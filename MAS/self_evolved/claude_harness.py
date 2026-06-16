"""Claude Agent SDK backend for the self-evolved agent harness.

``ClaudeAgentSDKClient`` satisfies the same ``generate`` contract as
``OpenRouterLLMClient``, so ``_execute_agent_stage`` stays the shared Agent
Harness: identical prompts, artifact coercion, and trace emission. Repo tool
dicts (``{"name", "description", "parameters", "handler"}``) are bridged into
an in-process MCP server; ``allowed_tools`` is limited to those bridged tools
so the agent keeps the run's context boundary (no file/bash access). Tool
executions are recorded by the bridge handlers themselves, so
``LLMResult.tool_calls`` reflects what actually ran.

The dependency is optional: install with ``pip install -e ".[claude]"``. Auth is
resolved by the SDK, either from ``ANTHROPIC_API_KEY`` or local Claude
credentials.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any

from ..artifacts import _extract_json_payload
from ..llm import LLMResult, OpenRouterLLMClient

_SERVER_NAME = "mas"
_DEFAULT_QUERY_TIMEOUT_S = 600.0
_TOOL_OUTPUT_MAX_CHARS = 4000
_FINALIZE_NOTES_MAX_CHARS = 4000
_SYNTHETIC_FINAL_MAX_CHARS = 2400
# The finalize call needs >1 turn: with max_turns=1 the SDK raises "Reached maximum
# number of turns" before emitting a result. Two turns suffice (verified), three gives margin.
_FINALIZE_MAX_TURNS = 3
_FINALIZE_SYSTEM_PROMPT = (
    "You are a research finalizer. You have NO tools available and cannot search. "
    "Synthesize the best answer you can from the evidence already gathered and return it "
    "as a single JSON object matching the required keys. If the evidence is insufficient, "
    "say so in the answer rather than asking to search further."
)

# Shared structured-answer schema. Used (optionally) on the main query and always
# on the no-tools finalization query so the SDK returns clean ``structured_output``.
_ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "required": [
            "answer_artifact",
            "summary",
            "critique",
            "revision_request",
            "confidence",
            "unresolved_issues",
            "evidence_summary",
        ],
        "properties": {
            "answer_artifact": {},
            "summary": {},
            "critique": {},
            "revision_request": {},
            "confidence": {"type": "number"},
            "unresolved_issues": {},
            "evidence_summary": {},
        },
    },
}


def _env_truthy(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _usage_token_in(usage: dict[str, Any]) -> int:
    """Total input tokens, including prompt-cache reads/writes.

    The Claude Agent SDK reports ``input_tokens`` as only the small uncached delta;
    the bulk of the prompt rides on ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``. Summing all three recovers the real input size.
    """

    return int(
        (usage.get("input_tokens", 0) or 0)
        + (usage.get("cache_read_input_tokens", 0) or 0)
        + (usage.get("cache_creation_input_tokens", 0) or 0)
    )


class ClaudeAgentSDKClient:
    """OpenRouter-compatible ``generate`` backed by the Claude Agent SDK."""

    def __init__(self, models: dict[str, str]) -> None:
        self.models = dict(models)

    def model_for_agent_type(self, agent_type: str) -> str:
        return self.models.get(agent_type, self.models.get("default", "claude-sonnet-4-6"))

    def generate(
        self,
        *,
        prompt: Any,
        agent_type: str,
        task_id: str,
        run_index: int,
        agent_id: str,
        tools: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = 8,
        temperature: float = 0.0,
    ) -> LLMResult:
        sdk = self._import_sdk()
        auth_source = "api_key" if os.environ.get("ANTHROPIC_API_KEY") else "sdk_default"

        model = self.model_for_agent_type(agent_type)
        base_system_text, prompt_text = self._split_prompt(prompt)
        tool_records: list[dict[str, Any]] = []
        has_tools = bool(tools)
        options_kwargs = self._base_options_kwargs(model)
        options_kwargs["max_turns"] = int(max_tool_iterations) if has_tools else 1
        if _env_truthy("CLAUDE_AGENT_SDK_JSON_SCHEMA", default=False):
            options_kwargs["output_format"] = _ANSWER_JSON_SCHEMA
        bridged = self._bridge_tools(sdk, list(tools or []), tool_records)
        system_text = base_system_text
        if bridged:
            tool_names = [
                f"mcp__{_SERVER_NAME}__{getattr(item, 'name', '')}"
                for item in bridged
                if getattr(item, "name", "")
            ]
            hint = (
                "Claude SDK tool bridge: when evidence is missing, call the allowed MCP "
                f"tool by its exact name ({', '.join(tool_names)}). Do not merely say "
                "you will search or inspect evidence; actually invoke the tool, then "
                "answer from the tool result."
            )
            system_text = "\n\n".join(part for part in (system_text, hint) if part)
        if system_text:
            options_kwargs["system_prompt"] = system_text

        if bridged:
            server = sdk.create_sdk_mcp_server(name=_SERVER_NAME, version="1.0.0", tools=bridged)
            options_kwargs["mcp_servers"] = {_SERVER_NAME: server}
        # Only the bridged MCP tools are allowed — never the SDK's built-in
        # file/bash tools, which would break the run's context boundary.
        options_kwargs["allowed_tools"] = [
            f"mcp__{_SERVER_NAME}__{getattr(item, 'name', '')}" for item in bridged
        ]
        options = sdk.ClaudeAgentOptions(**options_kwargs)

        scan = self._scan_messages(self._run_query(sdk, prompt_text, options))
        text = self._primary_text(scan)
        extra_in = extra_out = 0
        extra_cost = 0.0
        finalized = False
        # When a tool-using agent is cut off mid-investigation (max turns) it never
        # emits the final JSON answer, so the parse fails downstream. Mirror the
        # OpenRouter ``_force_final_response`` path: one no-tools synthesis call.
        if has_tools and _extract_json_payload(text) is None:
            if _env_truthy("CLAUDE_AGENT_SDK_SKIP_FINALIZE", default=False):
                text = self._synthetic_answer_json(
                    assistant_text=scan["assistant_text"],
                    tool_records=tool_records,
                    reason="sdk_finalize_skipped",
                )
                scan.setdefault("run_meta", {})["sdk_finalize_skipped"] = True
            else:
                try:
                    fin = self._finalize_answer(
                        sdk,
                        model=model,
                        prompt_text=prompt_text,
                        assistant_text=scan["assistant_text"],
                        tool_records=tool_records,
                    )
                except Exception as exc:
                    text = self._synthetic_answer_json(
                        assistant_text=scan["assistant_text"],
                        tool_records=tool_records,
                        reason=f"sdk_finalize_failed: {exc}",
                    )
                    scan.setdefault("run_meta", {})["sdk_finalize_error"] = str(exc)
                else:
                    fin_text = self._primary_text(fin)
                    if fin_text.strip():
                        text = fin_text
                        extra_in, extra_out = fin["token_in"], fin["token_out"]
                        extra_cost = fin["cost_usd"]
                        finalized = True

        return self._build_result(
            scan,
            text=text,
            model=model,
            tool_records=tool_records,
            auth_source=auth_source,
            extra_in=extra_in,
            extra_out=extra_out,
            extra_cost=extra_cost,
            finalized=finalized,
        )

    @staticmethod
    def _base_options_kwargs(model: str) -> dict[str, Any]:
        """Common ``ClaudeAgentOptions`` kwargs (env-driven) shared by both queries."""

        options_kwargs: dict[str, Any] = {"model": model}
        permission_mode = os.environ.get("CLAUDE_AGENT_SDK_PERMISSION_MODE", "dontAsk").strip()
        if permission_mode:
            options_kwargs["permission_mode"] = permission_mode
        effort = os.environ.get("CLAUDE_AGENT_SDK_EFFORT", "low").strip()
        if effort:
            options_kwargs["effort"] = effort
        thinking = os.environ.get("CLAUDE_AGENT_SDK_THINKING", "disabled").strip()
        if thinking:
            if thinking == "disabled":
                options_kwargs["thinking"] = {"type": "disabled"}
            elif thinking.isdigit():
                options_kwargs["max_thinking_tokens"] = int(thinking)
        return options_kwargs

    def _finalize_answer(
        self,
        sdk: Any,
        *,
        model: str,
        prompt_text: str,
        assistant_text: str,
        tool_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """One no-tools call forcing a structured final answer from gathered evidence.

        Uses a dedicated minimal system prompt (not the agent persona, which references
        tools and would waste the turn budget on blocked tool attempts) plus the JSON
        schema ``output_format`` so the SDK returns clean ``structured_output``.
        """

        options_kwargs = self._base_options_kwargs(model)
        options_kwargs["max_turns"] = _FINALIZE_MAX_TURNS
        options_kwargs["allowed_tools"] = []
        options_kwargs["output_format"] = _ANSWER_JSON_SCHEMA
        options_kwargs["system_prompt"] = _FINALIZE_SYSTEM_PROMPT
        options = sdk.ClaudeAgentOptions(**options_kwargs)
        notes = self._build_investigation_notes(assistant_text, tool_records)
        finalize_prompt = (
            f"{prompt_text}\n\n## Investigation notes so far\n{notes}\n\n"
            "Based ONLY on the evidence gathered above, output exactly one JSON object with keys: "
            "answer_artifact, summary, critique, revision_request, confidence, unresolved_issues, "
            "evidence_summary. Do not call any tools and do not wrap it in markdown."
        )
        return self._scan_messages(self._run_query(sdk, finalize_prompt, options))

    @staticmethod
    def _build_investigation_notes(
        assistant_text: str, tool_records: list[dict[str, Any]]
    ) -> str:
        lines: list[str] = []
        if assistant_text.strip():
            lines.append(f"Reasoning so far:\n{assistant_text.strip()}")
        if tool_records:
            lines.append("Tool findings:")
            for rec in tool_records[-8:]:
                args = json.dumps(rec.get("arguments", {}), ensure_ascii=False, default=str)
                output = json.dumps(rec.get("output", ""), ensure_ascii=False, default=str)
                lines.append(f"- {rec.get('tool_name', '')}({args[:200]}) -> {output[:600]}")
        return "\n".join(lines)[:_FINALIZE_NOTES_MAX_CHARS]

    @staticmethod
    def _synthetic_answer_json(
        *, assistant_text: str, tool_records: list[dict[str, Any]], reason: str
    ) -> str:
        evidence_lines: list[str] = []
        for rec in tool_records[-6:]:
            args = json.dumps(rec.get("arguments", {}), ensure_ascii=False, default=str)
            output = json.dumps(rec.get("output", ""), ensure_ascii=False, default=str)
            evidence_lines.append(
                f"{rec.get('tool_name', '')}({args[:180]}) -> {output[:420]}"
            )
        answer = assistant_text.strip() or "Insufficient evidence gathered by Claude Agent SDK."
        payload = {
            "answer_artifact": answer[:_SYNTHETIC_FINAL_MAX_CHARS],
            "summary": "Structured fallback from Claude Agent SDK assistant text and tool evidence.",
            "critique": reason,
            "revision_request": "No revision requested by fallback packager.",
            "confidence": 0.05,
            "unresolved_issues": [reason],
            "evidence_summary": evidence_lines,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    # -- SDK plumbing ----------------------------------------------------------

    @staticmethod
    def _import_sdk() -> Any:
        try:
            import claude_agent_sdk
        except ImportError as exc:
            raise RuntimeError(
                "harness_backend=claude_agent_sdk requires the optional "
                "'claude-agent-sdk' package: pip install -e \".[claude]\""
            ) from exc
        return claude_agent_sdk

    @staticmethod
    def _split_prompt(prompt: Any) -> tuple[str, str]:
        """Chat messages -> (system_prompt, user prompt text)."""

        if isinstance(prompt, str):
            return "", prompt
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in prompt or []:
            role = str(message.get("role", "user"))
            content = str(message.get("content", "") or "")
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            else:
                user_parts.append(f"[{role}]\n{content}")
        return "\n\n".join(system_parts), "\n\n".join(user_parts)

    @staticmethod
    def _bridge_tools(
        sdk: Any,
        tools: list[dict[str, Any]],
        tool_records: list[dict[str, Any]],
    ) -> list[Any]:
        bridged: list[Any] = []
        used_names: set[str] = set()
        for tool_dict in tools:
            original_name = str(tool_dict.get("name", "")).strip()
            handler = tool_dict.get("handler")
            if not original_name or not callable(handler):
                continue
            api_name = OpenRouterLLMClient._sanitize_tool_name(original_name, used_names)
            description = str(tool_dict.get("description", "")).strip() or f"Call {original_name}"
            params = tool_dict.get("parameters")
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}, "required": []}

            def _make_runner(name: str, run_handler: Any) -> Any:
                async def _run(args: dict[str, Any]) -> dict[str, Any]:
                    status, error = "completed", None
                    try:
                        output = run_handler(dict(args or {}))
                    except Exception as exc:
                        status, error = "error", str(exc)
                        output = {"error": str(exc)}
                    tool_records.append(
                        {
                            "tool_name": name,
                            "arguments": dict(args or {}),
                            "status": status,
                            "error": error,
                            "output": output,
                        }
                    )
                    text = json.dumps(output, ensure_ascii=False, default=str)
                    return {
                        "content": [{"type": "text", "text": text[:_TOOL_OUTPUT_MAX_CHARS]}],
                        "is_error": status == "error",
                    }

                return _run

            bridged.append(
                sdk.tool(api_name, description, params)(_make_runner(original_name, handler))
            )
        return bridged

    @staticmethod
    def _run_query(sdk: Any, prompt_text: str, options: Any) -> list[Any]:
        """Drive the async SDK stream from sync code via a worker thread."""

        async def _collect() -> list[Any]:
            collected: list[Any] = []
            try:
                async for message in sdk.query(prompt=prompt_text, options=options):
                    collected.append(message)
            except Exception as exc:
                error_text = str(exc)
                if collected and (
                    "Claude Code returned an error result: success" in error_text
                    or "Reached maximum number of turns" in error_text
                ):
                    return collected
                raise
            return collected

        holder: dict[str, Any] = {}

        def _runner() -> None:
            try:
                holder["messages"] = asyncio.run(_collect())
            except Exception as exc:
                holder["error"] = exc

        thread = threading.Thread(target=_runner, daemon=True, name="claude-agent-sdk-query")
        thread.start()
        timeout_s = ClaudeAgentSDKClient._query_timeout_s()
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise RuntimeError(
                f"claude_agent_sdk query exceeded the {timeout_s:.0f}s hard timeout"
            )
        error = holder.get("error")
        if error is not None:
            raise RuntimeError(f"claude_agent_sdk query failed: {error}") from error
        messages = list(holder.get("messages", []))
        for message in messages:
            result = str(getattr(message, "result", "") or "")
            if "Invalid API key" in result:
                raise RuntimeError("claude_agent_sdk authentication failed: invalid API key")
        return messages

    @staticmethod
    def _query_timeout_s() -> float:
        raw = os.environ.get("CLAUDE_AGENT_SDK_QUERY_TIMEOUT_S", "").strip()
        if not raw:
            return _DEFAULT_QUERY_TIMEOUT_S
        try:
            value = float(raw)
        except ValueError:
            return _DEFAULT_QUERY_TIMEOUT_S
        return max(1.0, value)

    @staticmethod
    def _scan_messages(messages: list[Any]) -> dict[str, Any]:
        """Collect text, structured output, usage and run-status from the SDK stream.

        Unlike a last-message-wins scan, all assistant narration is accumulated so a
        mid-stream JSON answer is not lost and the full reasoning is available to feed
        the finalization call.
        """

        assistant_parts: list[str] = []
        result_text = ""
        structured_output: dict[str, Any] | None = None
        usage: dict[str, Any] = {}
        cost_usd = 0.0
        run_meta: dict[str, Any] = {}
        for message in messages:
            blocks = getattr(message, "content", None)
            if isinstance(blocks, list):
                parts = [
                    str(getattr(block, "text", ""))
                    for block in blocks
                    if getattr(block, "text", "")
                ]
                if parts:
                    assistant_parts.append("\n".join(parts))
            if hasattr(message, "total_cost_usd") or hasattr(message, "usage"):
                raw_usage = getattr(message, "usage", None)
                if isinstance(raw_usage, dict):
                    usage = raw_usage
                cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
                result_text = str(getattr(message, "result", "") or "")
                so = getattr(message, "structured_output", None)
                if isinstance(so, dict):
                    structured_output = so
                for key in (
                    "num_turns",
                    "duration_ms",
                    "session_id",
                    "subtype",
                    "stop_reason",
                    "is_error",
                ):
                    value = getattr(message, key, None)
                    if value is not None:
                        run_meta[key] = value
        return {
            "assistant_text": "\n".join(assistant_parts),
            "result_text": result_text,
            "structured_output": structured_output,
            "token_in": _usage_token_in(usage),
            "token_out": int(usage.get("output_tokens", 0) or 0),
            "cost_usd": cost_usd,
            "run_meta": run_meta,
        }

    @staticmethod
    def _primary_text(scan: dict[str, Any]) -> str:
        """Best parseable text: structured output (as JSON) > result > assistant text."""

        structured = scan.get("structured_output")
        if isinstance(structured, dict):
            return json.dumps(structured, ensure_ascii=False, default=str)
        return scan.get("result_text") or scan.get("assistant_text") or ""

    @staticmethod
    def _build_result(
        scan: dict[str, Any],
        *,
        text: str,
        model: str,
        tool_records: list[dict[str, Any]],
        auth_source: str,
        extra_in: int,
        extra_out: int,
        extra_cost: float,
        finalized: bool,
    ) -> LLMResult:
        metadata: dict[str, Any] = {
            "provider": "claude_agent_sdk",
            "auth_source": auth_source,
            "sdk_finalized": finalized,
        }
        metadata.update(scan.get("run_meta", {}))
        return LLMResult(
            text=text,
            token_in=int(scan["token_in"]) + int(extra_in),
            token_out=int(scan["token_out"]) + int(extra_out),
            cost_usd=float(scan["cost_usd"]) + float(extra_cost),
            model=model,
            mock_used=False,
            metadata=metadata,
            tool_calls=tool_records,
        )

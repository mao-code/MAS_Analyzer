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

from ..llm import LLMResult, OpenRouterLLMClient

_SERVER_NAME = "mas"
_DEFAULT_QUERY_TIMEOUT_S = 600.0
_TOOL_OUTPUT_MAX_CHARS = 4000


def _env_truthy(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


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
        system_text, prompt_text = self._split_prompt(prompt)
        tool_records: list[dict[str, Any]] = []
        has_tools = bool(tools)
        options_kwargs: dict[str, Any] = {
            "model": model,
            "max_turns": int(max_tool_iterations) if has_tools else 1,
        }
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
        if _env_truthy("CLAUDE_AGENT_SDK_JSON_SCHEMA", default=False):
            options_kwargs["output_format"] = {
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
        bridged = self._bridge_tools(sdk, list(tools or []), tool_records)
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

        messages = self._run_query(sdk, prompt_text, options)
        return self._to_llm_result(
            messages,
            model=model,
            tool_records=tool_records,
            auth_source=auth_source,
        )

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
    def _to_llm_result(
        messages: list[Any],
        *,
        model: str,
        tool_records: list[dict[str, Any]],
        auth_source: str,
    ) -> LLMResult:
        assistant_text = ""
        result_text = ""
        usage: dict[str, Any] = {}
        cost_usd = 0.0
        metadata: dict[str, Any] = {
            "provider": "claude_agent_sdk",
            "auth_source": auth_source,
        }
        for message in messages:
            blocks = getattr(message, "content", None)
            if isinstance(blocks, list):
                parts = [
                    str(getattr(block, "text", ""))
                    for block in blocks
                    if getattr(block, "text", "")
                ]
                if parts:
                    assistant_text = "\n".join(parts)
            if hasattr(message, "total_cost_usd") or hasattr(message, "usage"):
                raw_usage = getattr(message, "usage", None)
                if isinstance(raw_usage, dict):
                    usage = raw_usage
                cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
                result_text = str(getattr(message, "result", "") or "")
                for key in ("num_turns", "duration_ms", "session_id"):
                    value = getattr(message, key, None)
                    if value is not None:
                        metadata[key] = value
        return LLMResult(
            text=result_text or assistant_text,
            token_in=int(usage.get("input_tokens", 0) or 0),
            token_out=int(usage.get("output_tokens", 0) or 0),
            cost_usd=cost_usd,
            model=model,
            mock_used=False,
            metadata=metadata,
            tool_calls=tool_records,
        )

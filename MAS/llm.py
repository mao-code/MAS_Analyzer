from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import random
import re
import signal
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import OpenRouterConfig


@dataclass
class LLMResult:
    text: str
    token_in: int
    token_out: int
    cost_usd: float
    model: str
    mock_used: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _ToolLoopTurn:
    iteration: int
    assistant_msg: dict[str, Any]
    assistant_text: str
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_records: list[dict[str, Any]] = field(default_factory=list)


class _HardTimeoutExpired(RuntimeError):
    """Raised when an LLM request exceeds the configured wall-clock timeout."""


class OpenRouterLLMClient:
    """OpenRouter chat client with deterministic local fallback."""

    def __init__(self, config: OpenRouterConfig, models: dict[str, str]) -> None:
        self.config = config
        self.models = dict(models)
        self.client = None
        self._request_seq = 0
        self._request_seq_lock = threading.Lock()
        self.disable_live = self._env_flag("MAS_DISABLE_LIVE_LLM")
        self.require_live = (not self.disable_live) and self._env_flag("MAS_REQUIRE_LIVE_LLM")

        if self.disable_live:
            return

        if not config.api_key:
            if self.require_live:
                raise RuntimeError(
                    "MAS_REQUIRE_LIVE_LLM is enabled but OPENROUTER_API_KEY is missing."
                )
            return

        try:
            import openai  # type: ignore
        except Exception as exc:
            if self.require_live:
                raise RuntimeError(
                    "MAS_REQUIRE_LIVE_LLM is enabled but the 'openai' package is unavailable."
                ) from exc
            return

        headers: dict[str, str] = {}
        if config.http_referer:
            headers["HTTP-Referer"] = config.http_referer
        if config.x_title:
            headers["X-Title"] = config.x_title

        kwargs: dict[str, Any] = {
            "base_url": config.base_url,
            "api_key": config.api_key,
            "timeout": config.timeout_s,
        }
        if headers:
            kwargs["default_headers"] = headers

        try:
            self.client = openai.OpenAI(**kwargs)
        except Exception as exc:
            self.client = None
            if self.require_live:
                raise RuntimeError("Failed to initialize the OpenRouter client.") from exc

    def model_for_agent_type(self, agent_type: str) -> str:
        return self.models.get(agent_type, self.models.get("default", "qwen/qwen3-8b"))

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_float(name: str) -> float | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return float(raw)

    @staticmethod
    def _env_int(name: str) -> int | None:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        return int(raw)

    @staticmethod
    def _env_str(name: str) -> str | None:
        raw = os.getenv(name, "").strip()
        return raw or None

    def _apply_openrouter_sampling_overrides(
        self,
        kwargs: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        request_kwargs = dict(kwargs)

        temperature_override = self._env_float("OPENROUTER_TEMPERATURE")
        top_p_override = self._env_float("OPENROUTER_TOP_P")
        top_k_override = self._env_int("OPENROUTER_TOP_K")
        reasoning_effort_override = self._env_str("OPENROUTER_REASONING_EFFORT")

        effective_temperature = (
            temperature_override if temperature_override is not None else temperature
        )
        if effective_temperature is not None:
            request_kwargs["temperature"] = effective_temperature
        if top_p_override is not None:
            request_kwargs["top_p"] = top_p_override
        if top_k_override is not None:
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body["top_k"] = top_k_override
            request_kwargs["extra_body"] = extra_body
        if reasoning_effort_override is not None:
            extra_body = dict(request_kwargs.get("extra_body") or {})
            extra_body["reasoning"] = {
                "effort": reasoning_effort_override,
            }
            request_kwargs["extra_body"] = extra_body

        return request_kwargs

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
        model = self.model_for_agent_type(agent_type)
        tool_defs, tool_handlers, original_tool_names = self._normalize_tools(tools or [])
        request_id = self._next_request_id()
        started = time.perf_counter()
        message_count = len(prompt) if isinstance(prompt, list) else 1
        self._log(
            "START "
            f"request_id={request_id} task_id={task_id} run_index={run_index} "
            f"agent_id={agent_id} agent_type={agent_type} model={model} "
            f"messages={message_count} tools={len(tool_defs)} "
            f"max_tool_iterations={max(1, int(max_tool_iterations))}"
        )

        if self.client is not None:
            try:
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{"role": "user", "content": str(prompt)}]

                if tool_defs:
                    result = self._generate_with_tools(
                        model=model,
                        messages=messages,
                        tool_defs=tool_defs,
                        tool_handlers=tool_handlers,
                        original_tool_names=original_tool_names,
                        max_tool_iterations=max_tool_iterations,
                        temperature=temperature,
                        request_id=request_id,
                        task_id=task_id,
                        run_index=run_index,
                        agent_id=agent_id,
                    )
                    elapsed_s = max(time.perf_counter() - started, 0.0)
                    self._log(
                        "DONE "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} model={model} mode=tools "
                        f"elapsed_s={elapsed_s:.2f} token_in={result.token_in} "
                        f"token_out={result.token_out} tool_calls={len(result.tool_calls)}"
                    )
                    return result

                completion = self._create_chat_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                text = self._extract_text(completion)
                usage = getattr(completion, "usage", None)
                token_in = int(getattr(usage, "prompt_tokens", self._estimate_tokens(prompt)))
                token_out = int(getattr(usage, "completion_tokens", self._estimate_tokens(text)))

                result = LLMResult(
                    text=text,
                    token_in=token_in,
                    token_out=token_out,
                    cost_usd=0.0,
                    model=model,
                    mock_used=False,
                    metadata={
                        "provider": "openrouter",
                        "missing_cost_note": "OpenRouter response did not provide cost_usd; recorded as 0.0",
                    },
                )
                elapsed_s = max(time.perf_counter() - started, 0.0)
                self._log(
                    "DONE "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} model={model} mode=chat "
                    f"elapsed_s={elapsed_s:.2f} token_in={token_in} token_out={token_out}"
                )
                return result
            except Exception as exc:
                elapsed_s = max(time.perf_counter() - started, 0.0)
                self._log(
                    "ERROR "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} model={model} elapsed_s={elapsed_s:.2f} "
                    f"error={type(exc).__name__}:{exc}"
                )
                if self.require_live:
                    raise RuntimeError(f"Live OpenRouter generation failed: {exc}") from exc
                result = self._mock_result(
                    prompt=prompt,
                    agent_type=agent_type,
                    task_id=task_id,
                    run_index=run_index,
                    agent_id=agent_id,
                    model=model,
                    fallback_reason=str(exc),
                    tools_available=bool(tool_defs),
                )
                self._log(
                    "FALLBACK "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} model={model} mock_used=true "
                    f"reason={type(exc).__name__}:{exc}"
                )
                return result

        if self.require_live:
            raise RuntimeError("Live OpenRouter generation is required but the client is unavailable.")

        result = self._mock_result(
            prompt=prompt,
            agent_type=agent_type,
            task_id=task_id,
            run_index=run_index,
            agent_id=agent_id,
            model=model,
            fallback_reason="OpenRouter client unavailable or API key missing",
            tools_available=bool(tool_defs),
        )
        elapsed_s = max(time.perf_counter() - started, 0.0)
        self._log(
            "FALLBACK "
            f"request_id={request_id} task_id={task_id} run_index={run_index} "
            f"agent_id={agent_id} model={model} elapsed_s={elapsed_s:.2f} "
            "mock_used=true reason=client_unavailable_or_api_key_missing"
        )
        return result

    def _generate_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tool_defs: list[dict[str, Any]],
        tool_handlers: dict[str, Callable[[dict[str, Any]], Any]],
        original_tool_names: dict[str, str],
        max_tool_iterations: int,
        temperature: float,
        request_id: str,
        task_id: str,
        run_index: int,
        agent_id: str,
    ) -> LLMResult:
        base_messages = [dict(item) for item in messages]
        total_token_in = 0
        total_token_out = 0
        tool_call_records: list[dict[str, Any]] = []
        final_text = ""
        stopped_early = False
        duplicate_call_counts: dict[str, int] = {}
        tool_turns: list[_ToolLoopTurn] = []
        latest_context_stats = {
            "raw_turns": 0,
            "summarized_turns": 0,
            "summary_chars": 0,
        }

        for iteration_index in range(max(1, int(max_tool_iterations))):
            working_messages, latest_context_stats = self._build_tool_loop_messages(
                base_messages=base_messages,
                tool_turns=tool_turns,
            )
            if latest_context_stats["summarized_turns"] > 0:
                self._log(
                    "TOOL_CONTEXT "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} raw_turns={latest_context_stats['raw_turns']} "
                    f"summarized_turns={latest_context_stats['summarized_turns']} "
                    f"summary_chars={latest_context_stats['summary_chars']}"
                )
            self._log(
                "TOOL_LOOP "
                f"request_id={request_id} task_id={task_id} run_index={run_index} "
                f"agent_id={agent_id} iteration={iteration_index + 1}/"
                f"{max(1, int(max_tool_iterations))} messages={len(working_messages)}"
            )
            completion = self._create_chat_completion(
                model=model,
                messages=working_messages,
                temperature=temperature,
                tools=tool_defs,
                tool_choice="auto",
            )
            usage = getattr(completion, "usage", None)
            total_token_in += int(getattr(usage, "prompt_tokens", 0))
            total_token_out += int(getattr(usage, "completion_tokens", 0))

            if not total_token_in:
                total_token_in = self._estimate_tokens(working_messages)

            choice = completion.choices[0] if getattr(completion, "choices", None) else None
            message = getattr(choice, "message", None)
            if message is None:
                final_text = ""
                break

            message_content = getattr(message, "content", "")
            if isinstance(message_content, list):
                parts = []
                for item in message_content:
                    text = (
                        item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                    )
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                assistant_text = "\n".join(parts).strip()
            else:
                assistant_text = str(message_content or "")
            if assistant_text:
                final_text = assistant_text

            raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
            serialized_tool_calls: list[dict[str, Any]] = []
            for tool_call in raw_tool_calls:
                function = getattr(tool_call, "function", None)
                name = str(getattr(function, "name", "") or "")
                arguments_raw = str(getattr(function, "arguments", "") or "{}")
                serialized_tool_calls.append(
                    {
                        "id": str(getattr(tool_call, "id", "")),
                        "type": str(getattr(tool_call, "type", "function") or "function"),
                        "function": {
                            "name": name,
                            "arguments": arguments_raw,
                        },
                    }
                )

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text,
            }
            if serialized_tool_calls:
                assistant_msg["tool_calls"] = serialized_tool_calls
            current_turn = _ToolLoopTurn(
                iteration=iteration_index + 1,
                assistant_msg=dict(assistant_msg),
                assistant_text=assistant_text,
            )
            self._log(
                "TOOL_LOOP_RESULT "
                f"request_id={request_id} task_id={task_id} run_index={run_index} "
                f"agent_id={agent_id} iteration={iteration_index + 1} "
                f"tool_calls={len(serialized_tool_calls)} assistant_text_chars={len(assistant_text)}"
            )

            if not serialized_tool_calls:
                tool_turns.append(current_turn)
                break

            for tool_call in serialized_tool_calls:
                fn = tool_call["function"]
                api_tool_name = str(fn.get("name") or "")
                tool_name = original_tool_names.get(api_tool_name, api_tool_name)
                tool_id = str(tool_call.get("id") or "")
                args_text = str(fn.get("arguments") or "{}")
                try:
                    args = json.loads(args_text) if args_text.strip() else {}
                    if not isinstance(args, dict):
                        args = {"value": args}
                except Exception:
                    args = {}

                status = "completed"
                error = None
                output: Any
                handler = tool_handlers.get(api_tool_name)
                if handler is None:
                    status = "error"
                    error = f"Unknown tool: {tool_name}"
                    output = {"error": error}
                else:
                    try:
                        self._log(
                            "TOOL_CALL "
                            f"request_id={request_id} task_id={task_id} run_index={run_index} "
                            f"agent_id={agent_id} iteration={iteration_index + 1} "
                            f"tool_name={tool_name} args={self._truncate_for_log(args)}"
                        )
                        output = handler(args)
                        if inspect.isawaitable(output):
                            output = asyncio.run(output)
                    except Exception as exc:
                        status = "error"
                        error = str(exc)
                        output = {"error": error}
                self._log(
                    "TOOL_CALL_RESULT "
                    f"request_id={request_id} task_id={task_id} run_index={run_index} "
                    f"agent_id={agent_id} iteration={iteration_index + 1} "
                    f"tool_name={tool_name} status={status} "
                    f"output={self._truncate_for_log(output)}"
                )

                tool_call_records.append(
                    {
                        "tool_name": tool_name,
                        "arguments": args,
                        "status": status,
                        "error": error,
                        "output": output,
                    }
                )
                current_turn.tool_records.append(tool_call_records[-1])

                sig = f"{api_tool_name}:{args_text}"
                duplicate_call_counts[sig] = duplicate_call_counts.get(sig, 0) + 1
                if duplicate_call_counts[sig] >= 3:
                    self._log(
                        "TOOL_LOOP_STOP "
                        f"request_id={request_id} task_id={task_id} run_index={run_index} "
                        f"agent_id={agent_id} reason=duplicate_tool_call signature={sig}"
                    )
                    current_turn.tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": api_tool_name,
                            "content": (
                                "Repeated identical tool call detected. This exact call has been made "
                                "3 or more times with the same arguments and is returning the same result. "
                                "Please stop repeating this call and provide your final answer now."
                            ),
                        }
                    )
                    stopped_early = True
                    break

                if isinstance(output, str):
                    output_payload = output
                else:
                    output_payload = self._safe_json_dumps(output)
                current_turn.tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": api_tool_name,
                        "content": output_payload,
                    }
                )

            tool_turns.append(current_turn)
            if stopped_early:
                break
        else:
            stopped_early = True

        if stopped_early and not final_text:
            final_messages, latest_context_stats = self._build_tool_loop_messages(
                base_messages=base_messages,
                tool_turns=tool_turns,
            )
            self._log(
                "FORCE_FINAL "
                f"request_id={request_id} task_id={task_id} run_index={run_index} "
                f"agent_id={agent_id}"
            )
            final_text, extra_in, extra_out = self._force_final_response(
                model=model,
                messages=final_messages,
                temperature=temperature,
                max_tool_iterations=max_tool_iterations,
            )
            total_token_in += extra_in
            total_token_out += extra_out

        if not total_token_out:
            total_token_out = self._estimate_tokens(final_text)

        metadata = {
            "provider": "openrouter",
            "missing_cost_note": "OpenRouter response did not provide cost_usd; recorded as 0.0",
            "tool_context_raw_turns": latest_context_stats["raw_turns"],
            "tool_context_summarized_turns": latest_context_stats["summarized_turns"],
        }
        if latest_context_stats["summarized_turns"] > 0:
            metadata["tool_context_compacted"] = True
        if stopped_early:
            repeated = [sig for sig, cnt in duplicate_call_counts.items() if cnt >= 3]
            if repeated:
                metadata["tool_loop_stopped_reason"] = (
                    f"Duplicate tool call detected (repeated >=3 times): {repeated[0]}"
                )
            else:
                metadata["tool_loop_stopped_reason"] = (
                    f"Reached max_tool_iterations={max(1, int(max_tool_iterations))}"
                )
            if final_text:
                metadata["tool_loop_forced_final_answer"] = True

        return LLMResult(
            text=final_text,
            token_in=total_token_in,
            token_out=total_token_out,
            cost_usd=0.0,
            model=model,
            mock_used=False,
            metadata=metadata,
            tool_calls=tool_call_records,
        )

    def _mock_result(
        self,
        *,
        prompt: Any,
        agent_type: str,
        task_id: str,
        run_index: int,
        agent_id: str,
        model: str,
        fallback_reason: str,
        tools_available: bool,
    ) -> LLMResult:
        prompt_str = (
            prompt
            if isinstance(prompt, str)
            else " ".join(str(m.get("content", "")) for m in prompt)
        )
        seed_value = self._stable_seed(task_id, str(run_index), agent_id, agent_type, prompt_str)
        rng = random.Random(seed_value)

        words = [token for token in re.split(r"\s+", prompt_str.strip()) if token]
        if not words:
            words = ["empty", "prompt"]
        sampled = [words[rng.randrange(len(words))] for _ in range(min(8, len(words) + 2))]
        # Avoid square brackets so benchmark citation/docid regexes do not
        # treat mock metadata as retrieved evidence docids.
        answer = (
            f"MOCK({agent_id}|{agent_type}) "
            f"Synthesized response with seed={seed_value % 100000}: " + " ".join(sampled)
        )

        return LLMResult(
            text=answer,
            token_in=self._estimate_tokens(prompt),
            token_out=self._estimate_tokens(answer),
            cost_usd=0.0,
            model=model,
            mock_used=True,
            metadata={
                "provider": "mock",
                "fallback_reason": fallback_reason,
                "seed": seed_value,
                "tools_available": tools_available,
            },
        )

    @staticmethod
    def _normalize_tools(
        tools: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Callable[[dict[str, Any]], Any]],
        dict[str, str],
    ]:
        defs: list[dict[str, Any]] = []
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        original_names: dict[str, str] = {}
        used_names: set[str] = set()
        for tool in tools:
            original_name = str(tool.get("name", "")).strip()
            handler = tool.get("handler")
            if not original_name or not callable(handler):
                continue
            api_name = OpenRouterLLMClient._sanitize_tool_name(original_name, used_names)
            description = str(tool.get("description", "")).strip() or f"Call {original_name}"
            params = tool.get("parameters")
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}, "required": []}
            defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": api_name,
                        "description": description,
                        "parameters": params,
                    },
                }
            )
            handlers[api_name] = handler
            # Some models still emit the original dotted tool name after seeing it in
            # the prompt context, even when the API-exposed function name is sanitized.
            # Resolve both forms to the same callable so tool execution remains stable.
            handlers[original_name] = handler
            original_names[api_name] = original_name
            original_names[original_name] = original_name
        return defs, handlers, original_names

    def _build_tool_loop_messages(
        self,
        *,
        base_messages: list[dict[str, Any]],
        tool_turns: list[_ToolLoopTurn],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        raw_turn_window = max(1, self._env_int("MAS_TOOL_CONTEXT_RAW_TURNS") or 1)
        preview_chars = max(80, self._env_int("MAS_TOOL_CONTEXT_PREVIEW_CHARS") or 320)
        summary_max_chars = max(
            1024,
            self._env_int("MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS") or 6000,
        )
        summarized_turns = max(0, len(tool_turns) - raw_turn_window)
        older_turns = tool_turns[:summarized_turns]
        recent_turns = tool_turns[summarized_turns:]
        messages = [dict(item) for item in base_messages]
        summary_chars = 0

        if older_turns:
            summary = self._summarize_tool_turns(
                older_turns,
                preview_chars=preview_chars,
                max_chars=summary_max_chars,
            )
            summary_chars = len(summary)
            messages = self._insert_context_message(
                messages,
                {
                    "role": "system",
                    "content": summary,
                },
            )

        for turn in recent_turns:
            messages.append(dict(turn.assistant_msg))
            for tool_message in turn.tool_messages:
                messages.append(dict(tool_message))

        return messages, {
            "raw_turns": len(recent_turns),
            "summarized_turns": len(older_turns),
            "summary_chars": summary_chars,
        }

    def _summarize_tool_turns(
        self,
        tool_turns: list[_ToolLoopTurn],
        *,
        preview_chars: int,
        max_chars: int,
    ) -> str:
        tool_call_count = sum(len(turn.tool_records) for turn in tool_turns)
        lines = [
            "Tool-memory summary from earlier tool iterations.",
            (
                "This is a compressed summary of older tool-use. "
                "Prefer the latest raw tool messages later in the conversation if there is any conflict."
            ),
            f"Earlier iterations summarized: {len(tool_turns)}",
            f"Earlier tool calls summarized: {tool_call_count}",
        ]

        for turn in tool_turns:
            assistant_preview = self._preview_text(turn.assistant_text, limit=preview_chars)
            if assistant_preview:
                lines.append(f"Iteration {turn.iteration} assistant: {assistant_preview}")
            else:
                lines.append(f"Iteration {turn.iteration} assistant: [no direct answer text]")
            for record in turn.tool_records:
                lines.append(
                    (
                        f"Iteration {turn.iteration} tool {record['tool_name']} "
                        f"args={self._truncate_for_log(record.get('arguments'), limit=preview_chars)} "
                        f"status={record.get('status', 'completed')} "
                        f"output={self._truncate_for_log(record.get('output'), limit=preview_chars)}"
                    )
                )

        clipped: list[str] = []
        used = 0
        for line in lines:
            addition = ("\n" if clipped else "") + line
            if used + len(addition) <= max_chars:
                clipped.append(line)
                used += len(addition)
                continue

            remaining = max_chars - used - (1 if clipped else 0)
            if remaining > 32:
                clipped.append(line[: remaining - 20].rstrip() + "... [truncated]")
            elif not clipped:
                clipped.append("Tool-memory summary truncated.")
            else:
                clipped.append("... [truncated]")
            break

        return "\n".join(clipped)

    @staticmethod
    def _insert_context_message(
        base_messages: list[dict[str, Any]],
        context_message: dict[str, Any],
    ) -> list[dict[str, Any]]:
        insert_at = 0
        while insert_at < len(base_messages) and base_messages[insert_at].get("role") == "system":
            insert_at += 1
        return [
            *[dict(item) for item in base_messages[:insert_at]],
            dict(context_message),
            *[dict(item) for item in base_messages[insert_at:]],
        ]

    def _force_final_response(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tool_iterations: int,
    ) -> tuple[str, int, int]:
        follow_up_messages = [dict(item) for item in messages]
        follow_up_messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of tool calls "
                    f"({max(1, int(max_tool_iterations))}). "
                    "Based only on the information already gathered, provide your best final answer now. "
                    "Do not call any more tools."
                ),
            }
        )
        completion = self._create_chat_completion(
            model=model,
            messages=follow_up_messages,
            temperature=temperature,
        )
        text = self._extract_text(completion).strip()
        usage = getattr(completion, "usage", None)
        token_in = int(getattr(usage, "prompt_tokens", self._estimate_tokens(follow_up_messages)))
        token_out = int(getattr(usage, "completion_tokens", self._estimate_tokens(text)))
        return text, token_in, token_out

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        if self.client is None:
            raise RuntimeError("OpenRouter client unavailable")
        kwargs = self._apply_openrouter_sampling_overrides(
            kwargs,
            temperature=kwargs.get("temperature"),
        )
        kwargs.setdefault("timeout", self.config.timeout_s)
        timeout_value = float(self.config.timeout_s or 0.0)
        # On the main thread we can use SIGALRM for a true wall-clock deadline.
        # On worker threads (LangGraph Send branches) SIGALRM is unavailable, so
        # we run the HTTP call in a disposable thread and join with a timeout.
        # This ensures every API call is bounded regardless of which thread calls it.
        if (
            timeout_value > 0.0
            and threading.current_thread() is not threading.main_thread()
        ):
            import concurrent.futures

            # Do NOT use the context-manager form of ThreadPoolExecutor here.
            # Its __exit__ calls shutdown(wait=True), which would block forever
            # waiting for the background API thread even after a timeout fires.
            _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _future = _ex.submit(self.client.chat.completions.create, **kwargs)
            try:
                result = _future.result(timeout=timeout_value)
                _ex.shutdown(wait=False)
                return result
            except concurrent.futures.TimeoutError:
                _ex.shutdown(wait=False)
                self._log(
                    "TIMEOUT "
                    f"mode=worker_thread timeout_s={timeout_value:.3f} "
                    f"model={kwargs.get('model', '')}"
                )
                raise _HardTimeoutExpired(
                    f"LLM request exceeded configured timeout of {timeout_value:.3f}s"
                )
            except Exception:
                _ex.shutdown(wait=False)
                raise
        with self._hard_timeout(self.config.timeout_s):
            return self.client.chat.completions.create(**kwargs)

    @contextmanager
    def _hard_timeout(self, timeout_s: float | None):
        timeout_value = float(timeout_s or 0.0)
        signal_supported = (
            timeout_value > 0.0
            and os.name != "nt"
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and hasattr(signal, "ITIMER_REAL")
            and threading.current_thread() is threading.main_thread()
        )
        if not signal_supported:
            yield
            return

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_delay, previous_interval = signal.setitimer(signal.ITIMER_REAL, 0.0)

        def _handle_timeout(signum: int, frame: Any) -> None:
            self._log(
                "TIMEOUT "
                f"mode=main_thread timeout_s={timeout_value:.3f}"
            )
            raise _HardTimeoutExpired(
                f"LLM request exceeded configured timeout of {timeout_value:.3f}s"
            )

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_value)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_delay > 0.0 or previous_interval > 0.0:
                signal.setitimer(signal.ITIMER_REAL, previous_delay, previous_interval)

    @staticmethod
    def _sanitize_tool_name(name: str, used_names: set[str]) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name).strip("_")
        if not sanitized:
            sanitized = "tool"
        candidate = sanitized
        suffix = 2
        while candidate in used_names:
            candidate = f"{sanitized}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    @staticmethod
    def _extract_text(completion: Any) -> str:
        choices = getattr(completion, "choices", [])
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    @staticmethod
    def _estimate_tokens(text: Any) -> int:
        if isinstance(text, list):
            content = " ".join(str(m.get("content", "")) for m in text)
            if not content.strip():
                return 0
            return max(1, int(len(re.findall(r"\S+", content)) * 1.3))

        if not text or not str(text).strip():
            return 0
        return max(1, int(len(re.findall(r"\S+", str(text))) * 1.3))

    @staticmethod
    def _safe_json_dumps(value: Any) -> str:
        return json.dumps(OpenRouterLLMClient._to_jsonable(value), ensure_ascii=False)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): OpenRouterLLMClient._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [OpenRouterLLMClient._to_jsonable(item) for item in value]

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return OpenRouterLLMClient._to_jsonable(to_dict())
            except Exception:
                pass

        to_list = getattr(value, "tolist", None)
        if callable(to_list):
            try:
                return OpenRouterLLMClient._to_jsonable(to_list())
            except Exception:
                pass

        item = getattr(value, "item", None)
        if callable(item):
            try:
                return OpenRouterLLMClient._to_jsonable(item())
            except Exception:
                pass

        return str(value)

    @staticmethod
    def _stable_seed(*parts: str) -> int:
        data = "||".join(parts).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        return int(digest[:16], 16)

    def _next_request_id(self) -> str:
        with self._request_seq_lock:
            self._request_seq += 1
            return f"llm-{self._request_seq}"

    @staticmethod
    def _truncate_for_log(value: Any, *, limit: int = 240) -> str:
        text = OpenRouterLLMClient._safe_json_dumps(value)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _preview_text(text: Any, *, limit: int = 320) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", str(text)).strip()
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."

    @staticmethod
    def _log(message: str) -> None:
        print(f"[llm] {message}", flush=True)

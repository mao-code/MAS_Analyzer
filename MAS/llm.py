from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import random
import re
from collections.abc import Callable
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


class OpenRouterLLMClient:
    """OpenRouter chat client with deterministic local fallback."""

    def __init__(self, config: OpenRouterConfig, models: dict[str, str]) -> None:
        self.config = config
        self.models = dict(models)
        self.client = None
        self.require_live = os.getenv("MAS_REQUIRE_LIVE_LLM", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

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

        if self.client is not None:
            try:
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{"role": "user", "content": str(prompt)}]

                if tool_defs:
                    return self._generate_with_tools(
                        model=model,
                        messages=messages,
                        tool_defs=tool_defs,
                        tool_handlers=tool_handlers,
                        original_tool_names=original_tool_names,
                        max_tool_iterations=max_tool_iterations,
                        temperature=temperature,
                    )

                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                text = self._extract_text(completion)
                usage = getattr(completion, "usage", None)
                token_in = int(getattr(usage, "prompt_tokens", self._estimate_tokens(prompt)))
                token_out = int(getattr(usage, "completion_tokens", self._estimate_tokens(text)))

                return LLMResult(
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
            except Exception as exc:
                if self.require_live:
                    raise RuntimeError(f"Live OpenRouter generation failed: {exc}") from exc
                return self._mock_result(
                    prompt=prompt,
                    agent_type=agent_type,
                    task_id=task_id,
                    run_index=run_index,
                    agent_id=agent_id,
                    model=model,
                    fallback_reason=str(exc),
                    tools_available=bool(tool_defs),
                )

        if self.require_live:
            raise RuntimeError("Live OpenRouter generation is required but the client is unavailable.")

        return self._mock_result(
            prompt=prompt,
            agent_type=agent_type,
            task_id=task_id,
            run_index=run_index,
            agent_id=agent_id,
            model=model,
            fallback_reason="OpenRouter client unavailable or API key missing",
            tools_available=bool(tool_defs),
        )

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
    ) -> LLMResult:
        working_messages = [dict(item) for item in messages]
        total_token_in = 0
        total_token_out = 0
        tool_call_records: list[dict[str, Any]] = []
        final_text = ""
        stopped_early = False

        for _ in range(max(1, int(max_tool_iterations))):
            completion = self.client.chat.completions.create(
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
            working_messages.append(assistant_msg)

            if not serialized_tool_calls:
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
                        output = handler(args)
                        if inspect.isawaitable(output):
                            output = asyncio.run(output)
                    except Exception as exc:
                        status = "error"
                        error = str(exc)
                        output = {"error": error}

                tool_call_records.append(
                    {
                        "tool_name": tool_name,
                        "arguments": args,
                        "status": status,
                        "error": error,
                        "output": output,
                    }
                )

                if isinstance(output, str):
                    output_payload = output
                else:
                    output_payload = self._safe_json_dumps(output)
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "name": api_tool_name,
                        "content": output_payload,
                    }
                )
        else:
            stopped_early = True

        if stopped_early and not final_text:
            final_text, extra_in, extra_out = self._force_final_response(
                model=model,
                messages=working_messages,
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
        }
        if stopped_early:
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
            original_names[api_name] = original_name
        return defs, handlers, original_names

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
        completion = self.client.chat.completions.create(
            model=model,
            messages=follow_up_messages,
            temperature=temperature,
        )
        text = self._extract_text(completion).strip()
        usage = getattr(completion, "usage", None)
        token_in = int(getattr(usage, "prompt_tokens", self._estimate_tokens(follow_up_messages)))
        token_out = int(getattr(usage, "completion_tokens", self._estimate_tokens(text)))
        return text, token_in, token_out

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

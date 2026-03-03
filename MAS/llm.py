from __future__ import annotations

import hashlib
import random
import re
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


class OpenRouterLLMClient:
    """OpenRouter chat client with deterministic local fallback."""

    def __init__(self, config: OpenRouterConfig, models: dict[str, str]) -> None:
        self.config = config
        self.models = dict(models)
        self.client = None

        if not config.api_key:
            return

        try:
            import openai  # type: ignore
        except Exception:
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
        except Exception:
            self.client = None

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
        temperature: float = 0.0,
    ) -> LLMResult:
        model = self.model_for_agent_type(agent_type)

        if self.client is not None:
            try:
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{"role": "user", "content": str(prompt)}]

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
                return self._mock_result(
                    prompt=prompt,
                    agent_type=agent_type,
                    task_id=task_id,
                    run_index=run_index,
                    agent_id=agent_id,
                    model=model,
                    fallback_reason=str(exc),
                )

        return self._mock_result(
            prompt=prompt,
            agent_type=agent_type,
            task_id=task_id,
            run_index=run_index,
            agent_id=agent_id,
            model=model,
            fallback_reason="OpenRouter client unavailable or API key missing",
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
            },
        )

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
    def _stable_seed(*parts: str) -> int:
        data = "||".join(parts).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        return int(digest[:16], 16)

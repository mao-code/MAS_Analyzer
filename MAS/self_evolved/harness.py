"""Agent harness abstraction: one stage-execution contract, two LLM backends.

The harness protocol is exactly ``OpenRouterLLMClient.generate``'s keyword
signature returning ``LLMResult``. ``_execute_agent_stage`` is therefore the
shared Agent Harness for every backend: same role prompts, artifact coercion,
tool-retry contract, and trace emission regardless of provider.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..llm import LLMResult, OpenRouterLLMClient

HARNESS_BACKENDS = ("openrouter", "claude_agent_sdk")


class AgentHarness(Protocol):
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
    ) -> LLMResult: ...


def build_harness(backend: str, llm_client: OpenRouterLLMClient) -> AgentHarness:
    backend = str(backend or "openrouter").strip().lower()
    if backend == "openrouter":
        return llm_client
    if backend == "claude_agent_sdk":
        from .claude_harness import ClaudeAgentSDKClient

        return ClaudeAgentSDKClient(models=dict(llm_client.models))
    raise ValueError(
        f"Unknown harness backend '{backend}'. Expected one of: {', '.join(HARNESS_BACKENDS)}"
    )

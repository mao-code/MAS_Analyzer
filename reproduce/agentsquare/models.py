from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MODULE_TYPES = ("planning", "reasoning", "tooluse", "memory")


@dataclass(frozen=True)
class AgentSquareModule:
    """One standardized AgentSquare module choice.

    The upstream project models agents as a four-slot tuple:
    Planning, Reasoning, Tool Use, and Memory.  This class stores the
    reproducible module identity plus a prompt-level implementation suitable
    for this repo's benchmark adapters.
    """

    name: str
    module_type: str
    thought: str
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "module_type": self.module_type,
            "thought": self.thought,
            "prompt": self.prompt,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentSquareSpec:
    planning: AgentSquareModule | None
    reasoning: AgentSquareModule
    tooluse: AgentSquareModule | None = None
    memory: AgentSquareModule | None = None

    def module_for(self, module_type: str) -> AgentSquareModule | None:
        return {
            "planning": self.planning,
            "reasoning": self.reasoning,
            "tooluse": self.tooluse,
            "memory": self.memory,
        }.get(module_type)

    def to_payload(self) -> dict[str, Any]:
        return {
            "planning": self.planning.to_payload() if self.planning else None,
            "reasoning": self.reasoning.to_payload(),
            "tooluse": self.tooluse.to_payload() if self.tooluse else None,
            "memory": self.memory.to_payload() if self.memory else None,
        }


@dataclass(frozen=True)
class AgentSquareConfig:
    model_agent_type: str = "default"
    temperature: float = 1.0
    max_plan_steps: int = 3
    max_reasoning_samples: int = 3
    max_tokens: int | None = None

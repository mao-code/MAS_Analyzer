from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ADASConfig:
    model_agent_type: str = "default"
    temperature: float = 1.0
    max_tokens: int | None = None
    max_tool_iterations: int = 8


@dataclass
class ADASSolution:
    name: str
    thought: str
    code: str
    generation: int | str = "initial"
    fitness: float | None = None
    validation_scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "thought": self.thought,
            "code": self.code,
            "generation": self.generation,
            "fitness": self.fitness,
            "validation_scores": list(self.validation_scores),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ADASSolution":
        return cls(
            name=str(payload.get("name") or "Unnamed Agent"),
            thought=str(payload.get("thought") or ""),
            code=str(payload.get("code") or ""),
            generation=payload.get("generation", "initial"),
            fitness=(
                float(payload["fitness"])
                if payload.get("fitness") is not None and payload.get("fitness") != ""
                else None
            ),
            validation_scores=[float(v) for v in payload.get("validation_scores") or []],
            metadata=dict(payload.get("metadata") or {}),
        )

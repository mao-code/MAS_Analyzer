from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class DescriptorHook(Protocol):
    def on_monitor(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        """Called by the descriptor monitor node after each fan-in step."""

    def on_finalize(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        """Called during finalize to emit final metrics/annotations."""


@dataclass
class NullDescriptor:
    """Default no-op descriptor hook."""

    monitor_records: list[dict[str, Any]] = field(default_factory=list)

    def on_monitor(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        record = {
            "phase": snapshot.get("phase"),
            "round_index": snapshot.get("round_index"),
            "message_count": snapshot.get("message_count", 0),
            "dispatch_id": snapshot.get("dispatch_id", -1),
        }
        self.monitor_records.append(record)
        return record

    def on_finalize(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        return {
            "monitor_steps": len(self.monitor_records),
            "final_phase": snapshot.get("phase"),
            "final_round_index": snapshot.get("round_index"),
        }

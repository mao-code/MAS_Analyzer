from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

EVENT_TYPES = {
    "plan",
    "act",
    "tool_call",
    "tool_result",
    "verify",
    "revise",
    "finalize",
    "error",
}

REQUIRED_FIELDS = {
    "timestamp_start",
    "timestamp_end",
    "actor",
    "event_type",
    "payload",
    "token_in",
    "token_out",
    "latency_ms",
    "cost_usd",
}


@dataclass(frozen=True)
class TraceEvent:
    """Stable trace event schema.

    This schema is the backbone for reproducible metric extraction.
    """

    timestamp_start: float
    timestamp_end: float
    actor: str
    event_type: str
    payload: Dict[str, Any]
    token_in: int
    token_out: int
    latency_ms: float
    cost_usd: float
    state_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.timestamp_end - self.timestamp_start)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict: bool = True) -> "TraceEvent":
        validate_event_dict(data, strict=strict)

        payload = data.get("payload", {})
        if payload is None:
            payload = {}

        extra = {k: v for k, v in data.items() if k not in REQUIRED_FIELDS and k != "state_id"}

        return cls(
            timestamp_start=float(data["timestamp_start"]),
            timestamp_end=float(data["timestamp_end"]),
            actor=str(data["actor"]),
            event_type=str(data["event_type"]),
            payload=dict(payload),
            token_in=int(data["token_in"]),
            token_out=int(data["token_out"]),
            latency_ms=float(data["latency_ms"]),
            cost_usd=float(data["cost_usd"]),
            state_id=data.get("state_id"),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "actor": self.actor,
            "event_type": self.event_type,
            "payload": self.payload,
            "token_in": self.token_in,
            "token_out": self.token_out,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }
        if self.state_id is not None:
            data["state_id"] = self.state_id
        for key, value in self.extra.items():
            if key not in data:
                data[key] = value
        return data


def validate_event_dict(data: Dict[str, Any], *, strict: bool = True) -> None:
    missing = REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"TraceEvent missing required fields: {sorted(missing)}")

    event_type = data.get("event_type")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Invalid event_type '{event_type}'. Expected one of {sorted(EVENT_TYPES)}")

    if strict:
        if data.get("timestamp_end") < data.get("timestamp_start"):
            raise ValueError("timestamp_end must be >= timestamp_start")
        if int(data.get("token_in", 0)) < 0 or int(data.get("token_out", 0)) < 0:
            raise ValueError("token_in/token_out must be non-negative")
        if float(data.get("latency_ms", 0.0)) < 0:
            raise ValueError("latency_ms must be non-negative")
        if float(data.get("cost_usd", 0.0)) < 0:
            raise ValueError("cost_usd must be non-negative")


def validate_trace_events(events: List[TraceEvent], *, strict: bool = True) -> None:
    if not events:
        raise ValueError("Trace must contain at least one event")
    if strict:
        last_end = events[0].timestamp_start
        for event in events:
            if event.timestamp_start < last_end:
                raise ValueError("Trace events must be non-decreasing in timestamp_start")
            last_end = event.timestamp_end

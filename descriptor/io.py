from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .schema import TraceEvent, validate_trace_events


def read_trace_jsonl(path: str | Path, *, strict: bool = True) -> List[TraceEvent]:
    events: List[TraceEvent] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            events.append(TraceEvent.from_dict(data, strict=strict))
    if strict:
        validate_trace_events(events, strict=True)
    return events


def write_trace_jsonl(events: Iterable[TraceEvent], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), sort_keys=True))
            handle.write("\n")


@dataclass
class TraceLogger:
    """Minimal trace logger for systems emitting events."""

    events: List[TraceEvent] = field(default_factory=list)

    def log_event(self, event: TraceEvent) -> None:
        self.events.append(event)

    def log(
        self,
        *,
        timestamp_start: float,
        timestamp_end: float,
        actor: str,
        event_type: str,
        payload: dict,
        token_in: int,
        token_out: int,
        latency_ms: float,
        cost_usd: float,
        state_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            actor=actor,
            event_type=event_type,
            payload=payload,
            token_in=token_in,
            token_out=token_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            state_id=state_id,
            extra=extra or {},
        )
        self.events.append(event)
        return event

    def flush(self, path: str | Path) -> None:
        write_trace_jsonl(self.events, path)

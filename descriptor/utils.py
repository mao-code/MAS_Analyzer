from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Optional

from .schema import TraceEvent

SUCCESS_STATUSES = {"success", "ok", "pass", "passed", "complete", "completed"}
FAIL_STATUSES = {"fail", "failed", "error", "timeout", "cancelled", "canceled"}


def is_tool_error(event: TraceEvent) -> bool:
    if event.event_type == "tool_result":
        payload = event.payload or {}
        if payload.get("error") is True:
            return True
        status = str(payload.get("status", "")).lower()
        if status in FAIL_STATUSES:
            return True
        if payload.get("error_code") or payload.get("exception"):
            return True
    if event.event_type == "error":
        payload = event.payload or {}
        source = str(payload.get("source", "")).lower()
        if source == "tool":
            return True
    return False


def extract_tool_name(event: TraceEvent) -> Optional[str]:
    payload = event.payload or {}
    for key in ("tool_name", "tool", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def infer_completion(events: Iterable[TraceEvent]) -> bool:
    for event in events:
        if event.event_type == "finalize":
            return True
        payload = event.payload or {}
        if payload.get("completed") is True:
            return True
        if payload.get("artifact_hash"):
            return True
    return False


def infer_success(events: Iterable[TraceEvent]) -> bool:
    success: Optional[bool] = None
    error_seen = False
    for event in events:
        if event.event_type == "error":
            error_seen = True
        if event.event_type == "finalize":
            payload = event.payload or {}
            if isinstance(payload.get("success"), bool):
                success = payload["success"]
                continue
            status = payload.get("status")
            if status is not None:
                status_str = str(status).lower()
                if status_str in SUCCESS_STATUSES:
                    success = True
                elif status_str in FAIL_STATUSES:
                    success = False
                continue
            success = True
    if success is None and error_seen:
        success = False
    if success is None:
        success = False
    return success


def compute_loop_score(events: Iterable[TraceEvent]) -> float:
    events_list = list(events)
    state_ids = [event.state_id for event in events_list if event.state_id]
    if len(state_ids) >= 2:
        counts = Counter(state_ids)
        duplicates = sum(count - 1 for count in counts.values())
        return duplicates / len(state_ids)
    if len(events_list) < 2:
        return 0.0
    seen = set()
    repeated = 0
    total = 0
    for idx in range(len(events_list) - 1):
        pair = (events_list[idx].event_type, events_list[idx + 1].event_type)
        total += 1
        if pair in seen:
            repeated += 1
        else:
            seen.add(pair)
    return repeated / total if total else 0.0


def compute_avg_branching(events: Iterable[TraceEvent]) -> float:
    events_list = list(events)
    if len(events_list) < 2:
        return 0.0
    transitions: Dict[str, set[str]] = {}
    for idx in range(len(events_list) - 1):
        src = events_list[idx].event_type
        dst = events_list[idx + 1].event_type
        transitions.setdefault(src, set()).add(dst)
    if not transitions:
        return 0.0
    return sum(len(next_set) for next_set in transitions.values()) / len(transitions)


def compute_failure_mode_hist(events: Iterable[TraceEvent]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for event in events:
        if event.event_type == "error" or is_tool_error(event):
            payload = event.payload or {}
            code = payload.get("error_code") or payload.get("error") or payload.get("exception")
            code_str = str(code) if code is not None else "unknown"
            hist[code_str] = hist.get(code_str, 0) + 1
    return hist

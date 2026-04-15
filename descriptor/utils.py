from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .schema import TraceEvent

SUCCESS_STATUSES = {"success", "ok", "pass", "passed"}
COMPLETION_STATUSES = SUCCESS_STATUSES | {"complete", "completed", "done", "finalized"}
FAIL_STATUSES = {"fail", "failed", "error", "timeout", "cancelled", "canceled"}
SYSTEM_MEDIATED_PACKET_KINDS = frozenset(
    {
        "task_package",
        "orchestrator_feedback",
        "specialist_report",
        "peer_summary",
        "root_task_package",
        "manager_task_package",
        "child_report",
        "manager_report",
    }
)


@dataclass(frozen=True)
class CommunicationCounts:
    total: int
    agent_to_agent: int
    system_mediated: int


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


def extract_tool_name(event: TraceEvent) -> str | None:
    payload = event.payload or {}
    for key in ("tool_name", "tool", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def infer_completion(
    events: Iterable[TraceEvent],
    *,
    final_answer: str | None = None,
    run_metadata: Mapping[str, object] | None = None,
) -> bool:
    metadata = run_metadata or {}

    explicit_completed = metadata.get("completed")
    if isinstance(explicit_completed, bool):
        return explicit_completed

    terminated = metadata.get("terminated")
    truncated = metadata.get("truncated")
    if isinstance(terminated, bool) or isinstance(truncated, bool):
        return bool(terminated) and not bool(truncated)

    for key in ("error", "exception", "agentbench_error"):
        value = metadata.get(key)
        if value not in (None, "", False):
            return False

    for key in ("status", "final_status", "agentbench_status"):
        value = metadata.get(key)
        if value in (None, ""):
            continue
        status = str(value).strip().lower()
        if status in FAIL_STATUSES:
            return False
        if status in COMPLETION_STATUSES:
            return True

    if str(final_answer or "").strip():
        return True

    for event in events:
        if event.event_type == "finalize":
            payload = event.payload or {}
            status = str(payload.get("status", "")).strip().lower()
            if status in FAIL_STATUSES:
                return False
            if status in COMPLETION_STATUSES:
                return True
            if str(payload.get("final_answer", "")).strip():
                return True
            return True
        payload = event.payload or {}
        if payload.get("completed") is True:
            return True
        if payload.get("artifact_hash"):
            return True
    return False


def infer_success(events: Iterable[TraceEvent]) -> bool:
    success: bool | None = None
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
    transitions: dict[str, set[str]] = {}
    for idx in range(len(events_list) - 1):
        src = events_list[idx].event_type
        dst = events_list[idx + 1].event_type
        transitions.setdefault(src, set()).add(dst)
    if not transitions:
        return 0.0
    return sum(len(next_set) for next_set in transitions.values()) / len(transitions)


def compute_failure_mode_hist(events: Iterable[TraceEvent]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for event in events:
        if event.event_type == "error" or is_tool_error(event):
            payload = event.payload or {}
            code = payload.get("error_code") or payload.get("error") or payload.get("exception")
            code_str = str(code) if code is not None else "unknown"
            hist[code_str] = hist.get(code_str, 0) + 1
    return hist


def _extract_recipients(payload: dict[str, object], *, sender: str) -> set[str]:
    recipients: set[str] = set()
    for key in (
        "to",
        "recipients",
        "receiver",
        "recipient",
        "target_agent",
        "to_agent",
    ):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            candidate = value.strip()
            if candidate and candidate != sender:
                recipients.add(candidate)
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                candidate = str(item).strip()
                if candidate and candidate != sender:
                    recipients.add(candidate)
            continue
    return recipients


def compute_communication_counts(events: Iterable[TraceEvent]) -> CommunicationCounts:
    agent_to_agent = 0
    system_mediated = 0
    for event in events:
        payload = event.payload or {}
        if event.event_type != "tool_call":
            continue
        if str(payload.get("tool_name", "")) != "inter_agent_send":
            continue
        sender = str(event.actor or "").strip()
        recipients = _extract_recipients(payload, sender=sender)
        edge_count = len(recipients)
        if edge_count == 0:
            continue
        packet_kind = str(payload.get("kind", "")).strip().lower()
        if packet_kind in SYSTEM_MEDIATED_PACKET_KINDS or sender == "system":
            system_mediated += edge_count
        elif sender:
            agent_to_agent += edge_count
    return CommunicationCounts(
        total=agent_to_agent + system_mediated,
        agent_to_agent=agent_to_agent,
        system_mediated=system_mediated,
    )


def compute_communication_count(events: Iterable[TraceEvent]) -> int:
    return compute_communication_counts(events).total


def compute_handoff_count(events: Iterable[TraceEvent]) -> int:
    actors = [
        str(event.actor).strip()
        for event in events
        if str(event.actor).strip() and str(event.actor).strip() != "system"
    ]
    if len(actors) < 2:
        return 0
    return sum(1 for idx in range(1, len(actors)) if actors[idx] != actors[idx - 1])

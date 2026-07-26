"""Safe derived tools for stateful self-evolved runs."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

_CALENDAR_ACTION_RE = re.compile(
    r"\b(?:book|schedule|create|arrange|set\s+up)\b.{0,120}"
    r"\b(?:meeting|event|appointment|call)\b"
    r"|\b(?:meeting|event|appointment|call)\b.{0,120}"
    r"\b(?:book|schedule|create|arrange|set\s+up)\b",
    re.IGNORECASE | re.DOTALL,
)
_FLEXIBLE_TIME_RE = re.compile(
    r"\b(?:first|earliest|next)\b.{0,80}\b(?:free|available)\b"
    r"|\b(?:free|available)\b.{0,80}\b(?:time|slot)\b"
    r"|\bwhen\s+(?:i(?:'m|\s+am)|we(?:'re|\s+are))\s+(?:free|available)\b",
    re.IGNORECASE | re.DOTALL,
)
_FAILED_MUTATION_OUTPUT_RE = re.compile(
    r"\b(?:not provided|missing required|missing argument|required argument)"
    r"|\b(?:not found|does not exist|failed|failure)\b",
    re.IGNORECASE,
)


def _user_prompt_text(task_prompt: Any) -> str:
    if not isinstance(task_prompt, list):
        return str(task_prompt or "")
    user_parts: list[str] = []
    for item in task_prompt:
        if not isinstance(item, dict) or str(item.get("role", "")).casefold() != "user":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            user_parts.append(content)
        elif isinstance(content, list):
            user_parts.extend(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("text")
            )
    return "\n".join(user_parts)


def calendar_scheduling_mode(task_prompt: Any) -> str:
    """Classify explicit calendar-create intent as ``fixed`` or ``flexible``.

    The derived availability verifier is appropriate only when the user asks the
    system to choose a free slot. A request that supplies its own time must not
    acquire an extra availability precondition.
    """

    text = _user_prompt_text(task_prompt)
    if not _CALENDAR_ACTION_RE.search(text):
        return ""
    if _FLEXIBLE_TIME_RE.search(text):
        return "flexible"
    return "fixed"


def successful_mutation_record(
    record: dict[str, Any], mutation_tool_names: set[str]
) -> bool:
    """Return whether a tool record proves a mutation completed successfully."""

    if str(record.get("tool_name", "")) not in mutation_tool_names:
        return False
    if str(record.get("status", "")).casefold() not in {"completed", "ok", "success"}:
        return False
    if record.get("error"):
        return False
    output = record.get("output")
    if isinstance(output, dict) and (
        output.get("success") is False or output.get("error")
    ):
        return False
    return not (
        isinstance(output, str) and _FAILED_MUTATION_OUTPUT_RE.search(output)
    )


def parse_duration_minutes(raw: Any, default: int = 30) -> int:
    """Parse the duration spellings models actually produce into whole minutes.

    Accepts plain minutes ("30", 30, 30.0), unit suffixes ("30m", "30 minutes",
    "2h", "2 hours", "1.5 hours"), clock spans ("1:30", "01:30:00"), and ISO-8601
    ("PT1H30M"). Empty input falls back to ``default``; unrecognizable text raises
    ValueError naming the accepted formats, so the tool error is actionable.
    """
    if raw is None:
        return default
    text = str(raw).strip().lower()
    if not text:
        return default
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return max(1, round(float(text)))
    clock = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if clock:
        return max(1, int(clock.group(1)) * 60 + int(clock.group(2)))
    units = re.fullmatch(
        r"(?:pt)?\s*(?:(\d+(?:\.\d+)?)\s*h(?:ours?|rs?)?)?\s*"
        r"(?:(\d+(?:\.\d+)?)\s*m(?:in(?:ute)?s?)?)?",
        text,
    )
    if units and (units.group(1) or units.group(2)):
        hours = float(units.group(1) or 0)
        minutes = float(units.group(2) or 0)
        return max(1, round(hours * 60 + minutes))
    raise ValueError(
        f"unrecognized duration {raw!r}; use minutes such as '30', or '30m', "
        "'1:30:00', '2 hours', 'PT1H30M'"
    )


def augment_with_transaction_tools(
    tools: list[dict[str, Any]], *, task_prompt: Any = None
) -> list[dict[str, Any]]:
    """Add read-only deterministic verifiers composed from available APIs."""

    augmented = list(tools)
    by_name = {str(tool.get("name", "")): tool for tool in tools if isinstance(tool, dict)}
    search = by_name.get("calendar.search_events")
    if search is None or not callable(search.get("handler")):
        return augmented
    if "calendar.find_first_available_slot" in by_name:
        return augmented
    if task_prompt is not None and calendar_scheduling_mode(task_prompt) != "flexible":
        return augmented

    search_handler = search["handler"]

    def find_first_available_slot(arguments: dict[str, Any]) -> dict[str, Any]:
        time_min = str(arguments.get("time_min", "")).strip()
        time_max = str(arguments.get("time_max", "")).strip()
        duration_minutes = parse_duration_minutes(arguments.get("duration"))
        window_start = datetime.fromisoformat(time_min)
        window_end = datetime.fromisoformat(time_max)
        records = search_handler({"query": "", "time_min": time_min, "time_max": time_max})
        if not isinstance(records, list):
            records = []

        intervals: list[tuple[datetime, datetime, str]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                start = datetime.fromisoformat(str(record.get("event_start", "")))
                minutes = int(float(str(record.get("duration", "0"))))
            except (TypeError, ValueError):
                continue
            intervals.append(
                (start, start + timedelta(minutes=minutes), str(record.get("event_id", "")))
            )
        intervals.sort(key=lambda item: item[0])

        cursor = window_start
        needed = timedelta(minutes=duration_minutes)
        considered: list[dict[str, str]] = []
        for start, end, event_id in intervals:
            considered.append(
                {
                    "event_id": event_id,
                    "start": start.isoformat(sep=" "),
                    "end": end.isoformat(sep=" "),
                }
            )
            if cursor + needed <= start:
                break
            if end > cursor:
                cursor = end
        available = cursor + needed <= window_end
        return {
            "available": available,
            "event_start": cursor.isoformat(sep=" ") if available else None,
            "duration": str(duration_minutes),
            "considered_intervals": considered,
        }

    augmented.append(
        {
            "name": "calendar.find_first_available_slot",
            "description": (
                "Read-only verifier that searches the calendar and deterministically returns "
                "the first gap large enough for the requested duration. Prefer this over "
                "manually inferring availability from event starts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {"type": "string"},
                    "time_max": {"type": "string"},
                    "duration": {
                        "type": "string",
                        "description": (
                            "Meeting length in whole minutes, e.g. '30'. "
                            "'30m', '1:30:00', '2 hours', and 'PT1H30M' are also accepted."
                        ),
                    },
                },
                "required": ["time_min", "time_max", "duration"],
            },
            "handler": find_first_available_slot,
        }
    )
    return augmented

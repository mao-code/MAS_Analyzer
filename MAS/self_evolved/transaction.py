"""Safe derived tools for stateful self-evolved runs."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def augment_with_transaction_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add read-only deterministic verifiers composed from available APIs."""

    augmented = list(tools)
    by_name = {str(tool.get("name", "")): tool for tool in tools if isinstance(tool, dict)}
    search = by_name.get("calendar.search_events")
    if search is None or not callable(search.get("handler")):
        return augmented
    if "calendar.find_first_available_slot" in by_name:
        return augmented

    search_handler = search["handler"]

    def find_first_available_slot(arguments: dict[str, Any]) -> dict[str, Any]:
        time_min = str(arguments.get("time_min", "")).strip()
        time_max = str(arguments.get("time_max", "")).strip()
        duration_minutes = int(str(arguments.get("duration", "30")).strip())
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
                    "duration": {"type": "string"},
                },
                "required": ["time_min", "time_max", "duration"],
            },
            "handler": find_first_available_slot,
        }
    )
    return augmented

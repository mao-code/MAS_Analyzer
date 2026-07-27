"""Small shared helpers for the command-line layer."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 fallback
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _log_progress(message: str) -> None:
    print(f"[{_now_stamp()}] {message}", flush=True)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OutputPaths:
    output_layout: str
    experiment_id: str
    experiment_root: Path
    benchmark_root: Path
    run_root: Path
    system_label: str


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    items = [item.strip() for item in str(raw).split(",")]
    values = [int(item) for item in items if item]
    return values or None


def _parse_str_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    return values or None


def _write_summary_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sanitized_row = {
                key: ("" if isinstance(value, float) and math.isnan(value) else value)
                for key, value in row.items()
            }
            writer.writerow(sanitized_row)


def _redact_secrets(data: Any, *, parent_key: str = "") -> Any:
    secret_markers = ("api_key", "token", "secret", "password")
    key_lower = parent_key.lower()

    if isinstance(data, dict):
        return {key: _redact_secrets(value, parent_key=str(key)) for key, value in data.items()}
    if isinstance(data, list):
        return [_redact_secrets(value, parent_key=parent_key) for value in data]
    if isinstance(data, tuple):
        return tuple(_redact_secrets(value, parent_key=parent_key) for value in data)
    if isinstance(data, str) and any(marker in key_lower for marker in secret_markers):
        return "***REDACTED***" if data else ""
    return data


def _prompt_preview(prompt: Any, *, limit: int = 280) -> str:
    if isinstance(prompt, list):
        parts = []
        for item in prompt[:6]:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            parts.append(f"{role}: {content}")
        text = "\n".join(parts)
    else:
        text = str(prompt)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _normalized_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _text_preview(text: Any, *, limit: int = 220) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _append_markdown_fence(lines: list[str], content: Any, *, language: str = "text") -> None:
    text = str(content)
    fence = "```"
    while fence in text:
        fence += "`"
    lines.append(f"{fence}{language}")
    lines.append(text)
    lines.append(fence)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

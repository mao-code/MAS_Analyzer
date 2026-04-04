from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_NON_ANSWER_PREFIXES = (
    "thought:",
    "analysis:",
    "thinking:",
    "plan:",
    "scratchpad:",
    "note:",
)
_NON_ANSWER_EXACT = {
    "thought",
    "analysis",
    "thinking",
    "plan",
    "scratchpad",
    "none",
    "null",
}
_PLAN_INTROS = (
    "to complete the request, i will execute the following plan",
    "i will execute the following plan",
    "i'll execute the following plan",
    "the following plan will be used",
    "here is the plan",
    "search strategy:",
)
_DIRECT_ANSWER_KEYS = (
    "final_answer",
    "answer",
    "result",
    "winner",
    "institution_name",
    "individual_name",
    "entity_name",
    "organization_name",
    "company_name",
    "school_name",
    "university_name",
    "candidate_name",
    "name",
    "institution",
    "individual",
    "entity",
    "candidate",
    "brand",
    "city",
    "country",
)
_PLANNING_KEYS = {
    "plan",
    "plans",
    "sub_questions",
    "search_strategy",
    "search_queries",
    "queries",
    "steps",
    "step",
    "task_package",
    "task_packages",
    "tool",
    "tools",
    "parameters",
    "purpose",
    "dependencies",
    "description",
    "rationale",
}
_SUPPORT_KEYS = {
    "summary",
    "critique",
    "revision_request",
    "confidence",
    "unresolved_issues",
    "evidence_summary",
    "verification_details",
    "evidence",
    "citations",
    "sources",
    "details",
    "metadata",
    "notes",
    "reasoning",
    "explanation",
    "location",
}


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _looks_like_plan_text(raw: str) -> bool:
    lowered = raw.strip().lower()
    if not lowered:
        return False
    if lowered.startswith(_NON_ANSWER_PREFIXES):
        return True
    if lowered in _NON_ANSWER_EXACT:
        return True
    if lowered.startswith(_PLAN_INTROS):
        return True
    if lowered.startswith(
        (
            "{'plan':",
            '{"plan":',
            "{'sub_questions':",
            '{"sub_questions":',
            "{'search_strategy':",
            '{"search_strategy":',
            "[{'tool':",
            '[{"tool":',
        )
    ):
        return True
    return False


def _parse_structured_value(raw: str) -> Any | None:
    raw = raw.strip()
    if not raw or raw[0] not in "{[" or raw[-1] not in "}]":
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(raw)
        except Exception:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _is_plan_mapping(value: Mapping[str, Any]) -> bool:
    keys = {str(key).strip().lower() for key in value.keys()}
    if not keys:
        return False
    has_direct_answer = any(key in _DIRECT_ANSWER_KEYS or key.endswith("_name") for key in keys)
    has_planning = any(key in _PLANNING_KEYS for key in keys)
    if has_planning and not has_direct_answer:
        return True
    return keys <= (_PLANNING_KEYS | _SUPPORT_KEYS)


def _extract_from_mapping(value: Mapping[str, Any]) -> str:
    lowered = {str(key).strip().lower(): item for key, item in value.items()}

    for key in _DIRECT_ANSWER_KEYS:
        if key not in lowered:
            continue
        candidate = extract_substantive_answer(lowered[key])
        if candidate:
            return candidate

    for key, item in lowered.items():
        if key.endswith("_name") and key not in _PLANNING_KEYS and key not in _SUPPORT_KEYS:
            candidate = extract_substantive_answer(item)
            if candidate:
                return candidate

    if _is_plan_mapping(lowered):
        return ""
    return ""


def _extract_from_sequence(value: Sequence[Any]) -> str:
    items = list(value)
    if not items:
        return ""
    if len(items) == 1:
        return extract_substantive_answer(items[0])
    if all(_is_scalar(item) for item in items):
        normalized = [_normalized_text(item) for item in items if _normalized_text(item)]
        return "; ".join(normalized)
    if all(isinstance(item, Mapping) and _is_plan_mapping(item) for item in items):
        return ""
    return ""


def extract_substantive_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return _normalized_text(_extract_from_mapping(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _normalized_text(_extract_from_sequence(value))

    raw = _normalized_text(value)
    if not raw or _looks_like_plan_text(raw):
        return ""

    parsed = _parse_structured_value(raw)
    if parsed is not None:
        return _normalized_text(extract_substantive_answer(parsed))

    return raw


def has_substantive_answer(value: Any) -> bool:
    return bool(extract_substantive_answer(value))

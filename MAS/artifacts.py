from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from typing_extensions import TypedDict

from answer_utils import extract_substantive_answer


class ArtifactRecord(TypedDict, total=False):
    """Structured output produced by one agent stage."""

    artifact_id: str
    dispatch_id: int
    node_name: str
    stage_role: str
    round_index: int
    discussion_index: int
    agent_id: str
    role: str
    answer: str
    summary: str
    critique: str
    revision_request: str
    confidence: float
    unresolved_issues: list[str]
    evidence_summary: list[str]
    source_artifact_ids: list[str]
    status: str
    raw_response: str
    tool_records: list[dict[str, Any]]
    llm: dict[str, Any]


def _iter_artifact_ids(value: Any) -> list[str]:
    raw_values = value if isinstance(value, list) else [value]
    artifact_ids: list[str] = []
    for raw in raw_values:
        for part in str(raw or "").split(","):
            cleaned = part.strip()
            if cleaned:
                artifact_ids.append(cleaned)
    return artifact_ids


class RelayPacket(TypedDict, total=False):
    """Bounded structured communication payload passed between agents."""

    message_id: str
    dispatch_id: int
    sender: str
    recipients: list[str]
    kind: str
    phase: str
    round: int
    discussion_index: int
    artifact_id: str | None
    content: str
    payload: dict[str, Any]


class TerminationDecision(TypedDict, total=False):
    """Explicit termination decision used by conditional routing."""

    should_stop: bool
    next_step: str
    reason: str
    reason_detail: str
    stage_name: str
    round_index: int
    discussion_index: int
    consensus_ratio: float
    consensus_mode: str
    consensus_source: str
    consensus_signature: str
    consensus_count: int
    consensus_valid_count: int
    consensus_groups: list[list[int]]
    consensus_explanation: str
    consensus_is_substantive: bool | None
    consensus_gate_blocked: bool
    consensus_gate_reason: str
    progress_source: str
    progress_status: str | None
    expected_improvement: str | None
    progress_explanation: str
    average_confidence: float
    mean_delta: float | None
    valid_artifact_count: int
    control_token_in: int
    control_token_out: int
    control_cost_usd: float
    control_latency_ms: float


def build_artifact(
    *,
    text: str,
    artifact_id: str,
    dispatch_id: int,
    node_name: str,
    stage_role: str,
    round_index: int,
    discussion_index: int,
    agent_id: str,
    role: str,
    source_artifact_ids: list[str],
    tool_records: list[dict[str, Any]],
    llm_payload: dict[str, Any],
) -> ArtifactRecord:
    """Coerce raw model text into the shared artifact schema."""

    payload = _extract_json_payload(text)
    parsed = payload is not None

    answer_value = (
        ((payload.get("answer_artifact") or payload.get("answer")) if parsed else None) or text
    )
    if isinstance(answer_value, (dict, list)):
        answer_value = json.dumps(answer_value, ensure_ascii=False, default=str, sort_keys=True)
    answer = _bounded_text(answer_value, max_chars=6000)
    summary = _bounded_text(
        (payload.get("summary") if parsed else None) or answer,
        max_chars=400,
    )
    critique = _bounded_text(payload.get("critique") if parsed else None, max_chars=1200)
    revision_request = _bounded_text(
        payload.get("revision_request") if parsed else None,
        max_chars=1200,
    )
    unresolved_issues = _coerce_list(payload.get("unresolved_issues") if parsed else None, limit=6)
    evidence_summary = _coerce_list(payload.get("evidence_summary") if parsed else None, limit=6)
    confidence = _coerce_confidence(payload.get("confidence") if parsed else None)

    status = "ok" if parsed else "fallback"
    if not answer.strip():
        answer = _bounded_text(text, max_chars=6000)
        status = "invalid" if not answer.strip() else status

    return ArtifactRecord(
        artifact_id=artifact_id,
        dispatch_id=int(dispatch_id),
        node_name=node_name,
        stage_role=stage_role,
        round_index=int(round_index),
        discussion_index=int(discussion_index),
        agent_id=agent_id,
        role=role,
        answer=answer,
        summary=summary,
        critique=critique,
        revision_request=revision_request,
        confidence=confidence,
        unresolved_issues=unresolved_issues,
        evidence_summary=evidence_summary,
        source_artifact_ids=list(source_artifact_ids),
        status=status,
        raw_response=str(text),
        tool_records=list(tool_records),
        llm=dict(llm_payload),
    )


def packet_payload_from_artifact(
    artifact: ArtifactRecord,
    *,
    max_chars: int = 0,
) -> dict[str, Any]:
    """Create a relay payload from a full artifact.

    Full fidelity by default (``max_chars <= 0``). A positive budget triggers
    structural compaction via :func:`compact_packet_payload` — dropping
    low-priority fields and preferring the agent's own summary — rather than a
    blunt mid-string truncation of every field.
    """

    payload = {
        "artifact_id": artifact.get("artifact_id"),
        "summary": _bounded_text(artifact.get("summary", ""), max_chars=0),
        "answer_artifact": _bounded_text(artifact.get("answer", ""), max_chars=0),
        "critique": _bounded_text(artifact.get("critique", ""), max_chars=0),
        "revision_request": _bounded_text(artifact.get("revision_request", ""), max_chars=0),
        "confidence": float(artifact.get("confidence", 0.5)),
        "unresolved_issues": list(artifact.get("unresolved_issues", [])),
        "evidence_summary": list(artifact.get("evidence_summary", [])),
    }
    return compact_packet_payload(payload, max_chars=int(max_chars))


def packet_content(payload: dict[str, Any], *, max_chars: int = 0) -> str:
    """Render a single-string packet summary for logs and traces.

    Full text by default; a positive ``max_chars`` sentence-bounds the chosen
    field (summary, else answer, else revision request) at a sentence/whitespace
    boundary instead of a mid-token cut.
    """

    for key in ("summary", "answer_artifact", "revision_request"):
        text = _bounded_text(payload.get(key), max_chars=0)
        if text:
            if int(max_chars) <= 0:
                return text
            return _sentence_bounded_text(text, max_chars=int(max_chars))
    return "No bounded content provided."


_COMPACT_LIST_LIMIT = 4
_COMPACT_TEXT_KEYS = ("summary", "answer_artifact", "critique", "revision_request")
_COMPACT_LIST_KEYS = ("evidence_summary", "unresolved_issues")


def compact_packet_payload(payload: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    """Deterministically shrink a relay payload toward ``max_chars`` characters.

    Full fidelity when ``max_chars <= 0``. Over budget, reduce *structurally*
    rather than chopping the answer mid-token: drop ``revision_request``, then
    ``critique``; then drop the long ``answer_artifact`` when a ``summary``
    exists; then trim the evidence/issue lists; and only as a last resort
    sentence-bound the remaining primary text field at a sentence/whitespace
    boundary. The model's answer is never cut mid-token — the whole field is
    dropped in favor of the agent's own summary instead.
    """

    compacted = dict(payload)
    if int(max_chars) <= 0:
        return compacted
    budget = int(max_chars)

    def field_len(value: Any) -> int:
        if isinstance(value, list):
            return sum(len(str(item)) for item in value)
        return len(str(value or ""))

    def total() -> int:
        return sum(
            field_len(compacted.get(key)) for key in (*_COMPACT_TEXT_KEYS, *_COMPACT_LIST_KEYS)
        )

    # 1. Drop low-priority free-text fields first.
    for key in ("revision_request", "critique"):
        if total() <= budget:
            return compacted
        if compacted.get(key):
            compacted[key] = ""

    # 2. Prefer the agent's own summary over the long answer.
    if total() > budget and compacted.get("summary") and compacted.get("answer_artifact"):
        compacted["answer_artifact"] = ""

    # 3. Trim evidence / unresolved lists.
    if total() > budget:
        for key in _COMPACT_LIST_KEYS:
            if compacted.get(key):
                compacted[key] = list(compacted[key])[:_COMPACT_LIST_LIMIT]

    # 4. Last resort: sentence-bound the remaining primary text field.
    if total() > budget:
        for key in ("summary", "answer_artifact"):
            if compacted.get(key):
                compacted[key] = _sentence_bounded_text(compacted[key], max_chars=budget)
                break
    return compacted


def answer_signature(text: str) -> str:
    """Canonicalize an answer for deterministic voting and agreement checks."""

    substantive = extract_substantive_answer(text)
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", substantive.lower())).strip()
    if not normalized:
        return ""
    return normalized


def artifacts_by_id(artifacts: list[ArtifactRecord]) -> dict[str, ArtifactRecord]:
    """Index artifact records by artifact_id."""

    indexed: dict[str, ArtifactRecord] = {}
    for artifact in artifacts:
        artifact_id = str(artifact.get("artifact_id", "")).strip()
        if artifact_id:
            indexed[artifact_id] = artifact
    return indexed


def collect_artifact_lineage(
    artifacts: list[ArtifactRecord],
    selected_artifact_id: str | list[str] | None,
) -> list[ArtifactRecord]:
    """Return the selected artifact lineage in dependency order."""

    indexed = artifacts_by_id(artifacts)
    ordered: list[ArtifactRecord] = []
    seen: set[str] = set()

    def visit(artifact_id: str) -> None:
        if not artifact_id or artifact_id in seen:
            return
        seen.add(artifact_id)
        artifact = indexed.get(artifact_id)
        if artifact is None:
            return
        for source_id in _iter_artifact_ids(artifact.get("source_artifact_ids", [])):
            visit(source_id)
        ordered.append(artifact)

    for artifact_id in _iter_artifact_ids(selected_artifact_id):
        visit(artifact_id)
    return ordered


def collect_lineage_tool_records(
    artifacts: list[ArtifactRecord],
    selected_artifact_id: str | list[str] | None,
) -> list[dict[str, Any]]:
    """Collect tool records from the selected artifact lineage only."""

    collected: list[dict[str, Any]] = []
    for artifact in collect_artifact_lineage(artifacts, selected_artifact_id):
        records = artifact.get("tool_records", [])
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                collected.append(dict(record))
    return collected


def compute_consensus(artifacts: list[ArtifactRecord]) -> dict[str, Any]:
    """Measure current answer agreement among valid artifacts."""

    valid = [artifact for artifact in artifacts if answer_signature(artifact.get("answer", ""))]
    if not valid:
        return {
            "ratio": 0.0,
            "signature": "",
            "count": 0,
            "valid_count": 0,
        }

    counts: dict[str, int] = {}
    for artifact in valid:
        signature = answer_signature(artifact.get("answer", ""))
        counts[signature] = counts.get(signature, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    winner, count = ranked[0]
    return {
        "ratio": count / len(valid),
        "signature": winner,
        "count": count,
        "valid_count": len(valid),
    }


def compute_mean_delta(
    previous_artifacts: list[ArtifactRecord],
    current_artifacts: list[ArtifactRecord],
) -> float | None:
    """Estimate whether revisions materially changed compared with the prior step."""

    previous_by_agent = {
        str(artifact.get("agent_id", "")): artifact for artifact in previous_artifacts
    }
    deltas: list[float] = []
    for artifact in current_artifacts:
        agent_id = str(artifact.get("agent_id", ""))
        previous = previous_by_agent.get(agent_id)
        if previous is None:
            continue
        before = answer_signature(previous.get("answer", ""))
        after = answer_signature(artifact.get("answer", ""))
        if not before and not after:
            deltas.append(0.0)
            continue
        if before == after:
            deltas.append(0.0)
            continue
        similarity = SequenceMatcher(None, before, after).ratio()
        deltas.append(max(0.0, 1.0 - similarity))

    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def average_confidence(artifacts: list[ArtifactRecord]) -> float:
    valid = [float(artifact.get("confidence", 0.5)) for artifact in artifacts]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def latest_artifact_by_agent(artifacts: list[ArtifactRecord]) -> dict[str, ArtifactRecord]:
    """Return each agent's most recent artifact.

    Recency is ordered by (round_index, discussion_index, dispatch_id) and then by
    append order (list position) as the final tie-break. Append order is a robust
    chronological signal for sequential stages; it replaces a previous lexical
    node_name tie-break that only tracked recency by alphabetical accident and could
    in principle select a stale artifact over a newer one.
    """

    latest: dict[str, ArtifactRecord] = {}
    latest_key: dict[str, tuple[int, int, int, int]] = {}
    for index, artifact in enumerate(artifacts):
        agent_id = str(artifact.get("agent_id", ""))
        if not agent_id:
            continue
        key = (
            int(artifact.get("round_index", 0)),
            int(artifact.get("discussion_index", 0)),
            int(artifact.get("dispatch_id", -1)),
            index,
        )
        if agent_id not in latest_key or key >= latest_key[agent_id]:
            latest[agent_id] = artifact
            latest_key[agent_id] = key
    return latest


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    candidates: list[str] = []
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text or "", flags=re.DOTALL)
    candidates.extend(fenced)

    raw = (text or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        candidates.append(raw)

    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(raw[first : last + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _coerce_list(value: Any, *, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in re.split(r"[\n;,]+", value) if item.strip()]
        return items[:limit]
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items[:limit]
    return [str(value).strip()] if str(value).strip() else []


def _coerce_confidence(value: Any) -> float:
    if value is None:
        return 0.5
    try:
        confidence = float(value)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, confidence))


def _bounded_text(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _sentence_bounded_text(value: Any, *, max_chars: int) -> str:
    """Boundary-aware truncation: cut at the last sentence terminator or
    whitespace before ``max_chars`` (never mid-token) and append an ellipsis.
    ``max_chars <= 0`` returns the whitespace-normalized text unchanged."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    budget = max(1, max_chars - 3)
    window = text[:budget]
    cut = max(window.rfind("."), window.rfind("!"), window.rfind("?"))
    if cut >= budget // 2:
        cut += 1  # keep the sentence terminator
    else:
        space = window.rfind(" ")
        cut = space if space > 0 else budget
    snippet = text[:cut].rstrip()
    return (snippet + "...") if snippet else window.rstrip() + "..."

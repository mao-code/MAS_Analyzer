from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from typing_extensions import TypedDict


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

    answer = _bounded_text(
        payload.get("answer_artifact") if parsed else None
        or payload.get("answer") if parsed else None
        or text,
        max_chars=6000,
    )
    summary = _bounded_text(
        payload.get("summary") if parsed else None or answer,
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
    max_chars: int = 320,
) -> dict[str, Any]:
    """Create a bounded relay payload from a full artifact."""

    bounded = max(32, int(max_chars))
    return {
        "artifact_id": artifact.get("artifact_id"),
        "summary": _bounded_text(artifact.get("summary", ""), max_chars=bounded),
        "answer_artifact": _bounded_text(artifact.get("answer", ""), max_chars=bounded),
        "critique": _bounded_text(artifact.get("critique", ""), max_chars=bounded),
        "revision_request": _bounded_text(artifact.get("revision_request", ""), max_chars=bounded),
        "confidence": float(artifact.get("confidence", 0.5)),
        "unresolved_issues": list(artifact.get("unresolved_issues", []))[:4],
        "evidence_summary": list(artifact.get("evidence_summary", []))[:4],
    }


def packet_content(payload: dict[str, Any], *, max_chars: int = 320) -> str:
    """Render a compact single-string packet summary for logs and traces."""

    bounded = max(32, int(max_chars))
    summary = _bounded_text(payload.get("summary"), max_chars=bounded)
    if summary:
        return summary
    answer = _bounded_text(payload.get("answer_artifact"), max_chars=bounded)
    if answer:
        return answer
    revision = _bounded_text(payload.get("revision_request"), max_chars=bounded)
    if revision:
        return revision
    return "No bounded content provided."


def answer_signature(text: str) -> str:
    """Canonicalize an answer for deterministic voting and agreement checks."""

    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()
    return normalized


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
    latest: dict[str, ArtifactRecord] = {}
    for artifact in artifacts:
        agent_id = str(artifact.get("agent_id", ""))
        if not agent_id:
            continue
        key = (
            int(artifact.get("round_index", 0)),
            int(artifact.get("discussion_index", 0)),
            int(artifact.get("dispatch_id", -1)),
            str(artifact.get("node_name", "")),
        )
        current = latest.get(agent_id)
        if current is None:
            latest[agent_id] = artifact
            continue
        current_key = (
            int(current.get("round_index", 0)),
            int(current.get("discussion_index", 0)),
            int(current.get("dispatch_id", -1)),
            str(current.get("node_name", "")),
        )
        if key >= current_key:
            latest[agent_id] = artifact
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
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."

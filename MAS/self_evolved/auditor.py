"""Trace Auditor meta-agent.

Inspects the just-executed turn using process-observable signals only (no
benchmark ground truth is available in-run) and reports detected failure
modes plus whether a topology/context repair is recommended. Mode names
follow the taxonomy in ``scripts/generate_mas_failure_analysis_report.py``
where a process-level analogue exists.

Deterministic heuristics run first; ``audit_mode = "llm_judge"`` adds an
LLM refinement pass with a deterministic fallback (same contract as the
termination judge).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from answer_utils import classify_answer_mode
from descriptor.utils import FAIL_STATUSES

from ..artifacts import ArtifactRecord, _extract_json_payload, compute_consensus
from ..config import SelfEvolvedConfig
from .spec import TopologySpec

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}

_NEGATIVE_EVIDENCE_SNIPPETS = (
    "no evidence",
    "none",
    "not retrieved",
    "not gathered",
    "no search",
    "unknown",
    "insufficient",
)


class TraceAuditorAgent:
    def __init__(self, llm_client: Any, se_config: SelfEvolvedConfig) -> None:
        self.llm_client = llm_client
        self.se_config = se_config

    def audit(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        *,
        turn_index: int,
    ) -> dict[str, Any]:
        detected = self._heuristic_modes(state, spec, turn_index=turn_index)
        report: dict[str, Any] = {
            "turn_index": int(turn_index),
            "source": "heuristic",
            "detected_modes": detected,
            "repair_recommended": any(
                _SEVERITY_RANK.get(str(mode.get("severity", "low")), 0) >= 1 for mode in detected
            ),
            "recommendation": self._recommendation_text(detected),
            "llm": {},
        }
        if self.se_config.audit_mode == "llm_judge":
            report = self._llm_refine(state, spec, report)
        return report

    # -- deterministic heuristics ----------------------------------------------

    def _heuristic_modes(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        *,
        turn_index: int,
    ) -> list[dict[str, Any]]:
        artifacts = [
            artifact
            for artifact in state.get("artifacts", [])
            if int(artifact.get("round_index", -1)) == turn_index
        ]
        contributions = [
            artifact
            for artifact in artifacts
            if str(artifact.get("stage_role", "")) in {"worker", "critic"}
        ]
        aggregations = [
            artifact
            for artifact in artifacts
            if str(artifact.get("stage_role", "")) == "aggregator"
        ]
        modes: list[dict[str, Any]] = []

        # tool_error_cascade: agents whose tool calls failed this turn.
        failures_by_agent: dict[str, int] = {}
        for record in state.get("tool_records_log", []):
            if int(record.get("round_index", -1)) != turn_index:
                continue
            if str(record.get("status", "")).lower() in FAIL_STATUSES:
                agent_id = str(record.get("agent_id", ""))
                failures_by_agent[agent_id] = failures_by_agent.get(agent_id, 0) + 1
        if failures_by_agent:
            total_failures = sum(failures_by_agent.values())
            modes.append(
                {
                    "mode": "tool_error_cascade",
                    "severity": "high" if total_failures >= 2 else "medium",
                    "agent_ids": sorted(failures_by_agent),
                    "detail": (
                        f"{total_failures} failed tool call(s) across "
                        f"{len(failures_by_agent)} agent(s)."
                    ),
                }
            )

        # branch_collapse: contributions that stayed blocked/planning/empty.
        collapsed = [
            str(artifact.get("agent_id", ""))
            for artifact in contributions
            if classify_answer_mode(str(artifact.get("answer", ""))) in {"blocked", "plan", "empty"}
        ]
        if collapsed:
            severity = "high" if len(collapsed) > 1 else "medium"
            modes.append(
                {
                    "mode": "branch_collapse",
                    "severity": severity,
                    "agent_ids": sorted(set(collapsed)),
                    "detail": f"{len(collapsed)} contribution(s) produced no substantive answer.",
                }
            )

        # evidence_lost_before_synthesis: members gathered evidence, the
        # aggregate carries none forward.
        member_evidence = sum(self._evidence_count(a) for a in contributions)
        if aggregations and member_evidence > 0:
            starved = [
                str(artifact.get("agent_id", ""))
                for artifact in aggregations
                if self._evidence_count(artifact) == 0
            ]
            if starved:
                modes.append(
                    {
                        "mode": "evidence_lost_before_synthesis",
                        "severity": "medium",
                        "agent_ids": sorted(set(starved)),
                        "detail": (
                            "Member contributions carried evidence but the synthesis "
                            "artifact lists none."
                        ),
                    }
                )

        # premature_consensus: high agreement with low confidence or open issues.
        if len(contributions) > 1:
            consensus = compute_consensus(contributions)
            unresolved = any(artifact.get("unresolved_issues") for artifact in contributions)
            confidences = [float(a.get("confidence", 0.5)) for a in contributions]
            avg_confidence = sum(confidences) / len(confidences)
            if float(consensus.get("ratio", 0.0)) >= 0.75 and (avg_confidence < 0.5 or unresolved):
                modes.append(
                    {
                        "mode": "premature_consensus",
                        "severity": "medium",
                        "agent_ids": [],
                        "detail": (
                            f"Answer agreement {float(consensus['ratio']):.2f} despite "
                            f"avg confidence {avg_confidence:.2f} and "
                            f"{'open' if unresolved else 'no'} unresolved issues."
                        ),
                    }
                )

        # message_compaction_loss: bounded packets truncated this turn.
        truncated = [
            message
            for message in state.get("messages", [])
            if int(message.get("round", -1)) == turn_index
            and str(message.get("content", "")).endswith("...")
        ]
        if truncated:
            modes.append(
                {
                    "mode": "message_compaction_loss",
                    "severity": "low",
                    "agent_ids": sorted({str(m.get("sender", "")) for m in truncated}),
                    "detail": f"{len(truncated)} relay packet(s) were truncated at the bound.",
                }
            )

        # Tool-call signals for this turn (retrieval coverage + duplicate writes).
        turn_calls = [
            record
            for record in state.get("tool_records_log", [])
            if int(record.get("round_index", -1)) == turn_index
        ]
        run_tools = state.get("tools") or []
        has_search_tool = any(
            isinstance(tool, dict) and str(tool.get("name", "")) == "search" for tool in run_tools
        )
        has_get_document = any(
            isinstance(tool, dict) and str(tool.get("name", "")) == "get_document"
            for tool in run_tools
        )

        # insufficient_search_coverage: a retrieval/search run that under-gathered —
        # searched from snippets but never opened a document, or had too few agents
        # searching for a broad-coverage question. This is the dominant browsecomp
        # failure (a single searcher / a non-searching verifier instead of breadth).
        if has_search_tool or has_get_document:
            searchers = {
                str(record.get("agent_id", ""))
                for record in turn_calls
                if str(record.get("tool_name", "")) == "search"
            }
            search_calls = sum(
                1 for record in turn_calls if str(record.get("tool_name", "")) == "search"
            )
            reads = sum(
                1 for record in turn_calls if str(record.get("tool_name", "")) == "get_document"
            )
            coverage_problem: tuple[str, str] | None = None
            if has_get_document and search_calls > 0 and reads == 0:
                coverage_problem = (
                    "high",
                    "Search ran but no document was opened with get_document; "
                    "answers from snippets alone are unreliable.",
                )
            elif search_calls > 0 and len(searchers) < 2:
                coverage_problem = (
                    "medium",
                    f"Only {len(searchers)} agent(s) searched; broad retrieval needs "
                    "several searchers covering different facets.",
                )
            if coverage_problem is not None:
                modes.append(
                    {
                        "mode": "insufficient_search_coverage",
                        "severity": coverage_problem[0],
                        "agent_ids": sorted(searchers),
                        "detail": coverage_problem[1],
                    }
                )

        # duplicate_state_mutation: the same non-read tool call (tool + arguments) was
        # issued by >= 2 agents this turn. For side-effecting tools (calendar/email/etc.)
        # this double-applies and corrupts the evaluated state (the dominant workbench
        # failure). The executor's per-run dedup net collapses the recorded call; this
        # flags the structural cause so a repair can serialize the write next turn.
        sig_agents: dict[str, set[str]] = {}
        for record in turn_calls:
            name = str(record.get("tool_name", "")).strip()
            if not name or name in {"search", "get_document", "inter_agent_send"}:
                continue
            arguments = record.get("arguments")
            try:
                arg_sig = json.dumps(arguments, sort_keys=True, default=str)
            except Exception:
                arg_sig = str(arguments)
            sig_agents.setdefault(f"{name}({arg_sig})", set()).add(str(record.get("agent_id", "")))
        duplicated = {sig: agents for sig, agents in sig_agents.items() if len(agents) >= 2}
        if duplicated:
            offenders = sorted({agent for agents in duplicated.values() for agent in agents})
            modes.append(
                {
                    "mode": "duplicate_state_mutation",
                    "severity": "high",
                    "agent_ids": offenders,
                    "detail": (
                        f"{len(duplicated)} tool call(s) were issued identically by "
                        "multiple agents; a state-changing call must execute once."
                    ),
                }
            )

        # missing_validator: low-confidence answers with no critic downstream.
        has_validator = any(node.stage_role == "critic" for node in spec.agents) or any(
            group.pattern == "debate" for group in spec.groups
        )
        if contributions and not has_validator:
            confidences = [float(a.get("confidence", 0.5)) for a in contributions]
            if sum(confidences) / len(confidences) < 0.6:
                modes.append(
                    {
                        "mode": "missing_validator",
                        "severity": "medium",
                        "agent_ids": [],
                        "detail": (
                            "No critic/debate stage exists downstream of low-confidence answers."
                        ),
                    }
                )

        return modes

    @staticmethod
    def _evidence_count(artifact: ArtifactRecord) -> int:
        evidence = artifact.get("evidence_summary", [])
        if not isinstance(evidence, list):
            return 0
        count = 0
        for item in evidence:
            text = re.sub(r"\s+", " ", str(item or "")).strip().lower()
            if text and not any(snippet in text for snippet in _NEGATIVE_EVIDENCE_SNIPPETS):
                count += 1
        return count

    @staticmethod
    def _recommendation_text(detected: list[dict[str, Any]]) -> str:
        if not detected:
            return "No repair needed; the turn shows no process failure signals."
        hints = {
            "tool_error_cascade": (
                "expand the failing agent into a star subgroup so tool work is split"
            ),
            "branch_collapse": (
                "expand the blocked agent into a star subgroup with focused subtasks"
            ),
            "evidence_lost_before_synthesis": (
                "widen the aggregator's evidence access so member evidence survives synthesis"
            ),
            "premature_consensus": (
                "convert one branch into a fully-linked debate to challenge the shared answer"
            ),
            "message_compaction_loss": "raise packet bounds for the affected senders",
            "missing_validator": "add a verifier critic or a debate subgroup before synthesis",
            "insufficient_search_coverage": (
                "add searcher workers (parallel, each on a different facet) and open "
                "documents before answering"
            ),
            "duplicate_state_mutation": (
                "serialize the state-changing tool through exactly one executor; collapse "
                "the parallel workers into a chain or singleton"
            ),
        }
        parts = [
            f"{mode['mode']}: {hints.get(str(mode['mode']), 'consider a topology repair')}"
            for mode in detected
        ]
        return "; ".join(parts)

    # -- optional LLM refinement -------------------------------------------------

    def _llm_refine(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._refine_prompt(state, spec, report)
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                agent_type="auditor",
                task_id=str(state.get("task_id", "")),
                run_index=int(state.get("run_index", 0)),
                agent_id="trace_auditor",
                temperature=0.0,
            )
        except Exception:
            logger.warning("Auditor LLM refinement failed; keeping heuristics", exc_info=True)
            return report

        llm_payload = {
            "model": str(result.model),
            "mock_used": bool(result.mock_used),
            "token_in": int(result.token_in),
            "token_out": int(result.token_out),
            "cost_usd": float(result.cost_usd),
        }
        if bool(result.mock_used):
            return {**report, "llm": llm_payload}

        payload = _extract_json_payload(str(result.text or ""))
        if not isinstance(payload, dict) or "repair_recommended" not in payload:
            return {**report, "llm": llm_payload}

        return {
            **report,
            "source": "llm_judge",
            "repair_recommended": bool(payload.get("repair_recommended", False)),
            "recommendation": str(payload.get("recommendation", report["recommendation"]))[:800],
            "llm": llm_payload,
        }

    def _refine_prompt(
        self,
        state: dict[str, Any],
        spec: TopologySpec,
        report: dict[str, Any],
    ) -> list[dict[str, str]]:
        groups_summary = "; ".join(
            f"{group.group_id}({group.pattern}: {', '.join(group.member_ids)})"
            for group in spec.groups
        )
        findings = (
            "\n".join(
                f"- {mode['mode']} [{mode['severity']}] {mode['detail']}"
                for mode in report.get("detected_modes", [])
            )
            or "- (no heuristic findings)"
        )
        system_msg = (
            "You are a trace auditor for a multi-agent system. Given heuristic "
            "process signals from one execution turn, decide whether a single "
            "topology or context repair is worth one extra turn. Be conservative: "
            "recommend repair only when the signals indicate a structural problem "
            "a topology change can fix."
        )
        user_msg = (
            f"Current topology groups: {groups_summary}\n\n"
            f"Heuristic findings for turn {report.get('turn_index', 0)}:\n{findings}\n\n"
            "Return ONLY one JSON object: "
            '{"repair_recommended": true|false, "recommendation": "one sentence"}'
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

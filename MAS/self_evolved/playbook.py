"""Topology Playbook: persistent and in-run memory of topology decisions.

The playbook is a JSON file keyed by benchmark plus coarse deterministic task
features (no embeddings — keys stay auditable).

There are two memory horizons:

- short-term, turn-level memory lives only inside one run and is updated after
  each audited turn from process signals;
- long-term, session-level memory is persisted post-hoc by
  ``scripts/update_topology_playbook.py`` after joining run candidates with
  ``run_*.eval.json`` (``benchmark.evaluate(...).success`` remains the only
  correctness authority).

Runs never write the persistent playbook file. This keeps parallel experiments
independent while still giving the online repair planner access to fresh
turn-level topology/context signals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PLAYBOOK_VERSION = 1
_MAX_FAILURE_MEMORIES = 8
_MAX_ENTRIES = 200
_MAX_SHORT_TERM_ENTRIES = 8


def task_key(benchmark_name: str, task: Any, *, tools_available: bool) -> str:
    """Deterministic coarse task key: benchmark + tool availability + prompt size."""

    prompt = str(getattr(task, "prompt", "") or "")
    if len(prompt) < 400:
        size = "short"
    elif len(prompt) < 1600:
        size = "medium"
    else:
        size = "long"
    tools = "tools" if tools_available else "no_tools"
    return f"{benchmark_name or 'unknown'}::{tools}::{size}"


def pattern_sketch(spec_payload: dict[str, Any]) -> dict[str, Any]:
    """Compact plan-DSL-shaped sketch of a TopologySpec payload."""

    groups = spec_payload.get("groups", [])
    root_id = spec_payload.get("root_group_id", "")
    root = next((g for g in groups if g.get("group_id") == root_id), {})
    expansions = [
        {
            "parent_agent_id": group.get("parent_agent_id"),
            "pattern": group.get("pattern"),
            "num_subagents": len(group.get("member_ids", [])),
        }
        for group in groups
        if group.get("parent_agent_id")
    ]
    return {
        "pattern": root.get("pattern", ""),
        "num_agents": len(spec_payload.get("agents", [])),
        "expansions": expansions,
    }


class TopologyPlaybook:
    def __init__(self, path: str | Path, payload: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        payload = payload or {"version": PLAYBOOK_VERSION, "entries": []}
        self.version = int(payload.get("version", PLAYBOOK_VERSION))
        entries = payload.get("entries", [])
        self.entries: list[dict[str, Any]] = [
            dict(entry) for entry in entries if isinstance(entry, dict)
        ]

    @classmethod
    def load(cls, path: str | Path) -> TopologyPlaybook:
        path = Path(path)
        if not path.exists():
            return cls(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("playbook root must be a JSON object")
            return cls(path, payload)
        except Exception:
            logger.warning("Failed to load playbook at %s; using empty", path, exc_info=True)
            return cls(path)

    def save(self) -> None:
        payload = {"version": self.version, "entries": self.entries[:_MAX_ENTRIES]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    # -- read side (used by the planner during runs) --------------------------

    def lookup(self, benchmark_name: str, key: str, *, limit: int = 5) -> list[dict[str, Any]]:
        """Entries for the exact key first, then other entries for the benchmark."""

        exact = [entry for entry in self.entries if entry.get("key") == key]
        same_benchmark = [
            entry
            for entry in self.entries
            if entry.get("benchmark") == benchmark_name and entry.get("key") != key
        ]
        return (exact + same_benchmark)[:limit]

    @staticmethod
    def planner_view(entry: dict[str, Any]) -> dict[str, Any]:
        """Bounded shape fed into planner prompts."""

        scope = str(entry.get("scope", "long_term"))
        support = entry.get("support", {})
        runs = int(support.get("runs", 0))
        successes = int(support.get("successes", 0))
        if scope == "short_term":
            support_text = "turn-level process memory"
        else:
            support_text = f"{successes}/{runs} successful"
        memories = entry.get("failure_memories", [])
        mutation_notes = "; ".join(
            f"{memory.get('mode')}->{memory.get('mutation')}({memory.get('outcome')})"
            for memory in memories[:3]
            if isinstance(memory, dict)
        )
        return {
            "scope": scope,
            "key": str(entry.get("key", "")),
            "turn_index": entry.get("turn_index"),
            "pattern": entry.get("pattern", {}),
            "support": support_text,
            "notes": str(entry.get("notes", ""))[:200],
            "mutation": mutation_notes,
        }

    # -- write side (post-hoc updater only) ------------------------------------

    def merge_candidate(self, candidate: dict[str, Any], *, success: bool) -> None:
        key = str(candidate.get("key", "")).strip()
        if not key:
            return
        entry = next((item for item in self.entries if item.get("key") == key), None)
        if entry is None:
            entry = {
                "key": key,
                "benchmark": str(candidate.get("benchmark", "")),
                "pattern": dict(candidate.get("initial_pattern", {})),
                "support": {"runs": 0, "successes": 0},
                "notes": "",
                "failure_memories": [],
            }
            self.entries.append(entry)

        support = entry.setdefault("support", {"runs": 0, "successes": 0})
        support["runs"] = int(support.get("runs", 0)) + 1
        support["successes"] = int(support.get("successes", 0)) + int(bool(success))

        mutation = candidate.get("mutation_summary")
        if mutation:
            memories = entry.setdefault("failure_memories", [])
            memories.append(
                {
                    "mode": ",".join(candidate.get("audit_modes", [])) or "unknown",
                    "mutation": str(mutation),
                    "outcome": "success" if success else "failure",
                    "termination_reason": str(candidate.get("termination_reason", "")),
                }
            )
            del memories[:-_MAX_FAILURE_MEMORIES]


class PlaybookMaintainer:
    """Builds the in-run update candidate (process signals only, no ground truth)."""

    @staticmethod
    def build_update_candidate(
        *,
        key: str,
        benchmark_name: str,
        spec_versions: list[dict[str, Any]],
        audit_reports: list[dict[str, Any]],
        mutation_payload: dict[str, Any] | None,
        termination_decision: dict[str, Any],
    ) -> dict[str, Any]:
        audit_modes = sorted(
            {
                str(mode.get("mode", ""))
                for report in audit_reports
                for mode in report.get("detected_modes", [])
                if mode.get("mode")
            }
        )
        mutation_summary = ""
        if mutation_payload and mutation_payload.get("mutation"):
            ops = mutation_payload["mutation"].get("ops", [])
            mutation_summary = ",".join(
                f"{op.get('op')}:{op.get('args', {}).get('pattern', '')}".rstrip(":")
                for op in ops
                if isinstance(op, dict)
            )
        return {
            "key": key,
            "benchmark": benchmark_name,
            "initial_pattern": pattern_sketch(spec_versions[0]) if spec_versions else {},
            "final_pattern": pattern_sketch(spec_versions[-1]) if spec_versions else {},
            "mutated": len(spec_versions) > 1,
            "mutation_summary": mutation_summary,
            "audit_modes": audit_modes,
            "termination_reason": str(termination_decision.get("reason", "")),
            "consensus_ratio": float(termination_decision.get("consensus_ratio", 0.0)),
            "average_confidence": float(termination_decision.get("average_confidence", 0.0)),
        }


class ShortTermTopologyPlaybook:
    """In-memory turn-level playbook for one self-evolved run.

    It records process observations before benchmark correctness is known. The
    entries are planner priors for the next in-run repair decision, not
    persistent success evidence.
    """

    def __init__(self, *, max_entries: int = _MAX_SHORT_TERM_ENTRIES) -> None:
        self.max_entries = max(1, int(max_entries))
        self.entries: list[dict[str, Any]] = []

    def record_turn(
        self,
        *,
        key: str,
        benchmark_name: str,
        turn_index: int,
        spec_payload: dict[str, Any],
        audit_report: dict[str, Any] | None,
        termination_decision: dict[str, Any],
    ) -> dict[str, Any]:
        audit_report = audit_report or {}
        detected_modes = [
            str(mode.get("mode", ""))
            for mode in audit_report.get("detected_modes", [])
            if isinstance(mode, dict) and mode.get("mode")
        ]
        repair_recommended = bool(audit_report.get("repair_recommended", False))
        recommendation = str(audit_report.get("recommendation", "")).strip()
        reason = str(termination_decision.get("reason", "")).strip()
        notes_parts = [
            f"turn={int(turn_index)}",
            f"repair_recommended={repair_recommended}",
            f"modes={','.join(detected_modes) if detected_modes else 'none'}",
        ]
        if reason:
            notes_parts.append(f"decision={reason}")
        if recommendation:
            notes_parts.append(recommendation)

        entry = {
            "scope": "short_term",
            "key": key,
            "benchmark": benchmark_name,
            "turn_index": int(turn_index),
            "spec_version": int(spec_payload.get("version", 0)),
            "pattern": pattern_sketch(spec_payload),
            "support": {"kind": "process_signal", "runs": 1, "successes": 0},
            "notes": "; ".join(notes_parts)[:400],
            "repair_recommended": repair_recommended,
            "termination_reason": reason,
            "failure_memories": [
                {
                    "mode": mode,
                    "mutation": "pending_trace_backed_repair"
                    if repair_recommended
                    else "no_repair",
                    "outcome": "process_signal",
                }
                for mode in detected_modes[:3]
            ],
        }
        self.entries.append(entry)
        del self.entries[: -self.max_entries]
        return entry

    def planner_entries(self) -> list[dict[str, Any]]:
        return [TopologyPlaybook.planner_view(entry) for entry in self.entries]

    def to_payload(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.entries]

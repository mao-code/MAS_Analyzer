"""Topology Planning Skill: the long-term playbook as an agent-maintained SKILL.md.

Unlike the structured JSON playbook (``playbook.py``), this is a long-form markdown
document — the planner loads it in full at plan time (like a skill), and an LLM
**reflection agent** rewrites it from run outcomes (``SkillReflector``).

Division of responsibility:
- read side: ``TopologySkill.load(path).prompt_section()`` returns the markdown the
  planner injects. When the file exists it is the planner's primary long-term memory;
  the JSON playbook is the deterministic fallback when it is absent.
- write side: ``SkillReflector.reflect(...)`` is given the current skill plus run
  outcomes labelled by PROCESS SIGNALS ONLY (``is_process_clean`` — auditor findings +
  decision-grade consensus, never ``benchmark.evaluate(...).success``) and returns a
  revised markdown skill. The default writer is the online ``OnlineSkillUpdater`` (every
  N runs); ``scripts/reflect_topology_skill.py`` is the offline equivalent. Keeping the
  benchmark verdict out of the skill is what stops the study from being biased.

The reflection LLM falls back to leaving the skill unchanged when mocked or unusable,
mirroring the termination/auditor judges, so offline tests stay deterministic.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import SelfEvolvedConfig
from .playbook import is_process_clean

logger = logging.getLogger(__name__)


def summary_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """One run's playbook candidate → a process-only reflection summary.

    Uses **no ground truth**: the outcome label is a process proxy
    (``is_process_clean`` — the auditor flagged no failure modes and the run reached
    decision-grade consensus), so reflection never learns from the held-out benchmark
    verdict. The single source of the summary shape consumed by
    ``SkillReflector.reflect``, shared by the online updater (in-memory ``run_metadata``
    candidate) and the offline ``scripts/reflect_topology_skill.py``, so both feed the
    reflector identical rows.
    """

    final = candidate.get("final_pattern") or candidate.get("initial_pattern") or {}
    pattern = f"{final.get('pattern', '?')}/{final.get('num_agents', '?')}"
    return {
        "key": str(candidate.get("key", "")),
        "benchmark": str(candidate.get("benchmark", "")),
        "pattern": pattern,
        "process_outcome": "clean" if is_process_clean(candidate) else "flagged",
        "audit_modes": list(candidate.get("audit_modes", []) or []),
        "termination_reason": str(candidate.get("termination_reason", "")),
        "average_confidence": round(float(candidate.get("average_confidence", 0.0) or 0.0), 2),
    }

# Sections the reflection agent must keep (refine wording only, never delete) so a weak
# reflector cannot erase the planner's core method while growing the lessons.
PROTECTED_SECTIONS = ("## Standing principles", "## How to choose a topology")


class TopologySkill:
    """The long-term playbook rendered as an agent-maintained markdown skill."""

    def __init__(self, path: str | Path, text: str = "") -> None:
        self.path = Path(path)
        self.text = text

    @classmethod
    def load(cls, path: str | Path) -> TopologySkill:
        p = Path(path)
        if not p.exists():
            return cls(p, "")
        try:
            return cls(p, p.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read topology skill at %s; treating as empty", p)
            return cls(p, "")

    def exists(self) -> bool:
        return bool(self.text.strip())

    def prompt_section(self, *, max_chars: int = 8000) -> str:
        """Bounded markdown for the planner prompt (full skill, generously capped)."""

        text = self.text.strip()
        if not text:
            return ""
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n\n…(skill truncated)…"
        return text

    def save(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        tmp.replace(self.path)


@dataclass(frozen=True)
class ReflectionResult:
    skill_markdown: str
    changed: bool
    reason: str
    llm: dict[str, Any] = field(default_factory=dict)


def _strip_code_fences(text: str) -> str:
    """Drop a leading/trailing ```markdown fence if the model wrapped its output."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


class SkillReflector:
    """LLM agent that revises the markdown skill from process-labelled run outcomes."""

    def __init__(self, llm_client: Any, se_config: SelfEvolvedConfig) -> None:
        self.llm_client = llm_client
        self.se_config = se_config

    def reflect(
        self, *, current_skill: str, run_summaries: list[dict[str, Any]]
    ) -> ReflectionResult:
        if not run_summaries:
            return ReflectionResult(current_skill, False, "no_runs")

        prompt = self._build_prompt(current_skill, run_summaries)
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                agent_type="general",
                task_id="skill_reflection",
                run_index=0,
                agent_id="skill_reflector",
                tools=[],
                max_tool_iterations=1,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Skill reflection failed; keeping current skill", exc_info=True)
            return ReflectionResult(current_skill, False, f"llm_error:{exc}")

        llm_payload = {
            "model": str(result.model),
            "mock_used": bool(result.mock_used),
            "token_in": int(result.token_in),
            "token_out": int(result.token_out),
            "cost_usd": float(result.cost_usd),
        }
        if bool(result.mock_used):
            return ReflectionResult(current_skill, False, "mock", llm_payload)

        text = _strip_code_fences(str(result.text or ""))
        # Guardrails: keep only a substantive doc that preserved the protected sections.
        if len(text) < 200 or not text.lstrip().startswith("#"):
            return ReflectionResult(current_skill, False, "unusable_output", llm_payload)
        if current_skill.strip():
            missing = [s for s in PROTECTED_SECTIONS if s in current_skill and s not in text]
            if missing:
                return ReflectionResult(
                    current_skill, False, "dropped_protected_section", llm_payload
                )
        return ReflectionResult(text, True, "updated", llm_payload)

    # -- prompt ----------------------------------------------------------------

    @staticmethod
    def _aggregate(run_summaries: list[dict[str, Any]]) -> str:
        """Compact clean-by-(shape, pattern) table plus auditor failure-mode counts.

        Process signals only — the clean/total counts come from ``is_process_clean``,
        never from the benchmark verdict.
        """

        by_pattern: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        modes: dict[str, int] = defaultdict(int)
        for row in run_summaries:
            shape = str(row.get("key", "")) or "unknown"
            pattern = str(row.get("pattern", "?"))
            stat = by_pattern[(shape, pattern)]
            stat[0] += 1
            stat[1] += 1 if row.get("process_outcome") == "clean" else 0
            for mode in row.get("audit_modes", []) or []:
                modes[str(mode)] += 1
        lines = [
            "process outcomes by task shape and topology "
            "(clean/total — process signals only, NOT ground-truth correctness):"
        ]
        for (shape, pattern), (runs, clean) in sorted(by_pattern.items()):
            lines.append(f"  - {shape} | {pattern}: {clean}/{runs} ran clean")
        if modes:
            lines.append("process failure modes flagged by the auditor (across all runs):")
            for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {mode}: {count}")
        return "\n".join(lines)

    def _build_prompt(
        self, current_skill: str, run_summaries: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        system_msg = (
            "You maintain a 'Topology Planning Skill': a markdown document that teaches a "
            "planner how to choose a multi-agent topology for a task. You are given the "
            "current skill and outcomes from recent runs. IMPORTANT: the outcomes are "
            "labelled by PROCESS SIGNALS ONLY (whether the run's trace auditor flagged "
            "failure modes and whether it reached decision-grade consensus) — there is NO "
            "ground-truth correctness here. Revise the skill so it captures which "
            "topologies run cleanly and which trigger process failures.\n"
            "Rules:\n"
            f"- PRESERVE these sections, refining wording only, never deleting them: "
            f"{', '.join(PROTECTED_SECTIONS)}.\n"
            "- Grow the '## Lessons from experience' section: add or refine concise, "
            "actionable lessons grounded in the process signals (cite the evidence, e.g. "
            "'chain/3 ran clean 2/2 on tool-using medium retrieval; star/3 flagged "
            "duplicate_state_mutation 3x'). Speak of running cleanly / avoiding process "
            "failure modes, NOT of being 'correct' or 'right'.\n"
            "- Keep every lesson GENERAL — key it on task characteristics (task type, tools, "
            "size, state mutation, search breadth), not on benchmark-specific trivia.\n"
            "- Prefer revising an existing lesson over duplicating it; drop lessons the new "
            "evidence contradicts. Keep the document tight and readable.\n"
            "- Output the COMPLETE updated markdown document and nothing else (no fences)."
        )
        user_msg = (
            "## Current skill\n"
            f"{current_skill.strip() or '(empty — create the document)'}\n\n"
            "## Recent run outcomes (process signals only — no ground truth)\n"
            f"{self._aggregate(run_summaries)}\n\n"
            "Return the full updated skill markdown."
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]


class OnlineSkillUpdater:
    """Online (in-experiment) writer for the long-term skill.

    Accumulates per-run playbook candidates (process signals only — no ground truth)
    from freshly executed self_evolved runs and, every ``batch_size`` runs, reflects
    that batch into the skill markdown and fires ``on_update`` so the planner reloads
    the revised skill for the rest of the experiment. Because the experiment loop is single-threaded, ``record`` blocks
    on the reflection LLM call at a batch boundary — the run loop pauses, the skill
    updates, then the loop resumes. ``batch_size <= 0`` makes every method a no-op,
    preserving the parallel-safe, post-hoc-only default.

    A trailing partial batch (< ``batch_size`` runs) is intentionally left for the
    post-hoc ``scripts/reflect_topology_skill.py`` to catch — online updates fire only
    on exact batch boundaries.
    """

    def __init__(
        self,
        *,
        reflector: SkillReflector,
        skill_path: str | Path,
        batch_size: int,
        on_update: Callable[[], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.reflector = reflector
        self.skill_path = Path(skill_path)
        self.batch_size = int(batch_size)
        self._on_update = on_update
        self._log = log or (lambda _message: None)
        self._buffer: list[dict[str, Any]] = []
        self.updates_applied = 0

    @property
    def enabled(self) -> bool:
        return self.batch_size > 0

    def record(self, candidate: dict[str, Any] | None) -> None:
        """Buffer one run's process-signal outcome; flush when the batch is full.

        Takes no ground-truth label — the reflection input is process-only
        (``summary_from_candidate``)."""

        if not self.enabled or not candidate:
            return
        self._buffer.append(summary_from_candidate(candidate))
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        batch = self._buffer
        self._buffer = []
        if not batch:
            return
        # Re-read the on-disk skill so reflection always builds on the latest text.
        skill = TopologySkill.load(self.skill_path)
        clean = sum(1 for row in batch if row.get("process_outcome") == "clean")
        result = self.reflector.reflect(current_skill=skill.text, run_summaries=batch)
        if result.changed:
            skill.save(result.skill_markdown)
            self.updates_applied += 1
            if self._on_update is not None:
                self._on_update()
            self._log(
                f"SKILL_UPDATE runs={len(batch)} clean={clean} "
                f"reason={result.reason} path={self.skill_path}"
            )
        else:
            self._log(
                f"SKILL_UPDATE_SKIP runs={len(batch)} clean={clean} reason={result.reason}"
            )

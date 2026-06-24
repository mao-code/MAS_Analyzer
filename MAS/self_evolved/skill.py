"""Topology Planning Skill: the long-term playbook as an agent-maintained SKILL.md.

Unlike the structured JSON playbook (``playbook.py``), this is a long-form markdown
document — the planner loads it in full at plan time (like a skill), and an LLM
**reflection agent** rewrites it post-hoc from run outcomes (``SkillReflector``).

Division of responsibility:
- read side: ``TopologySkill.load(path).prompt_section()`` returns the markdown the
  planner injects. When the file exists it is the planner's primary long-term memory;
  the JSON playbook is the deterministic fallback when it is absent.
- write side: ``SkillReflector.reflect(...)`` is given the current skill plus run
  outcomes labelled with ground-truth ``benchmark.evaluate(...).success`` and returns a
  revised markdown skill. It is invoked post-hoc by
  ``scripts/reflect_topology_skill.py`` — runs never write the file (parallel-safe).

The reflection LLM falls back to leaving the skill unchanged when mocked or unusable,
mirroring the termination/auditor judges, so offline tests stay deterministic.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import SelfEvolvedConfig

logger = logging.getLogger(__name__)

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
    """LLM agent that revises the markdown skill from success-labelled run outcomes."""

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
        """Compact success-by-(shape, pattern) table plus failure-mode counts."""

        by_pattern: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        modes: dict[str, int] = defaultdict(int)
        for row in run_summaries:
            shape = str(row.get("key", "")) or "unknown"
            pattern = str(row.get("pattern", "?"))
            stat = by_pattern[(shape, pattern)]
            stat[0] += 1
            stat[1] += 1 if row.get("success") else 0
            for mode in row.get("audit_modes", []) or []:
                if not row.get("success"):
                    modes[str(mode)] += 1
        lines = ["outcomes by task shape and topology (successes/runs):"]
        for (shape, pattern), (runs, succ) in sorted(by_pattern.items()):
            lines.append(f"  - {shape} | {pattern}: {succ}/{runs}")
        if modes:
            lines.append("failure modes observed on failed runs:")
            for mode, count in sorted(modes.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {mode}: {count}")
        return "\n".join(lines)

    def _build_prompt(
        self, current_skill: str, run_summaries: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        system_msg = (
            "You maintain a 'Topology Planning Skill': a markdown document that teaches a "
            "planner how to choose a multi-agent topology for a task. You are given the "
            "current skill and outcomes from recent runs, each labelled with ground-truth "
            "success or failure. Revise the skill so it captures what actually worked.\n"
            "Rules:\n"
            f"- PRESERVE these sections, refining wording only, never deleting them: "
            f"{', '.join(PROTECTED_SECTIONS)}.\n"
            "- Grow the '## Lessons from experience' section: add or refine concise, "
            "actionable lessons grounded in the outcomes (cite the evidence, e.g. "
            "'chain/3 succeeded 2/2 on tool-using medium retrieval').\n"
            "- Keep every lesson GENERAL — key it on task characteristics (task type, tools, "
            "size, state mutation, search breadth), not on benchmark-specific trivia.\n"
            "- Prefer revising an existing lesson over duplicating it; drop lessons the new "
            "evidence contradicts. Keep the document tight and readable.\n"
            "- Output the COMPLETE updated markdown document and nothing else (no fences)."
        )
        user_msg = (
            "## Current skill\n"
            f"{current_skill.strip() or '(empty — create the document)'}\n\n"
            "## Recent run outcomes (ground truth)\n"
            f"{self._aggregate(run_summaries)}\n\n"
            "Return the full updated skill markdown."
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

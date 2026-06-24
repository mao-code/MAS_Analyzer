import tempfile
import unittest
from pathlib import Path
from typing import Any

from MAS.config import SelfEvolvedConfig
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.self_evolved.planner import TopologyPlannerAgent
from MAS.self_evolved.skill import SkillReflector, TopologySkill

SEED_SKILL = (
    "# Topology Planning Skill\n\n"
    "## Standing principles\n1. one executor for writes.\n\n"
    "## How to choose a topology\n- retrieval -> searchers.\n\n"
    "## Lessons from experience\n- (none yet)\n"
)


class _StubLLM(OpenRouterLLMClient):
    def __init__(self, text: str, *, mock_used: bool = False) -> None:
        self._text = text
        self._mock_used = mock_used

    def generate(self, **kwargs: Any) -> LLMResult:
        return LLMResult(
            text=self._text,
            token_in=10,
            token_out=10,
            cost_usd=0.0,
            model="stub",
            mock_used=self._mock_used,
            metadata={},
        )


class _Task:
    task_id = "t"
    prompt = "Create a calendar event and email the team."


class TestTopologySkill(unittest.TestCase):
    def test_load_missing_file_is_empty(self) -> None:
        skill = TopologySkill.load("/nonexistent/topology_skill.md")
        self.assertFalse(skill.exists())
        self.assertEqual(skill.prompt_section(), "")

    def test_roundtrip_and_bounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill.md"
            TopologySkill(path).save(SEED_SKILL)
            skill = TopologySkill.load(path)
            self.assertTrue(skill.exists())
            self.assertIn("Standing principles", skill.prompt_section())
            self.assertTrue(skill.prompt_section(max_chars=40).endswith("…(skill truncated)…"))


class TestSkillInPlannerPrompt(unittest.TestCase):
    def _planner(self, plan_text: str = "{}") -> TopologyPlannerAgent:
        return TopologyPlannerAgent(_StubLLM(plan_text), SelfEvolvedConfig())

    def test_skill_text_injected_and_replaces_inline_guidance(self) -> None:
        messages = self._planner()._build_initial_prompt(
            task=_Task(),
            benchmark_name="workbench",
            max_agents=3,
            playbook_entries=[],
            principles=["fallback principle"],
            skill_text=SEED_SKILL,
        )
        blob = messages[0]["content"] + messages[1]["content"]
        self.assertIn("Topology planning skill", blob)
        self.assertIn("## Standing principles", blob)
        # The skill replaces the built-in guidance block and the JSON principles fallback.
        self.assertNotIn("How to choose the topology (match it to your analysis)", blob)
        self.assertNotIn("fallback principle", blob)

    def test_without_skill_falls_back_to_guidance(self) -> None:
        messages = self._planner()._build_initial_prompt(
            task=_Task(),
            benchmark_name="workbench",
            max_agents=3,
            playbook_entries=[],
            principles=["fallback principle"],
            skill_text="",
        )
        blob = messages[0]["content"] + messages[1]["content"]
        self.assertIn("How to choose the topology (match it to your analysis)", blob)
        self.assertIn("fallback principle", blob)


class TestSkillReflector(unittest.TestCase):
    SUMMARIES = [
        {
            "key": "workbench::tools::medium",
            "benchmark": "workbench",
            "pattern": "chain/2",
            "success": True,
            "audit_modes": [],
            "termination_reason": "consensus_reached",
        },
        {
            "key": "browsecomp::tools::medium",
            "benchmark": "browsecomp",
            "pattern": "chain/2",
            "success": False,
            "audit_modes": ["insufficient_search_coverage"],
            "termination_reason": "max_rounds_reached",
        },
    ]

    def _reflector(self, text: str, *, mock: bool = False) -> SkillReflector:
        return SkillReflector(_StubLLM(text, mock_used=mock), SelfEvolvedConfig())

    def test_mock_llm_leaves_skill_unchanged(self) -> None:
        result = self._reflector("anything", mock=True).reflect(
            current_skill=SEED_SKILL, run_summaries=self.SUMMARIES
        )
        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "mock")
        self.assertEqual(result.skill_markdown, SEED_SKILL)

    def test_no_runs_is_noop(self) -> None:
        result = self._reflector("x").reflect(current_skill=SEED_SKILL, run_summaries=[])
        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "no_runs")

    def test_dropping_protected_section_is_rejected(self) -> None:
        # Output omits "## How to choose a topology" -> guardrail keeps the old skill.
        bad = "# Topology Planning Skill\n\n## Standing principles\n1. x.\n\n## Lessons\n- y.\n" + (
            "padding " * 40
        )
        result = self._reflector(bad).reflect(
            current_skill=SEED_SKILL, run_summaries=self.SUMMARIES
        )
        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "dropped_protected_section")

    def test_valid_revision_is_accepted(self) -> None:
        good = (
            "# Topology Planning Skill\n\n"
            "## Standing principles\n1. one executor for writes.\n\n"
            "## How to choose a topology\n- retrieval -> searchers.\n\n"
            "## Lessons from experience\n"
            "- chain/2 succeeded 1/1 on workbench tool-using medium state-mutation tasks.\n"
            + ("More detail to clear the length guardrail. " * 5)
        )
        result = self._reflector(good).reflect(
            current_skill=SEED_SKILL, run_summaries=self.SUMMARIES
        )
        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "updated")
        self.assertIn("chain/2 succeeded", result.skill_markdown)


if __name__ == "__main__":
    unittest.main()

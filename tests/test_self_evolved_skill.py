import tempfile
import unittest
from pathlib import Path
from typing import Any

from MAS.config import SelfEvolvedConfig
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.self_evolved.planner import TopologyPlannerAgent
from MAS.self_evolved.skill import (
    OnlineSkillUpdater,
    SkillReflector,
    TopologySkill,
    summary_from_candidate,
)

VALID_REVISION = (
    "# Topology Planning Skill\n\n"
    "## Standing principles\n1. one executor for writes.\n\n"
    "## How to choose a topology\n- retrieval -> searchers.\n\n"
    "## Lessons from experience\n"
    "- chain/2 ran clean 1/1 on workbench tool-using medium state-mutation tasks.\n"
    + ("More detail to clear the length guardrail. " * 5)
)

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
        self.assertIn("Count independent evidence sources", blob)
        self.assertIn("exactly one retriever", blob)
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

    def test_skill_injected_into_mutation_prompt(self) -> None:
        planner = self._planner()
        spec = planner.fallback_initial_spec(2)
        messages = planner._build_mutation_prompt(
            task=_Task(),
            spec=spec,
            audit_report={"detected_modes": [], "recommendation": ""},
            playbook_entries=[],
            principles=["fallback principle"],
            skill_text=SEED_SKILL,
        )
        blob = messages[0]["content"] + messages[1]["content"]
        self.assertIn("Topology planning skill", blob)
        self.assertIn("counterfactual information gain", blob)
        self.assertIn("cannot repair data that the source did not return", blob)
        # The operational ops cheatsheet always stays; the skill replaces the JSON priors.
        self.assertIn("Available ops", blob)
        self.assertNotIn("fallback principle", blob)

    def test_mutation_prompt_without_skill_keeps_principles(self) -> None:
        planner = self._planner()
        spec = planner.fallback_initial_spec(2)
        messages = planner._build_mutation_prompt(
            task=_Task(),
            spec=spec,
            audit_report={"detected_modes": [], "recommendation": ""},
            playbook_entries=[],
            principles=["fallback principle"],
            skill_text="",
        )
        blob = messages[0]["content"] + messages[1]["content"]
        self.assertNotIn("Topology planning skill", blob)
        self.assertIn("fallback principle", blob)


class TestSkillReflector(unittest.TestCase):
    SUMMARIES = [
        {
            "key": "workbench::tools::medium",
            "benchmark": "workbench",
            "pattern": "chain/2",
            "process_outcome": "clean",
            "audit_modes": [],
            "termination_reason": "consensus_reached",
        },
        {
            "key": "browsecomp::tools::medium",
            "benchmark": "browsecomp",
            "pattern": "chain/2",
            "process_outcome": "flagged",
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
            "- chain/2 ran clean 1/1 on workbench tool-using medium state-mutation tasks.\n"
            + ("More detail to clear the length guardrail. " * 5)
        )
        result = self._reflector(good).reflect(
            current_skill=SEED_SKILL, run_summaries=self.SUMMARIES
        )
        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "updated")
        self.assertIn("chain/2 ran clean", result.skill_markdown)


class TestOnlineSkillUpdater(unittest.TestCase):
    @staticmethod
    def _candidate(key: str = "workbench::tools::medium") -> dict:
        # Process-clean by default (no audit modes + decision-grade consensus); no eval.
        return {
            "key": key,
            "benchmark": "workbench",
            "final_pattern": {"pattern": "chain", "num_agents": 2},
            "audit_modes": [],
            "termination_reason": "consensus_reached",
        }

    def test_disabled_batch_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill.md"
            TopologySkill(path).save(SEED_SKILL)
            updater = OnlineSkillUpdater(
                reflector=SkillReflector(_StubLLM(VALID_REVISION), SelfEvolvedConfig()),
                skill_path=path,
                batch_size=0,
            )
            self.assertFalse(updater.enabled)
            for _ in range(5):
                updater.record(self._candidate())
            self.assertEqual(updater.updates_applied, 0)
            self.assertIn("(none yet)", TopologySkill.load(path).text)

    def test_flushes_every_n_runs_and_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill.md"
            TopologySkill(path).save(SEED_SKILL)
            reloaded: list[bool] = []
            updater = OnlineSkillUpdater(
                reflector=SkillReflector(_StubLLM(VALID_REVISION), SelfEvolvedConfig()),
                skill_path=path,
                batch_size=3,
                on_update=lambda: reloaded.append(True),
            )
            updater.record(self._candidate())
            updater.record(self._candidate())
            self.assertEqual(updater.updates_applied, 0)  # batch not full yet
            self.assertEqual(reloaded, [])

            updater.record(self._candidate())  # 3rd run -> flush
            self.assertEqual(updater.updates_applied, 1)
            self.assertEqual(reloaded, [True])  # engine cache invalidated
            self.assertIn("chain/2 ran clean", TopologySkill.load(path).text)

    def test_none_candidate_does_not_advance_batch(self) -> None:
        # Runs surfacing no playbook candidate must not consume a batch slot.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill.md"
            TopologySkill(path).save(SEED_SKILL)
            updater = OnlineSkillUpdater(
                reflector=SkillReflector(_StubLLM(VALID_REVISION), SelfEvolvedConfig()),
                skill_path=path,
                batch_size=2,
            )
            for _ in range(5):
                updater.record(None)
            self.assertEqual(updater.updates_applied, 0)
            updater.record(self._candidate())
            updater.record(self._candidate())
            self.assertEqual(updater.updates_applied, 1)

    def test_mock_reflection_keeps_skill_but_clears_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "skill.md"
            TopologySkill(path).save(SEED_SKILL)
            reloaded: list[bool] = []
            updater = OnlineSkillUpdater(
                reflector=SkillReflector(_StubLLM("x", mock_used=True), SelfEvolvedConfig()),
                skill_path=path,
                batch_size=2,
                on_update=lambda: reloaded.append(True),
            )
            updater.record(self._candidate())
            updater.record(self._candidate())
            # Mock reflection leaves the skill unchanged; no update, no reload.
            self.assertEqual(updater.updates_applied, 0)
            self.assertEqual(reloaded, [])
            self.assertIn("(none yet)", TopologySkill.load(path).text)


class TestSummaryFromCandidate(unittest.TestCase):
    def test_falls_back_to_initial_pattern_and_is_process_only(self) -> None:
        # No audit modes but not consensus-terminated -> flagged (process proxy).
        row = summary_from_candidate(
            {"key": "k", "benchmark": "b", "initial_pattern": {"pattern": "star", "num_agents": 4}}
        )
        self.assertEqual(row["pattern"], "star/4")
        self.assertEqual(row["process_outcome"], "flagged")
        self.assertNotIn("success", row)  # ground truth never enters the summary

    def test_clean_when_no_modes_and_consensus(self) -> None:
        row = summary_from_candidate(
            {
                "key": "k",
                "benchmark": "b",
                "final_pattern": {"pattern": "chain", "num_agents": 2},
                "audit_modes": [],
                "termination_reason": "consensus_reached",
            }
        )
        self.assertEqual(row["process_outcome"], "clean")


if __name__ == "__main__":
    unittest.main()

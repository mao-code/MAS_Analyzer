import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from MAS.config import SelfEvolvedConfig
from MAS.langgraph_engine import ExperimentSpec
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.self_evolved.engine import SelfEvolvedEngine
from MAS.self_evolved.playbook import ShortTermTopologyPlaybook, TopologyPlaybook, task_key
from scripts.update_topology_playbook import main as update_playbook_main

_VALID_PLAN = json.dumps(
    {
        "rationale": "Star fits this decomposable lookup.",
        "pattern": "star",
        "num_agents": 3,
        "expansions": [],
    }
)


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _entry(
    key: str = "finance_agent::no_tools::short", benchmark: str = "finance_agent"
) -> dict[str, Any]:
    return {
        "key": key,
        "benchmark": benchmark,
        "pattern": {"pattern": "star", "num_agents": 3, "expansions": []},
        "support": {"runs": 4, "successes": 3},
        "notes": "Star-3 reliably covers short finance lookups.",
        "failure_memories": [],
    }


def _candidate(
    key: str = "finance_agent::no_tools::short",
    *,
    mutation_summary: str = "",
    audit_modes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "benchmark": key.split("::")[0],
        "initial_pattern": {"pattern": "star", "num_agents": 3, "expansions": []},
        "final_pattern": {"pattern": "star", "num_agents": 3, "expansions": []},
        "mutated": bool(mutation_summary),
        "mutation_summary": mutation_summary,
        "audit_modes": audit_modes or [],
        "termination_reason": "consensus_reached",
        "consensus_ratio": 1.0,
        "average_confidence": 0.8,
    }


class _CapturingLLM(OpenRouterLLMClient):
    """Returns a valid plan for the planner (capturing its prompt) and a
    converging answer for everyone else."""

    def __init__(self) -> None:
        self.planner_prompts: list[str] = []

    def generate(
        self,
        *,
        prompt,
        agent_type,
        task_id,
        run_index,
        agent_id,
        tools=None,
        max_tool_iterations=8,
        temperature=0.0,
    ) -> LLMResult:
        prompt_text = (
            prompt
            if isinstance(prompt, str)
            else " ".join(str(m.get("content", "")) for m in prompt)
        )
        if agent_id == "topology_planner":
            self.planner_prompts.append(prompt_text)
            text = _VALID_PLAN
        else:
            text = "The answer is 42."
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="capture",
            mock_used=False,
            metadata={},
        )


def _run_engine(playbook_path: Path, *, playbook_read: bool = True) -> tuple[Any, _CapturingLLM]:
    client = _CapturingLLM()
    engine = SelfEvolvedEngine(
        client,
        SelfEvolvedConfig(
            playbook_path=str(playbook_path),
            # No skill file -> exercise the JSON-playbook fallback path here; the
            # skill-primary path is covered in test_self_evolved_skill.py.
            skill_path=str(playbook_path.parent / "absent_skill.md"),
            playbook_read=playbook_read,
        ),
    )
    spec = ExperimentSpec(
        topology="self_evolved",
        num_agents=3,
        rounds=2,
        communication_budget_per_agent=2,
        termination_consensus_mode="lexical",
        final_vote_mode="deterministic",
        benchmark_name="finance_agent",
        enable_dynamic_roles=False,
    )
    result = engine.run(
        task=_Task(task_id="pb-task", prompt="What is the figure?"),
        run_index=0,
        seed=7,
        spec=spec,
        agent_types=["general"],
        tools=[],
    )
    return result, client


class TestTaskKey(unittest.TestCase):
    def test_key_is_deterministic_over_benchmark_tools_and_size(self) -> None:
        short = _Task(task_id="t", prompt="x" * 399)
        medium = _Task(task_id="t", prompt="x" * 400)
        long = _Task(task_id="t", prompt="x" * 1600)

        self.assertEqual(
            task_key("finance_agent", short, tools_available=False),
            "finance_agent::no_tools::short",
        )
        self.assertEqual(
            task_key("finance_agent", medium, tools_available=True),
            "finance_agent::tools::medium",
        )
        self.assertEqual(
            task_key("browsecomp", long, tools_available=True),
            "browsecomp::tools::long",
        )

    def test_missing_benchmark_and_prompt(self) -> None:
        self.assertEqual(
            task_key("", _Task(task_id="t", prompt=None), tools_available=False),
            "unknown::no_tools::short",
        )


class TestPlaybookPersistence(unittest.TestCase):
    def test_save_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "playbook.json"
            playbook = TopologyPlaybook(path)
            playbook.entries.append(_entry())
            playbook.save()

            self.assertTrue(path.exists())
            # Atomic save leaves no temp file behind.
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["playbook.json"])

            reloaded = TopologyPlaybook.load(path)
            self.assertEqual(reloaded.entries, playbook.entries)
            self.assertEqual(reloaded.version, playbook.version)

    def test_load_missing_or_corrupt_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = TopologyPlaybook.load(Path(tmp) / "absent.json")
            self.assertEqual(missing.entries, [])

            corrupt_path = Path(tmp) / "corrupt.json"
            corrupt_path.write_text("not json{", encoding="utf-8")
            corrupt = TopologyPlaybook.load(corrupt_path)
            self.assertEqual(corrupt.entries, [])

    def test_lookup_prefers_exact_key_then_benchmark(self) -> None:
        playbook = TopologyPlaybook("unused.json")
        playbook.entries.extend(
            [
                _entry("finance_agent::tools::long"),
                _entry("finance_agent::no_tools::short"),
                _entry("browsecomp::tools::long", benchmark="browsecomp"),
            ]
        )
        entries = playbook.lookup("finance_agent", "finance_agent::no_tools::short")
        self.assertEqual(entries[0]["key"], "finance_agent::no_tools::short")
        self.assertEqual(entries[1]["key"], "finance_agent::tools::long")
        self.assertEqual(len(entries), 2)

    def test_lookup_transfers_same_shape_across_benchmarks(self) -> None:
        # A lesson learned on one benchmark must transfer to a same-shape task on a
        # benchmark the playbook has never seen (the skill-like behavior).
        playbook = TopologyPlaybook("unused.json")
        playbook.entries.extend(
            [
                _entry("workbench::tools::medium", benchmark="workbench"),
                _entry("browsecomp::no_tools::long", benchmark="browsecomp"),
            ]
        )
        entries = playbook.lookup("stabletoolbench", "stabletoolbench::tools::medium")
        keys = [e["key"] for e in entries]
        # Same shape (tools::medium) transfers; the different-shape entry does not.
        self.assertIn("workbench::tools::medium", keys)
        self.assertNotIn("browsecomp::no_tools::long", keys)

    def test_planner_view_is_bounded(self) -> None:
        entry = _entry()
        entry["notes"] = "n" * 500
        entry["failure_memories"] = [
            {
                "mode": "branch_collapse",
                "mutation": "expand_agent_to_group:star",
                "outcome": "success",
            }
        ]
        view = TopologyPlaybook.planner_view(entry)
        self.assertEqual(view["scope"], "long_term")
        self.assertEqual(view["support"], "3/4 successful")
        self.assertEqual(len(view["notes"]), 200)
        self.assertIn("branch_collapse->expand_agent_to_group:star(success)", view["mutation"])


class TestShortTermPlaybook(unittest.TestCase):
    def test_records_turn_level_process_memory_for_planner(self) -> None:
        playbook = ShortTermTopologyPlaybook()
        entry = playbook.record_turn(
            key="finance_agent::no_tools::short",
            benchmark_name="finance_agent",
            turn_index=0,
            spec_payload={
                "version": 0,
                "agents": [{"agent_id": "agent_0"}],
                "root_group_id": "g_root",
                "groups": [
                    {
                        "group_id": "g_root",
                        "pattern": "star",
                        "member_ids": ["agent_0"],
                    }
                ],
            },
            audit_report={
                "repair_recommended": True,
                "recommendation": "Split the weak branch.",
                "detected_modes": [{"mode": "branch_collapse"}],
            },
            termination_decision={"reason": "max_rounds_reached"},
        )

        self.assertEqual(entry["scope"], "short_term")
        self.assertTrue(entry["repair_recommended"])
        view = playbook.planner_entries()[0]
        self.assertEqual(view["scope"], "short_term")
        self.assertEqual(view["support"], "turn-level process memory")
        self.assertIn("branch_collapse->pending_trace_backed_repair", view["mutation"])


class TestMergeCandidate(unittest.TestCase):
    def test_support_is_success_conditioned(self) -> None:
        playbook = TopologyPlaybook("unused.json")
        playbook.merge_candidate(_candidate(), success=True)
        playbook.merge_candidate(_candidate(), success=False)

        self.assertEqual(len(playbook.entries), 1)
        entry = playbook.entries[0]
        self.assertEqual(entry["support"], {"runs": 2, "successes": 1})
        self.assertEqual(entry["pattern"], {"pattern": "star", "num_agents": 3, "expansions": []})
        # No mutation -> no failure memories recorded.
        self.assertEqual(entry["failure_memories"], [])

    def test_mutation_runs_append_bounded_failure_memories(self) -> None:
        playbook = TopologyPlaybook("unused.json")
        for index in range(10):
            playbook.merge_candidate(
                _candidate(
                    mutation_summary=f"expand_agent_to_group:star#{index}",
                    audit_modes=["branch_collapse"],
                ),
                success=index % 2 == 0,
            )
        memories = playbook.entries[0]["failure_memories"]
        self.assertEqual(len(memories), 8)  # trimmed to the most recent
        self.assertEqual(memories[-1]["mutation"], "expand_agent_to_group:star#9")
        self.assertEqual(memories[-1]["mode"], "branch_collapse")
        self.assertEqual(memories[-1]["outcome"], "failure")

    def test_candidate_without_key_is_ignored(self) -> None:
        playbook = TopologyPlaybook("unused.json")
        playbook.merge_candidate({"benchmark": "finance_agent"}, success=True)
        self.assertEqual(playbook.entries, [])


class TestEnginePlaybookIntegration(unittest.TestCase):
    def _seeded_playbook(self, tmp: str) -> Path:
        path = Path(tmp) / "playbook.json"
        playbook = TopologyPlaybook(path)
        playbook.entries.append(_entry())
        playbook.save()
        return path

    def test_planner_prompt_contains_playbook_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._seeded_playbook(tmp)
            result, client = _run_engine(path)

            self.assertEqual(len(client.planner_prompts), 1)
            prompt = client.planner_prompts[0]
            self.assertIn("Historical experience", prompt)
            self.assertIn("finance_agent::no_tools::short", prompt)
            self.assertIn("3/4 successful", prompt)

            planner_events = [
                event for event in result.trace_events if event.actor == "topology_planner"
            ]
            self.assertEqual(
                planner_events[0].payload["playbook_keys"], ["finance_agent::no_tools::short"]
            )

    def test_playbook_read_false_omits_priors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._seeded_playbook(tmp)
            result, client = _run_engine(path, playbook_read=False)

            self.assertNotIn("Historical experience", client.planner_prompts[0])
            planner_events = [
                event for event in result.trace_events if event.actor == "topology_planner"
            ]
            self.assertEqual(planner_events[0].payload["playbook_keys"], [])

    def test_run_updates_short_term_memory_but_not_persistent_playbook(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._seeded_playbook(tmp)
            before = path.read_bytes()
            result, _ = _run_engine(path)

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ["playbook.json"])

            # Short-term memory updates during the run.
            short_term = result.run_metadata["self_evolved"]["short_term_playbook_entries"]
            self.assertEqual(len(short_term), result.run_metadata["turns_executed"])
            self.assertEqual(short_term[0]["scope"], "short_term")

            # The long-term candidate lands in run_metadata only.
            candidate = result.run_metadata["self_evolved"]["playbook_update_candidate"]
            self.assertEqual(candidate["key"], "finance_agent::no_tools::short")
            maintainer_events = [
                event
                for event in result.trace_events
                if event.actor == "playbook_maintainer"
                and event.payload.get("node") == "playbook_candidate"
            ]
            self.assertEqual(len(maintainer_events), 1)
            self.assertEqual(
                maintainer_events[0].payload["update_candidate"]["key"], candidate["key"]
            )

            short_term_events = [
                event
                for event in result.trace_events
                if event.actor == "playbook_maintainer"
                and str(event.payload.get("node", "")).startswith("short_term_playbook_turn_")
            ]
            self.assertEqual(len(short_term_events), result.run_metadata["turns_executed"])

    def test_missing_playbook_path_is_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "absent.json"
            _run_engine(path)
            self.assertFalse(path.exists())


class TestUpdaterScript(unittest.TestCase):
    def _fabricate_run(
        self,
        root: Path,
        benchmark: str,
        task_id: str,
        run_name: str,
        candidate: dict[str, Any],
        *,
        success: bool,
    ) -> None:
        run_dir = root / benchmark / "self_evolved" / task_id
        run_dir.mkdir(parents=True, exist_ok=True)
        raw = {"run_metadata": {"self_evolved": {"playbook_update_candidate": candidate}}}
        (run_dir / f"{run_name}.raw.json").write_text(json.dumps(raw), encoding="utf-8")
        (run_dir / f"{run_name}.eval.json").write_text(
            json.dumps({"success": success}), encoding="utf-8"
        )

    def test_merges_experiment_runs_success_conditioned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            self._fabricate_run(
                root, "finance_agent", "task_0", "run_0", _candidate(), success=True
            )
            self._fabricate_run(
                root,
                "finance_agent",
                "task_0",
                "run_1",
                _candidate(
                    mutation_summary="expand_agent_to_group:star",
                    audit_modes=["branch_collapse"],
                ),
                success=False,
            )
            # Run without an eval verdict must be skipped.
            no_eval_dir = root / "finance_agent" / "self_evolved" / "task_1"
            no_eval_dir.mkdir(parents=True)
            (no_eval_dir / "run_0.raw.json").write_text(
                json.dumps(
                    {"run_metadata": {"self_evolved": {"playbook_update_candidate": _candidate()}}}
                ),
                encoding="utf-8",
            )

            playbook_path = Path(tmp) / "playbook.json"
            exit_code = update_playbook_main(
                ["--experiment-root", str(root), "--playbook", str(playbook_path)]
            )
            self.assertEqual(exit_code, 0)

            playbook = TopologyPlaybook.load(playbook_path)
            self.assertEqual(len(playbook.entries), 1)
            entry = playbook.entries[0]
            self.assertEqual(entry["support"], {"runs": 2, "successes": 1})
            self.assertEqual(len(entry["failure_memories"]), 1)
            self.assertEqual(entry["failure_memories"][0]["outcome"], "failure")

    def test_dry_run_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            self._fabricate_run(
                root, "finance_agent", "task_0", "run_0", _candidate(), success=True
            )
            playbook_path = Path(tmp) / "playbook.json"
            exit_code = update_playbook_main(
                [
                    "--experiment-root",
                    str(root),
                    "--playbook",
                    str(playbook_path),
                    "--dry-run",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertFalse(playbook_path.exists())

    def test_benchmark_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "experiment"
            self._fabricate_run(
                root, "finance_agent", "task_0", "run_0", _candidate(), success=True
            )
            self._fabricate_run(
                root,
                "browsecomp",
                "task_0",
                "run_0",
                _candidate("browsecomp::tools::long"),
                success=True,
            )
            playbook_path = Path(tmp) / "playbook.json"
            update_playbook_main(
                [
                    "--experiment-root",
                    str(root),
                    "--playbook",
                    str(playbook_path),
                    "--benchmark",
                    "browsecomp",
                ]
            )
            playbook = TopologyPlaybook.load(playbook_path)
            self.assertEqual(
                [entry["key"] for entry in playbook.entries], ["browsecomp::tools::long"]
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from reproduce.mass import (
    MASSCandidateExecutor,
    MASSConfig,
    MASSFramework,
    MIPROLikePromptOptimizer,
    SearchSpace,
    TemplateBenchmarkAdapter,
)
from reproduce.mass.interfaces import BenchmarkExample
from reproduce.mass.models import (
    AgentPromptBundle,
    CandidateEvaluation,
    MASSCandidate,
    WorkflowSpec,
)
from reproduce.mass.paper_baselines import (
    BASELINE_SPECS,
    DEFAULT_BASELINES,
    DEFAULT_MODEL,
    StandaloneOpenRouterClient,
    _majority_vote,
)
from reproduce.mass.run_existing_benchmarks import (
    DEFAULT_MODEL_TEMPERATURE,
    DEFAULT_TOPOLOGY_CANDIDATES,
    DEFAULT_TOPOLOGY_TEMPERATURE,
    DEFAULT_VALIDATION_REPEATS,
    _benchmark_family,
    _parse_args,
    _resolve_search_space,
)


class DummyBenchmark:
    def validation_examples(self, limit: int | None = None):
        examples = [
            BenchmarkExample(example_id="ex-1", prompt="task one"),
            BenchmarkExample(example_id="ex-2", prompt="task two"),
        ]
        return examples[:limit] if limit is not None else examples

    def evaluate_candidate(self, candidate, examples):
        workflow = candidate.workflow
        score = float(workflow.aggregate_width)
        score += 0.5 * float(workflow.reflect_rounds)
        score += 0.25 if workflow.execute_enabled else 0.0
        return CandidateEvaluation(
            score=score,
            details={"active_blocks": workflow.active_blocks(), "example_count": len(examples)},
        )

    def execute_candidate(self, candidate, example):
        executor = MASSCandidateExecutor()
        return executor.run_candidate(candidate, example)


def test_mass_framework_runs_three_stages() -> None:
    framework = MASSFramework(
        config=MASSConfig(
            task_name="dummy",
            search_space=SearchSpace(
                enabled_blocks=("aggregate", "reflect", "execute"),
                aggregate=(1, 3),
                reflect=(0, 1, 2),
                execute=(False, True),
                max_agent_budget=8,
            ),
            candidates_per_stage=4,
            random_seed=7,
        ),
        benchmark=DummyBenchmark(),
    )

    results = framework.run()

    assert set(results) == {
        "stage1_block_prompt",
        "stage2_topology",
        "stage3_workflow_prompt",
    }
    assert results["stage2_topology"].best_score >= results["stage1_block_prompt"].best_score
    assert "selection_probabilities" in results["stage1_block_prompt"].metadata


def test_workflow_enforces_agent_budget() -> None:
    framework = MASSFramework(
        config=MASSConfig(
            task_name="dummy",
            search_space=SearchSpace(
                enabled_blocks=("aggregate", "debate"),
                aggregate=(1, 3, 5, 7),
                debate=(0, 1, 2, 3),
                max_agent_budget=4,
            ),
            candidates_per_stage=16,
        ),
        benchmark=DummyBenchmark(),
    )

    results = framework.run()
    assert results["stage2_topology"].best_candidate.workflow.estimated_agent_count <= 4


def test_stage1_records_block_influence_scores() -> None:
    framework = MASSFramework(
        config=MASSConfig(
            task_name="dummy",
            search_space=SearchSpace(
                enabled_blocks=("aggregate", "reflect", "debate"),
                aggregate=(1, 3, 5),
                reflect=(0, 1, 2),
                debate=(0, 1, 2),
                max_agent_budget=8,
            ),
            candidates_per_stage=4,
            random_seed=3,
        ),
        benchmark=DummyBenchmark(),
    )

    results = framework.run()
    influence_scores = results["stage1_block_prompt"].metadata["influence_scores"]
    assert set(influence_scores) == {"aggregate", "reflect", "debate"}


def test_topology_stage_records_unique_sampled_topologies() -> None:
    framework = MASSFramework(
        config=MASSConfig(
            task_name="dummy",
            search_space=SearchSpace(
                enabled_blocks=("aggregate", "reflect", "debate"),
                aggregate=(1, 3, 5, 7, 9),
                reflect=(0, 1, 2, 3, 4),
                debate=(0, 1, 2, 3, 4),
                max_agent_budget=12,
            ),
            candidates_per_stage=8,
            random_seed=23,
        ),
        benchmark=DummyBenchmark(),
    )

    results = framework.run()
    sampled = results["stage2_topology"].metadata["sampled_candidates"]
    topology_keys = [tuple(candidate["topology_key"]) for candidate in sampled]

    assert len(topology_keys) == len(set(topology_keys))
    assert results["stage2_topology"].metadata["unique_topology_count"] == len(sampled)
    assert results["stage2_topology"].metadata["topology_sampling_exhausted"] is False


def test_mipro_like_optimizer_adds_exemplars_and_metadata() -> None:
    framework = MASSFramework(
        config=MASSConfig(
            task_name="dummy",
            search_space=SearchSpace(
                enabled_blocks=("aggregate", "reflect"),
                aggregate=(1, 3),
                reflect=(0, 1),
                max_agent_budget=6,
            ),
            candidates_per_stage=2,
            random_seed=11,
        ),
        benchmark=DummyBenchmark(),
        prompt_optimizer=MIPROLikePromptOptimizer(),
    )

    results = framework.run()
    predictor_prompt = (
        results["stage1_block_prompt"].metadata["base_candidate"].prompts["predictor"]
    )

    assert predictor_prompt.metadata["optimizer"] == "mipro_like"
    assert predictor_prompt.metadata["max_bootstrapped_demos"] == 3
    assert predictor_prompt.metadata["instruction_candidates"] == 10
    assert predictor_prompt.metadata["rounds_per_agent"] == 10
    assert predictor_prompt.metadata["proposed_instruction_count"] == 10
    assert len(predictor_prompt.metadata["candidate_search_trace"]) == 10
    assert "Candidate strategy" in predictor_prompt.system_instruction
    assert predictor_prompt.exemplar


def test_template_adapter_executes_candidate_with_observable_turns() -> None:
    adapter = TemplateBenchmarkAdapter(
        examples=[
            BenchmarkExample(example_id="ex-1", prompt="task one", reference_answer="aggregate"),
        ]
    )
    framework = MASSFramework(
        config=MASSConfig(
            task_name="dummy",
            search_space=SearchSpace(
                enabled_blocks=("aggregate", "reflect", "execute"),
                aggregate=(1, 3),
                reflect=(0, 1),
                execute=(False, True),
                max_agent_budget=6,
            ),
            candidates_per_stage=2,
            random_seed=13,
        ),
        benchmark=adapter,
    )

    results = framework.run()
    execution_details = results["stage3_workflow_prompt"].metadata["evaluation_details"][
        "executions"
    ][0]
    assert execution_details["turn_count"] >= 2
    assert execution_details["metadata"]["candidate_answer_count"] >= 1


def _prompt() -> AgentPromptBundle:
    return AgentPromptBundle(system_instruction="Solve the task.")


def test_executor_keeps_initial_workflow_single_agent() -> None:
    candidate = MASSCandidate(
        workflow=WorkflowSpec(),
        prompts={"predictor": _prompt()},
        stage="test",
    )

    execution = MASSCandidateExecutor().run_candidate(
        candidate,
        BenchmarkExample(example_id="ex-single", prompt="task"),
    )

    assert [turn.step for turn in execution.turns] == ["predict"]
    assert execution.metadata["aggregate_used"] is False
    assert execution.metadata["active_blocks"] == ["predictor"]


def test_executor_debate_uses_two_predictors_and_agent_turns() -> None:
    candidate = MASSCandidate(
        workflow=WorkflowSpec(aggregate_width=1, debate_rounds=1),
        prompts={"predictor": _prompt(), "debate": _prompt()},
        stage="test",
    )

    execution = MASSCandidateExecutor().run_candidate(
        candidate,
        BenchmarkExample(example_id="ex-debate", prompt="task"),
    )

    assert [turn.step for turn in execution.turns].count("predict") == 2
    assert [turn.step for turn in execution.turns].count("debate") == 2
    assert execution.metadata["debate_round_count"] == 1
    assert execution.metadata["aggregate_used"] is True


def test_executor_reflects_then_refines_answer() -> None:
    candidate = MASSCandidate(
        workflow=WorkflowSpec(reflect_rounds=1),
        prompts={"predictor": _prompt(), "reflect": _prompt()},
        stage="test",
    )

    execution = MASSCandidateExecutor().run_candidate(
        candidate,
        BenchmarkExample(example_id="ex-reflect", prompt="task"),
    )

    assert [turn.step for turn in execution.turns] == ["predict", "reflect", "refine"]
    assert execution.metadata["reflection_count"] == 1
    assert execution.final_answer.startswith("refine_0 response")


def test_executor_execute_feedback_flows_into_refinement() -> None:
    candidate = MASSCandidate(
        workflow=WorkflowSpec(reflect_rounds=1, execute_enabled=True),
        prompts={"predictor": _prompt(), "execute": _prompt(), "reflect": _prompt()},
        stage="test",
    )

    execution = MASSCandidateExecutor().run_candidate(
        candidate,
        BenchmarkExample(example_id="ex-execute", prompt="task"),
    )

    assert [turn.step for turn in execution.turns] == [
        "predict",
        "execute",
        "reflect",
        "refine",
    ]
    assert execution.metadata["execution_feedback_count"] == 1
    assert "with execution feedback" in execution.turns[2].content


def test_executor_uses_scicode_metadata_for_execute_feedback() -> None:
    candidate = MASSCandidate(
        workflow=WorkflowSpec(reflect_rounds=1, execute_enabled=True),
        prompts={"predictor": _prompt(), "execute": _prompt(), "reflect": _prompt()},
        stage="test",
    )
    example = BenchmarkExample(
        example_id="scicode-1",
        prompt="",
        metadata={
            "required_dependencies": "import numpy as np",
            "sub_steps": [
                {
                    "function_header": "def solve(x):",
                    "test_cases": ["assert solve(1) == 1", "assert solve(2) == 2"],
                }
            ],
        },
    )

    execution = MASSCandidateExecutor().run_candidate(candidate, example)

    assert execution.turns[1].metadata["execution_source"] == "example_metadata"
    assert "SciCode public tests" in execution.turns[1].content
    assert "public_test_cases=2" in execution.turns[1].content
    assert execution.metadata["execution_feedback_source"] == "example_metadata"


def test_executor_prefers_custom_execution_callback() -> None:
    candidate = MASSCandidate(
        workflow=WorkflowSpec(reflect_rounds=1, execute_enabled=True),
        prompts={"predictor": _prompt(), "execute": _prompt(), "reflect": _prompt()},
        stage="test",
    )
    executor = MASSCandidateExecutor(
        execution_callback=lambda answer, example, context: (
            f"tool feedback for {example.example_id}: {answer[:20]}"
        )
    )

    execution = executor.run_candidate(
        candidate,
        BenchmarkExample(
            example_id="with-tests",
            prompt="task",
            metadata={"test_cases": ["assert answer"]},
        ),
    )

    assert execution.turns[1].metadata["execution_source"] == "execution_callback"
    assert execution.turns[1].content.startswith("tool feedback for with-tests")
    assert execution.metadata["execution_feedback_source"] == "execution_callback"


def test_paper_baseline_defaults_match_mass_paper_specs() -> None:
    assert DEFAULT_MODEL == "google/gemma-4-31b-it"
    assert DEFAULT_BASELINES == ("cot", "self_consistency", "self_refine", "debate")
    assert BASELINE_SPECS["self_consistency"].calls_worst_case == 9
    assert BASELINE_SPECS["self_refine"].calls_worst_case == 11
    assert BASELINE_SPECS["debate"].calls_worst_case == 10


def test_paper_baseline_client_has_local_mock_without_mas_runtime(monkeypatch) -> None:
    monkeypatch.setenv("MAS_DISABLE_LIVE_LLM", "1")
    client = StandaloneOpenRouterClient(model=DEFAULT_MODEL, api_key="")

    result = client.generate(
        messages=[{"role": "user", "content": "Please think step by step and solve 1+1."}],
        task_id="task-1",
        agent_id="cot",
        temperature=0.7,
    )

    assert result.mock_used is True
    assert result.model == DEFAULT_MODEL
    assert "MOCK(cot)" in result.text


def test_paper_baseline_majority_vote_normalizes_answers() -> None:
    assert _majority_vote(["Answer: Blue.", "blue", "red"]) == "Answer: Blue."


def test_mass_runner_defaults_match_paper_setup(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_existing_benchmarks.py"])
    args = _parse_args()

    assert args.model == "google/gemma-4-31b-it"
    assert args.temperature == DEFAULT_MODEL_TEMPERATURE == 0.7
    assert args.candidates_per_stage == DEFAULT_TOPOLOGY_CANDIDATES == 10
    assert args.validation_repeats == DEFAULT_VALIDATION_REPEATS == 3
    assert args.topology_temperature == DEFAULT_TOPOLOGY_TEMPERATURE == 0.05
    assert args.max_tokens == 4096


def test_mass_runner_uses_task_family_search_spaces(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_existing_benchmarks.py"])
    args = _parse_args()

    assert _benchmark_family("browsecomp") == "long_context"
    assert _resolve_search_space("browsecomp", args).enabled_blocks == (
        "summarize",
        "aggregate",
        "reflect",
        "debate",
    )
    assert _benchmark_family("scicode") == "coding"
    assert _resolve_search_space("scicode", args).enabled_blocks == (
        "aggregate",
        "reflect",
        "debate",
        "execute",
    )

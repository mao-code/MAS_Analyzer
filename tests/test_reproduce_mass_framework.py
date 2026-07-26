from __future__ import annotations

from types import SimpleNamespace

from reproduce.mass import (
    MASSCandidateExecutor,
    MASSConfig,
    MASSFramework,
    MASSRuntimeRunner,
    MIPROLikeConfig,
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
    DEFAULT_FINAL_EVALUATION_REPEATS,
    DEFAULT_MAX_AGENT_BUDGET,
    DEFAULT_MODEL_TEMPERATURE,
    DEFAULT_TOPOLOGY_CANDIDATES,
    DEFAULT_TOPOLOGY_TEMPERATURE,
    DEFAULT_VALIDATION_REPEATS,
    _benchmark_family,
    _parse_args,
    _resolve_search_space,
    _split_tasks_for_mass,
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


def test_prompt_optimizer_resumes_candidate_scores(tmp_path) -> None:
    calls = {"count": 0}
    optimizer = MIPROLikePromptOptimizer(
        MIPROLikeConfig(
            instruction_candidates=2,
            rounds_per_agent=2,
            checkpoint_dir=tmp_path,
        )
    )
    examples = [BenchmarkExample(example_id="ex-1", prompt="task one")]
    workflow = WorkflowSpec()
    seed_prompt = AgentPromptBundle(system_instruction="Solve.")

    def evaluate(prompts, workflow, examples):
        calls["count"] += 1
        return CandidateEvaluation(score=float(calls["count"]))

    first = optimizer.optimize_block_prompt(
        block_name="predictor",
        seed_prompt=seed_prompt,
        base_prompts={"predictor": seed_prompt},
        examples=examples,
        workflow=workflow,
        evaluate=evaluate,
    )
    second = optimizer.optimize_block_prompt(
        block_name="predictor",
        seed_prompt=seed_prompt,
        base_prompts={"predictor": seed_prompt},
        examples=examples,
        workflow=workflow,
        evaluate=evaluate,
    )

    assert first.system_instruction == second.system_instruction
    assert calls["count"] == 3


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


class ValidationDrivenBenchmark:
    """Benchmark whose score depends on the prompt text and exposes per-example success."""

    def validation_examples(self, limit: int | None = None):
        examples = [
            BenchmarkExample(example_id="ex-1", prompt="task one", reference_answer="alpha"),
            BenchmarkExample(example_id="ex-2", prompt="task two", reference_answer="beta"),
        ]
        return examples[:limit] if limit is not None else examples

    def evaluate_candidate(self, candidate, examples):
        predictor = candidate.prompts.get("predictor")
        text = predictor.system_instruction if predictor is not None else ""
        # Reward the third proposed instruction candidate so selection is observable.
        per_example = 1.0 if "Candidate strategy 3" in text else 0.4
        return CandidateEvaluation(
            score=per_example,
            details={
                "scores": [per_example for _ in examples],
                "executions": [
                    {"example_id": ex.example_id, "final_answer": f"pred::{ex.example_id}"}
                    for ex in examples
                ],
                "benchmark_evaluations": [
                    {"example_id": ex.example_id, "success": per_example > 0.5, "score": per_example}
                    for ex in examples
                ],
            },
        )

    def execute_candidate(self, candidate, example):
        return MASSCandidateExecutor().run_candidate(candidate, example)


def test_optimizer_selects_prompt_by_validation_score() -> None:
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
            random_seed=5,
        ),
        benchmark=ValidationDrivenBenchmark(),
        prompt_optimizer=MIPROLikePromptOptimizer(),
    )

    results = framework.run()
    predictor_prompt = (
        results["stage1_block_prompt"].metadata["base_candidate"].prompts["predictor"]
    )

    assert predictor_prompt.metadata["selection_mode"] == "validation"
    assert predictor_prompt.metadata["validation_score"] == 1.0
    assert "Candidate strategy 3" in predictor_prompt.system_instruction
    trace = predictor_prompt.metadata["candidate_search_trace"]
    assert trace[0]["selection_mode"] == "validation"
    assert trace[0]["validation_example_count"] == 2
    assert trace[2]["candidate_score"] == 1.0
    assert "Candidate strategy 3" in trace[2]["instruction_preview"]
    # Demos bootstrapped from the model's own correct predictions on validation.
    assert predictor_prompt.metadata["demo_source"] == "bootstrapped_correct_predictions"
    assert "pred::ex-1" in predictor_prompt.exemplar


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
    assert args.max_agent_budget == DEFAULT_MAX_AGENT_BUDGET == 10
    assert args.instruction_candidates == 10
    assert args.prompt_search_rounds == 10
    assert args.llm_prompt_proposals is False
    assert args.validation_repeats == DEFAULT_VALIDATION_REPEATS == 3
    assert args.final_evaluation_repeats == DEFAULT_FINAL_EVALUATION_REPEATS == 3
    assert args.topology_temperature == DEFAULT_TOPOLOGY_TEMPERATURE == 0.05
    assert args.max_tokens == 0
    assert args.validation_task_offset == 0
    assert args.validation_task_limit is None
    assert args.final_task_limit is None
    assert args.final_task_offset is None


def test_mass_runner_can_split_validation_and_final_tasks(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_existing_benchmarks.py",
            "--validation-task-limit",
            "2",
            "--final-task-limit",
            "2",
        ],
    )
    args = _parse_args()
    tasks = [SimpleNamespace(task_id=f"task-{index}") for index in range(5)]

    validation_tasks, final_tasks, split_payload = _split_tasks_for_mass(args=args, tasks=tasks)

    assert [task.task_id for task in validation_tasks] == ["task-0", "task-1"]
    assert [task.task_id for task in final_tasks] == ["task-2", "task-3"]
    assert split_payload["held_out"] is True


def test_mass_runner_split_falls_back_when_no_final_tasks(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_existing_benchmarks.py",
            "--validation-task-limit",
            "2",
            "--final-task-offset",
            "9",
        ],
    )
    args = _parse_args()
    tasks = [SimpleNamespace(task_id=f"task-{index}") for index in range(2)]

    validation_tasks, final_tasks, split_payload = _split_tasks_for_mass(args=args, tasks=tasks)

    assert [task.task_id for task in final_tasks] == [task.task_id for task in validation_tasks]
    assert split_payload["held_out"] is False


def test_mass_runner_can_use_separate_validation_offset(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_existing_benchmarks.py",
            "--validation-task-offset",
            "35",
            "--validation-task-limit",
            "10",
            "--final-task-offset",
            "5",
            "--final-task-limit",
            "30",
        ],
    )
    args = _parse_args()
    tasks = [SimpleNamespace(task_id=f"task-{index}") for index in range(50)]

    validation_tasks, final_tasks, split_payload = _split_tasks_for_mass(args=args, tasks=tasks)

    assert [task.task_id for task in validation_tasks] == [
        f"task-{index}" for index in range(35, 45)
    ]
    assert [task.task_id for task in final_tasks] == [f"task-{index}" for index in range(5, 35)]
    assert split_payload["held_out"] is True


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
    assert _benchmark_family("workbench") == "tool_or_web"
    assert _resolve_search_space("stabletoolbench", args).max_agent_budget == 10


def test_paper_templates_cover_enabled_blocks_per_family() -> None:
    from reproduce.mass.prompt_templates import family_prompt_templates

    long_context = family_prompt_templates("long_context", benchmark_name="browsecomp")
    assert set(long_context) >= {"predictor", "summarize", "reflect", "debate", "aggregate"}
    assert "retrieve relevant information" in long_context["summarize"].system_instruction
    assert long_context["predictor"].input_fields == ("Question", "Available tools")
    assert "Answer" in long_context["predictor"].output_fields
    assert "search tool" in long_context["predictor"].output_contract

    coding = family_prompt_templates("coding", benchmark_name="scicode")
    assert set(coding) >= {"predictor", "reflect", "execute", "debate", "aggregate"}
    assert "python" in coding["predictor"].system_instruction.lower()

    plancraft = family_prompt_templates("general", benchmark_name="plancraft")
    assert "Action" in plancraft["predictor"].output_fields
    assert "reference" in plancraft["predictor"].output_contract
    assert "match/mismatch" in plancraft["predictor"].output_contract

    stabletoolbench = family_prompt_templates("tool_or_web", benchmark_name="stabletoolbench")
    assert "Available tools" in stabletoolbench["predictor"].input_fields
    assert "Use provided tools" in stabletoolbench["predictor"].output_contract

    workbench = family_prompt_templates("tool_or_web", benchmark_name="workbench")
    assert "python" not in workbench["predictor"].system_instruction.lower()
    assert "Available tools" in workbench["predictor"].input_fields

    assert family_prompt_templates("general") == {}


def test_paper_templates_override_framework_defaults() -> None:
    from reproduce.mass.models import AgentPromptBundle as _Bundle
    from reproduce.mass.prompt_templates import family_prompt_templates

    framework = MASSFramework(
        config=MASSConfig(
            task_name="coding",
            search_space=SearchSpace(
                enabled_blocks=("aggregate", "reflect", "execute"),
                aggregate=(1, 3),
                reflect=(0, 1),
                execute=(False, True),
                max_agent_budget=6,
            ),
            candidates_per_stage=2,
            random_seed=4,
            prompt_templates=family_prompt_templates("coding"),
        ),
        benchmark=DummyBenchmark(),
    )
    prompts = framework._default_prompts(framework.config.search_space)
    assert isinstance(prompts["predictor"], _Bundle)
    assert "python" in prompts["predictor"].system_instruction.lower()


def test_prompt_optimizer_preserves_fixed_io_contract() -> None:
    seed_prompt = AgentPromptBundle(
        system_instruction="Solve.",
        input_fields=("Question", "Context"),
        output_fields=("Answer",),
        output_contract="Only return the answer.",
    )
    optimizer = MIPROLikePromptOptimizer(
        MIPROLikeConfig(
            instruction_candidates=1,
            rounds_per_agent=1,
            instruction_proposer=lambda *_: ["Use evidence carefully."],
        )
    )

    optimized = optimizer.optimize_block_prompt(
        block_name="predictor",
        seed_prompt=seed_prompt,
        base_prompts={"predictor": seed_prompt},
        examples=[BenchmarkExample(example_id="ex-1", prompt="task")],
        workflow=WorkflowSpec(),
        evaluate=lambda prompts, workflow, examples: CandidateEvaluation(score=1.0),
    )

    assert optimized.system_instruction == "Use evidence carefully."
    assert optimized.input_fields == ("Question", "Context")
    assert optimized.output_fields == ("Answer",)
    assert optimized.output_contract == "Only return the answer."


def test_workflow_prompt_optimizer_scores_joint_prompt_sets() -> None:
    prompts = {
        "predictor": AgentPromptBundle(
            system_instruction="Predict seed.",
            input_fields=("Question",),
            output_fields=("Answer",),
        ),
        "aggregate": AgentPromptBundle(
            system_instruction="Aggregate seed.",
            input_fields=("Question", "Solutions"),
            output_fields=("Answer",),
        ),
    }

    def proposer(block_name, *_args):
        return [f"{block_name} bad", f"{block_name} good"]

    optimizer = MIPROLikePromptOptimizer(
        MIPROLikeConfig(
            instruction_candidates=2,
            rounds_per_agent=1,
            instruction_proposer=proposer,
            bootstrap_demos=False,
        )
    )

    def evaluate(candidate_prompts, workflow, examples):
        instructions = {key: value.system_instruction for key, value in candidate_prompts.items()}
        score = 1.0 if instructions == {
            "predictor": "predictor good",
            "aggregate": "aggregate good",
        } else 0.0
        return CandidateEvaluation(score=score, details={"instructions": instructions})

    optimized = optimizer.optimize_workflow_prompts(
        workflow=WorkflowSpec(aggregate_width=3),
        prompts=prompts,
        examples=[BenchmarkExample(example_id="ex-1", prompt="task")],
        evaluate=evaluate,
    )

    assert optimized["predictor"].system_instruction == "predictor good"
    assert optimized["aggregate"].system_instruction == "aggregate good"
    assert optimized["predictor"].metadata["selection_mode"] == "validation_joint"
    trace = optimized["predictor"].metadata["joint_candidate_search_trace"]
    assert [item["score"] for item in trace] == [0.0, 1.0]


def test_plancraft_prompt_proposer_filters_reference_matching() -> None:
    seed_prompt = AgentPromptBundle(
        system_instruction="Choose a valid PlanCraft action.",
        input_fields=("Observation", "Target item"),
        output_fields=("Action",),
        output_contract="Return move, smelt, or impossible.",
    )
    optimizer = MIPROLikePromptOptimizer(
        MIPROLikeConfig(
            instruction_candidates=2,
            rounds_per_agent=1,
            benchmark_name="plancraft",
            instruction_proposer=lambda *_: [
                "Verify whether the output matches the reference answer.",
                "Choose the next valid move action from the inventory.",
            ],
        )
    )

    optimized = optimizer.optimize_block_prompt(
        block_name="predictor",
        seed_prompt=seed_prompt,
        base_prompts={"predictor": seed_prompt},
        examples=[BenchmarkExample(example_id="ex-1", prompt="Craft a stick")],
        workflow=WorkflowSpec(),
        evaluate=lambda prompts, workflow, examples: CandidateEvaluation(
            score=1.0
            if "move action" in prompts["predictor"].system_instruction
            else 0.0
        ),
    )

    assert "reference" not in optimized.system_instruction.lower()
    assert "move action" in optimized.system_instruction


def test_run_existing_benchmarks_keep_best_selects_best_stage_payload() -> None:
    from reproduce.mass.run_existing_benchmarks import _select_final_stage_payload

    payload = {
        "final_stage_name": "stage3_workflow_prompt",
        "final_stage": {"best_score": 0.0, "best_candidate": {"stage": "stage3"}},
        "best_score": 0.0,
        "stages": {
            "stage1_block_prompt": {
                "best_score": 0.4,
                "best_candidate": {"stage": "stage1"},
            },
            "stage2_topology": {
                "best_score": 0.7,
                "best_candidate": {"stage": "stage2"},
            },
            "stage3_workflow_prompt": {
                "best_score": 0.0,
                "best_candidate": {"stage": "stage3"},
            },
        },
    }

    selected = _select_final_stage_payload(
        payload,
        keep_best_after_global_prompt_stage=True,
    )

    assert selected["final_stage_name"] == "stage2_topology"
    assert selected["best_score"] == 0.7
    assert selected["final_stage"]["best_candidate"]["stage"] == "stage2"


def test_executor_renders_paper_style_signature() -> None:
    executor = MASSCandidateExecutor()
    prompt = AgentPromptBundle(
        system_instruction="Answer from context.",
        input_fields=("Question", "Context"),
        output_fields=("Answer",),
        output_contract="Only return the answer.",
    )

    rendered = executor._render_prompt_text(prompt, context={})

    assert "Follow the following format." in rendered
    assert "Question: ${question}" in rendered
    assert "Context: ${context}" in rendered
    assert "Answer: ${answer}" in rendered
    assert "Only return the answer." in rendered


def test_runtime_runner_is_exported() -> None:
    assert MASSRuntimeRunner.__name__ == "MASSRuntimeRunner"


def test_runtime_runner_normalizes_plancraft_action_output() -> None:
    assert (
        MASSRuntimeRunner._normalize_plancraft_action(
            "move: from [0] to [I1] with quantity 1 "
            "(Note: this slot was mentioned in prior context.)"
        )
        == "move: from [0] to [I1] with quantity 1"
    )
    assert (
        MASSRuntimeRunner._normalize_plancraft_action(
            "Action: move: from [I2] to [A1] with quantity 3"
        )
        == "move: from [I2] to [A1] with quantity 3"
    )
    assert (
        MASSRuntimeRunner._normalize_plancraft_action(
            "impossible: missing sticks to craft fishing rod\nReasoning: extra text"
        )
        == "impossible: missing sticks to craft fishing rod"
    )

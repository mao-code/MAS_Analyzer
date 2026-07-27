from __future__ import annotations

import os

from benchmark.base import BenchmarkTask
from reproduce.agentsquare.models import AgentSquareModule
from reproduce.agentsquare.modules import spec_from_names
from reproduce.agentsquare.preflight import run_preflight
from reproduce.agentsquare.run_existing_benchmarks import (
    _build_predictor_prompt,
    _candidate_specs,
    _evolution_specs,
    _inject_benchmark_runtime_config,
    _module_archive_payload,
    _modules_from_archive_payload,
    _parse_args,
    _parse_module_proposals,
    _parse_predictor_response,
    _predict_candidate_scores,
)
from reproduce.agentsquare.runtime_runner import AgentSquareRuntimeRunner
from reproduce.agentsquare.status import collect_status
from reproduce.agentsquare.summarize_results import _format_latex_row, summarize_run


class DummyLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1

        class Result:
            text = "Answer: 4"
            token_in = 10
            token_out = 3
            cost_usd = 0.0
            model = "mock"
            mock_used = True
            metadata = {}
            tool_calls = []

        return Result()


class PredictorLLM:
    def __init__(self, text: str, *, mock_used: bool = False) -> None:
        self.text = text
        self.mock_used = mock_used
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        text = self.text
        mock_used = self.mock_used

        class Result:
            token_in = 10
            token_out = 3
            cost_usd = 0.0
            model = "mock"
            metadata = {}
            tool_calls = []

            def __init__(self) -> None:
                self.text = text
                self.mock_used = mock_used

        return Result()


def test_agentsquare_runner_executes_default_spec() -> None:
    task = BenchmarkTask(task_id="t1", prompt="What is 2+2?", reference_answer="4")
    runner = AgentSquareRuntimeRunner(spec=spec_from_names(), llm_client=DummyLLM())  # type: ignore[arg-type]

    result = runner.run_task(task, run_index=0, seed=42, benchmark_name="dummy")

    assert result.final_answer == "4"
    assert result.run_metadata["agentsquare_reproduce"] is True
    assert result.run_metadata["spec"]["reasoning"]["name"] == "IO"
    assert len(result.trace_events) == 1


def test_agentsquare_spec_can_enable_all_module_slots() -> None:
    spec = spec_from_names(planning="DEPS", reasoning="COT", tooluse="IO", memory="Summary")

    payload = spec.to_payload()

    assert payload["planning"]["name"] == "DEPS"
    assert payload["reasoning"]["name"] == "COT"
    assert payload["tooluse"]["name"] == "IO"
    assert payload["memory"]["name"] == "Summary"


def test_agentsquare_search_candidates_start_with_seed_spec() -> None:
    class Args:
        planning = "None"
        reasoning = "IO"
        tooluse = "None"
        memory = "None"
        max_search_candidates = 3

    candidates = _candidate_specs(Args())

    assert candidates[0] == {
        "planning": "None",
        "reasoning": "IO",
        "tooluse": "None",
        "memory": "None",
    }
    assert len(candidates) == 3


def test_agentsquare_cot_sc_runs_three_reasoning_samples() -> None:
    task = BenchmarkTask(task_id="t1", prompt="What is 2+2?", reference_answer="4")
    llm = DummyLLM()
    runner = AgentSquareRuntimeRunner(
        spec=spec_from_names(reasoning="COT-SC"),
        llm_client=llm,  # type: ignore[arg-type]
    )

    result = runner.run_task(task, run_index=0, seed=42, benchmark_name="dummy")

    assert result.final_answer == "4"
    assert llm.calls == 3
    assert result.run_metadata["messages_sent_total"] == 3
    assert len(result.trace_events) == 3


def test_agentsquare_evolution_mutates_one_module_slot() -> None:
    current = {
        "planning": "None",
        "reasoning": "IO",
        "tooluse": "None",
        "memory": "None",
    }

    evolved = _evolution_specs(current)

    assert current not in evolved
    assert {
        "planning": "IO",
        "reasoning": "IO",
        "tooluse": "None",
        "memory": "None",
    } in evolved
    assert all(
        sum(1 for slot, value in candidate.items() if current[slot] != value) == 1
        for candidate in evolved
    )


def test_agentsquare_predictor_prefers_measured_candidates() -> None:
    current = {
        "planning": "None",
        "reasoning": "IO",
        "tooluse": "None",
        "memory": "None",
    }
    measured = {
        tuple(
            sorted(
                {"planning": "IO", "reasoning": "IO", "tooluse": "None", "memory": "None"}.items()
            )
        ): 0.7
    }

    payload = _predict_candidate_scores(
        candidates=[
            current,
            {"planning": "IO", "reasoning": "IO", "tooluse": "None", "memory": "None"},
        ],
        current_agent=current,
        measured_scores=measured,
    )
    predictions = payload["ranked_candidates"]

    assert max(predictions, key=lambda item: item["predicted_score"])["reason"] == (
        "previous_validation_score"
    )
    assert payload["predictor"]["mode"] == "heuristic"


def test_agentsquare_parse_predictor_response_extracts_json() -> None:
    candidates = [
        {"planning": "None", "reasoning": "IO", "tooluse": "None", "memory": "None"},
        {"planning": "IO", "reasoning": "COT", "tooluse": "None", "memory": "None"},
    ]

    parsed = _parse_predictor_response(
        'prefix {"ranked_candidates":[{"index":1,"predicted_score":0.8,"reason":"better"}]}',
        candidates=candidates,
    )

    assert parsed == [
        {
            "spec_names": candidates[1],
            "predicted_score": 0.8,
            "reason": "better",
        }
    ]


def test_agentsquare_llm_predictor_ranks_candidates() -> None:
    class Args:
        predictor_mode = "llm"
        model_agent_type = "default"
        predictor_max_tokens = 1200

    candidates = [
        {"planning": "None", "reasoning": "IO", "tooluse": "None", "memory": "None"},
        {"planning": "IO", "reasoning": "COT", "tooluse": "None", "memory": "None"},
    ]
    llm = PredictorLLM(
        '{"ranked_candidates":[{"index":1,"predicted_score":0.9,"reason":"stronger plan"}]}'
    )

    payload = _predict_candidate_scores(
        candidates=candidates,
        current_agent=candidates[0],
        measured_scores={},
        llm_client=llm,  # type: ignore[arg-type]
        args=Args(),
        benchmark_name="dummy",
        iteration=0,
        tested_cases=[],
    )

    assert llm.calls == 1
    assert payload["predictor"]["mode"] == "llm"
    assert payload["ranked_candidates"][0]["spec_names"] == candidates[1]
    assert payload["ranked_candidates"][0]["predictor_source"] == "llm"


def test_agentsquare_llm_predictor_falls_back_on_mock() -> None:
    class Args:
        predictor_mode = "llm"
        model_agent_type = "default"
        predictor_max_tokens = 1200

    current = {"planning": "None", "reasoning": "IO", "tooluse": "None", "memory": "None"}
    llm = PredictorLLM('{"ranked_candidates":[]}', mock_used=True)

    payload = _predict_candidate_scores(
        candidates=[current],
        current_agent=current,
        measured_scores={},
        llm_client=llm,  # type: ignore[arg-type]
        args=Args(),
        benchmark_name="dummy",
        iteration=0,
        tested_cases=[],
    )

    assert payload["predictor"]["mode"] == "heuristic"
    assert payload["predictor"]["fallback_used"] is True
    assert payload["ranked_candidates"][0]["predictor_source"] == "heuristic"


def test_agentsquare_parse_generated_module_proposals() -> None:
    modules = _parse_module_proposals(
        """
        {
          "modules": [
            {
              "module_type": "reasoning",
              "name": "Evidence First",
              "thought": "Ground answers in retrieved evidence before finalizing.",
              "prompt": "First identify evidence, then answer concisely.",
              "code": "class ReasoningEvidenceFirst: pass"
            }
          ]
        }
        """,
        benchmark_name="browsecomp",
        iteration=2,
    )

    module = modules["reasoning"]
    assert module.name.startswith("GEN_browsecomp_2_reasoning_Evidence_First")
    assert module.metadata["generated_by"] == "agentsquare_module_evolution"
    assert "evidence" in module.prompt.lower()


def test_agentsquare_generated_modules_enter_candidate_pool_and_archive() -> None:
    class Args:
        planning = "None"
        reasoning = "IO"
        tooluse = "None"
        memory = "None"
        max_search_candidates = 50

    generated = {
        "reasoning": {
            "GEN_dummy_0_reasoning_Checker": AgentSquareModule(
                name="GEN_dummy_0_reasoning_Checker",
                module_type="reasoning",
                thought="Check assumptions before answering.",
                prompt="Check assumptions before answering.",
            )
        }
    }

    candidates = _candidate_specs(Args(), generated_modules=generated)
    archived = _modules_from_archive_payload(_module_archive_payload(generated))

    assert any(
        candidate["reasoning"] == "GEN_dummy_0_reasoning_Checker" for candidate in candidates
    )
    assert archived["reasoning"]["GEN_dummy_0_reasoning_Checker"].prompt == (
        "Check assumptions before answering."
    )


def test_agentsquare_predictor_prompt_includes_generated_module_context() -> None:
    generated = {
        "reasoning": {
            "GEN_dummy_0_reasoning_Checker": AgentSquareModule(
                name="GEN_dummy_0_reasoning_Checker",
                module_type="reasoning",
                thought="Check assumptions before answering.",
                prompt="Verify evidence before final answer.",
            )
        }
    }
    candidate = {
        "planning": "None",
        "reasoning": "GEN_dummy_0_reasoning_Checker",
        "tooluse": "None",
        "memory": "None",
    }

    prompt = _build_predictor_prompt(
        candidates=[candidate],
        current_agent={
            "planning": "None",
            "reasoning": "IO",
            "tooluse": "None",
            "memory": "None",
        },
        measured_scores={},
        generated_modules=generated,
        benchmark_name="dummy",
        iteration=0,
        tested_cases=[],
    )

    assert "Verify evidence before final answer." in prompt[1]["content"]


def test_agentsquare_summarizer_uses_population_std(tmp_path) -> None:
    run_root = tmp_path / "run"
    bench_dir = run_root / "math500"
    bench_dir.mkdir(parents=True)
    (bench_dir / "results.json").write_text(
        """
        {
          "score": 0.5,
          "task_count": 30,
          "run_count": 90,
          "per_run_scores": {"0": 0.4, "1": 0.5, "2": 0.6},
          "search": {
            "best_spec_names": {
              "planning": "None",
              "reasoning": "IO",
              "tooluse": "None",
              "memory": "None"
            }
          }
        }
        """,
        encoding="utf-8",
    )

    payload = summarize_run(run_root=run_root, benchmarks=("math500",))

    row = payload["benchmarks"]["math500"]
    assert row["mean_pct"] == 50.0
    assert round(row["std_pct"], 1) == 8.2
    assert payload["average"]["mean_pct"] == 50.0


def test_agentsquare_summarizer_formats_five_benchmark_table_row(tmp_path) -> None:
    run_root = tmp_path / "run"
    benchmarks = ("browsecomp", "math500", "plancraft", "stabletoolbench", "workbench")
    for index, benchmark in enumerate(benchmarks, start=1):
        bench_dir = run_root / benchmark
        bench_dir.mkdir(parents=True)
        score = index / 10
        (bench_dir / "results.json").write_text(
            __import__("json").dumps(
                {
                    "score": score,
                    "task_count": 30,
                    "run_count": 90,
                    "per_run_scores": {"0": score, "1": score, "2": score},
                }
            ),
            encoding="utf-8",
        )

    payload = summarize_run(run_root=run_root, benchmarks=benchmarks)
    row = _format_latex_row(payload, system_name="AgentSquare")

    assert row == (
        r"AgentSquare & $10.0_{\pm 0.0}$ & $20.0_{\pm 0.0}$ & "
        r"$30.0_{\pm 0.0}$ & $40.0_{\pm 0.0}$ & $50.0_{\pm 0.0}$ & "
        r"$30.0_{\pm 0.0}$ \\"
    )


def test_agentsquare_search_resume_uses_existing_results(tmp_path) -> None:
    from reproduce.agentsquare.run_existing_benchmarks import _run_search

    output_dir = tmp_path / "search"
    output_dir.mkdir()
    expected = {
        "best_spec_names": {
            "planning": "None",
            "reasoning": "IO",
            "tooluse": "None",
            "memory": "None",
        },
        "best_score": 0.5,
        "module_archive": {},
    }
    (output_dir / "search_results.json").write_text(
        __import__("json").dumps(expected),
        encoding="utf-8",
    )

    class Args:
        resume = True

    class ExplodingBenchmark:
        def run(self, *args, **kwargs):
            raise AssertionError("should not run benchmark when search resume exists")

    payload = _run_search(
        benchmark=ExplodingBenchmark(),
        benchmark_name="dummy",
        args=Args(),
        llm_client=None,  # type: ignore[arg-type]
        validation_tasks=[object()],
        output_dir=output_dir,
    )

    assert payload == expected


def test_agentsquare_status_reports_completed_runs(tmp_path) -> None:
    run_root = tmp_path / "run"
    bench_root = run_root / "math500"
    (bench_root / "final" / "runs" / "task_a").mkdir(parents=True)
    (bench_root / "split.json").write_text(
        __import__("json").dumps({"final_task_ids": ["task_a"]}),
        encoding="utf-8",
    )
    (bench_root / "results.json").write_text(
        __import__("json").dumps({"score": 1.0, "task_count": 1, "run_count": 1}),
        encoding="utf-8",
    )
    (bench_root / "search").mkdir()
    (bench_root / "search" / "search_results.json").write_text(
        __import__("json").dumps({"best_spec_names": {"reasoning": "IO"}}),
        encoding="utf-8",
    )
    (bench_root / "final" / "runs" / "task_a" / "run_0.json").write_text(
        __import__("json").dumps({"score": 1.0}),
        encoding="utf-8",
    )

    payload = collect_status(run_root=run_root, benchmarks=("math500",))

    row = payload["benchmarks"]["math500"]
    assert row["search_done"] is True
    assert row["results_done"] is True
    assert row["completed_runs"] == 1
    assert row["expected_runs"] == 1


def test_agentsquare_browsecomp_injects_openrouter_judge_config(monkeypatch) -> None:
    class Args:
        openrouter_base_url = "https://openrouter.ai/api/v1"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config: dict[str, object] = {}

    _inject_benchmark_runtime_config(
        benchmark_cfg=config,
        benchmark_name="browsecomp",
        args=Args(),
    )

    assert config["openrouter"] == {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
    }


def test_agentsquare_formal_preflight_config_keeps_official_eval_modes() -> None:
    class Args:
        config = "config/reproduce_agentsquare.example.toml"
        openrouter_base_url = "https://openrouter.ai/api/v1"
        benchmark = ["browsecomp", "stabletoolbench"]
        task_limit = 40
        validation_task_limit = 10
        final_task_offset = 10
        final_task_limit = 30

    payload = run_preflight(args=Args())

    assert payload["benchmarks"]["browsecomp"]["ok"] is True
    assert payload["benchmarks"]["browsecomp"]["eval_mode"] == "llm_judge"
    assert payload["benchmarks"]["browsecomp"]["max_tool_iterations"] == 8
    assert payload["benchmarks"]["stabletoolbench"]["ok"] is True
    assert payload["benchmarks"]["stabletoolbench"]["eval_mode"] == "fac"
    assert payload["benchmarks"]["stabletoolbench"]["max_tool_iterations"] == 8


def test_agentsquare_cli_defaults_to_formal_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--benchmark", "math500"],
    )

    args = _parse_args()

    assert args.config == "config/reproduce_agentsquare.example.toml"


def test_agentsquare_max_tokens_zero_clears_openrouter_limit(monkeypatch) -> None:
    from reproduce.agentsquare.run_existing_benchmarks import main

    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "123")

    class DummyConfig:
        def __init__(self) -> None:
            self.openrouter = type("OpenRouter", (), {"base_url": "", "timeout_s": 0})()
            self.models = {}

    class DummyClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--benchmark",
            "math500",
            "--task-limit",
            "1",
            "--max-tokens",
            "0",
        ],
    )
    monkeypatch.setattr(
        "reproduce.agentsquare.run_existing_benchmarks.load_experiment_config",
        lambda _path: DummyConfig(),
    )
    monkeypatch.setattr(
        "reproduce.agentsquare.run_existing_benchmarks.OpenRouterLLMClient",
        DummyClient,
    )
    monkeypatch.setattr(
        "reproduce.agentsquare.run_existing_benchmarks._run_one_benchmark",
        lambda **kwargs: {"score": 1.0},
    )

    main()

    assert "OPENROUTER_MAX_TOKENS" not in os.environ


def test_agentsquare_formal_launcher_keeps_table_contract() -> None:
    script = (
        __import__("pathlib").Path("scripts/baselines/run_agentsquare_formal.sh").read_text(encoding="utf-8")
    )

    for benchmark in ("browsecomp", "math500", "plancraft", "stabletoolbench", "workbench"):
        assert f"--benchmark {benchmark}" in script
    for expected in (
        'search_iterations="${AGENTSQUARE_SEARCH_ITERATIONS:-3}"',
        'max_search_candidates="${AGENTSQUARE_MAX_SEARCH_CANDIDATES:-3}"',
        "--task-limit 40",
        "--validation-task-limit 10",
        "--final-task-offset 10",
        "--final-task-limit 30",
        "--runs-per-task 3",
        "--validation-repeats 1",
        "--search",
        "--search-iterations",
        "--module-evolution-mode llm",
        "--predictor-mode llm",
        "--model google/gemma-4-31b-it",
        "--temperature 1",
        "--max-tokens 0",
        "--resume",
        "--keep-going",
        "stabletoolbench_virtual_server.py",
        "http://127.0.0.1:8080/virtual/healthz",
        "--cache-root benchmark/stabletoolbench/tool_response_cache",
    ):
        assert expected in script

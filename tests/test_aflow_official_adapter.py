from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.base import BenchmarkEvaluation, BenchmarkTask
from MAS.llm import LLMResult
from reproduce.aflow.official import optimizer as aflow_optimizer
from reproduce.aflow.official.optimizer import OfficialAFlowBenchmarkOptimizer


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> LLMResult:
        self.calls.append(kwargs)
        agent_id = str(kwargs.get("agent_id", ""))
        if agent_id == "aflow_optimizer":
            text = """<modification>Use answer_generate for a concise final answer.</modification>
<graph>
class Workflow(OfficialWorkflowBase):
    async def __call__(self, problem: str):
        generated = await self.answer_generate(input=problem)
        return generated.get("answer", generated.get("response", "")), self.llm.get_usage_summary()["total_cost"]
</graph>
<prompt>
# no custom prompt needed
</prompt>"""
        else:
            text = "<thought>done</thought><answer>correct answer</answer>"
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="fake-model",
            mock_used=False,
            metadata={"provider": "fake", "finish_reason": "stop", "hit_output_limit": False},
            tool_calls=[],
        )

    def model_for_agent_type(self, agent_type: str) -> str:
        return "fake-model"


class FakeBenchmark:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def load_tasks(self, task_limit: int | None = None) -> list[BenchmarkTask]:
        tasks = []
        for index in range(4):
            tasks.append(
                BenchmarkTask(
                    task_id=f"task_{index}",
                    prompt=f"Return the correct answer for task {index}.",
                    reference_answer="correct answer",
                    metadata={},
                )
            )
        return tasks[:task_limit]

    def run(
        self,
        task: BenchmarkTask,
        runner: Any,
        run_index: int,
        seed: int,
    ):
        return runner.run_task(task, run_index=run_index, seed=seed, benchmark_name="fakebench")

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        success = "correct answer" in prediction
        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=1.0 if success else 0.0,
            success=success,
            details={"prediction": prediction, "run_metadata": run_metadata or {}},
        )


def test_official_aflow_optimizer_writes_round_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(aflow_optimizer, "get_benchmark", lambda *_args, **_kwargs: FakeBenchmark())

    optimizer = OfficialAFlowBenchmarkOptimizer(
        benchmark_name="fakebench",
        benchmark_config={},
        llm_client=FakeLLMClient(),
        output_dir=tmp_path,
        task_limit=2,
        validation_rounds=1,
        test_task_limit=2,
        test_offset=1,
        runs_per_task=1,
        max_rounds=2,
        sample=1,
        seed=42,
        model_agent_type="default",
        temperature=1.0,
        allow_mock=False,
    )

    payload = optimizer.optimize()

    assert payload["method"] == "aflow_official_adapter"
    assert payload["best_score"] == 1.0
    assert payload["test_score"] == 1.0
    assert (tmp_path / "workflows/round_1/graph.py").exists()
    assert (tmp_path / "workflows/round_2/graph.py").exists()
    assert (tmp_path / "workflows/round_2/prompt.py").exists()
    assert (tmp_path / "workflows/round_2/experience.json").exists()
    assert (tmp_path / "best_workflow/graph.py").exists()

    results = json.loads((tmp_path / "workflows/results.json").read_text(encoding="utf-8"))
    assert [row["round"] for row in results] == [1, 2]

    experience = json.loads(
        (tmp_path / "workflows/round_2/experience.json").read_text(encoding="utf-8")
    )
    assert experience["father node"] == 1
    assert "answer_generate" in experience["modification"]

    trace_path = tmp_path / "workflows/round_2/runs/task_0/run_0.trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))["trace"]
    assert trace[0]["event_type"] == "act"
    assert trace[0]["payload"]["node"] == "aflow_official:AnswerGenerate"

    validation_summary = json.loads(
        (tmp_path / "workflows/round_2/summary.json").read_text(encoding="utf-8")
    )
    test_summary = json.loads(
        (tmp_path / f"test/round_{payload['best_round']}/summary.json").read_text(encoding="utf-8")
    )
    assert validation_summary["round"]["task_ids"] == ["task_0"]
    assert test_summary["round"]["task_ids"] == ["task_1", "task_2"]
    assert not set(validation_summary["round"]["task_ids"]).intersection(
        test_summary["round"]["task_ids"]
    )

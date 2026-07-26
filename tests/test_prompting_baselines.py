from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from benchmark.base import BenchmarkTask
from descriptor.schema import TraceEvent
from MAS.prompting_baselines import PromptingBaselineRunner
from MAS.runner import MASRunResult


def _event(actor: str = "agent", output: str = "") -> TraceEvent:
    return TraceEvent(
        timestamp_start=0.0,
        timestamp_end=1.0,
        actor=actor,
        event_type="finalize",
        payload={"output": output},
        token_in=1,
        token_out=1,
        latency_ms=1.0,
        cost_usd=0.0,
    )


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        text = "feedback" if "feedback" in str(kwargs.get("agent_id")) else "revised"

        @dataclass
        class Result:
            text: str
            token_in: int = 1
            token_out: int = 1
            cost_usd: float = 0.0
            model: str = "mock"
            mock_used: bool = True

        return Result(text=text)


class FakeBaseRunner:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = list(answers or ["answer"])
        self.calls: list[dict[str, Any]] = []
        self.config = object()
        self.openrouter_client = FakeLLMClient()
        self.llm_client = self.openrouter_client
        self.engine = object()

    def run_task(self, task: Any, run_index: int, seed: int, **kwargs: Any) -> MASRunResult:
        call_index = len(self.calls)
        self.calls.append({"task": task, "run_index": run_index, "seed": seed, "kwargs": kwargs})
        answer = self.answers[min(call_index, len(self.answers) - 1)]
        return MASRunResult(
            final_answer=answer,
            trace_events=[_event(output=answer)],
            run_metadata={"tool_calls_total": call_index, "seed": seed},
        )

    def reload_self_evolved_skill(self) -> None:
        return None


def test_cot_injects_step_by_step_instruction() -> None:
    base = FakeBaseRunner()
    runner = PromptingBaselineRunner(base, baseline="cot")
    task = BenchmarkTask(task_id="t1", prompt="Question?", reference_answer="")

    result = runner.run_task(task=task, run_index=0, seed=42)

    assert result.run_metadata["prompting_baseline"] == "cot"
    assert "Think step by step" in base.calls[0]["task"].prompt


def test_self_consistency_samples_three_and_selects_majority() -> None:
    base = FakeBaseRunner(["A", "B", "B"])
    runner = PromptingBaselineRunner(base, baseline="self_consistency")
    task = BenchmarkTask(task_id="t1", prompt="Question?", reference_answer="")

    result = runner.run_task(task=task, run_index=0, seed=42)

    assert len(base.calls) == 3
    assert result.final_answer == "B"
    assert result.run_metadata["self_consistency_samples"] == 3
    assert result.run_metadata["self_consistency_selected_index"] == 1
    assert [call["seed"] for call in base.calls] == [42, 100042, 200042]


def test_self_consistency_writes_incremental_checkpoint(tmp_path) -> None:
    base = FakeBaseRunner(["A", "B", "B"])
    runner = PromptingBaselineRunner(base, baseline="self_consistency")
    runner.set_run_checkpoint_context(task_dir=tmp_path, run_index=0)
    task = BenchmarkTask(task_id="t1", prompt="Question?", reference_answer="")

    runner.run_task(task=task, run_index=0, seed=42)

    checkpoint = json.loads((tmp_path / "run_0.prompting_checkpoint.json").read_text())
    assert checkpoint["prompting_baseline"] == "self_consistency"
    assert checkpoint["phase"] == "self_consistency_complete"
    assert checkpoint["complete"] is True
    assert checkpoint["samples_completed"] == 3
    assert checkpoint["selected_answer"] == "B"


def test_self_refine_runs_three_feedback_revision_rounds_by_default() -> None:
    base = FakeBaseRunner(["initial"])
    runner = PromptingBaselineRunner(base, baseline="self_refine")
    task = BenchmarkTask(task_id="t1", prompt="Question?", reference_answer="")

    result = runner.run_task(task=task, run_index=0, seed=42)

    assert len(base.calls) == 1
    assert len(base.openrouter_client.calls) == 6
    assert result.final_answer == "revised"
    assert result.run_metadata["self_refine_rounds"] == 3
    assert len(result.run_metadata["self_refine_records"]) == 3


def test_self_refine_writes_incremental_checkpoint(tmp_path) -> None:
    base = FakeBaseRunner(["initial"])
    runner = PromptingBaselineRunner(base, baseline="self_refine")
    runner.set_run_checkpoint_context(task_dir=tmp_path, run_index=0)
    task = BenchmarkTask(task_id="t1", prompt="Question?", reference_answer="")

    runner.run_task(task=task, run_index=0, seed=42)

    checkpoint = json.loads((tmp_path / "run_0.prompting_checkpoint.json").read_text())
    assert checkpoint["prompting_baseline"] == "self_refine"
    assert checkpoint["phase"] == "self_refine_complete"
    assert checkpoint["complete"] is True
    assert checkpoint["rounds_completed"] == 3
    assert checkpoint["final_answer"] == "revised"

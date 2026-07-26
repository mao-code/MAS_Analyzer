from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.base import BenchmarkEvaluation, BenchmarkTask
from MAS.llm import LLMResult
from reproduce.aflow.official import optimizer as aflow_optimizer
from reproduce.aflow.official.optimizer import OfficialAFlowBenchmarkOptimizer
from reproduce.aflow.official.runtime import (
    AFlowExecutionContext,
    OfficialLLM,
    Operators,
    normalize_workflow_answer,
)


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
        runs_per_task=2,
        workers=2,
        retries=1,
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
    assert (tmp_path / "workflows/round_1/runs/task_0/run_1.json").exists()

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
    checkpoint_path = tmp_path / "workflows/round_2/runs/task_0/checkpoints/run_0.attempt_0.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "completed"
    assert checkpoint["operator_calls_completed"] == 1

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


def test_answer_generate_preserves_benchmark_message_prompt(tmp_path: Path) -> None:
    llm_client = FakeLLMClient()
    context = AFlowExecutionContext(
        llm_client=llm_client,
        agent_type="default",
        task_id="plancraft:task_0",
        run_index=0,
        temperature=1.0,
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    operators = Operators(OfficialLLM(context))
    messages = [
        {"role": "system", "content": "official system prompt"},
        {"role": "user", "content": "official observation prompt"},
    ]

    result = __import__("asyncio").run(operators.answer_generate(messages))

    assert result["answer"] == "<thought>done</thought><answer>correct answer</answer>"
    assert llm_client.calls[0]["prompt"] == messages


def test_custom_preserves_benchmark_message_prompt(tmp_path: Path) -> None:
    llm_client = FakeLLMClient()
    context = AFlowExecutionContext(
        llm_client=llm_client,
        agent_type="default",
        task_id="plancraft:task_0",
        run_index=0,
        temperature=1.0,
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    operators = Operators(OfficialLLM(context))
    messages = [
        {"role": "system", "content": "official system prompt"},
        {"role": "user", "content": "official observation prompt"},
    ]

    result = __import__("asyncio").run(operators.custom(messages, ""))

    assert result["response"] == "<thought>done</thought><answer>correct answer</answer>"
    assert llm_client.calls[0]["prompt"] == messages


def test_call_xml_fills_missing_final_field(tmp_path: Path) -> None:
    class PartialXmlClient(FakeLLMClient):
        def generate(self, **kwargs: Any) -> LLMResult:
            self.calls.append(kwargs)
            return LLMResult(
                text="<thought>not enough evidence</thought>",
                token_in=10,
                token_out=5,
                cost_usd=0.0,
                model="fake-model",
                mock_used=False,
                metadata={"provider": "fake", "finish_reason": "stop", "hit_output_limit": False},
                tool_calls=[],
            )

    context = AFlowExecutionContext(
        llm_client=PartialXmlClient(),
        agent_type="default",
        task_id="browsecomp:task_0",
        run_index=0,
        temperature=1.0,
        checkpoint_path=tmp_path / "checkpoint.json",
    )

    result = __import__("asyncio").run(
        OfficialLLM(context).call_xml("question", ["thought", "answer"], operator="AnswerGenerate")
    )

    assert result["thought"] == "not enough evidence"
    assert result["answer"] == "<thought>not enough evidence</thought>"


def test_normalize_workflow_answer_extracts_xml_and_caps_broken_thought() -> None:
    assert normalize_workflow_answer("<thought>brief</thought><answer>Stefanel</answer>") == "Stefanel"
    assert normalize_workflow_answer("plain answer") == "plain answer"
    assert normalize_workflow_answer("<thought>" + "long " * 500) == "no-answer"
    assert normalize_workflow_answer(("analysis\n" * 500) + "Final Answer: **Brigitte Bardot**") == (
        "Brigitte Bardot"
    )
    assert normalize_workflow_answer("analysis only\n" * 500) == "no-answer"
    assert (
        normalize_workflow_answer(
            "Based on the available information, the individual is **Pouneh Shabani-Jadidi**. "
            "She meets the criteria."
        )
        == "Pouneh Shabani-Jadidi"
    )
    assert normalize_workflow_answer("I am unable to identify the specific learning institution.") == (
        "no-answer"
    )
    assert normalize_workflow_answer("Real-Life Relative:") == "no-answer"
    assert (
        normalize_workflow_answer(
            "Based on the information provided in the search results, there is insufficient "
            "evidence to identify the specific individual."
        )
        == "no-answer"
    )
    assert (
        normalize_workflow_answer(
            "The specific details were not found in the search results. A definitive answer "
            "cannot be derived from the provided documents."
        )
        == "no-answer"
    )


def test_normalize_workflow_answer_keeps_tool_benchmark_answers_intact() -> None:
    answer = (
        "To help you plan a memorable surprise birthday party, here are the suggested popular "
        "sites and main keywords.\n\n"
        "**Popular Sites for Inspiration:**\n"
        "* **Pinterest (pinterest.com)**\n"
        "* **Martha Stewart (marthastewart.com)**\n\n"
        "**Main Keywords:**\n"
        "* **Party**\n"
        "* **Ideas**\n"
        "* **Birthday**\n\n"
        "**Planning Tip:** combine the keywords with the friend's interests."
    )

    assert normalize_workflow_answer(answer, benchmark_name="stabletoolbench") == answer
    assert normalize_workflow_answer(answer, benchmark_name="workbench") == answer

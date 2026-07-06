from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from benchmark import get_benchmark
from descriptor.metrics import compute_run_metrics, compute_task_metrics
from MAS import OpenRouterLLMClient

from . import prompts
from .runtime import (
    OPERATOR_DESCRIPTIONS,
    OfficialAFlowRunnerAdapter,
    ensure_initial_workspace,
    load_prompt_module,
    load_workflow_class,
    parse_xml_fields,
    select_round,
    write_graph_file,
)


class OfficialAFlowBenchmarkOptimizer:
    """AFlow optimizer adapted from FoundationAgents/AFlow to this repo's benchmarks."""

    def __init__(
        self,
        *,
        benchmark_name: str,
        benchmark_config: dict[str, Any],
        llm_client: OpenRouterLLMClient,
        output_dir: Path,
        task_limit: int,
        validation_rounds: int,
        test_task_limit: int,
        test_offset: int,
        runs_per_task: int,
        retries: int,
        max_rounds: int,
        sample: int,
        seed: int,
        model_agent_type: str,
        temperature: float,
        allow_mock: bool,
        operators: list[str] | None = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.benchmark_config = dict(benchmark_config)
        self.llm_client = llm_client
        self.output_dir = output_dir
        self.task_limit = task_limit
        self.validation_rounds = validation_rounds
        self.test_task_limit = test_task_limit
        self.test_offset = test_offset
        self.runs_per_task = runs_per_task
        self.retries = retries
        self.max_rounds = max_rounds
        self.sample = sample
        self.seed = seed
        self.model_agent_type = model_agent_type
        self.temperature = temperature
        self.allow_mock = allow_mock
        self.operators = operators or [
            "Custom",
            "AnswerGenerate",
            "ScEnsemble",
            "Review",
            "Revise",
            "Format",
        ]
        self.workflows_dir = self.output_dir / "workflows"
        self.results_path = self.workflows_dir / "results.json"

    def optimize(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ensure_initial_workspace(self.workflows_dir, self.operators)
        results = self._load_results()

        if not any(int(item.get("round", 0)) == 1 for item in results):
            print(
                f"[{_now_stamp()}] AFLOW_OFFICIAL_EVAL_INITIAL benchmark={self.benchmark_name}",
                flush=True,
            )
            results.append(self._evaluate_round(1, initial=True, split="validation"))
            self._save_results(results)

        for current_round in range(1, max(1, self.max_rounds)):
            next_round = current_round + 1
            if any(int(item.get("round", 0)) == next_round for item in results):
                print(
                    f"[{_now_stamp()}] AFLOW_OFFICIAL_ROUND_RESUME benchmark={self.benchmark_name} "
                    f"round={next_round}",
                    flush=True,
                )
                continue
            selected = select_round(results, sample=self.sample)
            print(
                f"[{_now_stamp()}] AFLOW_OFFICIAL_OPTIMIZE benchmark={self.benchmark_name} "
                f"round={next_round} father={selected['round']} father_score={selected['score']}",
                flush=True,
            )
            response = self._propose_graph(selected)
            round_dir = self.workflows_dir / f"round_{next_round}"
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / "__init__.py").write_text("", encoding="utf-8")
            write_graph_file(round_dir / "graph.py", response["graph"])
            (round_dir / "prompt.py").write_text(
                response.get("prompt", "").strip() + "\n", encoding="utf-8"
            )
            experience = {
                "father node": int(selected["round"]),
                "modification": response.get("modification", ""),
                "before": float(selected["score"]),
                "after": None,
                "succeed": None,
            }
            try:
                row = self._evaluate_round(next_round, initial=False, split="validation")
            except Exception as exc:
                row = {
                    "round": next_round,
                    "score": 0.0,
                    "avg_cost": 0.0,
                    "total_cost": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                    "time": _now_stamp(),
                }
            experience["after"] = float(row.get("score", 0.0))
            experience["succeed"] = bool(float(row.get("score", 0.0)) > float(selected["score"]))
            (round_dir / "experience.json").write_text(
                json.dumps(experience, indent=2), encoding="utf-8"
            )
            results.append(row)
            self._save_results(results)

        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        best_round = int(best["round"])
        best_payload = self._materialize_best_round(best_round)
        test_payload = self._evaluate_best_on_test(best_round)
        payload = {
            "method": "aflow_official_adapter",
            "benchmark": self.benchmark_name,
            "best_round": best_round,
            "best_score": float(best.get("score", 0.0)),
            "test_score": test_payload["score"],
            "best": best,
            "results": results,
            "best_workflow": best_payload,
            "test": test_payload,
            "settings": {
                "task_limit": self.task_limit,
                "validation_rounds": self.validation_rounds,
                "test_task_limit": self.test_task_limit,
                "test_offset": self.test_offset,
                "runs_per_task": self.runs_per_task,
                "retries": self.retries,
                "max_rounds": self.max_rounds,
                "sample": self.sample,
                "seed": self.seed,
                "model_agent_type": self.model_agent_type,
                "temperature": self.temperature,
                "operators": self.operators,
            },
        }
        _write_json(self.output_dir / "aflow_results.json", payload)
        return payload

    def _evaluate_round(self, round_number: int, *, initial: bool, split: str) -> dict[str, Any]:
        round_dir = self.workflows_dir / f"round_{round_number}"
        workflow_class = load_workflow_class(round_dir)
        prompt_module = load_prompt_module(round_dir)
        benchmark = get_benchmark(self.benchmark_name, config=self.benchmark_config)
        all_tasks = list(benchmark.load_tasks(task_limit=self._load_task_limit()))
        eval_tasks = self._select_tasks(all_tasks, split=split)
        if not eval_tasks:
            raise RuntimeError(f"No tasks loaded for benchmark '{self.benchmark_name}'")
        artifact_dir = self._artifact_dir(round_number, split)
        task_payloads: list[dict[str, Any]] = []
        total_cost = 0.0
        for task_index, task in enumerate(eval_tasks):
            run_payloads: list[dict[str, Any]] = []
            run_metrics: list[dict[str, Any]] = []
            for run_index in range(max(1, self.runs_per_task)):
                run_payload = self._load_run_artifact(artifact_dir, str(task.task_id), run_index)
                if run_payload is not None:
                    run_payloads.append(run_payload)
                    run_metrics.append(dict(run_payload["metrics"]))
                    total_cost += float(
                        run_payload.get("metrics", {}).get("C3_cost_total", 0.0) or 0.0
                    )
                    print(
                        f"[{_now_stamp()}] AFLOW_OFFICIAL_RUN_RESUME benchmark={self.benchmark_name} "
                        f"split={split} round={round_number} task_id={task.task_id} run_index={run_index}",
                        flush=True,
                    )
                    continue
                run_seed = int(self.seed) + task_index * 1000 + run_index
                result, prediction, run_metadata, evaluation, metrics = self._run_with_retries(
                    benchmark=benchmark,
                    task=task,
                    artifact_dir=artifact_dir,
                    workflow_class=workflow_class,
                    prompt_module=prompt_module,
                    run_index=run_index,
                    run_seed=run_seed,
                )
                total_cost += float(metrics.get("C3_cost_total", 0.0) or 0.0)
                run_payload = {
                    "round": round_number,
                    "task_id": str(task.task_id),
                    "run_index": run_index,
                    "seed": run_seed,
                    "prediction": prediction,
                    "score": float(evaluation.score),
                    "success": bool(evaluation.success),
                    "metrics": metrics,
                    "trace": [event.to_dict() for event in result.trace_events],
                    "run_metadata": run_metadata,
                    "evaluation_details": evaluation.details,
                }
                self._write_run_artifact(artifact_dir, str(task.task_id), run_index, run_payload)
                self._mark_checkpoint_completed(artifact_dir, str(task.task_id), run_index)
                self._append_log(artifact_dir, run_payload)
                run_payloads.append(run_payload)
                run_metrics.append(metrics)
            task_metrics = compute_task_metrics(run_metrics)
            task_payloads.append(
                {
                    "task_id": str(task.task_id),
                    "score": float(task_metrics["eval_avg_score"]),
                    "success": bool(task_metrics["success_rate"] > 0.0),
                    "success_rate": float(task_metrics["success_rate"]),
                    "prediction": run_payloads[-1]["prediction"],
                    "metrics": task_metrics,
                    "runs": run_payloads,
                }
            )
        score = sum(float(task["score"]) for task in task_payloads) / len(task_payloads)
        round_summary = {
            "round": round_number,
            "score": score,
            "avg_cost": total_cost / max(1, len(task_payloads)),
            "total_cost": total_cost,
            "time": _now_stamp(),
            "initial": initial,
            "split": split,
            "task_count": len(task_payloads),
            "task_ids": [item["task_id"] for item in task_payloads],
        }
        _write_json(artifact_dir / "summary.json", {"round": round_summary, "tasks": task_payloads})
        self._write_summary_csv(artifact_dir / "summary.csv", task_payloads)
        print(
            f"[{_now_stamp()}] AFLOW_OFFICIAL_EVAL_DONE benchmark={self.benchmark_name} "
            f"split={split} round={round_number} score={score}",
            flush=True,
        )
        return round_summary

    def _run_with_retries(
        self,
        *,
        benchmark: Any,
        task: Any,
        artifact_dir: Path,
        workflow_class: Any,
        prompt_module: Any,
        run_index: int,
        run_seed: int,
    ) -> tuple[Any, str, dict[str, Any], Any, dict[str, Any]]:
        max_attempts = max(1, int(self.retries) + 1)
        last_exc: Exception | None = None
        for attempt_index in range(max_attempts):
            checkpoint_path = self._checkpoint_path(
                artifact_dir, str(task.task_id), run_index, attempt_index
            )
            runner = OfficialAFlowRunnerAdapter(
                workflow_class=workflow_class,
                prompt_module=prompt_module,
                llm_client=self.llm_client,
                benchmark_name=self.benchmark_name,
                agent_type=self.model_agent_type,
                temperature=self.temperature,
                allow_mock=self.allow_mock,
                checkpoint_path=checkpoint_path,
            )
            try:
                result = benchmark.run(
                    task=task, runner=runner, run_index=run_index, seed=run_seed + attempt_index
                )
                prediction = str(result.final_answer or "")
                run_metadata = {
                    **dict(result.run_metadata),
                    "attempt_index": attempt_index,
                    "attempts_used": attempt_index + 1,
                    "checkpoint_path": str(checkpoint_path.resolve()),
                }
                evaluation = benchmark.evaluate(task, prediction, run_metadata=run_metadata)
                metrics = compute_run_metrics(
                    list(result.trace_events),
                    evaluation=evaluation,
                    final_answer=prediction,
                    run_metadata=run_metadata,
                )
                return result, prediction, run_metadata, evaluation, metrics
            except Exception as exc:
                last_exc = exc
                self._write_attempt_error(
                    artifact_dir=artifact_dir,
                    task_id=str(task.task_id),
                    run_index=run_index,
                    attempt_index=attempt_index,
                    exc=exc,
                    checkpoint_path=checkpoint_path,
                )
                print(
                    f"[{_now_stamp()}] AFLOW_OFFICIAL_RUN_RETRY benchmark={self.benchmark_name} "
                    f"task_id={task.task_id} run_index={run_index} attempt={attempt_index + 1}/{max_attempts} "
                    f"error={type(exc).__name__}:{exc}",
                    flush=True,
                )
        assert last_exc is not None
        raise last_exc

    def _evaluate_best_on_test(self, best_round: int) -> dict[str, Any]:
        print(
            f"[{_now_stamp()}] AFLOW_OFFICIAL_TEST_START benchmark={self.benchmark_name} "
            f"round={best_round}",
            flush=True,
        )
        return self._evaluate_round(best_round, initial=False, split="test")

    def _propose_graph(self, selected: dict[str, Any]) -> dict[str, str]:
        father_round = int(selected["round"])
        round_dir = self.workflows_dir / f"round_{father_round}"
        graph = _extract_workflow_class((round_dir / "graph.py").read_text(encoding="utf-8"))
        prompt = (round_dir / "prompt.py").read_text(encoding="utf-8")
        experience = self._format_experience(father_round)
        operator_description = self._operator_description()
        log_data = self._load_log(father_round)
        optimize_prompt = (
            prompts.WORKFLOW_INPUT.format(
                experience=experience,
                score=float(selected["score"]),
                graph=graph,
                prompt=prompt,
                operator_description=operator_description,
                log=log_data,
            )
            + prompts.WORKFLOW_CUSTOM_USE
            + prompts.WORKFLOW_OPTIMIZE_PROMPT.format(type="tool-use benchmark")
            + prompts.XML_RESPONSE_INSTRUCTION
        )
        result = self._call_optimizer_with_retries(optimize_prompt, father_round)
        if result.mock_used and not self.allow_mock:
            raise RuntimeError(
                "Live OpenRouter AFlow optimizer expected, but mock fallback was used."
            )
        fields = parse_xml_fields(result.text, ["modification", "graph", "prompt"])
        if not fields.get("graph"):
            fields["graph"] = graph
            fields["modification"] = (
                "Fallback: optimizer response did not include graph; kept parent graph."
            )
        if not fields.get("prompt"):
            fields["prompt"] = prompt
        _write_json(
            self.output_dir
            / "optimizer_calls"
            / f"round_{father_round}_to_{father_round + 1}.json",
            {
                "father_round": father_round,
                "prompt": optimize_prompt,
                "response": result.text,
                "parsed": fields,
                "token_in": result.token_in,
                "token_out": result.token_out,
                "metadata": result.metadata,
            },
        )
        return fields

    def _call_optimizer_with_retries(self, optimize_prompt: str, father_round: int) -> Any:
        max_attempts = max(1, int(self.retries) + 1)
        last_exc: Exception | None = None
        for attempt_index in range(max_attempts):
            try:
                return self.llm_client.generate(
                    prompt=[{"role": "user", "content": optimize_prompt}],
                    agent_type=self.model_agent_type,
                    task_id=f"{self.benchmark_name}:aflow_optimizer",
                    run_index=father_round,
                    agent_id="aflow_optimizer",
                    tools=[],
                    max_tool_iterations=1,
                    temperature=self.temperature,
                )
            except Exception as exc:
                last_exc = exc
                _write_json(
                    self.output_dir
                    / "optimizer_calls"
                    / f"round_{father_round}.attempt_{attempt_index}.error.json",
                    {
                        "father_round": father_round,
                        "attempt_index": attempt_index,
                        "error": f"{type(exc).__name__}: {exc}",
                        "time": _now_stamp(),
                    },
                )
                print(
                    f"[{_now_stamp()}] AFLOW_OFFICIAL_OPTIMIZER_RETRY "
                    f"benchmark={self.benchmark_name} round={father_round + 1} "
                    f"attempt={attempt_index + 1}/{max_attempts} error={type(exc).__name__}:{exc}",
                    flush=True,
                )
        assert last_exc is not None
        raise last_exc

    def _materialize_best_round(self, best_round: int) -> dict[str, Any]:
        round_dir = self.workflows_dir / f"round_{best_round}"
        graph = (round_dir / "graph.py").read_text(encoding="utf-8")
        prompt = (round_dir / "prompt.py").read_text(encoding="utf-8")
        best_dir = self.output_dir / "best_workflow"
        best_dir.mkdir(parents=True, exist_ok=True)
        (best_dir / "graph.py").write_text(graph, encoding="utf-8")
        (best_dir / "prompt.py").write_text(prompt, encoding="utf-8")
        return {
            "round": best_round,
            "graph_path": str(best_dir / "graph.py"),
            "prompt_path": str(best_dir / "prompt.py"),
        }

    def _load_results(self) -> list[dict[str, Any]]:
        if not self.results_path.exists():
            return []
        try:
            data = json.loads(self.results_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save_results(self, results: list[dict[str, Any]]) -> None:
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    def _load_task_limit(self) -> int:
        return max(
            max(1, self.validation_rounds),
            max(0, self.test_offset) + max(1, self.test_task_limit),
            max(1, self.task_limit),
        )

    def _select_tasks(self, tasks: list[Any], *, split: str) -> list[Any]:
        if split == "validation":
            return tasks[: max(1, self.validation_rounds)]
        if split == "test":
            start = max(0, int(self.test_offset))
            end = start + max(1, int(self.test_task_limit))
            selected = tasks[start:end]
            if len(selected) < max(1, int(self.test_task_limit)):
                raise RuntimeError(
                    f"Not enough test tasks for benchmark '{self.benchmark_name}': "
                    f"requested offset={start} limit={self.test_task_limit}, loaded={len(tasks)}"
                )
            return selected
        raise ValueError(f"Unknown AFlow split: {split}")

    def _artifact_dir(self, round_number: int, split: str) -> Path:
        if split == "validation":
            return self.workflows_dir / f"round_{round_number}"
        return self.output_dir / "test" / f"round_{round_number}"

    def _run_dir(self, artifact_dir: Path, task_id: str) -> Path:
        return artifact_dir / "runs" / _safe(task_id)

    def _checkpoint_path(
        self, artifact_dir: Path, task_id: str, run_index: int, attempt_index: int
    ) -> Path:
        return (
            self._run_dir(artifact_dir, task_id)
            / "checkpoints"
            / f"run_{run_index}.attempt_{attempt_index}.json"
        )

    def _load_run_artifact(
        self, artifact_dir: Path, task_id: str, run_index: int
    ) -> dict[str, Any] | None:
        path = self._run_dir(artifact_dir, task_id) / f"run_{run_index}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(payload, dict) or "metrics" not in payload:
            return None
        return payload

    def _write_attempt_error(
        self,
        *,
        artifact_dir: Path,
        task_id: str,
        run_index: int,
        attempt_index: int,
        exc: Exception,
        checkpoint_path: Path,
    ) -> None:
        run_dir = self._run_dir(artifact_dir, task_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "error",
            "task_id": task_id,
            "run_index": run_index,
            "attempt_index": attempt_index,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "error": f"{type(exc).__name__}: {exc}",
            "time": _now_stamp(),
        }
        (run_dir / f"run_{run_index}.attempt_{attempt_index}.error.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def _mark_checkpoint_completed(self, artifact_dir: Path, task_id: str, run_index: int) -> None:
        run_dir = self._run_dir(artifact_dir, task_id)
        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return
        for checkpoint_path in sorted(checkpoint_dir.glob(f"run_{run_index}.attempt_*.json")):
            try:
                payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload["status"] = "completed"
            checkpoint_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _write_run_artifact(
        self, artifact_dir: Path, task_id: str, run_index: int, payload: dict[str, Any]
    ) -> None:
        run_dir = self._run_dir(artifact_dir, task_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"run_{run_index}.json"
        trace_path = run_dir / f"run_{run_index}.trace.json"
        payload["run_artifact_path"] = str(path.resolve())
        payload["trace_path"] = str(trace_path.resolve())
        trace_path.write_text(
            json.dumps({"trace": payload["trace"]}, indent=2, default=str), encoding="utf-8"
        )
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def _append_log(self, artifact_dir: Path, payload: dict[str, Any]) -> None:
        path = artifact_dir / "log.json"
        existing: list[Any] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []
        existing.append(
            {
                "task_id": payload["task_id"],
                "run_index": payload["run_index"],
                "prediction": payload["prediction"][:1000],
                "score": payload["score"],
                "success": payload["success"],
                "tool_calls_total": payload["run_metadata"].get("tool_calls_total", 0),
            }
        )
        path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")

    def _load_log(self, round_number: int) -> str:
        path = self.workflows_dir / f"round_{round_number}" / "log.json"
        if not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        rows = data if isinstance(data, list) else [data]
        return "\n\n".join(json.dumps(row, indent=2, ensure_ascii=False) for row in rows[:3])

    def _format_experience(self, father_round: int) -> str:
        entries = []
        for path in sorted(self.workflows_dir.glob("round_*/experience.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if int(data.get("father node", -1)) != father_round:
                continue
            verdict = "succeeded" if data.get("succeed") else "failed"
            entries.append(f"- {verdict}: {data.get('modification')} -> {data.get('after')}")
        return (
            "\n".join(entries) if entries else f"No experience data found for round {father_round}."
        )

    def _operator_description(self) -> str:
        lines = []
        for idx, name in enumerate(self.operators, start=1):
            item = OPERATOR_DESCRIPTIONS.get(name)
            if item:
                lines.append(
                    f"{idx}. {name}: {item['description']}, with interface {item['interface']}."
                )
        return "\n".join(lines)

    def _write_summary_csv(self, path: Path, task_payloads: list[dict[str, Any]]) -> None:
        import csv

        rows = []
        metric_keys = sorted({key for task in task_payloads for key in task["metrics"]})
        for task in task_payloads:
            row = {"task_id": task["task_id"], "score": task["score"], "success": task["success"]}
            row.update({key: task["metrics"].get(key) for key in metric_keys})
            rows.append(row)
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _extract_workflow_class(graph_text: str) -> str:
    match = re.search(r"class Workflow\(.*", graph_text, re.DOTALL)
    if match:
        return match.group(0)
    match = re.search(r"class Workflow:.*", graph_text, re.DOTALL)
    return match.group(0) if match else graph_text


def _safe(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(value).strip())
    return safe or "task"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str, sort_keys=True), encoding="utf-8")


def _now_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

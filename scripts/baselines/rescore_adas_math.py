#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.base import BenchmarkTask
from benchmark.math500 import Math500Benchmark
from reproduce.adas.runtime_runner import _parse_json_fields


def _raw_final_text(row: dict[str, Any]) -> str:
    acts = [event for event in row.get("trace", []) if event.get("event_type") == "act"]
    return str((acts[-1].get("payload") or {}).get("text") or "") if acts else ""


def rescore(payload: dict[str, Any]) -> dict[str, Any]:
    benchmark = Math500Benchmark()
    corrected = deepcopy(payload)
    changes: list[dict[str, Any]] = []
    for row in corrected.get("runs", []):
        details = dict(row.get("evaluation_details") or {})
        raw_text = _raw_final_text(row)
        parsed = _parse_json_fields(raw_text, ["thinking", "answer"])
        prediction = str(parsed.get("answer") or raw_text).strip()
        task = BenchmarkTask(
            task_id=str(row.get("task_id") or ""),
            prompt="",
            reference_answer=str(details.get("reference_answer") or ""),
            metadata={"subject": details.get("subject", ""), "level": details.get("level")},
        )
        evaluation = benchmark.evaluate(task, prediction)
        old_success = bool(row.get("success"))
        row["prediction"] = prediction
        row["score"] = evaluation.score
        row["success"] = evaluation.success
        row["evaluation_details"] = evaluation.details
        row["rescore_provenance"] = {
            "source": "last act payload.text",
            "parser": "reproduce.adas.runtime_runner._parse_json_fields",
        }
        if old_success != evaluation.success:
            changes.append(
                {
                    "task_id": task.task_id,
                    "run_index": row.get("run_index"),
                    "old_success": old_success,
                    "new_success": evaluation.success,
                }
            )

    runs = corrected.get("runs", [])
    run_indices = sorted({int(row.get("run_index", 0)) for row in runs})
    rates = [
        100.0
        * statistics.fmean(bool(row.get("success")) for row in runs if row.get("run_index") == idx)
        for idx in run_indices
    ]
    corrected["rescore_summary"] = {
        "run_success_rates_percent": rates,
        "mean_percent": statistics.fmean(rates) if rates else 0.0,
        "population_std_percent": statistics.pstdev(rates) if len(rates) > 1 else 0.0,
        "changed_outcomes": changes,
        "changed_outcome_count": len(changes),
    }
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore ADAS MATH from immutable raw LLM traces.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    corrected = rescore(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corrected, indent=2, default=str), encoding="utf-8")
    print(json.dumps(corrected["rescore_summary"], indent=2))


if __name__ == "__main__":
    main()

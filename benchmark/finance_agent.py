from __future__ import annotations

import ast
import csv
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .base import BenchmarkEvaluation, BenchmarkTask

PUBLIC_CSV_URL = (
    "https://raw.githubusercontent.com/vals-ai/finance-agent/"
    "aad00743ce54b348678a2073aac51fba825ca901/data/public.csv"
)


class FinanceAgentBenchmark:
    """Lightweight adapter for the Vals FinanceAgent public benchmark CSV."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.dataset_url = str(cfg.get("dataset_url", PUBLIC_CSV_URL))
        self.cache_dir = Path(str(cfg.get("cache_dir", ".cache/finance_agent")))
        self.local_csv_path = cfg.get("local_csv_path")
        self.success_threshold = float(cfg.get("success_threshold", 0.5))

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        csv_path = self._resolve_csv_path()
        tasks: List[BenchmarkTask] = []

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                task_id = str(idx)
                rubric = self._parse_rubric(row.get("Rubric", ""))
                metadata = {
                    "question_type": row.get("Question Type", ""),
                    "expert_time_mins": self._safe_float(row.get("Expert time (mins)", "")),
                    "rubric": rubric,
                    "source": "finance_agent_public_csv",
                }
                tasks.append(
                    BenchmarkTask(
                        task_id=task_id,
                        prompt=(row.get("Question") or "").strip(),
                        reference_answer=(row.get("Answer") or "").strip(),
                        metadata=metadata,
                    )
                )
                if task_limit is not None and len(tasks) >= task_limit:
                    break

        return tasks

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: Dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        run_metadata = run_metadata or {}
        pred_norm = self._normalize_text(prediction)
        rubric = list(task.metadata.get("rubric", []))

        correctness_total = 0
        contradiction_total = 0
        correctness_hits = 0
        contradiction_hits = 0

        criterion_results: List[Dict[str, Any]] = []
        for criterion in rubric:
            operator = str(criterion.get("operator", "")).strip().lower()
            text = str(criterion.get("criteria", "")).strip()
            if not operator or not text:
                continue

            text_norm = self._normalize_text(text)
            matched = bool(text_norm) and text_norm in pred_norm

            if operator == "correctness":
                correctness_total += 1
                correctness_hits += int(matched)
            elif operator == "contradiction":
                contradiction_total += 1
                contradiction_hits += int(matched)

            criterion_results.append(
                {
                    "operator": operator,
                    "criteria": text,
                    "matched": matched,
                }
            )

        correctness_ratio = (
            correctness_hits / correctness_total if correctness_total > 0 else 0.0
        )
        contradiction_ratio = (
            contradiction_hits / contradiction_total if contradiction_total > 0 else 0.0
        )

        score = max(0.0, min(1.0, correctness_ratio - contradiction_ratio))
        success = score >= self.success_threshold

        details: Dict[str, Any] = {
            "correctness_hits": correctness_hits,
            "correctness_total": correctness_total,
            "contradiction_hits": contradiction_hits,
            "contradiction_total": contradiction_total,
            "correctness_ratio": correctness_ratio,
            "contradiction_ratio": contradiction_ratio,
            "success_threshold": self.success_threshold,
            "criterion_results": criterion_results,
            "prediction": prediction,
            "reference_answer": task.reference_answer,
            "task_metadata": task.metadata,
            "run_metadata": run_metadata,
        }

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=score,
            success=success,
            details=details,
        )

    def requirements(self) -> Dict[str, Any]:
        return {
            "benchmark": "finance_agent",
            "dataset_source": self.dataset_url,
            "notes": [
                "This adapter provides dataset loading and rubric-proxy scoring.",
                "It does not replicate the full Vals tool harness or leaderboard evaluation.",
            ],
        }

    def _resolve_csv_path(self) -> Path:
        if self.local_csv_path:
            local = Path(str(self.local_csv_path)).expanduser().resolve()
            if not local.exists():
                raise FileNotFoundError(f"FinanceAgent CSV not found: {local}")
            return local

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self.cache_dir / "public.csv"
        if cached.exists():
            return cached

        with urllib.request.urlopen(self.dataset_url, timeout=30) as response:
            data = response.read()
        cached.write_bytes(data)
        return cached

    @staticmethod
    def _parse_rubric(value: str) -> List[Dict[str, Any]]:
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                out: List[Dict[str, Any]] = []
                for item in parsed:
                    if isinstance(item, dict):
                        out.append(dict(item))
                return out
        except (SyntaxError, ValueError):
            return []
        return []

    @staticmethod
    def _normalize_text(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text.strip().lower())
        return collapsed

    @staticmethod
    def _safe_float(value: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

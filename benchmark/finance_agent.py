"""Vals Finance Agent Benchmark adapter.

Data source: https://github.com/vals-ai/finance-agent
Leaderboard: https://www.vals.ai/benchmarks/finance_agent

CSV schema (public.csv):
    Question, Answer, Question Type, Expert time (mins), Rubric

Rubric is a JSON list of criteria dicts, each with:
    {"operator": "correctness"|"contradiction", "criteria": "<text>"}

Evaluation modes:
    1. llm_judge (default)  - Uses an LLM to judge each criterion against the
       prediction. Mirrors the official Vals evaluation: mode-of-3 LLM calls
       per criterion.
    2. substring            - Simple normalised substring matching (fast, no
       API cost, lower accuracy). Useful for debugging and quick iteration.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import re
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from statistics import mode
from typing import TYPE_CHECKING, Any

from .base import BenchmarkEvaluation, BenchmarkTask

if TYPE_CHECKING:
    from MAS.runner import MASRunResult

logger = logging.getLogger(__name__)

# Pinned commit so results are reproducible even if upstream main changes.
PUBLIC_CSV_URL = (
    "https://raw.githubusercontent.com/vals-ai/finance-agent/"
    "aad00743ce54b348678a2073aac51fba825ca901/data/public.csv"
)

# ---------------------------------------------------------------------------
# LLM-as-Judge prompt template
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert financial analyst acting as a judge. Your job is to
determine whether a PREDICTION satisfies a specific evaluation CRITERION
with respect to a REFERENCE ANSWER.

Rules:
- Focus on factual accuracy. Minor formatting or phrasing differences are OK.
- For numerical values: allow up to 2% relative tolerance unless the
  criterion specifies exact figures.  "$3.25 Billion" matches "$3.25B",
  "3,250,000,000", "3.25 billion", etc.
- For percentages: "22.6%" matches "22.60%", "~23%" does NOT match "22.6%".
- Rounding: if the prediction rounds an intermediate number but arrives at
  the same final answer, that is acceptable.
- If the criterion operator is "contradiction", you are checking whether
  the prediction CONTRADICTS the reference.  Return "yes" if there IS a
  contradiction, "no" if there is no contradiction.

Respond with ONLY a JSON object: {"judgment": "yes" | "no"}
"yes" = the criterion is satisfied (correctness met, or contradiction found).
"no"  = the criterion is NOT satisfied.
"""

_JUDGE_USER_TEMPLATE = """\
REFERENCE ANSWER:
{reference_answer}

PREDICTION:
{prediction}

CRITERION ({operator}):
{criteria}
"""


class FinanceAgentBenchmark:
    """Adapter for the Vals Finance Agent public benchmark CSV.

    Config keys (all optional):
        dataset_url         URL to the public CSV (default: pinned commit).
        cache_dir           Local cache directory (default: .cache/finance_agent).
        local_csv_path      Path to a pre-downloaded CSV (overrides download).
        success_threshold   Score >= this is considered success (default: 0.5).
        eval_mode           "llm_judge" or "substring" (default: "llm_judge").
        judge_model         Model name for LLM judge (default: "openai/gpt-4o").
        judge_repeats       Number of repeated judge calls; final answer is
                            the mode (default: 3, matching Vals methodology).
        judge_temperature   Temperature for judge calls (default: 0.0).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.dataset_url = str(cfg.get("dataset_url", PUBLIC_CSV_URL))
        self.cache_dir = Path(str(cfg.get("cache_dir", ".cache/finance_agent")))
        self.local_csv_path = cfg.get("local_csv_path")
        self.success_threshold = float(cfg.get("success_threshold", 0.5))

        # Evaluation settings
        self.eval_mode: str = str(cfg.get("eval_mode", "llm_judge"))
        self.judge_model: str = str(cfg.get("judge_model", "openai/gpt-4o"))
        self.judge_repeats: int = int(cfg.get("judge_repeats", 3))
        self.judge_temperature: float = float(cfg.get("judge_temperature", 0.0))

        # Lazily initialised LLM client (only if eval_mode == "llm_judge")
        self._llm_client: Any | None = None
        self._llm_config: dict[str, Any] = dict(cfg.get("openrouter", {}))

    # ------------------------------------------------------------------
    # BenchmarkAdapter interface
    # ------------------------------------------------------------------

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        csv_path = self._resolve_csv_path()
        tasks: list[BenchmarkTask] = []

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

    def run(
        self,
        task: BenchmarkTask,
        runner: Any,
        run_index: int,
        seed: int,
    ) -> MASRunResult:
        """One-shot benchmark: delegate entirely to the runner."""
        return runner.run_task(task=task, run_index=run_index, seed=seed)

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        run_metadata = run_metadata or {}
        rubric = list(task.metadata.get("rubric", []))

        if self.eval_mode == "llm_judge":
            return self._evaluate_llm_judge(task, prediction, rubric, run_metadata)
        return self._evaluate_substring(task, prediction, rubric, run_metadata)

    def requirements(self) -> dict[str, Any]:
        notes = [
            "Data: public CSV from vals-ai/finance-agent (537 questions).",
            f"Eval mode: {self.eval_mode}.",
        ]
        if self.eval_mode == "llm_judge":
            notes.append(
                f"LLM judge: {self.judge_model}, mode-of-{self.judge_repeats}. "
                "Requires 'openai' package and a valid API key / OpenRouter key."
            )
        else:
            notes.append(
                "Substring mode: fast but lower accuracy. "
                "Set eval_mode='llm_judge' for higher-fidelity evaluation."
            )
        return {
            "benchmark": "finance_agent",
            "version": "1.1",
            "dataset_source": self.dataset_url,
            "question_types": [
                "Simple retrieval - Quantitative",
                "Simple retrieval - Qualitative",
                "Numerical Reasoning",
                "Complex Retrieval",
                "Adjustments",
                "Beat or Miss",
                "Trends",
                "Financial Modeling",
                "Market Analysis",
            ],
            "notes": notes,
        }

    # ------------------------------------------------------------------
    # Evaluation: LLM-as-Judge (mirrors Vals methodology)
    # ------------------------------------------------------------------

    def _evaluate_llm_judge(
        self,
        task: BenchmarkTask,
        prediction: str,
        rubric: list[dict[str, Any]],
        run_metadata: dict[str, Any],
    ) -> BenchmarkEvaluation:
        correctness_total = 0
        contradiction_total = 0
        correctness_hits = 0
        contradiction_hits = 0

        criterion_results: list[dict[str, Any]] = []

        for criterion in rubric:
            operator = str(criterion.get("operator", "")).strip().lower()
            text = str(criterion.get("criteria", "")).strip()
            if not operator or not text:
                continue

            matched = self._judge_criterion(
                prediction=prediction,
                reference_answer=task.reference_answer,
                operator=operator,
                criteria=text,
            )

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
                    "eval_mode": "llm_judge",
                }
            )

        correctness_ratio = correctness_hits / correctness_total if correctness_total > 0 else 0.0
        contradiction_ratio = (
            contradiction_hits / contradiction_total if contradiction_total > 0 else 0.0
        )

        score = max(0.0, min(1.0, correctness_ratio - contradiction_ratio))
        success = score >= self.success_threshold

        details: dict[str, Any] = {
            "eval_mode": "llm_judge",
            "judge_model": self.judge_model,
            "judge_repeats": self.judge_repeats,
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
            "question_type": task.metadata.get("question_type", ""),
            "run_metadata": run_metadata,
        }

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=score,
            success=success,
            details=details,
        )

    def _judge_criterion(
        self,
        *,
        prediction: str,
        reference_answer: str,
        operator: str,
        criteria: str,
    ) -> bool:
        """Call the LLM judge `judge_repeats` times and return mode result."""
        client = self._get_llm_client()

        user_msg = _JUDGE_USER_TEMPLATE.format(
            reference_answer=reference_answer,
            prediction=prediction,
            operator=operator,
            criteria=criteria,
        )

        votes: list[bool] = []
        for _ in range(self.judge_repeats):
            try:
                response = client.chat.completions.create(
                    model=self.judge_model,
                    temperature=self.judge_temperature,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                judgment = json.loads(content).get("judgment", "no")
                votes.append(judgment.strip().lower() == "yes")
            except Exception:
                logger.warning(
                    "LLM judge call failed for criterion: %s",
                    criteria[:80],
                    exc_info=True,
                )
                votes.append(False)

        # Mode-of-N: majority vote (matches Vals methodology)
        try:
            return mode(votes)
        except Exception:
            # If no clear mode (shouldn't happen with odd N), default False
            return sum(votes) > len(votes) / 2

    def _get_llm_client(self) -> Any:
        """Lazily create an OpenAI-compatible client."""
        if self._llm_client is not None:
            return self._llm_client

        import openai  # type: ignore

        kwargs: dict[str, Any] = {}

        # Support OpenRouter if configured
        base_url = self._llm_config.get("base_url")
        api_key = self._llm_config.get("api_key")

        if base_url:
            kwargs["base_url"] = str(base_url)
        if api_key:
            kwargs["api_key"] = str(api_key)

        self._llm_client = openai.OpenAI(**kwargs)
        return self._llm_client

    # ------------------------------------------------------------------
    # Evaluation: Substring fallback (fast, no API cost)
    # ------------------------------------------------------------------

    def _evaluate_substring(
        self,
        task: BenchmarkTask,
        prediction: str,
        rubric: list[dict[str, Any]],
        run_metadata: dict[str, Any],
    ) -> BenchmarkEvaluation:
        pred_norm = self._normalize_text(prediction)

        correctness_total = 0
        contradiction_total = 0
        correctness_hits = 0
        contradiction_hits = 0

        criterion_results: list[dict[str, Any]] = []
        for criterion in rubric:
            operator = str(criterion.get("operator", "")).strip().lower()
            text = str(criterion.get("criteria", "")).strip()
            if not operator or not text:
                continue

            matched = self._substring_match(pred_norm, text)

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
                    "eval_mode": "substring",
                }
            )

        correctness_ratio = correctness_hits / correctness_total if correctness_total > 0 else 0.0
        contradiction_ratio = (
            contradiction_hits / contradiction_total if contradiction_total > 0 else 0.0
        )

        score = max(0.0, min(1.0, correctness_ratio - contradiction_ratio))
        success = score >= self.success_threshold

        details: dict[str, Any] = {
            "eval_mode": "substring",
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
            "question_type": task.metadata.get("question_type", ""),
            "run_metadata": run_metadata,
        }

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=score,
            success=success,
            details=details,
        )

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------

    @classmethod
    def _substring_match(cls, pred_norm: str, criteria_text: str) -> bool:
        """Check if normalised criteria appears in normalised prediction.

        Handles common financial formatting variants:
          - "$3.25 Billion" ↔ "$3.25B" ↔ "3.25 billion"
          - "22.6%" ↔ "22.60%"
          - "2,865,507" ↔ "2865507"
        """
        criteria_norm = cls._normalize_text(criteria_text)

        # Direct substring
        if criteria_norm and criteria_norm in pred_norm:
            return True

        # Try numeric extraction: if criteria is a single number, check
        # if it appears in prediction within 2% tolerance.
        criteria_numbers = cls._extract_numbers(criteria_text)
        if criteria_numbers:
            pred_numbers = cls._extract_numbers(pred_norm)
            for ref_num in criteria_numbers:
                for pred_num in pred_numbers:
                    if cls._is_close(pred_num, ref_num, rel_tol=0.02):
                        return True

        return False

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        """Extract numbers from text, handling commas and common suffixes."""
        # Remove commas in numbers (e.g. "2,865,507" -> "2865507")
        cleaned = re.sub(r"(\d),(\d)", r"\1\2", text)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        out: list[float] = []
        for value in numbers:
            try:
                out.append(float(value))
            except ValueError:
                continue
        return out

    @staticmethod
    def _is_close(a: float, b: float, *, rel_tol: float = 0.02) -> bool:
        """Check if two numbers are within relative tolerance."""
        if b == 0:
            return abs(a) < 1e-9
        return abs(a - b) / abs(b) <= rel_tol

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

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

        logger.info("Downloading FinanceAgent CSV from %s", self.dataset_url)
        with urllib.request.urlopen(self.dataset_url, timeout=30) as response:
            data = response.read()
        cached.write_bytes(data)
        return cached

    @staticmethod
    def _parse_rubric(value: str) -> list[dict[str, Any]]:
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                out: list[dict[str, Any]] = []
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

"""MATH-500 benchmark adapter.

Dataset : HuggingFaceH4/MATH-500 (500 problems sampled from the MATH test set,
          the subset used in OpenAI's "Let's Verify Step by Step").
Columns : problem, solution, answer, subject, level, unique_id.

MATH-500 is a pure reasoning benchmark: one-shot generation, no tools.
The agent is asked to reason step by step and put the final answer in
``\\boxed{...}``. Evaluation extracts the last boxed expression from the
prediction (with fallbacks for "the answer is ..." phrasing) and compares it
to the reference answer using the canonical Hendrycks MATH normalization
(``strip_string`` / ``is_equiv``), plus a numeric fallback.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.base import BenchmarkEvaluation, BenchmarkTask

if TYPE_CHECKING:
    from MAS.runner import MASRunResult

QUERY_TEMPLATE = """\
Solve the following math problem step by step.

{problem}

Think carefully and show your reasoning. Then give your final answer on the \
last line in the form: \\boxed{{<answer>}}."""


# ---------------------------------------------------------------------------
# Answer extraction (mirrors hendrycks/math `last_boxed_only_string`)
# ---------------------------------------------------------------------------


def _last_boxed_only_string(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return text[idx : right_brace_idx + 1]


def _remove_boxed(boxed: str) -> str:
    if boxed.startswith("\\boxed "):
        return boxed[len("\\boxed ") :]
    for left in ("\\boxed{", "\\fbox{"):
        if boxed.startswith(left) and boxed.endswith("}"):
            return boxed[len(left) : -1]
    return boxed


_ANSWER_PHRASE_RE = re.compile(
    r"(?:final answer is|answer is|answer:)\s*(.+?)\s*(?:\.\s*)?$",
    re.IGNORECASE,
)


def extract_answer(prediction: str) -> tuple[str, str]:
    """Extract the final answer from a model response.

    Returns (answer, match_type) where match_type is "boxed", "phrase",
    "last_line", or "empty".
    """
    boxed = _last_boxed_only_string(prediction)
    if boxed is not None:
        return _remove_boxed(boxed).strip(), "boxed"

    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    for line in reversed(lines):
        match = _ANSWER_PHRASE_RE.search(line)
        if match:
            return match.group(1).strip().strip("$").rstrip("."), "phrase"

    if lines:
        return lines[-1].strip("$").rstrip("."), "last_line"
    return "", "empty"


# ---------------------------------------------------------------------------
# Answer equivalence (verbatim port of hendrycks/math `strip_string`/`is_equiv`)
# ---------------------------------------------------------------------------


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a)
        b_int = int(b)
        if string == f"{a_int}/{b_int}":
            return f"\\frac{{{a_int}}}{{{b_int}}}"
    except ValueError:
        return string
    return string


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs when describing units
    if "\\text{ " in string:
        return string.split("\\text{ ")[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


# Text-formatting wrappers that wrap a whole answer (e.g. word/categorical
# answers like `\text{Evelyn}` or `\textbf{yes}`), keeping their inner content.
# Applied after unit removal so the unit form `5\text{ cm}` is handled first.
_TEXT_WRAPPER_RE = re.compile(
    r"\\(?:text|textbf|textrm|textit|textsf|mbox|mathrm|mathbf|mathsf)\{([^{}]*)\}"
)


def _remove_text_wrappers(string: str) -> str:
    prev = None
    while prev != string:
        prev = string
        string = _TEXT_WRAPPER_RE.sub(r"\1", string)
    return string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = _remove_text_wrappers(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _to_float(string: str) -> float | None:
    try:
        return float(string.replace(",", ""))
    except ValueError:
        return None


def is_equiv(prediction: str, reference: str) -> bool:
    if prediction is None or reference is None:
        return False
    try:
        pred = _strip_string(prediction)
        ref = _strip_string(reference)
        if pred == ref:
            return True
        pred_num = _to_float(pred)
        ref_num = _to_float(ref)
        if pred_num is not None and ref_num is not None:
            return abs(pred_num - ref_num) < 1e-6
        return False
    except Exception:
        return prediction.strip() == reference.strip()


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class Math500Benchmark:
    """MATH-500 adapter for MAS_Analyzer.

    Config keys (all optional):
        split          Dataset split (default: "test").
        dataset_name   HuggingFace dataset name (default: "HuggingFaceH4/MATH-500").
        dataset_path   Local JSONL path; takes precedence over the HF dataset
                       (columns: problem, answer, and optionally solution,
                       subject, level, unique_id).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.split: str = str(cfg.get("split", "test"))
        self.dataset_name: str = str(cfg.get("dataset_name", "HuggingFaceH4/MATH-500"))
        self.dataset_path: str | None = (
            str(cfg.get("dataset_path")) if cfg.get("dataset_path") else None
        )

    # ---- load_tasks --------------------------------------------------------

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        tasks: list[BenchmarkTask] = []
        for idx, row in enumerate(self._iter_rows()):
            problem = str(row.get("problem", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not problem or not answer:
                continue
            raw_id = str(row.get("unique_id") or idx)
            task_id = raw_id.replace("/", "_").removesuffix(".json")
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    prompt=QUERY_TEMPLATE.format(problem=problem),
                    reference_answer=answer,
                    metadata={
                        "source": "math500",
                        "subject": str(row.get("subject", "")),
                        "level": row.get("level"),
                        "solution": str(row.get("solution", "")),
                    },
                )
            )
            if task_limit is not None and len(tasks) >= task_limit:
                break
        return tasks

    def _iter_rows(self):
        if self.dataset_path:
            path = Path(self.dataset_path)
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return

        from datasets import load_dataset

        yield from load_dataset(self.dataset_name, split=self.split)

    # ---- run ---------------------------------------------------------------

    def run(
        self,
        task: BenchmarkTask,
        runner: Any,
        run_index: int,
        seed: int,
    ) -> MASRunResult:
        """One-shot reasoning benchmark: no tools, single generation."""
        return runner.run_task(
            task=task,
            run_index=run_index,
            seed=seed,
            benchmark_name="math500",
        )

    # ---- evaluate ----------------------------------------------------------

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        extracted, match_type = extract_answer(prediction or "")
        success = bool(extracted) and is_equiv(extracted, task.reference_answer)
        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=1.0 if success else 0.0,
            success=success,
            details={
                "prediction": (prediction or "")[:500],
                "extracted_answer": extracted,
                "match_type": match_type,
                "reference_answer": task.reference_answer,
                "subject": task.metadata.get("subject", ""),
                "level": task.metadata.get("level"),
            },
        )

    def requirements(self) -> dict[str, Any]:
        return {
            "benchmark": "math500",
            "version": "1.0",
            "dataset_source": f"https://huggingface.co/datasets/{self.dataset_name}",
            "metrics": [
                "accuracy (boxed-answer exact match after Hendrycks MATH normalization)",
            ],
            "notes": [
                "Pure reasoning benchmark: one-shot, no tools.",
                "Prediction answer extracted from the last \\boxed{...} "
                "(fallback: 'the answer is ...' phrasing, then last line).",
                "Requires 'datasets' unless a local dataset_path JSONL is configured.",
            ],
        }

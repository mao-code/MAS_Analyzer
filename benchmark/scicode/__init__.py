"""SciCode benchmark adapter.

Faithfully replicates the official SciCode evaluation pipeline
(https://github.com/scicode-bench/SciCode) so that prompts, code assembly,
and test execution are identical to the upstream repository.
"""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from benchmark.base import (
    BenchmarkEvaluation,
    BenchmarkTask,
    init_run_metadata_aggregate,
    merge_step_run_metadata,
)
from MAS import MASRunner, MASRunResult

with contextlib.suppress(ImportError):
    from datasets import load_dataset

with contextlib.suppress(ImportError):
    from scicode.parse.parse import extract_function_name, get_function_from_code

# ---------------------------------------------------------------------------
# Prompt templates (verbatim copies from the official repo)
# ---------------------------------------------------------------------------

MULTISTEP_TEMPLATE = """\
PROBLEM DESCRIPTION:
You will be provided with problem steps along with background knowledge necessary for solving the problem. Your task will be to develop a Python solution focused on the next step of the problem-solving process.

PROBLEM STEPS AND FUNCTION CODE:
Here, you'll find the Python code for the initial steps of the problem-solving process. This code is integral to building the solution.

{problem_steps_str}

NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. A function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.

{dependencies}

RESPONSE GUIDELINES:
Now, based on the instructions and information provided above, write the complete and executable Python program for the next step in a single block.
Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
Your response should NOT include the dependencies and functions of all previous steps. If your next step function calls functions from previous steps, please make sure it uses the headers provided without modification.
DO NOT generate EXAMPLE USAGE OR TEST CODE in your response. Please make sure your response python code in format of ```python```.
"""

BACKGROUND_COMMENT_TEMPLATE = """\
PROBLEM DESCRIPTION:
You will be provided with the main description of the problem, previous steps, and the next step. Your task will be to generate the disciplinary knowledge necessary for solving the next step and then develop a Python solution focused on this step.

PREVIOUS STEPS DESCRIPTION:
{problem_steps_str}

NEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:
This part will describe the next step in the problem-solving process. First, provide the necessary scientific background knowledge as a comment at the beginning of your response, starting with 'Background: '. Then, a function header will be provided, and your task is to develop the Python code for this next step based on the provided description and function header.

{next_step_str}

DEPENDENCIES:
Use only the following dependencies in your solution. Do not include these dependencies at the beginning of your code.
{dependencies}

RESPONSE GUIDELINES:
1. Start with the scientific background required for the next step, formatted as a comment.
2. Then write the complete and executable Python program for the next step in a single block.
3. Your response should focus exclusively on implementing the solution for the next step, adhering closely to the specified function header and the context provided by the initial steps.
4. DO NOT include previous function code, example usage or test code in your response.
5. Ensure your response is in the format of ```python``` and includes the necessary background as a comment at the top.
6. Put a newline immediately after ```python.
7. Put the background comment on its own line, starting with exactly `# Background:`.
8. Put the function definition on a new line after the background comment. Never place `def`, `async def`, or `class` on the same line as the background comment.

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```
"""

# Hardcoded special steps that the official repo skips (they ship ground-truth
# .txt files for these instead of generating them).
SKIPPED_STEPS: set[tuple[str, int]] = {
    ("13", 5),
    ("62", 0),
    ("76", 2),
}


def _extract_python_script(text: str) -> str:
    """Extract python code from a markdown-formatted LLM response.

    Mirrors the official ``extract_python_script`` in
    ``src/scicode/gen/models.py`` – including the ``re.sub`` that strips
    import statements so that they don't clash with the ``dependencies``
    block that is prepended separately.
    """
    if "```" in text:
        if "```python" in text:
            python_script = text.split("```python")[1].split("```")[0]
        else:
            python_script = text.split("```")[1].split("```")[0]
    else:
        python_script = text

    # Official repo strips all import lines from the generated code because
    # the required dependencies are prepended separately.
    python_script = re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)",
        "",
        python_script,
        flags=re.MULTILINE,
    )
    return python_script


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------


class SciCodeBenchmark:
    """Benchmark adapter for SciCode.

    Lifecycle (mirrors the official repo):
      load_tasks  → one BenchmarkTask per *main problem*
      run         → loops over sub-steps, calling the LLM once per step
      evaluate    → assembles full .py files and runs assert-based tests
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.split: str = str(cfg.get("split", "test"))
        self.with_background: bool = bool(cfg.get("with_background", False))
        self.h5py_file: str = str(cfg.get("h5py_file", "data/test_data.h5"))
        self.execution_timeout_s: int = max(1, int(cfg.get("execution_timeout_s", 1800)))
        self.llm_repair: bool = bool(cfg.get("llm_repair", False))
        self.llm_repair_model: str | None = (
            str(cfg.get("llm_repair_model")).strip() if cfg.get("llm_repair_model") else None
        )
        # Match upstream gencode.py mapping:
        #   with_background=True  -> multistep template
        #   with_background=False -> background-comment template
        self.template = MULTISTEP_TEMPLATE if self.with_background else BACKGROUND_COMMENT_TEMPLATE
        self._ensure_data_exists()

    def _ensure_data_exists(self) -> None:
        """Download the numeric test results from Hugging Face if missing.

        Official source is Google Drive (manual), but a stable mirror exists
        on Hugging Face for programmatic access.
        """
        data_path = Path(self.h5py_file)
        if data_path.exists():
            return

        logger.info(
            f"SciCode evaluation data not found at {data_path}. "
            "Attempting to download from Hugging Face mirror..."
        )
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Mirror hosted on Hugging Face (akshathmangudi/SciCode)
        url = "https://huggingface.co/datasets/akshathmangudi/SciCode/resolve/main/raw/raw_ground.h5?download=true"

        try:
            # -L follows redirects, -o writes to file
            subprocess.run(["curl", "-L", url, "-o", str(data_path)], check=True)
            logger.info(f"Successfully downloaded SciCode data to {data_path}")
        except Exception as e:
            logger.error(
                f"Failed to download SciCode data: {e}\n"
                "Please manually download 'test_data.h5' and place it at: "
                f"{data_path.resolve()}\n"
                "Manual download link: https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link"
            )

    # ---- load_tasks --------------------------------------------------------

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        dataset = load_dataset("SciCode1/SciCode", split=self.split)

        tasks: list[BenchmarkTask] = []
        for row in dataset:
            tasks.append(
                BenchmarkTask(
                    task_id=str(row["problem_id"]),
                    prompt="",  # built dynamically per sub-step inside run()
                    reference_answer="",
                    metadata={
                        "sub_steps": row["sub_steps"],
                        "required_dependencies": row.get("required_dependencies", ""),
                    },
                )
            )
            if task_limit is not None and len(tasks) >= task_limit:
                break

        return tasks

    # ---- run ---------------------------------------------------------------

    def run(
        self,
        task: BenchmarkTask,
        runner: MASRunner,
        run_index: int,
        seed: int,
    ) -> MASRunResult:
        all_events: list = []
        aggregate_metadata = init_run_metadata_aggregate()
        sub_steps = task.metadata["sub_steps"]
        dependencies: str = task.metadata["required_dependencies"]
        prob_id = task.task_id
        tot_steps = len(sub_steps)

        # Mirrors official `self.previous_llm_code`:
        # stores the *extracted function code* for each step (used in prompts).
        previous_llm_code: list[str | None] = [None] * tot_steps

        # Stores the *full runnable file* for each step (deps + all prev code + this step code).
        # This is what gets tested.  Mirrors official `save_response_with_steps`.
        full_code_per_step: list[str] = [""] * tot_steps
        raw_responses_per_step: list[str] = [""] * tot_steps
        repaired_responses_per_step: list[str] = [""] * tot_steps

        for idx in range(tot_steps):
            num_steps = idx + 1  # 1-based

            # Skip hardcoded special steps (official repo ships ground-truth for these)
            if (prob_id, idx) in SKIPPED_STEPS:
                continue

            # -- Build prompt (identical to official `generate_prompt_with_steps`) --
            output_lines: list[str] = []
            previous_code_parts: list[str] = []

            for i in range(num_steps - 1):
                step_desc = sub_steps[i]["step_description_prompt"]
                if self.with_background:
                    step_desc = step_desc + "\n" + sub_steps[i]["step_background"]
                output_lines.append(step_desc)
                output_lines.append(previous_llm_code[i] or "")
                previous_code_parts.append(previous_llm_code[i] or "")
                output_lines.append("------")

            # Next step description + function header
            next_desc = sub_steps[num_steps - 1]["step_description_prompt"]
            if self.with_background:
                next_desc = next_desc + "\n" + sub_steps[num_steps - 1]["step_background"]

            header_docstring = sub_steps[num_steps - 1]["function_header"]
            return_str = sub_steps[num_steps - 1]["return_line"]

            problem_steps_str = "\n\n".join(output_lines[:-1]) if output_lines else ""
            next_step_str = f"{next_desc}\n\n{header_docstring}\n\n{return_str}"

            prompt = self.template.format(
                problem_steps_str=problem_steps_str,
                next_step_str=next_step_str,
                dependencies=dependencies,
            )

            # `previous_code` is what the official repo prepends when saving the .py file
            previous_code = f"{dependencies}\n" + "\n".join(previous_code_parts) + "\n"

            # -- Call MAS runner --
            step_task = BenchmarkTask(
                task_id=f"{prob_id}.{num_steps}",
                prompt=[{"role": "user", "content": prompt}],
                reference_answer=task.reference_answer,
                metadata=task.metadata,
            )

            mas_result = runner.run_task(task=step_task, run_index=run_index, seed=seed, benchmark_name="scicode")
            all_events.extend(mas_result.trace_events)
            merge_step_run_metadata(
                aggregate_metadata,
                dict(mas_result.run_metadata),
                outer_step_index=idx,
                step_task_id=step_task.task_id,
                final_answer=mas_result.final_answer,
            )

            # -- Extract and store code (mirrors official logic) --
            raw_response = mas_result.final_answer
            raw_responses_per_step[idx] = raw_response
            repaired_response = self._repair_response_if_needed(
                raw_response=raw_response,
                runner=runner,
                function_header=sub_steps[idx]["function_header"],
                step_task_id=step_task.task_id,
            )
            repaired_responses_per_step[idx] = repaired_response
            python_code = _extract_python_script(repaired_response)

            # Store the extracted *function only* for use in future prompts
            try:
                func_name = extract_function_name(sub_steps[idx]["function_header"])
                function_code = get_function_from_code(python_code, func_name)
                previous_llm_code[idx] = function_code if function_code else python_code
            except Exception:
                previous_llm_code[idx] = python_code

            # Store the full runnable file (deps + all prev funcs + this func)
            # This mirrors `save_response_with_steps`: f'{previous_code}\n{python_code}'
            full_code_per_step[idx] = f"{previous_code}\n{python_code}"

        return MASRunResult(
            final_answer=full_code_per_step[-1] if full_code_per_step else "",
            trace_events=all_events,
            run_metadata={
                "full_code_per_step": full_code_per_step,
                "previous_llm_code": [c or "" for c in previous_llm_code],
                "raw_responses_per_step": raw_responses_per_step,
                "repaired_responses_per_step": repaired_responses_per_step,
                **aggregate_metadata,
            },
        )

    def _repair_response_if_needed(
        self,
        *,
        raw_response: str,
        runner: MASRunner,
        function_header: str,
        step_task_id: str,
    ) -> str:
        if not self.llm_repair:
            return raw_response
        llm_client = getattr(runner, "llm_client", None)
        client = getattr(llm_client, "client", None)
        if client is None:
            return raw_response

        model = self.llm_repair_model or llm_client.model_for_agent_type("default")
        repair_prompt = self._build_repair_prompt(
            raw_response=raw_response,
            function_header=function_header,
        )
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=repair_prompt,
                temperature=0.0,
            )
            repaired = llm_client._extract_text(completion).strip()
            return repaired or raw_response
        except Exception as exc:
            logger.debug(f"SciCode {step_task_id} repair failed: {exc}")
            return raw_response

    @staticmethod
    def _build_repair_prompt(
        *,
        raw_response: str,
        function_header: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict code-format repair tool. "
                    "Your job is to repair formatting only. "
                    "Do not solve the task again, do not add new helper functions, "
                    "do not change the algorithm, and do not add explanations. "
                    "Return exactly one ```python``` code block."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Repair the following model output into a single valid Python code block.\n"
                    "Keep the original code content as much as possible.\n"
                    "Allowed fixes:\n"
                    "- move prose outside the code block out of the response\n"
                    "- ensure ```python starts on its own line\n"
                    "- ensure '# Background:' is on its own line\n"
                    "- ensure 'def', 'async def', and 'class' start on a new line\n"
                    "- preserve the requested function header exactly if it already appears\n\n"
                    f"Target function header:\n{function_header}\n\n"
                    "Raw model output:\n"
                    f"{raw_response}"
                ),
            },
        ]

    # ---- evaluate ----------------------------------------------------------

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        run_metadata = run_metadata or {}
        full_code_per_step: list[str] = run_metadata.get("full_code_per_step", [])
        sub_steps = task.metadata["sub_steps"]
        prob_id = task.task_id

        total_steps = len(sub_steps)
        total_correct = 0
        step_results: dict[str, str] = {}

        h5_file_path = Path(self.h5py_file).resolve().as_posix()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            for idx in range(total_steps):
                step_id = sub_steps[idx]["step_number"]

                # Skip hardcoded special steps (official repo auto-passes these)
                if (prob_id, idx) in SKIPPED_STEPS:
                    total_correct += 1
                    step_results[step_id] = "skipped (official)"
                    continue

                test_cases = sub_steps[idx]["test_cases"]

                # The full code for this step already contains dependencies + all
                # previous functions + this step's function (assembled in run()).
                code_content = full_code_per_step[idx] if idx < len(full_code_per_step) else ""
                if not code_content.strip():
                    step_results[step_id] = "empty code"
                    continue

                # Assemble the test script exactly like official test_generated_code.py:
                #   1. The full code
                #   2. Import process_hdf5_to_tuple
                #   3. Load targets
                #   4. Run assertions
                assert_file = tmp_path / f"{step_id}.py"
                with assert_file.open("w", encoding="utf-8") as f:
                    f.write(code_content)
                    f.write("\n\nfrom scicode.parse.parse import process_hdf5_to_tuple\n\n")
                    f.write(
                        f"targets = process_hdf5_to_tuple('{step_id}', "
                        f"{len(test_cases)}, '{h5_file_path}')\n"
                    )
                    for i, test_case in enumerate(test_cases):
                        f.write(f"target = targets[{i}]\n\n")
                        for line in test_case.split("\n"):
                            f.write(line + "\n")

                try:
                    res = subprocess.run(
                        [sys.executable, str(assert_file)],
                        capture_output=True,
                        text=True,
                        timeout=self.execution_timeout_s,
                    )
                    if res.returncode == 0:
                        total_correct += 1
                        step_results[step_id] = "pass"
                    else:
                        step_results[step_id] = f"fail: {res.stderr[:200]}"
                        logger.debug(
                            f"SciCode {prob_id}.{step_id} assertion failed:\n{res.stderr[:500]}"
                        )
                except subprocess.TimeoutExpired:
                    step_results[step_id] = "timeout"
                    logger.debug(f"SciCode {prob_id}.{step_id} timed out.")
                except Exception as e:
                    step_results[step_id] = f"error: {e}"
                    logger.debug(f"SciCode {prob_id}.{step_id} execution error: {e}")

        success = total_correct == total_steps
        score = 1.0 if success else 0.0

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=score,
            success=success,
            details={
                "prediction": prediction[:500],
                "Total Correct": total_correct,
                "Total Steps": total_steps,
                "Main Problem Resolve Rate": 1.0 if success else 0.0,
                "Subproblem Resolve Rate": (
                    total_correct / total_steps if total_steps > 0 else 0.0
                ),
                "step_results": step_results,
            },
        )

    def requirements(self) -> dict[str, Any]:
        return {"scicode": ">=1.0.0", "datasets": ">=2.0.0"}

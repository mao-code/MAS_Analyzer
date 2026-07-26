from __future__ import annotations

import re
import time
import json
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.base import BenchmarkTask
from descriptor.schema import TraceEvent

from .runner import MASRunResult, MASRunner

BASELINE_DIRECT = "direct"
BASELINE_COT = "cot"
BASELINE_SELF_CONSISTENCY = "self_consistency"
BASELINE_SELF_REFINE = "self_refine"

PROMPTING_BASELINES = {
    BASELINE_DIRECT,
    BASELINE_COT,
    BASELINE_SELF_CONSISTENCY,
    BASELINE_SELF_REFINE,
}


def normalize_prompting_baseline(value: str | None) -> str:
    raw = str(value or BASELINE_DIRECT).strip().lower().replace("-", "_")
    aliases = {
        "none": BASELINE_DIRECT,
        "sas": BASELINE_DIRECT,
        "single_agent": BASELINE_DIRECT,
        "chain_of_thought": BASELINE_COT,
        "selfconsistency": BASELINE_SELF_CONSISTENCY,
        "sc": BASELINE_SELF_CONSISTENCY,
        "selfrefine": BASELINE_SELF_REFINE,
        "sr": BASELINE_SELF_REFINE,
    }
    normalized = aliases.get(raw, raw)
    if normalized not in PROMPTING_BASELINES:
        raise ValueError(
            f"Unknown prompting baseline '{value}'. Expected one of {sorted(PROMPTING_BASELINES)}."
        )
    return normalized


class PromptingBaselineRunner:
    """Runner wrapper for fixed prompting baselines.

    These are inference-time baselines, not topology variants. The wrapper keeps
    benchmark adapters and evaluator contracts unchanged: adapters still call
    ``runner.run_task(...)``, and benchmark correctness remains solely
    ``benchmark.evaluate(...).success``.
    """

    def __init__(
        self,
        base_runner: MASRunner,
        *,
        baseline: str,
        self_consistency_samples: int = 3,
        self_refine_rounds: int = 3,
    ) -> None:
        self.base_runner = base_runner
        self.baseline = normalize_prompting_baseline(baseline)
        self.self_consistency_samples = max(1, int(self_consistency_samples))
        self.self_refine_rounds = max(1, int(self_refine_rounds))
        self.config = base_runner.config
        self.openrouter_client = base_runner.openrouter_client
        self.llm_client = base_runner.llm_client
        self.engine = base_runner.engine
        self._checkpoint_task_dir: Path | None = None
        self._checkpoint_run_index: int | None = None

    def set_run_checkpoint_context(self, *, task_dir: Path | str | None, run_index: int | None) -> None:
        self._checkpoint_task_dir = Path(task_dir) if task_dir is not None else None
        self._checkpoint_run_index = int(run_index) if run_index is not None else None

    def run_task(
        self,
        task: Any,
        run_index: int,
        seed: int,
        **kwargs: Any,
    ) -> MASRunResult:
        if self.baseline == BASELINE_DIRECT:
            result = self.base_runner.run_task(task=task, run_index=run_index, seed=seed, **kwargs)
            return self._tag_result(result)
        if self.baseline == BASELINE_COT:
            cot_task = _with_prompt_instruction(
                task,
                "Think step by step, then provide the final answer in the required format.",
            )
            result = self.base_runner.run_task(
                task=cot_task,
                run_index=run_index,
                seed=seed,
                **kwargs,
            )
            return self._tag_result(result)
        if self.baseline == BASELINE_SELF_CONSISTENCY:
            return self._run_self_consistency(task=task, run_index=run_index, seed=seed, **kwargs)
        if self.baseline == BASELINE_SELF_REFINE:
            return self._run_self_refine(task=task, run_index=run_index, seed=seed, **kwargs)
        raise AssertionError(f"Unhandled prompting baseline: {self.baseline}")

    def _run_self_consistency(
        self,
        *,
        task: Any,
        run_index: int,
        seed: int,
        **kwargs: Any,
    ) -> MASRunResult:
        samples: list[MASRunResult] = []
        for sample_index in range(self.self_consistency_samples):
            sample_seed = int(seed) + (sample_index * 100_000)
            sample_task = _with_prompt_instruction(
                task,
                (
                    "Solve independently. Think step by step internally if useful, then provide "
                    "only the final answer/action in the required format."
                ),
            )
            samples.append(
                self.base_runner.run_task(
                    task=sample_task,
                    run_index=run_index,
                    seed=sample_seed,
                    **kwargs,
                )
            )
            self._write_prompting_checkpoint(
                task=task,
                run_index=run_index,
                phase="self_consistency_sample_complete",
                complete=False,
                payload={
                    "sample_index": sample_index,
                    "samples_completed": len(samples),
                    "samples_total": self.self_consistency_samples,
                    "answers": [
                        {
                            "sample_index": idx,
                            "answer": sample.final_answer,
                            "normalized_answer": _normalize_answer(sample.final_answer),
                            "metadata": _compact_sample_metadata(sample.run_metadata),
                        }
                        for idx, sample in enumerate(samples)
                    ],
                },
            )

        selected_index = _select_majority_answer_index([sample.final_answer for sample in samples])
        selected = samples[selected_index]
        trace_events = _merge_sample_traces(samples, selected_index=selected_index)
        metadata = {
            **dict(selected.run_metadata),
            "prompting_baseline": self.baseline,
            "self_consistency_samples": self.self_consistency_samples,
            "self_consistency_selected_index": selected_index,
            "self_consistency_answers": [
                {
                    "sample_index": idx,
                    "answer": sample.final_answer,
                    "normalized_answer": _normalize_answer(sample.final_answer),
                    "metadata": _compact_sample_metadata(sample.run_metadata),
                }
                for idx, sample in enumerate(samples)
            ],
        }
        self._write_prompting_checkpoint(
            task=task,
            run_index=run_index,
            phase="self_consistency_complete",
            complete=True,
            payload={
                "selected_index": selected_index,
                "selected_answer": selected.final_answer,
                "samples_completed": len(samples),
                "samples_total": self.self_consistency_samples,
                "answers": metadata["self_consistency_answers"],
            },
        )
        return MASRunResult(
            final_answer=selected.final_answer,
            trace_events=trace_events,
            run_metadata=metadata,
        )

    def _run_self_refine(
        self,
        *,
        task: Any,
        run_index: int,
        seed: int,
        **kwargs: Any,
    ) -> MASRunResult:
        initial_task = _with_prompt_instruction(
            task,
            "Provide your best initial answer/action in the required format.",
        )
        initial = self.base_runner.run_task(
            task=initial_task,
            run_index=run_index,
            seed=seed,
            **kwargs,
        )
        current_answer = str(initial.final_answer or "")
        trace_events = list(initial.trace_events)
        refine_records: list[dict[str, Any]] = []
        self._write_prompting_checkpoint(
            task=task,
            run_index=run_index,
            phase="self_refine_initial_complete",
            complete=False,
            payload={
                "initial_answer": initial.final_answer,
                "rounds_completed": 0,
                "rounds_total": self.self_refine_rounds,
                "records": [],
            },
        )

        for refine_index in range(self.self_refine_rounds):
            feedback_text, feedback_event = self._call_refine_llm(
                task=task,
                run_index=run_index,
                seed=seed,
                refine_index=refine_index,
                current_answer=current_answer,
                mode="feedback",
            )
            self._write_prompting_checkpoint(
                task=task,
                run_index=run_index,
                phase="self_refine_feedback_complete",
                complete=False,
                payload={
                    "round": refine_index + 1,
                    "feedback": feedback_text,
                    "current_answer": current_answer,
                    "rounds_completed": refine_index,
                    "rounds_total": self.self_refine_rounds,
                    "records": refine_records,
                },
            )
            revised_answer, revise_event = self._call_refine_llm(
                task=task,
                run_index=run_index,
                seed=seed,
                refine_index=refine_index,
                current_answer=current_answer,
                feedback=feedback_text,
                mode="revise",
            )
            trace_events.extend([feedback_event, revise_event])
            refine_records.append(
                {
                    "round": refine_index + 1,
                    "feedback": feedback_text,
                    "previous_answer": current_answer,
                    "revised_answer": revised_answer,
                }
            )
            if revised_answer.strip():
                current_answer = revised_answer.strip()
            self._write_prompting_checkpoint(
                task=task,
                run_index=run_index,
                phase="self_refine_round_complete",
                complete=False,
                payload={
                    "round": refine_index + 1,
                    "current_answer": current_answer,
                    "rounds_completed": refine_index + 1,
                    "rounds_total": self.self_refine_rounds,
                    "records": refine_records,
                },
            )

        metadata = {
            **dict(initial.run_metadata),
            "prompting_baseline": self.baseline,
            "self_refine_rounds": self.self_refine_rounds,
            "self_refine_initial_answer": initial.final_answer,
            "self_refine_records": refine_records,
            "self_refine_note": (
                "Feedback/revision calls do not execute side-effect tools; benchmark adapters "
                "still score with the original run metadata plus the final revised text."
            ),
        }
        self._write_prompting_checkpoint(
            task=task,
            run_index=run_index,
            phase="self_refine_complete",
            complete=True,
            payload={
                "final_answer": current_answer,
                "initial_answer": initial.final_answer,
                "rounds_completed": len(refine_records),
                "rounds_total": self.self_refine_rounds,
                "records": refine_records,
            },
        )
        return MASRunResult(
            final_answer=current_answer,
            trace_events=trace_events,
            run_metadata=metadata,
        )

    def _write_prompting_checkpoint(
        self,
        *,
        task: Any,
        run_index: int,
        phase: str,
        complete: bool,
        payload: dict[str, Any],
    ) -> None:
        task_dir = self._checkpoint_task_dir
        context_run_index = self._checkpoint_run_index
        if task_dir is None or context_run_index != int(run_index):
            return
        task_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = task_dir / f"run_{run_index}.prompting_checkpoint.json"
        checkpoint_payload = {
            "task_id": str(getattr(task, "task_id", "")),
            "run_index": int(run_index),
            "prompting_baseline": self.baseline,
            "phase": phase,
            "complete": bool(complete),
            "updated_at": time.time(),
            **payload,
        }
        tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(checkpoint_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp_path.replace(checkpoint_path)

    def _call_refine_llm(
        self,
        *,
        task: Any,
        run_index: int,
        seed: int,
        refine_index: int,
        current_answer: str,
        mode: str,
        feedback: str = "",
    ) -> tuple[str, TraceEvent]:
        start = time.time()
        prompt = _self_refine_prompt(
            task_prompt=getattr(task, "prompt", ""),
            current_answer=current_answer,
            mode=mode,
            feedback=feedback,
        )
        result = self.openrouter_client.generate(
            prompt=prompt,
            agent_type="default",
            task_id=str(getattr(task, "task_id", "")),
            run_index=run_index,
            agent_id=f"self_refine_{mode}_{refine_index}",
            tools=[],
            max_tool_iterations=1,
            temperature=0.0,
            max_tokens=_self_refine_max_tokens(mode),
        )
        end = time.time()
        event = TraceEvent(
            timestamp_start=start,
            timestamp_end=end,
            actor=f"self_refine_{mode}_{refine_index}",
            event_type="verify" if mode == "feedback" else "revise",
            payload={
                "prompting_baseline": self.baseline,
                "mode": mode,
                "round": refine_index + 1,
                "input_answer": current_answer,
                "feedback": feedback,
                "output": result.text,
                "model": result.model,
                "mock_used": result.mock_used,
            },
            token_in=int(result.token_in),
            token_out=int(result.token_out),
            latency_ms=(end - start) * 1000.0,
            cost_usd=float(result.cost_usd),
        )
        return result.text, event

    def _tag_result(self, result: MASRunResult) -> MASRunResult:
        metadata = dict(result.run_metadata)
        metadata["prompting_baseline"] = self.baseline
        return MASRunResult(
            final_answer=result.final_answer,
            trace_events=result.trace_events,
            run_metadata=metadata,
        )

    def reload_self_evolved_skill(self) -> None:
        self.base_runner.reload_self_evolved_skill()


def _with_prompt_instruction(task: Any, instruction: str) -> Any:
    prompt = getattr(task, "prompt", "")
    updated_prompt = _append_instruction(prompt, instruction)
    if isinstance(task, BenchmarkTask):
        return replace(task, prompt=updated_prompt)
    try:
        return replace(task, prompt=updated_prompt)
    except Exception:
        task.prompt = updated_prompt
        return task


def _append_instruction(prompt: Any, instruction: str) -> Any:
    if isinstance(prompt, str):
        return f"{prompt}\n\nInstruction: {instruction}"
    if isinstance(prompt, list):
        messages = [dict(item) if isinstance(item, dict) else item for item in prompt]
        for index in range(len(messages) - 1, -1, -1):
            item = messages[index]
            if isinstance(item, dict) and item.get("role") == "user":
                item["content"] = f"{item.get('content', '')}\n\nInstruction: {instruction}"
                messages[index] = item
                return messages
        return [{"role": "system", "content": instruction}, *messages]
    return f"{prompt}\n\nInstruction: {instruction}"


def _select_majority_answer_index(answers: list[str]) -> int:
    normalized = [_normalize_answer(answer) for answer in answers]
    counts = Counter(normalized)
    best_key, _ = counts.most_common(1)[0]
    return normalized.index(best_key)


def _normalize_answer(answer: str) -> str:
    text = str(answer or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^answer\s*:\s*", "", text)
    return text


def _merge_sample_traces(samples: list[MASRunResult], *, selected_index: int) -> list[TraceEvent]:
    merged: list[TraceEvent] = []
    for sample_index, sample in enumerate(samples):
        for event in sample.trace_events:
            extra = dict(event.extra)
            extra["self_consistency_sample_index"] = sample_index
            extra["self_consistency_selected"] = sample_index == selected_index
            merged.append(
                TraceEvent(
                    timestamp_start=event.timestamp_start,
                    timestamp_end=event.timestamp_end,
                    actor=event.actor,
                    event_type=event.event_type,
                    payload=dict(event.payload),
                    token_in=event.token_in,
                    token_out=event.token_out,
                    latency_ms=event.latency_ms,
                    cost_usd=event.cost_usd,
                    state_id=event.state_id,
                    extra=extra,
                )
            )
    return merged


def _compact_sample_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "topology",
        "run_status",
        "fallback",
        "needs_rerun",
        "tool_calls_total",
        "messages_sent_total",
        "final_reason",
    ]
    return {key: metadata.get(key) for key in keys if key in metadata}


def _self_refine_max_tokens(mode: str) -> int | None:
    if os.getenv("MAS_SELF_REFINE_NO_MAX_TOKENS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    if mode == "feedback":
        return 512
    return 768


def _self_refine_prompt(
    *,
    task_prompt: Any,
    current_answer: str,
    mode: str,
    feedback: str = "",
) -> list[dict[str, str]]:
    task_text = _prompt_to_text(task_prompt)
    if mode == "feedback":
        user = (
            "Review the answer for the task. Keep the critique concise and actionable.\n"
            "Return exactly this format, with at most 3 bullets:\n"
            "Correctness: True or False\n"
            "Feedback:\n"
            "- <one concrete issue, missing evidence, or 'No changes needed.'>\n\n"
            f"Task:\n{task_text}\n\nCurrent answer:\n{current_answer}"
        )
    else:
        user = (
            "Revise the answer using the feedback. Return only the final answer/action in the "
            "task's required format. Do not include explanations, critique, markdown, or extra "
            "sections. If the previous answer is already correct, repeat it unchanged.\n\n"
            f"Task:\n{task_text}\n\nPrevious answer:\n{current_answer}\n\nFeedback:\n{feedback}"
        )
    return [
        {
            "role": "system",
            "content": "You are the fixed Self-Refine baseline feedback/refinement model.",
        },
        {"role": "user", "content": user},
    ]


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        chunks: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                chunks.append(f"{item.get('role', 'message')}: {item.get('content', '')}")
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(prompt)

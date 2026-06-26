from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from benchmark import get_benchmark, list_benchmarks

DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_EXCLUDED_BENCHMARKS = {"finance_agent"}
DEFAULT_BASELINES = ("cot", "self_consistency", "self_refine", "debate", "adas", "aflow")


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    paper_name: str
    description: str
    calls_worst_case: int
    implementation_note: str = ""


BASELINE_SPECS = {
    "cot": BaselineSpec(
        name="cot",
        paper_name="CoT",
        description='Zero-shot chain-of-thought: "Please think step by step and then solve the task."',
        calls_worst_case=1,
    ),
    "self_consistency": BaselineSpec(
        name="self_consistency",
        paper_name="Self-Consistency",
        description="SC@9 with diversified CoT traces at temperature 0.8 and rule-based majority vote.",
        calls_worst_case=9,
    ),
    "self_refine": BaselineSpec(
        name="self_refine",
        paper_name="Self-Refine",
        description="One predictor plus one reflector/refiner loop, maximum 5 reflection rounds.",
        calls_worst_case=11,
    ),
    "debate": BaselineSpec(
        name="debate",
        paper_name="Multi-Agent Debate",
        description="Three debating agents for 3 rounds, followed by one judging aggregator.",
        calls_worst_case=10,
    ),
    "adas": BaselineSpec(
        name="adas",
        paper_name="ADAS",
        description=(
            "LLM meta-agent proposes agentic designs conditioned on former baseline evaluations; "
            "paper setup uses 30 rounds with 3 validation evaluations per round."
        ),
        calls_worst_case=-1,
        implementation_note=(
            "Safe standalone reproduction: the optimizer chooses among fixed workflow families instead "
            "of executing arbitrary generated Python code."
        ),
    ),
    "aflow": BaselineSpec(
        name="aflow",
        paper_name="AFlow",
        description=(
            "Automatic workflow design over predefined operators; paper setup uses 20 rounds, "
            "5 validation runs per round, and k=3."
        ),
        calls_worst_case=-1,
        implementation_note=(
            "Safe standalone reproduction: MCTS-style search is approximated over fixed operator "
            "workflows and does not execute generated scripts."
        ),
    ),
}

WORKFLOW_VARIANTS = ("cot", "review", "self_refine_once", "debate_once", "ensemble3", "test_review")


@dataclass
class ChatResult:
    text: str
    model: str
    token_in: int = 0
    token_out: int = 0
    mock_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class StandaloneOpenRouterClient:
    """Minimal OpenRouter client so baselines do not depend on the repo MAS runtime."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = 600.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.disable_live = _env_flag("MAS_DISABLE_LIVE_LLM")
        self.require_live = (not self.disable_live) and _env_flag("MAS_REQUIRE_LIVE_LLM")

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        task_id: str,
        agent_id: str,
        temperature: float,
    ) -> ChatResult:
        if self.disable_live or not self.api_key:
            if self.require_live:
                raise RuntimeError(
                    "MAS_REQUIRE_LIVE_LLM is enabled but OPENROUTER_API_KEY is missing."
                )
            return self._mock(messages=messages, task_id=task_id, agent_id=agent_id)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            text=str(message.get("content") or ""),
            model=str(data.get("model") or self.model),
            token_in=int(usage.get("prompt_tokens") or 0),
            token_out=int(usage.get("completion_tokens") or 0),
            mock_used=False,
            metadata={"elapsed_s": round(time.perf_counter() - started, 3)},
        )

    def _mock(self, *, messages: list[dict[str, str]], task_id: str, agent_id: str) -> ChatResult:
        seed = hashlib.sha256(
            json.dumps(messages, sort_keys=True).encode("utf-8")
            + task_id.encode("utf-8")
            + agent_id.encode("utf-8")
        ).hexdigest()[:10]
        user_text = " ".join(
            message.get("content", "") for message in messages if message.get("role") == "user"
        )
        words = re.findall(r"[A-Za-z0-9_.$:/%-]+", user_text)[:12]
        text = f"MOCK({agent_id}) seed={seed}: {' '.join(words) or 'answer'}"
        return ChatResult(
            text=text,
            model=self.model,
            token_in=sum(len(message.get("content", "").split()) for message in messages),
            token_out=len(text.split()),
            mock_used=True,
        )


def run_baseline_suite(args: argparse.Namespace) -> dict[str, Any]:
    _load_env_file(args.env_file)
    client = StandaloneOpenRouterClient(
        model=args.model,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=args.openrouter_base_url,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
    )
    benchmarks = _resolve_benchmarks(args)
    run_id = args.run_id or _now_stamp()
    output_root = Path(args.output_dir).expanduser().resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "paper": "arXiv:2502.02533v2",
        "method": "mass_paper_baselines",
        "model": args.model,
        "config_path": str(Path(args.config).expanduser().resolve()) if args.config else None,
        "benchmarks": {},
        "baselines": list(args.baseline),
        "excluded_benchmarks": sorted(DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)),
        "settings": vars(args),
    }

    for benchmark_name in benchmarks:
        print(f"[{_now_stamp()}] BASELINE_BENCH_START benchmark={benchmark_name}", flush=True)
        try:
            payload = _run_benchmark(
                benchmark_name=benchmark_name,
                client=client,
                args=args,
                output_root=output_root,
                benchmark_config=_benchmark_section_config(args.config, benchmark_name),
            )
            summary["benchmarks"][benchmark_name] = payload
            print(f"[{_now_stamp()}] BASELINE_BENCH_DONE benchmark={benchmark_name}", flush=True)
        except Exception as exc:
            error_payload = {"error": f"{type(exc).__name__}: {exc}"}
            summary["benchmarks"][benchmark_name] = error_payload
            _write_json(output_root / benchmark_name / "error.json", error_payload)
            print(
                f"[{_now_stamp()}] BASELINE_BENCH_ERROR benchmark={benchmark_name} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            if not args.keep_going:
                raise

    _write_json(output_root / "summary.json", summary)
    print(f"[{_now_stamp()}] BASELINE_RUN_DONE output={output_root}", flush=True)
    return summary


def _run_benchmark(
    *,
    benchmark_name: str,
    client: StandaloneOpenRouterClient,
    args: argparse.Namespace,
    output_root: Path,
    benchmark_config: dict[str, Any],
) -> dict[str, Any]:
    benchmark = get_benchmark(benchmark_name, config=benchmark_config)
    tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
    if not tasks:
        raise RuntimeError(f"No tasks loaded for benchmark '{benchmark_name}'")

    benchmark_dir = output_root / benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "benchmark": benchmark_name,
        "task_count": len(tasks),
        "baselines": {},
    }
    for baseline_name in args.baseline:
        spec = BASELINE_SPECS[baseline_name]
        method_dir = benchmark_dir / baseline_name
        method_dir.mkdir(parents=True, exist_ok=True)
        scores: list[float] = []
        task_payloads = []
        for task in tasks:
            prediction, trace = _run_baseline(
                baseline_name=baseline_name,
                client=client,
                benchmark=benchmark,
                task=task,
                benchmark_name=benchmark_name,
                task_id=str(task.task_id),
                prompt=task.prompt,
                args=args,
            )
            evaluation = benchmark.evaluate(
                task,
                prediction,
                run_metadata={
                    "mass_paper_baseline": baseline_name,
                    "paper": "arXiv:2502.02533v2",
                    "model": args.model,
                    "trace": trace,
                },
            )
            scores.append(float(evaluation.score))
            task_payloads.append(
                {
                    "task_id": str(task.task_id),
                    "prediction": prediction,
                    "score": float(evaluation.score),
                    "success": bool(evaluation.success),
                    "evaluation_details": _jsonable(evaluation.details),
                    "trace": trace,
                }
            )
        baseline_payload = {
            "spec": spec.__dict__,
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "tasks": task_payloads,
        }
        payload["baselines"][baseline_name] = baseline_payload
        _write_json(method_dir / "results.json", baseline_payload)

    _write_json(benchmark_dir / "paper_baseline_results.json", payload)
    return payload


def _run_baseline(
    *,
    baseline_name: str,
    client: StandaloneOpenRouterClient,
    benchmark: Any,
    task: Any,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    if baseline_name == "cot":
        return _run_cot(client, benchmark_name, task_id, prompt, args.temperature)
    if baseline_name == "self_consistency":
        return _run_self_consistency(client, benchmark_name, task_id, prompt, args.sc_samples)
    if baseline_name == "self_refine":
        return _run_self_refine(
            client, benchmark_name, task_id, prompt, args.temperature, args.self_refine_rounds
        )
    if baseline_name == "debate":
        return _run_debate(
            client,
            benchmark_name,
            task_id,
            prompt,
            args.temperature,
            args.debate_agents,
            args.debate_rounds,
        )
    if baseline_name == "adas":
        return _run_adas(
            client=client,
            benchmark=benchmark,
            task=task,
            benchmark_name=benchmark_name,
            task_id=task_id,
            prompt=prompt,
            temperature=args.temperature,
            rounds=args.adas_rounds,
            validation_repeats=args.adas_validation_repeats,
        )
    if baseline_name == "aflow":
        return _run_aflow(
            client=client,
            benchmark=benchmark,
            task=task,
            benchmark_name=benchmark_name,
            task_id=task_id,
            prompt=prompt,
            temperature=args.temperature,
            rounds=args.aflow_rounds,
            validation_runs=args.aflow_validation_runs,
            k=args.aflow_k,
        )
    raise ValueError(f"Unsupported baseline: {baseline_name}")


def _run_cot(
    client: StandaloneOpenRouterClient,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    result = _chat(
        client,
        task_id=f"{benchmark_name}:{task_id}:cot",
        agent_id="cot",
        system="You are a careful zero-shot chain-of-thought solver.",
        user=f"Please think step by step and then solve the task.\n\nTask:\n{prompt}",
        temperature=temperature,
    )
    return result.text, {"calls": [_call_payload("cot", result)]}


def _run_self_consistency(
    client: StandaloneOpenRouterClient,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    samples: int,
) -> tuple[str, dict[str, Any]]:
    calls = []
    answers = []
    for index in range(samples):
        result = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:sc:{index}",
            agent_id=f"sc_{index}",
            system="You are a careful zero-shot chain-of-thought solver. Produce an independent reasoning trace.",
            user=f"Please think step by step and then solve the task.\n\nTask:\n{prompt}",
            temperature=0.8,
        )
        calls.append(_call_payload(f"sc_{index}", result))
        answers.append(_extract_answer(result.text))
    prediction = _majority_vote(answers) or (answers[-1] if answers else "")
    return prediction, {"samples": samples, "answers": answers, "calls": calls}


def _run_self_refine(
    client: StandaloneOpenRouterClient,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    temperature: float,
    rounds: int,
) -> tuple[str, dict[str, Any]]:
    calls = []
    predictor = _chat(
        client,
        task_id=f"{benchmark_name}:{task_id}:self_refine:initial",
        agent_id="predictor",
        system="You are the predictor in a Self-Refine baseline.",
        user=f"Please think step by step and then solve the task.\n\nTask:\n{prompt}",
        temperature=temperature,
    )
    answer = predictor.text
    calls.append(_call_payload("predictor_initial", predictor))
    for round_index in range(rounds):
        reflection = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:self_refine:reflect:{round_index}",
            agent_id="reflector",
            system="You are a self-reflector. Criticize the answer. If absolutely correct, include the word Correct.",
            user=f"Task:\n{prompt}\n\nCurrent answer:\n{answer}\n\nGive concise feedback and correctness.",
            temperature=temperature,
        )
        calls.append(_call_payload(f"reflector_{round_index}", reflection))
        if _is_correct_signal(reflection.text):
            break
        refined = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:self_refine:refine:{round_index}",
            agent_id="refiner",
            system="You are a refiner. Use the feedback to improve the answer.",
            user=(
                f"Task:\n{prompt}\n\nPrevious answer:\n{answer}\n\n"
                f"Reflection:\n{reflection.text}\n\nReturn the updated final answer."
            ),
            temperature=temperature,
        )
        answer = refined.text
        calls.append(_call_payload(f"refiner_{round_index}", refined))
    return answer, {"max_rounds": rounds, "calls": calls}


def _run_debate(
    client: StandaloneOpenRouterClient,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    temperature: float,
    agents: int,
    rounds: int,
) -> tuple[str, dict[str, Any]]:
    calls = []
    opinions: list[str] = []
    for round_index in range(rounds):
        previous = (
            "\n\n".join(f"Agent {idx}: {opinion}" for idx, opinion in enumerate(opinions))
            or "(none yet)"
        )
        next_opinions = []
        for agent_index in range(agents):
            result = _chat(
                client,
                task_id=f"{benchmark_name}:{task_id}:debate:{round_index}:{agent_index}",
                agent_id=f"debate_agent_{agent_index}",
                system="You are one agent in a multi-agent debate baseline.",
                user=(
                    f"Task:\n{prompt}\n\nPrevious debate opinions:\n{previous}\n\n"
                    "Justify your answer, consider other agents, and finish with your updated answer."
                ),
                temperature=temperature,
            )
            next_opinions.append(result.text)
            calls.append(_call_payload(f"debate_r{round_index}_a{agent_index}", result))
        opinions = next_opinions
    judge = _chat(
        client,
        task_id=f"{benchmark_name}:{task_id}:debate:judge",
        agent_id="debate_judge",
        system="You are the aggregator judge in a multi-agent debate baseline.",
        user=(
            f"Task:\n{prompt}\n\nFinal debate opinions:\n"
            f"{chr(10).join(f'Agent {idx}: {opinion}' for idx, opinion in enumerate(opinions))}\n\n"
            "Choose the best answer and return the final prediction."
        ),
        temperature=temperature,
    )
    calls.append(_call_payload("debate_judge", judge))
    return judge.text, {"agents": agents, "rounds": rounds, "calls": calls}


def _run_adas(
    *,
    client: StandaloneOpenRouterClient,
    benchmark: Any,
    task: Any,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    temperature: float,
    rounds: int,
    validation_repeats: int,
) -> tuple[str, dict[str, Any]]:
    """ADAS-style meta-agent search without executing generated code."""

    search_trace = []
    leaderboard: list[dict[str, Any]] = []
    best_prediction = ""
    best_score = float("-inf")
    former_summary = "No former evaluations yet."

    for round_index in range(rounds):
        proposal = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:adas:proposal:{round_index}",
            agent_id="adas_meta_agent",
            system=(
                "You are an ADAS-style meta-agent. Select a safe workflow family for this task. "
                f"Allowed workflow families: {', '.join(WORKFLOW_VARIANTS)}."
            ),
            user=(
                f"Task:\n{prompt}\n\nFormer baseline/workflow evaluations:\n{former_summary}\n\n"
                "Return the best workflow family name and a short rationale."
            ),
            temperature=temperature,
        )
        variant = _choose_variant(proposal.text, fallback_index=round_index)
        repeat_scores = []
        repeat_payloads = []
        prediction = ""
        calls = [_call_payload(f"adas_meta_agent_{round_index}", proposal)]
        for repeat_index in range(validation_repeats):
            prediction, workflow_trace = _execute_workflow_variant(
                client=client,
                variant=variant,
                benchmark_name=benchmark_name,
                task_id=f"{task_id}:adas:{round_index}:{repeat_index}",
                prompt=prompt,
                temperature=temperature,
            )
            calls.extend(workflow_trace["calls"])
            score = _score_prediction(
                benchmark=benchmark,
                task=task,
                prediction=prediction,
                metadata={
                    "mass_paper_baseline": "adas_validation",
                    "variant": variant,
                    "round": round_index,
                    "repeat": repeat_index,
                },
            )
            repeat_scores.append(score)
            repeat_payloads.append(
                {"repeat": repeat_index, "score": score, "prediction": prediction}
            )

        avg_score = sum(repeat_scores) / len(repeat_scores) if repeat_scores else 0.0
        candidate_payload = {
            "round": round_index,
            "variant": variant,
            "proposal_text": proposal.text,
            "average_score": avg_score,
            "repeats": repeat_payloads,
            "calls": calls,
        }
        leaderboard.append(candidate_payload)
        search_trace.append(candidate_payload)
        if avg_score > best_score:
            best_score = avg_score
            best_prediction = prediction
        former_summary = _leaderboard_summary(leaderboard)

    return best_prediction, {
        "rounds": rounds,
        "validation_repeats": validation_repeats,
        "best_score_on_task_validation": best_score,
        "leaderboard": leaderboard,
        "search_trace": search_trace,
    }


def _run_aflow(
    *,
    client: StandaloneOpenRouterClient,
    benchmark: Any,
    task: Any,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    temperature: float,
    rounds: int,
    validation_runs: int,
    k: int,
) -> tuple[str, dict[str, Any]]:
    """AFlow-style operator workflow search over a safe predefined workflow set."""

    candidates = list(WORKFLOW_VARIANTS)
    leaderboard: list[dict[str, Any]] = []
    best_prediction = ""
    best_score = float("-inf")
    search_trace = []

    for round_index in range(rounds):
        ranked = sorted(leaderboard, key=lambda item: float(item["average_score"]), reverse=True)
        active = [item["variant"] for item in ranked[:k]]
        for variant in candidates:
            if variant not in active:
                active.append(variant)
            if len(active) >= max(1, k):
                break

        expansion = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:aflow:expand:{round_index}",
            agent_id="aflow_optimizer",
            system=(
                "You are an AFlow-style optimizer over predefined operators. "
                f"Allowed workflow families: {', '.join(WORKFLOW_VARIANTS)}."
            ),
            user=(
                f"Task:\n{prompt}\n\nCurrent leaderboard:\n{_leaderboard_summary(leaderboard)}\n\n"
                "Select or refine the next workflow family to evaluate."
            ),
            temperature=temperature,
        )
        proposed = _choose_variant(expansion.text, fallback_index=round_index)
        if proposed not in active:
            active[-1] = proposed

        round_payload = {
            "round": round_index,
            "active_variants": active,
            "calls": [_call_payload("aflow_optimizer", expansion)],
            "evaluations": [],
        }
        for variant in active:
            repeat_scores = []
            repeat_payloads = []
            prediction = ""
            calls = []
            for repeat_index in range(validation_runs):
                prediction, workflow_trace = _execute_workflow_variant(
                    client=client,
                    variant=variant,
                    benchmark_name=benchmark_name,
                    task_id=f"{task_id}:aflow:{round_index}:{variant}:{repeat_index}",
                    prompt=prompt,
                    temperature=temperature,
                )
                calls.extend(workflow_trace["calls"])
                score = _score_prediction(
                    benchmark=benchmark,
                    task=task,
                    prediction=prediction,
                    metadata={
                        "mass_paper_baseline": "aflow_validation",
                        "variant": variant,
                        "round": round_index,
                        "repeat": repeat_index,
                    },
                )
                repeat_scores.append(score)
                repeat_payloads.append(
                    {"repeat": repeat_index, "score": score, "prediction": prediction}
                )

            avg_score = sum(repeat_scores) / len(repeat_scores) if repeat_scores else 0.0
            candidate_payload = {
                "round": round_index,
                "variant": variant,
                "average_score": avg_score,
                "repeats": repeat_payloads,
                "calls": calls,
            }
            leaderboard.append(candidate_payload)
            round_payload["evaluations"].append(candidate_payload)
            if avg_score > best_score:
                best_score = avg_score
                best_prediction = prediction
        search_trace.append(round_payload)

    return best_prediction, {
        "rounds": rounds,
        "validation_runs": validation_runs,
        "k": k,
        "best_score_on_task_validation": best_score,
        "leaderboard": sorted(
            leaderboard, key=lambda item: float(item["average_score"]), reverse=True
        ),
        "search_trace": search_trace,
    }


def _execute_workflow_variant(
    *,
    client: StandaloneOpenRouterClient,
    variant: str,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    if variant == "cot":
        return _run_cot(client, benchmark_name, task_id, prompt, temperature)
    if variant == "self_refine_once":
        return _run_self_refine(client, benchmark_name, task_id, prompt, temperature, rounds=1)
    if variant == "debate_once":
        return _run_debate(client, benchmark_name, task_id, prompt, temperature, agents=2, rounds=1)
    if variant == "ensemble3":
        return _run_ensemble(client, benchmark_name, task_id, prompt, temperature, samples=3)
    if variant == "review":
        draft, draft_trace = _run_cot(client, benchmark_name, task_id, prompt, temperature)
        review = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:review",
            agent_id="reviewer",
            system="You are a reviewer. Improve the draft and return the final answer.",
            user=f"Task:\n{prompt}\n\nDraft answer:\n{draft}\n\nReturn the improved final answer.",
            temperature=temperature,
        )
        return review.text, {"calls": draft_trace["calls"] + [_call_payload("reviewer", review)]}
    if variant == "test_review":
        draft, draft_trace = _run_cot(client, benchmark_name, task_id, prompt, temperature)
        tester = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:test",
            agent_id="tester",
            system="You are a tester. Identify likely errors in the draft answer.",
            user=f"Task:\n{prompt}\n\nDraft answer:\n{draft}\n\nGive concise test feedback.",
            temperature=temperature,
        )
        reviewer = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:test_review",
            agent_id="test_reviewer",
            system="You revise answers using test feedback.",
            user=(
                f"Task:\n{prompt}\n\nDraft answer:\n{draft}\n\n"
                f"Test feedback:\n{tester.text}\n\nReturn the final answer."
            ),
            temperature=temperature,
        )
        return reviewer.text, {
            "calls": draft_trace["calls"]
            + [_call_payload("tester", tester), _call_payload("test_reviewer", reviewer)]
        }
    raise ValueError(f"Unknown workflow variant: {variant}")


def _run_ensemble(
    client: StandaloneOpenRouterClient,
    benchmark_name: str,
    task_id: str,
    prompt: Any,
    temperature: float,
    samples: int,
) -> tuple[str, dict[str, Any]]:
    calls = []
    answers = []
    for index in range(samples):
        result = _chat(
            client,
            task_id=f"{benchmark_name}:{task_id}:ensemble:{index}",
            agent_id=f"ensemble_{index}",
            system="You are one independent solver in an ensemble.",
            user=f"Please think step by step and then solve the task.\n\nTask:\n{prompt}",
            temperature=temperature,
        )
        answers.append(_extract_answer(result.text))
        calls.append(_call_payload(f"ensemble_{index}", result))
    judge = _chat(
        client,
        task_id=f"{benchmark_name}:{task_id}:ensemble:judge",
        agent_id="ensemble_judge",
        system="You aggregate ensemble answers into one final prediction.",
        user=f"Task:\n{prompt}\n\nCandidate answers:\n{_format_candidates(answers)}\n\nReturn the final answer.",
        temperature=temperature,
    )
    calls.append(_call_payload("ensemble_judge", judge))
    return judge.text, {"samples": samples, "answers": answers, "calls": calls}


def _chat(
    client: StandaloneOpenRouterClient,
    *,
    task_id: str,
    agent_id: str,
    system: str,
    user: str,
    temperature: float,
) -> ChatResult:
    return client.generate(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        task_id=task_id,
        agent_id=agent_id,
        temperature=temperature,
    )


def _call_payload(agent_id: str, result: ChatResult) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "text": result.text,
        "model": result.model,
        "token_in": result.token_in,
        "token_out": result.token_out,
        "mock_used": result.mock_used,
        "metadata": result.metadata,
    }


def _extract_answer(text: str) -> str:
    tag_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_match:
        return tag_match.group(1).strip()
    answer_match = re.search(
        r"(?:final answer|answer)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        return answer_match.group(1).strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _majority_vote(answers: list[str]) -> str:
    normalized = [_normalize_answer(answer) for answer in answers if answer.strip()]
    if not normalized:
        return ""
    winner, _ = Counter(normalized).most_common(1)[0]
    for answer in answers:
        if _normalize_answer(answer) == winner:
            return answer
    return winner


def _normalize_answer(answer: str) -> str:
    text = _extract_answer(answer).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff.:%/$ -]", "", text)
    return text[:500]


def _score_prediction(
    *, benchmark: Any, task: Any, prediction: str, metadata: dict[str, Any]
) -> float:
    evaluation = benchmark.evaluate(task, prediction, run_metadata=metadata)
    return float(evaluation.score)


def _choose_variant(text: str, fallback_index: int) -> str:
    normalized = text.lower()
    aliases = {
        "self_refine_once": ("self_refine_once", "self-refine", "self refine", "reflect"),
        "debate_once": ("debate_once", "debate"),
        "ensemble3": ("ensemble3", "ensemble", "self-consistency", "self consistency"),
        "test_review": ("test_review", "test review", "test"),
        "review": ("review", "revise"),
        "cot": ("cot", "chain-of-thought", "chain of thought"),
    }
    for variant in WORKFLOW_VARIANTS:
        if any(alias in normalized for alias in aliases[variant]):
            return variant
    return WORKFLOW_VARIANTS[fallback_index % len(WORKFLOW_VARIANTS)]


def _leaderboard_summary(leaderboard: list[dict[str, Any]], limit: int = 8) -> str:
    if not leaderboard:
        return "(empty)"
    ranked = sorted(leaderboard, key=lambda item: float(item["average_score"]), reverse=True)
    return "\n".join(
        f"{idx + 1}. {item['variant']} score={float(item['average_score']):.4f}"
        for idx, item in enumerate(ranked[:limit])
    )


def _format_candidates(answers: list[str]) -> str:
    return "\n\n".join(f"[{idx}] {answer}" for idx, answer in enumerate(answers)) or "(none)"


def _is_correct_signal(text: str) -> bool:
    lower = text.lower()
    false_signals = ("incorrect", "not correct", "wrong", "false")
    return "correct" in lower and not any(signal in lower for signal in false_signals)


def _resolve_benchmarks(args: argparse.Namespace) -> list[str]:
    excluded = DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)
    requested = args.benchmark or list_benchmarks()
    return [name for name in requested if name not in excluded]


def _benchmark_section_config(config_path: str | None, benchmark_name: str) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw = data.get(benchmark_name) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"[{benchmark_name}] config section must be a table when present.")
    return dict(raw)


def _load_env_file(env_file: str | None) -> None:
    if not env_file:
        return
    path = Path(env_file).expanduser()
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paper-faithful MASS baselines on repo benchmarks."
    )
    parser.add_argument("--config", default="config/experiment.example.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output-dir", default="outputs_mass_baselines")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--exclude-benchmark", action="append", default=[])
    parser.add_argument("--baseline", action="append", choices=sorted(BASELINE_SPECS), default=[])
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--openrouter-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sc-samples", type=int, default=9)
    parser.add_argument("--self-refine-rounds", type=int, default=5)
    parser.add_argument("--debate-agents", type=int, default=3)
    parser.add_argument("--debate-rounds", type=int, default=3)
    parser.add_argument("--adas-rounds", type=int, default=30)
    parser.add_argument("--adas-validation-repeats", type=int, default=3)
    parser.add_argument("--aflow-rounds", type=int, default=20)
    parser.add_argument("--aflow-validation-runs", type=int, default=5)
    parser.add_argument("--aflow-k", type=int, default=3)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    if not args.baseline:
        args.baseline = list(DEFAULT_BASELINES)
    return args


def main() -> None:
    run_baseline_suite(parse_args())


if __name__ == "__main__":
    main()

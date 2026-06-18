from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark import get_benchmark, list_benchmarks
from MAS import OpenRouterLLMClient, load_experiment_config

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = UTC


DEFAULT_EXCLUDED_BENCHMARKS = {"finance_agent"}


@dataclass
class AgentState:
    agent_id: str
    role: str
    active: bool = True
    last_answer: str = ""
    importance: float = 0.0


def main() -> None:
    args = _parse_args()
    config = load_experiment_config(args.config)
    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    benchmarks = _resolve_benchmarks(args)
    run_id = args.run_id or _now_stamp()
    output_root = Path(args.output_dir).expanduser().resolve() / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "method": "dylan_style",
        "benchmarks": {},
        "excluded_benchmarks": sorted(DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)),
        "settings": vars(args),
    }

    for benchmark_name in benchmarks:
        print(f"[{_now_stamp()}] DYLAN_BENCH_START benchmark={benchmark_name}", flush=True)
        bench_dir = output_root / benchmark_name
        bench_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload = _run_benchmark(
                benchmark_name=benchmark_name,
                config=config,
                llm_client=llm_client,
                args=args,
                output_dir=bench_dir,
            )
            summary["benchmarks"][benchmark_name] = payload
            print(
                f"[{_now_stamp()}] DYLAN_BENCH_DONE benchmark={benchmark_name} "
                f"avg_score={payload['average_score']}",
                flush=True,
            )
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
            summary["benchmarks"][benchmark_name] = payload
            _write_json(bench_dir / "error.json", payload)
            print(
                f"[{_now_stamp()}] DYLAN_BENCH_ERROR benchmark={benchmark_name} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
            if not args.keep_going:
                raise

    _write_json(output_root / "summary.json", summary)
    print(f"[{_now_stamp()}] DYLAN_RUN_DONE output={output_root}", flush=True)


def _run_benchmark(
    *,
    benchmark_name: str,
    config: Any,
    llm_client: OpenRouterLLMClient,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    benchmark = get_benchmark(
        benchmark_name, config=_benchmark_section_config(config, benchmark_name)
    )
    tasks = list(benchmark.load_tasks(task_limit=args.task_limit))
    if not tasks:
        raise RuntimeError(f"No tasks loaded for benchmark '{benchmark_name}'")

    task_payloads = []
    scores = []
    for task in tasks:
        prediction, trace = _run_dylan_task(
            llm_client=llm_client,
            task_prompt=task.prompt,
            task_id=f"{benchmark_name}:{task.task_id}",
            roles=_roles(args.roles, args.agents),
            rounds=max(1, args.rounds),
            keep_top_k=max(1, args.keep_top_k),
            consensus_threshold=args.consensus_threshold,
            model_agent_type=args.model_agent_type,
            temperature=args.temperature,
        )
        evaluation = benchmark.evaluate(
            task,
            prediction,
            run_metadata={
                "dylan_reproduce": True,
                "roles": _roles(args.roles, args.agents),
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
                "trace": trace,
            }
        )

    payload = {
        "benchmark": benchmark_name,
        "task_count": len(tasks),
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "tasks": task_payloads,
    }
    _write_json(output_dir / "dylan_results.json", payload)
    return payload


def _run_dylan_task(
    *,
    llm_client: OpenRouterLLMClient,
    task_prompt: Any,
    task_id: str,
    roles: list[str],
    rounds: int,
    keep_top_k: int,
    consensus_threshold: float,
    model_agent_type: str,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    agents = [AgentState(agent_id=f"agent_{idx}", role=role) for idx, role in enumerate(roles)]
    messages: list[dict[str, Any]] = []
    final_answer = ""
    stopped_by = "max_rounds"

    for round_index in range(rounds):
        active_agents = [agent for agent in agents if agent.active]
        for agent in active_agents:
            result = llm_client.generate(
                prompt=[
                    {
                        "role": "system",
                        "content": (
                            "You are one participant in a DyLAN-style dynamic LLM-agent network. "
                            "Solve carefully, consider peer messages, and return a concise final answer."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _agent_prompt(task_prompt, agent, round_index, messages),
                    },
                ],
                agent_type=model_agent_type,
                task_id=task_id,
                run_index=0,
                agent_id=agent.agent_id,
                temperature=temperature,
            )
            agent.last_answer = result.text
            messages.append(
                {
                    "round": round_index + 1,
                    "agent_id": agent.agent_id,
                    "role": agent.role,
                    "text": result.text,
                    "model": result.model,
                    "token_in": result.token_in,
                    "token_out": result.token_out,
                    "mock_used": result.mock_used,
                }
            )

        consensus_answer, consensus_ratio = _consensus([agent.last_answer for agent in agents])
        final_answer = consensus_answer
        _update_importance(agents, final_answer)
        if consensus_ratio >= consensus_threshold:
            stopped_by = "consensus"
            break

        if round_index >= 1:
            _activate_top_agents(agents, keep_top_k)

    if not final_answer:
        final_answer = _majority_answer([agent.last_answer for agent in agents])

    trace = {
        "rounds_requested": rounds,
        "rounds_executed": max((item["round"] for item in messages), default=0),
        "stopped_by": stopped_by,
        "messages": messages,
        "agents": [
            {
                "agent_id": agent.agent_id,
                "role": agent.role,
                "active": agent.active,
                "importance": agent.importance,
                "last_answer": agent.last_answer,
            }
            for agent in agents
        ],
        "final_answer": final_answer,
    }
    return final_answer, trace


def _agent_prompt(
    task_prompt: Any, agent: AgentState, round_index: int, messages: list[dict[str, Any]]
) -> str:
    if round_index == 0 or not messages:
        peer_context = "(no previous peer messages)"
    else:
        recent = messages[-8:]
        peer_context = "\n\n".join(
            f"{item['agent_id']} ({item['role']}): {item['text']}" for item in recent
        )
    return (
        f"Role: {agent.role}\n\n"
        f"Task:\n{task_prompt}\n\n"
        f"Previous peer messages:\n{peer_context}\n\n"
        "Return only the final answer for this task. If useful, revise based on peers."
    )


def _consensus(answers: list[str]) -> tuple[str, float]:
    normalized = [_normalize_answer(answer) for answer in answers if answer.strip()]
    if not normalized:
        return "", 0.0
    counts = Counter(normalized)
    answer, count = counts.most_common(1)[0]
    return answer, count / len(normalized)


def _majority_answer(answers: list[str]) -> str:
    non_empty = [answer for answer in answers if answer.strip()]
    if not non_empty:
        return ""
    normalized_winner, _ = _consensus(non_empty)
    for answer in reversed(non_empty):
        if _normalize_answer(answer) == normalized_winner:
            return answer
    return non_empty[-1]


def _update_importance(agents: list[AgentState], final_answer: str) -> None:
    target = _normalize_answer(final_answer)
    for agent in agents:
        if _normalize_answer(agent.last_answer) == target:
            agent.importance += 1.0


def _activate_top_agents(agents: list[AgentState], keep_top_k: int) -> None:
    ranked = sorted(
        agents, key=lambda agent: (agent.importance, bool(agent.last_answer)), reverse=True
    )
    keep = {agent.agent_id for agent in ranked[:keep_top_k]}
    for agent in agents:
        agent.active = agent.agent_id in keep


def _normalize_answer(answer: str) -> str:
    text = answer.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff.:%/$ -]", "", text)
    return text[:500]


def _roles(raw_roles: str, agent_count: int) -> list[str]:
    roles = [role.strip() for role in raw_roles.split(",") if role.strip()]
    if not roles:
        roles = ["Solver"]
    while len(roles) < agent_count:
        roles.append(roles[len(roles) % len(roles)])
    return roles[:agent_count]


def _benchmark_section_config(config: Any, benchmark_name: str) -> dict[str, Any]:
    cfg = dict(getattr(config, benchmark_name, {}) or {})
    cfg.setdefault("openrouter", {})
    if config.openrouter.api_key and "api_key" not in cfg["openrouter"]:
        cfg["openrouter"]["api_key"] = config.openrouter.api_key
    if config.openrouter.base_url and "base_url" not in cfg["openrouter"]:
        cfg["openrouter"]["base_url"] = config.openrouter.base_url
    return cfg


def _resolve_benchmarks(args: argparse.Namespace) -> list[str]:
    excluded = DEFAULT_EXCLUDED_BENCHMARKS | set(args.exclude_benchmark)
    requested = args.benchmark or list_benchmarks()
    return [name for name in requested if name not in excluded]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DyLAN-style dynamic agents on existing benchmarks."
    )
    parser.add_argument("--config", default="config/experiment.example.toml")
    parser.add_argument("--output-dir", default="outputs_dylan_reproduce")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--benchmark", action="append", default=[])
    parser.add_argument("--exclude-benchmark", action="append", default=[])
    parser.add_argument("--task-limit", type=int, default=1)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--roles", default="Solver,Critic,Verifier,Summarizer")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--keep-top-k", type=int, default=2)
    parser.add_argument("--consensus-threshold", type=float, default=0.67)
    parser.add_argument("--model-agent-type", default="default")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

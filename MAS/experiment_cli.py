from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .config import load_experiment_config
from .experiment import build_runtime_config, run_experiment


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LangGraph MAS topology experiments")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--agents", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--discussion-rounds", type=int, default=1)
    parser.add_argument("--prompt", default="Solve the task and provide a concise final answer.")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--config", default=None, help="Optional existing experiment.toml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/langgraph_topologies")
    parser.add_argument("--task-id", default="adhoc_task")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    prompt = str(args.prompt)
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")

    if args.config:
        config = load_experiment_config(args.config)
    else:
        config = build_runtime_config(
            topology=args.topology,
            agents=args.agents,
            rounds=args.rounds,
            discussion_rounds=args.discussion_rounds,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            config.openrouter.api_key = api_key

    run_stamp = _timestamp()
    run_dir = Path(args.output_dir).expanduser().resolve() / args.topology / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_path = run_dir / "run_0.trace.jsonl"
    result = run_experiment(
        topology=args.topology,
        agents=args.agents,
        prompt=prompt,
        rounds=args.rounds,
        discussion_rounds=args.discussion_rounds,
        run_index=args.run_index,
        seed=args.seed,
        config=config,
        task_id=args.task_id,
        output_trace_path=trace_path,
    )

    metadata_path = run_dir / "run_0.metadata.json"
    answer_path = run_dir / "run_0.answer.txt"

    metadata_path.write_text(
        json.dumps(result.run_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    answer_path.write_text(result.final_answer, encoding="utf-8")

    print(str(trace_path))
    print(str(metadata_path))
    print(str(answer_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

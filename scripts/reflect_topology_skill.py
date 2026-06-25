#!/usr/bin/env python3
"""Agent-maintained topology skill: offline LLM reflection over run outcomes.

The long-term playbook is a markdown skill (``config/topology_skill.md``) the planner
reads at plan time. The default writer is the *online* updater (every N runs, in
``main.py``); this script is the offline equivalent for re-reflecting over a finished
experiment. An LLM **reflection agent** reads the current skill plus per-run outcomes
labelled by PROCESS SIGNALS ONLY (``is_process_clean`` — auditor findings + decision-grade
consensus, never ``run_*.eval.json``) and rewrites the skill's "Lessons from experience"
section. Keeping the benchmark verdict out of the skill avoids biasing the study.

Usage:
    python scripts/reflect_topology_skill.py \
        --experiment-root artifacts/full_experiment/<experiment-id> \
        --config config/experiment.toml \
        [--skill config/topology_skill.md] [--benchmark <name>] [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MAS import OpenRouterLLMClient, load_experiment_config  # noqa: E402
from MAS.self_evolved.skill import (  # noqa: E402
    SkillReflector,
    TopologySkill,
    summary_from_candidate,
)
from scripts.update_topology_playbook import collect_candidates  # noqa: E402


def build_run_summaries(experiment_root: Path, *, benchmark: str | None = None) -> list[dict]:
    """Candidates -> compact, process-only run summaries.

    Shares ``summary_from_candidate`` with the online updater so offline and
    in-experiment reflection feed the reflector identical (ground-truth-free) rows."""

    return [
        summary_from_candidate(candidate)
        for candidate in collect_candidates(experiment_root, benchmark=benchmark)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--config", required=True, help="Experiment TOML (LLM client + model)")
    parser.add_argument("--skill", type=Path, default=ROOT / "config" / "topology_skill.md")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment_root = args.experiment_root.expanduser().resolve()
    if not experiment_root.is_dir():
        print(f"error: experiment root not found: {experiment_root}", file=sys.stderr)
        return 2

    summaries = build_run_summaries(experiment_root, benchmark=args.benchmark)
    if not summaries:
        print("No self_evolved run outcomes found; nothing to reflect on.")
        return 0

    config = load_experiment_config(args.config)
    llm_client = OpenRouterLLMClient(config.openrouter, config.models)
    skill = TopologySkill.load(args.skill)
    reflector = SkillReflector(llm_client, config.self_evolved)

    clean = sum(1 for s in summaries if s["process_outcome"] == "clean")
    print(f"Reflecting over {len(summaries)} run(s) ({clean} ran clean, process signals only)...")
    result = reflector.reflect(current_skill=skill.text, run_summaries=summaries)
    print(f"Reflection outcome: changed={result.changed} reason={result.reason}")
    if not result.changed:
        print("Skill left unchanged (mock LLM, no usable output, or guardrail).")
        return 0
    if args.dry_run:
        print("--- proposed skill (dry run, not written) ---")
        print(result.skill_markdown)
        return 0
    skill.save(result.skill_markdown)
    print(f"Wrote updated skill to {args.skill}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

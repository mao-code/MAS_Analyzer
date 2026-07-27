#!/usr/bin/env python3
"""Re-run PlanCraft systems pinned to bf16 providers, killing quantization-routing variance.

Why
---
`google/gemma-4-31b-it` is served by ~14 OpenRouter provider endpoints spanning bf16 (full
precision) down to fp4 (4-bit). `:nitro` routes to whichever is fastest per call, so different
runs of the same task hit different quantizations -- and the low-quant endpoints both
over-provision the topology planner AND hallucinate false "impossible" give-ups. That mixes
epochs across a system's 3 runs and inflates the between-run std.

Pinning routing to the three bf16 endpoints (WandB / Novita / Venice) and ignoring the rest
makes every run use the same quantization, which collapses the between-run std (validated:
SAS std 11.5 -> 1.9; MANTA -> ~2.8). Results are written to a NEW experiment id
(`plancraft_bf16_pinned`), so the original mixed-epoch artifacts are left untouched.

Resume-safe: main.py skips runs whose artifacts already exist, so re-running the same command
continues where a killed run left off. Each system's config is reconstructed from that system's
`experiment_settings.json:raw_config_snapshot` and written under `config/_bf16/`.

Usage
-----
    # resume / run MANTA (self_evolved); 'manta' is an alias
    .venv/bin/python scripts/experiments/rerun_plancraft_bf16.py --systems manta

    # the remaining static systems
    .venv/bin/python scripts/experiments/rerun_plancraft_bf16.py --systems only_voting,group_chat_debate,\
fully_linked_debate,orchestrator_no_discussion,orchestrator_with_discussion,\
orchestrator_tree_structure

    # everything, then summarize
    .venv/bin/python scripts/experiments/rerun_plancraft_bf16.py --systems sas,only_voting,group_chat_debate,\
fully_linked_debate,orchestrator_no_discussion,orchestrator_with_discussion,\
orchestrator_tree_structure,manta
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from rerun_plancraft_runs12 import DEFAULT_ROOTS, render_config_toml  # noqa: E402

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# The three full-precision (bf16) endpoints; everything else is ignored so a fallback can never
# land on an fp4/fp8 endpoint. Keeps the whole re-run on one quantization.
BF16_ORDER = "WandB,Novita,Venice"
BF16_IGNORE = "Chutes,DeepInfra,SiliconFlow,Parasail,Phala,ModelRun,Together,SambaNova,Cerebras"
NEW_EXPERIMENT_ID = "plancraft_bf16_pinned"
ALIAS = {"manta": "self_evolved"}


def find_system_dir(label: str) -> Path | None:
    for root in DEFAULT_ROOTS:
        p = root / "plancraft" / label
        if (p / "experiment_settings.json").exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--systems", default="self_evolved", help="Comma list; 'manta'=self_evolved.")
    ap.add_argument("--task-limit", type=int, default=30)
    ap.add_argument("--runs-per-task", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-summarize", action="store_true", help="Skip the final rollup.")
    args = ap.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (put it in .env) -- would fall back to mock.")

    labels = [ALIAS.get(s.strip().lower(), s.strip()) for s in args.systems.split(",") if s.strip()]

    env = dict(os.environ)
    env["MAS_OPENROUTER_PROVIDER_ORDER"] = BF16_ORDER
    env["MAS_OPENROUTER_IGNORE_PROVIDERS"] = BF16_IGNORE

    cfg_dir = ROOT / "config" / "_bf16"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "artifacts" / "full_experiment"

    for label in labels:
        sd = find_system_dir(label)
        if sd is None:
            print(f"[SKIP] {label}: no plancraft system with experiment_settings.json found")
            continue
        snapshot = json.loads((sd / "experiment_settings.json").read_text())["raw_config_snapshot"]
        cfg = cfg_dir / f"{label}.toml"
        cfg.write_text(render_config_toml(snapshot))
        topology = snapshot["mas"]["topology"]
        cmd = [
            sys.executable, str(ROOT / "main.py"), "run",
            "--config", str(cfg),
            "--benchmark", "plancraft",
            "--output-dir", str(out_dir),
            "--output-layout", "hierarchical",
            "--experiment-id", NEW_EXPERIMENT_ID,
            "--system-label", label,
            "--runs-per-task", str(args.runs_per_task),
            "--seed", str(args.seed),
            "--task-limit", str(args.task_limit),
        ]
        if topology == "self_evolved":
            cmd += ["--skill-update-batch-size", "0"]
        print(f"\n[RUN] {label} (bf16-pinned, resumes existing runs)\n  " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(ROOT), env=env)
        if result.returncode != 0:
            raise SystemExit(f"{label}: main.py run exited {result.returncode}")

    if not args.no_summarize:
        print("\n[SUMMARIZE]")
        subprocess.run(
            [
                sys.executable, str(ROOT / "main.py"), "summarize-experiment",
                "--experiment-root", str(out_dir / NEW_EXPERIMENT_ID),
            ],
            cwd=str(ROOT),
            env=env,
        )
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

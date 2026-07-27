#!/usr/bin/env python3
"""EXPERIMENTAL / throwaway: PlanCraft MANTA (self_evolved) with max_turns=5 + multi-repair.

Goal
----
Test whether giving the self-evolved topology loop MORE evolution turns helps PlanCraft, vs the
max_turns=2 baseline at ``artifacts/full_experiment/plancraft_bf16_skill_learning_fixed``.

Two gates in the core code make ``max_turns=5`` a no-op on its own:
  1. ``SelfEvolvedConfig.validate`` rejects ``max_turns`` outside {1, 2}.
  2. ``SelfEvolvedEngine.run`` caps repairs with ``mutations_used < 1`` -- so after ONE mutation the
     loop always stops, regardless of ``max_turns``.

This script does NOT edit core code. It relaxes both gates at RUNTIME by monkey-patching:
  1. ``validate``      : allow ``max_turns`` up to 10.
  2. ``run``           : mutation budget ``mutations_used < 1`` -> ``< (max_turns - 1)`` (<=4 repairs).
Each patch is applied via ``inspect.getsource`` + a checked ``str.replace`` and then asserted to have
taken effect, so if the core source ever drifts this script fails LOUDLY instead of silently running
the unpatched baseline and wasting API budget.

Everything else is reconstructed verbatim from the baseline's ``raw_config_snapshot`` (same model,
judge, agents, plancraft settings). Results go to a NEW experiment id so the baseline is untouched.
bf16 provider pinning matches the baseline's naming (WandB / Novita / Venice).

Usage
-----
    .venv/bin/python scripts/experiments/exp_plancraft_maxturns5.py --dry-run        # verify patches only
    .venv/bin/python scripts/experiments/exp_plancraft_maxturns5.py                  # run 15 tasks x 1 run
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from rerun_plancraft_runs12 import render_config_toml  # noqa: E402

BASELINE = ROOT / "artifacts/full_experiment/plancraft_bf16_skill_learning_fixed/plancraft/self_evolved"
NEW_EXPERIMENT_ID = "plancraft_maxturns5_multirepair"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-maoxunhuang-Desktop-MAS-Analyzer/"
    "2d78b8b0-e7dd-4ab4-802e-41a597f345a1/scratchpad"
)
# bf16 endpoints (full precision) — matches the baseline's `bf16` naming; everything else ignored so
# a fallback can never land on an fp4/fp8 endpoint (see scripts/experiments/rerun_plancraft_bf16.py).
BF16_ORDER = "WandB,Novita,Venice"
BF16_IGNORE = "Chutes,DeepInfra,SiliconFlow,Parasail,Phala,ModelRun,Together,SambaNova,Cerebras"
MAX_TURNS = 5


def _patch_method(cls, name: str, replacements, module_globals: dict) -> str:
    """Rebind ``cls.name`` from its own source with checked string replacements.

    ``replacements`` is a list of ``(old, new, expected_count)``. Each ``old`` must occur exactly
    ``expected_count`` times or we raise — a drift guard so we never run an unpatched engine.
    """
    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    for old, new, expected in replacements:
        found = src.count(old)
        if found != expected:
            raise SystemExit(
                f"PATCH ABORT: {cls.__name__}.{name}: expected {expected}x {old!r}, found {found}. "
                "Core source drifted — refusing to run an unpatched (baseline-equivalent) engine."
            )
        src = src.replace(old, new)
    ns: dict = {}
    exec(compile(src, f"<patched {cls.__name__}.{name}>", "exec"), module_globals, ns)  # noqa: S102
    setattr(cls, name, ns[name])
    return src


def apply_patches() -> None:
    import MAS.config as cfgmod
    import MAS.self_evolved.engine as engmod

    # 1. Allow max_turns up to 10 (core: `if not 1 <= self.max_turns <= 2:`).
    _patch_method(
        cfgmod.SelfEvolvedConfig,
        "validate",
        [("1 <= self.max_turns <= 2", "1 <= self.max_turns <= 10", 1)],
        cfgmod.__dict__,
    )
    # 2. Repair budget: up to (max_turns - 1) trace-backed mutations instead of exactly 1.
    new_src = _patch_method(
        engmod.SelfEvolvedEngine,
        "run",
        [("mutations_used < 1", "mutations_used < (int(self.se_config.max_turns) - 1)", 1)],
        engmod.__dict__,
    )

    # --- assert the patches actually took effect ---
    cfgmod.SelfEvolvedConfig(max_turns=MAX_TURNS).validate()  # would raise on the old bound
    if "int(self.se_config.max_turns) - 1" not in new_src:
        raise SystemExit("PATCH ABORT: engine mutation-budget replacement not present in new source.")
    print("[patch] SelfEvolvedConfig.validate: max_turns bound relaxed to <= 10  ✓")
    print("[patch] SelfEvolvedEngine.run: repair budget = max_turns - 1 (=", MAX_TURNS - 1, ") ✓")


def build_config(task_limit: int, runs_per_task: int) -> Path:
    settings = json.loads((BASELINE / "experiment_settings.json").read_text())
    snap = settings["raw_config_snapshot"]
    snap.setdefault("experiment", {})
    snap["experiment"]["task_limit"] = task_limit
    snap["experiment"]["runs_per_task"] = runs_per_task
    snap.setdefault("mas", {})["max_turns"] = MAX_TURNS  # unused by SE loop, kept consistent
    se = snap.setdefault("self_evolved", {})
    se["max_turns"] = MAX_TURNS
    se["skill_update_batch_size"] = 0  # no online skill drift; isolate the topology-evolution lever
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cfg = SCRATCH / "exp_maxturns5_self_evolved.toml"
    cfg.write_text(render_config_toml(snap))
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-limit", type=int, default=15)
    ap.add_argument("--runs-per-task", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Apply+verify patches and build config, then exit.")
    args = ap.parse_args()

    if not (BASELINE / "experiment_settings.json").exists():
        raise SystemExit(f"Baseline settings not found under {BASELINE}")

    apply_patches()
    cfg = build_config(args.task_limit, args.runs_per_task)
    print(f"[config] wrote {cfg}")

    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (put it in .env) — would fall back to mock.")

    os.environ["MAS_OPENROUTER_PROVIDER_ORDER"] = BF16_ORDER
    os.environ["MAS_OPENROUTER_IGNORE_PROVIDERS"] = BF16_IGNORE

    argv = [
        "run",
        "--config", str(cfg),
        "--benchmark", "plancraft",
        "--output-dir", str(ROOT / "artifacts" / "full_experiment"),
        "--output-layout", "hierarchical",
        "--experiment-id", NEW_EXPERIMENT_ID,
        "--system-label", "self_evolved",
        "--runs-per-task", str(args.runs_per_task),
        "--seed", str(args.seed),
        "--task-limit", str(args.task_limit),
        "--skill-update-batch-size", "0",
    ]
    print("[run] main.py " + " ".join(argv))
    if args.dry_run:
        print("[dry-run] patches verified, config built; not executing the experiment.")
        return 0

    import main as cli  # imported after patches; runner picks up the patched engine class

    rc = cli.main(argv)
    if rc != 0:
        raise SystemExit(f"main.py run exited {rc}")

    print("\n[summarize]")
    cli.main(["summarize-experiment", "--experiment-root", str(ROOT / "artifacts" / "full_experiment" / NEW_EXPERIMENT_ID)])
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

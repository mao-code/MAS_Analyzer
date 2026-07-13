#!/usr/bin/env python3
"""Re-run PlanCraft runs 1 and 2 (keep run 0), then re-aggregate the experiment summaries.

Why this exists
---------------
PlanCraft's `get_environment()` handed the *cached* dataset example straight to
`PlancraftEnvironment`, which stores inventory item dicts **by reference** and mutates
their quantities in place during play (`plancraft/environment/env.py` reset/set_quantity).
Because the examples list is cached for the whole process, the first run of a task
depleted the shared inventory and every later run of the same task started from a
corrupted (partially consumed) world -- so run 0 scored ~83% while runs 1 and 2 were
byte-identical at ~50% (the agent correctly reported "impossible" for a world that no
longer matched the gold label). `benchmark/plancraft/__init__.py` now deep-copies the
example per environment, so runs are independent going forward.

This script repairs the already-written artifacts. Run 0 was clean, so it is kept and
re-used via the harness resume path; only runs 1 and 2 are re-executed with the fix, and
then per-system (`summary.*`, per-task `descriptor/analysis/task_summary`) and
experiment-level (`experiment_summary.*`, `benchmark_summary.*`) rollups are regenerated.

Each system is re-run with the EXACT config it originally used, reconstructed from that
system's `experiment_settings.json:raw_config_snapshot` (the current
`scripts/full_experiment.py` SYSTEMS matrix no longer matches these 2026-04 artifacts).

Usage
-----
    # live re-run (needs a real key; env wins over the redacted snapshot value)
    OPENROUTER_API_KEY=sk-... .venv/bin/python scripts/rerun_plancraft_runs12.py

    # preview only -- no deletion, no runs, no summarize
    .venv/bin/python scripts/rerun_plancraft_runs12.py --dry-run

    # limit / customise
    .venv/bin/python scripts/rerun_plancraft_runs12.py --jobs 4 --runs 1,2
    .venv/bin/python scripts/rerun_plancraft_runs12.py --roots artifacts/full_experiment/<id>

    # re-run only specific systems (runs 1&2). 'manta' == self_evolved.
    .venv/bin/python scripts/rerun_plancraft_runs12.py --systems sas,manta
    # ...and force a redo even though they are already repaired:
    .venv/bin/python scripts/rerun_plancraft_runs12.py --systems sas,manta --force
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The API key usually lives in .env (auto-loaded by MAS/config.py inside the subprocess).
# Load it here too so the pre-flight guard sees it and every child inherits it explicitly.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Only roots that actually contain a plancraft/ tree are touched; others are skipped with
# a note (full_selfevo_bw has no plancraft, so it is a no-op even though listed).
DEFAULT_ROOTS = [
    ROOT / "artifacts/full_experiment/20260427T134706Z__google_gemma_4_31b_it_nitro",
    ROOT / "artifacts/full_experiment/full_selfevo_bw__google_gemma_4_31b_it_nitro",
    ROOT / "artifacts/full_experiment/full_selfevo_ps__google_gemma_4_31b_it_nitro",
]

BENCHMARK = "plancraft"


# --------------------------------------------------------------------------- TOML output
def _toml_scalar(value: object) -> str | None:
    """Serialise a scalar/list to a TOML literal. Returns None if it must be omitted."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (list, tuple)):
        parts = [_toml_scalar(v) for v in value]
        if any(p is None for p in parts):
            return None
        return "[" + ", ".join(p for p in parts if p is not None) + "]"
    return None


def render_config_toml(snapshot: dict) -> str:
    """Re-serialise a raw_config_snapshot into a TOML string.

    Null values, empty tables, and the redacted openrouter.api_key are dropped: the api
    key is supplied from the environment (OPENROUTER_API_KEY wins in load config), and
    absent keys fall back to the same code defaults that produced the snapshot.
    """
    lines: list[str] = []
    for section, body in snapshot.items():
        if not isinstance(body, dict) or not body:
            continue
        rendered: list[str] = []
        for key, value in body.items():
            if section == "openrouter" and key == "api_key":
                continue  # taken from env; snapshot value is redacted
            literal = _toml_scalar(value)
            if literal is None:
                continue
            rendered.append(f"{key} = {literal}")
        if not rendered:
            continue
        lines.append(f"[{section}]")
        lines.extend(rendered)
        lines.append("")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- system repair
def find_task_dirs(system_dir: Path) -> list[Path]:
    """Immediate subdirs that are real per-task output dirs (have a run_0 answer)."""
    return sorted(
        p.parent for p in system_dir.glob("*/run_0.answer.txt") if p.parent.is_dir()
    )


def _rerun_complete(task_dir: Path, idx: int, cutoff: float) -> bool:
    """True if run <idx> is a complete run re-executed AFTER the fix (mtime >= cutoff)."""
    req = [
        task_dir / f"run_{idx}.answer.txt",
        task_dir / f"run_{idx}.metadata.json",
        task_dir / f"run_{idx}.eval.json",
        task_dir / f"run_{idx}.trace.jsonl",
    ]
    if not all(p.exists() for p in req):
        return False
    return (task_dir / f"run_{idx}.metadata.json").stat().st_mtime >= cutoff


def purge_reruns(
    task_dir: Path, run_indices: list[int], cutoff: float, *, dry_run: bool, force: bool = False
) -> int:
    """Invalidate ONLY the stale/partial reran indices so the harness re-executes them.

    Idempotent across kill/relaunch: a rerun index that is already complete AND was written
    after `cutoff` (the fix time) is kept, so its fresh result is reused via harness resume
    instead of being deleted and recomputed. Run 0 is never touched. Returns files removed.

    `force=True` bypasses the idempotency skip: every reran index (fresh or not) is deleted
    and re-executed. Use it to redo already-repaired systems from scratch.
    """
    fresh = {idx: (not force and _rerun_complete(task_dir, idx, cutoff)) for idx in run_indices}
    ts = task_dir / "task_summary.json"
    ts_fresh = ts.exists() and ts.stat().st_mtime >= cutoff
    # Fully repaired already: leave everything so main.py's task-resume skips this task.
    if not force and ts_fresh and all(fresh.values()):
        return 0

    removed = 0
    # Drop task_summary so the task is re-aggregated after the runs settle (a stale/original
    # one would otherwise let task-resume skip a half-repaired task).
    if ts.exists():
        removed += 1
        if not dry_run:
            ts.unlink()
    for idx in run_indices:
        if fresh[idx]:
            continue  # keep a completed post-fix rerun; harness per-run resume reuses it
        # trailing dot so run_1.* never matches run_10.* (defensive; max index is 2)
        for path in task_dir.glob(f"run_{idx}.*"):
            removed += 1
            if not dry_run:
                path.unlink()
    return removed


def rerun_system(
    system_dir: Path,
    config_dir: Path | None,
    run_indices: list[int],
    cutoff: float,
    *,
    dry_run: bool,
    force: bool = False,
) -> None:
    settings_path = system_dir / "experiment_settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    snapshot = settings["raw_config_snapshot"]
    runtime = settings.get("runtime", {})
    bench = settings.get("benchmark", {})

    # Derive experiment_id / output_dir from the ON-DISK path, NOT from settings: for
    # systems that used a non-nitro model the recorded experiment_id embeds the model slug
    # (e.g. ...gemma_4_31b_it) and differs from the actual folder (...gemma_4_31b_it_nitro),
    # so trusting settings would write a brand-new folder instead of replacing artifacts.
    # system_dir == <output_dir>/<experiment_id>/plancraft/<system_label>.
    experiment_id = system_dir.parents[1].name
    output_dir = str(system_dir.parents[2])
    system_label = system_dir.name
    topology = snapshot["mas"]["topology"]
    seed = int(runtime.get("seed", snapshot.get("experiment", {}).get("seed", 42)))
    runs_per_task = int(runtime.get("runs_per_task", 3))
    task_dirs = find_task_dirs(system_dir)
    task_limit = int(bench.get("task_limit") or len(task_dirs) or 0)

    max_rerun = max(run_indices)
    if runs_per_task <= max_rerun:
        raise SystemExit(
            f"{system_label}: runs_per_task={runs_per_task} cannot cover rerun index {max_rerun}"
        )

    removed = sum(purge_reruns(td, run_indices, cutoff, dry_run=dry_run, force=force) for td in task_dirs)
    kept = sum(1 for td in task_dirs if all(_rerun_complete(td, i, cutoff) for i in run_indices))
    print(
        f"  [{system_label}] topology={topology} tasks={len(task_dirs)} "
        f"seed={seed} runs_per_task={runs_per_task} kept=run_0 rerun={run_indices} "
        f"{'force=on ' if force else ''}already_repaired={kept} purged_files={removed}"
    )

    # Default: write the reconstructed config next to the artifacts it describes (under the
    # gitignored experiment root) so experiment_settings.json:config_path resolves and git
    # stays clean. A global --config-dir overrides this.
    cfg_dir = config_dir if config_dir is not None else (system_dir.parents[1] / "_rerun_configs")
    config_path = cfg_dir / f"{system_label}.toml"
    if not dry_run:
        cfg_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(render_config_toml(snapshot), encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "run",
        "--config", str(config_path),
        "--benchmark", BENCHMARK,
        "--output-dir", str(output_dir),
        "--output-layout", "hierarchical",
        "--experiment-id", experiment_id,
        "--system-label", system_label,
        "--runs-per-task", str(runs_per_task),
        "--seed", str(seed),
        "--task-limit", str(task_limit),
    ]
    if topology == "self_evolved":
        # Do not rewrite config/topology_skill.md mid-rerun; keep the run deterministic.
        cmd += ["--skill-update-batch-size", "0"]

    if dry_run:
        print("    DRY-RUN cmd:", " ".join(cmd))
        return

    print("    RUN:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"{system_label}: main.py run exited {result.returncode}")


def summarize_root(root: Path, *, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "summarize-experiment",
        "--experiment-root", str(root),
    ]
    if dry_run:
        print("  DRY-RUN summarize:", " ".join(cmd))
        return
    print("  SUMMARIZE:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"summarize-experiment failed for {root} (exit {result.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="*", default=None, help="Override experiment roots to repair.")
    parser.add_argument("--runs", default="1,2", help="Comma-separated run indices to re-execute (default 1,2).")
    parser.add_argument(
        "--systems",
        default=None,
        help="Comma-separated system labels to re-run (default: all). 'manta' is an alias for "
        "self_evolved. Example: --systems sas,manta",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run runs 1&2 even if already repaired (purge fresh results and re-execute). "
        "Without this, already-repaired systems are skipped via harness resume.",
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel systems per root (default 4).")
    parser.add_argument("--config-dir", default=None, help="Where to write reconstructed configs.")
    parser.add_argument(
        "--since",
        default=None,
        help="Freshness cutoff (epoch seconds). A rerun already completed at/after this time is "
        "kept (resume). Default: the mtime of the fixed benchmark/plancraft/__init__.py, so "
        "original artifacts are purged and post-fix reruns survive a kill/relaunch.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan only; no deletion, no runs.")
    args = parser.parse_args()

    run_indices = sorted({int(x) for x in args.runs.split(",") if x.strip() != ""})
    if not run_indices or min(run_indices) < 1:
        raise SystemExit("--runs must be positive indices (run 0 is kept); e.g. --runs 1,2")

    # Freshness cutoff: original PlanCraft artifacts predate the fix; every post-fix rerun
    # postdates it. Using the fixed adapter file's mtime cleanly separates the two so a
    # kill/relaunch keeps completed reruns instead of purging and recomputing them.
    cutoff = float(args.since) if args.since else (ROOT / "benchmark" / "plancraft" / "__init__.py").stat().st_mtime

    if not args.dry_run and not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set -- a live re-run would fall back to mock mode. "
            "Export it (or put it in .env) before running, or pass --dry-run."
        )

    roots = [Path(r).resolve() for r in args.roots] if args.roots else DEFAULT_ROOTS
    # None => per-root <root>/_rerun_configs/ (default); a value => single shared dir.
    config_dir = Path(args.config_dir).resolve() if args.config_dir else None

    # System selection: 'manta' is the human-facing name for the self_evolved topology.
    system_aliases = {"manta": "self_evolved"}
    wanted_systems: set[str] | None = None
    if args.systems:
        wanted_systems = {
            system_aliases.get(s.strip().lower(), s.strip())
            for s in args.systems.split(",")
            if s.strip()
        }

    for root in roots:
        bench_dir = root / BENCHMARK
        if not bench_dir.is_dir():
            print(f"[SKIP] {root.name}: no {BENCHMARK}/ subtree")
            continue
        system_dirs = sorted(
            p for p in bench_dir.iterdir() if p.is_dir() and (p / "experiment_settings.json").exists()
        )
        if wanted_systems is not None:
            system_dirs = [p for p in system_dirs if p.name in wanted_systems]
        if not system_dirs:
            reason = (
                f"none of the requested systems {sorted(wanted_systems)}"
                if wanted_systems is not None
                else "no plancraft systems with experiment_settings.json"
            )
            print(f"[SKIP] {root.name}: {reason}")
            continue

        print(f"\n=== {root.name}: {len(system_dirs)} plancraft system(s) ===")
        errors: list[str] = []
        if args.jobs > 1 and len(system_dirs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futs = {
                    pool.submit(
                        rerun_system, sd, config_dir, run_indices, cutoff, dry_run=args.dry_run, force=args.force
                    ): sd
                    for sd in system_dirs
                }
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{futs[fut].name}: {exc}")
        else:
            for sd in system_dirs:
                try:
                    rerun_system(sd, config_dir, run_indices, cutoff, dry_run=args.dry_run, force=args.force)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{sd.name}: {exc}")

        if errors:
            print(f"[ERROR] {root.name}: {len(errors)} system(s) failed:")
            for e in errors:
                print("   -", e)
            print("  Skipping summarize for this root; fix and re-run the failed systems.")
            continue

        summarize_root(root, dry_run=args.dry_run)

    print("\nDone." + (" (dry-run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

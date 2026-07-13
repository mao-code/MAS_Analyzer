#!/usr/bin/env python
"""Redo WorkBench runs that FAILED because the OpenRouter provider returned an
empty completion (``failure_category == "empty_completion"``), in place, in the
same experiment folder.

These runs are NOT flagged ``needs_rerun`` by the harness (an empty completion is
treated as a clean stop), so the normal resume skips them. This script forces a
targeted redo by deleting exactly those runs' artifacts and invalidating the two
summaries that gate the skip logic:

  * per run  : delete ``run_<n>.*`` files            -> run-level resume re-executes it
  * per task : delete ``task_summary.json``          -> task-level resume falls to per-run loop
  * per system: delete ``summary.json`` + ``.csv``   -> batch driver stops skipping the system

Good runs are untouched and reused on resume. After running this with --apply,
resume with:

    KEEP_OLD=1 TARGETS=static bash scripts/rerun_workbench_crm_fix.sh

Dry-run by default (prints what it WOULD delete). Pass --apply to delete.

Note: same folder => same model (google/gemma-4-31b-it:nitro), which is the source
of the empty completions. Re-runs get fresh attempts, but ~8.7% of generations are
empty, so a single pass won't clear all of them for the high-call-count MAS systems.
Run this + resume 2-3 times, or harden routing on resume (see script epilogue).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ROOT = Path(
    "artifacts/full_experiment/20260427T134706Z__google_gemma_4_31b_it_nitro/workbench"
)
STATIC_SYSTEMS = [
    "sas",
    "only_voting",
    "orchestrator_no_discussion",
    "orchestrator_with_discussion",
    "orchestrator_tree_structure",
    "fully_linked_debate",
    "group_chat_debate",
]
RUN_SUFFIXES = [
    "answer.txt",
    "metadata.json",
    "result.json",
    "raw.json",
    "trace.jsonl",
    "eval.json",
    "trace_metrics.json",
    "trajectory.json",
    "trajectory.md",
]


def _walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _run_hit_empty_completion(metadata_path: Path) -> bool:
    """True if this run's tool loop was cut off by a provider empty completion."""
    try:
        md = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for node in _walk(md):
        if not isinstance(node, dict):
            continue
        meta = node.get("metadata")
        if isinstance(meta, dict) and meta.get("failure_category") == "empty_completion":
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-root", default=str(DEFAULT_ROOT), help="workbench/ folder to repair")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument(
        "--include-succeeded",
        action="store_true",
        help="also redo runs that had an empty completion but still succeeded (default: only redo FAILED ones)",
    )
    args = ap.parse_args()

    root = Path(args.experiment_root)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    total_runs = 0
    affected_systems: set[str] = set()
    affected_tasks: set[Path] = set()
    to_delete: list[Path] = []

    for system in STATIC_SYSTEMS:
        sysdir = root / system
        if not sysdir.is_dir():
            continue
        for taskdir in sorted(sysdir.glob("multi_domain_*")):
            for eval_path in sorted(taskdir.glob("run_*.eval.json")):
                run_index = eval_path.name.split("_")[1].split(".")[0]
                md_path = taskdir / f"run_{run_index}.metadata.json"
                if not md_path.exists():
                    continue
                try:
                    succeeded = bool(json.loads(eval_path.read_text()).get("success"))
                except Exception:
                    succeeded = False
                if succeeded and not args.include_succeeded:
                    continue
                if not _run_hit_empty_completion(md_path):
                    continue
                total_runs += 1
                affected_systems.add(system)
                affected_tasks.add(taskdir)
                for suffix in RUN_SUFFIXES:
                    p = taskdir / f"run_{run_index}.{suffix}"
                    if p.exists():
                        to_delete.append(p)

    # per-task and per-system invalidation files
    for taskdir in affected_tasks:
        p = taskdir / "task_summary.json"
        if p.exists():
            to_delete.append(p)
    for system in affected_systems:
        for name in ("summary.json", "summary.csv"):
            p = root / system / name
            if p.exists():
                to_delete.append(p)

    mode = "APPLY (deleting)" if args.apply else "DRY RUN (nothing deleted)"
    print(f"=== redo empty-completion runs — {mode} ===")
    print(f"root: {root}")
    per_sys: dict[str, int] = {}
    for system in STATIC_SYSTEMS:
        n = sum(
            1
            for t in affected_tasks
            if t.parent.name == system
        )
        if n:
            per_sys[system] = n
    print(f"error runs to redo: {total_runs}")
    print(f"affected tasks: {len(affected_tasks)}  affected systems: {len(affected_systems)}")
    print("affected tasks per system:")
    for system in STATIC_SYSTEMS:
        cnt = sum(1 for t in affected_tasks if t.parent.name == system)
        if cnt:
            print(f"   {system:32s} {cnt} tasks")
    print(f"files to delete: {len(to_delete)}")

    if args.apply:
        for p in to_delete:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        print("\nDeleted. Now resume (same folder, re-executes only the deleted runs):")
        print("   KEEP_OLD=1 TARGETS=static bash scripts/rerun_workbench_crm_fix.sh")
    else:
        print("\n(dry run) re-run with --apply to delete, then resume with:")
        print("   KEEP_OLD=1 TARGETS=static bash scripts/rerun_workbench_crm_fix.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

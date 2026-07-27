#!/usr/bin/env python3
"""Pre-seed + report helper for the browsecomp read-net fix-validation re-run.

This is a DIAGNOSTIC tool, not an experiment runner. It supports re-running only
the previously-failed browsecomp self_evolved runs with the read-net fix applied,
into a SEPARATE folder, and then reporting whether they flip to success -- broken
down by failure category so the real fix signal is not conflated with resampling
noise.

Two subcommands:

  preseed   Copy the PASSING runs of each failed task from the source (headline)
            run into the destination (fix-check) folder so the harness's resume
            logic skips them, and OMIT the failed runs so the harness re-executes
            exactly those with the fixed code. Task-level summary/descriptor files
            are intentionally not copied, so the per-run resume path is taken.

  report    Compare each previously-failed run's old verdict (source) against its
            re-executed verdict (destination), grouped by failure category:

              fix_target  : null "UNSUPPORTED" answer AND gold fully retrieved
                            (recall_gold == 1.0)  -> the read-net fix targets these;
                            a flip here is real fix signal.
              gold_wrong  : gold fully retrieved but a wrong (non-null) answer
                            -> reasoning miss; the finalize fix cannot help. A flip
                            here is resampling luck.
              low_recall  : gold not fully retrieved -> search/coverage miss; the
                            finalize fix cannot help. A flip here is resampling luck.

IMPORTANT (research integrity):
  * The destination folder is a fix-VALIDATION scratch folder. Because `preseed`
    copies old-code passing runs alongside newly re-executed failed runs, the
    destination's own summary.csv / descriptor is a MIX of two code versions and is
    NOT a valid success rate. This reporter -- which reads only the re-executed
    failed runs and labels every flip by category -- is the authoritative output.
  * Do NOT splice the recovered failures back into the headline run's success rate.
    That is a one-directional ratchet (it can only go up) and is not defensible.
  * For a paper number, re-run the FULL task set with the fix into a fresh folder
    and report that, with the same code version across every run.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

UNSUPPORTED_MARKER = "Unable to determine a supported final answer"

DEFAULT_SRC = (
    ROOT
    / "artifacts/full_experiment/full_selfevo_bw__google_gemma_4_31b_it_nitro"
    / "browsecomp/self_evolved"
)
DEFAULT_DST = (
    ROOT
    / "artifacts/full_experiment/full_selfevo_bw_fixcheck__google_gemma_4_31b_it_nitro"
    / "browsecomp/self_evolved"
)

# Per-run artifact files written by main.py (run_<n>.*). Used to copy passers and to
# detect/clear failed runs.
RUN_SUFFIXES = (
    ".answer.txt",
    ".eval.json",
    ".metadata.json",
    ".raw.json",
    ".result.json",
    ".trace.jsonl",
    ".trace_metrics.json",
    ".trajectory.json",
    ".trajectory.md",
)

# Task-level files that, if present, would let the harness skip the whole task or
# present a (now stale / mixed) aggregate. Never copy these into the fix-check folder.
TASK_LEVEL_FILES = ("task_summary.json", "descriptor.json", "descriptor.csv", "analysis.json")


def _find_recall_gold(obj: object) -> float | None:
    """Recursively search a parsed trace-metrics object for a recall_gold value."""
    if isinstance(obj, dict):
        if "recall_gold" in obj:
            try:
                return float(obj["recall_gold"])
            except (TypeError, ValueError):
                return None
        for value in obj.values():
            found = _find_recall_gold(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_recall_gold(item)
            if found is not None:
                return found
    return None


def _recall_gold(task_dir: Path, run: str) -> float | None:
    tm = task_dir / f"{run}.trace_metrics.json"
    if not tm.exists():
        return None
    try:
        return _find_recall_gold(json.loads(tm.read_text(encoding="utf-8")))
    except Exception:
        # Fall back to a tolerant regex on the raw text.
        m = re.search(r'"recall_gold"\s*:\s*([0-9.]+)', tm.read_text(encoding="utf-8"))
        return float(m.group(1)) if m else None


def _is_null_answer(task_dir: Path, run: str) -> bool:
    ans = task_dir / f"{run}.answer.txt"
    if not ans.exists():
        return False
    return UNSUPPORTED_MARKER in ans.read_text(encoding="utf-8")


def _eval_success(eval_path: Path) -> bool | None:
    if not eval_path.exists():
        return None
    try:
        return bool(json.loads(eval_path.read_text(encoding="utf-8")).get("success", False))
    except Exception:
        return None


def _answer_preview(task_dir: Path, run: str, width: int = 90) -> str:
    ans = task_dir / f"{run}.answer.txt"
    if not ans.exists():
        return "<no answer file>"
    text = re.sub(r"\s+", " ", ans.read_text(encoding="utf-8")).strip()
    return (text[:width] + "…") if len(text) > width else text


def _category(task_dir: Path, run: str) -> str:
    recall = _recall_gold(task_dir, run)
    null_answer = _is_null_answer(task_dir, run)
    if null_answer and recall == 1.0:
        return "fix_target"
    if recall == 1.0:
        return "gold_wrong"
    return "low_recall"


def _enumerate_failures(src: Path) -> list[dict]:
    """Return every failed run under src as dicts with task, run, category."""
    failures: list[dict] = []
    for eval_path in sorted(src.glob("*/run_*.eval.json")):
        if _eval_success(eval_path):
            continue
        task_dir = eval_path.parent
        run = eval_path.name[: -len(".eval.json")]
        failures.append(
            {
                "task": task_dir.name,
                "run": run,
                "category": _category(task_dir, run),
                "recall_gold": _recall_gold(task_dir, run),
                "null_answer": _is_null_answer(task_dir, run),
            }
        )
    return failures


def _passing_runs_by_task(src: Path) -> dict[str, set[str]]:
    """Map task id -> set of run names (run_0, ...) that PASSED in the source."""
    passing: dict[str, set[str]] = {}
    for eval_path in sorted(src.glob("*/run_*.eval.json")):
        if not _eval_success(eval_path):
            continue
        task = eval_path.parent.name
        run = eval_path.name[: -len(".eval.json")]
        passing.setdefault(task, set()).add(run)
    return passing


# ---------------------------------------------------------------------------
# preseed
# ---------------------------------------------------------------------------
def cmd_preseed(args: argparse.Namespace) -> int:
    src: Path = args.src
    dst: Path = args.dst
    if not src.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 2

    failures = _enumerate_failures(src)
    failed_tasks = sorted({f["task"] for f in failures})
    passing = _passing_runs_by_task(src)

    print(f"[preseed] src={src}")
    print(f"[preseed] dst={dst}")
    print(f"[preseed] failed runs={len(failures)} across {len(failed_tasks)} tasks")
    print(f"[preseed] failed task ids: {','.join(failed_tasks)}")

    copied_passers = 0
    cleared_failures = 0
    for task in failed_tasks:
        src_task = src / task
        dst_task = dst / task
        dst_task.mkdir(parents=True, exist_ok=True)

        # Always carry the task manifest so the re-run sees the same task payload.
        manifest = src_task / "task.json"
        if manifest.exists():
            shutil.copy2(manifest, dst_task / "task.json")

        # Copy PASSING runs only -> resume skips them (no re-execution, no cost).
        for run in sorted(passing.get(task, set())):
            for suffix in RUN_SUFFIXES:
                f = src_task / f"{run}{suffix}"
                if f.exists():
                    shutil.copy2(f, dst_task / f.name)
            copied_passers += 1

        # Ensure FAILED runs are absent in dst -> harness re-executes them with the fix.
        failed_runs = {f["run"] for f in failures if f["task"] == task}
        for run in failed_runs:
            removed_any = False
            for suffix in RUN_SUFFIXES:
                f = dst_task / f"{run}{suffix}"
                if f.exists():
                    f.unlink()
                    removed_any = True
            cleared_failures += 1
            if removed_any:
                print(f"[preseed]   cleared stale {task}/{run}.* in dst")

        # Never leave task-level aggregates that would trigger task-resume / a mixed summary.
        for name in TASK_LEVEL_FILES:
            stale = dst_task / name
            if stale.exists():
                stale.unlink()

    print(
        f"[preseed] pre-seeded {copied_passers} passing run(s); "
        f"{cleared_failures} failed run(s) will re-execute with the fix."
    )
    print(f"[preseed] TASK_IDS={','.join(failed_tasks)}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
CATEGORY_ORDER = ["fix_target", "gold_wrong", "low_recall"]
CATEGORY_BLURB = {
    "fix_target": "null answer + gold retrieved -> read-net fix targets these (flip = REAL signal)",
    "gold_wrong": "gold retrieved but wrong answer -> reasoning miss (flip = resampling noise)",
    "low_recall": "gold not retrieved -> search/coverage miss (flip = resampling noise)",
}


def cmd_report(args: argparse.Namespace) -> int:
    src: Path = args.src
    dst: Path = args.dst
    if not src.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 2
    if not dst.exists():
        print(f"ERROR: fix-check folder not found (run preseed + the re-run first): {dst}", file=sys.stderr)
        return 2

    failures = _enumerate_failures(src)

    print("=" * 92)
    print("BROWSECOMP READ-NET FIX-VALIDATION REPORT (diagnostic, NOT a headline success rate)")
    print("=" * 92)
    print(f"source (old code):  {src}")
    print(f"fixcheck (new code):{dst}")
    print()

    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORY_ORDER}
    not_rerun: list[dict] = []
    for f in failures:
        dst_eval = dst / f["task"] / f"{f['run']}.eval.json"
        new_success = _eval_success(dst_eval)
        if new_success is None:
            not_rerun.append(f)
            continue
        f = {**f, "new_success": new_success}
        by_cat[f["category"]].append(f)

    grand_flips = 0
    grand_total = 0
    for cat in CATEGORY_ORDER:
        rows = by_cat[cat]
        if not rows:
            continue
        flips = sum(1 for r in rows if r["new_success"])
        grand_flips += flips
        grand_total += len(rows)
        print(f"── {cat}  ({flips}/{len(rows)} now pass) — {CATEGORY_BLURB[cat]}")
        for r in sorted(rows, key=lambda x: (x["task"], x["run"])):
            flag = "PASS ✓" if r["new_success"] else "fail ✗"
            rg = "—" if r["recall_gold"] is None else f"{r['recall_gold']:.2f}"
            preview = _answer_preview(dst / r["task"], r["run"])
            print(f"     {r['task']}/{r['run']}  recall_gold={rg}  old=fail -> new={flag}  | {preview}")
        print()

    if not_rerun:
        print(f"── not re-run yet ({len(not_rerun)}): no eval found in fixcheck folder")
        for r in sorted(not_rerun, key=lambda x: (x["task"], x["run"])):
            print(f"     {r['task']}/{r['run']}  ({r['category']})")
        print()

    fix_rows = by_cat["fix_target"]
    fix_flips = sum(1 for r in fix_rows if r["new_success"])
    noise_rows = by_cat["gold_wrong"] + by_cat["low_recall"]
    noise_flips = sum(1 for r in noise_rows if r["new_success"])

    print("-" * 92)
    print("SUMMARY")
    print(
        f"  REAL fix signal (fix_target): {fix_flips}/{len(fix_rows)} previously-null runs "
        f"now produce a supported answer that passes."
    )
    print(
        f"  Resampling noise (gold_wrong + low_recall): {noise_flips}/{len(noise_rows)} flipped — "
        f"the finalize fix cannot affect these; treat as regression-to-the-mean, not a fix effect."
    )
    print(f"  Total re-run: {grand_flips}/{grand_total} failed runs now pass.")
    print("-" * 92)
    print(
        "  ⚠ Do NOT report 'total now pass' as a success-rate gain. Only the fix_target line is\n"
        "    attributable to the code change. For a paper number, re-run the FULL set with the fix."
    )
    return 0


def cmd_taskids(args: argparse.Namespace) -> int:
    """Print the comma-separated task ids that have >=1 failed run (for TASK_IDS)."""
    failures = _enumerate_failures(args.src)
    print(",".join(sorted({f["task"] for f in failures})))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("preseed", "report", "taskids"):
        p = sub.add_parser(name)
        p.add_argument("--src", type=lambda s: Path(s).expanduser().resolve(), default=DEFAULT_SRC)
        p.add_argument("--dst", type=lambda s: Path(s).expanduser().resolve(), default=DEFAULT_DST)

    args = parser.parse_args(argv)
    if args.command == "preseed":
        return cmd_preseed(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "taskids":
        return cmd_taskids(args)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

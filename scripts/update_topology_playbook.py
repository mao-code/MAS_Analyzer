#!/usr/bin/env python3
"""Offline, process-only topology playbook updates.

Runs of the self-evolved system record a process-signal update candidate in
``run_metadata.self_evolved.playbook_update_candidate``. The default learning path is
online (the skill reflection agent, every N runs); this script is the offline equivalent
for the JSON playbook. It merges those candidates via a deterministic merge ranked by the
PROCESS proxy (``is_process_clean``) — it deliberately does NOT read ``run_*.eval.json``,
so the benchmark verdict never leaks into the planner's long-term memory.

Usage:
    python scripts/update_topology_playbook.py \
        --experiment-root artifacts/full_experiment/<experiment-id> \
        [--playbook config/topology_playbook.json] [--benchmark <name>] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MAS.self_evolved.playbook import TopologyPlaybook, is_process_clean  # noqa: E402

SYSTEM_DIR_NAME = "self_evolved"


def _load_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def collect_candidates(experiment_root: Path, *, benchmark: str | None = None) -> list[dict]:
    """Collect process-signal update candidates from every self_evolved run dir.

    Reads only ``run_metadata.self_evolved.playbook_update_candidate`` — never the
    eval verdict — so the planner's long-term memory stays free of ground truth.
    """

    candidates: list[dict] = []
    for raw_path in sorted(experiment_root.glob(f"*/{SYSTEM_DIR_NAME}/*/run_*.raw.json")):
        benchmark_name = raw_path.parts[-4]
        if benchmark is not None and benchmark_name != benchmark:
            continue
        raw_payload = _load_json(raw_path)
        if raw_payload is None:
            continue
        candidate = (
            raw_payload.get("run_metadata", {})
            .get("self_evolved", {})
            .get("playbook_update_candidate")
        )
        if not isinstance(candidate, dict) or not candidate.get("key"):
            continue
        candidates.append(candidate)
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--playbook", type=Path, default=ROOT / "config" / "topology_playbook.json")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    experiment_root = args.experiment_root.expanduser().resolve()
    if not experiment_root.is_dir():
        print(f"error: experiment root not found: {experiment_root}", file=sys.stderr)
        return 2

    candidates = collect_candidates(experiment_root, benchmark=args.benchmark)
    if not candidates:
        print("No self_evolved playbook candidates found; nothing to do.")
        return 0

    playbook = TopologyPlaybook.load(args.playbook)
    for candidate in candidates:
        playbook.merge_candidate(candidate)

    clean = sum(1 for candidate in candidates if is_process_clean(candidate))
    print(
        f"Merged {len(candidates)} run candidate(s) ({clean} ran clean, process signals only) "
        f"into {args.playbook} ({len(playbook.entries)} entries)."
    )
    if args.dry_run:
        print("Dry run: playbook not written.")
        return 0
    playbook.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

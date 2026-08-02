#!/usr/bin/env bash
# Run MANTA on the same held-out WorkBench positions used by the adaptive
# baseline launchers: positions 10-39 (30 tasks, three runs by default).
#
# This is a split-matched rerun. It deliberately reuses full_selfevo_bw.sh so
# model routing, retry policy, topology settings, and online skill updates stay
# identical to the canonical MANTA WorkBench run; only the task selection and
# experiment id change.
#
# Usage:
#   DRY_RUN=1 bash scripts/experiments/run_manta_workbench_10_39.sh
#   bash scripts/experiments/run_manta_workbench_10_39.sh
#   RUNS_PER_TASK=1 bash scripts/experiments/run_manta_workbench_10_39.sh
#   MODELS=google/gemma-4-31b-it bash scripts/experiments/run_manta_workbench_10_39.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  python_bin="${REPO_ROOT}/.venv/bin/python"
else
  python_bin="$(command -v python3)"
fi

# Resolve the positional split through the benchmark adapter and fail closed if
# the v1 WorkBench ordering no longer maps positions 10-39 to these task ids.
task_ids="$("${python_bin}" - <<'PY'
import tomllib
from pathlib import Path

from benchmark import get_benchmark

config_path = Path("config/benchmarks/workbench_10.toml")
config = tomllib.loads(config_path.read_text(encoding="utf-8"))["workbench"]
tasks = list(get_benchmark("workbench", config=config).load_tasks(task_limit=40))
selected = [str(task.task_id) for task in tasks[10:40]]
expected = [f"multi_domain_{index}" for index in range(10, 40)]
if selected != expected:
    raise SystemExit(
        "WorkBench task order changed; refusing to run an unverified split. "
        f"Expected {expected}, got {selected}."
    )
print(",".join(selected))
PY
)"

export ONLY_SYSTEMS="self_evolved"
export BENCHMARKS="workbench"
export TASK_LIMIT="30"
export TASK_IDS="${task_ids}"
export RUNS_PER_TASK="${RUNS_PER_TASK:-3}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-manta_workbench_10_39}"
export MODELS="${MODELS:-google/gemma-4-31b-it:nitro}"

echo "[manta-workbench-10-39] task_count=30 task_ids=${TASK_IDS}" >&2
echo "[manta-workbench-10-39] runs_per_task=${RUNS_PER_TASK} models=${MODELS} experiment_id=${EXPERIMENT_ID}" >&2

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[manta-workbench-10-39] DRY_RUN=1; preflight passed, no experiment started." >&2
  exit 0
fi

exec bash "${REPO_ROOT}/scripts/full_selfevo_bw.sh" "$@"

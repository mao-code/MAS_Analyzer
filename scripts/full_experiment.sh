#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper defaults. Override these with environment variables or CLI flags.
# Examples:
#   TASK_LIMIT=2 RUNS_PER_TASK=1 bash scripts/full_experiment.sh
#   BENCHMARKS=workbench,scicode SKIP_SETUP=1 bash scripts/full_experiment.sh

TASK_LIMIT="${TASK_LIMIT:-}"
RUNS_PER_TASK="${RUNS_PER_TASK:-1}"
BENCHMARKS="${BENCHMARKS:-}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
CONFIG_DIR="${CONFIG_DIR:-}"
SKIP_SETUP="${SKIP_SETUP:-0}"
SETUP_ONLY="${SETUP_ONLY:-0}"

args=()

if [[ -n "${TASK_LIMIT}" ]]; then
  args+=(--task-limit "${TASK_LIMIT}")
fi
if [[ -n "${RUNS_PER_TASK}" ]]; then
  args+=(--runs-per-task "${RUNS_PER_TASK}")
fi
if [[ -n "${BENCHMARKS}" ]]; then
  args+=(--benchmarks "${BENCHMARKS}")
fi
if [[ -n "${EXPERIMENT_ID}" ]]; then
  args+=(--experiment-id "${EXPERIMENT_ID}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
  args+=(--output-root "${OUTPUT_ROOT}")
fi
if [[ -n "${CONFIG_DIR}" ]]; then
  args+=(--config-dir "${CONFIG_DIR}")
fi
if [[ "${SKIP_SETUP}" == "1" ]]; then
  args+=(--skip-setup)
fi
if [[ "${SETUP_ONLY}" == "1" ]]; then
  args+=(--setup-only)
fi

if command -v conda >/dev/null 2>&1; then
  exec conda run -n agents python scripts/full_experiment.py "${args[@]}" "$@"
fi

exec python scripts/full_experiment.py "${args[@]}" "$@"

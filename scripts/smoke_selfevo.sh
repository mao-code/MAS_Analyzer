#!/usr/bin/env bash
#
# Quick smoke test for the self-evolved topology system over BOTH benchmarks of
# interest (browsecomp + workbench), at a tiny scale: 3 tasks per benchmark,
# 1 run per task. It is a thin wrapper over scripts/full_selfevo_bw.sh, so it
# inherits the exact same driver, model, tool configs and ONLINE skill learning.
#
# Use it to verify the pipeline end-to-end (planning -> spawn -> execute ->
# audit -> repair -> finalize -> online skill update) before committing to the
# full 30-task run.
#
# Output lands at:
#   artifacts/full_experiment/<EXPERIMENT_ID>__google_gemma_4_31b_it_nitro/
#       {browsecomp,workbench}/self_evolved/
#
# Usage:
#   bash scripts/smoke_selfevo.sh                       # browsecomp + workbench, 3x1 each
#   BENCHMARKS=browsecomp bash scripts/smoke_selfevo.sh # one benchmark only
#   EXPERIMENT_ID=my_smoke bash scripts/smoke_selfevo.sh
# Any extra args pass through to scripts/full_selfevo_bw.sh -> full_experiment.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Load secrets if present (full_selfevo_bw.sh does NOT source .env itself).
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# --- smoke scale: 3 tasks per benchmark, 1 run per task ---
export BENCHMARKS="${BENCHMARKS:-browsecomp,workbench}"
export TASK_LIMIT="${TASK_LIMIT:-3}"
export RUNS_PER_TASK="${RUNS_PER_TASK:-1}"
export EXPERIMENT_ID="${EXPERIMENT_ID:-smoke_selfevo}"
# Reflect the long-term skill at the smoke scale (every 3 runs = once per benchmark).
export SKILL_UPDATE_BATCH_SIZE="${SKILL_UPDATE_BATCH_SIZE:-3}"

echo "[smoke_selfevo] BENCHMARKS=${BENCHMARKS} TASK_LIMIT=${TASK_LIMIT} RUNS_PER_TASK=${RUNS_PER_TASK} EXPERIMENT_ID=${EXPERIMENT_ID} SKILL_UPDATE_BATCH_SIZE=${SKILL_UPDATE_BATCH_SIZE}" >&2

exec bash "${REPO_ROOT}/scripts/full_selfevo_bw.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

worker_id="${1:?worker id required}"
task_offset="${2:?task offset required}"
task_limit="${3:?task limit required}"

cd /home/lai/github/MAS_Analyzer

set -a
if [ -f .env ]; then
  # shellcheck disable=SC1091
  . ./.env
fi
set +a

export MAS_REQUIRE_LIVE_LLM=1
export OPENROUTER_TEMPERATURE=1.0
export OPENROUTER_TOP_P=1.0
export OPENROUTER_TOP_K=0
export OPENROUTER_REASONING_EFFORT=
export MAS_LLM_RETRY_ATTEMPTS=5
export MAS_LLM_EMPTY_COMPLETION_RETRY_ATTEMPTS=1
export MAS_LLM_TIMEOUT_RETRY_ATTEMPTS=2
export MAS_LLM_RETRY_BACKOFF_S=8
export MAS_LLM_RETRY_MAX_BACKOFF_S=180
export MAS_OPENROUTER_PROVIDER_ORDER=DeepInfra,Chutes,Novita,Together
export MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER=1
export MAS_SELF_REFINE_NO_MAX_TOKENS=1

run_baseline() {
  local baseline="$1"
  shift
  echo "[math500 worker=${worker_id}] START baseline=${baseline} offset=${task_offset} limit=${task_limit} $(date -Is)"
  uv run python main.py run \
    --config config/benchmarks/math500_10.toml \
    --benchmark math500 \
    --output-dir artifacts/full_experiment \
    --output-layout hierarchical \
    --experiment-id math500_prompting_gemma_30x3 \
    --system-label "${baseline}" \
    --topology sas \
    --agents 1 \
    --mas-rounds 1 \
    --discussion-rounds 1 \
    --communication-budget 0 \
    --task-offset "${task_offset}" \
    --task-limit "${task_limit}" \
    --runs-per-task 3 \
    --seed 42 \
    --default-model google/gemma-4-31b-it \
    --prompting-baseline "${baseline}" \
    "$@"
  echo "[math500 worker=${worker_id}] DONE baseline=${baseline} offset=${task_offset} limit=${task_limit} $(date -Is)"
}

run_baseline cot
run_baseline self_consistency --self-consistency-samples 3
run_baseline self_refine --self-refine-rounds 3

#!/usr/bin/env bash
set -euo pipefail

task_id="${1:?task id required}"
benchmark="${2:-browsecomp}"
baseline="${3:-self_refine}"

cd /home/lai/github/MAS_Analyzer

set -a
if [ -f .env ]; then
  # shellcheck disable=SC1091
  . ./.env
fi
set +a

export MAS_STRICT_TOOL_JSON_PROMPT=1
export MAS_REQUIRE_LIVE_LLM=1
export OPENROUTER_TEMPERATURE=1.0
export OPENROUTER_TOP_P=1.0
export OPENROUTER_TOP_K=0
# Gemma on OpenRouter may spend the whole visible completion budget on hidden
# reasoning when this is enabled, returning content=None with finish_reason=length.
export OPENROUTER_REASONING_EFFORT=
export MAS_LLM_RETRY_ATTEMPTS=5
export MAS_LLM_EMPTY_COMPLETION_RETRY_ATTEMPTS=1
export MAS_LLM_TIMEOUT_RETRY_ATTEMPTS=2
export MAS_LLM_RETRY_BACKOFF_S=8
export MAS_LLM_RETRY_MAX_BACKOFF_S=180
export MAS_OPENROUTER_PROVIDER_ORDER=DeepInfra,Chutes,Novita,Together
export MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER=1
export MAS_TRANSCRIPT_COMPACTION_ENABLED=1
export MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER=2
export MAS_TOOL_CONTEXT_RAW_TURNS=2
export MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS=6000
export MAS_TOOL_CONTEXT_PREVIEW_CHARS=160
export MAS_TOOL_CONTEXT_DOCUMENT_CHARS=12000
export MAS_SELF_REFINE_NO_MAX_TOKENS=1

case "${baseline}" in
  cot)
    baseline_args=(--prompting-baseline cot)
    ;;
  self_consistency)
    baseline_args=(--prompting-baseline self_consistency --self-consistency-samples 3)
    ;;
  self_refine)
    baseline_args=(--prompting-baseline self_refine --self-refine-rounds 3)
    ;;
  *)
    echo "Unknown prompting baseline: ${baseline}" >&2
    exit 2
    ;;
esac

echo "[STRICT RERUN-BAD compact ${baseline} ${benchmark} task ${task_id}] $(date -Is)"

.venv/bin/python main.py run \
  --config "config/benchmarks/${benchmark}_10.toml" \
  --benchmark "${benchmark}" \
  --output-dir artifacts/full_experiment \
  --output-layout hierarchical \
  --experiment-id prompting_baselines_gemma_30x3 \
  --system-label "${baseline}" \
  --topology sas \
  --agents 1 \
  --mas-rounds 1 \
  --discussion-rounds 1 \
  --communication-budget 0 \
  --task-ids "${task_id}" \
  --runs-per-task 3 \
  --seed 42 \
  --default-model google/gemma-4-31b-it \
  --judge-model google/gemma-4-31b-it \
  --benchmark-eval-judge-model google/gemma-4-31b-it \
  "${baseline_args[@]}"

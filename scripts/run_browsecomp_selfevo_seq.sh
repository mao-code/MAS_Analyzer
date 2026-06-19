#!/usr/bin/env bash
# Run browsecomp self_evolved ONE task per process (via --task-offset), so a task
# that completes is saved to disk even if a later task is OOM-killed. Writes into the
# same hierarchical experiment root full_experiment.sh uses, then summarizes so the
# per-system summary.csv the comparison reads is produced. Matches full_experiment.sh
# self_evolved flags + sampling env for apples-to-apples with the static baselines.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${OPENROUTER_API_KEY:-}" && -f ".env" ]]; then set -a; . ./.env; set +a; fi
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (or put it in .env).}"

PY="${PY:-/Users/maoxunhuang/miniconda3/envs/agents/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/.venv/bin/python"
CONFIG="${CONFIG:-config/benchmarks/browsecomp_10.toml}"
MODEL="${MODEL:-google/gemma-4-31b-it:nitro}"
EXPERIMENT_ID="${EXPERIMENT_ID:-eval_selfevo__google_gemma_4_31b_it_nitro}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/full_experiment}"
SAMPLES="${SAMPLES:-2}"
START_OFFSET="${START_OFFSET:-0}"  # resume from a later task if an earlier run died

# full_experiment.sh sampling/retry/tool-context defaults (apples-to-apples).
export OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-medium}"
export OPENROUTER_TEMPERATURE="${OPENROUTER_TEMPERATURE:-1.0}"
export OPENROUTER_TOP_P="${OPENROUTER_TOP_P:-1.0}"
export OPENROUTER_TOP_K="${OPENROUTER_TOP_K:-0}"
export MAS_LLM_RETRY_ATTEMPTS="${MAS_LLM_RETRY_ATTEMPTS:-5}"
export MAS_LLM_TIMEOUT_RETRY_ATTEMPTS="${MAS_LLM_TIMEOUT_RETRY_ATTEMPTS:-2}"
export MAS_LLM_RETRY_BACKOFF_S="${MAS_LLM_RETRY_BACKOFF_S:-8}"
export MAS_LLM_RETRY_MAX_BACKOFF_S="${MAS_LLM_RETRY_MAX_BACKOFF_S:-180}"
export MAS_OPENROUTER_PROVIDER_ORDER="${MAS_OPENROUTER_PROVIDER_ORDER:-DeepInfra,Chutes,Novita,Together}"
export MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER="${MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER:-1}"
export MAS_TRANSCRIPT_COMPACTION_ENABLED="${MAS_TRANSCRIPT_COMPACTION_ENABLED:-1}"
export MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER="${MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER:-2}"
export MAS_TOOL_CONTEXT_RAW_TURNS="${MAS_TOOL_CONTEXT_RAW_TURNS:-2}"
export MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS="${MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS:-6000}"
export MAS_TOOL_CONTEXT_PREVIEW_CHARS="${MAS_TOOL_CONTEXT_PREVIEW_CHARS:-160}"
export MAS_TOOL_CONTEXT_DOCUMENT_CHARS="${MAS_TOOL_CONTEXT_DOCUMENT_CHARS:-12000}"

LOGDIR="artifacts/_smoke_logs"; mkdir -p "$LOGDIR"
echo "python=$PY  samples=$SAMPLES  exp=$EXPERIMENT_ID"

for ((i=START_OFFSET; i<SAMPLES; i++)); do
  echo "=== browsecomp self_evolved task offset=$i ==="
  "$PY" main.py run \
    --config "$CONFIG" \
    --benchmark browsecomp \
    --output-dir "$OUTPUT_DIR" \
    --output-layout hierarchical \
    --experiment-id "$EXPERIMENT_ID" \
    --system-label self_evolved \
    --topology self_evolved \
    --agents 5 --mas-rounds 2 --discussion-rounds 1 --communication-budget 2 \
    --default-model "$MODEL" --judge-model "$MODEL" \
    --task-limit 1 --task-offset "$i" --runs-per-task 1 \
    >"$LOGDIR/browsecomp_se_offset${i}.log" 2>&1
  rc=$?
  echo "  offset=$i exit=$rc (log: $LOGDIR/browsecomp_se_offset${i}.log)"
done

echo "=== summarize experiment root ==="
"$PY" main.py summarize-experiment --experiment-root "$OUTPUT_DIR/$EXPERIMENT_ID" >"$LOGDIR/browsecomp_se_summarize.log" 2>&1
echo "done."

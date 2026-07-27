#!/usr/bin/env bash
# Run stabletoolbench self_evolved ONE task per process (via --task-offset), so a task
# that completes is saved to disk even if a sibling is OOM-killed (SIGTERM -15) or stalls.
# Each task runs RUNS_PER_TASK times for a fair (multi-run) comparison vs the static
# baselines. A per-process wall-clock timeout kills a hung LLM call instead of letting it
# block the whole pass. Starts/stops the StableToolBench virtual server itself, then
# summarizes so the per-system summary.csv the comparison reads is produced. Matches the
# full_experiment self_evolved flags + sampling env for apples-to-apples.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -z "${OPENROUTER_API_KEY:-}" && -f ".env" ]]; then set -a; . ./.env; set +a; fi
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (or put it in .env).}"

PY="${PY:-/Users/maoxunhuang/miniconda3/envs/agents/bin/python}"
[[ -x "$PY" ]] || PY="$ROOT/.venv/bin/python"
CONFIG="${CONFIG:-config/benchmarks/stabletoolbench_10.toml}"
MODEL="${MODEL:-google/gemma-4-31b-it:nitro}"
EXPERIMENT_ID="${EXPERIMENT_ID:-eval_selfevo__google_gemma_4_31b_it_nitro}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/full_experiment}"
SAMPLES="${SAMPLES:-3}"
RUNS_PER_TASK="${RUNS_PER_TASK:-3}"
START_OFFSET="${START_OFFSET:-0}"
PER_TASK_TIMEOUT_S="${PER_TASK_TIMEOUT_S:-1500}"  # kill a hung task after 25 min
SERVER_PORT="${SERVER_PORT:-8080}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT/benchmark/stabletoolbench/tool_response_cache}"

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

# Portable per-process timeout (macOS has no coreutils `timeout` by default).
run_with_timeout() {
  local secs="$1"; shift
  "$@" & local pid=$!
  ( sleep "$secs" && kill -TERM "$pid" 2>/dev/null && sleep 8 && kill -KILL "$pid" 2>/dev/null ) &
  local watcher=$!
  wait "$pid" 2>/dev/null; local rc=$?
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  return "$rc"
}

# --- Start the StableToolBench virtual server ---
echo "starting virtual server on :$SERVER_PORT (cache=$CACHE_ROOT)"
"$PY" scripts/stabletoolbench_virtual_server.py --host localhost --port "$SERVER_PORT" \
  --path /virtual --cache-root "$CACHE_ROOT" >"$LOGDIR/stb_server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT
# Wait for readiness (up to ~30s).
for _ in $(seq 1 30); do
  grep -q "ready" "$LOGDIR/stb_server.log" 2>/dev/null && break
  sleep 1
done
echo "python=$PY  samples=$SAMPLES  runs_per_task=$RUNS_PER_TASK  exp=$EXPERIMENT_ID  timeout=${PER_TASK_TIMEOUT_S}s"

for ((i=START_OFFSET; i<SAMPLES; i++)); do
  echo "=== stabletoolbench self_evolved task offset=$i (x$RUNS_PER_TASK runs) ==="
  run_with_timeout "$PER_TASK_TIMEOUT_S" \
    "$PY" main.py run \
      --config "$CONFIG" \
      --benchmark stabletoolbench \
      --output-dir "$OUTPUT_DIR" \
      --output-layout hierarchical \
      --experiment-id "$EXPERIMENT_ID" \
      --system-label self_evolved \
      --topology self_evolved \
      --agents 5 --mas-rounds 2 --discussion-rounds 1 --communication-budget 2 \
      --default-model "$MODEL" --judge-model "$MODEL" \
      --task-limit 1 --task-offset "$i" --runs-per-task "$RUNS_PER_TASK" \
      >"$LOGDIR/stb_se_offset${i}.log" 2>&1
  rc=$?
  echo "  offset=$i exit=$rc (log: $LOGDIR/stb_se_offset${i}.log)"
done

echo "=== summarize experiment root ==="
"$PY" main.py summarize-experiment --experiment-root "$OUTPUT_DIR/$EXPERIMENT_ID" >"$LOGDIR/stb_se_summarize.log" 2>&1
echo "done."

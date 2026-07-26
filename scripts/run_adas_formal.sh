#!/usr/bin/env bash
set -euo pipefail

cd /home/lai/github/MAS_Analyzer

RUN_ID="${RUN_ID:-adas_gemma_10val_30test_T1_$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS="${WORKERS:-8}"
SEARCH_GENERATIONS="${SEARCH_GENERATIONS:-3}"
DEBUG_MAX="${DEBUG_MAX:-3}"
META_RETRY_ATTEMPTS="${META_RETRY_ATTEMPTS:-3}"
TASK_LIMIT="${TASK_LIMIT:-40}"
VALIDATION_TASK_LIMIT="${VALIDATION_TASK_LIMIT:-10}"
FINAL_TASK_OFFSET="${FINAL_TASK_OFFSET:-10}"
FINAL_TASK_LIMIT="${FINAL_TASK_LIMIT:-30}"
RUNS_PER_TASK="${RUNS_PER_TASK:-3}"
VALIDATION_REPEATS="${VALIDATION_REPEATS:-1}"
MODEL="${MODEL:-google/gemma-4-31b-it}"
TEMPERATURE="${TEMPERATURE:-1}"
MAX_TOKENS="${MAX_TOKENS:-0}"
MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_adas_reproduce}"
CONFIG="${CONFIG:-config/reproduce_agentsquare.example.toml}"
TIMEOUT_S="${TIMEOUT_S:-240}"

mkdir -p run_logs "$OUTPUT_DIR"
LOG_PATH="${LOG_PATH:-run_logs/adas_${RUN_ID}.log}"
SUMMARY_PATH="$OUTPUT_DIR/$RUN_ID/adas_summary.json"
STB_SERVER_LOG="${STB_SERVER_LOG:-run_logs/adas_${RUN_ID}_stabletoolbench_server.log}"
stb_server_pid=""

cleanup() {
  if [ -n "${stb_server_pid}" ] && kill -0 "${stb_server_pid}" 2>/dev/null; then
    echo "[adas] stopping StableToolBench virtual server pid=${stb_server_pid}"
    kill "${stb_server_pid}" 2>/dev/null || true
    wait "${stb_server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

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
export MAS_LLM_RETRY_ATTEMPTS="${MAS_LLM_RETRY_ATTEMPTS:-5}"
export MAS_LLM_EMPTY_COMPLETION_RETRY_ATTEMPTS="${MAS_LLM_EMPTY_COMPLETION_RETRY_ATTEMPTS:-1}"
export MAS_LLM_TIMEOUT_RETRY_ATTEMPTS="${MAS_LLM_TIMEOUT_RETRY_ATTEMPTS:-2}"
export MAS_LLM_RETRY_BACKOFF_S="${MAS_LLM_RETRY_BACKOFF_S:-8}"
export MAS_LLM_RETRY_MAX_BACKOFF_S="${MAS_LLM_RETRY_MAX_BACKOFF_S:-180}"
export MAS_OPENROUTER_PROVIDER_ORDER="${MAS_OPENROUTER_PROVIDER_ORDER:-DeepInfra,Chutes,Novita,Together}"
export MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER="${MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER:-1}"

{
echo "[adas] START $(date -Is)"
echo "[adas] run_id=${RUN_ID}"
echo "[adas] workers=${WORKERS} search_generations=${SEARCH_GENERATIONS} debug_max=${DEBUG_MAX} meta_retry_attempts=${META_RETRY_ATTEMPTS}"
echo "[adas] split task_limit=${TASK_LIMIT} val=${VALIDATION_TASK_LIMIT} final_offset=${FINAL_TASK_OFFSET} final=${FINAL_TASK_LIMIT} runs_per_task=${RUNS_PER_TASK}"
echo "[adas] model=${MODEL} temperature=${TEMPERATURE} max_tokens=${MAX_TOKENS} timeout_s=${TIMEOUT_S}"
extra_args=()
if [ -n "$MAX_TOOL_ITERATIONS" ]; then
  extra_args+=(--max-tool-iterations "$MAX_TOOL_ITERATIONS")
  echo "[adas] override max_tool_iterations=${MAX_TOOL_ITERATIONS}"
else
  echo "[adas] max_tool_iterations=benchmark_config"
fi

if ! curl -fsS "http://127.0.0.1:8080/virtual/healthz" >/dev/null 2>&1; then
  echo "[adas] START_STABLETOOLBENCH_SERVER log=${STB_SERVER_LOG}"
  uv run python scripts/stabletoolbench_virtual_server.py \
    --host 127.0.0.1 \
    --port 8080 \
    --path /virtual \
    --cache-root benchmark/stabletoolbench/tool_response_cache \
    >"${STB_SERVER_LOG}" 2>&1 &
  stb_server_pid="$!"
  for _ in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:8080/virtual/healthz" >/dev/null 2>&1; then
      echo "[adas] STABLETOOLBENCH_SERVER_READY pid=${stb_server_pid}"
      break
    fi
    sleep 1
  done
  curl -fsS "http://127.0.0.1:8080/virtual/healthz" >/dev/null
else
  echo "[adas] STABLETOOLBENCH_SERVER_ALREADY_READY"
fi

uv run python -m reproduce.adas.run_existing_benchmarks \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --benchmark browsecomp \
  --benchmark stabletoolbench \
  --benchmark plancraft \
  --benchmark workbench \
  --benchmark math500 \
  --task-limit "$TASK_LIMIT" \
  --validation-task-limit "$VALIDATION_TASK_LIMIT" \
  --final-task-offset "$FINAL_TASK_OFFSET" \
  --final-task-limit "$FINAL_TASK_LIMIT" \
  --runs-per-task "$RUNS_PER_TASK" \
  --validation-repeats "$VALIDATION_REPEATS" \
  --search \
  --search-generations "$SEARCH_GENERATIONS" \
  --debug-max "$DEBUG_MAX" \
  --meta-retry-attempts "$META_RETRY_ATTEMPTS" \
  --workers "$WORKERS" \
  --resume \
  --keep-going \
  --model "$MODEL" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --timeout-s "$TIMEOUT_S" \
  "${extra_args[@]}"

uv run python -m reproduce.adas.summarize_results \
  --run-root "$OUTPUT_DIR/$RUN_ID" \
  --output "$SUMMARY_PATH"

echo "[adas] DONE $(date -Is)"
echo "[adas] summary=${SUMMARY_PATH}"
} 2>&1 | tee -a "$LOG_PATH"

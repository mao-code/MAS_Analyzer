#!/usr/bin/env bash
set -euo pipefail

cd /home/lai/github/MAS_Analyzer

run_id="${1:-agentsquare_gemma_10val_30test_T1_$(date +%Y%m%dT%H%M%S)}"
workers="${AGENTSQUARE_WORKERS:-4}"
search_iterations="${AGENTSQUARE_SEARCH_ITERATIONS:-3}"
predictor_top_k="${AGENTSQUARE_PREDICTOR_TOP_K:-2}"
max_search_candidates="${AGENTSQUARE_MAX_SEARCH_CANDIDATES:-3}"
timeout_s="${AGENTSQUARE_TIMEOUT_S:-240}"

mkdir -p run_logs outputs_agentsquare_reproduce
log_path="run_logs/agentsquare_${run_id}.log"
summary_path="outputs_agentsquare_reproduce/${run_id}/agentsquare_summary.json"
stb_server_log="run_logs/agentsquare_${run_id}_stabletoolbench_server.log"
stb_server_pid=""

cleanup() {
  if [ -n "${stb_server_pid}" ] && kill -0 "${stb_server_pid}" 2>/dev/null; then
    echo "[agentsquare] stopping StableToolBench virtual server pid=${stb_server_pid}"
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

echo "[agentsquare] run_id=${run_id}"
echo "[agentsquare] log=${log_path}"
echo "[agentsquare] workers=${workers} search_iterations=${search_iterations} predictor_top_k=${predictor_top_k} max_search_candidates=${max_search_candidates}"

{
  echo "[agentsquare] START $(date -Is)"
  if ! curl -fsS "http://127.0.0.1:8080/virtual/healthz" >/dev/null 2>&1; then
    echo "[agentsquare] START_STABLETOOLBENCH_SERVER $(date -Is) log=${stb_server_log}"
    uv run python scripts/stabletoolbench_virtual_server.py \
      --host 127.0.0.1 \
      --port 8080 \
      --path /virtual \
      --cache-root benchmark/stabletoolbench/tool_response_cache \
      >"${stb_server_log}" 2>&1 &
    stb_server_pid="$!"
    for _ in $(seq 1 30); do
      if curl -fsS "http://127.0.0.1:8080/virtual/healthz" >/dev/null 2>&1; then
        echo "[agentsquare] STABLETOOLBENCH_SERVER_READY pid=${stb_server_pid}"
        break
      fi
      sleep 1
    done
    curl -fsS "http://127.0.0.1:8080/virtual/healthz" >/dev/null
  else
    echo "[agentsquare] STABLETOOLBENCH_SERVER_ALREADY_READY"
  fi

  echo "[agentsquare] PREFLIGHT $(date -Is)"
  uv run python -m reproduce.agentsquare.preflight \
    --config config/reproduce_agentsquare.example.toml

  uv run python -m reproduce.agentsquare.run_existing_benchmarks \
    --config config/reproduce_agentsquare.example.toml \
    --benchmark browsecomp \
    --benchmark math500 \
    --benchmark plancraft \
    --benchmark stabletoolbench \
    --benchmark workbench \
    --task-limit 40 \
    --validation-task-limit 10 \
    --final-task-offset 10 \
    --final-task-limit 30 \
    --runs-per-task 3 \
    --validation-repeats 1 \
    --search \
    --search-iterations "${search_iterations}" \
    --module-evolution-mode llm \
    --predictor-mode llm \
    --predictor-top-k "${predictor_top_k}" \
    --max-search-candidates "${max_search_candidates}" \
    --planning None \
    --reasoning IO \
    --tooluse None \
    --memory None \
    --model google/gemma-4-31b-it \
    --temperature 1 \
    --max-tokens 0 \
    --workers "${workers}" \
    --timeout-s "${timeout_s}" \
    --resume \
    --keep-going \
    --run-id "${run_id}" \
    --output-dir outputs_agentsquare_reproduce

  echo "[agentsquare] SUMMARIZE $(date -Is)"
  uv run python -m reproduce.agentsquare.summarize_results \
    --run-root "outputs_agentsquare_reproduce/${run_id}" \
    --output "${summary_path}"
  echo "[agentsquare] DONE $(date -Is)"
} 2>&1 | tee -a "${log_path}"

echo "[agentsquare] summary=${summary_path}"

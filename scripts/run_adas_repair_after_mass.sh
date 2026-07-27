#!/usr/bin/env bash
set -euo pipefail

cd /home/lai/github/MAS_Analyzer

WAIT_SESSION="${WAIT_SESSION:-mass_plancraft_searchfix_unseen}"
RUN_ID="${RUN_ID:-adas_repaired_browsecomp_workbench_10val30test_T1_20260726}"
LOG_PATH="${LOG_PATH:-run_logs/${RUN_ID}.log}"
ADAS_WORKERS="${ADAS_WORKERS:-10}"

mkdir -p run_logs outputs_adas_reproduce
set -a
# shellcheck disable=SC1091
. ./.env
set +a

echo "[adas-repair] waiting for tmux session ${WAIT_SESSION}" | tee -a "$LOG_PATH"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 60
done

MASS_ROOT="outputs_mass_reproduce/mass_plancraft_searchfix_unseen_30x3_20260726"
MASS_SUMMARY="$MASS_ROOT/summary.json"
MASS_CHECKPOINTS="$MASS_ROOT/plancraft/checkpoints/final_evaluation"
# The MASS runner writes summary.json at the run root and may not emit a
# separate marker on a clean resume. A complete final evaluation is the
# authoritative gate for starting the downstream ADAS run.
while [[ ! -f "$MASS_SUMMARY" || $(find "$MASS_CHECKPOINTS" -type f -name '*.json' 2>/dev/null | wc -l) -lt 90 ]]; do
  echo "[adas-repair] MASS session ended without completion marker; preserving checkpoints and waiting for resume" | tee -a "$LOG_PATH"
  tmux new-session -d -s "$WAIT_SESSION" "cd /home/lai/github/MAS_Analyzer && exec uv run --frozen python -m reproduce.mass.run_existing_benchmarks --config config/experiment.example.toml --env-file .env --benchmark plancraft --output-dir outputs_mass_reproduce --run-id mass_plancraft_searchfix_unseen_30x3_20260726 --task-limit 80 --validation-task-offset 35 --validation-task-limit 10 --final-task-offset 45 --final-task-limit 30 --final-evaluation-repeats 3 --prompt-search-source outputs_mass_reproduce/mass_official_plancraft_rescued_final_20260706T130548Z/plancraft/checkpoints/prompt_search.json --model google/gemma-4-31b-it --temperature 1 --max-tokens 0 --num-workers 4 >> run_logs/mass_plancraft_searchfix_unseen_30x3_20260726.log 2>&1"
  sleep 60
  while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do sleep 60; done
done

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

echo "[adas-repair] starting ${RUN_ID} at $(date -Is)" | tee -a "$LOG_PATH"
uv run --frozen python -m reproduce.adas.run_existing_benchmarks \
  --config config/reproduce_agentsquare.example.toml \
  --env-file .env \
  --output-dir outputs_adas_reproduce \
  --run-id "$RUN_ID" \
  --benchmark browsecomp \
  --benchmark workbench \
  --task-limit 40 \
  --validation-task-limit 10 \
  --final-task-offset 10 \
  --final-task-limit 30 \
  --runs-per-task 3 \
  --validation-repeats 1 \
  --search \
  --search-generations 3 \
  --debug-max 3 \
  --meta-retry-attempts 3 \
  --workers "$ADAS_WORKERS" \
  --resume \
  --keep-going \
  --model google/gemma-4-31b-it \
  --temperature 1 \
  --max-tokens 0 \
  --timeout-s 600 \
  >>"$LOG_PATH" 2>&1

uv run --frozen python -m reproduce.adas.summarize_results \
  --run-root "outputs_adas_reproduce/$RUN_ID" \
  --output "outputs_adas_reproduce/$RUN_ID/adas_summary.json" \
  >>"$LOG_PATH" 2>&1

echo "[adas-repair] done at $(date -Is)" | tee -a "$LOG_PATH"

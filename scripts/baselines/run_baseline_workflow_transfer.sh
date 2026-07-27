#!/usr/bin/env bash
set -uo pipefail

ROOT="/home/lai/github/MAS_Analyzer"
AFLOW_ROOT="/home/lai/github/MAS_Analyzer_aflow"
RUN_ID="${RUN_ID:-workflow_transfer_pc_wb_20260727}"
WORKERS="${WORKERS:-4}"
LOG_ROOT="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_ROOT"

run_job() {
  local name="$1"
  local cwd="$2"
  shift 2
  printf '[%s] START %s\n' "$(date -Is)" "$name" | tee -a "$LOG_ROOT/driver.log"
  if (cd "$cwd" && "$@") >"$LOG_ROOT/$name.log" 2>&1; then
    printf '[%s] DONE %s\n' "$(date -Is)" "$name" | tee -a "$LOG_ROOT/driver.log"
  else
    local status=$?
    printf '[%s] FAILED %s status=%s\n' "$(date -Is)" "$name" "$status" \
      | tee -a "$LOG_ROOT/driver.log"
  fi
}

COMMON=(--task-limit 40 --validation-task-limit 0 --final-task-offset 10
  --final-task-limit 30 --runs-per-task 1 --workers "$WORKERS" --temperature 1
  --max-tokens 0 --timeout-s 600 --seed 42 --resume)

run_job adas_pc_to_wb "$ROOT" "$ROOT/.venv/bin/python" -m reproduce.adas.run_existing_benchmarks \
  --config config/reproduce_agentsquare.toml --benchmark workbench \
  --output-dir outputs_transferability --run-id "$RUN_ID/adas_pc_to_wb" \
  --search-source outputs_adas_reproduce/adas_plancraft_fair_nosearch_valsmall_max30_20260727/plancraft/search/search_results.json \
  "${COMMON[@]}"

run_job adas_wb_to_pc "$ROOT" "$ROOT/.venv/bin/python" -m reproduce.adas.run_existing_benchmarks \
  --config config/reproduce_plancraft_fair.toml --benchmark plancraft \
  --output-dir outputs_transferability --run-id "$RUN_ID/adas_wb_to_pc" \
  --search-source outputs_adas_reproduce/adas_repaired_browsecomp_workbench_10val30test_T1_20260726/workbench/search/search_results.json \
  "${COMMON[@]}"

run_job agentsquare_pc_to_wb "$ROOT" "$ROOT/.venv/bin/python" -m reproduce.agentsquare.run_existing_benchmarks \
  --config config/reproduce_agentsquare.toml --benchmark workbench \
  --output-dir outputs_transferability --run-id "$RUN_ID/agentsquare_pc_to_wb" \
  --search-source outputs_agentsquare_reproduce/agentsquare_plancraft_fair_nosearch_valsmall_max30_20260727/plancraft/search/search_results.json \
  "${COMMON[@]}"

run_job agentsquare_wb_to_pc "$ROOT" "$ROOT/.venv/bin/python" -m reproduce.agentsquare.run_existing_benchmarks \
  --config config/reproduce_plancraft_fair.toml --benchmark plancraft \
  --output-dir outputs_transferability --run-id "$RUN_ID/agentsquare_wb_to_pc" \
  --search-source outputs_agentsquare_reproduce/agentsquare_gemma_10val_30test_T1_fastsearch_noenv_20260719T172820/workbench/search/search_results.json \
  "${COMMON[@]}"

MASS_COMMON=(--task-limit 40 --validation-task-limit 0 --final-task-offset 10
  --final-task-limit 30 --final-evaluation-repeats 1 --num-workers "$WORKERS"
  --temperature 1 --max-tokens 0 --timeout-s 600 --seed 42 --resume)

run_job mass_pc_to_wb "$ROOT" "$ROOT/.venv/bin/python" -m reproduce.mass.run_existing_benchmarks \
  --config config/reproduce_agentsquare.toml --benchmark workbench \
  --output-dir outputs_transferability --run-id "$RUN_ID/mass_pc_to_wb" \
  --prompt-search-source outputs_mass_reproduce/mass_official_plancraft_rescued_final_20260706T130548Z/plancraft \
  "${MASS_COMMON[@]}"

run_job mass_wb_to_pc "$ROOT" "$ROOT/.venv/bin/python" -m reproduce.mass.run_existing_benchmarks \
  --config config/reproduce_plancraft_fair.toml --benchmark plancraft \
  --output-dir outputs_transferability --run-id "$RUN_ID/mass_wb_to_pc" \
  --prompt-search-source outputs_mass_reproduce/mass_official_workbench_20260705T062551Z/workbench \
  "${MASS_COMMON[@]}"

AFLOW_COMMON=(--task-limit 40 --test-task-limit 30 --test-offset 10
  --runs-per-task 1 --workers "$WORKERS" --retries 2 --temperature 1 --seed 42
  --env-file "$ROOT/.env")

run_job aflow_pc_to_wb "$AFLOW_ROOT" "$AFLOW_ROOT/.venv/bin/python" \
  -m reproduce.aflow.evaluate_best_workflow --config "$ROOT/config/reproduce_agentsquare.toml" \
  --benchmark workbench --output-dir "$ROOT/outputs_transferability" \
  --run-id "$RUN_ID/aflow_pc_to_wb" --round-number 4 \
  --workflow-dir outputs_aflow_reproduce/aflow_plancraft_10val30test_T1_gemma_20260727/plancraft/workflows/round_4 \
  "${AFLOW_COMMON[@]}"

run_job aflow_wb_to_pc "$AFLOW_ROOT" "$AFLOW_ROOT/.venv/bin/python" \
  -m reproduce.aflow.evaluate_best_workflow --config "$ROOT/config/reproduce_plancraft_fair.toml" \
  --benchmark plancraft --output-dir "$ROOT/outputs_transferability" \
  --run-id "$RUN_ID/aflow_wb_to_pc" --round-number 5 \
  --workflow-dir outputs_aflow_reproduce/aflow_official_gemma_10val_30test_T1_promptfix_clientfix_workbench_workflowhint_w8_setsid/workbench/workflows/round_5 \
  "${AFLOW_COMMON[@]}"

printf '[%s] ALL JOBS FINISHED\n' "$(date -Is)" | tee -a "$LOG_ROOT/driver.log"

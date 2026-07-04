#!/usr/bin/env bash
#
# Re-run ONLY the WorkBench benchmark and REPLACE the workbench data in the
# existing gemma-4-31b experiment folders, after the CRM-tool fix.
#
# Why: multi_domain WorkBench tasks label the CRM domain "crm", but the tool
# specs use "customer_relationship_manager". With tool_selection="domains" the
# adapter dropped every CRM tool for those tasks, so agents could never look up
# a customer's assignee and the crm/* multi_domain tasks were unwinnable. Fixed
# in benchmark/workbench.py via DOMAIN_ALIASES {"crm": "customer_relationship_manager"}.
#
# This wrapper regenerates the workbench arm for:
#   1. Static MAS + SAS  -> artifacts/full_experiment/20260427T134706Z__<model_slug>/workbench/
#        (7 systems: sas, orchestrator_tree_structure, orchestrator_no_discussion,
#         orchestrator_with_discussion, only_voting, fully_linked_debate, group_chat_debate)
#   2. Self-evolved MAS  -> artifacts/full_experiment/full_selfevo_bw__<model_slug>/workbench/
#        (delegates to scripts/full_selfevo_bw.sh scoped to workbench)
#
# The third folder you mentioned, full_selfevo_ps__<model_slug>, contains only
# plancraft + stabletoolbench (the "ps" self-evolved run) and has NO workbench
# data, so there is nothing to replace there — it is intentionally skipped.
#
# "Replace" semantics: the batch driver SKIPS systems that already have a clean
# summary.json, so this script DELETES each target's workbench/ subtree first,
# then re-runs. Only the workbench/ subdir is removed; the other benchmarks in
# each folder (browsecomp, plancraft, stabletoolbench, finance_agent, ...) are
# left untouched, and the end-of-run `summarize-experiment` re-scans the whole
# folder so the top-level experiment_summary.{json,csv} rollup stays complete.
#
# It reuses the exact same drivers/settings as the originals:
#   - scripts/full_experiment.sh  (static/SAS arms + all sampling/provider/retry env)
#   - scripts/full_selfevo_bw.sh  (self-evolved arm + online skill learning)
# so the only intended change vs. the original runs is the CRM-tool fix.
#
# ---------------------------------------------------------------------------
# Model note (important):
#   In the current folders the workbench arm is a MERGE of two routings — three
#   systems ran on "google/gemma-4-31b-it:nitro" and four on the plain
#   "google/gemma-4-31b-it". The plain-routing sibling folder does not exist on
#   disk (its data was merged into the *_nitro folder). To keep every re-run in
#   the *_nitro folder you asked to replace, all systems are re-run under a
#   single routing: MODEL (default :nitro). nitro vs non-nitro is the same model
#   weights, only OpenRouter provider routing differs. Override with MODEL=... if
#   you want the plain routing instead (that will write to a *_google_gemma_4_31b_it
#   folder, not the *_nitro one).
# ---------------------------------------------------------------------------
#
# Usage:
#   bash scripts/rerun_workbench_crm_fix.sh                       # full: 30 tasks x 3 runs, both folders
#   DRY_RUN=1 bash scripts/rerun_workbench_crm_fix.sh            # print the plan, delete/run nothing
#   TASK_LIMIT=2 RUNS_PER_TASK=1 bash scripts/rerun_workbench_crm_fix.sh   # quick smoke
#   TARGETS=static  bash scripts/rerun_workbench_crm_fix.sh      # only the static/SAS folder
#   TARGETS=selfevo bash scripts/rerun_workbench_crm_fix.sh      # only the self-evolved folder
#   MODEL="google/gemma-4-31b-it" bash scripts/rerun_workbench_crm_fix.sh  # plain routing
#   KEEP_OLD=1 bash scripts/rerun_workbench_crm_fix.sh           # do NOT delete; rely on needs-rerun/skip
#
# Requires: OPENROUTER_API_KEY in the environment / .env (real LLM calls, hours of compute).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# --- knobs -----------------------------------------------------------------
MODEL="${MODEL:-google/gemma-4-31b-it:nitro}"
TASK_LIMIT="${TASK_LIMIT:-30}"
RUNS_PER_TASK="${RUNS_PER_TASK:-3}"
TARGETS="${TARGETS:-static,selfevo}"          # which arms to regenerate
STATIC_EXPERIMENT_ID="${STATIC_EXPERIMENT_ID:-20260427T134706Z}"
SELFEVO_EXPERIMENT_ID="${SELFEVO_EXPERIMENT_ID:-full_selfevo_bw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/full_experiment}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_OLD="${KEEP_OLD:-0}"

# The 7 static baselines + SAS (everything in SYSTEMS except self_evolved).
STATIC_SYSTEMS="sas,orchestrator_tree_structure,orchestrator_no_discussion,orchestrator_with_discussion,only_voting,fully_linked_debate,group_chat_debate"

# Same slugging rule as scripts/full_experiment.py::_model_slug.
model_slug() {
  python3 - "$1" <<'PY'
import sys
model = sys.argv[1]
text = "".join(ch.lower() if ch.isalnum() else "_" for ch in model.strip())
print("_".join(p for p in text.split("_") if p) or "model")
PY
}

MODEL_SLUG="$(model_slug "${MODEL}")"
STATIC_ROOT="${OUTPUT_ROOT}/${STATIC_EXPERIMENT_ID}__${MODEL_SLUG}"
SELFEVO_ROOT="${OUTPUT_ROOT}/${SELFEVO_EXPERIMENT_ID}__${MODEL_SLUG}"

log() { echo "[rerun-workbench] $*" >&2; }

wants() { [[ ",${TARGETS}," == *",$1,"* ]]; }

# Delete only the workbench/ subtree of an experiment folder (the "replace" step).
replace_prep() {
  local root="$1"
  local wb="${root}/workbench"
  if [[ "${KEEP_OLD}" == "1" ]]; then
    log "KEEP_OLD=1 -> not deleting ${wb} (driver will skip clean systems / rerun only needs-rerun)"
    return
  fi
  if [[ -d "${wb}" ]]; then
    log "REPLACE: removing existing workbench data at ${wb}"
    [[ "${DRY_RUN}" == "1" ]] || rm -rf "${wb}"
  else
    log "no existing workbench data at ${wb} (fresh generate)"
  fi
  # stale per-benchmark logs from the batch driver, if any
  local wblog="${root}/logs/workbench"
  if [[ -d "${wblog}" && "${KEEP_OLD}" != "1" ]]; then
    [[ "${DRY_RUN}" == "1" ]] || rm -rf "${wblog}"
  fi
}

log "repo=${REPO_ROOT}"
log "MODEL=${MODEL} (slug=${MODEL_SLUG})  TASK_LIMIT=${TASK_LIMIT}  RUNS_PER_TASK=${RUNS_PER_TASK}  TARGETS=${TARGETS}"
log "static  -> ${STATIC_ROOT}/workbench"
log "selfevo -> ${SELFEVO_ROOT}/workbench"
if [[ "${DRY_RUN}" == "1" ]]; then log "DRY_RUN=1 -> nothing will be deleted or executed"; fi

# ---------------------------------------------------------------------------
# 1) Static MAS + SAS
# ---------------------------------------------------------------------------
if wants static; then
  log "=== [1/2] static MAS + SAS (workbench) ==="
  replace_prep "${STATIC_ROOT}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    ONLY_SYSTEMS="${STATIC_SYSTEMS}" \
    MODELS="${MODEL}" \
    BENCHMARKS="workbench" \
    TASK_LIMIT="${TASK_LIMIT}" \
    RUNS_PER_TASK="${RUNS_PER_TASK}" \
    EXPERIMENT_ID="${STATIC_EXPERIMENT_ID}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
      bash "${REPO_ROOT}/scripts/full_experiment.sh"
  else
    log "DRY_RUN plan: ONLY_SYSTEMS=${STATIC_SYSTEMS} MODELS=${MODEL} BENCHMARKS=workbench \\"
    log "              TASK_LIMIT=${TASK_LIMIT} RUNS_PER_TASK=${RUNS_PER_TASK} EXPERIMENT_ID=${STATIC_EXPERIMENT_ID} \\"
    log "              bash scripts/full_experiment.sh"
  fi
fi

# ---------------------------------------------------------------------------
# 2) Self-evolved MAS  (delegates to the canonical bw runner, scoped to workbench)
#    full_selfevo_bw.sh forces ONLY_SYSTEMS=self_evolved, EXPERIMENT_ID=full_selfevo_bw,
#    MAX_PARALLEL=1, and the online skill-learning args. We only narrow BENCHMARKS
#    to workbench so browsecomp in that folder is left intact.
#    NOTE: online skill learning (SKILL_UPDATE_BATCH_SIZE, default 8) rewrites
#    config/topology_skill.md during the run — set SKILL_UPDATE_BATCH_SIZE=0 to disable.
# ---------------------------------------------------------------------------
if wants selfevo; then
  log "=== [2/2] self-evolved MAS (workbench) ==="
  replace_prep "${SELFEVO_ROOT}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    MODELS="${MODEL}" \
    BENCHMARKS="workbench" \
    TASK_LIMIT="${TASK_LIMIT}" \
    RUNS_PER_TASK="${RUNS_PER_TASK}" \
    EXPERIMENT_ID="${SELFEVO_EXPERIMENT_ID}" \
      bash "${REPO_ROOT}/scripts/full_selfevo_bw.sh"
  else
    log "DRY_RUN plan: MODELS=${MODEL} BENCHMARKS=workbench TASK_LIMIT=${TASK_LIMIT} \\"
    log "              RUNS_PER_TASK=${RUNS_PER_TASK} EXPERIMENT_ID=${SELFEVO_EXPERIMENT_ID} \\"
    log "              bash scripts/full_selfevo_bw.sh"
  fi
fi

log "done."

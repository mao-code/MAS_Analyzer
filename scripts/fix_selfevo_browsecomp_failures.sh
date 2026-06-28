#!/usr/bin/env bash
#
# Fix-VALIDATION re-run for browsecomp self_evolved failures.
#
# WHAT THIS IS (and is NOT):
#   This is a DIAGNOSTIC that re-runs ONLY the previously-failed browsecomp runs
#   with the read-net fix applied (MAS/self_evolved/engine.py _needs_final_synthesis:
#   the "Unable to determine a supported final answer" sentinel now triggers the
#   read-net instead of being mistaken for a real answer). It writes to a SEPARATE
#   folder and reports whether the failures flip to success, BROKEN DOWN by failure
#   category so the real fix signal is not conflated with resampling luck.
#
#   It is NOT a paper number. Re-running only failures and counting successes mixes
#   the ~5 runs the fix can legitimately recover (null answer + gold retrieved) with
#   regression-to-the-mean flips among the ~30 runs the finalize fix cannot touch.
#   For a defensible headline, re-run the FULL task set with the fix in a fresh
#   folder (e.g. EXPERIMENT_ID=full_selfevo_bw_fixed bash scripts/full_selfevo_bw.sh)
#   and report THAT, with one code version across every run. Do not splice these
#   recovered failures back into the headline run's success rate.
#
# HOW IT WORKS:
#   1. preseed: copy the PASSING runs of each failed task into the fix-check folder
#      so the harness resume logic skips them (no cost); omit the failed runs so the
#      harness re-executes exactly those with the fixed code.
#   2. run: delegate to scripts/full_experiment.sh with the SAME canonical settings
#      as the headline run (model, provider order, sampling, self_evolved 5-agent /
#      max_turns 2 row), scoped via --task-ids to the failed tasks, skill learning
#      OFF (SKILL_UPDATE_BATCH_SIZE=0) so the code fix is the ONLY changed variable.
#   3. report: compare old vs new per run, grouped by fix_target / gold_wrong /
#      low_recall. Only fix_target flips are attributable to the fix.
#
# NOTE: the fix-check folder's own summary.csv mixes old-code (resumed) passers with
#   new-code (re-run) failures and is NOT a valid success rate. The printed report is
#   the authoritative output.
#
# Usage:
#   bash scripts/fix_selfevo_browsecomp_failures.sh            # preseed + run + report
#   REPORT_ONLY=1 bash scripts/fix_selfevo_browsecomp_failures.sh   # just re-print the report
#   DRY_RUN=1 bash scripts/fix_selfevo_browsecomp_failures.sh       # preseed only, no LLM run
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  set -u
else
  echo "WARNING: .venv not found; relying on full_experiment.sh python detection." >&2
fi

# --- fixed paths for THIS run (gemma-4-31b browsecomp self_evolved) ---
MODEL_SLUG="google_gemma_4_31b_it_nitro"
HEADLINE_ID="full_selfevo_bw"
FIXCHECK_ID="${EXPERIMENT_ID:-full_selfevo_bw_fixcheck}"
ART="${REPO_ROOT}/artifacts/full_experiment"
SRC="${ART}/${HEADLINE_ID}__${MODEL_SLUG}/browsecomp/self_evolved"
DST="${ART}/${FIXCHECK_ID}__${MODEL_SLUG}/browsecomp/self_evolved"

HELPER="${REPO_ROOT}/scripts/fixcheck_browsecomp_failures.py"

if [[ ! -d "${SRC}" ]]; then
  echo "ERROR: headline browsecomp run not found: ${SRC}" >&2
  exit 2
fi

# Report-only short-circuit (re-print after a completed run).
if [[ "${REPORT_ONLY:-0}" == "1" ]]; then
  exec python3 "${HELPER}" report --src "${SRC}" --dst "${DST}"
fi

echo "[fixcheck] === STEP 1/3: pre-seed passers, clear failed runs ==="
python3 "${HELPER}" preseed --src "${SRC}" --dst "${DST}"

TASK_IDS="$(python3 "${HELPER}" taskids --src "${SRC}")"
if [[ -z "${TASK_IDS}" ]]; then
  echo "[fixcheck] no failed runs found in ${SRC}; nothing to re-run." >&2
  exit 0
fi
echo "[fixcheck] re-running task ids: ${TASK_IDS}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[fixcheck] DRY_RUN=1 set — pre-seed done, skipping the LLM re-run." >&2
  exit 0
fi

echo "[fixcheck] === STEP 2/3: re-run failed runs with the fix (skill learning OFF) ==="
# Delegate to the canonical driver, scoped to the failed tasks. Skill learning is OFF
# so the read-net code fix is the only changed variable vs the headline run.
export ONLY_SYSTEMS="self_evolved"
export MODELS="google/gemma-4-31b-it:nitro"
export BENCHMARKS="browsecomp"
export TASK_IDS
export RUNS_PER_TASK="${RUNS_PER_TASK:-3}"
export EXPERIMENT_ID="${FIXCHECK_ID}"
export SKILL_UPDATE_BATCH_SIZE="0"
export SELF_EVOLVED_ARGS="${SELF_EVOLVED_ARGS:-} --skill-update-batch-size 0"
export MAX_PARALLEL="1"
export SKIP_SETUP="${SKIP_SETUP:-1}"

echo "[fixcheck] ONLY_SYSTEMS=${ONLY_SYSTEMS} BENCHMARKS=${BENCHMARKS} EXPERIMENT_ID=${EXPERIMENT_ID} TASK_IDS=${TASK_IDS}" >&2
echo "[fixcheck] SKILL_UPDATE_BATCH_SIZE=0 (skill OFF) MAX_PARALLEL=1 SKIP_SETUP=${SKIP_SETUP}" >&2

# NOTE: plain `bash` (not `exec`) so control returns here for the report step.
set +e
bash "${REPO_ROOT}/scripts/full_experiment.sh"
RUN_RC=$?
set -e
if [[ "${RUN_RC}" != "0" ]]; then
  echo "[fixcheck] WARNING: re-run exited rc=${RUN_RC}; reporting whatever completed." >&2
fi

echo "[fixcheck] === STEP 3/3: report old vs new by failure category ==="
python3 "${HELPER}" report --src "${SRC}" --dst "${DST}"

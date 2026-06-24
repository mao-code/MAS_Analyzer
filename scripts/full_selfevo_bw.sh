#!/usr/bin/env bash
#
# Re-run the self-evolved topology experiment with the SAME settings as
#   artifacts/full_experiment/full_selfevo__google_gemma_4_31b_it_nitro
# but scoped to browsecomp + workbench only (the two benchmarks of interest).
#
# It delegates to scripts/full_experiment.sh, so it inherits the *exact same
# driver and canonical settings* as the original run:
#   - self_evolved SYSTEMS row: 5 agents, max_turns 2, discussion_rounds 1,
#     communication budget 2          (scripts/full_experiment.py:39)
#   - OpenRouter sampling / provider order / retry / tool-context env defaults
#     (temperature 1.0, provider order DeepInfra,Chutes,Novita,Together, etc.)
#                                       (scripts/full_experiment.sh)
#   - hierarchical layout; benchmark-specific tool configs from
#     config/benchmarks/{browsecomp,workbench}_10.toml
#     (browsecomp: tools on, tool_k 5, get_document, max_tool_iterations 6;
#      workbench: multi_domain, max_tool_iterations 20)
#
# Only three things differ from the full run: the system set (self_evolved only,
# no static baseline arms), the benchmark subset, and the experiment id.
#
# Output lands at:
#   artifacts/full_experiment/<EXPERIMENT_ID>__google_gemma_4_31b_it_nitro/
#       {browsecomp,workbench}/self_evolved/
# right next to the existing static baseline
#   artifacts/full_experiment/20260427T134706Z__google_gemma_4_31b_it_nitro/
# so the two are directly comparable (same tasks, same 3 runs, same model).
#
# Usage:
#   bash scripts/full_selfevo_bw.sh                              # full: 30 tasks x 3 runs
#   TASK_LIMIT=2 RUNS_PER_TASK=1 bash scripts/full_selfevo_bw.sh # quick smoke
#   SKIP_SETUP=1 bash scripts/full_selfevo_bw.sh                 # skip benchmark data setup
#   BENCHMARKS=browsecomp bash scripts/full_selfevo_bw.sh        # one benchmark
#   EXPERIMENT_ID=my_rerun bash scripts/full_selfevo_bw.sh       # custom id
# Any extra args are passed through to scripts/full_experiment.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Use the repo's virtualenv interpreter (3.14, with deps installed).
# full_experiment.sh probes for a >=3.11 python (conda 'agents' / python3.11 /
# python3); on this machine the bare python3 is too old, so we activate .venv to
# provide a compatible `python3` that full_experiment.sh will pick up.
if [[ -f ".venv/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  set -u
else
  echo "WARNING: .venv not found; relying on full_experiment.sh python detection." >&2
fi

# --- the only deltas from the original full_selfevo run ---
export ONLY_SYSTEMS="${ONLY_SYSTEMS:-self_evolved}"     # self-evolved only (no static arms)
export MODELS="${MODELS:-google/gemma-4-31b-it:nitro}"  # same model as the baseline
export BENCHMARKS="${BENCHMARKS:-browsecomp,workbench}"  # the two benchmarks of interest
export TASK_LIMIT="${TASK_LIMIT:-30}"                    # same task count
export RUNS_PER_TASK="${RUNS_PER_TASK:-3}"               # same repeats per task
export EXPERIMENT_ID="${EXPERIMENT_ID:-full_selfevo_bw}" # -> full_selfevo_bw__google_gemma_4_31b_it_nitro

echo "[full_selfevo_bw] python=$(command -v python3) ($(python3 -c 'import sys;print(sys.version.split()[0])'))" >&2
echo "[full_selfevo_bw] ONLY_SYSTEMS=${ONLY_SYSTEMS} MODELS=${MODELS} BENCHMARKS=${BENCHMARKS} TASK_LIMIT=${TASK_LIMIT} RUNS_PER_TASK=${RUNS_PER_TASK} EXPERIMENT_ID=${EXPERIMENT_ID}" >&2

exec bash "${REPO_ROOT}/scripts/full_experiment.sh" "$@"

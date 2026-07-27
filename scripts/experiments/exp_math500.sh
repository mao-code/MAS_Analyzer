#!/usr/bin/env bash
# =============================================================================
# MATH-500 experiment runner: SAS + all static MAS topologies + MANTA.
#
# Thin wrapper over scripts/full_experiment.sh, pinned to the `math500` benchmark
# with the SAME run settings as the other benchmarks (30 tasks x 3 runs by
# default). The batch driver already enumerates all 8 systems, so this runs:
#   sas
#   orchestrator_tree_structure
#   orchestrator_no_discussion
#   orchestrator_with_discussion
#   only_voting
#   fully_linked_debate
#   group_chat_debate
#   self_evolved            (MANTA)
#
# Everything is overridable via env. Common examples:
#   # full run: all 8 systems, 30 tasks x 3 runs (standard model routing)
#   bash scripts/experiments/exp_math500.sh
#
#   # MANTA-only correctness smoke: 10 tasks x 1 run, one model
#   ONLY_SYSTEMS=self_evolved TASK_LIMIT=10 RUNS_PER_TASK=1 \
#     MODELS=google/gemini-3-flash-preview bash scripts/experiments/exp_math500.sh
#
#   # SAS + MANTA only
#   ONLY_SYSTEMS=sas,self_evolved bash scripts/experiments/exp_math500.sh
#
# Notes:
#   - ONLY_SYSTEMS filters the system list (comma-separated exact labels above).
#   - MANTA online long-term skill-learning is ENABLED by default (the general MANTA
#     default: skill_update_batch_size = 12, ≈4 tasks × 3 runs). Every 12 completed
#     self_evolved runs are reflected into config/topology_skill.md and reloaded, so
#     the skill self-evolves DURING the run and that tracked file WILL change — commit
#     or stash it first if you want a clean learning delta. This is race-free here
#     because it is a single benchmark: at most one self_evolved process exists at a
#     time (models run sequentially; one self_evolved job per model). To disable, set
#     SELF_EVOLVED_ARGS="--skill-update-batch-size 0" (parallel-safe, no drift).
#     ⚠ If you extend this to MULTIPLE benchmarks at once, several self_evolved jobs
#     would write the shared skill concurrently — keep learning off or serialize then.
#   - Any extra flags are passed straight through to full_experiment.sh /
#     full_experiment.py, e.g.:  bash scripts/experiments/exp_math500.sh --max-parallel 2
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Secrets: load .env (OPENROUTER_API_KEY etc.) without echoing it.
if [[ -z "${OPENROUTER_API_KEY:-}" && -f ".env" ]]; then set -a; . ./.env; set +a; fi

# ---- Run settings (all overridable via env) --------------------------------
export BENCHMARKS="math500"                    # pinned benchmark
export TASK_LIMIT="${TASK_LIMIT:-30}"          # same as other benchmarks
export RUNS_PER_TASK="${RUNS_PER_TASK:-3}"     # same as other benchmarks

# STABLE experiment id => RESUMABLE. Without this, full_experiment.py defaults
# --experiment-id to a fresh UTC timestamp per invocation, so a re-run would land
# in a new folder and resume nothing. Pinning it means kill + re-run the SAME
# command continues where it left off: the driver skips already-clean (system)
# jobs and main.py skips completed tasks/runs inside a partially-done system.
# Override EXPERIMENT_ID to run a separate, independent experiment.
export EXPERIMENT_ID="${EXPERIMENT_ID:-math500}"

# Online long-term skill-learning is ON by default (inherits the MANTA config default,
# skill_update_batch_size = 12). full_experiment.sh does NOT export SELF_EVOLVED_ARGS
# itself, so set+export here; empty default => no override => config default (12) applies.
# Set SELF_EVOLVED_ARGS="--skill-update-batch-size 0" to disable (see header note).
export SELF_EVOLVED_ARGS="${SELF_EVOLVED_ARGS:-}"

# ONLY_SYSTEMS is read by full_experiment.py via the inherited environment.
if [[ -n "${ONLY_SYSTEMS:-}" ]]; then export ONLY_SYSTEMS; fi

echo "[exp_math500] benchmark=math500 task_limit=${TASK_LIMIT} runs_per_task=${RUNS_PER_TASK}" \
     "systems=${ONLY_SYSTEMS:-all(8)} models=${MODELS:-<full_experiment default>}"

exec bash scripts/full_experiment.sh "$@"

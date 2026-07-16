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
#   bash scripts/exp_math500.sh
#
#   # MANTA-only correctness smoke: 10 tasks x 1 run, one model
#   ONLY_SYSTEMS=self_evolved TASK_LIMIT=10 RUNS_PER_TASK=1 \
#     MODELS=google/gemini-3-flash-preview bash scripts/exp_math500.sh
#
#   # SAS + MANTA only
#   ONLY_SYSTEMS=sas,self_evolved bash scripts/exp_math500.sh
#
# Notes:
#   - ONLY_SYSTEMS filters the system list (comma-separated exact labels above).
#   - MANTA online skill-learning is DISABLED by default here so the tracked
#     long-term skill (config/topology_skill.md) is never rewritten mid-run and
#     the runner stays parallel-safe / idempotent. Set SELF_EVOLVED_ARGS=""
#     (or "--skill-update-batch-size 8") to re-enable online learning — but then
#     use a single sequential process (ONLY_SYSTEMS=self_evolved, MAX_PARALLEL=1).
#   - Any extra flags are passed straight through to full_experiment.sh /
#     full_experiment.py, e.g.:  bash scripts/exp_math500.sh --max-parallel 2
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Secrets: load .env (OPENROUTER_API_KEY etc.) without echoing it.
if [[ -z "${OPENROUTER_API_KEY:-}" && -f ".env" ]]; then set -a; . ./.env; set +a; fi

# ---- Run settings (all overridable via env) --------------------------------
export BENCHMARKS="math500"                    # pinned benchmark
export TASK_LIMIT="${TASK_LIMIT:-30}"          # same as other benchmarks
export RUNS_PER_TASK="${RUNS_PER_TASK:-3}"     # same as other benchmarks

# Keep the tracked long-term skill clean by default (see header note).
# full_experiment.sh does NOT export SELF_EVOLVED_ARGS itself, so set+export here.
export SELF_EVOLVED_ARGS="${SELF_EVOLVED_ARGS:---skill-update-batch-size 0}"

# ONLY_SYSTEMS is read by full_experiment.py via the inherited environment.
if [[ -n "${ONLY_SYSTEMS:-}" ]]; then export ONLY_SYSTEMS; fi

echo "[exp_math500] benchmark=math500 task_limit=${TASK_LIMIT} runs_per_task=${RUNS_PER_TASK}" \
     "systems=${ONLY_SYSTEMS:-all(8)} models=${MODELS:-<full_experiment default>}"

exec bash scripts/full_experiment.sh "$@"

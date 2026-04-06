#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper defaults. Override these with environment variables or CLI flags.
# Examples:
#   TASK_LIMIT=2 RUNS_PER_TASK=1 bash scripts/full_experiment.sh
#   BENCHMARKS=workbench,scicode SKIP_SETUP=1 bash scripts/full_experiment.sh
#   FINAL_VOTE_MODE=deterministic bash scripts/full_experiment.sh
#   DISABLE_DYNAMIC_ROLES=1 bash scripts/full_experiment.sh   # use structural roles only

TASK_LIMIT="${TASK_LIMIT:-3}"
RUNS_PER_TASK="${RUNS_PER_TASK:-1}"
BENCHMARKS="${BENCHMARKS:-browsecomp,workbench}"
RETRY_FAILURES="${RETRY_FAILURES:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}" # A "job" here is one (benchmark, system) pair, not one individual task.
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
CONFIG_DIR="${CONFIG_DIR:-}"
SKIP_SETUP="${SKIP_SETUP:-0}"
SETUP_ONLY="${SETUP_ONLY:-0}"
FINAL_VOTE_MODE="${FINAL_VOTE_MODE:-}"
# Set DISABLE_DYNAMIC_ROLES=1 to skip LLM-based role assignment (uses structural roles only).
DISABLE_DYNAMIC_ROLES="${DISABLE_DYNAMIC_ROLES:-0}"

# ============================================================================
# Global MAS override args
# These are appended to every `main.py run ...` command launched by
# `scripts/full_experiment.py`.
#
# Example:
# MAS_GLOBAL_ARGS="--default-model google/gemini-3-flash-preview --judge-model google/gemini-3-flash-preview --peer-artifact-max-chars 240"
# ============================================================================
# google/gemini-3.1-flash-lite-preview:nitro
# google/gemini-3-flash-preview
MAS_GLOBAL_ARGS="${MAS_GLOBAL_ARGS:---default-model google/gemini-3.1-flash-lite-preview:nitro --judge-model google/gemini-3.1-flash-lite-preview:nitro}"

# ============================================================================
# SAS
# Example:
# SAS_ARGS="--judge-model openai/gpt-4.1-mini --peer-artifact-max-chars 200"
# ============================================================================
SAS_ARGS="${SAS_ARGS:-}"

# ============================================================================
# Orchestrator Tree Structure
# Example:
# ORCHESTRATOR_TREE_STRUCTURE_ARGS="--termination-consensus-mode llm_judge --judge-model openai/gpt-4.1-mini --peer-artifact-max-chars 240"
# ============================================================================
ORCHESTRATOR_TREE_STRUCTURE_ARGS="${ORCHESTRATOR_TREE_STRUCTURE_ARGS:-}"

# ============================================================================
# Orchestrator No Discussion
# Example:
# ORCHESTRATOR_NO_DISCUSSION_ARGS="--termination-consensus-mode llm_judge --judge-model openai/gpt-4.1-mini --peer-artifact-max-chars 240"
# ============================================================================
ORCHESTRATOR_NO_DISCUSSION_ARGS="${ORCHESTRATOR_NO_DISCUSSION_ARGS:-}"

# ============================================================================
# Orchestrator With Discussion
# Example:
# ORCHESTRATOR_WITH_DISCUSSION_ARGS="--termination-consensus-mode llm_judge --judge-model openai/gpt-4.1-mini --peer-artifact-max-chars 220"
# ============================================================================
ORCHESTRATOR_WITH_DISCUSSION_ARGS="${ORCHESTRATOR_WITH_DISCUSSION_ARGS:-}"

# ============================================================================
# Only Voting
# Example:
# ONLY_VOTING_ARGS="--judge-model openai/gpt-4.1-mini --peer-artifact-max-chars 200"
# ============================================================================
ONLY_VOTING_ARGS="${ONLY_VOTING_ARGS:-}"

# ============================================================================
# Fully Linked Debate
# Example:
# FULLY_LINKED_DEBATE_ARGS="--termination-consensus-mode llm_judge --judge-model openai/gpt-4.1 --peer-artifact-max-chars 220"
# ============================================================================
FULLY_LINKED_DEBATE_ARGS="${FULLY_LINKED_DEBATE_ARGS:-}"

# ============================================================================
# Group Chat Debate
# Example:
# GROUP_CHAT_DEBATE_ARGS="--termination-consensus-mode llm_judge --judge-model openai/gpt-4.1 --peer-artifact-max-chars 220"
# ============================================================================
GROUP_CHAT_DEBATE_ARGS="${GROUP_CHAT_DEBATE_ARGS:-}"

args=()

if [[ -n "${TASK_LIMIT}" ]]; then
  args+=(--task-limit "${TASK_LIMIT}")
fi
if [[ -n "${RUNS_PER_TASK}" ]]; then
  args+=(--runs-per-task "${RUNS_PER_TASK}")
fi
if [[ -n "${BENCHMARKS}" ]]; then
  args+=(--benchmarks "${BENCHMARKS}")
fi
if [[ -n "${RETRY_FAILURES}" ]]; then
  args+=(--retry-failures "${RETRY_FAILURES}")
fi
if [[ -n "${MAX_PARALLEL}" ]]; then
  args+=(--max-parallel "${MAX_PARALLEL}")
fi
if [[ -n "${EXPERIMENT_ID}" ]]; then
  args+=(--experiment-id "${EXPERIMENT_ID}")
fi
if [[ -n "${OUTPUT_ROOT}" ]]; then
  args+=(--output-root "${OUTPUT_ROOT}")
fi
if [[ -n "${CONFIG_DIR}" ]]; then
  args+=(--config-dir "${CONFIG_DIR}")
fi
if [[ "${SKIP_SETUP}" == "1" ]]; then
  args+=(--skip-setup)
fi
if [[ "${SETUP_ONLY}" == "1" ]]; then
  args+=(--setup-only)
fi
if [[ -n "${FINAL_VOTE_MODE}" ]]; then
  args+=(--final-vote-mode "${FINAL_VOTE_MODE}")
fi
if [[ "${DISABLE_DYNAMIC_ROLES}" == "1" ]]; then
  args+=(--no-dynamic-roles)
fi

export MAS_GLOBAL_ARGS
export SAS_ARGS
export ORCHESTRATOR_TREE_STRUCTURE_ARGS
export ORCHESTRATOR_NO_DISCUSSION_ARGS
export ORCHESTRATOR_WITH_DISCUSSION_ARGS
export ONLY_VOTING_ARGS
export FULLY_LINKED_DEBATE_ARGS
export GROUP_CHAT_DEBATE_ARGS

python_is_compatible() {
  local python_cmd="${1}"
  "${python_cmd}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

if command -v conda >/dev/null 2>&1; then
  if conda run -n agents python -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then
    exec conda run -n agents python scripts/full_experiment.py "${args[@]}" "$@"
  fi
fi

if command -v python3.11 >/dev/null 2>&1; then
  exec python3.11 scripts/full_experiment.py "${args[@]}" "$@"
fi

if command -v python3 >/dev/null 2>&1 && python_is_compatible python3; then
  exec python3 scripts/full_experiment.py "${args[@]}" "$@"
fi

if command -v python >/dev/null 2>&1 && python_is_compatible python; then
  exec python scripts/full_experiment.py "${args[@]}" "$@"
fi

cat >&2 <<'EOF'
scripts/full_experiment.sh requires Python 3.11+.

Recommended fix:
  conda env create -f environment.yml
  conda run -n agents python -V
  bash scripts/full_experiment.sh

If the 'agents' environment already exists but is stale, recreate or update it:
  conda env remove -n agents
  conda env create -f environment.yml
EOF
exit 1

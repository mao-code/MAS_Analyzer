#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper defaults. Override these with environment variables or CLI flags.
# Examples:
#   TASK_LIMIT=2 RUNS_PER_TASK=1 bash scripts/full_experiment.sh
#   BENCHMARKS=workbench,scicode SKIP_SETUP=1 bash scripts/full_experiment.sh
#   bash scripts/full_experiment.sh --benchmark workbench --benchmark scicode
#   RUNS_PER_TASK=8 bash scripts/full_experiment.sh --benchmarks browsecomp,workbench
#   FINAL_VOTE_MODE=deterministic bash scripts/full_experiment.sh
#   DISABLE_DYNAMIC_ROLES=1 bash scripts/full_experiment.sh   # use structural roles only
#   MODELS="qwen/qwen3-32b" bash scripts/full_experiment.sh

TASK_LIMIT="${TASK_LIMIT:-30}"
RUNS_PER_TASK="${RUNS_PER_TASK:-3}"
# BENCHMARKS="${BENCHMARKS:-workbench,scicode,browsecomp,plancraft,webshop,agentbench,stabletoolbench}"
BENCHMARKS="${BENCHMARKS:-browsecomp,plancraft,stabletoolbench,workbench}"
RETRY_FAILURES="${RETRY_FAILURES:-3}"
MAX_PARALLEL="${MAX_PARALLEL:-4}" # A "job" here is one (benchmark, system) pair, not one individual task.
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
CONFIG_DIR="${CONFIG_DIR:-}"

# ============================================================================
# Models to run
# This is the main batch-level model selector for MAS.
# Set a comma-separated list to run a full experiment once per model.
# Each selected model overrides both MAS --default-model and --judge-model.
# Leave empty to use the benchmark/config-defined MAS model routing.
#
# Current OpenRouter model IDs:
#   qwen/qwen3-32b
#
# Example:
# MODELS="qwen/qwen3-32b"
# ============================================================================
MODELS="${MODELS:-google/gemma-4-31b-it:nitro,openai/gpt-oss-120b}"

SKIP_SETUP="${SKIP_SETUP:-0}"
SETUP_ONLY="${SETUP_ONLY:-0}"
FINAL_VOTE_MODE="${FINAL_VOTE_MODE:-}"
BENCHMARK_EVAL_JUDGE_MODEL="${BENCHMARK_EVAL_JUDGE_MODEL:-}"
# Set DISABLE_DYNAMIC_ROLES=1 to skip LLM-based role assignment (uses structural roles only).
DISABLE_DYNAMIC_ROLES="${DISABLE_DYNAMIC_ROLES:-0}"

# ============================================================================
# OpenRouter sampling overrides
# These are exported so MAS OpenRouter calls can apply consistent sampling
# during full experiment runs.
# ============================================================================
OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-medium}"
OPENROUTER_TEMPERATURE="${OPENROUTER_TEMPERATURE:-1.0}"
OPENROUTER_TOP_P="${OPENROUTER_TOP_P:-1.0}"
OPENROUTER_TOP_K="${OPENROUTER_TOP_K:-0}"
MAS_LLM_RETRY_ATTEMPTS="${MAS_LLM_RETRY_ATTEMPTS:-5}"
MAS_LLM_TIMEOUT_RETRY_ATTEMPTS="${MAS_LLM_TIMEOUT_RETRY_ATTEMPTS:-2}"
MAS_LLM_RETRY_BACKOFF_S="${MAS_LLM_RETRY_BACKOFF_S:-8}"
MAS_LLM_RETRY_MAX_BACKOFF_S="${MAS_LLM_RETRY_MAX_BACKOFF_S:-180}"
MAS_OPENROUTER_PROVIDER_ORDER="${MAS_OPENROUTER_PROVIDER_ORDER:-DeepInfra,Chutes,Novita,Together}"
MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER="${MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER:-1}"
MAS_TRANSCRIPT_COMPACTION_ENABLED="${MAS_TRANSCRIPT_COMPACTION_ENABLED:-1}"
MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER="${MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER:-2}"
MAS_TOOL_CONTEXT_RAW_TURNS="${MAS_TOOL_CONTEXT_RAW_TURNS:-2}"
MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS="${MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS:-6000}"
MAS_TOOL_CONTEXT_PREVIEW_CHARS="${MAS_TOOL_CONTEXT_PREVIEW_CHARS:-160}"
MAS_TOOL_CONTEXT_DOCUMENT_CHARS="${MAS_TOOL_CONTEXT_DOCUMENT_CHARS:-12000}"
FULL_EXPERIMENT_PROGRESS_INTERVAL_S="${FULL_EXPERIMENT_PROGRESS_INTERVAL_S:-300}"

# ============================================================================
# Global MAS override args
# These are appended to every `main.py run ...` command launched by
# `scripts/full_experiment.py`.
# Use this for shared non-model MAS flags.
# Do not use this for --default-model / --judge-model; use MODELS instead.
#
# Example:
# MAS_GLOBAL_ARGS="--peer-artifact-max-chars 240"
# ============================================================================
MAS_GLOBAL_ARGS="${MAS_GLOBAL_ARGS:-}"

# ============================================================================
# SAS
# Example:
# SAS_ARGS="--peer-artifact-max-chars 200"
# ============================================================================
SAS_ARGS="${SAS_ARGS:-}"

# ============================================================================
# Orchestrator Tree Structure
# Example:
# ORCHESTRATOR_TREE_STRUCTURE_ARGS="--termination-consensus-mode llm_judge --peer-artifact-max-chars 240"
# ============================================================================
ORCHESTRATOR_TREE_STRUCTURE_ARGS="${ORCHESTRATOR_TREE_STRUCTURE_ARGS:---communication-budget 50}"

# ============================================================================
# Orchestrator No Discussion
# Example:
# ORCHESTRATOR_NO_DISCUSSION_ARGS="--termination-consensus-mode llm_judge --peer-artifact-max-chars 240"
# ============================================================================
ORCHESTRATOR_NO_DISCUSSION_ARGS="${ORCHESTRATOR_NO_DISCUSSION_ARGS:---communication-budget 50}"

# ============================================================================
# Orchestrator With Discussion
# Example:
# ORCHESTRATOR_WITH_DISCUSSION_ARGS="--termination-consensus-mode llm_judge --peer-artifact-max-chars 220"
# ============================================================================
ORCHESTRATOR_WITH_DISCUSSION_ARGS="${ORCHESTRATOR_WITH_DISCUSSION_ARGS:---communication-budget 50}"

# ============================================================================
# Only Voting
# Example:
# ONLY_VOTING_ARGS="--peer-artifact-max-chars 200"
# ============================================================================
ONLY_VOTING_ARGS="${ONLY_VOTING_ARGS:-}"

# ============================================================================
# Fully Linked Debate
# Example:
# FULLY_LINKED_DEBATE_ARGS="--termination-consensus-mode llm_judge --peer-artifact-max-chars 220"
# ============================================================================
FULLY_LINKED_DEBATE_ARGS="${FULLY_LINKED_DEBATE_ARGS:---communication-budget 50}"

# ============================================================================
# Group Chat Debate
# Example:
# GROUP_CHAT_DEBATE_ARGS="--termination-consensus-mode llm_judge --peer-artifact-max-chars 220"
# ============================================================================
GROUP_CHAT_DEBATE_ARGS="${GROUP_CHAT_DEBATE_ARGS:---communication-budget 50}"

args=()
cli_has_benchmark_selection=0

for arg in "$@"; do
  case "${arg}" in
    --benchmark|--benchmark=*|--benchmarks|--benchmarks=*|--list-benchmarks)
      cli_has_benchmark_selection=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/full_experiment.sh [wrapper options passed through to scripts/full_experiment.py]

Defaults:
  - Runs all discovered benchmark configs when BENCHMARKS is unset and no benchmark CLI flags are passed.
  - Uses TASK_LIMIT, RUNS_PER_TASK, BENCHMARKS, and the other environment variables below as wrapper defaults.

Useful examples:
  bash scripts/full_experiment.sh --list-benchmarks
  bash scripts/full_experiment.sh --benchmark workbench --benchmark scicode
  bash scripts/full_experiment.sh --benchmarks browsecomp,workbench
  BENCHMARKS=workbench,scicode bash scripts/full_experiment.sh
  RUNS_PER_TASK=8 bash scripts/full_experiment.sh --benchmarks browsecomp,workbench
  MODELS="google/gemma-4-31b-it,qwen/qwen3.5-flash-02-23,openai/gpt-oss-120b" bash scripts/full_experiment.sh

Environment variable defaults:
  TASK_LIMIT
  RUNS_PER_TASK
  BENCHMARKS
  RETRY_FAILURES
  MAX_PARALLEL
  EXPERIMENT_ID
  OUTPUT_ROOT
  CONFIG_DIR
  MODELS
  SKIP_SETUP
  SETUP_ONLY
  FINAL_VOTE_MODE
  BENCHMARK_EVAL_JUDGE_MODEL
  DISABLE_DYNAMIC_ROLES
  FULL_EXPERIMENT_PROGRESS_INTERVAL_S

Notes:
  - CLI benchmark flags override BENCHMARKS.
  - MODELS is the main batch-level selector for MAS internal models.
  - MAS_GLOBAL_ARGS is for shared non-model MAS CLI args.
  - BENCHMARK_EVAL_JUDGE_MODEL overrides benchmark-side evaluation judge_model, not the MAS internal judge model.
  - FULL_EXPERIMENT_PROGRESS_INTERVAL_S controls live console snapshots; use 0 to disable them.
  - For Python-level help, run: python scripts/full_experiment.py --help
EOF
      exit 0
      ;;
  esac
done

if [[ -n "${TASK_LIMIT}" ]]; then
  args+=(--task-limit "${TASK_LIMIT}")
fi
if [[ -n "${TASK_IDS:-}" ]]; then
  args+=(--task-ids "${TASK_IDS}")
fi
if [[ -n "${RUNS_PER_TASK}" ]]; then
  args+=(--runs-per-task "${RUNS_PER_TASK}")
fi
if [[ -n "${BENCHMARKS}" && "${cli_has_benchmark_selection}" != "1" ]]; then
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
if [[ -n "${MODELS}" ]]; then
  args+=(--models "${MODELS}")
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
if [[ -n "${BENCHMARK_EVAL_JUDGE_MODEL}" ]]; then
  args+=(--benchmark-eval-judge-model "${BENCHMARK_EVAL_JUDGE_MODEL}")
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
export OPENROUTER_REASONING_EFFORT
export OPENROUTER_TEMPERATURE
export OPENROUTER_TOP_P
export OPENROUTER_TOP_K
export MAS_LLM_RETRY_ATTEMPTS
export MAS_LLM_TIMEOUT_RETRY_ATTEMPTS
export MAS_LLM_RETRY_BACKOFF_S
export MAS_LLM_RETRY_MAX_BACKOFF_S
export MAS_OPENROUTER_PROVIDER_ORDER
export MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER
export MAS_TRANSCRIPT_COMPACTION_ENABLED
export MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER
export MAS_TOOL_CONTEXT_RAW_TURNS
export MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS
export MAS_TOOL_CONTEXT_PREVIEW_CHARS
export MAS_TOOL_CONTEXT_DOCUMENT_CHARS
export FULL_EXPERIMENT_PROGRESS_INTERVAL_S
unset MAS_OPENROUTER_FALLBACK_MODELS

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
  uv sync
  uv run bash scripts/full_experiment.sh

Or with a plain virtualenv on a 3.11+ interpreter:
  python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -e .
  bash scripts/full_experiment.sh
EOF
exit 1

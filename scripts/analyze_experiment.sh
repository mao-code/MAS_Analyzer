#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper defaults. Override these with environment variables or CLI flags.
# Examples:
#   bash scripts/analyze_experiment.sh
#   EXPERIMENT_ROOT=artifacts/full_experiment/20260330T010539Z bash scripts/analyze_experiment.sh
#   OUTPUT_DIR=artifacts/full_experiment/20260330T010539Z/custom_analysis bash scripts/analyze_experiment.sh
#   bash scripts/analyze_experiment.sh --experiment-root artifacts/full_experiment/20260330T010539Z

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_EXPERIMENT_ROOT=""
if [[ -d "${REPO_ROOT}/artifacts/full_experiment" ]]; then
  latest_experiment="$(find "${REPO_ROOT}/artifacts/full_experiment" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 || true)"
  if [[ -n "${latest_experiment}" ]]; then
    DEFAULT_EXPERIMENT_ROOT="${latest_experiment}"
  fi
fi

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${DEFAULT_EXPERIMENT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment-root)
      EXPERIMENT_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash scripts/analyze_experiment.sh [--experiment-root PATH] [--output-dir PATH]

Parameters:
  --experiment-root PATH   Experiment directory to analyze.
                           Default: latest directory under artifacts/full_experiment/
  --output-dir PATH        Directory for generated analysis artifacts.
                           Default: <experiment-root>/analysis

Environment variable equivalents:
  EXPERIMENT_ROOT
  OUTPUT_DIR
EOF
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${EXPERIMENT_ROOT}" ]]; then
  echo "error: experiment root is required and no default experiment was found." >&2
  exit 1
fi

python_args=(scripts/analyze_experiment.py --experiment-root "${EXPERIMENT_ROOT}")
if [[ -n "${OUTPUT_DIR}" ]]; then
  python_args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ ${#args[@]} -gt 0 ]]; then
  python_args+=("${args[@]}")
fi

cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  exec conda run -n agents python "${python_args[@]}"
fi

exec python "${python_args[@]}"

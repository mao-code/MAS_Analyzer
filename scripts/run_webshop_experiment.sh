#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

# Core run options
MODE="sas" # sas | mas
TASK_LIMIT="3"
RUNS_PER_TASK="1"
SEED="42"
OUTPUT_DIR="outputs/webshop_runs"

# LLM defaults
DEFAULT_MODEL="openai/gpt-4o-mini"

# MAS defaults
LEVELS="2"
INTRA_LINK_RATIO="0.8"
FULL_LINKED="false"
NUMBER_OF_AGENTS="4"
AGENTS_PER_LEVEL=""
AGENT_TYPES="planner,researcher"
COMM_COUNT="2"
TURN_MODE="multi_turn"
MAX_TURNS="2"

# WebShop defaults
DATA_MODE="small" # Change to "full" to run the full WebShop asset set.
SPLIT="test"
AUTO_DOWNLOAD="true"
HUMAN_GOALS="true"
MAX_STEPS="12"
HISTORY_WINDOW="4"
SHOW_ATTRS="false"
DATA_DIR="benchmark/webshop/data"

print_help() {
  cat <<'EOF'
Run WebShop experiments with SAS or MAS.

Usage:
  bash scripts/run_webshop_experiment.sh [options]

Options:
  --mode sas|mas
  --task-limit N
  --runs-per-task N
  --seed N
  --output-dir PATH
  --default-model MODEL

MAS options (used when --mode mas):
  --levels N
  --intra-link-ratio FLOAT
  --full-linked true|false
  --number-of-agents N
  --agents-per-level a,b,c
  --agent-types planner,researcher,...
  --communication-count N
  --turn-mode single_turn|multi_turn
  --max-turns N

WebShop options:
  --data-mode small|full
  --split test|eval|train|all
  --auto-download true|false
  --human-goals true|false
  --max-steps N
  --history-window N
  --show-attrs true|false
  --data-dir PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --task-limit) TASK_LIMIT="$2"; shift 2 ;;
    --runs-per-task) RUNS_PER_TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --default-model) DEFAULT_MODEL="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
    --intra-link-ratio) INTRA_LINK_RATIO="$2"; shift 2 ;;
    --full-linked) FULL_LINKED="$2"; shift 2 ;;
    --number-of-agents) NUMBER_OF_AGENTS="$2"; shift 2 ;;
    --agents-per-level) AGENTS_PER_LEVEL="$2"; shift 2 ;;
    --agent-types) AGENT_TYPES="$2"; shift 2 ;;
    --communication-count) COMM_COUNT="$2"; shift 2 ;;
    --turn-mode) TURN_MODE="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --data-mode) DATA_MODE="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --auto-download) AUTO_DOWNLOAD="$2"; shift 2 ;;
    --human-goals) HUMAN_GOALS="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --history-window) HISTORY_WINDOW="$2"; shift 2 ;;
    --show-attrs) SHOW_ATTRS="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

if [[ "${MODE}" != "sas" && "${MODE}" != "mas" ]]; then
  echo "Invalid --mode: ${MODE}. Use sas or mas."
  exit 1
fi

if [[ "${MODE}" == "sas" ]]; then
  LEVELS="1"
  INTRA_LINK_RATIO="1.0"
  FULL_LINKED="true"
  NUMBER_OF_AGENTS="1"
  AGENTS_PER_LEVEL=""
  AGENT_TYPES="general"
  COMM_COUNT="0"
  TURN_MODE="single_turn"
  MAX_TURNS="1"
fi

agent_types_toml="$(echo "${AGENT_TYPES}" | awk -F',' '{
  out="[";
  for (i=1; i<=NF; i++) {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i);
    if (length($i) > 0) {
      if (out != "[") out=out ", ";
      out=out "\"" $i "\"";
    }
  }
  out=out "]";
  print out;
}')"

agents_per_level_line=""
if [[ -n "${AGENTS_PER_LEVEL}" ]]; then
  agents_per_level_clean="$(echo "${AGENTS_PER_LEVEL}" | sed 's/[[:space:]]//g')"
  agents_per_level_toml="[${agents_per_level_clean}]"
  agents_per_level_line="agents_per_level = ${agents_per_level_toml}"
fi

TMP_CONFIG="$(mktemp -t webshop_run_config)"
trap 'rm -f "${TMP_CONFIG}"' EXIT

cat > "${TMP_CONFIG}" <<EOF
[openrouter]
api_key = ""
base_url = "https://openrouter.ai/api/v1"
http_referer = ""
x_title = "MAS Analyzer"
timeout_s = 60.0

[experiment]
output_dir = "${OUTPUT_DIR}"
runs_per_task = ${RUNS_PER_TASK}
seed = ${SEED}
task_limit = ${TASK_LIMIT}

[mas]
levels = ${LEVELS}
intra_level_link_ratio = ${INTRA_LINK_RATIO}
full_linked = ${FULL_LINKED}
number_of_agents = ${NUMBER_OF_AGENTS}
${agents_per_level_line}
agent_types = ${agent_types_toml}
communication_count_internally = ${COMM_COUNT}
turn_mode = "${TURN_MODE}"
max_turns = ${MAX_TURNS}

[models]
default = "${DEFAULT_MODEL}"
planner = "${DEFAULT_MODEL}"
researcher = "${DEFAULT_MODEL}"
general = "${DEFAULT_MODEL}"

[webshop]
data_mode = "${DATA_MODE}"
split = "${SPLIT}"
auto_download = ${AUTO_DOWNLOAD}
human_goals = ${HUMAN_GOALS}
max_steps = ${MAX_STEPS}
history_window = ${HISTORY_WINDOW}
show_attrs = ${SHOW_ATTRS}
data_dir = "${DATA_DIR}"
EOF

MODE_UPPER="$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')"
echo "Running WebShop in ${MODE_UPPER} mode with config: ${TMP_CONFIG}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/main.py" run \
  --config "${TMP_CONFIG}" \
  --benchmark webshop \
  --task-limit "${TASK_LIMIT}" \
  --runs-per-task "${RUNS_PER_TASK}"

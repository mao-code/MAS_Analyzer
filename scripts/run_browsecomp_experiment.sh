#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

# Core run options
MODE="sas" # sas | mas
TASK_LIMIT="5"
RUNS_PER_TASK="1"
SEED="42"
OUTPUT_DIR="outputs/browsecomp_runs"

# LLM / eval options
DEFAULT_MODEL="openai/gpt-4o-mini"
EVAL_MODE="substring" # substring | llm_judge
JUDGE_MODEL="openai/gpt-4.1"

# MAS defaults
LEVELS="2"
INTRA_LINK_RATIO="0.8"
FULL_LINKED="false"
NUMBER_OF_AGENTS="4"
AGENTS_PER_LEVEL=""
AGENT_TYPES="planner,researcher"
COMM_COUNT="2"
TURN_MODE="multi_turn"
MAX_TURNS="3"

# BrowseComp data/tool defaults
DECRYPTED_PATH="benchmark/browsecomp/data/browsecomp_plus_decrypted.jsonl"
QREL_EVIDENCE_PATH="benchmark/browsecomp/topics-qrels/qrel_evidence.txt"
QREL_GOLDS_PATH="benchmark/browsecomp/topics-qrels/qrel_golds.txt"
ENABLE_TOOLS="true"
TOOL_K="5"
TOOL_SNIPPET_MAX_TOKENS="512"
INCLUDE_GET_DOCUMENT="true"
MAX_TOOL_ITERATIONS="8"

print_help() {
  cat <<'EOF'
Run BrowseComp experiments with SAS or MAS.

Usage:
  bash scripts/run_browsecomp_experiment.sh [options]

Options:
  --mode sas|mas
  --task-limit N
  --runs-per-task N
  --seed N
  --output-dir PATH
  --default-model MODEL
  --eval-mode substring|llm_judge
  --judge-model MODEL

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

BrowseComp options:
  --decrypted-path PATH
  --qrel-evidence-path PATH
  --qrel-golds-path PATH
  --enable-tools true|false
  --tool-k N
  --tool-snippet-max-tokens N
  --include-get-document true|false
  --max-tool-iterations N

Examples:
  bash scripts/run_browsecomp_experiment.sh --mode sas --task-limit 3
  bash scripts/run_browsecomp_experiment.sh --mode mas --number-of-agents 6 --levels 3 --agent-types planner,researcher,critic
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
    --eval-mode) EVAL_MODE="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --levels) LEVELS="$2"; shift 2 ;;
    --intra-link-ratio) INTRA_LINK_RATIO="$2"; shift 2 ;;
    --full-linked) FULL_LINKED="$2"; shift 2 ;;
    --number-of-agents) NUMBER_OF_AGENTS="$2"; shift 2 ;;
    --agents-per-level) AGENTS_PER_LEVEL="$2"; shift 2 ;;
    --agent-types) AGENT_TYPES="$2"; shift 2 ;;
    --communication-count) COMM_COUNT="$2"; shift 2 ;;
    --turn-mode) TURN_MODE="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    --decrypted-path) DECRYPTED_PATH="$2"; shift 2 ;;
    --qrel-evidence-path) QREL_EVIDENCE_PATH="$2"; shift 2 ;;
    --qrel-golds-path) QREL_GOLDS_PATH="$2"; shift 2 ;;
    --enable-tools) ENABLE_TOOLS="$2"; shift 2 ;;
    --tool-k) TOOL_K="$2"; shift 2 ;;
    --tool-snippet-max-tokens) TOOL_SNIPPET_MAX_TOKENS="$2"; shift 2 ;;
    --include-get-document) INCLUDE_GET_DOCUMENT="$2"; shift 2 ;;
    --max-tool-iterations) MAX_TOOL_ITERATIONS="$2"; shift 2 ;;
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

TMP_CONFIG="$(mktemp -t browsecomp_run_config)"
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

[browsecomp]
decrypted_path = "${DECRYPTED_PATH}"
qrel_evidence_path = "${QREL_EVIDENCE_PATH}"
qrel_golds_path = "${QREL_GOLDS_PATH}"
auto_download = false
eval_mode = "${EVAL_MODE}"
judge_model = "${JUDGE_MODEL}"
judge_temperature = 0.7
judge_max_tokens = 4096
enable_tools = ${ENABLE_TOOLS}
tool_k = ${TOOL_K}
tool_snippet_max_tokens = ${TOOL_SNIPPET_MAX_TOKENS}
include_get_document = ${INCLUDE_GET_DOCUMENT}
max_tool_iterations = ${MAX_TOOL_ITERATIONS}
EOF

MODE_UPPER="$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')"
echo "Running BrowseComp in ${MODE_UPPER} mode with config: ${TMP_CONFIG}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/main.py" run \
  --config "${TMP_CONFIG}" \
  --benchmark browsecomp \
  --task-limit "${TASK_LIMIT}" \
  --runs-per-task "${RUNS_PER_TASK}"

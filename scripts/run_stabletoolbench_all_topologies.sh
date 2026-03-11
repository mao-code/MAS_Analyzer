#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TASK_LIMIT="5"
RUNS_PER_TASK="1"
SEED="42"
OUTPUT_ROOT="outputs/stabletoolbench_all_topologies"
TASK_SETS="G1_instruction"
MAS_PROVIDER="mock"          # mock | openai | openrouter
MODEL=""
EVAL_MODE="heuristic"        # heuristic | llm_judge
JUDGE_MODEL="gpt-4.1-mini"
QUERY_ROOT="benchmark/stabletoolbench/data/solvable_queries"
AUTO_DOWNLOAD="true"
VIRTUAL_SERVER_URL="${STABLETOOLBENCH_VIRTUAL_SERVER_URL:-http://localhost:8080/virtual}"
ENABLE_TOOLS="true"
MAX_TOOL_ITERATIONS="8"
REQUEST_TIMEOUT_S="120"

print_help() {
  cat <<'EOF'
Run all SAS/MAS topologies on StableToolBench with 5 samples each.

This script expects the upstream StableToolBench virtual server to already be
running. See benchmark/stabletoolbench/README.md for setup.

Usage:
  bash scripts/run_stabletoolbench_all_topologies.sh [options]

Options:
  --task-limit N
  --runs-per-task N
  --seed N
  --output-root PATH
  --task-sets a,b,c
  --mas-provider mock|openai|openrouter
  --model MODEL
  --eval-mode heuristic|llm_judge
  --judge-model MODEL
  --query-root PATH
  --auto-download true|false
  --virtual-server-url URL
  --enable-tools true|false
  --max-tool-iterations N
  --request-timeout-s N

Examples:
  bash scripts/run_stabletoolbench_all_topologies.sh
  bash scripts/run_stabletoolbench_all_topologies.sh --mas-provider openai --eval-mode llm_judge
  bash scripts/run_stabletoolbench_all_topologies.sh --task-sets G1_instruction,G1_tool --task-limit 5
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-limit) TASK_LIMIT="$2"; shift 2 ;;
    --runs-per-task) RUNS_PER_TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --task-sets) TASK_SETS="$2"; shift 2 ;;
    --mas-provider) MAS_PROVIDER="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --eval-mode) EVAL_MODE="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --query-root) QUERY_ROOT="$2"; shift 2 ;;
    --auto-download) AUTO_DOWNLOAD="$2"; shift 2 ;;
    --virtual-server-url) VIRTUAL_SERVER_URL="$2"; shift 2 ;;
    --enable-tools) ENABLE_TOOLS="$2"; shift 2 ;;
    --max-tool-iterations) MAX_TOOL_ITERATIONS="$2"; shift 2 ;;
    --request-timeout-s) REQUEST_TIMEOUT_S="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

if [[ "${MAS_PROVIDER}" != "mock" && "${MAS_PROVIDER}" != "openai" && "${MAS_PROVIDER}" != "openrouter" ]]; then
  echo "Invalid --mas-provider: ${MAS_PROVIDER}"
  exit 1
fi

if [[ "${EVAL_MODE}" != "heuristic" && "${EVAL_MODE}" != "llm_judge" ]]; then
  echo "Invalid --eval-mode: ${EVAL_MODE}"
  exit 1
fi

if [[ -z "${MODEL}" ]]; then
  if [[ "${MAS_PROVIDER}" == "openai" ]]; then
    MODEL="gpt-4.1-mini"
  else
    MODEL="openai/gpt-4.1-mini"
  fi
fi

PYTHON_CMD=(python)
if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "agents"; then
    PYTHON_CMD=(conda run -n agents python)
  fi
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD=("${PROJECT_ROOT}/.venv/bin/python")
fi

if [[ "${EVAL_MODE}" == "llm_judge" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is required when --eval-mode llm_judge"
  exit 1
fi

LLM_BASE_URL="https://openrouter.ai/api/v1"
case "${MAS_PROVIDER}" in
  openai)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "OPENAI_API_KEY is required when --mas-provider openai"
      exit 1
    fi
    LLM_BASE_URL="https://api.openai.com/v1"
    ;;
  openrouter)
    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
      echo "OPENROUTER_API_KEY is required when --mas-provider openrouter"
      exit 1
    fi
    ;;
esac

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}/${STAMP}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

task_sets_toml="$(echo "${TASK_SETS}" | awk -F',' '{
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

TOPOS=(
  "sas"
  "orchestrator_tree_structure"
  "orchestrator_no_discussion"
  "orchestrator_with_discussion"
  "only_voting"
  "fully_linked_debate"
  "group_chat_debate"
)

write_config() {
  local topology="$1"
  local cfg_path="$2"

  local levels="1"
  local intra="0.0"
  local full="false"
  local agents="1"
  local agents_per_level_line=""
  local group_sizes_line=""
  local agent_types='["general"]'
  local comm="1"
  local turn_mode="multi_turn"
  local max_turns="2"
  local discussion_rounds="1"

  case "${topology}" in
    sas)
      levels="1"
      intra="1.0"
      full="true"
      agents="1"
      agent_types='["general"]'
      comm="0"
      turn_mode="single_turn"
      max_turns="1"
      discussion_rounds="1"
      ;;
    orchestrator_tree_structure)
      levels="3"
      intra="0.0"
      full="false"
      agents="5"
      agents_per_level_line="agents_per_level = [1, 2, 2]"
      agent_types='["planner", "researcher", "researcher", "critic", "critic"]'
      comm="2"
      turn_mode="multi_turn"
      max_turns="2"
      discussion_rounds="1"
      ;;
    orchestrator_no_discussion)
      levels="2"
      intra="0.0"
      full="false"
      agents="5"
      agents_per_level_line="agents_per_level = [1, 4]"
      agent_types='["planner", "researcher", "researcher", "critic", "critic"]'
      comm="2"
      turn_mode="multi_turn"
      max_turns="2"
      discussion_rounds="1"
      ;;
    orchestrator_with_discussion)
      levels="2"
      intra="0.0"
      full="false"
      agents="5"
      agents_per_level_line="agents_per_level = [1, 4]"
      agent_types='["planner", "researcher", "researcher", "critic", "critic"]'
      comm="3"
      turn_mode="multi_turn"
      max_turns="2"
      discussion_rounds="2"
      ;;
    only_voting)
      levels="1"
      intra="0.0"
      full="false"
      agents="5"
      agent_types='["general", "general", "general", "general", "general"]'
      comm="0"
      turn_mode="single_turn"
      max_turns="1"
      discussion_rounds="1"
      ;;
    fully_linked_debate)
      levels="1"
      intra="1.0"
      full="true"
      agents="5"
      agent_types='["general", "general", "general", "general", "general"]'
      comm="4"
      turn_mode="multi_turn"
      max_turns="3"
      discussion_rounds="1"
      ;;
    group_chat_debate)
      levels="1"
      intra="0.0"
      full="false"
      agents="5"
      group_sizes_line="group_sizes = [2, 3]"
      agent_types='["general", "general", "general", "general", "general"]'
      comm="3"
      turn_mode="multi_turn"
      max_turns="3"
      discussion_rounds="2"
      ;;
    *)
      echo "Unsupported topology: ${topology}"
      exit 1
      ;;
  esac

  cat > "${cfg_path}" <<EOF
[openrouter]
api_key = ""
base_url = "${LLM_BASE_URL}"
http_referer = ""
x_title = "MAS Analyzer StableToolBench All Topologies"
timeout_s = 60.0

[experiment]
output_dir = "${RUN_ROOT}/${topology}"
runs_per_task = ${RUNS_PER_TASK}
seed = ${SEED}
task_limit = ${TASK_LIMIT}

[mas]
levels = ${levels}
intra_level_link_ratio = ${intra}
full_linked = ${full}
topology = "${topology}"
number_of_agents = ${agents}
${agents_per_level_line}
${group_sizes_line}
agent_types = ${agent_types}
communication_count_internally = ${comm}
turn_mode = "${turn_mode}"
max_turns = ${max_turns}
discussion_rounds = ${discussion_rounds}

[models]
default = "${MODEL}"
planner = "${MODEL}"
researcher = "${MODEL}"
critic = "${MODEL}"
general = "${MODEL}"

[stabletoolbench]
query_root = "${PROJECT_ROOT}/${QUERY_ROOT}"
task_sets = ${task_sets_toml}
auto_download = ${AUTO_DOWNLOAD}
virtual_server_url = "${VIRTUAL_SERVER_URL}"
enable_tools = ${ENABLE_TOOLS}
max_tool_iterations = ${MAX_TOOL_ITERATIONS}
request_timeout_s = ${REQUEST_TIMEOUT_S}
eval_mode = "${EVAL_MODE}"
judge_model = "${JUDGE_MODEL}"
judge_api_base = "https://api.openai.com/v1"
judge_x_title = "MAS Analyzer StableToolBench Judge"
EOF
}

run_config() {
  local cfg_path="$1"

  case "${MAS_PROVIDER}" in
    mock)
      OPENROUTER_API_KEY="" "${PYTHON_CMD[@]}" "${PROJECT_ROOT}/main.py" run \
        --config "${cfg_path}" \
        --benchmark stabletoolbench \
        --task-limit "${TASK_LIMIT}" \
        --runs-per-task "${RUNS_PER_TASK}"
      ;;
    openai)
      OPENROUTER_API_KEY="${OPENAI_API_KEY}" "${PYTHON_CMD[@]}" "${PROJECT_ROOT}/main.py" run \
        --config "${cfg_path}" \
        --benchmark stabletoolbench \
        --task-limit "${TASK_LIMIT}" \
        --runs-per-task "${RUNS_PER_TASK}"
      ;;
    openrouter)
      "${PYTHON_CMD[@]}" "${PROJECT_ROOT}/main.py" run \
        --config "${cfg_path}" \
        --benchmark stabletoolbench \
        --task-limit "${TASK_LIMIT}" \
        --runs-per-task "${RUNS_PER_TASK}"
      ;;
  esac
}

for topology in "${TOPOS[@]}"; do
  cfg_path="${CONFIG_DIR}/${topology}.toml"
  write_config "${topology}" "${cfg_path}"

  echo "[run] topology=${topology} task_limit=${TASK_LIMIT} runs_per_task=${RUNS_PER_TASK} mas_provider=${MAS_PROVIDER} eval_mode=${EVAL_MODE}"
  run_config "${cfg_path}"
done

echo "Completed all topologies."
echo "Configs: ${CONFIG_DIR}"
echo "Outputs root: ${RUN_ROOT}"

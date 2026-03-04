#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Shared defaults
TASK_LIMIT="3"
RUNS_PER_TASK="1"
SEED="42"
OUTPUT_ROOT="outputs/browsecomp_all_topologies"
MODEL="openai/gpt-4o-mini"
USE_LIVE_LLM="true"  # true -> use OPENROUTER_API_KEY from env/.env

DECRYPTED_PATH="benchmark/browsecomp/data/browsecomp_plus_decrypted.jsonl"
QREL_EVIDENCE_PATH="benchmark/browsecomp/topics-qrels/qrel_evidence.txt"
QREL_GOLDS_PATH="benchmark/browsecomp/topics-qrels/qrel_golds.txt"

print_help() {
  cat <<'EOF'
Run all 7 MAS topologies on BrowseComp with 3 samples each.

Usage:
  bash scripts/run_browsecomp_all_topologies.sh [options]

Options:
  --task-limit N
  --runs-per-task N
  --seed N
  --output-root PATH
  --model MODEL
  --use-live-llm true|false
  --decrypted-path PATH
  --qrel-evidence-path PATH
  --qrel-golds-path PATH

Notes:
  - This script writes explicit per-topology configs to:
      <output-root>/<timestamp>/configs/*.toml
  - Default --use-live-llm=false forces local mock LLM fallback (no API cost).
  - Set --use-live-llm=true to use OPENROUTER_API_KEY (env or .env).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-limit) TASK_LIMIT="$2"; shift 2 ;;
    --runs-per-task) RUNS_PER_TASK="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --use-live-llm) USE_LIVE_LLM="$2"; shift 2 ;;
    --decrypted-path) DECRYPTED_PATH="$2"; shift 2 ;;
    --qrel-evidence-path) QREL_EVIDENCE_PATH="$2"; shift 2 ;;
    --qrel-golds-path) QREL_GOLDS_PATH="$2"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) echo "Unknown option: $1"; print_help; exit 1 ;;
  esac
done

PYTHON_CMD=(python)
if command -v conda >/dev/null 2>&1; then
  if conda env list | awk '{print $1}' | grep -qx "agents"; then
    PYTHON_CMD=(conda run -n agents python)
  fi
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_CMD=("${PROJECT_ROOT}/.venv/bin/python")
fi

if [[ ! -f "${PROJECT_ROOT}/${DECRYPTED_PATH}" ]]; then
  echo "BrowseComp decrypted data not found: ${PROJECT_ROOT}/${DECRYPTED_PATH}"
  exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}/${STAMP}"
CONFIG_DIR="${RUN_ROOT}/configs"
mkdir -p "${CONFIG_DIR}"

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
base_url = "https://openrouter.ai/api/v1"
http_referer = ""
x_title = "MAS Analyzer BrowseComp All Topologies"
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

[browsecomp]
decrypted_path = "${PROJECT_ROOT}/${DECRYPTED_PATH}"
qrel_evidence_path = "${PROJECT_ROOT}/${QREL_EVIDENCE_PATH}"
qrel_golds_path = "${PROJECT_ROOT}/${QREL_GOLDS_PATH}"
auto_download = false
eval_mode = "substring"
judge_model = "openai/gpt-4.1"
judge_temperature = 0.7
judge_max_tokens = 4096
enable_tools = true
tool_k = 5
tool_snippet_max_tokens = 512
include_get_document = true
max_tool_iterations = 8
EOF
}

for topology in "${TOPOS[@]}"; do
  cfg_path="${CONFIG_DIR}/${topology}.toml"
  write_config "${topology}" "${cfg_path}"

  echo "[run] topology=${topology} task_limit=${TASK_LIMIT} runs_per_task=${RUNS_PER_TASK}"
  if [[ "${USE_LIVE_LLM}" == "true" ]]; then
    "${PYTHON_CMD[@]}" "${PROJECT_ROOT}/main.py" run \
      --config "${cfg_path}" \
      --benchmark browsecomp \
      --task-limit "${TASK_LIMIT}" \
      --runs-per-task "${RUNS_PER_TASK}"
  else
    OPENROUTER_API_KEY="" "${PYTHON_CMD[@]}" "${PROJECT_ROOT}/main.py" run \
      --config "${cfg_path}" \
      --benchmark browsecomp \
      --task-limit "${TASK_LIMIT}" \
      --runs-per-task "${RUNS_PER_TASK}"
  fi
done

echo "Completed all topologies."
echo "Configs: ${CONFIG_DIR}"
echo "Outputs root: ${RUN_ROOT}"

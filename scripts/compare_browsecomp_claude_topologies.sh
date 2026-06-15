#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAMPLES="${SAMPLES:-3}"
RUNS_PER_TASK="${RUNS_PER_TASK:-1}"
HARNESS_BACKEND="${HARNESS_BACKEND:-claude_agent_sdk}"
EXPERIMENT_ID="${EXPERIMENT_ID:-browsecomp_${HARNESS_BACKEND}_topology_compare_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-artifacts/browsecomp_${HARNESS_BACKEND}_topology_compare}"
CONFIG_PATH="${CONFIG_PATH:-config/browsecomp_claude_topology_compare.toml}"
NO_DYNAMIC_ROLES="${NO_DYNAMIC_ROLES:-1}"
ENABLE_TOOLS="${ENABLE_TOOLS:-false}"
FAST_MODE="${FAST_MODE:-1}"
CLAUDE_AGENT_SDK_EFFORT="${CLAUDE_AGENT_SDK_EFFORT:-low}"
CLAUDE_AGENT_SDK_QUERY_TIMEOUT_S="${CLAUDE_AGENT_SDK_QUERY_TIMEOUT_S:-90}"
CLAUDE_AGENT_SDK_PERMISSION_MODE="${CLAUDE_AGENT_SDK_PERMISSION_MODE:-dontAsk}"
CLAUDE_AGENT_SDK_THINKING="${CLAUDE_AGENT_SDK_THINKING:-disabled}"
CLAUDE_AGENT_SDK_JSON_SCHEMA="${CLAUDE_AGENT_SDK_JSON_SCHEMA:-0}"
export CLAUDE_AGENT_SDK_EFFORT
export CLAUDE_AGENT_SDK_QUERY_TIMEOUT_S
export CLAUDE_AGENT_SDK_PERMISSION_MODE
export CLAUDE_AGENT_SDK_THINKING
export CLAUDE_AGENT_SDK_JSON_SCHEMA

UV_RUN=(uv run)
if [[ "$HARNESS_BACKEND" == "claude_agent_sdk" ]]; then
  MODEL="${MODEL:-claude-sonnet-4-6}"
  UV_RUN=(uv run --extra claude)
elif [[ "$HARNESS_BACKEND" == "openrouter" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" && -f ".env" ]]; then
    set -a
    . ./.env
    set +a
  fi
  if [[ "${MOCK_LLM:-0}" != "1" ]]; then
    : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY, or run with MOCK_LLM=1 for a pipeline-only smoke test.}"
  fi
  MODEL="${MODEL:-google/gemini-3-flash-preview}"
else
  echo "Unsupported HARNESS_BACKEND=$HARNESS_BACKEND; expected openrouter or claude_agent_sdk." >&2
  exit 2
fi

mkdir -p "$(dirname "$CONFIG_PATH")" "$OUT_ROOT/$EXPERIMENT_ID/logs"

cat > "$CONFIG_PATH" <<TOML
[openrouter]
api_key = ""
base_url = "https://openrouter.ai/api/v1"
timeout_s = 600

[experiment]
output_dir = "$OUT_ROOT"
runs_per_task = $RUNS_PER_TASK
seed = 42
task_limit = $SAMPLES

[models]
default = "$MODEL"

[mas]
levels = 1
intra_level_link_ratio = 1.0
full_linked = true
topology = "self_evolved"
number_of_agents = 4
agent_types = ["general"]
communication_count_internally = 2
turn_mode = "single_turn"
max_turns = 1
discussion_rounds = 1
termination_consensus_mode = "lexical"
final_vote_mode = "deterministic"
peer_artifact_max_chars = 320

[self_evolved]
harness_backend = "$HARNESS_BACKEND"
max_initial_agents = 4
max_total_agents = 8
max_turns = 2
audit_mode = "heuristic"
playbook_path = "config/topology_playbook.json"
playbook_read = true
default_packet_max_chars = 320

[browsecomp]
decrypted_path = "benchmark/browsecomp/data/browsecomp_plus_decrypted.jsonl"
qrel_evidence_path = "benchmark/browsecomp/topics-qrels/qrel_evidence.txt"
qrel_golds_path = "benchmark/browsecomp/topics-qrels/qrel_golds.txt"
auto_download = true
eval_mode = "substring"
enable_tools = $ENABLE_TOOLS
tool_k = 5
include_get_document = false
tool_snippet_max_tokens = 160
max_tool_iterations = 4
TOML

if [[ "$FAST_MODE" == "1" ]]; then
  SYSTEMS=(
    "sas|sas|1|1|1|0"
    "orchestrator_tree_structure|orchestrator_tree_structure|3|1|1|1"
    "orchestrator_no_discussion|orchestrator_no_discussion|2|1|1|1"
    "orchestrator_with_discussion|orchestrator_with_discussion|2|1|1|1"
    "only_voting|only_voting|2|1|1|0"
    "fully_linked_debate|fully_linked_debate|2|1|1|1"
    "group_chat_debate|group_chat_debate|2|1|1|1"
    "self_evolved|self_evolved|2|1|1|1"
  )
else
  SYSTEMS=(
    "sas|sas|1|1|1|0"
    "orchestrator_tree_structure|orchestrator_tree_structure|5|2|1|2"
    "orchestrator_no_discussion|orchestrator_no_discussion|4|2|1|2"
    "orchestrator_with_discussion|orchestrator_with_discussion|4|2|2|2"
    "only_voting|only_voting|4|1|1|0"
    "fully_linked_debate|fully_linked_debate|4|2|1|2"
    "group_chat_debate|group_chat_debate|4|2|2|2"
    "self_evolved|self_evolved|5|2|1|2"
  )
fi

echo "Config: $CONFIG_PATH"
echo "Experiment root: $OUT_ROOT/$EXPERIMENT_ID"
echo "Harness backend: $HARNESS_BACKEND"
echo "Samples: $SAMPLES"
echo "Runs per task: $RUNS_PER_TASK"
echo "Model: $MODEL"
echo "No dynamic roles: $NO_DYNAMIC_ROLES"
echo "BrowseComp tools enabled: $ENABLE_TOOLS"
echo "Fast topology profile: $FAST_MODE"
if [[ "$HARNESS_BACKEND" == "claude_agent_sdk" ]]; then
  echo "Claude SDK effort: $CLAUDE_AGENT_SDK_EFFORT"
  echo "Claude SDK query timeout: ${CLAUDE_AGENT_SDK_QUERY_TIMEOUT_S}s"
  echo "Claude SDK permission mode: $CLAUDE_AGENT_SDK_PERMISSION_MODE"
  echo "Claude SDK thinking: $CLAUDE_AGENT_SDK_THINKING"
  echo "Claude SDK JSON schema: $CLAUDE_AGENT_SDK_JSON_SCHEMA"
  if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "Claude auth: ANTHROPIC_API_KEY"
  else
    echo "Claude auth: local Claude SDK credentials"
  fi
else
  if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OpenRouter auth: OPENROUTER_API_KEY"
  else
    echo "OpenRouter auth: mock/no key"
  fi
fi

for entry in "${SYSTEMS[@]}"; do
  IFS="|" read -r label topology agents rounds discussion comm <<<"$entry"
  log_path="$OUT_ROOT/$EXPERIMENT_ID/logs/${label}.log"
  args=(
    python main.py run
    --config "$CONFIG_PATH"
    --benchmark browsecomp
    --task-limit "$SAMPLES"
    --runs-per-task "$RUNS_PER_TASK"
    --output-layout hierarchical
    --experiment-id "$EXPERIMENT_ID"
    --system-label "$label"
    --topology "$topology"
    --agents "$agents"
    --mas-rounds "$rounds"
    --discussion-rounds "$discussion"
    --communication-budget "$comm"
    --termination-consensus-mode lexical
    --final-vote-mode deterministic
  )
  if [[ "$NO_DYNAMIC_ROLES" == "1" ]]; then
    args+=(--no-dynamic-roles)
  fi

  echo
  echo "=== Running $label ==="
  "${UV_RUN[@]}" "${args[@]}" 2>&1 | tee "$log_path"
done

uv run python scripts/summarize_browsecomp_topology_compare.py \
  --experiment-root "$OUT_ROOT/$EXPERIMENT_ID" \
  --output "$OUT_ROOT/$EXPERIMENT_ID/OBSERVATION_REPORT.md"

echo
echo "Report: $OUT_ROOT/$EXPERIMENT_ID/OBSERVATION_REPORT.md"
echo "Experiment root: $OUT_ROOT/$EXPERIMENT_ID"

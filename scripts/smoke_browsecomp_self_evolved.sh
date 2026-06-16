#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SAMPLES="${SAMPLES:-3}"
RUNS_PER_TASK="${RUNS_PER_TASK:-1}"
EXPERIMENT_ID="${EXPERIMENT_ID:-browsecomp_self_evolved_smoke_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-artifacts/smoke_browsecomp_self_evolved}"
CONFIG_PATH="${CONFIG_PATH:-config/smoke_browsecomp_self_evolved.toml}"
HARNESS_BACKEND="${HARNESS_BACKEND:-openrouter}"
MODEL="${MODEL:-}"
ENABLE_TOOLS="${ENABLE_TOOLS:-true}"
NO_DYNAMIC_ROLES="${NO_DYNAMIC_ROLES:-0}"
MAS_NUMBER_OF_AGENTS="${MAS_NUMBER_OF_AGENTS:-4}"
MAS_COMMUNICATION_BUDGET="${MAS_COMMUNICATION_BUDGET:-2}"
PEER_ARTIFACT_MAX_CHARS="${PEER_ARTIFACT_MAX_CHARS:-320}"
DEFAULT_PACKET_MAX_CHARS="${DEFAULT_PACKET_MAX_CHARS:-320}"
TERMINATION_CONSENSUS_MODE="${TERMINATION_CONSENSUS_MODE:-lexical}"
FINAL_VOTE_MODE="${FINAL_VOTE_MODE:-deterministic}"
SELF_EVOLVED_MAX_INITIAL_AGENTS="${SELF_EVOLVED_MAX_INITIAL_AGENTS:-4}"
SELF_EVOLVED_MAX_TOTAL_AGENTS="${SELF_EVOLVED_MAX_TOTAL_AGENTS:-8}"
SELF_EVOLVED_MAX_TURNS="${SELF_EVOLVED_MAX_TURNS:-2}"
BROWSECOMP_MAX_TOOL_ITERATIONS="${BROWSECOMP_MAX_TOOL_ITERATIONS:-8}"
INCLUDE_GET_DOCUMENT="${INCLUDE_GET_DOCUMENT:-true}"
TOOL_SNIPPET_MAX_TOKENS="${TOOL_SNIPPET_MAX_TOKENS:-512}"
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
  if [[ "${MOCK_LLM:-0}" != "1" ]]; then
    : "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY, or run with MOCK_LLM=1 for a pipeline-only smoke test.}"
  fi
  MODEL="${MODEL:-google/gemini-3-flash-preview}"
else
  echo "Unsupported HARNESS_BACKEND=$HARNESS_BACKEND; expected openrouter or claude_agent_sdk." >&2
  exit 2
fi

mkdir -p "$(dirname "$CONFIG_PATH")" "$OUT_ROOT/$EXPERIMENT_ID"

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
number_of_agents = $MAS_NUMBER_OF_AGENTS
agent_types = ["general"]
communication_count_internally = $MAS_COMMUNICATION_BUDGET
turn_mode = "single_turn"
max_turns = 1
termination_consensus_mode = "$TERMINATION_CONSENSUS_MODE"
final_vote_mode = "$FINAL_VOTE_MODE"
peer_artifact_max_chars = $PEER_ARTIFACT_MAX_CHARS

[self_evolved]
harness_backend = "$HARNESS_BACKEND"
max_initial_agents = $SELF_EVOLVED_MAX_INITIAL_AGENTS
max_total_agents = $SELF_EVOLVED_MAX_TOTAL_AGENTS
max_turns = $SELF_EVOLVED_MAX_TURNS
audit_mode = "heuristic"
playbook_path = "config/topology_playbook.json"
playbook_read = true
default_packet_max_chars = $DEFAULT_PACKET_MAX_CHARS

[browsecomp]
decrypted_path = "benchmark/browsecomp/data/browsecomp_plus_decrypted.jsonl"
qrel_evidence_path = "benchmark/browsecomp/topics-qrels/qrel_evidence.txt"
qrel_golds_path = "benchmark/browsecomp/topics-qrels/qrel_golds.txt"
auto_download = true
eval_mode = "substring"
enable_tools = $ENABLE_TOOLS
tool_k = 5
include_get_document = $INCLUDE_GET_DOCUMENT
tool_snippet_max_tokens = $TOOL_SNIPPET_MAX_TOKENS
max_tool_iterations = $BROWSECOMP_MAX_TOOL_ITERATIONS
TOML

echo "Config: $CONFIG_PATH"
echo "Experiment: $OUT_ROOT/$EXPERIMENT_ID"
echo "BrowseComp tools enabled: $ENABLE_TOOLS"
echo "No dynamic roles: $NO_DYNAMIC_ROLES"
echo "MAS agents: $MAS_NUMBER_OF_AGENTS"
echo "MAS communication budget: $MAS_COMMUNICATION_BUDGET"
echo "Self-evolved max initial agents: $SELF_EVOLVED_MAX_INITIAL_AGENTS"
echo "Self-evolved max total agents: $SELF_EVOLVED_MAX_TOTAL_AGENTS"
echo "Self-evolved max turns: $SELF_EVOLVED_MAX_TURNS"
echo "BrowseComp max tool iterations: $BROWSECOMP_MAX_TOOL_ITERATIONS"
echo "BrowseComp include get_document: $INCLUDE_GET_DOCUMENT"
echo "BrowseComp snippet max tokens: $TOOL_SNIPPET_MAX_TOKENS"
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
fi

RUN_ARGS=(
  python main.py run
  --config "$CONFIG_PATH"
  --benchmark browsecomp
  --task-limit "$SAMPLES"
  --runs-per-task "$RUNS_PER_TASK"
  --output-layout hierarchical
  --experiment-id "$EXPERIMENT_ID"
)
if [[ "$NO_DYNAMIC_ROLES" == "1" ]]; then
  RUN_ARGS+=(--no-dynamic-roles)
fi

"${UV_RUN[@]}" "${RUN_ARGS[@]}" 2>&1 | tee "$OUT_ROOT/$EXPERIMENT_ID/run.log"

"${UV_RUN[@]}" python main.py summarize-experiment \
  --experiment-root "$OUT_ROOT/$EXPERIMENT_ID" \
  2>&1 | tee "$OUT_ROOT/$EXPERIMENT_ID/summary.log"

echo
echo "Per-task artifacts:"
echo "  $OUT_ROOT/$EXPERIMENT_ID/browsecomp/self_evolved/<task_id>/"
echo
echo "Key files per task:"
echo "  run_0.trace.jsonl"
echo "  run_0.metadata.json"
echo "  run_0.raw.json"
echo "  run_0.trajectory.md"
echo "  run_0.trajectory.json"
echo "  run_0.trace_metrics.json"
echo "  descriptor.json"
echo "  analysis.json"

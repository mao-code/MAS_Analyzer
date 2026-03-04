#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=${OUTPUT_DIR:-outputs/langgraph_topologies_batch}
PROMPT=${PROMPT:-"Solve the task and provide a concise final answer."}

bash scripts/run_langgraph_experiment.sh --topology sas --agents 1 --rounds 1 --output-dir "$OUTPUT_DIR" --prompt "$PROMPT"
bash scripts/run_langgraph_experiment.sh --topology orchestrator_tree_structure --agents 5 --rounds 2 --output-dir "$OUTPUT_DIR" --prompt "$PROMPT"
bash scripts/run_langgraph_experiment.sh --topology orchestrator_no_discussion --agents 5 --rounds 2 --output-dir "$OUTPUT_DIR" --prompt "$PROMPT"
bash scripts/run_langgraph_experiment.sh --topology orchestrator_with_discussion --agents 5 --rounds 2 --discussion-rounds 2 --output-dir "$OUTPUT_DIR" --prompt "$PROMPT"
bash scripts/run_langgraph_experiment.sh --topology only_voting --agents 5 --rounds 1 --output-dir "$OUTPUT_DIR" --prompt "$PROMPT"
bash scripts/run_langgraph_experiment.sh --topology fully_linked_debate --agents 5 --rounds 3 --output-dir "$OUTPUT_DIR" --prompt "$PROMPT"
bash scripts/run_langgraph_experiment.sh --topology group_chat_debate --agents 5 --rounds 3 --discussion-rounds 2 --output-dir "$OUTPUT_DIR" --prompt "$PROMPT"

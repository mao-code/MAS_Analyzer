# Trace schema and artifact semantics

The trace is designed so that every run-level metric is recomputable from the logs.

Each run writes a JSONL trace. A trace event contains:

- `timestamp_start`, `timestamp_end`
- `actor`
- `event_type`
- `payload`
- `token_in`, `token_out`
- `latency_ms`, `cost_usd`
- optional `state_id`

Supported event types:

- `plan`
- `act`
- `tool_call`
- `tool_result`
- `verify`
- `revise`
- `finalize`
- `error`

The schema is designed so all run-level trace metrics are recomputable from logs.

## Artifact semantics

For each run:

- `run_<n>.trace.jsonl`: raw trace events
- `run_<n>.answer.txt`: final answer text
- `run_<n>.metadata.json`: runtime metadata from the MAS execution
- `run_<n>.eval.json`: benchmark-native score and correctness
- `run_<n>.trace_metrics.json`: run-level outcome + trace totals + stage metrics
- `run_<n>.result.json`: compact run summary
- `run_<n>.trajectory.json` / `.md`: communication trajectory export

For each task:

- `descriptor.json`: aggregated task descriptor
- `descriptor.csv`: flat CSV version of the descriptor
- `analysis.json`: evaluation summary, descriptor, stage bottleneck hints
- `task_summary.json`: task-level summary across runs

For each system:

- `mas_graph.png` / `.mmd`: agent-topology graph
- `workflow_graph.png` / `.mmd`: workflow/control-flow graph
- `summary.json`: task summaries for the system
- `summary.csv`: one row per task for the system

For a hierarchical batch experiment:

- `artifacts/full_experiment/<experiment-id>/<benchmark>/<system>/<task_id>/...`
- benchmark/system rollups under the same root
- `experiment_summary.json` and `experiment_summary.csv` at the experiment root

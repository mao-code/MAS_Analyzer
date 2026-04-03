# Economic Analysis Framework for Multi-Agent Collaboration

This repo is the experiment harness behind [PAPER.md](PAPER.md). It runs single-agent systems (SAS) and multi-agent systems (MAS) on a shared benchmark suite, records structured execution traces, and converts repeated runs into a trace-derived descriptor that supports:

- quality vs cost analysis
- MAS vs SAS gain/cost comparison
- coordination diagnostics
- topology-level summary and Pareto analysis

The core goal is not only to ask whether MAS can help, but to measure when collaboration improves task outcomes enough to justify its execution and coordination cost.

## Core idea

Each task is executed one or more times under a fixed system configuration. For every run, the repo stores:

- benchmark-native evaluation output
- structured trace events
- run-level trace metrics
- a task-level descriptor aggregated over repeated runs

The descriptor follows the paper’s `Q / C / D / R / P` split:

- `Q`: outcome quality
- `C`: direct execution cost
- `D`: coordination diagnostics
- `R`: run-to-run reliability
- `P`: process structure

The paper defines higher-level economic quantities such as utility `U = Q - C`, collaboration gain `G`, and coordination cost `K`. This repo produces the trace-derived ingredients needed for those analyses.

## Exact metric contract

The metric contract is intentionally strict and reproducible.

### Run-level outcome variables

For one run `r`:

- `success_r = 1` iff `benchmark.evaluate(...).success` is `True`
- `completion_r = 1` iff the run produced a final artifact / final answer and did not terminate with an explicit runtime failure signal

Important:

- `success` is benchmark correctness
- `completion` is execution completion
- `completion` does **not** imply correctness

So a wrong answer can still have `completion = 1` and `success = 0`.

At the benchmark level for a fixed MAS:

- `success_rate` means: among all benchmark sample runs, what fraction were solved correctly
- `completion_rate` means: among all benchmark sample runs, what fraction finished execution and produced a final answer/artifact without an explicit runtime failure

Equivalently:

- `success_rate`: "How many samples did this MAS actually solve?"
- `completion_rate`: "How many sample runs did this MAS complete successfully as executions?"

### Run-level trace totals

For one run `r`, the trace code computes:

- `latency_total_r = sum(event.latency_ms)`
- `tokens_total_r = sum(event.token_in + event.token_out)`
- `cost_total_r = sum(event.cost_usd)`
- `tool_calls_total_r = count(event_type == "tool_call")`
- `tool_fail_total_r = count(tool failures)`
- `steps_total_r = number of trace events`
- `backtrack_rate_r = (#revise events + payload.redo) / steps_total_r`
- `loop_score_r = repeated-state or repeated-pattern ratio from the trace`
- `verification_density_r = #verify / steps_total_r`
- `communication_count_r = directed relay/message edges from all inter-agent sends, including system-mediated sends`
- `communication_agent_to_agent_count_r = directed send edges whose sender is a non-system agent`
- `communication_system_mediated_count_r = directed send edges whose sender is system / mediator`
- `handoff_count_r = actor switches across consecutive non-system events`

### Task-level descriptor aggregation

Given `N` repeated runs for the same task and system:

**Quality**

- `Q1_success_rate = mean_r(success_r)`
- `Q2_completion_rate = mean_r(completion_r)`

**Execution cost**

- `C1_latency_p95 = p95_r(latency_total_r)`
- `C2_tokens_total = mean_r(tokens_total_r)`
- `C3_cost_total = mean_r(cost_total_r)`
- `C4_tool_calls_total = mean_r(tool_calls_total_r)`

**Coordination diagnostics**

- `D1_tool_error_rate = sum_r(tool_fail_total_r) / sum_r(tool_calls_total_r)`
- `D2_communication_count = mean_r(communication_count_r)`
- `D2_agent_to_agent_communication_count = mean_r(communication_agent_to_agent_count_r)`
- `D2_system_mediated_communication_count = mean_r(communication_system_mediated_count_r)`
- `D3_handoff_count = mean_r(handoff_count_r)`

These `D*` metrics are logged as coordination diagnostics. They are not part of the paper’s direct execution-cost definition `C`.

**Reliability**

- `R1_success_var = Var_r(success_r)`
- `R2_latency_var = Var_r(latency_total_r)`
- `R3_tokens_var = Var_r(tokens_total_r)`

**Process**

- `P1_steps_total = mean_r(steps_total_r)`
- `P2_backtrack_rate = mean_r(backtrack_rate_r)`
- `P3_loop_score = mean_r(loop_score_r)`
- `P4_verification_density = mean_r(verification_density_r)`

### What appears in `summary.csv`

Per task and system, `summary.csv` includes:

- `eval_avg_score`: benchmark-native mean score across runs
- `eval_success_rate`: benchmark-native mean boolean success across runs
- `eval_completion_rate`: runtime completion rate across runs
- descriptor fields such as `Q1_success_rate`, `C2_tokens_total`, `D2_communication_count`, `P3_loop_score`, etc.

By design:

- `Q1_success_rate` should match `eval_success_rate`
- `Q2_completion_rate` should match `eval_completion_rate`

If those pairs disagree, that indicates a bug in the artifact pipeline.

Interpretation by level:

- per run: `success` and `completion` are binary `0/1`
- per task with repeated runs: `Q1_success_rate` and `Q2_completion_rate` are proportions over that task's repeated runs
- per benchmark for one MAS: average those task-level values across all samples in the benchmark to get the benchmark-level success/completion rates

## Workflow termination logic

Looped MAS stages such as debate, representative exchange, and orchestrator cycles do not stop implicitly. A controller node calls `_termination_decision(...)` in `MAS/langgraph_engine.py` and computes explicit stop statistics from the current stage artifacts.

### Inputs

For one controller decision:

- `candidate_artifacts`: the current artifacts that would be revised if the loop continues
- `previous_candidate_artifacts`: the previous-step artifacts for the same agents, used to measure change
- `consensus_artifacts`: the artifacts whose answers are compared for agreement
- `expected_count`: how many active branches or agents were expected to produce an artifact

### Valid artifact count

The code first counts:

- `valid_artifact_count = count(artifact.answer.strip() != "")`

If `valid_artifact_count < ceil(expected_count / 2)`, the stage stops with `invalid_or_failed_branch`.

Interpretation:

- this is a branch-survival check
- if fewer than half of the expected branches produced a non-empty answer, the collaboration stage is considered too broken to continue

### Consensus ratio

By default, the repo computes termination consensus with an LLM judge:

- `mas.termination_consensus_mode = "llm_judge"` by default
- the judge uses the system model route `models.judge` if provided, otherwise `models.default`
- the controller sends the current task prompt plus the candidate answers to the judge
- the judge returns JSON groups of semantically equivalent answers

The JSON schema is:

- `groups`: lists of artifact indices that express the same final answer
- `invalid_indices`: indices the judge considers unusable or non-answers
- `is_substantive`: whether the largest agreement group is an actual task answer
- `progress_status`: `improving | stalled | unclear`
- `expected_improvement`: `high | medium | low`
- `should_stop_for_no_progress`: whether another round is unlikely to materially improve correctness
- `explanation`: short rationale

The controller then computes:

- `winner_count = size of the largest judged equivalence group`
- `valid_count = number of valid answers after removing invalid_indices`
- `consensus_ratio = winner_count / valid_count`

The stage stops with `consensus_reached` when:

- `valid_count > 1`
- `consensus_ratio >= 0.75`
- the semantic judge marks the majority answer as substantive

Interpretation:

- consensus here is semantic agreement as judged by the termination judge, not exact string identity
- if the judge clusters 3 of 4 valid answers together, `consensus_ratio = 0.75`
- this consensus check is still a workflow-control heuristic, not the benchmark evaluator and not the final correctness decision

Fallback behavior:

- if `mas.termination_consensus_mode = "lexical"`, the repo uses deterministic normalized-string voting
- if `mas.termination_consensus_mode = "llm_judge"` but the judge is unavailable, running in mock mode, or returns unusable JSON, the controller falls back to lexical consensus

The lexical fallback canonicalizes each answer by lowercasing, removing non-alphanumeric characters, and collapsing whitespace, then computes the same `winner_count / valid_count` ratio over exact normalized matches.

Final answer aggregation is separate from this stop-condition ratio. Final answer selection is configurable and can fall back to deterministic `vote_artifacts(...)` after the loop ends.

You can also configure final answer selection separately:

- `mas.final_vote_mode = "llm_judge"` by default
- the final judge sees the task prompt plus the candidate answers and returns JSON with semantic groups, a `winner_index`, optional `invalid_indices`, and a short explanation
- if the final judge is unavailable, running in mock mode, or returns unusable JSON, the repo falls back to deterministic `vote_artifacts(...)`

### Average confidence

Each artifact carries a `confidence` field produced by the agent JSON output schema. During artifact construction:

- the parsed value is converted to `float`
- it is clipped into `[0, 1]`
- if missing or unparsable, it defaults to `0.5`

Then:

- `average_confidence = mean(artifact.confidence)`

across the current `candidate_artifacts` (or `consensus_artifacts` if needed).

Interpretation:

- this is self-reported model confidence averaged over the active artifacts
- it is logged as a diagnostic only and does not directly terminate a run
- the prompt now defines confidence as confidence in the current `answer_artifact`, not general optimism

### Progress / stall judgment

In `llm_judge` mode, `no_meaningful_change` is semantic. The termination judge sees the current candidate artifacts plus each agent's previous answer when available and decides whether another round is likely to materially improve correctness.

The stage stops with `no_meaningful_change` when:

- previous comparable artifacts exist
- the semantic judge returns `should_stop_for_no_progress = true`

Fallback behavior:

- if the termination judge is unavailable, mocked, or unparsable, the repo falls back to lexical change detection
- lexical fallback computes `mean_delta` with `difflib.SequenceMatcher`
- lexical fallback stops when `mean_delta <= 0.05`

`mean_delta` is still logged for compatibility, but in successful `llm_judge` mode it is diagnostic rather than the stop criterion.

### Max-round stop

The stage stops with `max_rounds_reached` when the topology-specific configured round or discussion limit has been exhausted.

### Stop order

The checks are applied in this order:

1. `invalid_or_failed_branch`
2. `consensus_reached`
3. `no_meaningful_change`
4. `max_rounds_reached`

So if multiple conditions are true, the first one in this list is the recorded stop reason.

### What gets logged

Each termination decision logs:

- `reason`
- `reason_detail`
- `consensus_mode`
- `consensus_source`
- `consensus_ratio`
- `consensus_groups`
- `consensus_explanation`
- `progress_source`
- `progress_status`
- `expected_improvement`
- `progress_explanation`
- `average_confidence`
- `mean_delta`
- `valid_artifact_count`
- control-step `token_in`, `token_out`, `latency_ms`, `cost_usd` when an LLM judge call is used

These values are workflow-control diagnostics. They determine whether a collaboration loop continues, but they are not benchmark quality metrics like `success_rate`.

## Trace schema

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

## Repository layout

- `benchmark/`: benchmark adapters and evaluation logic
- `benchmarks/`: benchmark overview docs
- `MAS/`: SAS/MAS runtimes and LangGraph topologies
- `descriptor/`: trace schema, run metrics, task descriptor aggregation, topology analysis
- `scripts/`: experiment and analysis helpers
- `config/`: experiment configs
- `main.py`: CLI entrypoint

## Quickstart

### 1. Install

```bash
uv sync
```

or

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Create a config

```bash
cp config/experiment.example.toml config/experiment.toml
```

OpenRouter credentials can be set in `config/experiment.toml` or through `OPENROUTER_API_KEY`.

### 3. Inspect available benchmarks

```bash
python main.py list-benchmarks
python main.py benchmark-info --benchmark browsecomp --config config/experiment.toml
```

### 4. Run one experiment

```bash
python main.py run \
  --config config/experiment.toml \
  --benchmark browsecomp \
  --task-limit 1 \
  --runs-per-task 1
```

### 5. Summarize a hierarchical experiment

```bash
python main.py summarize-experiment --experiment-root artifacts/full_experiment/<experiment-id>
```

## Batch experiments

The main batch wrapper is:

```bash
bash scripts/full_experiment.sh
```

Useful environment variables:

- `TASK_LIMIT`
- `RUNS_PER_TASK`
- `BENCHMARKS`
- `EXPERIMENT_ID`
- `OUTPUT_ROOT`

Hierarchical outputs are written under:

```text
artifacts/full_experiment/<experiment-id>/<benchmark>/<system>/<task_id>/
```

## Topology analysis

`descriptor/topology_analysis.py` provides:

- descriptor scaling
- Mahalanobis distance
- Pareto frontier extraction
- PCA / optional UMAP embeddings

This is useful when comparing SAS and multiple MAS topologies over the same task set.

## Benchmark notes

See [benchmarks/README.md](benchmarks/README.md) for the paper-aligned benchmark map, benchmark-specific success definitions, and setup notes.

## Package naming

- Canonical benchmark package: `benchmark/`
- `benchmarks/` is a documentation / compatibility shim

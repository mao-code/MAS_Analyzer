# Exact metric contract

This is the source of truth for what `success` and `completion` mean and how every
`Q*/C*/D*/R*/P*` field is computed. Do not change a definition here without changing
the code that produces it.

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

### Paper-facing task metrics

The task descriptor also writes paper-facing fields directly so downstream scripts do not need to reconstruct them:

- `success_rate = Q1_success_rate`
- `pass_at_1`, `pass_at_3`, `pass_at_5`, `pass_at_8` using the paper’s pass@k estimator over repeated runs
- `stability = clip(1 - R1_success_var / 0.25, 0, 1)` when `N >= 2`, otherwise blank
- `eval_avg_score = mean_r(score_r)`
- `tokens_total = C2_tokens_total`
- `cost_per_success = tokens_total / success_rate` when `success_rate > 0`, otherwise blank
- `tokens_cv = std_r(tokens_total_r) / mean_r(tokens_total_r)` when `N >= 2` and mean tokens are positive, otherwise blank
- `tool_calls_total = C4_tool_calls_total`
- diagnostic aliases: `tool_error_rate`, `communication_count`, `handoff_count`

Interpretation notes:

- `stability` and `tokens_cv` require repeated runs and are blank for single-run tasks
- `pass_at_k` is blank when fewer than `k` repeated runs are available
- `cost_per_success` is blank when the system never succeeds on that task

### What appears in `summary.csv`

Per task and system, `summary.csv` includes:

- `eval_avg_score`: benchmark-native mean score across runs
- `eval_success_rate`: benchmark-native mean boolean success across runs
- `eval_completion_rate`: runtime completion rate across runs
- paper-facing descriptor fields such as `success_rate`, `pass_at_3`, `stability`, `tokens_total`, `cost_per_success`, `tokens_cv`
- compatibility fields such as `Q1_success_rate`, `C2_tokens_total`, `D2_communication_count`, `P3_loop_score`, etc.

By design:

- `Q1_success_rate` should match `eval_success_rate`
- `Q2_completion_rate` should match `eval_completion_rate`

If those pairs disagree, that indicates a bug in the artifact pipeline.

Interpretation by level:

- per run: `success` and `completion` are binary `0/1`
- per task with repeated runs: `Q1_success_rate` and `Q2_completion_rate` are proportions over that task's repeated runs
- per benchmark for one MAS: average those task-level values across all samples in the benchmark to get the benchmark-level success/completion rates

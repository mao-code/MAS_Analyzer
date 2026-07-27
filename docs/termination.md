# Workflow termination logic

Controller nodes decide when a loop stops — never the agents.

Looped MAS stages such as debate, representative exchange, and orchestrator cycles do not stop implicitly. A controller node calls `_termination_decision(...)` in `MAS/langgraph_engine.py` and computes explicit stop statistics from the current stage artifacts.

### Inputs

For one controller decision:

- `candidate_artifacts`: the current artifacts that would be revised if the loop continues
- `previous_candidate_artifacts`: the previous-step artifacts for the same agents, used to measure change
- `consensus_artifacts`: the artifacts whose answers are compared for agreement
- `expected_count`: how many active branches or agents were expected to produce an artifact

### Branch artifact count

The code first counts:

- `valid_artifact_count = count(non-empty branch artifacts available at the current controller step)`

If `valid_artifact_count < ceil(expected_count / 2)`, the stage stops with `invalid_or_failed_branch`.

Interpretation:

- this is a branch-survival check
- if fewer than half of the expected branches produced any usable artifact at all, the collaboration stage is considered too broken to continue
- blocked or planning artifacts do not count as good final answers, but they no longer trigger branch-collapse handling by themselves

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
- the agreement is **decision-grade**: average confidence is `>= 0.5` and no agent still
  lists unresolved issues — *or* no further step (another round / a remaining repair) is available

Interpretation:

- consensus here is semantic agreement as judged by the termination judge, not exact string identity
- if the judge clusters 3 of 4 valid answers together, `consensus_ratio = 0.75`
- the decision-grade gate mirrors the Trace Auditor's `premature_consensus` check, so the controller never stops on a uniformly low-confidence (or unresolved) agreement while a repair or another round could still improve it; when no step remains, agreement always stops the loop (no infinite loops). The `consensus_ratio` metric itself is unchanged, and the decision logs `consensus_gate_blocked` / `consensus_gate_reason`.
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

Important:

- `mas.minimum_discussion_rounds` applies only to discussion/debate controllers
- outer collaboration cycles are controlled by `rounds`
- `rounds=1` means one outer cycle; it does not force a second pass

### Stop order

The checks are applied in this order:

1. `invalid_or_failed_branch`
2. `consensus_reached` (only when the agreement is decision-grade — see Consensus above)
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
- `consensus_gate_blocked` / `consensus_gate_reason`
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

# AFlow Official-Code Adapter

This branch adapts the official FoundationAgents/AFlow workflow-optimization
structure to the repository's benchmark adapters. The runner keeps the AFlow
shape of generated `Workflow` code, `prompt.py`, round directories,
`experience.json`, `results.json`, and best-workflow materialization, while
replacing the upstream benchmark/evaluator layer with this repository's
`benchmark.evaluate(...).success` contract.

## Paper Alignment

- Starts from `workflows/round_1/graph.py` and `prompt.py`.
- Evaluates workflow code on validation tasks and appends round scores to
  `workflows/results.json`.
- Samples a high-scoring parent round using the official top-round selection
  idea.
- Calls the optimizer LLM to propose `<modification>`, `<graph>`, and
  `<prompt>` for the next round.
- Writes `round_N/graph.py`, `round_N/prompt.py`, `round_N/experience.json`,
  per-run traces, and a materialized `best_workflow/`.

The main adaptation is that generated workflows subclass
`OfficialWorkflowBase` instead of importing the upstream `workspace.<dataset>`
package. Operators are OpenRouter-backed wrappers for the same AFlow operator
roles (`Custom`, `AnswerGenerate`, `ScEnsemble`, `Review`, `Revise`, `Format`)
and receive this repo's benchmark tools when the benchmark exposes tools.

## Run

```bash
.venv/bin/python -m reproduce.aflow.run_existing_benchmarks \
  --config config/reproduce_gemma/baseline_gemma_30x3.toml \
  --benchmark browsecomp \
  --benchmark plancraft \
  --benchmark stabletoolbench \
  --benchmark workbench \
  --task-limit 30 \
  --validation-rounds 10 \
  --test-task-limit 30 \
  --test-offset 10 \
  --runs-per-task 3 \
  --max-rounds 5 \
  --sample 4 \
  --temperature 1 \
  --run-id aflow_official_gemma_30x3_T1 \
  --keep-going
```

`--max-rounds` is optimizer rounds. `--sample` is the top-round candidate pool
used when selecting a parent workflow for the next optimization round.
`--validation-rounds` controls workflow-search tasks. The best workflow is then
run on held-out test tasks selected by `--test-offset` and `--test-task-limit`.
With the command above, tasks 0-9 are validation and tasks 10-39 are test.

`finance_agent` is excluded by default. Add benchmarks with repeated
`--benchmark` flags, or omit `--benchmark` to run all non-finance benchmarks.

For a no-cost wiring smoke test:

```bash
MAS_DISABLE_LIVE_LLM=1 .venv/bin/python -m reproduce.aflow.run_existing_benchmarks \
  --config config/experiment.example.toml \
  --benchmark workbench \
  --task-limit 1 \
  --validation-rounds 1 \
  --test-task-limit 1 \
  --test-offset 1 \
  --runs-per-task 1 \
  --max-rounds 2 \
  --sample 1 \
  --run-id smoke_official_aflow_workbench \
  --allow-mock \
  --keep-going
```

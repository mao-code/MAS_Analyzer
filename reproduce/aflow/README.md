# AFlow-Style Reproduction Runner

This branch implements a lightweight AFlow-style runner for the repository's
existing benchmark adapters. It is intended for practical reproduction
experiments when the goal is to compare the framework behavior on our benches,
not to run the original AFlow benchmark suite.

## Paper Alignment

- Uses workflow candidates composed from operator nodes such as `Generate`,
  `Review`, `Test`, and `Ensemble`.
- Evaluates multiple candidate workflows on validation tasks.
- Selects the best workflow by benchmark score.
- Writes per-candidate traces so the selected workflow can be inspected.

The official AFlow implementation has a richer optimizer/search loop. This
runner keeps the same workflow-optimization shape while using our benchmark
registry and scoring.

## Run

```bash
python -m reproduce.aflow.run_existing_benchmarks \
  --config config/experiment.example.toml \
  --benchmark stabletoolbench \
  --task-limit 1 \
  --sample 2 \
  --max-rounds 1 \
  --validation-rounds 1 \
  --keep-going
```

`finance_agent` is excluded by default. Add benchmarks with repeated
`--benchmark` flags, or omit `--benchmark` to run all non-finance benchmarks.

For a no-cost wiring smoke test:

```bash
MAS_DISABLE_LIVE_LLM=1 python -m reproduce.aflow.run_existing_benchmarks \
  --config config/experiment.example.toml \
  --benchmark stabletoolbench \
  --task-limit 1 \
  --sample 1 \
  --run-id smoke_stabletoolbench \
  --keep-going
```

# DyLAN-Style Reproduction Runner

This branch implements a lightweight DyLAN-style dynamic agent runner for the
repository's existing benchmark adapters. It is meant to reproduce the framework
shape closely enough to run our benches, without depending on DyLAN's original
task-specific scripts.

## Paper Alignment

- Runs multiple LLM agents with role prompts over several discussion rounds.
- Shares recent peer messages between agents.
- Stops early when normalized answers reach consensus.
- Tracks simple per-agent importance and prunes to top agents after warm-up
  rounds, matching DyLAN's dynamic activation idea.

The official DyLAN code has task-specific pipelines and a more detailed
importance evaluator. This runner keeps the dynamic multi-agent network behavior
while plugging directly into our benchmark registry and scoring.

## Run

```bash
python -m reproduce.dylan.run_existing_benchmarks \
  --config config/experiment.example.toml \
  --benchmark stabletoolbench \
  --task-limit 1 \
  --agents 4 \
  --rounds 3 \
  --keep-top-k 2 \
  --keep-going
```

`finance_agent` is excluded by default. Add benchmarks with repeated
`--benchmark` flags, or omit `--benchmark` to run all non-finance benchmarks.

For a no-cost wiring smoke test:

```bash
MAS_DISABLE_LIVE_LLM=1 python -m reproduce.dylan.run_existing_benchmarks \
  --config config/experiment.example.toml \
  --benchmark stabletoolbench \
  --task-limit 1 \
  --agents 2 \
  --rounds 2 \
  --keep-top-k 1 \
  --run-id smoke_stabletoolbench \
  --keep-going
```

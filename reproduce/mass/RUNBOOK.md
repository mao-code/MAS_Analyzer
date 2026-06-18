# MASS Reproduction Runbook

This runbook is for the standalone MASS-style reproduction path under `reproduce/mass`.

## Current Entry Point

Run existing repository benchmarks with the MASS reproduction framework:

```bash
.venv/bin/python -m reproduce.mass.run_existing_benchmarks \
  --config config/experiment.example.toml \
  --keep-going
```

`finance_agent` is excluded by default.

## Overnight Run Command

The command below runs one task per supported benchmark, with paper-like three-stage MASS search:

```bash
.venv/bin/python -m reproduce.mass.run_existing_benchmarks \
  --config config/experiment.example.toml \
  --task-limit 1 \
  --max-validation-examples 1 \
  --candidates-per-stage 2 \
  --run-id overnight_non_finance_mass \
  --keep-going
```

For a larger run, increase `--task-limit`, `--max-validation-examples`, and `--candidates-per-stage`.

By default, Stage 3 follows the paper and returns the workflow-level prompt-optimized candidate even if its validation score is lower than the Stage 2 candidate. Use `--keep-best-after-global-prompt-stage` only for an engineering-oriented safer variant.

## Benchmarks

Observed with mock LLM smoke test:

- `plancraft`: runnable
- `scicode`: runnable
- `stabletoolbench`: runnable
- `webshop`: runnable
- `workbench`: runnable
- `agentbench`: blocked unless the local AgentBench controller is running at `localhost:5000`
- `browsecomp`: blocked unless the decrypted dataset is configured or auto-download is enabled
- `finance_agent`: intentionally excluded

## Output

Results are written under:

```text
outputs_mass_reproduce/<run-id>/
```

Each benchmark gets:

- `mass_results.json` when it runs
- `error.json` when it is blocked

The top-level `summary.json` records all benchmark statuses.

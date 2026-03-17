# SciCode Benchmark Adapter

This folder contains notes for the SciCode benchmark integration in MAS
Analyzer.

## What This Benchmark Does

- Uses the upstream
  [`scicode-bench/SciCode`](https://github.com/scicode-bench/SciCode) dataset.
- Runs MAS/SAS agents through the official multi-step code-generation workflow.
- Evaluates generated code by assembling per-step solutions and executing the
  official test pipeline.

## Folder Layout

- `README.md`: adapter notes and usage.
- `../scicode.py`: benchmark adapter implementation.
- `data/test_data.h5`: numeric evaluation data file used by the official test
  logic.

## Prompting Behavior

The adapter mirrors the official multi-step setup:

- one `BenchmarkTask` per main problem
- one LLM generation per sub-step
- prompt construction from previous steps plus the next-step function header

Supported prompt styles:

- `with_background = false`: background-comment template
- `with_background = true`: multi-step template

## Evaluation Behavior

The adapter evaluates by:

- extracting python code from each sub-step response,
- assembling runnable files,
- executing the official-style test subprocess,
- scoring based on passing the generated tests.

This benchmark is code-execution based rather than string-match based.

## Typical Config

```toml
[scicode]
split = "test"
with_background = false
h5py_file = "data/test_data.h5"
```

## Data Sync

If `data/test_data.h5` is missing, the adapter attempts to download it from a
Hugging Face mirror automatically.

Manual fallback:

- download the file from the official linked mirror or Drive source
- place it at `data/test_data.h5`

## Run Example

```bash
uv run python main.py run \
  --config config/experiment.toml \
  --benchmark scicode \
  --task-limit 5 \
  --runs-per-task 1
```

## Notes

- The adapter follows the upstream multi-step reasoning and execution flow
  closely.
- It uses a MAS-compatible wrapper around the official stepwise generation
  pattern rather than the original upstream CLI/evaluator entrypoint.

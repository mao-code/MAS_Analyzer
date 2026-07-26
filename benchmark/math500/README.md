# MATH-500 Benchmark Adapter

This folder contains notes for the MATH-500 benchmark integration in MAS
Analyzer.

## What This Benchmark Does

- Uses the [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
  dataset: 500 competition math problems sampled from the MATH test set (the
  subset used in OpenAI's *Let's Verify Step by Step*).
- Pure reasoning benchmark: one-shot generation, **no tools**. Each run is a
  single `runner.run_task(...)` call.
- The prompt asks the agent to reason step by step and put the final answer in
  `\boxed{...}`.

## Evaluation Behavior

- The final answer is extracted from the **last** `\boxed{...}` in the
  prediction. Fallbacks: a trailing "the answer is ..." phrase, then the last
  non-empty line.
- Equivalence uses the canonical Hendrycks MATH normalization
  (`strip_string` / `is_equiv`): fraction canonicalization, `\left`/`\right`
  and unit stripping, sqrt brace fixing, etc., plus a small numeric fallback
  (`abs(a - b) < 1e-6`).
- `success` is exact match after normalization; `score` is 1.0 / 0.0.

## Config Keys (all optional)

```toml
[math500]
split = "test"                          # dataset split (only "test" exists)
dataset_name = "HuggingFaceH4/MATH-500" # HF dataset id
dataset_path = ""                       # local JSONL override (offline/tests)
```

A local `dataset_path` JSONL needs `problem` and `answer` fields per line;
`solution`, `subject`, `level`, and `unique_id` are carried into metadata when
present.

## Usage

```bash
python main.py run --config config/experiment.toml --benchmark math500 --task-limit 10 --runs-per-task 1
bash scripts/full_experiment.sh --benchmark math500   # uses config/benchmarks/math500_10.toml
```

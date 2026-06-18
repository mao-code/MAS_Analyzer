# Reproduce Notes: AFlow (`arXiv:2410.10762`)

Paper: [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762)

Official repo: https://github.com/FoundationAgents/AFlow

Branch prepared: `reproduce/aflow-2410-10762`

## What AFlow Is

AFlow optimizes agentic workflows represented as code/graphs. It uses a Monte Carlo Tree Search style loop with:

- workflow graph selection
- LLM-driven graph modification
- operator descriptions
- execution feedback
- validation score updates
- convergence checks

## Official Repo Status

The official repo is available and actively structured for reproduction.

Key files:

- `run.py`: main optimization entry point
- `scripts/optimizer.py`: graph optimization loop
- `scripts/evaluator.py`: dataset evaluator routing
- `scripts/operators.py`: Generate / Review / Revise / Ensemble / Test style operators
- `scripts/workflow.py`: workflow representation
- `benchmarks/`: built-in benchmark implementations
- `data/download_data.py`: dataset and raw experiment download helper

Official supported datasets in `run.py`:

- `DROP`
- `HotpotQA`
- `MATH`
- `GSM8K`
- `MBPP`
- `HumanEval`
- `LiveCodeBench`

## Dependencies

AFlow uses modern OpenAI client dependencies:

- `openai==1.82.0`
- `pydantic==2.11.5`
- `pandas==2.2.3`
- `numpy==2.0.2`
- `aiofiles`
- `tree-sitter`

This is closer to our current environment than DyLAN, but should still be isolated if we run the official repo directly.

## Fit For Our Benchmarks

AFlow is better suited than DyLAN for plugging in custom benchmarks because the repo already defines a `BaseBenchmark` interface requiring:

- `evaluate_problem`
- `calculate_score`
- `get_result_columns`

Best reproduction path for our benchmarks:

1. Keep official AFlow repo behavior as close as possible.
2. Add a wrapper benchmark class that maps our `BenchmarkTask` into AFlow's `BaseBenchmark`.
3. Reuse AFlow's optimizer loop and operators.
4. Configure OpenRouter-compatible models through `config/config2.yaml`.

## First Experiment Recommendation

Start by reproducing AFlow on one of our stable benchmarks through a small custom benchmark adapter:

- benchmark: `stabletoolbench` or `workbench`
- sample: `1`
- max rounds: `1-2`
- validation rounds: `1`
- output path: `outputs_aflow_reproduce/<benchmark>`

For official paper-style reproduction, use built-in datasets first:

```bash
python run.py --dataset MATH --sample 4 --max_rounds 20 --validation_rounds 5
```

For our project, the more valuable path is adapting `BaseBenchmark` to our benchmark registry.

# Standalone MASS Reproduction Framework

This package is intentionally separate from the production `MAS/` runtime.

Use it when:

- you only want to reproduce the MASS paper's search pattern
- you want to plug in your own benchmark
- you do not want topology/runtime experiments to affect the main codepath

## Core pieces

- `models.py`: search-space, workflow, candidate, and result dataclasses
- `interfaces.py`: benchmark adapter and prompt optimizer interfaces
- `executor.py`: standalone MASS-style candidate executor with optional external execution feedback
- `adapters.py`: benchmark adapter template for custom tasks
- `topology.py`: MASS-style workflow enumeration
- `optimizer.py`: MIPRO-like instruction+exemplar candidate search, identity fallback, and optional DSPy adapter shell
- `framework.py`: 3-stage MASS-style search loop with paper-like block warm-up, influence scoring, and pruned topology sampling
- `run_existing_benchmarks.py`: paper-default MASS runner for this repo's benchmarks
- `paper_baselines.py`: standalone paper baseline suite using only repo benchmark loading/evaluation

## Expected benchmark adapter

Your benchmark only needs to implement:

1. `validation_examples(limit=None)`
2. `execute_candidate(candidate, example)`
3. `evaluate_candidate(candidate, examples)`

The framework does not assume any specific dataset, task format, or executor.

## Typical usage

```python
from reproduce.mass import (
    BenchmarkExample,
    MASSConfig,
    MASSFramework,
    SearchSpace,
    TemplateBenchmarkAdapter,
)

search_space = SearchSpace(
    enabled_blocks=("aggregate", "reflect", "execute"),
    aggregate=(1, 3, 5),
    reflect=(0, 1, 2, 3, 4),
    execute=(False, True),
    max_agent_budget=8,
)

framework = MASSFramework(
    config=MASSConfig(task_name="my_benchmark", search_space=search_space),
    benchmark=TemplateBenchmarkAdapter(
        examples=[
            BenchmarkExample(example_id="1", prompt="task prompt", reference_answer="expected"),
        ],
    ),
)
results = framework.run()
```

## Notes

- The default prompt optimizer is `MIPROLikePromptOptimizer`, which proposes 10 instruction
  candidates, records a 10-round candidate search trace, and bootstraps up to 3 exemplars.
- This is a research scaffold, not a claim of exact author code parity.
- Exact paper-faithful reproduction still depends on your evaluator, prompts, and search budget.
- The topology stage now follows the paper more closely: block influence is measured in Stage 1, converted into softmax probabilities, and used for rejection-style pruning before workflow sampling.
- A true DSPy/MIPRO backend is still optional; the included optimizer is a dependency-light
  approximation with deterministic candidate scoring.
- The executor makes topology blocks observable and lets coding/tool tasks provide an
  `execution_callback`; if none is supplied, it surfaces public-test/tool metadata when available
  before falling back to a model-generated execute response.

## Existing Benchmark Runner

Run MASS core on repo benchmarks with paper-like defaults:

```bash
.venv/bin/python -m reproduce.mass.run_existing_benchmarks \
  --config config/experiment.example.toml \
  --benchmark stabletoolbench \
  --task-limit 1 \
  --keep-going
```

Defaults are set to match the paper setup where applicable:

- `--model google/gemma-4-31b-it`
- `--temperature 0.7`
- `--max-tokens 4096`
- `--candidates-per-stage 10`
- `--validation-repeats 3`
- `--topology-temperature 0.05`

The runner maps repo benchmarks onto Table 2-style task families to choose enabled blocks.
Override with repeated `--enabled-block` flags when needed.

## Paper Baselines

Run the paper's manually specified baselines without using the production `MAS/` runtime:

```bash
python -m reproduce.mass.run_paper_baselines \
  --benchmark stabletoolbench \
  --task-limit 1 \
  --model google/gemma-4-31b-it \
  --keep-going
```

Implemented baseline specs from App. B.2:

- `cot`: zero-shot chain-of-thought with "Please think step by step and then solve the task."
- `self_consistency`: SC@9, temperature 0.8, rule-based majority vote.
- `self_refine`: one predictor plus reflector/refiner loop, up to 5 reflection rounds.
- `debate`: 3 agents for 3 debate rounds plus one judging aggregator.

The runner imports repo code only for `benchmark.get_benchmark()`, `load_tasks()`, and
`evaluate()`. It uses its own minimal OpenRouter HTTP client and defaults to
`google/gemma-4-31b-it`.

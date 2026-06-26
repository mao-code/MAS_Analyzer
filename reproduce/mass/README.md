# Standalone MASS Reproduction Framework

This package is intentionally separate from the production `MAS/` runtime.

Use it when:

- you only want to reproduce the MASS paper's search pattern
- you want to plug in your own benchmark
- you do not want topology/runtime experiments to affect the main codepath

## Core pieces

- `models.py`: search-space, workflow, candidate, and result dataclasses
- `interfaces.py`: benchmark adapter and prompt optimizer interfaces
- `executor.py`: standalone MASS-style candidate execution skeleton
- `adapters.py`: benchmark adapter template for custom tasks
- `topology.py`: MASS-style workflow enumeration
- `optimizer.py`: MIPRO-like instruction+exemplar optimizer, identity fallback, and optional DSPy adapter shell
- `framework.py`: 3-stage MASS-style search loop with paper-like block warm-up, influence scoring, and pruned topology sampling
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

- The default prompt optimizer is `MIPROLikePromptOptimizer`, which rewrites both instructions and exemplars.
- This is a research scaffold, not a claim of exact author code parity.
- Exact paper-faithful reproduction still depends on your evaluator, prompts, and search budget.
- The topology stage now follows the paper more closely: block influence is measured in Stage 1, converted into softmax probabilities, and used for rejection-style pruning before workflow sampling.
- A true DSPy/MIPRO backend is still optional; the included optimizer is a dependency-light approximation.
- The executor skeleton makes topology blocks observable in execution, but you should still replace the model callback and scorer for real experiments.

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
- `adas`: 30 rounds, 3 validation evaluations per round, conditioned on former baseline/workflow evaluations.
- `aflow`: 20 rounds, 5 validation runs per round, `k=3`, over predefined workflow operators.

The runner imports repo code only for `benchmark.get_benchmark()`, `load_tasks()`, and
`evaluate()`. It uses its own minimal OpenRouter HTTP client and defaults to
`google/gemma-4-31b-it`.

For ADAS and AFlow, this runner uses a safe standalone reproduction: it searches among fixed
workflow families instead of executing arbitrary LLM-generated Python code.

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

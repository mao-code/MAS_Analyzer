# ADAS / Meta Agent Search Reproduction

This package adapts `ShengranHu/ADAS` (Apache-2.0, upstream commit
`2702bee8fefda42255efc5be9f60e3bd3db96ae4`) to this repository's benchmark
runner, judges, OpenRouter client, checkpointing, and trace schema.

The upstream implementation is domain-specific scripts (`_mgsm/search.py`,
`_mmlu/search.py`, etc.). This reproduction keeps the paper method shape:

- initialize the upstream seven-agent archive: CoT, self-consistency,
  self-refine, debate, step-back abstraction, quality-diversity, and dynamic
  role assignment;
- ask a meta-agent to generate a new `forward(self, taskInfo)` agent in code;
- apply two meta-agent reflexion prompts to revise the candidate;
- evaluate candidates on validation tasks;
- debug failed candidates with LLM feedback;
- update the archive and select the best validation candidate;
- run the selected candidate on held-out test tasks.

Generated code is executed in a restricted namespace. It can call
`LLMAgentBase`, `Info`, and small Python helpers, but cannot access filesystem,
subprocess, sockets, or arbitrary imports. Benchmark environments, tools, and
judges remain owned by the existing benchmark adapters.

## Smoke

```bash
uv run python -m reproduce.adas.run_existing_benchmarks \
  --benchmark math500 \
  --task-limit 3 \
  --validation-task-limit 1 \
  --final-task-offset 1 \
  --final-task-limit 1 \
  --runs-per-task 1 \
  --search \
  --search-generations 1 \
  --workers 1 \
  --resume \
  --model google/gemma-4-31b-it \
  --temperature 1 \
  --max-tokens 0
```

## Formal Run

Use `scripts/baselines/run_adas_formal.sh`. It runs the five table benchmarks in this
order:

`BrowseComp, StableToolBench, PlanCraft, WorkBench, MATH, Average`

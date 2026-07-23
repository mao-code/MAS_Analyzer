# PlanCraft Benchmark Adapter

This folder contains notes for the PlanCraft benchmark integration in MAS
Analyzer.

## What This Benchmark Does

- Uses the official
  [`gautierdag/plancraft`](https://github.com/gautierdag/plancraft) package.
- Runs MAS/SAS agents in an interactive Minecraft crafting environment.
- Evaluates whether the agent crafts the target item, or correctly outputs
  `impossible` for impossible tasks.

## Folder Layout

- `README.md`: adapter notes and usage.
- `../plancraft.py`: benchmark adapter implementation.

## Agent Interaction Format

The environment uses the official text action format:

- `move: from [I2] to [A1] with quantity 3`
- `smelt: from [I5] to [I6] with quantity 1`
- `impossible: <reason>`

The MAS runner is called one step at a time inside the adapter's interactive
loop.

## Evaluation Behavior

The adapter scores a run as successful when either:

- the environment returns positive reward for crafting the target item
- the task is impossible and the final prediction is exactly `impossible`

Reported details include:

- `reward`
- `num_steps`
- `terminated`
- `truncated`
- `recipe_type`
- `complexity`

## Typical Config

```toml
[plancraft]
split = "val"
max_steps = 30
resolution = "high"
recipe_search = true  # upstream read-only `search: <item>` action; no evaluation labels
```

Useful split values include:

- `val`
- `test`
- `val.small`
- `test.small`

## Run Example

```bash
uv run python main.py run \
  --config config/experiment.toml \
  --benchmark plancraft \
  --task-limit 10 \
  --runs-per-task 1
```

## Notes

- The adapter uses the official prompt helpers from the `plancraft` package.
- Recipe search renders upstream's `RECIPES` objects exhaustively, including every accepted
  ingredient alternative rather than one randomly sampled valid grid. It returns recipe
  instructions as a normal environment observation, consumes an environment step, and never
  exposes the example's `impossible` label, optimal path, reward, or reference answer.
- The action format is aligned with the current upstream package.
- This is MAS-compatible, but it is not a 1:1 copy of the upstream
  `Evaluator` harness.

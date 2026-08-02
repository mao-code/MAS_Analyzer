# Reproducing the experiments

This is the end-to-end path from a fresh clone to comparable numbers. Read it in order —
each step assumes the previous one succeeded.

## 0. Requirements

- Python **3.11+** (`pyproject.toml` pins `requires-python = ">=3.11"`)
- An [OpenRouter](https://openrouter.ai) API key — every system routes its LLM calls through it
- Roughly 1–2 GB of disk for benchmark assets, plus experiment artifacts (a full 4-benchmark
  × 8-system × 30-task × 3-run sweep produces tens of GB under `artifacts/`)
- Optional, per benchmark only: `TOOLBENCH_KEY` (StableToolBench), `TAVILY_API_KEY` and
  `SEC_EDGAR_API_KEY` (FinanceAgent), `HF_TOKEN` (assets behind a gated HF repo),
  `ANTHROPIC_API_KEY` (only for the `claude_agent_sdk` harness backend), Docker (AgentBench)

## 1. Install

```bash
uv sync
```

`pyproject.toml` is the single source of truth for dependencies. If you do not use `uv`:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

Verify the install:

```bash
python main.py list-benchmarks
```

## 2. Configure

Two files, both gitignored, both copied from a checked-in example:

```bash
cp config/experiment.example.toml config/experiment.toml
cp .env.example .env
```

Put `OPENROUTER_API_KEY` in `.env` (or `openrouter.api_key` in the TOML — the env var wins).
`config/experiment.example.toml` documents every `[<benchmark>]` section inline; the
`[mas]` section controls topology and runtime, and `[self_evolved]` controls the dynamic
topology system.

## 3. Verify the harness offline

The suite runs without an API key. `MAS/llm.py` falls back to a deterministic mock mode when
no key is configured, so this exercises the whole pipeline for free:

```bash
pytest
```

480 tests should pass. If they do, config → benchmark → MAS run → trace → descriptor is wired
up correctly and any later failure is a credentials or benchmark-asset problem, not a code one.

## 4. One real run

```bash
python main.py run \
  --config config/experiment.toml \
  --benchmark browsecomp \
  --task-limit 1 \
  --runs-per-task 1
```

This writes a full artifact tree for one task. Inspect `run_0.trace.jsonl`,
`run_0.eval.json` and `analysis.json` before scaling up — see [traces.md](traces.md) for
what each file contains and [metrics.md](metrics.md) for how the numbers are defined.

## 5. Benchmark assets

Most adapters download what they need on first use into `.cache/` or `benchmark/*/data/`.
Two need manual setup:

| Benchmark | Setup |
|---|---|
| StableToolBench | Start the virtual tool server: `python scripts/stabletoolbench_virtual_server.py --port 8080`, then set `STABLETOOLBENCH_VIRTUAL_SERVER_URL`. Details in [`benchmark/stabletoolbench/README.md`](../benchmark/stabletoolbench/README.md). |
| AgentBench | Needs Docker and a separate clone of the upstream repo. Details in [benchmarks.md](benchmarks.md). |

Per-benchmark success definitions and asset layouts are in [benchmarks.md](benchmarks.md).

## 6. Batch experiments

`scripts/full_experiment.sh` is the wrapper that produces the comparable numbers. It sweeps
every **system × benchmark × model** and is resumable — re-running skips completed tasks.

The eight systems it runs:

| Label | Topology |
|---|---|
| `sas` | single agent (the baseline) |
| `orchestrator_tree_structure` | 5 agents, tree |
| `orchestrator_no_discussion` | 4 agents, no peer discussion |
| `orchestrator_with_discussion` | 4 agents, 2 discussion rounds |
| `only_voting` | 4 agents, vote only |
| `fully_linked_debate` | 4 agents, all-to-all |
| `group_chat_debate` | 4 agents, shared channel |
| `self_evolved` | topology planned per task ([self-evolved.md](self-evolved.md)) |

```bash
# everything the config directory knows about
bash scripts/full_experiment.sh

# scoped, small
TASK_LIMIT=2 RUNS_PER_TASK=1 bash scripts/full_experiment.sh --benchmarks browsecomp,workbench

# one system only
ONLY_SYSTEMS=self_evolved bash scripts/full_experiment.sh
```

Common knobs: `TASK_LIMIT`, `RUNS_PER_TASK`, `BENCHMARKS`, `MODELS`, `ONLY_SYSTEMS`,
`EXPERIMENT_ID`, `OUTPUT_ROOT`, `MAX_PARALLEL`.

Output lands in:

```text
artifacts/full_experiment/<experiment-id>/<benchmark>/<system>/<task_id>/
```

### The two paper runs

```bash
bash scripts/full_selfevo_bw.sh    # BrowseComp + WorkBench
bash scripts/full_selfevo_ps.sh    # PlanCraft + StableToolBench  (run after the first)
```

> **Run these sequentially, in one process.** Online skill learning is on by default
> (`self_evolved.skill_update_batch_size = 12`), so a run rewrites `config/topology_skill.md`
> partway through. Two concurrent runs will interleave their writes and corrupt the learned
> skill. Set `skill_update_batch_size = 0` if you need to parallelize.

## 7. Analyze

```bash
python main.py summarize-experiment --experiment-root artifacts/full_experiment/<experiment-id>
```

`summarize-experiment` produces the per-task rollups and `summary.csv`. For failure-mode
breakdowns use `scripts/generate_mas_failure_analysis_report.py`, and for cross-topology
comparison see [topology-analysis.md](topology-analysis.md).

## Reproducibility caveats

These are properties of the setup, not bugs — report them alongside any numbers you publish.

- **Provider non-determinism.** Models are served through OpenRouter. Some routes (notably
  `google/gemma-4-31b-it:nitro`) are non-deterministic even at temperature 0, and throughput
  routes occasionally time out. Single-run results carry provider noise; use `RUNS_PER_TASK`
  ≥ 3 and report variance. The structural mechanisms (dedup net, read-net, breadth) are
  deterministic and provider-independent.
- **Quantization drift.** A given model slug can be served at different quantizations by
  different providers. Pin routing with `MAS_OPENROUTER_PROVIDER_ORDER` when it matters.
- **The learned skill is state.** `config/topology_skill.md` is rewritten during a run. To
  reproduce from a clean slate, reset it to the committed version first; to reproduce a
  *continuation*, keep the version you had.
- **Ground truth never feeds the planner.** The long-term playbook and skill are built from
  process signals only (`is_process_clean`), never from `benchmark.evaluate(...).success`.
  This is deliberate — feeding the held-out verdict back into planner memory would bias the
  study. See [self-evolved.md](self-evolved.md).

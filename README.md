# Trace-Derived Task Descriptors for MAS Architecture Selection

This repo implements an engineering-first research framework that turns agent interaction traces into a reproducible **Task Descriptor** vector, then uses that descriptor to support **Multi-Agent System (MAS) architecture selection** and explainable failure-boundary analysis.

## Why this project exists

Recent controlled studies show MAS gains are not stable: performance depends on task structure, topology, and model/system behavior. The missing piece is not "can MAS be better", but an operational way to answer:

1. For a given agentic task, when does MAS outperform a strong single-agent system?
2. If MAS helps, which topology is best (centralized, decentralized, hybrid, independent)?
3. When MAS becomes negative-return, what is the failure mechanism and boundary?

This repo focuses on the **Task-side**: we compute a portable, trace-derived descriptor `x(task)` as a standard input to architecture selection, rather than relying on ad-hoc heuristics.

## Key idea

For each task, we run a fixed **probe system P** a small number of times (typically 3 to 5), log structured traces, then compute a high-dimensional descriptor vector:

`x(task) = [xQ, xC, xR, xP]`

- `xQ`: success and quality signals
- `xC`: cost and efficiency signals
- `xR`: stability and reliability signals
- `xP`: process and structure signals

Because `x(task)` is derived from traces and repeated runs, it captures uncertainty and failure modes as first-class properties.

## Repository layout

- `benchmark/`
  - Task sets and benchmark runners, each packaged as a Python module.
  - Each benchmark provides tasks plus (optional) evaluation hooks (exact-match, unit tests, judge stubs).
- `MAS/`
  - Agent systems and MAS topologies.
  - Each system must emit trace events following the shared schema.
- `descriptor/`
  - Trace schema, trace IO, metric extraction, descriptor construction.
  - Robust scaling, Mahalanobis distance, Pareto frontier, ideal-point selection.
  - Optional: stage-level bottlenecks and 2D embeddings.
- `scripts/`
  - Shell helpers, test scripts, and testing configuration files.
- `config/`
  - User-specific experiment configs (gitignored). See `config/experiment.example.toml` for a template.
- `main.py`
  - CLI entrypoint. Choose benchmark, MAS candidates, probe system, number of runs, seeds, outputs.

All modules are connected through small interfaces so you can swap benchmarks and MAS implementations without touching descriptor code.

## Trace schema (stable contract)

Each run writes a JSONL trace: an ordered event sequence `e1..eT`. Each event includes:

- `timestamp_start`, `timestamp_end`
- `actor` (LLM role, tool, env, evaluator)
- `event_type`: `plan | act | tool_call | tool_result | verify | revise | finalize | error`
- `payload` (minimal summaries: tool name, error code, artifact hash)
- `token_in`, `token_out`, `latency_ms`, `cost_usd`
- `state_id` (optional but recommended for loop detection)

This schema is designed so metrics are fully recomputable from logs.

## Metrics (minimum viable set)

The descriptor implements the following trace-derived metrics (grouped by Q/C/R/P).

### Q: Success and Quality
- `Q1 success_rate`: successes / N
- `Q2 completion_rate`: produced final artifact / N

### C: Cost and Efficiency
- `C1 latency_p95`: p95 over run latencies
- `C2 tokens_total`: sum(token_in + token_out)
- `C3 cost_total`: sum(cost_usd)
- `C4 tool_calls_total`: count(tool_call)
- `C5 tool_error_rate`: tool_fail / tool_calls
- `C6 communication_count`: total directed inter-agent edges (for each message, count each sender→recipient pair; broadcasts count multiple)
- `C7 handoff_count`: count of active-agent transitions across events, excluding `system` actor events

### R: Stability and Reliability
- `R1 success_var`: Bernoulli variance across repeated runs (or bootstrap)
- `R2 latency_var`: variance or IQR across runs
- `R3 tokens_var`: variance or IQR across runs

### P: Process and Structure
- `P1 steps_total`: total events `T`
- `P2 backtrack_rate`: (#revise + #redo) / T
- `P3 loop_score`: repeated `state_id` ratio or repeated event-pattern ratio (defined in code)
- `P4 verification_density`: #verify / T

Optional extensions behind flags:
- `avg_branching`, `unique_tools`, `failure_mode_hist`, `executability_score`

## Selection utilities
This repo provides reusable utilities for architecture selection:

1. **Robust scaling**: `(x - median) / IQR` per dimension to reduce outlier sensitivity.
2. **Mahalanobis distance**: distance with correlation awareness across descriptor dimensions.
3. **Pareto frontier**: compute non-dominated candidate topologies under multi-objective trade-offs.
4. **Ideal point selection**: pick best on Pareto set by weighted distance to an ideal point:
   `d_ideal(x) = || W (x - x*) ||_2`

## Stage-level bottlenecks
Traces can be segmented into stages (plan/retrieve/act/verify/revise/finalize) and per-stage metrics computed to surface bottlenecks:
- verify is sparse → hallucination risk
- retrieve has tool failures → repeated retries and latency spikes
- revise dominates tokens → unstable planning/execution loop

## Quickstart

### 1. Install dependencies

```bash
# Python 3.11+ is required (pyproject: requires-python >=3.11)
# Using uv (recommended):
uv sync

# Or using pip:
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Create experiment config

```bash
cp config/experiment.example.toml config/experiment.toml
```

- `openrouter.api_key` can be set in `config/experiment.toml`.
- `OPENROUTER_API_KEY` environment variable overrides the config value when both are set.
- If no valid key is present, MAS runtime uses deterministic local mock fallback so experiments remain runnable.

### 3. Inspect benchmark adapters

```bash
python main.py list-benchmarks
python main.py benchmark-info --benchmark finance_agent --config config/experiment.toml
python main.py benchmark-info --benchmark browsecomp --config config/experiment.toml
python main.py benchmark-info --benchmark stabletoolbench --config config/experiment.toml
```

### 4. Run an experiment

```bash
python main.py run --config config/experiment.toml --benchmark finance_agent --task-limit 1 --runs-per-task 1
```

Outputs are written to:

- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.trace.jsonl`
- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.answer.txt`
- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.metadata.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.result.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.trajectory.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.trajectory.md`
- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.eval.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/descriptor.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/descriptor.csv`
- `outputs/<timestamp>/<benchmark>/<task_id>/analysis.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/task.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/task_summary.json`
- `outputs/<timestamp>/summary.json`
- `outputs/<timestamp>/summary.csv`

## OpenRouter Setup

The client uses OpenRouter via the OpenAI-compatible endpoint:

- Base URL: `https://openrouter.ai/api/v1`
- Chat endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Optional attribution headers: `HTTP-Referer`, `X-Title`

Model routing is controlled by `[models]` in config:

- `models.default` is required.
- Per-agent-type model selection uses `models.<agent_type>` when present.

## LangGraph Topology Experiments

`MAS.runner.MASRunner` now executes through **LangGraph** with a topology-aware relay layer.
Supported topology names:

- `sas`
- `orchestrator_tree_structure`
- `orchestrator_no_discussion`
- `orchestrator_with_discussion`
- `only_voting`
- `fully_linked_debate`
- `group_chat_debate`

### Python API

```python
from MAS import run_experiment

run_experiment(topology="sas", agents=1, prompt="Solve: 2+2")
run_experiment(topology="fully_linked_debate", agents=5, rounds=3, prompt="Solve: 2+2")
run_experiment(topology="orchestrator_no_discussion", agents=4, rounds=2, prompt="Solve: 2+2")
```

You can inject a descriptor hook (monitor/evaluation node) via `descriptor=...`.
Each run records:

- inter-agent relay messages
- per-agent message views
- full trace events (`TraceEvent` JSONL compatible)

### CLI / Shell Scripts

```bash
bash scripts/full_experiment.sh --topology sas --agents 1 --rounds 1 --prompt "Solve: 2+2"
bash scripts/full_experiment.sh --experiment-id 20260328T120000Z --task-limit 3
bash scripts/full_experiment.sh --setup-only --benchmarks agentbench,stabletoolbench
```

`scripts/full_experiment.sh` now supports two modes:

- Legacy single-run passthrough:
  - `bash scripts/full_experiment.sh --topology sas --agents 1 --rounds 1 --prompt "Solve: 2+2"`
  - This still calls `python -m MAS.experiment_cli`.
- Batch benchmark mode:
  - Runs every benchmark config under `config/benchmarks/*.toml` across the defined MAS systems.
  - First bootstraps benchmark dependencies, data, and local services from a Python orchestrator.
  - AgentBench setup clones `THUDM/AgentBench` into `.cache/external/AgentBench`, installs its v0.2 requirements in an isolated venv, builds the OS Docker images, and starts the local controller/worker for the configured task.
  - StableToolBench setup ensures query/cache assets exist and starts a lightweight local cache-backed `/virtual` server from this repo when the configured URL is local.
  - BrowseComp setup repairs the decrypted dataset path automatically when the repo-local dataset exists, or downloads it when needed.
  - Writes artifacts as `artifacts/full_experiment/<experiment-id>/<benchmark>/<system>/<task_id>/`.
  - Each system folder contains `mas_graph.png`, `mas_graph.mmd`, `mas_graph.json`, `experiment_settings.json`, `summary.json`, and `summary.csv`.
  - Each task folder contains the raw trace plus compact result, metadata, and full trajectory files.
  - `--setup-only` prepares everything without launching experiments.
  - It enforces live LLM execution by setting `MAS_REQUIRE_LIVE_LLM=1`, so provider failures abort the run instead of falling back to the local mock client.
  - The shell wrapper accepts environment-variable defaults for common knobs:
    `TASK_LIMIT`, `RUNS_PER_TASK`, `BENCHMARKS`, `EXPERIMENT_ID`, `OUTPUT_ROOT`,
    `CONFIG_DIR`, `SKIP_SETUP=1`, and `SETUP_ONLY=1`.

Recommended batch invocation:

```bash
TASK_LIMIT=2 RUNS_PER_TASK=1 bash scripts/full_experiment.sh
```

You can also request the hierarchical layout directly from `main.py`:

```bash
python main.py run \
  --config config/benchmarks/browsecomp_10.toml \
  --benchmark browsecomp \
  --output-layout hierarchical \
  --experiment-id 20260328T120000Z \
  --system-label sas \
  --topology sas \
  --agents 1 \
  --mas-rounds 1
```

## Benchmark Notes

See [benchmarks/README.md](benchmarks/README.md) for detailed setup and configuration notes for each supported benchmark adapter.

## Package Naming

- Canonical benchmark package is `benchmark/`.
- `benchmarks/` is kept as a compatibility shim that re-exports from `benchmark/`.

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
- `Q3 faithfulness`: evaluator hook (LLM-judge or rules), optional stub
- `Q4 context_relevancy`: evaluator hook (LLM-judge or IR metrics), optional stub

### C: Cost and Efficiency
- `C1 latency_p95`: p95 over run latencies
- `C2 tokens_total`: sum(token_in + token_out)
- `C3 cost_total`: sum(cost_usd)
- `C4 tool_calls_total`: count(tool_call)
- `C5 tool_error_rate`: tool_fail / tool_calls

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
```

### 4. Run an experiment

```bash
python main.py run --config config/experiment.toml --benchmark finance_agent --task-limit 1 --runs-per-task 1
```

Outputs are written to:

- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.trace.jsonl`
- `outputs/<timestamp>/<benchmark>/<task_id>/run_<n>.eval.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/descriptor.json`
- `outputs/<timestamp>/<benchmark>/<task_id>/descriptor.csv`
- `outputs/<timestamp>/<benchmark>/<task_id>/analysis.json`
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
bash scripts/run_langgraph_experiment.sh --topology sas --agents 1 --rounds 1 --prompt "Solve: 2+2"
bash scripts/run_all_topologies.sh
```

The shell wrappers run `python -m MAS.experiment_cli` through `conda run -n agents` when Conda is available.

Artifacts are written under `outputs/langgraph_topologies*` (trace JSONL, metadata JSON, final answer text).

## Benchmark Notes

### FinanceAgent adapter

- Loads the pinned public CSV from the referenced commit and caches it locally.
- Uses a rubric proxy score (`correctness` hit ratio minus `contradiction` hit ratio).
- This is intentionally lightweight and not leaderboard-parity with the full upstream tool harness.

### SciCode adapter

- Replicates the official multi-step reasoning and evaluation pipeline.
- Automatically attempts to download the 1.0GB `test_data.h5` from a Hugging Face mirror if it's not present in `data/test_data.h5`.
- Manual download (if auto-download fails): [Google Drive Link](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link).
- Place the file at `data/test_data.h5` before running evaluation.

### BrowseComp adapter

Preferred setup:

- BrowseComp assets are organized under `benchmark/browsecomp/`:
  - `benchmark/browsecomp/data/browsecomp_plus_decrypted.jsonl`
  - `benchmark/browsecomp/topics-qrels/qrel_evidence.txt`
  - `benchmark/browsecomp/topics-qrels/qrel_golds.txt`
- Provide local decrypted JSONL path via `browsecomp.decrypted_path` (optional if the default file above exists).
- Pull official benchmark assets (qrels + repo) via:
  - `bash scripts/pull_browsecomp_plus.sh`

Optional auto-download/decrypt mode:

- Requires `datasets` package and Hugging Face authentication for gated dataset access.
- For `Tevatron/browsecomp-plus`, public access is available in most environments.

Quick smoke-test setup (no judge API cost):

- Set `browsecomp.auto_download = true`
- Set `browsecomp.eval_mode = "substring"`
- Run with small limits, e.g. `--task-limit 2 --runs-per-task 1`

Tooling setup (official-style retrieval tool interface):

- `browsecomp.enable_tools = true` enables `search` and optional `get_document` tools.
- `browsecomp.tool_k` controls top-k retrieval results (default `5`, aligned with BrowseComp-Plus paper setup).
- `browsecomp.tool_snippet_max_tokens` controls snippet truncation budget (default `512` words/tokens approximation).
- `browsecomp.include_get_document = true` registers the document fetch tool in addition to search.
- `browsecomp.max_tool_iterations` controls per-agent tool-calling loop depth.

SAS vs MAS setup:

- SAS baseline: set `[mas]` to `number_of_agents = 1`, `levels = 1`, `turn_mode = "single_turn"`.
- MAS run: increase `number_of_agents`, use `levels > 1` and/or `turn_mode = "multi_turn"` with `max_turns > 1`.

Higher-fidelity setup (LLM judge):

- Set `browsecomp.eval_mode = "llm_judge"`
- Configure `browsecomp.judge_model` and OpenRouter/OpenAI credentials
- Keep `judge_temperature` and prompt format aligned with official settings

Official heavy-parity components (not required for this lightweight adapter):

- `pyserini` + Java 21 for BM25 parity.
- `faiss` + `tevatron` for dense retrieval parity.
- `vllm` + GPU for official LLM-judge parity.

### AgentBench adapter

Source: [THUDM/AgentBench](https://github.com/THUDM/AgentBench) · Paper: [arXiv 2308.03688](https://arxiv.org/abs/2308.03688)

AgentBench evaluates LLMs as autonomous agents in interactive environments (OS shell, database, web browsing, etc.). Each task runs inside a Docker container managed by the official AgentBench Task Server; our adapter replaces the official `AgentClient` with `MASRunner` so that all LLM calls flow through our trace system.

**Prerequisites:**

- Docker Desktop running
- AgentBench repo cloned and dependencies installed

**Setup (one-time):**

```bash
# In a separate directory (not inside MAS_Analyzer)
git clone https://github.com/THUDM/AgentBench
cd AgentBench
git checkout v0.2          # main branch removed start_task; v0.2 is required
pip install -r requirements.txt

# Build Docker images for OS tasks
docker build -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles --tag local-os/ubuntu
```

**Start the Task Server (keep running in a separate terminal):**

```bash
# Start controller + 1 OS worker
python -m src.start_task -a -s os-std 1
```

The `-a` flag auto-starts the Controller (Flask API on `http://localhost:5000/api`). Once started, Docker containers will be created for the task environment.

> **Note (macOS):** Port 5000 is often occupied by AirPlay Receiver. If you get an `address already in use` error, either disable AirPlay Receiver in System Settings → General → AirDrop & Handoff, or start the controller manually on a different port:
> ```bash
> python -m src.server.task_controller --port 5555 &
> python -m src.start_task --controller http://localhost:5555/api -s os-std 1
> ```
> Then update `controller_address` in your config accordingly.

**Run from MAS_Analyzer:**

```bash
uv run python main.py run \
  --config scripts/test_agentbench.toml \
  --benchmark agentbench \
  --task-limit 3
```

**Config reference (`[agentbench]` section in TOML):**

| Key | Default | Description |
|-----|---------|-------------|
| `controller_address` | `http://localhost:5000/api` | AgentBench Controller URL |
| `task_name` | `os-std` | Task to run (must match a registered worker name, e.g. `os-std`, `os-dev`, `dbbench-std`) |
| `max_turns` | `30` | Maximum interaction rounds per task |
| `timeout` | `120` | HTTP request timeout in seconds |

**Available tasks and resource requirements:**

| Task | Environment | Memory |
|------|-------------|--------|
| `os-std` / `os-dev` | Ubuntu shell | < 500 MB |
| `dbbench-std` | MySQL database | < 500 MB |
| `alfworld` | Virtual household | < 500 MB |
| `webshop` | Online shopping | ~15 GB ⚠️ |
| `knowledgegraph` | Freebase SPARQL | ~27 GB ⚠️ |

**Execution flow:**

1. `load_tasks()` → `GET /get_indices` to fetch available sample indices from the Task Server
2. `run()` → For each task:
   - `POST /start_sample` to initialize a Docker session
   - Loop while `status == "running"`: pass environment history to `MASRunner`, send agent response via `POST /interact`
   - Environment validates the answer and returns final status
3. `evaluate()` → `status == "completed"` means success (score 1.0); other statuses mean failure (score 0.0)

## Package Naming

- Canonical benchmark package is `benchmark/`.
- `benchmarks/` is kept as a compatibility shim that re-exports from `benchmark/`.

# Benchmark Adapters

This directory contains documentation and notes for the benchmark adapters
supported in MAS_Analyzer.

## FinanceAgent adapter

- Loads the pinned public CSV from the referenced commit and caches it locally.
- Uses a rubric proxy score (`correctness` hit ratio minus `contradiction` hit
  ratio).
- This is intentionally lightweight and not leaderboard-parity with the full
  upstream tool harness.

## SciCode adapter

- Replicates the official multi-step reasoning and evaluation pipeline.
- Automatically attempts to download the `test_data.h5` file from a Hugging
  Face mirror if it is not present in `data/test_data.h5`.
- Manual download fallback:
  [Google Drive Link](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link)
- Place the file at `data/test_data.h5` before running evaluation.

## BrowseComp adapter

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

## StableToolBench adapter

- Wraps the upstream StableToolBench solvable-query split and GPT-based virtual server.
- Each task's `api_list` is exposed as OpenAI-compatible tools to MAS agents.
- Tool calls are proxied to the upstream `/virtual` endpoint; the server itself must be started separately.
- Evaluation modes:
  - `heuristic`: cheap local smoke test
  - `llm_judge`: SoPR-style solve-status grading on solvable queries
- Small query assets can auto-download into `benchmark/stabletoolbench/data/solvable_queries/`.
- Large `tools/` and `tool_response_cache/` assets can auto-download when `stabletoolbench.auto_download_server_assets = true`.
- Setup details: [`benchmark/stabletoolbench/README.md`](../benchmark/stabletoolbench/README.md)

## AgentBench adapter

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

## PlanCraft adapter

- Uses the official `plancraft` package and MAS-owned interactive run loop.
- Supports official split names such as `val`, `test`, `val.small`, and `test.small`.
- Uses the current official action string format:
  `move: from [I2] to [A1] with quantity 3`
- This adapter is MAS-compatible but is not a 1:1 copy of the upstream `Evaluator` harness.

## WorkBench adapter

- Loads the official processed CSV task files and sandbox state from the upstream WorkBench repository into `.cache/workbench`.
- Exposes the workplace tools as OpenAI-compatible tools to MAS agents.
- Evaluation follows the upstream state-change semantics instead of grading the final natural-language answer.
- `company_directory.find_email_address` is always included, matching upstream toolkit behavior.
- The MAS runtime sanitizes dotted tool names before sending them to OpenAI-compatible providers, then restores the original names for traces and evaluation.

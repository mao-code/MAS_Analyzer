# StableToolBench Benchmark Adapter

This adapter integrates **StableToolBench** into MAS Analyzer with the
upstream **GPT-based virtual server**.

It does not reimplement the StableToolBench server. The server still runs
outside this repo. MAS Analyzer only:

- loads StableToolBench solvable query files,
- exposes each task's `api_list` as OpenAI-compatible tools,
- proxies tool calls to the StableToolBench `/virtual` endpoint,
- scores each run with a cheap local heuristic or a SoPR-style LLM judge.

## What Gets Benchmarked

StableToolBench is built on ToolBench and focuses on a more stable tool-use
environment:

- tasks come from the upstream `solvable_queries` split,
- the tool environment is served by the StableToolBench virtual API server,
- the official headline metric is **SoPR** (Solvable Pass Rate),
- pairwise model comparison uses **SoWR** (Solvable Win Rate).

In this repo:

- `main.py run --benchmark stabletoolbench` produces **per-run solve scores**,
- `eval_mode = "llm_judge"` is the closest match to official SoPR,
- official cross-model SoWR is documented here but not executed inside
  `main.py`, because SoWR needs pairwise comparison across two model outputs.

## Environment Setup

### 1. Start the upstream StableToolBench virtual server

Clone the official repo somewhere outside this project:

```bash
git clone https://github.com/THUNLP-MT/StableToolBench.git
cd StableToolBench
python -m venv .venv-stb
source .venv-stb/bin/activate
pip install -r requirements.txt
```

Download the upstream server assets:

- `server/tools/`
- `server/tool_response_cache/`

These are described in the official StableToolBench README. The GPT-based
virtual server uses:

- `tools/` for API documentation,
- `tool_response_cache/` for cached real responses,
- your `OPENAI_API_KEY` for fallback simulation when cache misses occur.

In this repo, those large assets can also be auto-downloaded into
`benchmark/stabletoolbench/` from Hugging Face if you enable:

```toml
[stabletoolbench]
auto_download_server_assets = true
```

The adapter fetches:

- `stabletoolbench/ToolEnv2404` -> `toolenv2404_filtered.tar.gz` -> `tools/`
- `stabletoolbench/Cache` -> `server_cache.zip` -> `tool_response_cache/`

This download is opt-in because the extracted assets are large.

Edit `server/config.yml` and set at least:

```yaml
api_key: "<your OpenAI key>"
api_base: "https://api.openai.com/v1"
model: "gpt-4.1-mini"
tools_folder: "./tools"
cache_folder: "./tool_response_cache"
toolbench_url: "http://8.218.239.54:8080/rapidapi"
port: 8080
```

Then start the server:

```bash
cd server
python main.py
```

By default MAS Analyzer expects:

```text
http://localhost:8080/virtual
```

You can override that with:

- config: `stabletoolbench.virtual_server_url`
- env: `STABLETOOLBENCH_VIRTUAL_SERVER_URL`

### 2. Configure MAS Analyzer

Small query assets can auto-download directly from GitHub. Large server assets
can also auto-download when explicitly enabled.

Minimal config example:

```toml
[openrouter]
api_key = ""
base_url = "https://openrouter.ai/api/v1"

[experiment]
output_dir = "outputs"
runs_per_task = 1
seed = 42
task_limit = 5

[mas]
levels = 1
intra_level_link_ratio = 1.0
full_linked = true
number_of_agents = 1
agent_types = ["general"]
communication_count_internally = 0
turn_mode = "single_turn"
max_turns = 1

[models]
default = "openai/gpt-4.1-mini"

[stabletoolbench]
auto_download = true
auto_download_server_assets = false
task_sets = ["G1_instruction"]
virtual_server_url = "http://localhost:8080/virtual"
enable_tools = true
max_tool_iterations = 8
eval_mode = "llm_judge"
judge_model = "gpt-4.1-mini"
```

## How The Adapter Works

### Task loading

The adapter reads:

- `solvable_queries/test_instruction/<group>.json`
- `solvable_queries/test_query_ids/<group>.json`

The supported groups are:

- `G1_instruction`
- `G1_category`
- `G1_tool`
- `G2_instruction`
- `G2_category`
- `G3_instruction`

If they are missing locally and `auto_download = true`, the adapter downloads
them from the upstream GitHub repo into:

```text
benchmark/stabletoolbench/data/solvable_queries/
```

### Server assets

If `tools/` or `tool_response_cache/` are missing locally and
`auto_download_server_assets = true`, the adapter downloads and extracts them
into:

```text
benchmark/stabletoolbench/tools/
benchmark/stabletoolbench/tool_response_cache/
```

These folders are large and are good candidates for `.gitignore`.

### Tool exposure

For each task, every item in the upstream `api_list` becomes one runtime tool.

Example upstream fields:

- `category_name`
- `tool_name`
- `api_name`
- `required_parameters`
- `optional_parameters`

The MAS runtime sees these as OpenAI-compatible tools. When the model calls a
tool, the adapter sends a POST request to the StableToolBench server:

```json
{
  "category": "...",
  "tool_name": "...",
  "api_name": "...",
  "tool_input": "{\"param\": \"value\"}",
  "strip": "truncate",
  "toolbench_key": ""
}
```

The virtual server returns the tool output, which is fed back into the model.

### Finish behavior

Upstream ToolBench trajectories include a `Finish` tool. MAS Analyzer does not
use that pattern. In this adapter, the agent returns a plain final answer
string instead of issuing a terminal `Finish` tool call.

## Evaluation

### `eval_mode = "heuristic"`

Cheap local smoke-test grading:

- empty answer -> `Unsolved`
- refusal/apology answer -> `Unsolved`
- very short answer -> `Unsure`
- otherwise -> `Solved`

This mode is for plumbing checks and regression tests, not benchmarking.

### `eval_mode = "llm_judge"`

This is the recommended mode in this repo.

It approximates official **SoPR** behavior on the solvable-query split:

- `Solved` -> score `1.0`
- `Unsure` -> score `0.5`
- `Unsolved` -> score `0.0`

The judge focuses on whether the answer satisfies the user request, not whether
it followed a specific trajectory format.

Judge credentials:

- config: `stabletoolbench.judge_api_key`
- env fallback: `OPENAI_API_KEY`

Judge base URL:

- config: `stabletoolbench.judge_api_base`
- env fallback: `OPENAI_API_BASE`
- default: `https://api.openai.com/v1`

### Official SoWR

Official StableToolBench also reports **SoWR**, which compares two candidate
systems pairwise. That is not integrated into `main.py` because this repo's
benchmark interface evaluates one run at a time.

The intended workflow here is:

1. run SAS and MAS variants through `main.py`,
2. compare `summary.json` / `summary.csv`,
3. add an external pairwise comparison step later if strict SoWR parity is
   required.

## Running This Benchmark

Single run:

```bash
python main.py run \
  --config config/experiment.toml \
  --benchmark stabletoolbench \
  --task-limit 5 \
  --runs-per-task 1
```

All topologies smoke test:

```bash
bash scripts/run_stabletoolbench_all_topologies.sh
```

The script defaults to:

- `5` samples,
- all seven SAS/MAS topologies,
- `heuristic` evaluation,
- mock MAS LLMs unless you opt into a live provider.

## Practical Notes

- The virtual server is the heavy dependency. This adapter only wraps it.
- If you want a real MAS model and only have `OPENAI_API_KEY`, you can still
  run the MAS client by pointing the repo's OpenAI-compatible client at
  `https://api.openai.com/v1`.
- If you already use OpenRouter in this repo, you can keep doing that for MAS
  inference while still using the StableToolBench GPT-based virtual server for
  tools and OpenAI for judge calls.

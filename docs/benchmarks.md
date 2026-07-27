# Benchmark adapters

Per-benchmark success definitions, capability framing, and setup notes for the supported suite.
The precise metric definitions live in [metrics.md](metrics.md); this document covers what
`success` means for each individual benchmark and what you need to set up to run it.

## Shared evaluation semantics

Across the repo:

- `success` always means benchmark-level correctness from `benchmark.evaluate(...)`
- `completion` always means the MAS run produced a final artifact / final answer without an explicit runtime failure
- `eval_avg_score` is the benchmark-native score scale and may differ across benchmarks

So:

- success is benchmark-specific
- completion is runtime-level
- completion does not imply correctness

## Paper-aligned benchmark map

| Benchmark | Capability focus | Workflow property | Success definition |
|---|---|---|---|
| FinanceAgent | Decomposable task execution | Parallelizable subtasks | Rubric proxy score meets `success_threshold` |
| BrowseComp-Plus | Open-ended information exploration | High-entropy search space | Substring match or LLM judge marks the final answer correct |
| PlanCraft | Sequential planning and execution | Strong dependency chain | Crafts target item or correctly declares the task impossible |
| WorkBench | Tool-intensive workflows | Frequent tool interactions | Upstream state-change evaluator marks the task correct |
| AgentBench | General task completion ability | Cross-domain interactive reasoning | Official task-server sample status is `completed` |
| StableToolBench | Tool-calling reliability | API coordination | Heuristic or LLM judge labels the answer `Solved` |
| WebShop | Interactive decision making | Multi-step exploration | Final environment reward is `1.0` |
| SciCode | Program synthesis and precise computation | Deterministic verification | All official sub-steps are correct |

## Adapter notes

### FinanceAgent

- Success is `score >= success_threshold`.
- Score is a lightweight rubric proxy, not leaderboard parity with the full upstream harness.
- The adapter caches the pinned public CSV locally.

### SciCode

- Replicates the official multi-step reasoning and verification flow closely.
- Success is `total_correct == total_steps`.
- If `data/test_data.h5` is missing, the adapter attempts to download it automatically.
- Manual fallback:
  [Google Drive Link](https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR?usp=drive_link)

### BrowseComp-Plus

- Success comes from either:
  - `eval_mode = "substring"`: normalized exact / numeric match
  - `eval_mode = "llm_judge"`: judge model says the extracted final answer is correct
- Retrieval and citation metrics are logged in `run_<n>.eval.json` details but are not used as the boolean success signal.

Preferred local asset layout:

- `benchmark/browsecomp/data/browsecomp_plus_decrypted.jsonl`
- `benchmark/browsecomp/topics-qrels/qrel_evidence.txt`
- `benchmark/browsecomp/topics-qrels/qrel_golds.txt`

Useful setup notes:

- `browsecomp.decrypted_path` can point to a local decrypted JSONL.
- With `browsecomp.auto_download = true` (the default), the adapter downloads and decrypts the
  dataset from Hugging Face on first use and caches it at `decrypted_path`. Set `HF_TOKEN` if
  the source repo requires authentication. Set `auto_download = false` to require a local file.
- `browsecomp.enable_tools = true` enables `search` and optional `get_document`.
- `browsecomp.tool_k` defaults to `5`, matching the BrowseComp-Plus paper setup.
- `browsecomp.eval_mode = "substring"` is the cheap smoke-test mode.
- `browsecomp.eval_mode = "llm_judge"` is the higher-fidelity setting.

### StableToolBench

- Wraps the upstream solvable-query split and virtual tool server.
- Success is:
  - `answer_status == "Solved"` in `heuristic` mode
  - `answer_status == "Solved"` from the LLM judge in `llm_judge` mode
- Tool calls are proxied through the upstream `/virtual` endpoint.
- Setup details live in [`benchmark/stabletoolbench/README.md`](../benchmark/stabletoolbench/README.md).

### AgentBench

Source: [THUDM/AgentBench](https://github.com/THUDM/AgentBench)  
Paper: [arXiv 2308.03688](https://arxiv.org/abs/2308.03688)

- Success follows the official task-server semantics: `SampleStatus.COMPLETED`.
- This adapter swaps the upstream `AgentClient` for `MASRunner` so all LLM calls flow through the shared trace system.

Prerequisites:

- Docker Desktop running
- AgentBench repo cloned separately

One-time setup:

```bash
git clone https://github.com/THUDM/AgentBench
cd AgentBench
git checkout v0.2
pip install -r requirements.txt

docker build -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles --tag local-os/ubuntu
```

Start the task server:

```bash
python -m src.start_task -a -s os-std 1
```

Then run from this repo:

```bash
uv run python main.py run \
  --config config/experiment.toml \
  --benchmark agentbench \
  --task-limit 3
```

(`config/benchmarks/agentbench_10.toml` holds the benchmark-specific settings used by the
batch wrapper.)

### PlanCraft

- Uses the official `plancraft` package with a MAS-owned interactive loop.
- Success means either:
  - the target was crafted, or
  - the task is impossible and the agent answered `impossible`
- Supports official split names such as `val`, `test`, `val.small`, and `test.small`.

### WorkBench

- Loads the processed upstream task files and sandbox state into `.cache/workbench`.
- Exposes workplace tools as OpenAI-compatible tools.
- Success follows the upstream state-change evaluator instead of grading the final natural-language answer.
- `company_directory.find_email_address` is always included to match upstream toolkit behavior.

### WebShop

- Runs the interactive WebShop environment through the MAS runtime.
- Success is `final_reward == 1.0`.
- The benchmark-native score stored in `eval_avg_score` is the final reward.

### Completion interpretation for all benchmarks

The repo-level `completion_rate` is intentionally benchmark-agnostic:

- a run counts as completed when it emits a final artifact / final answer and no explicit runtime failure is recorded
- a run can complete and still fail benchmark evaluation

This makes `completion_rate` a stability / executability signal, while `success_rate` remains the correctness signal used by the benchmark.

# WorkBench Benchmark Adapter

This folder contains notes for the WorkBench benchmark integration in MAS
Analyzer.

## What This Benchmark Does

- Uses tasks from the upstream
  [`olly-styles/WorkBench`](https://github.com/olly-styles/WorkBench) repo.
- Runs MAS/SAS agents against workplace-tool tasks across domains such as
  `calendar`, `email`, `analytics`, `project_management`,
  `customer_relationship_manager`, and `multi_domain`.
- Evaluates outputs by replaying predicted tool calls into a local sandbox and
  comparing the resulting state change with the official ground-truth action
  sequence.

## Folder Layout

- `README.md`: adapter notes and usage.
- `../workbench.py`: benchmark adapter implementation.
- `.cache/workbench/`: auto-downloaded upstream CSV assets and query files.

## Tools Exposed To Agents

Depending on the task domain, the adapter exposes OpenAI-compatible tools from
the following groups:

- `calendar.*`
- `email.*`
- `analytics.*`
- `project_management.*`
- `customer_relationship_manager.*`
- `company_directory.find_email_address`

Important behavior:

- `company_directory.find_email_address` is always included, matching upstream
  toolkit behavior.
- `tool_selection = "domains"` limits tools to the current task's declared
  domains.
- `tool_selection = "all"` exposes all core domain tools.

These are tracked in traces and run metadata:

- trace events: `tool_call`, `tool_result`
- run metadata: `tool_call_counts`, `tool_calls_total`, `function_calls`

## Evaluation Behavior

WorkBench is not scored by the final natural-language answer alone.

The adapter evaluates by:

- extracting tool calls from the MAS trace,
- replaying those calls inside a fresh WorkBench sandbox,
- comparing the resulting state against the state produced by the official
  ground-truth action list.

Reported details include:

- `exact_match`: side-effect tool calls match the official answer sequence
- `correct`: final sandbox state matches ground truth
- `unwanted_side_effects`: predicted actions changed state incorrectly

## Typical Config

```toml
[workbench]
domain = "multi_domain"
tool_selection = "domains"
max_tool_iterations = 20
```

Useful domain values:

- `analytics`
- `calendar`
- `customer_relationship_manager`
- `email`
- `multi_domain`
- `project_management`

## Data Sync

The adapter auto-downloads the required upstream CSV files into
`.cache/workbench/` on first use.

This includes:

- processed tool-state CSVs
- `queries_and_answers/*.csv`
- `data/raw/email_addresses.csv`

## Run Example

```bash
uv run python main.py run \
  --config config/experiment.toml \
  --benchmark workbench \
  --task-limit 5 \
  --runs-per-task 1
```

## Notes

- The adapter follows upstream state-based evaluation semantics.
- The MAS runtime sanitizes dotted tool names before sending them to
  OpenAI-compatible providers, then restores the original tool names for traces
  and evaluation.
- This is a MAS-compatible reimplementation, not the original upstream
  execution harness.

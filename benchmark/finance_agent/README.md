# FinanceAgent Benchmark Adapter

This folder contains notes for the FinanceAgent benchmark integration in MAS
Analyzer.

## What This Benchmark Does

- Uses the public benchmark data from the upstream
  [`vals-ai/finance-agent`](https://github.com/vals-ai/finance-agent) repo.
- Runs MAS/SAS agents on financial QA tasks with tool use.
- Evaluates outputs with either:
  - `llm_judge`: rubric-based criterion judging
  - `substring`: lightweight local matching for quick iteration

## Folder Layout

- `README.md`: adapter notes and usage.
- `__init__.py`: benchmark adapter implementation and tool wiring.
- `.cache/finance_agent/public.csv`: cached upstream dataset.

## Tools Exposed To Agents

The adapter exposes the four official FinanceAgent-style tools:

- `google_web_search`
- `edgar_search`
- `parse_html_page`
- `retrieve_information`

These are tracked in traces and run metadata through the MAS runtime.

## Evaluation Modes

- `llm_judge`:
  - uses rubric criteria from the dataset
  - supports repeated judge calls with mode voting
  - higher fidelity, but requires model/API access
- `substring`:
  - fast local proxy
  - no judge API cost
  - lower fidelity than the official benchmark setup

## Typical Config

```toml
[finance_agent]
eval_mode = "llm_judge"
judge_model = "openai/gpt-4o"
judge_repeats = 3
judge_temperature = 0.0
max_tool_iterations = 8
web_search_top_n = 10
```

Optional tool credentials can come from config or environment variables:

- `TAVILY_API_KEY`
- `SEC_EDGAR_API_KEY`

## Data Sync

The adapter downloads the pinned public CSV on first use and stores it in
`.cache/finance_agent/`.

You can also override this with:

- `local_csv_path`
- `dataset_url`

## Run Example

```bash
uv run python main.py run \
  --config config/experiment.toml \
  --benchmark finance_agent \
  --task-limit 5 \
  --runs-per-task 1
```

## Notes

- The adapter uses the official public CSV from the pinned upstream commit.
- The tool layer is high-fidelity, but this remains a MAS-compatible
  reimplementation rather than the original upstream harness.
- `llm_judge` is closer to the official evaluation style; `substring` is mainly
  for smoke tests and debugging.

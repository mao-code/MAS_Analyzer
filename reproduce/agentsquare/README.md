# AgentSquare Reproduction Scaffold

This package adapts the AgentSquare paper's modular agent abstraction to this
repo's benchmark adapters.

See `PAPER_ALIGNMENT.md` for the official-code mapping and the safety difference
around generated Python module code.

AgentSquare models an agent as four module slots:

- Planning
- Reasoning
- Tool Use
- Memory

The upstream repository's search script is ALFWorld-specific, so this scaffold
keeps the module interface and workflow idea while letting existing
`benchmark.run(...)` implementations own tools, environments, side effects, and
evaluation.

Smoke example:

```bash
.venv/bin/python -m reproduce.agentsquare.run_existing_benchmarks \
  --config config/reproduce_agentsquare.example.toml \
  --benchmark math500 \
  --task-limit 1 \
  --runs-per-task 1 \
  --planning None \
  --reasoning IO \
  --tooluse None \
  --memory None
```

Tool-use benchmarks should generally enable the Tool Use module:

```bash
.venv/bin/python -m reproduce.agentsquare.run_existing_benchmarks \
  --config config/reproduce_agentsquare.example.toml \
  --benchmark stabletoolbench \
  --task-limit 1 \
  --runs-per-task 1 \
  --reasoning COT \
  --tooluse IO
```

Current scope:

- fixed module-pool execution
- iterative module search: one-slot module evolution, recombination, predictor
  ranking, validation testing, and best-agent updates
- LLM module evolution that proposes new prompt-level Planning, Reasoning,
  Tool Use, and Memory modules and records any generated code as audit-only text
- LLM in-context performance predictor with recorded heuristic fallback
- COT-SC reasoning executes three samples and returns the majority answer
- benchmark-native trace/evaluation output
- resumable per-run JSON artifacts

Search smoke:

```bash
.venv/bin/python -m reproduce.agentsquare.run_existing_benchmarks \
  --config config/reproduce_agentsquare.example.toml \
  --benchmark math500 \
  --task-limit 40 \
  --validation-task-limit 10 \
  --final-task-offset 10 \
  --final-task-limit 30 \
  --runs-per-task 3 \
  --validation-repeats 1 \
  --search \
  --search-iterations 3 \
  --max-search-candidates 3 \
  --model google/gemma-4-31b-it \
  --temperature 1 \
  --max-tokens 0 \
  --workers 4
```

Official 10-val / 30-test style run for the current benchmark suite:

```bash
uv run python -m reproduce.agentsquare.preflight \
  --config config/reproduce_agentsquare.example.toml
```

The formal launcher uses `config/reproduce_agentsquare.example.toml`, reads
secrets from `.env`, and defaults to a tractable AgentSquare search budget:
3 search iterations and at most 3 candidates per iteration. It also checks
`http://127.0.0.1:8080/virtual/healthz` and starts the local StableToolBench
cache server automatically if needed.

```bash
bash scripts/run_agentsquare_formal.sh
```

To use a stable run id for resume:

```bash
bash scripts/run_agentsquare_formal.sh agentsquare_gemma_10val_30test_T1
```

To run detached:

```bash
tmux new -s agentsquare 'bash scripts/run_agentsquare_formal.sh agentsquare_gemma_10val_30test_T1'
```

Monitor progress:

```bash
tail -f run_logs/agentsquare_agentsquare_gemma_10val_30test_T1.log
```

Artifact-based status:

```bash
uv run python -m reproduce.agentsquare.status \
  --run-root outputs_agentsquare_reproduce/agentsquare_gemma_10val_30test_T1
```

Resume behavior:

- The formal launcher always passes `--resume`.
- Search resumes from `search/search_results.json` when that file exists.
- Final and validation runs resume at per-task/per-run JSON granularity under
  `final/runs/<task_id>/run_<idx>.json` and
  `search/iteration_*/candidate_*/runs/<task_id>/run_<idx>.json`.
- Successful run JSON files are not recomputed on restart; missing files are
  filled in.
- Progress is visible from both the log and the artifact status command above.

Summarize a finished run into the table row format:

```bash
uv run python -m reproduce.agentsquare.summarize_results \
  --run-root outputs_agentsquare_reproduce/<run-id> \
  --output outputs_agentsquare_reproduce/<run-id>/agentsquare_summary.json
```

Still to add for fuller paper parity:

- executing arbitrary LLM-generated Python module code. The current runner keeps
  generated code in artifacts for audit, but executes only validated prompt-level
  module instructions to avoid unsafe benchmark-side effects.

For offline tests or cost-controlled debugging, add `--predictor-mode heuristic`.
When `--predictor-mode llm` is used but the provider is unavailable or returns
unparseable JSON, the runner records the predictor failure in
`search/iteration_*/predictor_rankings.json` and falls back to the deterministic
heuristic ranking for that iteration.

For an even cheaper search that also disables LLM module proposals, add
`--module-evolution-mode off`.

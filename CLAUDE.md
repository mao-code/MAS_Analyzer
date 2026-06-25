# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Experiment harness for an economics-of-collaboration study (see `README.md`). It runs single-agent
(SAS) and multi-agent (MAS) systems over a shared benchmark suite, records structured execution
traces, and aggregates repeated runs into a trace-derived task descriptor (`Q / C / D / R / P`:
quality, execution cost, coordination diagnostics, reliability, process). The strict metric
contract — what `success` vs `completion` mean, how every `Q*/C*/D*/R*/P*` field is computed, and
what lands in `summary.csv` — is specified in detail in `README.md` and is the source of truth; do
not silently change a metric definition.

## Commands

```bash
# Install (preferred)
uv sync
# or: python -m venv .venv && source .venv/bin/activate && pip install -e .

# Tests (pytest configured in pyproject.toml; testpaths=tests, pythonpath=.)
pytest                                  # full suite
pytest tests/test_topology.py           # one file
pytest tests/test_topology.py::test_name -v   # one test
pytest -k metrics                       # by keyword

# Lint / format (ruff, line-length 100, py311)
ruff check .
ruff check --fix .
ruff format .
pre-commit run --all-files              # runs ruff + hygiene hooks

# CLI (main.py)
python main.py list-benchmarks
python main.py benchmark-info --benchmark browsecomp --config config/experiment.toml
python main.py run --config config/experiment.toml --benchmark browsecomp --task-limit 1 --runs-per-task 1
python main.py summarize-experiment --experiment-root artifacts/full_experiment/<experiment-id>

# Batch experiments (env-driven wrapper)
bash scripts/full_experiment.sh                              # all discovered benchmark configs
TASK_LIMIT=2 RUNS_PER_TASK=8 bash scripts/full_experiment.sh --benchmarks browsecomp,workbench
```

### Tests run offline by default

`MAS/llm.py::OpenRouterLLMClient` falls back to a **mock mode** when no API key is configured
(`config.api_key` empty / client unavailable), returning deterministic `MOCK(...)` results. The test
suite relies on this — most tests are smoke tests that exercise the pipeline without real LLM calls.
Keep mock mode working when touching the LLM client. Termination/final-vote judges also fall back to
deterministic lexical voting when the judge is mocked or returns unusable JSON.

## Config

- Copy `config/experiment.example.toml` → `config/experiment.toml` (gitignored). The example file
  documents every benchmark-specific `[<benchmark>]` section inline.
- Secrets via `.env` (copy from `.env.example`): `OPENROUTER_API_KEY` is primary; per-benchmark keys
  (`TOOLBENCH_KEY`, `TAVILY_API_KEY`, `HF_TOKEN`, etc.) only matter for the benchmark you run.
- `[mas]` controls topology and runtime: `number_of_agents`, `agent_types`, `turn_mode`,
  `termination_consensus_mode` (`llm_judge` default / `lexical`), `final_vote_mode`. Config dataclasses
  and validation live in `MAS/config.py`; `validate()` is the gate for legal values.
- `[self_evolved]` configures the dynamic topology system (`topology = "self_evolved"`):
  `harness_backend` (`openrouter` / `claude_agent_sdk`), `max_initial_agents`, `max_total_agents`,
  `max_turns` (1–2), `audit_mode`, `playbook_path` (default `config/topology_playbook.json`).
  Dataclass `SelfEvolvedConfig` in `MAS/config.py`.

## Architecture

The core flow **config → benchmark task → MAS run → trace → descriptor** is driven by these packages:

- **`benchmark/`** — benchmark adapters. `benchmark/registry.py` maps names → classes (add new
  benchmarks here). All adapters implement the `BenchmarkAdapter` protocol in `benchmark/base.py`:
  `load_tasks`, `run`, `evaluate` (returns `.success`), `requirements`. `benchmark.evaluate(...).success`
  is the **only** authority on correctness — never infer success elsewhere. (`benchmarks/` is just a
  docs/compatibility shim; the real package is `benchmark/`.)

- **`MAS/`** — the SAS/MAS runtime. `MAS/langgraph_engine.py` is the heart: it builds per-agent
  prompts (`_build_agent_prompt`, `_execute_agent_stage`), runs controller/worker nodes, and centralizes
  loop termination in `_termination_decision(...)`. Topology layout in `MAS/topology.py` +
  `MAS/relay.py`; shared run state in `MAS/state.py`; structured per-step artifacts in
  `MAS/artifacts.py`; provider-native OpenAI-compatible tool loop in `MAS/llm.py`; dynamic persona
  assignment in `MAS/role_pools.py` + `MAS/role_assigner.py`. Read `MAS/TOPOLOGY.md` (how each topology
  executes) and `MAS/README.md` (prompt contract) before changing runtime behavior.

- **`MAS/self_evolved/`** — query-conditioned **dynamic topology** system (`topology = "self_evolved"`).
  An LLM Topology Planner (`planner.py`) *analyzes the task* (type / attributes / failure risks) and
  proposes a per-task `TopologySpec` (`spec.py`) using general, task-characteristic topology guidance
  (no benchmark names); deterministic orchestrator code (`engine.py`) spawns and runs it via `TurnExecutor`
  (`executor.py`), a Trace Auditor (`auditor.py`) flags process failure modes (incl.
  `insufficient_search_coverage` and `duplicate_state_mutation`), and **at most one** trace-backed repair
  mutation runs before finalize. Three correctness nets live in code, not the prompt: `TurnExecutor` dedups
  identical `(tool, args)` calls per run so a write replays once; retrieval keeps the full agent budget
  (single turn) for search breadth; and a finalize **read-net** opens the top surfaced docids and feeds full
  text to the synthesizer when a retrieval run answered without reading. Visibility is pure code
  (`context.py`). The long-term playbook is an **agent-maintained markdown skill** (`config/topology_skill.md`,
  `skill.py`): the planner loads it in full and applies it in **both** the initial plan and the repair-mutation
  prompt. An LLM **reflection agent** rewrites its *Lessons from experience* section from run outcomes labelled by
  **process signals only** — `is_process_clean` (`playbook.py`): auditor flagged no modes + decision-grade
  consensus. **Ground truth (`benchmark.evaluate(...).success`) is deliberately NOT used to build the playbook**
  — feeding the held-out verdict back into the planner's memory would bias the study; the verdict stays the
  authority for *scoring* only. Standing-principles / How-to-choose sections are guardrail-protected. Default
  writer is **online**: when `self_evolved.skill_update_batch_size > 0` (default **8**), `OnlineSkillUpdater`
  (`skill.py`, driven from `main.py::run_command`) pauses every N freshly executed runs, reflects them into the
  skill, and reloads it (`SelfEvolvedEngine.reload_skill`) for the rest of the experiment. `scripts/reflect_topology_skill.py`
  is the offline equivalent; both share `summary_from_candidate` and read process signals only (never `eval.json`).
  The legacy structured JSON playbook (`topology_playbook.json`, `playbook.py`, `update_topology_playbook.py`) is
  the deterministic fallback when no skill file exists; its `lookup` transfers entries by task shape
  (`tools::size`) across benchmarks, ranked by the same process proxy. **Online updates are ON by default, so a
  `run` rewrites the skill mid-experiment — use a single sequential process; set `skill_update_batch_size = 0` for
  parallel experiments.** The orchestration is deterministic (agents never decide termination). See the
  "Self-evolved topology system" section + diagram in `README.md`.

- **`descriptor/`** — trace schema (`descriptor/schema.py`), run-level metrics, and task-level
  aggregation (`descriptor/experiment.py::analyze_task_runs`), plus comparison tooling in
  `descriptor/topology_analysis.py` (scaling, Mahalanobis distance, Pareto, PCA/UMAP).

- **`analysis/econ_eval/`** — economic post-analysis (utility, cost/quality regime classification);
  consumed by `scripts/analyze_experiment.py`.

- **`scripts/`** — `full_experiment.py`/`.sh` (batch wrapper), `analyze_experiment.py`,
  `generate_mas_failure_analysis_report.py`, `update_topology_playbook.py` (offline process-only playbook merge),
  and the StableToolBench virtual server.

- **`main.py`** — single CLI entrypoint (subcommands: `run`, `list-benchmarks`, `benchmark-info`,
  `summarize-experiment`). Large file; it wires config → runner → descriptor → artifact writing.

### Key invariants

- **Prompt instruction priority** (enforced in the engine): structural stage contract → tool-use
  contract → task/benchmark instructions → domain persona. Personas never override stage behavior.
- Agents pass **compacted relay packets** derived from structured artifacts, never raw chat
  transcripts (full fidelity by default; a positive budget triggers structural compaction —
  drop low-priority fields, prefer the agent's own summary — never a blunt mid-string truncation).
  Visibility is controlled in code via `message_selector(...)`, not left to prompts.
- Tool-enabled answer stages must call tools when evidence is weak; the runtime never fabricates tool
  calls after the fact. Blocked/planning/no-evidence outputs are non-substantive for voting/termination.
- Agents never decide when a loop stops — controller nodes do, via the ordered checks
  `invalid_or_failed_branch → consensus_reached → no_meaningful_change → max_rounds_reached`
  (`consensus_reached` fires only when the agreement is decision-grade — avg confidence ≥ 0.5 and
  no open unresolved issues — unless no repair/round remains).

## Artifact layout

Hierarchical batch outputs:
`artifacts/full_experiment/<experiment-id>/<benchmark>/<system>/<task_id>/`. Per-run files
(`run_<n>.trace.jsonl`, `.eval.json`, `.trace_metrics.json`, etc.), per-task (`descriptor.json/csv`,
`analysis.json`), per-system (`summary.csv`, topology graphs), and experiment rollups. The trace
schema is designed so all run-level metrics are recomputable from logs — preserve that property.

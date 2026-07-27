# Economic Analysis Framework for Multi-Agent Collaboration

Experiment harness for an economics-of-collaboration study. It runs single-agent systems (SAS)
and multi-agent systems (MAS) over a shared benchmark suite, records structured execution
traces, and converts repeated runs into a trace-derived task descriptor.

The question is not just *can* multi-agent collaboration help, but **when does collaboration
improve task outcomes enough to justify its execution and coordination cost.**

The harness supports:

- quality vs. cost analysis
- MAS vs. SAS gain/cost comparison
- coordination diagnostics
- topology-level summary and Pareto analysis

## Core idea

Each task is executed one or more times under a fixed system configuration. For every run the
repo stores the benchmark-native evaluation output, structured trace events, run-level trace
metrics, and a task-level descriptor aggregated over repeated runs.

The descriptor follows a `Q / C / D / R / P` split:

| | |
|---|---|
| `Q` | outcome quality |
| `C` | direct execution cost |
| `D` | coordination diagnostics |
| `R` | run-to-run reliability |
| `P` | process structure |

Higher-level economic quantities — utility `U = Q - C`, collaboration gain `G`, coordination
cost `K` — are derived from these. This repo produces the trace-derived ingredients.

Two rules hold everywhere in the codebase:

- **`benchmark.evaluate(...).success` is the only authority on correctness.** Success is never
  inferred anywhere else.
- **Agents never decide when a loop stops.** Controller nodes do, through an ordered set of
  checks. See [docs/termination.md](docs/termination.md).

## Quickstart

```bash
uv sync
cp config/experiment.example.toml config/experiment.toml
cp .env.example .env          # then set OPENROUTER_API_KEY

python main.py list-benchmarks
python main.py run --config config/experiment.toml --benchmark browsecomp \
                   --task-limit 1 --runs-per-task 1
```

The test suite runs fully offline — `MAS/llm.py` falls back to a deterministic mock when no API
key is set, so `pytest` verifies the whole pipeline for free.

**Reproducing the experiments end to end is documented in
[docs/reproducing.md](docs/reproducing.md).** Start there rather than here.

## Documentation

| Document | Contents |
|---|---|
| [docs/reproducing.md](docs/reproducing.md) | Fresh clone → comparable numbers, plus reproducibility caveats |
| [docs/metrics.md](docs/metrics.md) | The exact metric contract — `success` vs `completion`, every `Q*/C*/D*/R*/P*` field |
| [docs/termination.md](docs/termination.md) | How controller nodes decide a loop is finished |
| [docs/traces.md](docs/traces.md) | Trace schema and artifact semantics |
| [docs/prompting.md](docs/prompting.md) | Agent prompting and tool-use design |
| [docs/self-evolved.md](docs/self-evolved.md) | The query-conditioned dynamic topology system |
| [docs/topology-analysis.md](docs/topology-analysis.md) | Scaling, Mahalanobis distance, Pareto, PCA/UMAP |
| [docs/benchmarks.md](docs/benchmarks.md) | Per-benchmark success definitions and setup notes |
| [MAS/TOPOLOGY.md](MAS/TOPOLOGY.md) | How each topology executes |
| [MAS/README.md](MAS/README.md) | The prompt contract |
| [scripts/README.md](scripts/README.md) | What each script is for, and which are one-offs |

The metric contract in `docs/metrics.md` is the source of truth. Do not silently change a
metric definition.

## Repository layout

```text
main.py              CLI entrypoint (run, list-benchmarks, benchmark-info, summarize-experiment)
cli/                 implementation of the CLI: settings, artifacts, trajectory, graphs, resume
MAS/                 SAS/MAS runtime; langgraph_engine.py is the heart
  self_evolved/      query-conditioned dynamic topology system
benchmark/           benchmark adapters; registry.py maps names → classes
descriptor/          trace schema, run metrics, task aggregation, topology analysis
analysis/econ_eval/  economic post-analysis (utility, cost/quality regimes)
reproduce/           reproductions of external baselines (ADAS, AFlow, AgentSquare, MASS)
scripts/             batch runners and analysis tools (see scripts/README.md)
  baselines/           drivers for the reproduce/ baselines
  experiments/         one-off drivers kept for provenance
config/              experiment configs (gitignored except the examples)
docs/                documentation
tests/               offline test suite
```

Artifacts are written hierarchically:

```text
artifacts/full_experiment/<experiment-id>/<benchmark>/<system>/<task_id>/
```

The trace schema is designed so that every run-level metric is recomputable from the logs.

## Development

```bash
pytest                     # full offline suite
ruff check . && ruff format .
pre-commit run --all-files
```

Contributor and coding-agent guidance lives in [CLAUDE.md](CLAUDE.md).

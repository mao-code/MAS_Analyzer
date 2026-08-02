# MANTA: Multi-Agent Network Topology Adaptation

[![arXiv](https://img.shields.io/badge/arXiv-2607.28527-b31b1b.svg)](https://arxiv.org/abs/2607.28527)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Official implementation and evaluation harness for
**[MANTA: Multi-Agent Network Topology Adaptation for Self-Evolving Multi-Agent Systems](https://arxiv.org/abs/2607.28527)**.

Most LLM multi-agent systems treat their communication topology as a fixed design choice, or
optimize it offline before deployment. MANTA lets the *structure of collaboration itself* adapt at
inference time: it plans a task-conditioned topology, watches the collaboration trace while the task
runs, and applies bounded structural repairs when the current organization proves insufficient —
changing agent roles, communication links, execution order, information visibility, and validation
paths, while holding the task interface and agent budget fixed.

**MANTA reaches the highest average score (74.0) across five benchmarks, +5.8 points over the
strongest baseline, and the best result on BrowseComp and PlanCraft.**

---

## How MANTA works

Three components, all running at inference time and all orchestrated by deterministic code — agents
never decide when the loop stops:

| Stage | What happens |
|---|---|
| **Plan** | An LLM *Topology Planner* analyzes the task (type, attributes, failure risks) and proposes a per-task topology spec — number of agents, roles, links, visibility, validation points. It is conditioned on accumulated structural experience, never on benchmark identity. |
| **Audit → Repair** | A hybrid *Trace Auditor* combines deterministic failure modes with grounded open-set observations over the live execution trace. When it flags a problem, a bounded structural **mutation** rewires the topology and the turn re-executes. Bounded by `repair_budget` (default 4) and `max_turns` (default 5). |
| **Learn** | A cross-run **playbook** — an agent-maintained markdown skill — distills topology-selection and repair experience across runs, and is reloaded mid-experiment (true online self-evolution). |

**The playbook is built from process signals only** (auditor findings + consensus quality), never
from `benchmark.evaluate(...).success`. Feeding the held-out verdict back into the planner's memory
would bias the study, so the verdict stays the authority for *scoring* only.

Full description and diagram: **[docs/self-evolved.md](docs/self-evolved.md)**.

## Main results

Success rate (%) with Gemma 4 31B, medium reasoning effort. 30 tasks per benchmark, mean ± std over
three independent runs. Best per column in bold.

| Category | System | BrowseComp | StableToolBench | PlanCraft | WorkBench | MATH | **Average** |
|---|---|---|---|---|---|---|---|
| Single-agent | Single Agent | 34.4 ±4.2 | 74.4 ±7.9 | 61.1 ±1.6 | 41.1 ±5.7 | 85.6 ±6.3 | 59.3 ±2.5 |
| | CoT | 26.7 ±5.4 | 50.0 ±7.2 | 62.2 ±12.6 | 35.6 ±4.2 | 75.6 ±3.1 | 50.0 ±3.3 |
| | Self-Consistency | 37.8 ±1.6 | 51.1 ±1.6 | 61.1 ±15.7 | 15.6 ±1.6 | 78.9 ±4.2 | 48.9 ±3.3 |
| | Self-Refine | 14.4 ±3.1 | 68.9 ±1.6 | 62.2 ±15.0 | 35.6 ±4.2 | **96.7** ±2.7 | 55.6 ±3.2 |
| Static MAS | Voting | 43.3 ±2.7 | 85.6 ±1.6 | 61.1 ±1.6 | 41.1 ±1.6 | 92.2 ±1.6 | 64.7 ±0.8 |
| | Group Chat Debate | 61.1 ±3.1 | 82.2 ±5.7 | 72.2 ±3.1 | 21.1 ±4.2 | 91.1 ±4.2 | 65.5 ±1.9 |
| | Fully Linked Debate | 58.9 ±9.6 | 81.1 ±5.7 | 73.3 ±2.7 | 21.1 ±4.2 | 91.1 ±1.6 | 65.1 ±2.5 |
| | Orchestrator w/o Discussion | 53.3 ±2.7 | 82.2 ±1.6 | 74.4 ±1.6 | 23.3 ±4.7 | 94.4 ±3.1 | 65.5 ±1.3 |
| | Orchestrator w/ Discussion | 64.4 ±4.2 | 80.0 ±0.0 | 73.3 ±2.7 | 20.0 ±2.7 | 93.3 ±0.0 | 66.2 ±1.1 |
| | Orchestrator Tree | 54.4 ±5.7 | 78.9 ±3.1 | 62.2 ±3.1 | 16.7 ±2.7 | 94.4 ±1.6 | 61.3 ±1.6 |
| Adaptive MAS | AFlow | 12.2 ±3.1 | 66.7 ±5.4 | 21.1 ±4.2 | 61.1 ±4.2 | **96.7** ±0.0 | 51.6 ±1.7 |
| | ADAS | 48.9 ±1.6 | 77.8 ±4.2 | 57.8 ±13.4 | **66.7** ±0.0 | 90.0 ±0.0 | 68.2 ±3.2 |
| | AgentSquare | 32.2 ±1.6 | **88.9** ±5.7 | 34.4 ±6.8 | 62.2 ±3.1 | **96.7** ±2.7 | 62.9 ±1.7 |
| | MASS | 50.0 ±2.7 | 50.0 ±5.4 | 70.0 ±0.0 | 46.7 ±0.0 | 95.6 ±1.6 | 62.5 ±1.2 |
| **Ours** | **MANTA** | **76.7** ±4.7 | 82.2 ±3.1 | **76.7** ±3.3 | 43.3 ±2.7 | 91.1 ±5.7 | **74.0** ±1.8 |

## What is in this repository

Two things, in one harness:

1. **MANTA itself** — `MAS/self_evolved/`, selected with `topology = "self_evolved"`.
2. **The evaluation harness that produced every number above** — all single-agent and static-MAS
   baselines, nine benchmark adapters, reproductions of the four automated-design baselines
   (`reproduce/`: ADAS, AFlow, AgentSquare, MASS), and the tracing and scoring pipeline.

Every run records a structured execution trace — the same trace the Auditor reads during execution
and the playbook is distilled from — so results are inspectable and every run-level metric is
recomputable from the logs.

Two invariants hold everywhere in the codebase:

- **`benchmark.evaluate(...).success` is the only authority on correctness.** Success is never
  inferred anywhere else.
- **Agents never decide when a loop stops.** Controller nodes do, through an ordered set of checks
  (see [docs/termination.md](docs/termination.md)).

## Quickstart

Requires Python 3.11+ and an [OpenRouter](https://openrouter.ai) API key.

```bash
uv sync                                          # or: pip install -e .
cp config/experiment.example.toml config/experiment.toml
cp .env.example .env                             # then set OPENROUTER_API_KEY
```

```bash
python main.py list-benchmarks
python main.py run --config config/experiment.toml --benchmark browsecomp \
                   --topology self_evolved --task-limit 1 --runs-per-task 1
```

The test suite runs **fully offline** — `MAS/llm.py` falls back to a deterministic mock when no API
key is set, so `pytest` verifies the whole config → benchmark → run → trace → score pipeline for
free before you spend a single token.

```bash
pytest
```

## Running systems

Select the system with `--topology` (or the `[mas]` section of your config):

| `--topology` | Paper row |
|---|---|
| `sas` | Single Agent |
| `only_voting` | Voting |
| `group_chat_debate` | Group Chat Debate |
| `fully_linked_debate` | Fully Linked Debate |
| `orchestrator_no_discussion` | Orchestrator w/o Discussion |
| `orchestrator_with_discussion` | Orchestrator w/ Discussion |
| `orchestrator_tree` | Orchestrator Tree |
| `self_evolved` | **MANTA** |

Prompting baselines run on the SAS runtime via `--prompting-baseline {cot,self_consistency,self_refine}`.
The automated-design baselines (ADAS, AFlow, AgentSquare, MASS) live under `reproduce/`, driven by
`scripts/baselines/` — each reproduction's own README is the authority on how to run it.

Benchmark adapter names map to the paper as: `browsecomp` → BrowseComp, `stabletoolbench` →
StableToolBench, `plancraft` → PlanCraft, `workbench` → WorkBench, `math500` → MATH. Four further
adapters ship with the harness but are outside the paper table: `agentbench`, `finance_agent`,
`scicode`, `webshop`.

Batch sweeps use the env-driven wrapper:

```bash
TASK_LIMIT=30 RUNS_PER_TASK=3 bash scripts/full_experiment.sh --benchmarks browsecomp,workbench
```

> **Note on online learning.** MANTA rewrites `config/topology_skill.md` mid-experiment by default
> (`self_evolved.skill_update_batch_size = 12`). Use a **single sequential process**; set it to `0`
> if you want to run experiments in parallel and reflect the skill offline afterwards with
> `scripts/reflect_topology_skill.py`.

**The full path from a fresh clone to comparable numbers is
[docs/reproducing.md](docs/reproducing.md).** Start there rather than here.

## Documentation

| Document | Contents |
|---|---|
| [docs/self-evolved.md](docs/self-evolved.md) | MANTA in full: planner, auditor, repair mutations, playbook |
| [docs/reproducing.md](docs/reproducing.md) | Fresh clone → comparable numbers, plus reproducibility caveats |
| [docs/metrics.md](docs/metrics.md) | The exact metric contract — `success` vs `completion`, and how each is computed |
| [docs/benchmarks.md](docs/benchmarks.md) | Per-benchmark success definitions and setup notes |
| [docs/termination.md](docs/termination.md) | How controller nodes decide a loop is finished |
| [docs/traces.md](docs/traces.md) | Trace schema and artifact semantics |
| [docs/prompting.md](docs/prompting.md) | Agent prompting and tool-use design |
| [MAS/TOPOLOGY.md](MAS/TOPOLOGY.md) | How each static topology executes |
| [MAS/README.md](MAS/README.md) | The prompt contract |
| [scripts/README.md](scripts/README.md) | What each script is for, and which are one-offs |

The metric contract in `docs/metrics.md` is the source of truth. Do not silently change a metric
definition.

## Repository layout

```text
main.py              CLI entrypoint (run, list-benchmarks, benchmark-info, summarize-experiment)
cli/                 implementation of the CLI: settings, artifacts, trajectory, graphs, resume
MAS/                 SAS/MAS runtime; langgraph_engine.py is the heart
  self_evolved/      MANTA: planner, executor, auditor, repair mutations, playbook/skill
benchmark/           benchmark adapters; registry.py maps names → classes
descriptor/          trace schema, run metrics, and aggregation over repeated runs
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

## Development

```bash
pytest                     # full offline suite
ruff check . && ruff format .
pre-commit run --all-files
```

Contributor and coding-agent guidance lives in [CLAUDE.md](CLAUDE.md).

## Citation

```bibtex
@article{huang2026manta,
  title   = {MANTA: Multi-Agent Network Topology Adaptation for Self-Evolving Multi-Agent Systems},
  author  = {Huang, MaoXun and Wang, Jerry and Lai, Yi-Cheng and Zhang, Zhenxing and Cardie, Claire and Huang, Hen-Hsen},
  journal = {arXiv preprint arXiv:2607.28527},
  year    = {2026},
  url     = {https://arxiv.org/abs/2607.28527}
}
```

## Acknowledgements

This work builds on the BrowseComp, StableToolBench, PlanCraft, WorkBench, and MATH benchmarks, and
on the ADAS, AFlow, AgentSquare, and MASS reference implementations. See each adapter's and
reproduction's README for upstream attribution.

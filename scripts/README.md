# scripts/

Two tiers, by how likely you are to need them.

## Top level — the supported tooling

These are the scripts a new user runs. They are generic over benchmark, model and
topology, and they are what the main [README](../README.md) and
[docs/reproducing.md](../docs/reproducing.md) refer to.

| Script | Purpose |
|---|---|
| `full_experiment.py` | Batch experiment driver: every system × benchmark × model, resumable. |
| `full_experiment.sh` | Env-var wrapper around `full_experiment.py`. The usual entrypoint. |
| `full_selfevo_bw.sh` | Paper run A — self-evolved + static arms on BrowseComp + WorkBench. |
| `full_selfevo_ps.sh` | Paper run B — same, on PlanCraft + StableToolBench. Run after A. |
| `smoke_selfevo.sh` | Tiny end-to-end smoke of the self-evolved pipeline (3 tasks × 1 run). |
| `generate_mas_failure_analysis_report.py` | HTML failure-mode report from traces. |
| `run_manta_ablation.py` | MANTA ablation: one seed-42 sample per variant, 30 tasks/benchmark; supports benchmark-subset runs, including StableToolBench. |
| `run_playbook_mutation_experiment.py` | Long-term-playbook transfer + mutation-budget experiment. |
| `reflect_topology_skill.py` | Offline rewrite of `config/topology_skill.md` from run traces. |
| `update_topology_playbook.py` | Offline merge into the legacy JSON playbook fallback. |
| `stabletoolbench_virtual_server.py` | Virtual tool server required by the StableToolBench benchmark. |
| `stabletoolbench_sas_solvability.py` | Solvability report over the StableToolBench split. |

Both learning writers (`reflect_topology_skill.py`, `update_topology_playbook.py`) read
**process signals only** and never `eval.json` — see
[docs/self-evolved.md](../docs/self-evolved.md) for why that separation matters.

## `baselines/` — external-baseline drivers

Launchers for the reproductions under [`reproduce/`](../reproduce/) (ADAS, AgentSquare,
AFlow, MASS) and for the prompting baselines in `MAS/prompting_baselines.py`. Each
reproduction's own README is the authority on how to run it.

> `baselines/run_baseline_workflow_transfer.sh` contains hardcoded absolute paths
> (`/home/lai/...`) from the machine it was written on. It will not run elsewhere without
> editing `ROOT` and `AFLOW_ROOT` at the top.

Run every script from the repository root, e.g. `bash scripts/smoke_selfevo.sh`.

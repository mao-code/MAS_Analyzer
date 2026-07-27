# scripts/

Three tiers, by how likely you are to need them.

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
| `analyze_experiment.py` / `.sh` | Economic post-analysis over a finished experiment root. |
| `generate_mas_failure_analysis_report.py` | HTML failure-mode report from traces. |
| `plot_model_comparison.py` | Cross-model comparison plots. |
| `run_manta_ablation.py` | MANTA ablation: one seed-42 sample per variant, 30 tasks/benchmark. |
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

## `experiments/` — one-off drivers, kept for provenance

Each of these was written to run, repair or analyse **one specific experiment**. They hard-code
experiment ids, artifact paths, model slugs, and sometimes monkey-patch the runtime at import
time. They are kept so published numbers can be traced back to the exact code that produced
them — they are not a general-purpose toolkit, and several will not do anything useful without
the original artifact directories present.

Grouped by what they were for:

- **Self-evolved vs. static comparisons** — `compare_self_evolved_vs_static.py`,
  `compare_self_evolved_multi.py`, `compare_browsecomp_self_evolved_smoke.py`,
  `compare_browsecomp_claude_topologies.sh`, `summarize_browsecomp_topology_compare.py`
- **Give-up / refusal nets** — `exp_manta_giveup_respin.py`, `exp_plancraft_giveup_net.py`,
  `exp_plancraft_giveup_net_all.py`
- **Targeted repairs and re-runs** — `fix_selfevo_browsecomp_failures.sh`,
  `fixcheck_browsecomp_failures.py`, `rerun_plancraft_runs12.py`, `rerun_plancraft_bf16.py`,
  `rerun_workbench_crm_fix.sh`, `redo_workbench_error_runs.py`
- **Config sweeps** — `exp_plancraft_maxturns5.py`, `exp_math500.sh`
- **Full-run launchers (superseded by `full_selfevo_{bw,ps}.sh`)** — `run_full_selfevo.sh`,
  `run_browsecomp_selfevo_seq.sh`, `run_stb_selfevo_seq.sh`,
  `run_self_evolved_browsecomp_parallel.sh`, `smoke_browsecomp_self_evolved.sh`
- **Ad-hoc analysis** — `analyze_model_stats_failures.py`

Run them from the repository root, e.g. `bash scripts/experiments/exp_math500.sh`.

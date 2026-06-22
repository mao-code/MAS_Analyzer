# Experiment Analysis: analysis_input

- Experiment Root: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis_input`
- Benchmarks: 4
- Topologies: 1
- Task rows: 120

## Headline Findings

- `browsecomp`: strongest paper-aligned system is `aflow` with success `0.067`, stability `0.941` and mean tokens `15805.4`.

- `plancraft`: strongest paper-aligned system is `aflow` with success `0.378`, stability `0.881` and mean tokens `7024.6`.

- `stabletoolbench`: strongest paper-aligned system is `aflow` with success `0.811`, stability `0.793` and mean tokens `8023.1`.

- `workbench`: strongest paper-aligned system is `aflow` with success `0.322`, stability `0.941` and mean tokens `8008.2`.

## System Table

```text
benchmark       system_label  task_count  avg_accuracy  avg_eval_score  avg_success_rate  avg_stability  avg_pass_at_1  avg_pass_at_3  avg_pass_at_5  avg_pass_at_8  avg_latency_e2e  avg_token_total  avg_tokens_total  avg_cost_per_success  avg_tokens_cv  avg_tool_calls_total  avg_communication_count  avg_handoff_count
     browsecomp aflow        30          0.067         0.067           0.067             0.941          0.067          0.100          NaN            NaN            92587.945        15805.389        15805.389         28432.500             0.101          17.067                0.0                      3.0
      plancraft aflow        30          0.378         0.378           0.378             0.881          0.378          0.467          NaN            NaN             8137.144         7024.644         7024.644          6594.786             0.409           0.000                0.0                      0.0
stabletoolbench aflow        30          0.811         0.811           0.811             0.793          0.811          0.900          NaN            NaN            27872.940         8023.056         8023.056          8912.704             0.144           5.378                0.0                      1.0
      workbench aflow        30          0.322         0.322           0.322             0.941          0.322          0.367          NaN            NaN            23370.554         8008.211         8008.211         11480.121             0.180          10.222                0.0                      0.0
```

## Plot Guide

- `system_scorecard`: ranks each topology on success and token cost in one view; markers show stability and best available pass@k retry lift when present.
- `success_vs_tokens_frontier`: absolute quality vs cost tradeoff; point color encodes stability.
- `vs_sas_tradeoff`: how much success each MAS gains relative to SAS and how many extra tokens it costs.
- `coordination_breakdown`: communication and handoff overhead for each topology.
- `pass_at_k`: optional retry curve, only emitted when multiple pass@k values are available.

## Plot Inventory

- `browsecomp`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/browsecomp/RQ1/average/browsecomp_system_scorecard.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/browsecomp/RQ1/indivudal/browsecomp_utility_sas_vs_mas_indivudal.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/browsecomp/RQ1/average/browsecomp_success_vs_tokens_frontier_average.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/browsecomp/RQ1/indivudal/browsecomp_success_vs_tokens_frontier_indivudal.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/browsecomp/THEORY/average/browsecomp_pass_at_k_average.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/browsecomp/THEORY/indivudal/browsecomp_quality_cost_pareto_indivudal.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/browsecomp/THEORY/indivudal/browsecomp_mahalanobis_distance_diagnostics_indivudal.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/browsecomp/RQ2/average/browsecomp_coordination_breakdown_average.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/plancraft/RQ1/average/plancraft_system_scorecard.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/plancraft/RQ1/indivudal/plancraft_utility_sas_vs_mas_indivudal.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/plancraft/RQ1/average/plancraft_success_vs_tokens_frontier_average.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/plancraft/RQ1/indivudal/plancraft_success_vs_tokens_frontier_indivudal.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/plancraft/THEORY/average/plancraft_pass_at_k_average.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/plancraft/THEORY/indivudal/plancraft_quality_cost_pareto_indivudal.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/plancraft/THEORY/indivudal/plancraft_mahalanobis_distance_diagnostics_indivudal.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/plancraft/RQ2/average/plancraft_coordination_breakdown_average.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ1/average/stabletoolbench_system_scorecard.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ1/indivudal/stabletoolbench_utility_sas_vs_mas_indivudal.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ1/average/stabletoolbench_success_vs_tokens_frontier_average.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ1/indivudal/stabletoolbench_success_vs_tokens_frontier_indivudal.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/stabletoolbench/THEORY/average/stabletoolbench_pass_at_k_average.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/stabletoolbench/THEORY/indivudal/stabletoolbench_quality_cost_pareto_indivudal.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/stabletoolbench/THEORY/indivudal/stabletoolbench_mahalanobis_distance_diagnostics_indivudal.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ2/average/stabletoolbench_coordination_breakdown_average.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/workbench/RQ1/average/workbench_system_scorecard.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/workbench/RQ1/indivudal/workbench_utility_sas_vs_mas_indivudal.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/workbench/RQ1/average/workbench_success_vs_tokens_frontier_average.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/workbench/RQ1/indivudal/workbench_success_vs_tokens_frontier_indivudal.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/workbench/THEORY/average/workbench_pass_at_k_average.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/workbench/THEORY/indivudal/workbench_quality_cost_pareto_indivudal.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/workbench/THEORY/indivudal/workbench_mahalanobis_distance_diagnostics_indivudal.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_aflow/outputs_aflow_reproduce/aflow_gemma_30x3_T1_20260621/analysis/workbench/RQ2/average/workbench_coordination_breakdown_average.png`

## Notes

- `stability` and `tokens_cv` are left blank when a task has fewer than two runs.
- `pass_at_k` is left blank when the task has fewer than `k` repeated runs.
- `cost_per_success` is left blank when `success_rate = 0`.

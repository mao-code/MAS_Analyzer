# Experiment Analysis: analysis_input

- Experiment Root: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis_input`
- Benchmarks: 4
- Topologies: 1
- Task rows: 120

## Headline Findings

- `browsecomp`: strongest paper-aligned system is `dylan` with success `0.011`, stability `0.970` and mean tokens `33496.3`.

- `plancraft`: strongest paper-aligned system is `dylan` with success `0.000`, stability `1.000` and mean tokens `217128.1`.

- `stabletoolbench`: strongest paper-aligned system is `dylan` with success `0.444`, stability `0.793` and mean tokens `33683.7`.

- `workbench`: strongest paper-aligned system is `dylan` with success `0.178`, stability `0.881` and mean tokens `40773.8`.

## System Table

```text
benchmark       system_label  task_count  avg_accuracy  avg_eval_score  avg_success_rate  avg_stability  avg_pass_at_1  avg_pass_at_3  avg_pass_at_5  avg_pass_at_8  avg_latency_e2e  avg_token_total  avg_tokens_total  avg_cost_per_success  avg_tokens_cv  avg_tool_calls_total  avg_communication_count  avg_handoff_count
     browsecomp dylan        30          0.011         0.011           0.011             0.970          0.011          0.033          NaN            NaN            152761.977        33496.322        33496.322         86939.000            0.298          17.722                0.0                        7.133
      plancraft dylan        30          0.000         0.000           0.000             1.000          0.000          0.000          NaN            NaN            351008.320       217128.111       217128.111               NaN            0.108           0.000                0.0                      105.378
stabletoolbench dylan        30          0.444         0.444           0.444             0.793          0.444          0.567          NaN            NaN            187011.463        33683.711        33683.711         48566.471            0.078          14.322                0.0                        8.622
      workbench dylan        30          0.178         0.178           0.178             0.881          0.178          0.267          NaN            NaN            139040.768        40773.778        40773.778        111444.042            0.101          39.056                0.0                        8.133
```

## Plot Guide

- `system_scorecard`: ranks each topology on success and token cost in one view; markers show stability and best available pass@k retry lift when present.
- `success_vs_tokens_frontier`: absolute quality vs cost tradeoff; point color encodes stability.
- `vs_sas_tradeoff`: how much success each MAS gains relative to SAS and how many extra tokens it costs.
- `coordination_breakdown`: communication and handoff overhead for each topology.
- `pass_at_k`: optional retry curve, only emitted when multiple pass@k values are available.

## Plot Inventory

- `browsecomp`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/browsecomp/RQ1/average/browsecomp_system_scorecard.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/browsecomp/RQ1/indivudal/browsecomp_utility_sas_vs_mas_indivudal.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/browsecomp/RQ1/average/browsecomp_success_vs_tokens_frontier_average.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/browsecomp/RQ1/indivudal/browsecomp_success_vs_tokens_frontier_indivudal.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/browsecomp/THEORY/average/browsecomp_pass_at_k_average.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/browsecomp/THEORY/indivudal/browsecomp_quality_cost_pareto_indivudal.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/browsecomp/THEORY/indivudal/browsecomp_mahalanobis_distance_diagnostics_indivudal.png`
- `browsecomp`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/browsecomp/RQ2/average/browsecomp_coordination_breakdown_average.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/plancraft/RQ1/average/plancraft_system_scorecard.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/plancraft/RQ1/indivudal/plancraft_utility_sas_vs_mas_indivudal.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/plancraft/RQ1/average/plancraft_success_vs_tokens_frontier_average.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/plancraft/RQ1/indivudal/plancraft_success_vs_tokens_frontier_indivudal.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/plancraft/THEORY/average/plancraft_pass_at_k_average.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/plancraft/THEORY/indivudal/plancraft_quality_cost_pareto_indivudal.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/plancraft/THEORY/indivudal/plancraft_mahalanobis_distance_diagnostics_indivudal.png`
- `plancraft`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/plancraft/RQ2/average/plancraft_coordination_breakdown_average.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ1/average/stabletoolbench_system_scorecard.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ1/indivudal/stabletoolbench_utility_sas_vs_mas_indivudal.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ1/average/stabletoolbench_success_vs_tokens_frontier_average.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ1/indivudal/stabletoolbench_success_vs_tokens_frontier_indivudal.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/stabletoolbench/THEORY/average/stabletoolbench_pass_at_k_average.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/stabletoolbench/THEORY/indivudal/stabletoolbench_quality_cost_pareto_indivudal.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/stabletoolbench/THEORY/indivudal/stabletoolbench_mahalanobis_distance_diagnostics_indivudal.png`
- `stabletoolbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/stabletoolbench/RQ2/average/stabletoolbench_coordination_breakdown_average.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/workbench/RQ1/average/workbench_system_scorecard.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/workbench/RQ1/indivudal/workbench_utility_sas_vs_mas_indivudal.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/workbench/RQ1/average/workbench_success_vs_tokens_frontier_average.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/workbench/RQ1/indivudal/workbench_success_vs_tokens_frontier_indivudal.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/workbench/THEORY/average/workbench_pass_at_k_average.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/workbench/THEORY/indivudal/workbench_quality_cost_pareto_indivudal.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/workbench/THEORY/indivudal/workbench_mahalanobis_distance_diagnostics_indivudal.png`
- `workbench`: `/home/lai/github/MAS_Analyzer_dylan/outputs_dylan_reproduce/dylan_gemma_30x3_T1_20260621/analysis/workbench/RQ2/average/workbench_coordination_breakdown_average.png`

## Notes

- `stability` and `tokens_cv` are left blank when a task has fewer than two runs.
- `pass_at_k` is left blank when the task has fewer than `k` repeated runs.
- `cost_per_success` is left blank when `success_rate = 0`.

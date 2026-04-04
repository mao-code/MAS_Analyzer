# Experiment Analysis: 20260403T225842Z

- Experiment Root: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z`
- Benchmarks: 2
- Topologies: 7
- Task rows: 42
- Unique tasks: 6

## Headline Findings

- `browsecomp`: best mean score is `group_chat_debate` with score `0.333`, success `0.333`, and mean tokens `28875.0`.
- `browsecomp` SAS baseline: score `0.333`, success `0.333`, mean tokens `78184.3`.
- `browsecomp` strongest score lift vs SAS: `group_chat_debate` with mean score delta `+0.000` over `3` tasks.

- `workbench`: best mean score is `only_voting` with score `0.333`, success `0.333`, and mean tokens `16995.0`.
- `workbench` SAS baseline: score `0.000`, success `0.000`, mean tokens `10339.7`.
- `workbench` strongest score lift vs SAS: `only_voting` with mean score delta `+0.333` over `3` tasks.

## Topology Table

```text
benchmark  system_label                 topology                      task_count  avg_score  median_score  avg_success_rate  avg_latency_ms  median_latency_ms  avg_tokens  median_tokens  avg_cost_usd  avg_tool_calls  avg_communication  avg_handoffs  avg_steps  avg_loop_score  avg_verification_density  avg_agent_to_agent_communication  avg_system_mediated_communication
browsecomp            group_chat_debate            group_chat_debate 3           0.333      0.0           0.333              50899.329       52811.634          28875.000   30289.0       0.0           25.333           5.333             14.333         80.333    0.253           0.195                      3.333                            2.0                               
browsecomp                          sas                          sas 3           0.333      0.0           0.333              10499.316        6397.512          78184.333   77542.0       0.0            6.333           0.000              1.000         16.667    0.000           0.147                      0.000                            0.0                               
browsecomp orchestrator_with_discussion orchestrator_with_discussion 3           0.333      0.0           0.333              76078.899       66160.717         122052.667  112887.0       0.0           40.333           7.000             18.333        119.000    0.218           0.170                      7.000                            0.0                               
browsecomp                  only_voting                  only_voting 3           0.333      0.0           0.333              59035.057       54336.324         332450.000  379662.0       0.0           26.000           0.000              5.000         64.000    0.000           0.105                      0.000                            0.0                               
browsecomp          fully_linked_debate          fully_linked_debate 3           0.000      0.0           0.000              40517.790       40600.610          20313.667   23186.0       0.0           16.000           8.000             10.333         53.667    0.256           0.219                      8.000                            0.0                               
browsecomp  orchestrator_tree_structure  orchestrator_tree_structure 3           0.000      0.0           0.000              92179.034       81437.666          85679.333   79629.0       0.0           43.000          12.000             20.000        125.000    0.312           0.169                     10.000                            2.0                               
browsecomp   orchestrator_no_discussion   orchestrator_no_discussion 3           0.000      0.0           0.000              54682.808       56421.058         123430.333  127087.0       0.0           31.000           7.000             13.333         85.000    0.213           0.152                      7.000                            0.0                               
 workbench                  only_voting                  only_voting 3           0.333      0.0           0.333              29840.536       27414.305          16995.000   15335.0       0.0           14.667           0.000              5.000         41.333    0.000           0.146                      0.000                            0.0                               
 workbench          fully_linked_debate          fully_linked_debate 3           0.333      0.0           0.333              45178.919       42086.272          27002.333   27108.0       0.0           15.667          12.000             13.000         56.333    0.302           0.231                     12.000                            0.0                               
 workbench   orchestrator_no_discussion   orchestrator_no_discussion 3           0.333      0.0           0.333              42962.515       40894.626          28090.000   27322.0       0.0           22.667           5.000              8.000         60.333    0.000           0.133                      5.000                            0.0                               
 workbench            group_chat_debate            group_chat_debate 3           0.333      0.0           0.333              75534.576       86684.175          37028.667   37612.0       0.0           23.667           5.333             14.333         77.000    0.185           0.200                      3.333                            2.0                               
 workbench  orchestrator_tree_structure  orchestrator_tree_structure 3           0.333      0.0           0.333              74205.994       78216.350          43547.667   42843.0       0.0           39.333          12.000             20.000        117.667    0.263           0.179                     10.000                            2.0                               
 workbench orchestrator_with_discussion orchestrator_with_discussion 3           0.333      0.0           0.333             103199.562      118554.880          55018.000   62302.0       0.0           33.667           8.000             24.333        119.000    0.285           0.237                      8.000                            0.0                               
 workbench                          sas                          sas 3           0.000      0.0           0.000              13073.977        9521.892          10339.667    5945.0       0.0            6.667           0.000              1.000         17.333    0.000           0.180                      0.000                            0.0                               
```

## Topology Delta vs SAS

```text
benchmark  system_label                  task_count  mean_score_delta_vs_sas  mean_success_delta_vs_sas  mean_latency_delta_ms_vs_sas  mean_tokens_delta_vs_sas  mean_communication_delta_vs_sas  mean_tool_calls_delta_vs_sas  score_wins_vs_sas  score_ties_vs_sas  score_losses_vs_sas
browsecomp            group_chat_debate 3            0.000                    0.000                     40400.013                     -49309.333                 5.333                           19.000                        1                  1                  1                   
browsecomp orchestrator_with_discussion 3            0.000                    0.000                     65579.583                      43868.333                 7.000                           34.000                        1                  1                  1                   
browsecomp                  only_voting 3            0.000                    0.000                     48535.741                     254265.667                 0.000                           19.667                        1                  1                  1                   
browsecomp          fully_linked_debate 3           -0.333                   -0.333                     30018.474                     -57870.667                 8.000                            9.667                        0                  2                  1                   
browsecomp  orchestrator_tree_structure 3           -0.333                   -0.333                     81679.718                       7495.000                12.000                           36.667                        0                  2                  1                   
browsecomp   orchestrator_no_discussion 3           -0.333                   -0.333                     44183.493                      45246.000                 7.000                           24.667                        0                  2                  1                   
 workbench                  only_voting 3            0.333                    0.333                     16766.560                       6655.333                 0.000                            8.000                        1                  2                  0                   
 workbench          fully_linked_debate 3            0.333                    0.333                     32104.942                      16662.667                12.000                            9.000                        1                  2                  0                   
 workbench   orchestrator_no_discussion 3            0.333                    0.333                     29888.539                      17750.333                 5.000                           16.000                        1                  2                  0                   
 workbench            group_chat_debate 3            0.333                    0.333                     62460.599                      26689.000                 5.333                           17.000                        1                  2                  0                   
 workbench  orchestrator_tree_structure 3            0.333                    0.333                     61132.018                      33208.000                12.000                           32.667                        1                  2                  0                   
 workbench orchestrator_with_discussion 3            0.333                    0.333                     90125.585                      44678.333                 8.000                           27.000                        1                  2                  0                   
```

## Plot Inventory

- `overall_avg_task_score`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/overall_avg_task_score.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/browsecomp_task_score_boxplot.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/browsecomp_latency_ms_boxplot.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/browsecomp_tokens_total_boxplot.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/browsecomp_communication_count_boxplot.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/browsecomp_task_score_heatmap.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/browsecomp_accuracy_vs_tokens.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/workbench_task_score_boxplot.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/workbench_latency_ms_boxplot.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/workbench_tokens_total_boxplot.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/workbench_communication_count_boxplot.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/workbench_task_score_heatmap.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260403T225842Z/analysis/workbench_accuracy_vs_tokens.png`

## Notes

- `C3_cost_total` is omitted from most figures when it is identically zero across the experiment.
- This experiment uses only two tasks per benchmark and one run per task, so boxplots are descriptive rather than inferential.
- `workbench` appears to complete runs structurally but still scores zero on task evaluation across all tested topologies.

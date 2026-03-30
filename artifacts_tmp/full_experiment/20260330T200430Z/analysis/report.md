# Experiment Analysis: 20260330T200430Z

- Experiment Root: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z`
- Benchmarks: 2
- Topologies: 7
- Task rows: 70
- Unique tasks: 10

## Headline Findings

- `browsecomp`: best mean score is `group_chat_debate` with score `0.200`, success `0.200`, and mean tokens `12030.0`.
- `browsecomp` SAS baseline: score `0.200`, success `0.200`, mean tokens `72917.6`.
- `browsecomp` strongest score lift vs SAS: `group_chat_debate` with mean score delta `+0.000` over `5` tasks.

- `workbench`: best mean score is `fully_linked_debate` with score `0.400`, success `0.400`, and mean tokens `12731.2`.
- `workbench` SAS baseline: score `0.000`, success `0.000`, mean tokens `4709.2`.
- `workbench` strongest score lift vs SAS: `fully_linked_debate` with mean score delta `+0.400` over `5` tasks.

## Topology Table

```text
benchmark  system_label                 topology                      task_count  avg_score  median_score  avg_success_rate  avg_latency_ms  median_latency_ms  avg_tokens  median_tokens  avg_cost_usd  avg_tool_calls  avg_communication  avg_handoffs  avg_steps  avg_loop_score  avg_verification_density
browsecomp            group_chat_debate            group_chat_debate 5           0.2        0.0           0.2               27105.566       26139.725           12030.0     11790.0       0.0           14.0            0.0                 7.0          47.0       0.000           0.213                    
browsecomp                          sas                          sas 5           0.2        0.0           0.2                9433.852        6862.919           72917.6     37316.0       0.0            5.2            0.0                 1.0          14.4       0.000           0.153                    
browsecomp orchestrator_with_discussion orchestrator_with_discussion 5           0.2        0.0           0.2               45921.463       43319.681          139856.6    106885.0       0.0           28.8            6.8                12.8          80.8       0.039           0.161                    
browsecomp   orchestrator_no_discussion   orchestrator_no_discussion 5           0.2        0.0           0.2               37988.584       43762.226          182751.6    226345.0       0.0           25.0            5.0                 8.0          65.0       0.000           0.124                    
browsecomp          fully_linked_debate          fully_linked_debate 5           0.0        0.0           0.0               32053.956       31882.253           13965.8     16761.0       0.0           15.2            7.2                 9.8          51.4       0.231           0.223                    
browsecomp  orchestrator_tree_structure  orchestrator_tree_structure 5           0.0        0.0           0.0               41947.556       36586.392           93134.8     65496.0       0.0           28.6            6.8                12.0          81.8       0.064           0.158                    
browsecomp                  only_voting                  only_voting 5           0.0        0.0           0.0               59522.485       22561.194          137390.2     83170.0       0.0           13.8            0.0                 5.0          39.6       0.000           0.161                    
 workbench          fully_linked_debate          fully_linked_debate 5           0.4        0.0           0.4               25201.542       18166.518           12731.2      7147.0       0.0           10.0            4.8                 8.2          39.0       0.121           0.271                    
 workbench            group_chat_debate            group_chat_debate 5           0.4        0.0           0.4               41695.690       48562.166           21634.6     23867.0       0.0           18.0            2.4                11.8          61.0       0.148           0.215                    
 workbench  orchestrator_tree_structure  orchestrator_tree_structure 5           0.4        0.0           0.4               40436.199       33469.109           25887.8     22901.0       0.0           24.8            6.8                12.0          74.2       0.055           0.174                    
 workbench orchestrator_with_discussion orchestrator_with_discussion 5           0.2        0.0           0.2               31145.823       30776.534           21128.8     21168.0       0.0           18.6            5.0                 8.0          53.2       0.000           0.170                    
 workbench   orchestrator_no_discussion   orchestrator_no_discussion 5           0.2        0.0           0.2               47386.243       34623.399           24282.6     23466.0       0.0           21.0            5.0                 8.0          57.0       0.000           0.141                    
 workbench                          sas                          sas 5           0.0        0.0           0.0                9438.487        8693.672            4709.2      4430.0       0.0            4.4            0.0                 1.0          12.8       0.000           0.157                    
 workbench                  only_voting                  only_voting 5           0.0        0.0           0.0               36815.717       38756.377           18999.6     20049.0       0.0           18.8            0.0                 5.0          49.6       0.000           0.123                    
```

## Topology Delta vs SAS

```text
benchmark  system_label                  task_count  mean_score_delta_vs_sas  mean_success_delta_vs_sas  mean_latency_delta_ms_vs_sas  mean_tokens_delta_vs_sas  mean_communication_delta_vs_sas  mean_tool_calls_delta_vs_sas  score_wins_vs_sas  score_ties_vs_sas  score_losses_vs_sas
browsecomp            group_chat_debate 5            0.0                      0.0                       17671.715                     -60887.6                  0.0                               8.8                          1                  3                  1                   
browsecomp orchestrator_with_discussion 5            0.0                      0.0                       36487.612                      66939.0                  6.8                              23.6                          0                  5                  0                   
browsecomp   orchestrator_no_discussion 5            0.0                      0.0                       28554.733                     109834.0                  5.0                              19.8                          0                  5                  0                   
browsecomp          fully_linked_debate 5           -0.2                     -0.2                       22620.104                     -58951.8                  7.2                              10.0                          0                  4                  1                   
browsecomp  orchestrator_tree_structure 5           -0.2                     -0.2                       32513.705                      20217.2                  6.8                              23.4                          0                  4                  1                   
browsecomp                  only_voting 5           -0.2                     -0.2                       50088.633                      64472.6                  0.0                               8.6                          0                  4                  1                   
 workbench          fully_linked_debate 5            0.4                      0.4                       15763.055                       8022.0                  4.8                               5.6                          2                  3                  0                   
 workbench            group_chat_debate 5            0.4                      0.4                       32257.203                      16925.4                  2.4                              13.6                          2                  3                  0                   
 workbench  orchestrator_tree_structure 5            0.4                      0.4                       30997.712                      21178.6                  6.8                              20.4                          2                  3                  0                   
 workbench orchestrator_with_discussion 5            0.2                      0.2                       21707.336                      16419.6                  5.0                              14.2                          1                  4                  0                   
 workbench   orchestrator_no_discussion 5            0.2                      0.2                       37947.756                      19573.4                  5.0                              16.6                          1                  4                  0                   
 workbench                  only_voting 5            0.0                      0.0                       27377.230                      14290.4                  0.0                              14.4                          0                  5                  0                   
```

## Plot Inventory

- `overall_avg_task_score`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/overall_avg_task_score.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/browsecomp_task_score_boxplot.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/browsecomp_latency_ms_boxplot.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/browsecomp_tokens_total_boxplot.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/browsecomp_communication_count_boxplot.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/browsecomp_task_score_heatmap.png`
- `browsecomp`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/browsecomp_accuracy_vs_tokens.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/workbench_task_score_boxplot.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/workbench_latency_ms_boxplot.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/workbench_tokens_total_boxplot.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/workbench_communication_count_boxplot.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/workbench_task_score_heatmap.png`
- `workbench`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/artifacts/full_experiment/20260330T200430Z/analysis/workbench_accuracy_vs_tokens.png`

## Notes

- `C3_cost_total` is omitted from most figures when it is identically zero across the experiment.
- This experiment uses only two tasks per benchmark and one run per task, so boxplots are descriptive rather than inferential.
- `workbench` appears to complete runs structurally but still scores zero on task evaluation across all tested topologies.

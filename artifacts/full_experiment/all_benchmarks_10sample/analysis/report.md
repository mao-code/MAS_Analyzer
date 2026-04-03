# Experiment Analysis: all_benchmarks_10sample

- Experiment Root: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample`
- Benchmarks: 7
- Topologies: 7
- Task rows: 440
- Unique tasks: 70

## Headline Findings

- `agentbench`: best mean score is `sas` with score `0.400`, success `0.400`, and mean tokens `2585.7`.
- `agentbench` SAS baseline: score `0.400`, success `0.400`, mean tokens `2585.7`.
- `agentbench` strongest score lift vs SAS: `only_voting` with mean score delta `-0.300` over `10` tasks.

- `browsecomp`: best mean score is `sas` with score `0.300`, success `0.300`, and mean tokens `110334.1`.
- `browsecomp` SAS baseline: score `0.300`, success `0.300`, mean tokens `110334.1`.
- `browsecomp` strongest score lift vs SAS: `orchestrator_with_discussion` with mean score delta `-0.100` over `10` tasks.

- `finance_agent`: best mean score is `sas` with score `0.113`, success `0.100`, and mean tokens `118912.4`.
- `finance_agent` SAS baseline: score `0.113`, success `0.100`, mean tokens `118912.4`.
- `finance_agent` strongest score lift vs SAS: `orchestrator_no_discussion` with mean score delta `-0.013` over `10` tasks.

- `plancraft`: best mean score is `orchestrator_tree_structure` with score `0.900`, success `0.900`, and mean tokens `30347.2`.
- `plancraft` SAS baseline: score `0.800`, success `0.800`, mean tokens `3506.4`.
- `plancraft` strongest score lift vs SAS: `orchestrator_tree_structure` with mean score delta `+0.100` over `10` tasks.

- `scicode`: best mean score is `sas` with score `0.000`, success `0.000`, and mean tokens `27926.8`.
- `scicode` SAS baseline: score `0.000`, success `0.000`, mean tokens `27926.8`.
- `scicode` strongest score lift vs SAS: `only_voting` with mean score delta `+0.000` over `10` tasks.

- `stabletoolbench`: best mean score is `group_chat_debate` with score `0.700`, success `0.700`, and mean tokens `17512.6`.
- `stabletoolbench` SAS baseline: score `0.500`, success `0.500`, mean tokens `4466.2`.
- `stabletoolbench` strongest score lift vs SAS: `group_chat_debate` with mean score delta `+0.200` over `10` tasks.

- `workbench`: best mean score is `fully_linked_debate` with score `0.400`, success `0.400`, and mean tokens `6971.2`.
- `workbench` SAS baseline: score `0.100`, success `0.100`, mean tokens `5477.2`.
- `workbench` strongest score lift vs SAS: `fully_linked_debate` with mean score delta `+0.300` over `10` tasks.

## Topology Table

```text
benchmark       system_label                 topology                      task_count  avg_score  median_score  avg_success_rate  avg_latency_ms  median_latency_ms  avg_tokens  median_tokens  avg_cost_usd  avg_tool_calls  avg_communication  avg_handoffs  avg_steps  avg_loop_score  avg_verification_density
     agentbench                          sas                          sas 10          0.400      0.0           0.4                 6263.776        6552.295           2585.7      2308.5       0.0            0.0             0.0                3.4            8.8      0.500           0.500                    
     agentbench                  only_voting                  only_voting 10          0.100      0.0           0.1                14455.529       12526.664           5152.1      3744.0       0.0            0.0             0.0                6.8           15.6      0.150           0.500                    
     agentbench          fully_linked_debate          fully_linked_debate 10          0.000      0.0           0.0                13977.706       14528.514           5112.0      5086.5       0.0            0.0             0.0                5.0           15.0      0.000           0.533                    
     agentbench  orchestrator_tree_structure  orchestrator_tree_structure 10          0.000      0.0           0.0                22507.040       19203.064          10931.6      8933.0       0.0            8.4             7.2               12.2           42.0      0.100           0.314                    
     agentbench orchestrator_with_discussion orchestrator_with_discussion 10          0.000      0.0           0.0                23097.971       17295.347          12876.3      9498.0       0.0            6.5             6.5               10.7           33.8      0.117           0.346                    
     agentbench   orchestrator_no_discussion   orchestrator_no_discussion 10          0.000      0.0           0.0                24722.880       18023.433          13883.2      8440.0       0.0            7.5             7.5               12.5           37.5      0.142           0.320                    
     agentbench            group_chat_debate            group_chat_debate 10          0.000      0.0           0.0                33959.475       31942.133          17219.9     16549.0       0.0            3.4             0.4               11.8           36.3      0.247           0.430                    
     browsecomp                          sas                          sas 10          0.300      0.0           0.3                19545.488       22104.582         110334.1    132798.5       0.0            7.3             0.0                1.0           18.6      0.000           0.122                    
     browsecomp orchestrator_with_discussion orchestrator_with_discussion 10          0.200      0.0           0.2                61372.060       62208.608         122085.9    136776.5       0.0           29.4             7.1               14.4           85.6      0.098           0.174                    
     browsecomp                  only_voting                  only_voting 10          0.200      0.0           0.2                77793.843       72600.321         359765.3    313072.0       0.0           24.3             0.0                5.0           60.6      0.000           0.103                    
     browsecomp            group_chat_debate            group_chat_debate 10          0.100      0.0           0.1                35719.793       28947.203          22928.3     10579.0       0.0           15.8             0.4                7.8           51.6      0.027           0.208                    
     browsecomp  orchestrator_tree_structure  orchestrator_tree_structure 10          0.100      0.0           0.1                72371.317       75046.013         221128.8    171509.0       0.0           36.8             7.6               14.0          101.8      0.114           0.147                    
     browsecomp   orchestrator_no_discussion   orchestrator_no_discussion 10          0.100      0.0           0.1                67949.872       70154.131         296197.8    299114.5       0.0           28.2             5.0                8.0           71.4      0.000           0.114                    
     browsecomp          fully_linked_debate          fully_linked_debate 10          0.000      0.0           0.0                19744.801       17841.141           5510.0      4363.5       0.0            9.2             1.2                5.8           34.4      0.038           0.252                    
  finance_agent                          sas                          sas 10          0.112      0.0           0.1                31666.918       32260.491         118912.4     86059.5       0.0            4.9             0.0                1.0           13.8      0.000           0.183                    
  finance_agent   orchestrator_no_discussion   orchestrator_no_discussion 10          0.100      0.0           0.1               777615.803      349529.761         226034.8    201303.0       0.0           24.3             5.0                8.0           63.6      0.000           0.133                    
      plancraft  orchestrator_tree_structure  orchestrator_tree_structure 10          0.900      1.0           0.9                47475.100       54958.506          30347.2     35840.5       0.0           17.5            15.0               26.5           87.5      0.467           0.314                    
      plancraft                          sas                          sas 10          0.800      1.0           0.8                 6594.302        5382.524           3506.4      2886.0       0.0            0.0             0.0                4.0           10.0      0.442           0.500                    
      plancraft                  only_voting                  only_voting 10          0.800      1.0           0.8                39191.236       28935.844          22226.7     13902.0       0.0            0.0             0.0               19.4           40.8      0.517           0.500                    
      plancraft          fully_linked_debate          fully_linked_debate 10          0.800      1.0           0.8                44207.945       33419.843          24241.2     15747.0       0.0            0.0             0.0               18.2           48.0      0.508           0.533                    
      plancraft   orchestrator_no_discussion   orchestrator_no_discussion 10          0.800      1.0           0.8                47162.544       33314.928          32258.9     20410.0       0.0           15.0            15.0               26.0           75.0      0.449           0.320                    
      plancraft orchestrator_with_discussion orchestrator_with_discussion 10          0.800      1.0           0.8                85973.882       38925.476          71679.6     22485.5       0.0           22.5            22.5               39.5          117.0      0.496           0.346                    
      plancraft            group_chat_debate            group_chat_debate 10          0.700      1.0           0.7                48006.582       49864.647          26816.7     25401.5       0.0            4.4             0.0               16.6           50.6      0.467           0.435                    
        scicode                          sas                          sas 10          0.000      0.0           0.0                30358.575        7789.897          27926.8      3104.5       0.0            0.0             0.0               10.8           23.6      0.419           0.500                    
        scicode                  only_voting                  only_voting 10          0.000      0.0           0.0               119155.082       30802.559         111031.9     10626.0       0.0            0.0             0.0               34.4           70.8      0.419           0.500                    
        scicode          fully_linked_debate          fully_linked_debate 10          0.000      0.0           0.0               131733.937       35486.716         126918.3     15191.5       0.0            0.0             0.0               34.4           88.5      0.419           0.533                    
        scicode   orchestrator_no_discussion   orchestrator_no_discussion 10          0.000      0.0           0.0               161034.535       40631.924         162412.8     20108.5       0.0           29.5            29.5               52.1          147.5      0.419           0.320                    
        scicode orchestrator_with_discussion orchestrator_with_discussion 10          0.000      0.0           0.0               171106.113       42743.815         172515.3     23518.5       0.0           29.5            29.5               52.1          153.4      0.419           0.346                    
        scicode            group_chat_debate            group_chat_debate 10          0.000      0.0           0.0               201715.970       50949.755         204836.4     26287.0       0.0           11.8             0.0               46.2          135.7      0.419           0.435                    
        scicode  orchestrator_tree_structure  orchestrator_tree_structure 10          0.000      0.0           0.0               206921.516       51616.537         206362.9     22129.5       0.0           42.3            36.2               65.9          212.1      0.414           0.315                    
stabletoolbench            group_chat_debate            group_chat_debate 10          0.700      1.0           0.7                32201.081       31944.177          17512.6     16622.5       0.0           11.6             0.2                7.4           42.8      0.018           0.245                    
stabletoolbench  orchestrator_tree_structure  orchestrator_tree_structure 10          0.700      1.0           0.7                47504.621       45372.782          26826.4     26203.0       0.0           25.3             8.0               15.0           80.6      0.145           0.197                    
stabletoolbench          fully_linked_debate          fully_linked_debate 10          0.600      1.0           0.6                20669.749       21226.711           9421.1      9448.5       0.0            6.7             0.0                5.0           28.4      0.000           0.295                    
stabletoolbench                  only_voting                  only_voting 10          0.600      1.0           0.6                31591.274       28260.414          16926.3     11420.5       0.0           11.0             0.0                5.0           34.0      0.000           0.184                    
stabletoolbench                          sas                          sas 10          0.500      0.5           0.5                 9083.090        6862.532           4466.2      3303.5       0.0            2.9             0.0                1.0            9.8      0.000           0.213                    
stabletoolbench   orchestrator_no_discussion   orchestrator_no_discussion 10          0.400      0.0           0.4                32808.427       28658.161          18678.2     14811.0       0.0           15.0             5.3                8.8           46.2      0.021           0.190                    
stabletoolbench orchestrator_with_discussion orchestrator_with_discussion 10          0.300      0.0           0.3                41422.186       35228.499          23040.3     17431.5       0.0           16.7             5.6               10.7           54.7      0.053           0.215                    
      workbench          fully_linked_debate          fully_linked_debate 10          0.400      0.0           0.4                29078.633       25518.175           6971.2      6721.0       0.0            6.1             0.0                5.0           27.2      0.000           0.297                    
      workbench            group_chat_debate            group_chat_debate 10          0.400      0.0           0.4                49340.537       44795.652          17731.4     16248.0       0.0           14.0             1.0                9.0           49.6      0.066           0.229                    
      workbench  orchestrator_tree_structure  orchestrator_tree_structure 10          0.400      0.0           0.4                59096.653       48024.017          29430.6     21071.0       0.0           25.2             6.8               12.0           75.0      0.054           0.175                    
      workbench orchestrator_with_discussion orchestrator_with_discussion 10          0.300      0.0           0.3                57702.708       53663.702          23837.8     22639.5       0.0           21.1             5.6                9.7           60.9      0.016           0.171                    
      workbench                          sas                          sas 10          0.100      0.0           0.1                14825.932       14706.460           5477.2      5315.0       0.0            5.0             0.0                1.0           14.0      0.000           0.157                    
      workbench                  only_voting                  only_voting 10          0.100      0.0           0.1                58249.404       59728.599          19991.7     20554.0       0.0           19.6             0.0                5.0           51.2      0.000           0.118                    
      workbench   orchestrator_no_discussion   orchestrator_no_discussion 10          0.100      0.0           0.1                55005.765       52644.887          24541.0     23263.0       0.0           21.5             5.6                9.6           60.4      0.048           0.154                    
```

## Topology Delta vs SAS

```text
benchmark       system_label                  task_count  mean_score_delta_vs_sas  mean_success_delta_vs_sas  mean_latency_delta_ms_vs_sas  mean_tokens_delta_vs_sas  mean_communication_delta_vs_sas  mean_tool_calls_delta_vs_sas  score_wins_vs_sas  score_ties_vs_sas  score_losses_vs_sas
     agentbench                  only_voting 10          -0.300                   -0.3                         8191.753                       2566.4                  0.0                              0.0                          1                   5                 4                   
     agentbench          fully_linked_debate 10          -0.400                   -0.4                         7713.930                       2526.3                  0.0                              0.0                          0                   6                 4                   
     agentbench  orchestrator_tree_structure 10          -0.400                   -0.4                        16243.264                       8345.9                  7.2                              8.4                          0                   6                 4                   
     agentbench orchestrator_with_discussion 10          -0.400                   -0.4                        16834.195                      10290.6                  6.5                              6.5                          0                   6                 4                   
     agentbench   orchestrator_no_discussion 10          -0.400                   -0.4                        18459.104                      11297.5                  7.5                              7.5                          0                   6                 4                   
     agentbench            group_chat_debate 10          -0.400                   -0.4                        27695.699                      14634.2                  0.4                              3.4                          0                   6                 4                   
     browsecomp orchestrator_with_discussion 10          -0.100                   -0.1                        41826.572                      11751.8                  7.1                             22.1                          0                   9                 1                   
     browsecomp                  only_voting 10          -0.100                   -0.1                        58248.355                     249431.2                  0.0                             17.0                          0                   9                 1                   
     browsecomp            group_chat_debate 10          -0.200                   -0.2                        16174.305                     -87405.8                  0.4                              8.5                          0                   8                 2                   
     browsecomp  orchestrator_tree_structure 10          -0.200                   -0.2                        52825.830                     110794.7                  7.6                             29.5                          0                   8                 2                   
     browsecomp   orchestrator_no_discussion 10          -0.200                   -0.2                        48404.384                     185863.7                  5.0                             20.9                          0                   8                 2                   
     browsecomp          fully_linked_debate 10          -0.300                   -0.3                          199.313                    -104824.1                  1.2                              1.9                          0                   7                 3                   
  finance_agent   orchestrator_no_discussion 10          -0.012                    0.0                       745948.886                     107122.4                  5.0                             19.4                          0                   9                 1                   
      plancraft  orchestrator_tree_structure 10           0.100                    0.1                        40880.797                      26840.8                 15.0                             17.5                          2                   7                 1                   
      plancraft                  only_voting 10           0.000                    0.0                        32596.934                      18720.3                  0.0                              0.0                          0                  10                 0                   
      plancraft          fully_linked_debate 10           0.000                    0.0                        37613.642                      20734.8                  0.0                              0.0                          0                  10                 0                   
      plancraft   orchestrator_no_discussion 10           0.000                    0.0                        40568.241                      28752.5                 15.0                             15.0                          0                  10                 0                   
      plancraft orchestrator_with_discussion 10           0.000                    0.0                        79379.579                      68173.2                 22.5                             22.5                          0                  10                 0                   
      plancraft            group_chat_debate 10          -0.100                   -0.1                        41412.279                      23310.3                  0.0                              4.4                          1                   7                 2                   
        scicode                  only_voting 10           0.000                    0.0                        88796.506                      83105.1                  0.0                              0.0                          0                  10                 0                   
        scicode          fully_linked_debate 10           0.000                    0.0                       101375.361                      98991.5                  0.0                              0.0                          0                  10                 0                   
        scicode   orchestrator_no_discussion 10           0.000                    0.0                       130675.959                     134486.0                 29.5                             29.5                          0                  10                 0                   
        scicode orchestrator_with_discussion 10           0.000                    0.0                       140747.537                     144588.5                 29.5                             29.5                          0                  10                 0                   
        scicode            group_chat_debate 10           0.000                    0.0                       171357.395                     176909.6                  0.0                             11.8                          0                  10                 0                   
        scicode  orchestrator_tree_structure 10           0.000                    0.0                       176562.941                     178436.1                 36.2                             42.3                          0                  10                 0                   
stabletoolbench            group_chat_debate 10           0.200                    0.2                        23117.991                      13046.4                  0.2                              8.7                          2                   8                 0                   
stabletoolbench  orchestrator_tree_structure 10           0.200                    0.2                        38421.531                      22360.2                  8.0                             22.4                          2                   8                 0                   
stabletoolbench          fully_linked_debate 10           0.100                    0.1                        11586.659                       4954.9                  0.0                              3.8                          3                   5                 2                   
stabletoolbench                  only_voting 10           0.100                    0.1                        22508.183                      12460.1                  0.0                              8.1                          1                   9                 0                   
stabletoolbench   orchestrator_no_discussion 10          -0.100                   -0.1                        23725.337                      14212.0                  5.3                             12.1                          0                   9                 1                   
stabletoolbench orchestrator_with_discussion 10          -0.200                   -0.2                        32339.096                      18574.1                  5.6                             13.8                          0                   8                 2                   
      workbench          fully_linked_debate 10           0.300                    0.3                        14252.701                       1494.0                  0.0                              1.1                          3                   7                 0                   
      workbench            group_chat_debate 10           0.300                    0.3                        34514.605                      12254.2                  1.0                              9.0                          3                   7                 0                   
      workbench  orchestrator_tree_structure 10           0.300                    0.3                        44270.721                      23953.4                  6.8                             20.2                          3                   7                 0                   
      workbench orchestrator_with_discussion 10           0.200                    0.2                        42876.777                      18360.6                  5.6                             16.1                          2                   8                 0                   
      workbench                  only_voting 10           0.000                    0.0                        43423.472                      14514.5                  0.0                             14.6                          0                  10                 0                   
      workbench   orchestrator_no_discussion 10           0.000                    0.0                        40179.833                      19063.8                  5.6                             16.5                          0                  10                 0                   
```

## Plot Inventory

- `overall_avg_task_score`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/overall_avg_task_score.png`
- `agentbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/agentbench_task_score_boxplot.png`
- `agentbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/agentbench_latency_ms_boxplot.png`
- `agentbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/agentbench_tokens_total_boxplot.png`
- `agentbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/agentbench_communication_count_boxplot.png`
- `agentbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/agentbench_task_score_heatmap.png`
- `agentbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/agentbench_accuracy_vs_tokens.png`
- `browsecomp`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/browsecomp_task_score_boxplot.png`
- `browsecomp`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/browsecomp_latency_ms_boxplot.png`
- `browsecomp`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/browsecomp_tokens_total_boxplot.png`
- `browsecomp`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/browsecomp_communication_count_boxplot.png`
- `browsecomp`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/browsecomp_task_score_heatmap.png`
- `browsecomp`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/browsecomp_accuracy_vs_tokens.png`
- `finance_agent`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/finance_agent_task_score_boxplot.png`
- `finance_agent`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/finance_agent_latency_ms_boxplot.png`
- `finance_agent`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/finance_agent_tokens_total_boxplot.png`
- `finance_agent`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/finance_agent_communication_count_boxplot.png`
- `finance_agent`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/finance_agent_task_score_heatmap.png`
- `finance_agent`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/finance_agent_accuracy_vs_tokens.png`
- `plancraft`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/plancraft_task_score_boxplot.png`
- `plancraft`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/plancraft_latency_ms_boxplot.png`
- `plancraft`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/plancraft_tokens_total_boxplot.png`
- `plancraft`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/plancraft_communication_count_boxplot.png`
- `plancraft`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/plancraft_task_score_heatmap.png`
- `plancraft`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/plancraft_accuracy_vs_tokens.png`
- `scicode`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/scicode_task_score_boxplot.png`
- `scicode`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/scicode_latency_ms_boxplot.png`
- `scicode`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/scicode_tokens_total_boxplot.png`
- `scicode`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/scicode_communication_count_boxplot.png`
- `scicode`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/scicode_task_score_heatmap.png`
- `scicode`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/scicode_accuracy_vs_tokens.png`
- `stabletoolbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/stabletoolbench_task_score_boxplot.png`
- `stabletoolbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/stabletoolbench_latency_ms_boxplot.png`
- `stabletoolbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/stabletoolbench_tokens_total_boxplot.png`
- `stabletoolbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/stabletoolbench_communication_count_boxplot.png`
- `stabletoolbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/stabletoolbench_task_score_heatmap.png`
- `stabletoolbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/stabletoolbench_accuracy_vs_tokens.png`
- `workbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/workbench_task_score_boxplot.png`
- `workbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/workbench_latency_ms_boxplot.png`
- `workbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/workbench_tokens_total_boxplot.png`
- `workbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/workbench_communication_count_boxplot.png`
- `workbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/workbench_task_score_heatmap.png`
- `workbench`: `/Users/powerarena/Documents/GitHub/MAS_Analyzer/artifacts/full_experiment/all_benchmarks_10sample/analysis/workbench_accuracy_vs_tokens.png`

## Notes

- `C3_cost_total` is omitted from most figures when it is identically zero across the experiment.
- This experiment uses only two tasks per benchmark and one run per task, so boxplots are descriptive rather than inferential.
- `workbench` appears to complete runs structurally but still scores zero on task evaluation across all tested topologies.

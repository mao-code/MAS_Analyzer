# Topology Analysis Report

## Objectives

- `eval_avg_score`: `max`
- `Q1_success_rate`: `max`
- `C2_tokens_total`: `min`
- `C1_latency_p95`: `min`
- `P3_loop_score`: `min`

## Pareto Frontier

- `sas`

## Ranking (Ideal Point Distance)

- `1. sas` (d_ideal=0.7071, frontier=True)
- `2. orchestrator_no_discussion` (d_ideal=0.8149, frontier=False)
- `3. group_chat_debate` (d_ideal=0.9165, frontier=False)
- `4. orchestrator_tree_structure` (d_ideal=0.9176, frontier=False)
- `5. fully_linked_debate` (d_ideal=1.2268, frontier=False)
- `6. only_voting` (d_ideal=1.2834, frontier=False)
- `7. orchestrator_with_discussion` (d_ideal=1.8708, frontier=False)

## Generated Plots

- `pca_frontier_png`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/outputs/browsecomp_all_topologies/20260304T215653Z/topology_analysis/pca_frontier.png`
- `pareto_tradeoff_png`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/outputs/browsecomp_all_topologies/20260304T215653Z/topology_analysis/pareto_tradeoff.png`
- `mahalanobis_distance_png`: `/Users/maoxunhuang/Desktop/MAS_Analyzer/outputs/browsecomp_all_topologies/20260304T215653Z/topology_analysis/distance_mahalanobis.png`

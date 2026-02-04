# Trace-Derived Task Descriptors for MAS Architecture Selection

This repo implements an engineering-first research framework that turns agent interaction traces into a reproducible **Task Descriptor** vector, then uses that descriptor to support **Multi-Agent System (MAS) architecture selection** and explainable failure-boundary analysis.

## Why this project exists

Recent controlled studies show MAS gains are not stable: performance depends on task structure, topology, and model/system behavior. The missing piece is not "can MAS be better", but an operational way to answer:

1. For a given agentic task, when does MAS outperform a strong single-agent system?
2. If MAS helps, which topology is best (centralized, decentralized, hybrid, independent)?
3. When MAS becomes negative-return, what is the failure mechanism and boundary?

This repo focuses on the **Task-side**: we compute a portable, trace-derived descriptor `x(task)` as a standard input to architecture selection, rather than relying on ad-hoc heuristics.

## Key idea

For each task, we run a fixed **probe system P** a small number of times (typically 3 to 5), log structured traces, then compute a high-dimensional descriptor vector:

`x(task) = [xQ, xC, xR, xP]`

- `xQ`: success and quality signals
- `xC`: cost and efficiency signals
- `xR`: stability and reliability signals
- `xP`: process and structure signals

Because `x(task)` is derived from traces and repeated runs, it captures uncertainty and failure modes as first-class properties.

## Repository layout

- `benchmark/`
  - Task sets and benchmark runners.
  - Each benchmark provides tasks plus (optional) evaluation hooks (exact-match, unit tests, judge stubs).
- `MAS/`
  - Agent systems and MAS topologies.
  - Each system must emit trace events following the shared schema.
- `descriptor/`
  - Trace schema, trace IO, metric extraction, descriptor construction.
  - Robust scaling, Mahalanobis distance, Pareto frontier, ideal-point selection.
  - Optional: stage-level bottlenecks and 2D embeddings.
- `main.py`
  - CLI entrypoint. Choose benchmark, MAS candidates, probe system, number of runs, seeds, outputs.

All modules are connected through small interfaces so you can swap benchmarks and MAS implementations without touching descriptor code.

## Trace schema (stable contract)

Each run writes a JSONL trace: an ordered event sequence `e1..eT`. Each event includes:

- `timestamp_start`, `timestamp_end`
- `actor` (LLM role, tool, env, evaluator)
- `event_type`: `plan | act | tool_call | tool_result | verify | revise | finalize | error`
- `payload` (minimal summaries: tool name, error code, artifact hash)
- `token_in`, `token_out`, `latency_ms`, `cost_usd`
- `state_id` (optional but recommended for loop detection)

This schema is designed so metrics are fully recomputable from logs.

## Metrics (minimum viable set)

The descriptor implements the following trace-derived metrics (grouped by Q/C/R/P). :contentReference[oaicite:2]{index=2}

### Q: Success and Quality
- `Q1 success_rate`: successes / N
- `Q2 completion_rate`: produced final artifact / N
- `Q3 faithfulness`: evaluator hook (LLM-judge or rules), optional stub
- `Q4 context_relevancy`: evaluator hook (LLM-judge or IR metrics), optional stub

### C: Cost and Efficiency
- `C1 latency_p95`: p95 over run latencies
- `C2 tokens_total`: sum(token_in + token_out)
- `C3 cost_total`: sum(cost_usd)
- `C4 tool_calls_total`: count(tool_call)
- `C5 tool_error_rate`: tool_fail / tool_calls

### R: Stability and Reliability
- `R1 success_var`: Bernoulli variance across repeated runs (or bootstrap)
- `R2 latency_var`: variance or IQR across runs
- `R3 tokens_var`: variance or IQR across runs

### P: Process and Structure
- `P1 steps_total`: total events `T`
- `P2 backtrack_rate`: (#revise + #redo) / T
- `P3 loop_score`: repeated `state_id` ratio or repeated event-pattern ratio (defined in code)
- `P4 verification_density`: #verify / T

Optional extensions behind flags:
- `avg_branching`, `unique_tools`, `failure_mode_hist`, `executability_score`

## Selection utilities
This repo provides reusable utilities for architecture selection:

1. **Robust scaling**: `(x - median) / IQR` per dimension to reduce outlier sensitivity.
2. **Mahalanobis distance**: distance with correlation awareness across descriptor dimensions.
3. **Pareto frontier**: compute non-dominated candidate topologies under multi-objective trade-offs.
4. **Ideal point selection**: pick best on Pareto set by weighted distance to an ideal point:
   `d_ideal(x) = || W (x - x*) ||_2`

## Stage-level bottlenecks
Traces can be segmented into stages (plan/retrieve/act/verify/revise/finalize) and per-stage metrics computed to surface bottlenecks:
- verify is sparse → hallucination risk
- retrieve has tool failures → repeated retries and latency spikes
- revise dominates tokens → unstable planning/execution loop
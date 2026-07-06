# AFlow Official Alignment

Upstream reference:

- Repository: `FoundationAgents/AFlow`
- Inspected commit: `3f457218fc716093fe53f6df8a5d5e6379d66346`
- License: MIT, copyright FoundationAgents

## Preserved Structure

This adapter keeps the parts of AFlow that define the workflow optimization
methodology:

- round-based workflow workspace: `workflows/round_N/graph.py`
- prompt sidecar: `workflows/round_N/prompt.py`
- validation scores appended to `workflows/results.json`
- parent workflow selection from high-scoring rounds
- optimizer LLM response with `<modification>`, `<graph>`, and `<prompt>`
- per-round `experience.json`
- materialized `best_workflow/`
- operator roles matching AFlow's workflow search space:
  `Custom`, `AnswerGenerate`, `ScEnsemble`, `Review`, `Revise`, `Format`

## Required Adaptations

The upstream code imports `workspace.<dataset>.workflows...` and evaluates
against AFlow's bundled datasets. This repository needs the same optimization
shape on the local benchmark suite, so the adapter changes these boundaries:

- generated workflows subclass `OfficialWorkflowBase`
- operators call this repository's `OpenRouterLLMClient`
- benchmark execution uses `benchmark.run(...)`
- correctness uses `benchmark.evaluate(...).success`
- trace and cost fields are emitted as this repository's `TraceEvent` schema
- benchmark tools are passed into workflow operator calls when available

## Not Claimed

This is not a byte-for-byte execution of upstream `scripts/optimizer.py`; that
would require porting our benchmarks into the upstream `BaseBenchmark` API and
workspace import layout. It is an official-code-structure adapter intended to
preserve AFlow's workflow optimization methodology while running this
repository's benchmarks and metrics.

# Goal

Perform a qualitative failure-mode analysis of different Multi-Agent System (MAS) topologies across benchmarks. Identify:

1. Common/general failure modes across all MAS topologies.
2. Topology-specific failure modes.
3. Benchmark/task-specific failure modes.
4. Evidence-based explanations for WHY MAS fails in certain cases.
5. General design recommendations or solutions, such as dynamic MAS, LLM-as-tools, agent spawning, adaptive routing, or topology switching. This would be a novelty of this research project.

# Motivation

This analysis should be useful for both industry and academia. The goal is not only to describe where MAS fails, but to explain why it fails by connecting:

- quantitative benchmark results,
- qualitative trajectory evidence,
- agent communication patterns,
- tool-use behavior,
- discussion/debate dynamics,
- topology constraints,
- and final answer correctness.

The final report should synthesize these findings into actionable insights.

# Context

The experiment results are in:

- `artifacts/full_experiment/20260427T134706Z__google_gemma_4_31b_it_nitro`
- `artifacts/full_experiment/20260427T134706Z__openai_gpt_oss_120b`

These correspond to Gemma 4 and GPT-OSS experiments.

Use the trajectories, including:

- agent discussions,
- tool-use messages,
- communication patterns,
- intermediate reasoning traces,
- disagreement or consensus behavior,
- delegation patterns,
- topology behavior,
- and final outputs.

Ignore the FinanceAgent benchmark.

# Output

Create an interactive HTML report.

The report should be saved as a standalone HTML file, for example:

`mas_failure_analysis_report.html`

The web page should be minimal, clear, readable, and easy to navigate.

It should include:

- executive summary,
- methodology,
- quantitative overview,
- failure-mode taxonomy,
- topology-specific analysis,
- benchmark-specific analysis,
- cross-model comparison,
- representative trajectory case studies,
- WHY MAS fails,
- proposed general solution,
- limitations,
- and appendix or evidence table.

Use tables, collapsible sections, filters, or tabs where helpful.

# Working Instructions

You may spawn sub-agents to inspect different folders, benchmarks, models, or topology families.

# Constraints

- Ignore the FinanceAgent benchmark.
- Do not invent results that are not supported by files.
- Clearly distinguish evidence-backed findings from hypotheses.
- Prefer direct trajectory evidence over generic MAS theory.
- Keep the final report self-contained.
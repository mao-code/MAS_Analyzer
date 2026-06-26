# MASS Paper Alignment Audit

Paper: `arXiv:2502.02533`, "Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies"

This implementation is a paper-faithful reproduction scaffold, not the authors' original code.

## Aligned

- Uses the paper's three-stage structure:
  - Stage 1: block-level prompt warm-up
  - Stage 2: topology search
  - Stage 3: workflow-level prompt optimization
- Warms up the initial predictor as `a0*`.
- Optimizes each enabled block conditioned on the warmed predictor.
- Computes block influence as `E(ai*) / E(a0*)`.
- Converts influence scores into softmax selection probabilities.
- Uses rejection-style pruning with `u < p_ai`.
- Samples workflow candidates under an agent budget.
- Builds workflows with the paper's fixed construction order: `[summarize, reflect, debate, aggregate]`.
- Uses paper search dimensions by default:
  - `summarize`: `{0, 1, 2, 3, 4}`
  - `aggregate`: `{1, 3, 5, 7, 9}`
  - `reflect`: `{0, 1, 2, 3, 4}`
  - `debate`: `{0, 1, 2, 3, 4}`
  - `execute`: `{0, 1}`
- Uses the paper's minimum block definitions:
  - aggregate: `3 predictors + 1 aggregator`
  - reflect: `1 predictor + 1 reflector`
  - debate: `2 predictors + 1 debator`
  - execute: `1 predictor + 1 executor + 1 reflector`
- Stage 3 now returns the workflow-level prompt-optimized candidate by default, matching Algorithm 1.
- Adds a standalone paper-baseline runner for the manually specified App. B.2 baselines:
  - CoT
  - Self-Consistency SC@9 with temperature 0.8 and majority vote
  - Self-Refine with 5 reflection rounds
  - Multi-Agent Debate with 3 agents, 3 rounds, and 1 judge
  - ADAS with 30 search rounds and 3 validation evaluations per round
  - AFlow with 20 rounds, 5 validation runs per round, and k=3
- The paper-baseline runner defaults to `google/gemma-4-31b-it` and avoids the production `MAS/`
  runtime; it only uses repo benchmark loading and evaluation.

## Approximate

- `MIPROLikePromptOptimizer` approximates MIPRO's instruction + exemplar optimization, but it is not DSPy's real MIPRO optimizer.
- Prompt templates are generic approximations unless task-specific templates are supplied through `MASSConfig.prompt_templates`.
- Agent execution semantics are encoded in a lightweight standalone executor, not in the authors' runtime.
- The executor preserves observable block behavior, but internal prompting and message passing are not guaranteed to match the authors' hidden implementation.
- Existing repo benchmarks are not the paper's original benchmark suite, so reported scores are not comparable to the paper tables.

## Not Aligned Yet

- No exact author prompts or discovered best prompts.
- No exact MIPRO candidate proposal/evaluation loop.
- No validation repeated 3 or 5 times for ADAS/AFlow-style baselines.
- ADAS and AFlow are safe standalone reproductions. They preserve the paper's search budgets and
  conditioning pattern, but search fixed workflow families instead of executing arbitrary
  LLM-generated code.
- No exact paper datasets/splits unless separately configured.
- No real code execution tool integration for coding tasks; the `execute` block currently calls the model callback unless replaced.
- No task-family default search-space mapping for arbitrary custom benchmarks.

## Practical Conclusion

Use this as a close reproduction framework for running custom benchmarks in the MASS style.
Do not claim exact author-code parity or direct numerical comparability with the paper.

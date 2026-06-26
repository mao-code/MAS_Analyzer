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
- Tracks and rejects duplicate sampled workflows so Stage 2 searches distinct topologies when the
  pruned search space is large enough, matching the paper's "10 different topologies" setup.
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
- Executes the initial `a0` workflow as a single predictor without a spurious
  aggregator call.
- Expands MASS blocks into observable role-level calls:
  - repeated summarizer rounds pass summary context into later agents
  - debate uses at least two predictor answers and one debator pass per agent
  - reflect uses reflector feedback followed by predictor/refiner updates
  - execute produces execution feedback before reflector/refiner updates
  - execute can consume an external execution callback or benchmark public-test/tool metadata
    before falling back to a model-generated execute response
  - aggregate is only called when multiple candidate answers must be combined
- Stage 3 now returns the workflow-level prompt-optimized candidate by default, matching Algorithm 1.
- Core benchmark runner defaults to the paper setup:
  - model: `google/gemma-4-31b-it`
  - model temperature: `0.7`
  - max output tokens: `4096`
  - topology candidates: `10`
  - topology softmax temperature: `0.05`
  - validation repeats: `3`
- MIPRO-like optimizer exposes the paper's public prompt-search settings:
  - bootstrapped demos: `3`
  - instruction candidates: `10`
  - rounds per agent: `10`
- MIPRO-like optimizer now proposes and scores instruction candidates with a recorded
  per-round search trace instead of doing a single prompt rewrite.
- Existing-benchmark runner applies Table 2-style task-family search spaces:
  - math/discrete reasoning: `{Aggregate, Reflect, Debate}`
  - long-context: `{Summarize, Aggregate, Reflect, Debate}`
  - coding/tool/web: `{Aggregate, Reflect, Debate, Execute}`
- The core runner now uses a reproduction-local OpenRouter client and uses the repo only for
  benchmark loading/evaluation.
- Adds a standalone paper-baseline runner for the manually specified App. B.2 baselines:
  - CoT
  - Self-Consistency SC@9 with temperature 0.8 and majority vote
  - Self-Refine with 5 reflection rounds
  - Multi-Agent Debate with 3 agents, 3 rounds, and 1 judge
- The paper-baseline runner defaults to `google/gemma-4-31b-it` and avoids the production `MAS/`
  runtime; it only uses repo benchmark loading and evaluation.

## Approximate

- `MIPROLikePromptOptimizer` uses the paper's public MIPRO settings and records a candidate
  search loop, but its candidate scorer is a deterministic heuristic rather than DSPy's real
  model/evaluator-driven MIPRO objective.
- Prompt templates are generic approximations unless task-specific templates are supplied through `MASSConfig.prompt_templates`.
- Agent execution semantics are encoded in a lightweight standalone executor, not in the authors' runtime.
- The executor now matches the paper's public block composition more closely, but internal prompting,
  message formatting, and stopping rules are not guaranteed to match the authors' hidden implementation.
- Table 2 mapping is adapted to this repo's benchmark names, not the paper's original exact datasets.
- Existing repo benchmarks are not the paper's original benchmark suite, so reported scores are not comparable to the paper tables.

## Not Aligned Yet

- No exact author prompts or discovered best prompts.
- No exact DSPy MIPRO candidate proposal/evaluation loop.
- No ADAS or AFlow optimizer reproduction in this MASS branch; those are separate framework
  reproductions because they introduce their own search procedures.
- No exact paper datasets/splits unless separately configured.
- No built-in sandbox runner for arbitrary generated code; coding tasks can provide an
  `execution_callback`, and SciCode-style public-test metadata is surfaced as execution feedback.

## Practical Conclusion

Use this as a close reproduction framework for running custom benchmarks in the MASS style.
Do not claim exact author-code parity or direct numerical comparability with the paper.

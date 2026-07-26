# MASS Paper Section Audit

Paper: `2502.02533v2.pdf`, "Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies".

This audit compares the paper methodology section-by-section against this repo's MASS reproduction
path. It is scoped to running MASS as an adapted baseline on this repo's four target benchmarks:
BrowseComp, PlanCraft, StableToolBench, and WorkBench.

## Sec. 2.2 Workflow-Level Search Space Design

Paper requirement:
- Search space consists of agentic building blocks: Aggregate, Reflect, Debate, Summarize, and
  Tool-use.
- Summarize is task-specific for long-context settings.
- Tool-use is an insertion decision for coding/tool environments.

Current implementation:
- `SearchSpace` exposes `summarize`, `aggregate`, `reflect`, `debate`, and `execute`.
- `run_existing_benchmarks._resolve_search_space()` maps benchmark families to paper-style blocks:
  long-context uses Summarize/Aggregate/Reflect/Debate; coding and tool/web use
  Aggregate/Reflect/Debate/Execute.
- BrowseComp is mapped to long-context; PlanCraft is adapted with action-output prompts; StableToolBench
  and WorkBench are mapped to tool/web.

Status:
- Aligned in block coverage and adapted benchmark mapping.
- Not exact for paper datasets because these four benchmarks are not the paper's evaluation suite.

## Sec. 3 Stage 1: Block-Level Prompt Optimization

Paper requirement:
- Warm up initial predictor: `a0* <- O_D(a0)`.
- For every building block, optimize the minimum topology conditioned on `a0*`:
  `ai* <- O_D(ai | a0*)`.
- Store validation performance for influence estimation.

Current implementation:
- `MASSFramework._run_block_prompt_stage()` optimizes predictor first.
- Each enabled block is optimized with `base_prompts={"predictor": predictor_prompt}`.
- Minimum block workflow definitions follow Table 3:
  Aggregate = 3 predictors + 1 aggregator; Reflect = 1 predictor + 1 reflector;
  Debate = 2 predictors + 1 debator; Execute = predictor + executor + reflector.
- Candidate selection is validation-driven when benchmark evaluator is available.

Status:
- Methodologically aligned.
- Remaining approximation: prompt optimizer is MIPRO-like, not exact DSPy MIPROv2.

## Sec. 3 Stage 2: Workflow Topology Optimization

Paper requirement:
- Compute influence: `I_ai = E(ai*) / E(a0*)`.
- Convert influence to selection probability with `Softmax(I_a, t)`.
- Rejection-prune each search dimension using `u < p_ai`.
- Sample valid workflows under agent budget `N(a) < B`.
- Build workflow in predefined rule-based order: `[summarize, reflect, debate, aggregate]`.
- Evaluate `N` candidate workflows and select the best.

Current implementation:
- Influence and softmax are computed in `MASSFramework._run_block_prompt_stage()`.
- Rejection-style workflow sampling is implemented by `_sample_pruned_workflow()`.
- Duplicate topologies are rejected before evaluation.
- `SearchSpace.max_agent_budget` defaults to 10.
- `WorkflowSpec.order` defaults to `[summarize, reflect, debate, aggregate]`.
- `--candidates-per-stage` defaults to 10.

Status:
- Aligned at algorithm level.
- Slightly stricter than paper in one way: duplicate sampled topologies are skipped to ensure distinct
  candidates where possible.

## Sec. 3 Stage 3: Workflow-Level Prompt Optimization

Paper requirement:
- Treat the whole MAS design as an integrated entity.
- Jointly optimize prompts over all agents simultaneously, conditioned on the best topology:
  `W* <- O_D(Wc*)`.
- This stage adapts prompts to orchestration and inter-agent dependencies.

Current implementation:
- `MASSFramework._run_workflow_prompt_stage()` calls `optimize_workflow_prompts()` on the best Stage 2
  topology.
- `MIPROLikePromptOptimizer` now prepares prompt candidates for every prompt block, assembles complete
  workflow prompt sets, scores those sets with the validation evaluator, and selects the best full set.
- This replaced the earlier coordinate-ascent approximation.

Status:
- Closer to paper than before: workflow-level prompt selection is now joint at the candidate-set level.
- Remaining approximation: it does not run full DSPy MIPROv2 Bayesian/TPE search, and it does not
  enumerate the full Cartesian product of all per-agent prompt candidates.

## Experimental Setup

Paper requirement:
- Main models are Gemini 1.5 Pro/Flash, with 3 runs.
- MASS prompt optimizer is MIPRO.
- MIPRO settings: bootstrapped demos = 3, instruction candidates = 10, 10 rounds per agent.
- Topology optimization searches 10 different topologies.

Current implementation:
- Defaults expose instruction candidates = 10, prompt-search rounds = 10, bootstrapped demos = 3,
  topology candidates = 10, validation repeats = 3, final repeats = 3.
- User-requested model is `google/gemma-4-31b-it` through OpenRouter.
- `--max-tokens 0` omits max token cap from the runner; provider/model default applies.

Status:
- Search and repeat counts are aligned by default.
- Model differs by project choice; results are not numerically comparable to paper tables.

## Appendix B.1/B.3 Search Spaces and Construction Rules

Paper requirement:
- Task families map to search spaces:
  reasoning: Aggregate/Reflect/Debate;
  long-context: Summarize/Aggregate/Reflect/Debate;
  coding: Aggregate/Reflect/Debate/Executor.
- Table 3 dimensions:
  Summarize `{0,1,2,3,4}`, Aggregate `{1,3,5,7,9}`, Reflect `{0,1,2,3,4}`,
  Debate `{0,1,2,3,4}`, Execute `{0,1}`.

Current implementation:
- `SearchSpace` uses the same dimensions.
- BrowseComp uses long-context mapping.
- StableToolBench and WorkBench use tool/web mapping, which follows coding/tool-use search space.
- PlanCraft uses adapted action contract and can use the general Aggregate/Reflect/Debate mapping unless
  overridden.

Status:
- Aligned in dimensions.
- Adapted for new benchmark families.

## Appendix C.5 Prompt Optimizers

Paper requirement:
- MASS is prompt-optimizer agnostic.
- Paper integrates MIPRO because simultaneous instruction and exemplar optimization works better than
  instruction-only methods.

Current implementation:
- Optimizer interface is pluggable.
- `MIPROLikePromptOptimizer` jointly carries instruction, fixed I/O fields, exemplars, and validation
  scoring.
- `DSPyMIPROAdapter` remains a wrapper point but not a full installed DSPy integration.

Status:
- Aligned in architecture.
- Not exact in optimizer internals.

## Appendix D Prompt Templates

Paper requirement:
- Templates follow DSPy-style structure:
  Instruction, fixed Input/Output fields, and exemplars.
- Role templates include Predictor, Reflector, Refiner, Debator, Summarizer, and Executor where applicable.

Current implementation:
- `AgentPromptBundle` stores `system_instruction`, `input_fields`, `output_fields`,
  `output_contract`, and `exemplar`.
- Runtime renders "Follow the following format." with fixed fields.
- LLM proposal is constrained to rewrite only the Instruction.
- Templates are adapted to the four target benchmarks:
  BrowseComp QA, PlanCraft actions, StableToolBench tools, WorkBench tools/state changes.

Status:
- Aligned in template structure.
- Adapted rather than copied verbatim because the repo benchmarks differ from paper datasets.

## Appendix E Best Prompts Discovered

Paper observation:
- Best discovered prompts may include data summaries, demos, previous instructions, and proposed
  instructions from MIPRO.

Current implementation:
- Prompt proposal receives validation example previews, seed instruction, fixed I/O fields, output
  contract, and workflow payload.
- Candidate checkpoints store instruction and evaluation details.
- PlanCraft filters bad reference/match/mismatch proposals.

Status:
- Partially aligned.
- Remaining approximation: proposal prompt is not a verbatim DSPy/MIPRO meta-prompt and does not use a
  Bayesian search history beyond the recorded candidate trace.

## Practical Claim

Supported claim:
- "MASS-style adapted baseline on our benchmark suite, with paper-style search stages, search
  dimensions, benchmark-specific DSPy-like prompt contracts, validation-driven prompt/topology
  selection, and OpenRouter/Gemma execution."

Unsupported claim:
- "Exact author-code reproduction" or "numerically comparable reproduction of the paper tables."

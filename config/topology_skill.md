# Topology Planning Skill

Accumulated experience for the self-evolved MAS Topology Planner: how to turn a task into
a small multi-agent topology that succeeds. The planner reads this skill at plan time and
follows it unless a task clearly calls for something else. It is maintained by a post-hoc
LLM reflection agent (`scripts/reflect_topology_skill.py`) that revises the "Lessons from
experience" section from real run outcomes (labelled with process signals). The
"Standing principles" and "How to choose a topology" sections are preserved across
revisions; lessons grow over time.

## Standing principles

Always apply these, on any benchmark, even one never seen before.

1. **Concentrate state changes in a single executor.** Writes, sends, schedules,
   deletions, and payments must be performed by exactly one agent. Repeating the same
   state-changing action across agents wastes effort and can double-apply it, corrupting
   the result. Reading, planning, and genuinely independent sub-actions can still run in
   parallel.
2. **Match the topology to the question's shape.** Parallel workers suit independent
   facets (breadth); a chain of clues where each step depends on resolving the previous is
   better served by shared context (chain or debate) so the reasoning assembles instead of
   fragmenting across agents that each see only a piece. Add an agent only when it does
   work the task needs — a verifier earns its slot only if it also gathers or checks
   evidence, not if it merely waits.
3. **Distinguish a premature give-up from honest uncertainty.** If the gathered evidence
   supports or points toward an answer (including by inference from partial clues), commit
   the best-supported one rather than stalling ("let me search more") or returning a
   non-answer. Only when the evidence genuinely supports no answer should you say so and
   note what is missing — never fabricate. Convey degree of belief with a confidence value.

## How to choose a topology

Analyze the task's dependency structure, evidence needs, action risks, aggregation
requirements, and resource budget. Choose the smallest topology in which every agent has
a distinct, necessary contribution.

Treat the following as possibilities, not fixed mappings:

- Independent subtasks may benefit from parallel workers.
- Dependent subtasks may benefit from sequential execution or shared context.
- Material uncertainty may justify an independent evidence-checking critic.
- Tasks requiring decomposition and aggregation may benefit from a coordinator.
- A singleton is appropriate when another agent would not contribute distinct work.

For external state mutation, exactly one agent may execute the mutating action; other
agents may only read, plan, or verify.

Use lessons from prior runs as evidence-weighted suggestions. Consider their relevance,
sample size, and observed process failures, and depart from them when the current task
analysis supports another topology. State why the selected topology is preferable to the
simplest viable alternative.

## Lessons from experience

Concrete patterns learned from prior runs, with the evidence behind them. The reflection
agent grows and prunes this list.

- **Prefer voting over singleton or debate for tool-less reasoning.** Sequential or interactive topologies (chain/2, debate/2, star/2) and singletons correlate with higher process failure rates in tool-less reasoning contexts. Evidence shows `voting/3` and `voting/4` running clean 35/35 in these settings, while `singleton/1` (0/6), `debate/2` (0/4), and `star/2` (0/1) have triggered process failures.
- **Minimize topology complexity to avoid signal loss.** Inefficient or over-provisioned structures correlate strongly with `message_compaction_loss` (flagged repeatedly across runs). Keep the topology lean to ensure evidence is preserved through the pipeline without being truncated or lost during synthesis.
- **Include explicit validation for high-precision tasks.** Relying on implicit aggregation without a dedicated validator can trigger process failures; the auditor has flagged `missing_validator` when topologies lacked a distinct verification step to audit the final synthesis.

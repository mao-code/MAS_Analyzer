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

First analyze the task — its **type** (retrieval/search, reasoning, coding, tool use,
state mutation, verification, planning, comparison, summarization), its **attributes**
(ambiguity, need for breadth/debate/verification, hallucination risk, whether external
state is mutated, whether outputs aggregate), and its **failure risks** (duplicate writes,
thin search coverage, premature consensus, weak verification, poor decomposition). Then map
the analysis to a topology:

- **Broad retrieval / search** — provision several searcher workers (a star with workers,
  or voting) that each search a *different* facet or query; do not rely on a single
  searcher. If the clues form a dependent chain (each step needs the previous answer), use
  a chain or debate so the reasoning assembles in shared context.
- **Factuality / high hallucination risk** — include a verifier or critic, but only one
  that re-checks evidence (a debate, or a critic that re-derives the answer), never a
  passive agent that merely waits.
- **External state mutation** (create / update / delete / send / schedule / pay) — exactly
  one agent executes the mutating tool. Prefer a singleton; avoid chains or parallel
  workers to minimize the risk of duplicate execution. Reading and planning may still
  parallelize.
- **Ambiguous reasoning** with several defensible answers — debate or voting to surface and
  resolve disagreement.
- **Complex, multi-part tasks** — a star (coordinator + workers), or expansions (a tree)
  when sub-parts themselves decompose.
- **Specialized subtasks needing central control** — a star coordinator with
  role-specialized workers.
- **Simple single-step lookup or transform** — a singleton.

Prefer the smallest topology that covers the work — extra agents cost tokens and can
conflict — but do provision enough agents to cover the task (e.g. several searchers for
broad retrieval).

## Lessons from experience

Concrete patterns learned from prior runs, with the evidence behind them. The reflection
agent grows and prunes this list.

- **Tool-heavy medium-complexity tasks are highly unstable in star topologies.** Increasing agent count in a star does not improve stability and often degrades it. Evidence from medium tool-using tasks shows `star/3` (0/2), `star/4` (0/5), and `star/5` (0/1) all failed to run clean. Avoid expanding star size for tool-heavy tasks; prioritize the leanest possible configuration to minimize coordination overhead.
- **Deep coordination paths and wider stars trigger structural collapse.** High incidences of `message_compaction_loss` (7x) and `branch_collapse` (3x) indicate that moving evidence through multi-agent hierarchies strips critical data before it reaches the final synthesis step. This is particularly prevalent in larger star configurations. Keep interaction paths shallow and direct to avoid `evidence_lost_before_synthesis`.
- **Collaborative topologies are susceptible to premature consensus.** Recurring `premature_consensus` flags suggest that agents in multi-agent setups may agree on a result too quickly without sufficient evidentiary grounding. Ensure workers are strictly isolated by query/facet to prevent echoing.
- **Tool-using state-mutation tasks are prone to cascades.** Evidence of `tool_error_cascade` identifies a high risk of process failure when tools modify state. Prioritize a singleton executor to minimize these failure modes and avoid `duplicate_state_mutation`.
- **Evidence-heavy tasks frequently suffer from coverage gaps.** Persistent flags for `insufficient_search_coverage` demonstrate that the system often settles on thin results. A validator must be an active agent that re-derives or audits evidence rather than a passive observer.

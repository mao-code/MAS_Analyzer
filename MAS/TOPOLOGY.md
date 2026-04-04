# MAS Topology Execution Guide

This file explains how each MAS topology in this repo actually runs in code:

- which workflow nodes execute
- what messages each agent can see
- how packets are emitted between stages
- where termination is checked
- what happens before the final answer is selected

The implementation lives primarily in `MAS/langgraph_engine.py`, with shared state in `MAS/state.py`, shared artifacts in `MAS/artifacts.py`, and topology layouts in `MAS/relay.py`.

## One shared execution model

All topologies use the same basic runtime contract.

### 1. Agents do not pass raw chat transcripts

Each agent step produces a structured artifact with fields such as:

- `answer`
- `summary`
- `critique`
- `revision_request`
- `confidence`
- `unresolved_issues`
- `evidence_summary`

When one stage sends information to another stage, it sends a bounded relay packet derived from that artifact, not the full transcript.

### 2. Message visibility is explicit

An agent only sees packets selected by the stage's `message_selector(...)`:

- by recipient
- by packet kind
- by `round_index`
- by `discussion_index`
- usually with `latest_only=True`

So visibility is controlled in code, not left ambiguous in prompts.

### 3. Dispatch and work are separated

Most multi-agent topologies alternate between:

- a dispatch/controller node that chooses active agents and emits packets
- one or more parallel worker nodes that produce artifacts
- a controller node that decides whether to continue or stop

### 4. Termination is centralized

Agents do not decide when the topology stops. Controller nodes do.

The controller calls `_termination_decision(...)` in `MAS/langgraph_engine.py`, which checks:

- `invalid_or_failed_branch`
- `consensus_reached`
- `no_meaningful_change`
- `max_rounds_reached`

Those checks are workflow-control logic, not benchmark evaluation.

### 5. Consensus is now LLM-judged by default

The default config is:

- `mas.termination_consensus_mode = "llm_judge"`

That means the controller sends the current candidate answers to a judge model, which groups semantically equivalent answers into clusters. Then:

- `consensus_ratio = size_of_largest_cluster / number_of_valid_answers`

Fallbacks:

- if `termination_consensus_mode = "lexical"`, normalized string matching is used
- if `llm_judge` is configured but unavailable or unparsable, the controller falls back to lexical consensus

### 5a. Unified semantic termination judge

In `llm_judge` mode the controller now uses one termination-judge call to assess both:

1. semantic agreement across the current answers
2. whether another round is likely to materially improve correctness

The judge returns:

- semantic groups
- invalid indices
- `is_substantive`
- `progress_status`
- `expected_improvement`
- `should_stop_for_no_progress`

Consensus-based stopping is blocked when the judge says the majority answer is not substantive. Planning-only updates, "need more information", and task restatements are treated as non-substantive.

If the judge is unavailable or returns unusable JSON, the controller falls back to lexical consensus plus lexical `SequenceMatcher` delta for `no_meaningful_change`.

### 6. Final answer selection is separate from termination

Stopping a loop and selecting a final answer are different steps.

The default config is now:

- `mas.final_vote_mode = "llm_judge"`

That means the final voter/judge can call a judge model to group materially equivalent answers and choose the best final answer. If that judge is unavailable, running in mock mode, or returns unusable JSON, the system falls back to deterministic artifact voting.

Examples:

- `fully_linked_debate` stops in `debate_controller`, then chooses the final answer in `judge`
- `group_chat_debate` stops local debate in `group_controller`, stops representative debate in `representative_controller`, then chooses the final answer in `final_judge`
- orchestrator-style topologies usually stop after an orchestrator/root aggregate already exists, so `finalize` uses that aggregate directly

## Shared counters

Two counters matter:

- `round_index`: outer collaboration cycle
- `discussion_index`: inner revision/debate cycle inside a fixed outer round

Typical usage:

- `fully_linked_debate`: uses `round_index`
- `group_chat_debate`: local group debate uses `round_index`, representative debate uses `discussion_index`
- `orchestrator_with_discussion`: outer orchestration uses `round_index`, mediated specialist revision uses `discussion_index`

## Topology table

| Topology | Main communication pattern | Final answer source |
|---|---|---|
| `sas` | No communication | single agent artifact |
| `only_voting` | No communication between workers | configurable voter over worker artifacts, default LLM judge |
| `orchestrator_no_discussion` | orchestrator -> specialists -> orchestrator | latest orchestrator merge |
| `orchestrator_with_discussion` | orchestrator-mediated specialist summaries | latest orchestrator merge |
| `orchestrator_tree_structure` | root -> managers -> leaves -> managers -> root | latest root reducer artifact |
| `fully_linked_debate` | all-to-all bounded peer summaries | configurable judge over latest debate artifacts, default LLM judge |
| `group_chat_debate` | intra-group debate, then representative-only exchange | configurable final judge over representative artifacts, default LLM judge |

## 1. SAS

### Node order

`START -> single_agent -> descriptor_monitor -> finalize -> END`

### Exact workflow

1. `single_agent` runs once.
2. It sees no inter-agent packets.
3. It produces one artifact.
4. `finalize` uses that artifact as the final answer.

### Message flow

- none

### Termination

- no loop
- always single-turn

## 2. Only Voting

### Node order

`START -> dispatch_independent_agents -> worker -> voter -> descriptor_monitor -> finalize -> END`

### Exact workflow

1. `dispatch_independent_agents` activates all workers.
2. `worker` runs once per agent in parallel.
3. Each worker sees no peer packets at all.
4. Each worker produces its own artifact independently.
5. `voter` collects the latest worker artifacts and selects one answer.

### Message flow

- no relay packets are sent between workers
- this topology is intentionally communication-free

### Final aggregation

`voter` uses `mas.final_vote_mode`:

- default: `llm_judge`
- fallback or explicit deterministic mode: `vote_artifacts(...)`

Deterministic tie-break order:

- higher vote count
- tie-break 1: higher mean confidence
- tie-break 2: lexicographically smaller canonical answer

### Termination

- no iterative controller
- always single-turn

## 3. Orchestrator No Discussion

### Node order

`START -> orchestrator_plan -> dispatch_specialists -> specialist_worker -> relay_specialist_reports -> orchestrator_merge -> descriptor_monitor -> termination_checker -> dispatch_specialists|finalize`

### Exact workflow

1. `orchestrator_plan` creates the initial orchestrator artifact.
2. `dispatch_specialists` converts that artifact into one bounded packet per specialist.
3. Each `specialist_worker` runs in parallel.
4. Each specialist only sees its own orchestrator packet:
   - round 0: `task_package`
   - later rounds: `orchestrator_feedback`
5. `relay_specialist_reports` converts each specialist artifact into a bounded `specialist_report` packet back to the orchestrator.
6. `orchestrator_merge` reads the latest `specialist_report` packets and synthesizes one orchestrator artifact.
7. `termination_checker` decides whether to launch another outer round.

### Message flow

- orchestrator -> each specialist: `task_package` or `orchestrator_feedback`
- specialist -> orchestrator: `specialist_report`
- specialists never see peer specialist outputs

### What each specialist can see

- only orchestrator-origin packets addressed to itself
- never direct peer messages
- never raw peer transcripts

### Termination

`termination_checker` uses:

- `candidate_artifacts = latest orchestrator_merge artifact`
- `consensus_artifacts = latest specialist_worker artifacts`
- `previous_candidate_artifacts = previous orchestrator_merge artifact`

So:

- consensus is measured over specialist outputs
- no-change is measured over orchestrator aggregate outputs
- `rounds` bounds the number of outer orchestrator cycles

## 4. Orchestrator With Discussion

### Node order

`START -> orchestrator_plan -> dispatch_specialists -> specialists_initial_round -> relay_specialist_reports -> orchestrator_relay -> dispatch_revision_round|orchestrator_merge`

Then if discussion continues:

`dispatch_revision_round -> specialists_revision_round -> relay_specialist_reports -> orchestrator_relay`

When discussion stops:

`orchestrator_merge -> descriptor_monitor -> cycle_termination_checker -> dispatch_specialists|finalize`

### Exact workflow

1. `orchestrator_plan` creates initial task packages.
2. `dispatch_specialists` sends bounded orchestrator packets to specialists.
3. `specialists_initial_round` runs in parallel.
4. `relay_specialist_reports` sends bounded specialist reports back to the orchestrator.
5. `orchestrator_relay` examines the latest specialist artifacts and decides whether mediated discussion should continue.
6. If yes, the orchestrator emits one `peer_summary` packet per specialist.
7. `dispatch_revision_round` activates all specialists again.
8. `specialists_revision_round` runs in parallel. Each specialist sees only the orchestrator-generated peer summary bundle for that revision step.
9. Steps 4 to 8 repeat until `orchestrator_relay` stops the inner discussion loop.
10. `orchestrator_merge` synthesizes the latest specialist reports into one outer-cycle orchestrator artifact.
11. `cycle_termination_checker` decides whether to begin another outer orchestrator cycle.

### Message flow

- orchestrator -> specialist: `task_package` or `orchestrator_feedback`
- specialist -> orchestrator: `specialist_report`
- orchestrator -> specialist: `peer_summary`

Important:

- specialists never see raw peer transcripts
- specialists never send directly to each other
- all peer exposure is mediated by the orchestrator

### What each specialist can see

Initial round:

- only its own orchestrator task package

Revision round:

- only its own orchestrator-generated `peer_summary` packet
- that packet contains bounded summaries of peer artifacts, not full messages

### Termination

There are two stop points.

Inner mediated discussion:

- controller: `orchestrator_relay`
- bounded by `discussion_rounds`
- consensus measured over latest specialist artifacts in the current outer round
- no-change measured between specialist revisions in the current outer round

Outer orchestration cycle:

- controller: `cycle_termination_checker`
- bounded by `rounds`
- consensus measured over latest specialist artifacts
- no-change measured between orchestrator merge artifacts

## 5. Orchestrator Tree Structure

### Node order

`START -> root_plan -> manager_dispatch -> manager_nodes -> worker_dispatch -> worker_nodes -> worker_relay -> manager_reducers -> manager_relay -> root_reducer -> descriptor_monitor -> termination_checker -> manager_dispatch|finalize`

### Exact workflow

1. `root_plan` creates a root artifact.
2. `manager_dispatch` sends one `root_task_package` from root to each manager.
3. `manager_nodes` refine the root package into manager-local plans.
4. `worker_dispatch` sends one `manager_task_package` from each manager to its own children.
5. `worker_nodes` run at the leaves.
6. `worker_relay` converts each leaf artifact into a bounded `child_report` for its parent manager.
7. `manager_reducers` aggregate only direct child reports.
8. `manager_relay` sends one bounded `manager_report` per manager to the root.
9. `root_reducer` aggregates manager reports into the root artifact.
10. `termination_checker` decides whether to do another top-down/bottom-up pass.

### Message flow

- root -> managers: `root_task_package`
- manager -> own children only: `manager_task_package`
- leaf -> direct parent manager: `child_report`
- manager -> root: `manager_report`

Forbidden by construction:

- sibling-to-sibling messaging
- manager-to-manager messaging
- leaf-to-leaf messaging
- arbitrary backflow outside the parent-child DAG

### What each node can see

- managers only see `root_task_package` when planning downward, then `child_report` when aggregating upward
- leaves only see their own parent manager package
- root only sees `manager_report` packets during aggregation

### Termination

`termination_checker` uses:

- `candidate_artifacts = latest root_reducer artifact`
- `consensus_artifacts = latest manager_reducers artifacts`
- `previous_candidate_artifacts = previous root_reducer artifact`

So:

- consensus is checked over manager-level aggregate outputs
- no-change is checked over root aggregate outputs
- `rounds` bounds the number of full tree passes

## 6. Fully Linked Debate

This is the topology you asked about most directly.

### Node order

`START -> debate_init -> debate_controller -> debate_dispatch -> debate_round -> debate_controller -> debate_dispatch|judge -> descriptor_monitor -> finalize`

### Exact workflow

The code does not have a separate "initial thought" node. The first `debate_round` itself produces the initial individual answers.

Step by step:

1. `debate_init` initializes the debate process.
2. `debate_controller` runs before any debater artifacts exist.
3. Because no debate artifacts exist yet, the controller does not stop. It simply sets `next_step = debate_dispatch`.
4. `debate_dispatch` activates all agents. On this first pass it emits no peer-summary packets, because there is no previous debate artifact yet.
5. `debate_round` runs once per debater in parallel.
6. In round 0, each debater sees no peer debate packets, so it produces its own initial artifact from the task alone.
7. Control returns to `debate_controller`.
8. Now the controller has one latest artifact per debater. It checks termination.
9. If the controller decides to continue, it converts each debater artifact into a bounded packet and broadcasts it to all other debaters for the next round.
10. `debate_dispatch` activates all debaters again.
11. `debate_round` runs again. In this round, each debater sees the latest `debate_round` packets sent to it by all peers, then produces a revised artifact.
12. Steps 7 to 11 repeat until the controller stops.
13. When the controller stops, control goes to `judge`.
14. `judge` deterministically votes over the latest `debate_round` artifacts and selects the final answer.

### Message flow

Round 0:

- no peer packets
- each agent writes its own initial artifact

Round k > 0:

- for each debater artifact from round `k - 1`, the controller emits one bounded packet to all other debaters
- packet kind: `debate_round`
- each debater sees only the latest peer packets addressed to it for the current round

So yes: the behavior is effectively:

1. each agent produces an initial thought
2. the controller aggregates those into bounded peer-visible packets
3. each agent produces a revised thought
4. the controller checks whether another debate round is needed

But the implementation detail is important:

- the controller does not create one single merged global message
- it emits one packet per sender artifact
- each recipient sees a set of bounded peer packets, one from each other debater

### What each debater can see

In `debate_round`, agent `A` sees:

- all packets of kind `debate_round`
- whose recipient is `A`
- whose `round_index` matches the current round
- typically the latest packet from each peer

It does not see:

- its own packet as an inbound peer message
- raw peer transcripts
- unrelated packets from earlier rounds once `latest_only=True` selection is applied

### Termination

`debate_controller` uses:

- `candidate_artifacts = latest debate_round artifacts from current round`
- `previous_candidate_artifacts = latest debate_round artifacts from previous round`
- `consensus_artifacts = current debate_round artifacts`
- `expected_count = number of debaters`

Then it checks, in order:

1. too few valid artifacts survived
2. consensus ratio `>= 0.75` and the semantic judge says the answer is substantive
3. semantic no-progress on another round, or lexical delta `<= 0.05` in fallback mode
4. max rounds reached

If stop:

- `next_step = judge`

If continue:

- emit peer packets for next round
- increment `round_index`
- `next_step = debate_dispatch`

### Final answer selection

`judge` uses `mas.final_vote_mode` on the latest debate artifacts.

- default: `llm_judge`
- fallback or explicit deterministic mode: `vote_artifacts(...)`

That final judge is separate from the LLM judge used by termination consensus, even though both can use a judge model.

## 7. Group Chat Debate

### Node order

`START -> group_dispatch -> group_debate_round -> group_controller -> group_dispatch|representative_dispatch`

Then representative phase:

`representative_dispatch -> representative_merge -> representative_controller -> representative_dispatch|final_judge -> descriptor_monitor -> finalize`

### Exact workflow

Local group phase:

1. `group_dispatch` activates all agents.
2. `group_debate_round` runs once per group member in parallel.
3. In round 0, there are no prior group packets, so each member produces its initial local artifact.
4. `group_controller` checks whether local debate should continue.
5. If local debate continues, the controller emits one bounded packet per member artifact to the other members of the same group only.
6. `round_index` increments and another local group debate round begins.
7. If local debate stops, the controller emits one `group_summary` packet to each group's representative.

Representative phase:

8. `representative_dispatch` activates only the representatives.
9. `representative_merge` runs once per representative.
10. Each representative sees:
   - its own `group_summary`
   - any `representative_debate_round` packets sent by peer representatives in earlier representative discussion steps
11. `representative_controller` checks whether representative debate should continue.
12. If yes, it emits representative-only bounded peer packets and increments `discussion_index`.
13. If stop, control goes to `final_judge`.
14. `final_judge` applies `mas.final_vote_mode` over the latest representative artifacts.

### Message flow

Local phase:

- member -> other members in same group: `group_debate_round`

Promotion to representative phase:

- system -> each representative: `group_summary`

Representative phase:

- representative -> other representatives: `representative_debate_round`

No direct inter-group member traffic exists before representatives are selected.

### What each agent can see

Ordinary group member:

- only same-group packets
- never other groups' packets

Representative:

- own `group_summary`
- peer representative packets
- no raw member transcripts from other groups

### Termination

There are two controllers.

`group_controller`:

- checks local group artifacts
- bounded by `rounds`
- if stop, promotes group summaries upward

`representative_controller`:

- checks representative artifacts
- bounded by `discussion_rounds`
- if stop, routes to `final_judge`

## Workflow artifacts

Each system directory can now contain two graph views:

- `mas_graph.*`: communication topology between agents
- `workflow_graph.*`: actual workflow/control-flow nodes such as controllers, dispatchers, judges, and finalization

Use both together:

- `mas_graph` tells you who can communicate with whom
- `workflow_graph` tells you when each stage runs and where stopping happens

## Implementation pointers

- workflow builders: `MAS/langgraph_engine.py`
- message packet builders: `MAS/langgraph_engine.py`
- termination logic: `MAS/langgraph_engine.py`
- topology layouts: `MAS/relay.py`
- shared state: `MAS/state.py`
- shared artifact schema: `MAS/artifacts.py`

---

## 8. Dynamic Domain-Specific Role Assignment

### Motivation

The structural roles described above (orchestrator, specialist, debater, manager, leaf_worker, etc.) define an agent's *position* in the communication graph. They control routing, message visibility, and workflow transitions. However, they carry no domain expertise — a "specialist" in a financial QA task behaves identically to a "specialist" in a web retrieval task.

With 8 benchmarks spanning very different domains (web retrieval, finance, Minecraft crafting, scientific code generation, API orchestration, workplace tools, e-commerce, and cross-domain OS/DB/KG tasks), generic roles leave significant performance on the table. Agents benefit from domain-specific personas that guide their reasoning toward the expertise areas most relevant to the task.

This system implements dynamic, benchmark-aware role assignment inspired by the communication-aware agent design in *Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems* ([arXiv 2410.02506](https://arxiv.org/pdf/2410.02506)).

### Two-layer role architecture

Each agent now carries two orthogonal role labels:

| Layer | Source | Purpose | Example |
|---|---|---|---|
| **Structural role** | `TopologyLayout.roles` | Controls routing, message visibility, workflow transitions | `orchestrator`, `specialist`, `debater` |
| **Domain role + persona** | `WorkflowState.domain_personas` | Shapes the agent's reasoning, expertise, and behavioral style | `Financial Analyst`, `Web Search Strategist` |

Both are injected into the agent's system prompt. The structural role is never replaced — it is augmented with the domain persona.

### Role assignment workflow

```
┌──────────────────────────────────────────────────────────────┐
│  1. Load benchmark role pool                                  │
│     get_role_pool(benchmark_name) → list[DomainRole]         │
│                                                               │
│  2. Build assignment prompt                                   │
│     - Describe topology structure (agents, hierarchy, links) │
│     - List available domain roles with personas              │
│     - Include task preview                                    │
│                                                               │
│  3. Send to LLM                                               │
│     LLM returns JSON: {agent_id: role_name, ...}            │
│                                                               │
│  4. Parse and validate                                        │
│     - Match role names to pool (case-insensitive)            │
│     - Verify all agents are assigned                         │
│     - On failure → deterministic round-robin fallback        │
│                                                               │
│  5. Inject into WorkflowState                                 │
│     state["domain_personas"][agent_id] = {                   │
│       "role_name": "...", "persona": "..."                   │
│     }                                                         │
│                                                               │
│  6. Each _build_agent_prompt() call reads domain_personas    │
│     and appends Domain Role + Persona to the system message  │
└──────────────────────────────────────────────────────────────┘
```

This happens once before the main workflow graph executes (inside `LangGraphMASEngine.run()`), so it adds a single LLM call overhead per task.

### LLM assignment prompt template

The assigner LLM receives:

```
System: You are a multi-agent system architect. Your task is to assign
domain-specific roles to agents in a multi-agent topology.

User:
Assign domain-specific roles to the agents in a **{topology}** topology
for the **{benchmark_name}** benchmark.

## Topology Structure
- Topology type: {topology}
- Number of agents: {N}
- Agents (id, structural role, hierarchy level):
  - agent_0: structural_role=orchestrator, level=0
  - agent_1: structural_role=specialist, level=1
  ...
- Communication links:
  - agent_0 -> agent_1, agent_2
  ...

## Task Preview
{task_prompt_preview (max 600 chars)}

## Available Domain Roles
  1. **Role Name** — persona description
  2. ...

## Instructions
Assign exactly one domain role to each agent. Consider:
1. The agent's structural position ...
2. The specific task requirements ...
3. Diversity ...
4. For hierarchical topologies: higher=broader, lower=focused
5. You may assign the same role to multiple agents if needed.

Return a JSON object: {"agent_0": "Role Name A", ...}
Return ONLY the JSON object, no other text.
```

### Benchmark role pools

The following role pools are defined in `MAS/role_pools.py`. Each benchmark has 5–6 curated domain roles.

#### browsecomp — Open-ended web information retrieval

| Role | Persona |
|---|---|
| Web Search Strategist | Expert at formulating effective search queries. Decomposes complex questions into targeted sub-queries, considers alternate phrasings and synonyms, identifies the most promising search angles. Prioritizes precision over breadth. |
| Information Analyst | Specializes in analyzing and synthesizing information from multiple retrieved documents. Identifies relevant passages, cross-references claims across sources, resolves contradictions, extracts precise factual answers from noisy retrieval results. |
| Fact Verifier | Meticulous fact-checker. Verifies candidate answers against available evidence, checks for consistency with known facts, identifies unsupported claims, flags speculation. |
| Document Navigator | Excels at understanding document structure and relevance. Quickly assesses document utility, identifies informative sections, determines when to fetch more documents vs. use existing evidence. |
| Answer Synthesizer | Expert at producing concise, exact-match answers from complex evidence. Distills lengthy analyses into the precise format required, ensures specificity (names, numbers, dates). |
| Query Decomposer | Specializes in breaking complex multi-hop questions into simpler, answerable sub-questions. Identifies logical dependencies and optimal investigation order. |

#### finance_agent — Financial question answering (EDGAR / web search)

| Role | Persona |
|---|---|
| Financial Data Retriever | Expert at navigating financial data sources, particularly SEC EDGAR filings. Locates 10-K, 10-Q, 8-K, proxy statements. Extracts relevant tables and identifies correct reporting periods. |
| Financial Analyst | Specializes in interpreting financial statements, computing ratios, analyzing trends. Reads balance sheets, income statements, and cash flow statements to derive answers. |
| Regulatory Knowledge Expert | Deep knowledge of financial regulations, SEC requirements, GAAP/IFRS, corporate governance. Provides context on reporting implications. |
| Quantitative Reasoner | Excels at precise financial calculations: growth rates, margin analysis, YoY comparisons, weighted averages, unit conversions. Double-checks arithmetic. |
| Market Context Analyst | Understands broader market context, industry dynamics, macroeconomic factors. Distinguishes company-specific from sector-wide trends. |
| Answer Quality Auditor | Reviews financial answers for accuracy, completeness, proper units. Verifies numbers match sources, time periods are correct, and the answer addresses the question. |

#### plancraft — Sequential planning / Minecraft crafting

| Role | Persona |
|---|---|
| Recipe Knowledge Expert | Comprehensive knowledge of Minecraft crafting recipes. Knows ingredients, grid patterns, and intermediate components for complex recipes. |
| Inventory Manager | Tracks available resources meticulously. Knows inventory state at each step, anticipates resource consumption, identifies insufficiencies. |
| Plan Sequencer | Determines optimal crafting step order. Identifies prerequisites, finds shortest sequences, avoids dead-end paths that waste resources. |
| Action Executor | Translates high-level plans into precise, executable actions. Specifies slot placements, handles move-to-inventory steps, ensures syntactic validity. |
| Plan Verifier | Validates proposed plans by simulating inventory state after each step. Catches missing ingredients, impossible recipes, incorrect placements, resource conflicts. |

#### scicode — Scientific code generation

| Role | Persona |
|---|---|
| Algorithm Designer | Translates scientific problems into computational algorithms. Identifies appropriate numerical methods, data structures, and mathematical formulations. |
| Scientific Domain Expert | Broad knowledge of physics, chemistry, biology, applied mathematics. Understands scientific context and verifies implementations model phenomena correctly. |
| Code Implementer | Writes clean, correct, efficient scientific Python (NumPy, SciPy). Handles edge cases, numerical stability, array broadcasting. |
| Test Case Designer | Designs validation tests for scientific code. Creates inputs with known analytical solutions, checks boundaries, verifies conservation laws, ensures convergence. |
| Numerical Methods Specialist | Expert in ODE/PDE solvers, optimization, linear algebra, interpolation, integration. Selects solver parameters, understands convergence, diagnoses instabilities. |
| Code Reviewer | Reviews scientific code for correctness and best practices. Checks off-by-one errors, formula translations, unit handling, common pitfalls. |

#### stabletoolbench — API tool-calling coordination (1000+ tools)

| Role | Persona |
|---|---|
| API Discovery Specialist | Identifies relevant API tools from a large catalog. Understands categorization, reads descriptions efficiently, selects minimal tool sets. |
| API Call Planner | Designs API call sequences for complex tasks. Understands dependencies, handles pagination, plans error recovery. |
| Parameter Mapper | Correctly maps task requirements to API parameters. Understands types, required/optional fields, format constraints, value extraction. |
| Response Interpreter | Parses and interprets API responses accurately. Handles JSON/nested formats, extracts needed data points, identifies errors or partial results. |
| Tool Orchestration Coordinator | Coordinates multi-step workflows. Tracks gathered information, determines next calls, pipes outputs, decides when to finalize. |
| Error Recovery Specialist | Handles API failures gracefully. Diagnoses failures, devises alternatives, substitutes equivalent tools, ensures task completion. |

#### workbench — Workplace tool workflows (calendar, email, CRM)

| Role | Persona |
|---|---|
| Workflow Planner | Analyzes workplace tasks and decomposes them into tool operation sequences across calendar, email, CRM, and project management systems. |
| Calendar Operations Specialist | Expert at scheduling, availability checking, conflict resolution, recurring events, timezone conversions. |
| Email and Communication Expert | Handles email composition, retrieval, analysis. Understands threading, searches messages, drafts responses, extracts actionable information. |
| CRM Data Analyst | Navigates CRM data effectively. Looks up contacts, analyzes interactions, tracks pipelines, generates insights. |
| Cross-System Integrator | Specializes in cross-system tasks. Correlates data across calendar, email, CRM, project management. Ensures consistency. |
| Task Completion Verifier | Verifies operations executed correctly and completely. Checks all actions taken, data updated, final state matches requirements. |

#### webshop — Interactive e-commerce shopping

| Role | Persona |
|---|---|
| Product Search Specialist | Formulates effective product search queries. Identifies key attributes (brand, size, color, price), uses appropriate terms, filters efficiently. |
| Product Evaluator | Compares products against buyer requirements. Reads descriptions, specs, reviews. Assesses criteria match, ranks candidates by fit. |
| Navigation Strategist | Efficiently navigates e-commerce interfaces. Knows when to search, browse, filter, or click details. Minimizes steps to target item. |
| Purchase Decision Maker | Makes final buying decisions weighing all information: price-to-value, seller reliability, specs vs. requirements. |
| Attribute Matcher | Meticulously matches product attributes to requirements. Parses size charts, color variations, materials, compatibility information. |

#### agentbench — General cross-domain tasks (OS, DB, KG)

| Role | Persona |
|---|---|
| Operating System Expert | Proficient with Linux/Unix command-line. Navigates filesystems, manipulates files, writes scripts, manages processes. |
| Database Query Specialist | Expert at SQL queries. Understands schemas, writes complex joins/aggregations/subqueries, optimizes for requested information. |
| Knowledge Graph Navigator | Specializes in querying/traversing knowledge graphs. Understands entity-relationship structures, formulates SPARQL-like queries, navigates multi-hop relationships. |
| Task Decomposer | Breaks complex cross-domain tasks into subtasks. Identifies which domain each belongs to, determines dependencies, plans execution order. |
| Output Formatter | Ensures outputs match exact format requirements. Parses instructions carefully, extracts format specs, produces precisely conforming responses. |
| Error Diagnostician | Diagnoses and recovers from execution errors. Analyzes error messages, identifies root causes, devises corrected approaches. |

### Topology-specific assignment guidelines

The LLM assigner considers the agent's structural position when choosing domain roles:

| Topology | Structural position | Recommended domain role style |
|---|---|---|
| **sas** | single_agent | Assign the single most broadly capable role |
| **only_voting** | all voters | Assign diverse roles for perspective diversity |
| **orchestrator_no_discussion** | orchestrator | Coordinator / planner / auditor roles |
| | specialists | Focused execution / retrieval / analysis roles |
| **orchestrator_with_discussion** | orchestrator | Coordinator / synthesis / auditor roles |
| | specialists | Complementary analysis / verification roles |
| **orchestrator_tree_structure** | root_orchestrator | Broad coordinator / planner role |
| | managers | Domain-specific coordination roles |
| | leaf_workers | Focused execution / retrieval roles |
| **fully_linked_debate** | debaters | Diverse complementary roles for productive debate |
| **group_chat_debate** | representatives | Synthesis / coordination roles |
| | members | Focused analysis / verification roles |

### Edge cases

- **More agents than roles**: The LLM may assign the same role to multiple agents. The deterministic fallback uses round-robin.
- **Fewer agents than roles**: The LLM picks the most appropriate subset for the task.
- **SAS (1 agent)**: The single most relevant role is assigned.
- **LLM failure**: Falls back to deterministic round-robin assignment from the pool.
- **Unknown benchmark**: `get_role_pool()` returns an empty list; no persona injection occurs. Behavior is identical to the pre-dynamic-roles system.
- **`enable_dynamic_roles = false`**: Skips the LLM call entirely. Agents use only structural roles.

### Prompt integration

Before dynamic roles, each agent's system message looked like:

```
You are one agent in a deterministic multi-agent workflow.
Agent ID: agent_1
Agent Role: specialist
Stage Role: worker

Use only the task messages ...
```

After dynamic roles, it becomes:

```
You are one agent in a deterministic multi-agent workflow.
Agent ID: agent_1
Agent Role: specialist
Stage Role: worker
Domain Role: Financial Analyst
Persona: You specialize in interpreting financial statements, computing
financial ratios, analyzing revenue trends, and understanding corporate
financial disclosures. You can read balance sheets, income statements,
and cash flow statements to derive answers.

Use only the task messages ...
```

The stage context JSON payload also gains `domain_role` and `persona` keys:

```json
{
  "agent_id": "agent_1",
  "agent_role": "specialist",
  "stage_role": "worker",
  "domain_role": "Financial Analyst",
  "persona": "You specialize in interpreting financial statements ...",
  "directive": "...",
  ...
}
```

### Configuration

Add to `[mas]` in the TOML config to control dynamic role assignment:

```toml
[mas]
enable_dynamic_roles = true   # default: true; set false to disable
```

### Implementation pointers

- role pool definitions: `MAS/role_pools.py`
- LLM-based role assigner: `MAS/role_assigner.py`
- config flag: `MAS/config.py` (`MASConfig.enable_dynamic_roles`)
- state field: `MAS/state.py` (`WorkflowState.domain_personas`)
- integration point: `MAS/langgraph_engine.py` (`LangGraphMASEngine.run()`)
- prompt injection: `MAS/langgraph_engine.py` (`_build_agent_prompt()`)
- benchmark threading: each benchmark adapter passes `benchmark_name` to `runner.run_task()`

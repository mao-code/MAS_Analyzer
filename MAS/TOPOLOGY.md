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

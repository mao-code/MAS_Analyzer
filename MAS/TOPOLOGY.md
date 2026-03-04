## MAS Experiments Table
**Column guide**
- `Levels`: depth of hierarchy in the communication graph.
- `Same-level links`: lateral communication among peers at the same level.
- `Neighborhood fully connected`: whether each agent's local neighborhood is clique-like.
- `Agents tested`: agent counts / split patterns evaluated for that topology.
- `Configs`: number of experiment configurations.

| ID | Topology | Levels | Same-level links | Neighborhood fully connected | Agents tested | Turn mode | Configs |
|---|---|---:|---|---|---|---|---:|
| 1 | Orchestrator (Tree Structure) | 3 | None (0) | No | 3 (sequential); 5 ([1], [2], [1,1]); 7 ([1], [2], [2,2]) | Multi-turn | 3 |
| 2 | Orchestrator (No Discussion) | 2 | None (`0`) | No | 3, 4, 5 | Multi-turn | 3 |
| 3 | Orchestrator (With Discussion) | 2 | Full discussion relay via orchestrator (`1`) | No | 3, 4, 5 | Multi-turn | 3 |
| 4 | Only Voting | 1 | None (`0`) | No | 3, 4, 5 | Single-turn | 3 |
| 5 | Fully Linked Debate | 1 | Full (`1`, all-to-all) | Yes | 3, 4, 5 | Multi-turn | 3 |
| 6 | Group Chat + Debate | 1 | Grouped (intra-group links, limited/no inter-group links) | No | 3 (1,2); 4 (2,2); 5 (1,4 / 2,3) | Multi-turn | 4 |
| 7 | SAS (Single Agent System) | 1 | None (`0`) | N/A | 1 | Single-turn | 1 |

**Total configurations: 26**


## MAS Topologies (Definitions + Message Flow)

### 1) Orchestrator (Tree Structure)
**Definition:** A hierarchical coordinator (root) delegates to intermediate managers which delegate to leaf specialists. This is a structured “agents-as-tools / supervisor” pattern with multiple layers. 
**Message flow (typical):**
- User → Root Orchestrator (task intake)
- Root → Mid-level Orchestrators (subtask assignment)
- Mid-level → Leaf Agents (execution)
- Leaf → Mid-level (results)
- Mid-level → Root (aggregated results)
- Root → User (final)

**Diagram:**
- Root
  - Manager A
    - Worker A1
    - Worker A2
  - Manager B
    - Worker B1

**Relay mechanics:** parent-child routing; upward aggregation at each internal node; leaf agents do not talk laterally unless explicitly enabled.

---

### 2) Orchestrator (No Discussion)
**Definition:** A centralized supervisor dispatches tasks to specialists, but specialists do not see or comment on each other’s outputs; only the orchestrator integrates results. This matches “centralized” coordination patterns where a central node aggregates peer outputs.
**Message flow:**
- Orchestrator → Agent i (in parallel or sequence)
- Agent i → Orchestrator (result)
- Orchestrator → User (merge)

**Diagram:** `User → Orchestrator → {A,B,C} → Orchestrator → User`

**Relay mechanics:** star topology (hub-and-spoke). No peer-to-peer edges.

---

### 3) Orchestrator (With Discussion)
**Definition:** A centralized orchestrator still owns the routing, but it enables limited peer review by sharing one agent’s output (or a summary) with others for critique, then merges. This is “centralized with inter-agent debate,” where discussion is mediated by the central node (agents do not necessarily broadcast to everyone directly).  
**Message flow (one common variant):**
1. Orchestrator → {A,B,C}: produce initial candidates
2. Orchestrator → {A,B,C}: share peer candidates (or a distilled bundle)
3. {A,B,C} → Orchestrator: critiques/revisions
4. Orchestrator → User: final synthesis

**Diagram:** `User → O → {A,B,C} → O ↔ {A,B,C} → O → User`

**Relay mechanics:** orchestrator acts as the relay for “discussion rounds” (controls who sees what, and when).

---

### 4) Only Voting
**Definition:** Multiple agents run independently (no communication), then an aggregator selects the final answer by majority vote (or weighted vote). This is explicitly described as “Majority Voting” without debate.
**Message flow:**
- Prompt → {A,B,C,...} (parallel, independent)
- {A,B,C,...} → Aggregator (single-shot answers)
- Aggregator → User (vote result)

**Diagram:** `User → {A,B,C} → Vote → User`

**Relay mechanics:** no relaying between agents; only “answer collection” into the vote.

---

### 5) Fully Linked Debate
**Definition:** Fully connected communication graph: every agent can observe and respond to every other agent’s messages in each round (decentralized debate).  
**Message flow (round-based):**
- Round 0: each agent produces an initial answer
- Round t: each agent reads all others’ latest messages, then revises
- Final: selection via a judge/aggregator or consensus rule (implementation choice)

**Diagram:** complete graph `K_n` (all-to-all).

**Relay mechanics:** broadcast (or shared channel) per round; high token/latency; strong cross-influence.

---

### 6) Group Chat + Debate
**Definition:** Agents are partitioned into disconnected (or weakly connected) groups. Debate happens within each group, then a cross-group merge happens via an orchestrator or a representative. This is a “disconnected components” topology choice.
**Message flow (typical):**
1. Split agents into groups G1, G2, ...
2. Intra-group debate (multi-turn within each group)
3. Group representative → Orchestrator (group conclusion)
4. Orchestrator → User (final merge) or optional “final debate” among representatives

**Diagram:** `(A,B,C) fully/partially connected` + `(D,E,F) fully/partially connected`, minimal/no edges between the sets.

**Relay mechanics:** reduce cross-talk cost; preserve diversity by limiting global conformity; requires an explicit merge step.

---

### 7) SAS (Single Agent System)
**Definition:** One agent solves end-to-end; no inter-agent communication.
**Message flow:** `User → Agent → User`  
**Relay mechanics:** none.
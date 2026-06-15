# Self-Evolved Topology Planning — Implementation Plan

> **Status (2026-06): implemented.** All six phases below shipped under `MAS/self_evolved/`,
> and the persistent playbook seed lives at `config/topology_playbook.json`. This document is
> retained as the design record; for current usage see the "Self-evolved topology system"
> section (and workflow diagram) in `README.md`.

## Context

The research doc (`self_evolved_topology_planning_clear_agents_final.docx`) proposes replacing the
fixed MAS topology with a query-conditioned topology chosen by an LLM **Topology Planner**, executed
by an **Orchestrator** over a *dynamic* target agent system, audited by a **Trace Auditor** that can
trigger **at most one trace-backed repair** before finalization, with a persistent **Topology
Playbook** and a **Shared Context Controller** managing per-agent visibility. Target failure modes:
static topology mismatch, message compaction loss, evidence lost before synthesis, tool-error
cascades, missing validators. The core contribution is the meta-control mechanism, not new
hand-designed agents — all expert agents share one Agent Harness, specialized only by role prompt,
allowed tools, context boundary, and graph position.

**User decisions (binding):**
1. **Dual runtime with a config switch** — the harness runs on both the existing
   `OpenRouterLLMClient` (mock mode preserved for offline tests) and the **Claude Agent SDK**
   (`claude-agent-sdk`, requires `ANTHROPIC_API_KEY`).
2. **New system variant** — `self_evolved` registers like any topology so it flows through
   `python main.py run` / `scripts/full_experiment.sh`, writes to the standard artifact hierarchy,
   and is Q/C/D/R/P-comparable to SAS/fixed-MAS in `summary.csv`.
3. **Full design, phased** — all five components, in 6 verifiable phases.

**Repo invariants that must hold:** metric contract in README is source of truth;
`benchmark.evaluate(...).success` is the only correctness authority; mock mode keeps working;
bounded relay packets, visibility controlled in code; agents never decide loop termination; all
run-level metrics recomputable from the trace.

## Architecture

New subpackage `MAS/self_evolved/` (ships under existing `include = ["MAS*"]` packaging):

```
MAS/self_evolved/
  spec.py            # TopologySpec / AgentNode / GroupSpec / ContextPolicy / TopologyMutation
  context.py         # SharedContextController (deterministic infra)
  executor.py        # TurnExecutor: interprets a TopologySpec for one turn
  planner.py         # TopologyPlannerAgent (LLM + deterministic fallback)
  auditor.py         # TraceAuditorAgent (heuristics + optional llm_judge pass)
  playbook.py        # TopologyPlaybook persistence + PlaybookMaintainer
  engine.py          # SelfEvolvedEngine: meta-loop, one-repair logic, finalize
  harness.py         # AgentHarness protocol + build_harness() factory
  claude_harness.py  # ClaudeAgentSDKClient (lazy import of claude_agent_sdk)
```
Plus `scripts/update_topology_playbook.py` (post-hoc, success-conditioned playbook updates).

### TopologySpec (general, heterogeneous, nested)

Superset of the 7 uniform layouts. Frozen dataclasses in `spec.py`:
- `ContextPolicy`: `visible_kinds`, `visible_from` (parent/children/group/explicit ids),
  `share_scope`, `evidence_access` (own/branch/global), `summary_only`, `max_packet_chars`.
- `AgentNode`: `agent_id`, `structural_role`, `stage_role` (**must** be one of the engine's existing
  stage contracts: planner/worker/critic/aggregator), `group_id`, `allowed_tools`, `context`.
- `GroupSpec`: `group_id`, `pattern` (singleton/star/chain/debate/voting), `parent_agent_id`
  (nesting point), `member_ids`.
- `TopologySpec`: `version` (0 initial, 1 after repair), `agents`, `groups`, `edges`,
  `root_group_id`, `rationale`; methods `validate(max_agents)`, `to_layout()`,
  `group_execution_order()`.
- `TopologyMutation`: `rationale`, `target_failure_modes`, `ops: tuple[MutationOp,...]`,
  `apply(spec, max_agents) -> TopologySpec`. Ops: `expand_agent_to_group` (the doc's example: leaf →
  star of N subagents / debate group), `set_group_pattern`, `add_agent`, `add_edge`, `remove_edge`,
  `set_context_policy`.

`TopologySpec.to_layout()` projects to a real `MAS.relay.TopologyLayout` (`topology="self_evolved"`)
so three things work unchanged: `_execute_agent_stage` (reads `layout.roles`/`agent_ids`),
`run_metadata["topology_layout"] = layout.to_payload()` (trajectory/graph writers, topology
analysis), and `role_assigner.assign_domain_roles(...)`. `spec_from_layout(build_layout(...))`
converts any existing uniform layout into a spec — used for deterministic planner fallbacks.

### AgentHarness — one protocol, two backends

The protocol **is** `OpenRouterLLMClient.generate`'s keyword signature returning `LLMResult`
(verified at `MAS/llm.py:23-31, 208-219`). `_execute_agent_stage` in `langgraph_engine.py` is then
the shared Agent Harness for both backends — same prompts (`_build_agent_prompt`), artifact
coercion, tool-retry contract, trace emission.

- **OpenRouter backend**: `build_harness()` returns the existing client untouched — mock mode,
  tool loop, env flags inherited for free.
- **Claude Agent SDK backend** (`claude_harness.py::ClaudeAgentSDKClient`): lazy
  `import claude_agent_sdk`; clear error if package/`ANTHROPIC_API_KEY` missing. Bridges repo tool
  dicts (`{"name","description","parameters","handler"}`, format per `_normalize_tools`,
  `MAS/llm.py:860+`) → in-process MCP tools via `claude_agent_sdk.tool` +
  `create_sdk_mcp_server`; `max_tool_iterations` → `ClaudeAgentOptions(max_turns=...)`;
  `allowed_tools` limited to the bridged MCP tools (preserves context boundaries — no file/bash).
  Maps SDK stream back to `LLMResult`: tool uses → `tool_calls` records, `usage` → tokens,
  `total_cost_usd` → cost, `metadata["provider"]="claude_agent_sdk"`. Async query via
  `asyncio.run` in a worker thread (same pattern as llm.py's hard timeout). Because TraceEvents are
  emitted by `_execute_agent_stage` from `LLMResult` fields, **no separate trace bridge is needed**.
- `pyproject.toml`: optional extra `[project.optional-dependencies] claude = ["claude-agent-sdk>=0.1"]`.

### Meta-agents

| Doc concept | Implementation |
|---|---|
| Topology Planner | `planner.py` — `propose_initial(task, playbook_view)`, `propose_mutation(audit_report, spec, playbook_view)`. Modeled on `MAS/role_assigner.py::assign_domain_roles` (prompt → strict JSON parse → deterministic fallback + `used_fallback`). Fallback initial spec: `spec_from_layout(build_layout("orchestrator_no_discussion", n))`; fallback mutation: `None`. |
| Orchestrator | **Design decision (deviation from doc, flagged):** mechanical duties (spawn, route, visibility, apply mutation, termination) are deterministic code in `engine.py`/`executor.py`/`context.py`, honoring "agents never decide loop termination". The *cognitive* duty (decomposition, directives) is the LLM `coordinator` AgentNode inside the target spec — same as today's orchestrator stages. Mutations stay auditable. |
| Trace Auditor | `auditor.py::audit(state, spec, turn) -> AuditReport(detected_modes, severity, recommendation, per_branch_findings)`. Deterministic heuristics adapted from `scripts/generate_mas_failure_analysis_report.py:40-110`, restricted to **process-observable signals** (no eval ground truth in-run): failing tool_records → `tool_error_cascade`; zero-evidence artifacts feeding aggregator → `evidence_lost_before_synthesis`; high consensus + low confidence/unresolved issues → `premature_consensus`; blocked/invalid branch → `branch_collapse`; truncated packets → `message_compaction_loss`; no critic/verifier downstream of low confidence → `missing_validator`. Optional `audit_mode="llm_judge"` refinement with deterministic fallback (same pattern as `_compute_termination_assessment`). |
| Playbook Maintainer | `playbook.py` provides mixed memory. A short-term in-memory playbook is updated after each audited turn and can inform the one in-run repair. The persistent JSON playbook (default `config/topology_playbook.json`) is keyed by `benchmark_name` + coarse deterministic task features (no embeddings) and remains post-hoc: the engine records `run_metadata["playbook_update_candidate"]`; `scripts/update_topology_playbook.py` joins candidates with `run_*.eval.json` success and updates the file. This preserves correctness authority and avoids write races under parallel experiments. |
| Shared Context Controller | `context.py` — pure code. `visible_packets(state, agent_id)` generalizes the per-topology `message_selector` lambdas using each agent's `ContextPolicy` (reuses `_messages_for_recipient` semantics). `emit(...)` builds bounded packets via `packet_payload_from_artifact`/`packet_content` (`MAS/artifacts.py:165-198`), honoring `share_scope`/`summary_only`. Append-only `evidence_ledger` per stage; finalize aggregator always receives the global evidence digest (defense against "evidence lost before synthesis"). Recomputes visibility from new spec after mutation. |

## Execution model

`SelfEvolvedEngine` is a **direct interpreter** of TopologySpec, not LangGraph (compiled graphs are
static; per-turn recompilation would need new builders for arbitrary nested specs anyway). State is
a plain dict shaped like `WorkflowState` (same keys: `messages`, `artifacts`, `trace_payloads`,
`interaction_logs`, `termination_history`, counters) so downstream consumers are unchanged. It holds
a `LangGraphMASEngine` instance to reuse `_execute_agent_stage`, `_build_agent_prompt`,
`_draft_event`, `_termination_event`, `_materialize_trace_events`.

Turn lifecycle:
1. **PLAN** — long-term playbook lookup → `planner.propose_initial` → validate `spec_v0`. Emit
   `actor="topology_planner", event_type="plan"` with spec payload, rationale, `used_fallback`.
2. **SPAWN** — `layout = spec_v0.to_layout()`; personas via existing `assign_domain_roles`; init
   state. Emit `actor="orchestrator", event_type="plan"` (agents, context policies).
3. **EXECUTE turn 0** — `TurnExecutor.run_turn`: depth-first over `group_execution_order()`;
   pattern semantics (singleton/star/chain/debate/voting) composed from `_execute_agent_stage`
   calls + context-controller packet emission; group result relayed up as parent input. Stages emit
   the normal act/tool_call/tool_result/verify events with `actor=agent_id`.
4. **AUDIT** — `auditor.audit(...)`. Emit `actor="trace_auditor", event_type="verify"`.
5. **CONTROL** — `_meta_termination` ordered checks (code, never agents):
   (a) invalid_or_failed_branch → stop; (b) consensus_reached (`compute_consensus` ≥ 0.75 and
   substantive) → stop; (c) audit recommends repair AND `mutations_used < 1` AND planner returns a
   valid mutation → continue (the **only** continue path); (d) else stop. Emit via
   `_termination_event` payload shape.
6. **MUTATE** (path c only) — `planner.propose_mutation` → `mutation.apply(spec_v0)` → `spec_v1`;
   context controller revalidates; new agents get deterministic round-robin personas. Emit
   `actor="orchestrator", event_type="revise"` with ops + new layout payload.
7. **EXECUTE turn 1** — same as 3 with `round_index=1`; surviving agents keep prior-artifact
   continuity (resolved by agent_id already).
8. **AUDIT** — updates short-term playbook memory and final long-term candidate; cannot trigger another mutation; forced
   stop (`max_rounds_reached` or `consensus_reached`).
9. **FINALIZE** — candidates = root aggregator's latest substantive artifact else leaves;
   `final_vote_mode` deterministic (`relay.vote_artifacts`) or llm_judge with deterministic
   fallback. Emit `actor="system", event_type="finalize"`.
10. **RECORD** — `maintainer.build_update_candidate` → `run_metadata` only. Materialize trace.

All event types ∈ `descriptor/schema.py::EVENT_TYPES`; meta-agent LLM tokens/cost ride on their own
events so `C*` metrics include meta-control overhead; dispatch ids allocated monotonically so
`validate_trace_events` passes. `run_metadata` keeps the same keys as `LangGraphMASEngine.run`
plus: `topology_spec_versions`, `mutation`, `audit_reports`, `harness_backend`,
`playbook_update_candidate`.

## Config, CLI, registration

New `[self_evolved]` TOML section → `SelfEvolvedConfig` dataclass + `validate()` in `MAS/config.py`,
field on `ExperimentConfig`:
```toml
[self_evolved]
harness_backend = "openrouter"   # or "claude_agent_sdk"
max_initial_agents = 5
max_total_agents = 10            # hard cap after mutation
max_turns = 2                    # 1 + one repair
audit_mode = "heuristic"         # or "llm_judge"
playbook_path = "config/topology_playbook.json"
playbook_read = true
default_packet_max_chars = 320
```
Planner/auditor models route through the existing `[models]` table as agent_types
`planner`/`auditor` (fallback to `default` via `model_for_agent_type`).

Registration:
1. `MAS/relay.py` — add `TOPOLOGY_SELF_EVOLVED = "self_evolved"` to `SUPPORTED_TOPOLOGIES` +
   aliases; in `build_layout` raise a targeted error (layouts are built per-run by the new engine).
2. `MAS/runner.py::run_task` — top-of-function dispatch:
   `if resolved_topology == TOPOLOGY_SELF_EVOLVED:` → lazily-built `SelfEvolvedEngine` →
   `MASRunResult`. **Benchmark adapters need zero changes.**
3. `main.py` — `_default_system_label` already returns the topology name; guard
   `_write_system_graph_artifact` (main.py:1275 calls `build_layout`) to write a
   `{"topology":"self_evolved","dynamic":true}` placeholder instead. Optional CLI flags
   `--harness-backend`, `--playbook-path`.
4. `scripts/full_experiment.py` — append `("self_evolved", "self_evolved", 5, 2, 1, 2)` to
   `SYSTEMS` and `"self_evolved": "SELF_EVOLVED_ARGS"` to `SYSTEM_ARG_ENV_BY_LABEL`.
5. `config/experiment.example.toml` — document the section. Add `self_evolved` to
   `scripts/generate_mas_failure_analysis_report.py::SYSTEM_LABELS` (cosmetic).

## Phases (each mock-mode pytest-verifiable; `pytest` + `ruff check .` green per phase)

**Phase 1 — TopologySpec + TurnExecutor + harness seam + system registration.**
Create `spec.py`, `context.py` (minimal visibility), `executor.py`, `engine.py` (hardcoded
fallback spec; planner/auditor stubs), `harness.py` (OpenRouter pass-through). Modify `relay.py`,
`runner.py`, `config.py`, `main.py`, example TOML. Optional pre-refactor if desired: extract
`_execute_agent_stage`/`_build_agent_prompt`/trace helpers into `MAS/stage_exec.py` for a clean
seam (mechanical; otherwise reuse the private methods directly).
*Verify:* `tests/test_self_evolved_spec.py` (hand-built nested spec: coordinator + star subgroup +
debate subgroup → `validate()`, `to_layout()` assertions); `tests/test_self_evolved_engine_smoke.py`
(mock LLM run: `validate_trace_events` passes, per-stage act events, policy-filtered
`message_views`, non-empty final answer); main.py smoke clone of `tests/test_main_smoke.py` with
`topology="self_evolved"` → trace/descriptor/summary.csv exist.

**Phase 2 — Topology Planner.** Create `planner.py` (role_assigner pattern), wire
`propose_initial`, emit planner plan event.
*Verify:* `tests/test_self_evolved_planner.py` — fake LLM returns valid plan JSON → spec parsed;
malformed → fallback + `used_fallback=True`; e2e mock run contains the planner event.

**Phase 3 — Trace Auditor + one-repair loop + mutations.** Create `auditor.py`; add
`TopologyMutation` ops; `_meta_termination` + mutation application; `propose_mutation`.
*Verify:* `tests/test_self_evolved_auditor.py` / `test_self_evolved_repair.py` — injected failing
tool_records → `tool_error_cascade` flagged; stubbed mutation (doc's example: leaf→star-of-3 +
leaf→debate) applied exactly once (`topology_spec_versions` length 2; second recommendation refused
with `mutation_budget_exhausted`); high-consensus turn 0 → stop without mutation; exactly one
orchestrator `revise` event.

**Phase 4 — Playbook + maintainer.** Create `playbook.py`, `scripts/update_topology_playbook.py`,
seed `config/topology_playbook.json`.
*Verify:* `tests/test_self_evolved_playbook.py` — tmp-path persistence round-trip; deterministic
keying; planner prompt contains playbook entries (captured via fake LLM); updater merges a
fabricated experiment dir success-conditioned; engine never writes the playbook during `run`.

**Phase 5 — Full context-policy semantics.** Complete `context.py` (evidence ledger digests,
`summary_only` compaction, branch privacy) + `executor.py` ledger appends; aggregator always gets
the global evidence digest.
*Verify:* `tests/test_self_evolved_context.py` — visibility matrix per policy; aggregator prompt
contains all-branch evidence digest; private-branch packets never in sibling views; oversize
content bounded.

**Phase 6 — Claude Agent SDK backend.** Create `claude_harness.py`; `build_harness` switch;
optional extra in `pyproject.toml`; docs.
*Verify:* `tests/test_self_evolved_claude_harness.py` — fake `claude_agent_sdk` module injected
into `sys.modules` (no network/dep in CI): tool bridging, usage/cost mapping into `LLMResult`,
missing-key error, full engine run through the fake SDK producing schema-valid traces identical in
shape to the OpenRouter path.

## End-to-end verification

```bash
pytest                                   # offline, mock mode — full suite green
ruff check . && ruff format --check .
python main.py run --config config/experiment.toml --benchmark browsecomp \
  --task-limit 1 --runs-per-task 1      # with topology="self_evolved" in [mas]
python main.py summarize-experiment --experiment-root artifacts/full_experiment/<id>
TASK_LIMIT=1 RUNS_PER_TASK=1 bash scripts/full_experiment.sh --benchmarks browsecomp
# Live smoke (optional, needs keys): OPENROUTER_API_KEY for backend=openrouter;
# ANTHROPIC_API_KEY + pip install -e ".[claude]" for backend=claude_agent_sdk
python scripts/update_topology_playbook.py --experiment-root artifacts/full_experiment/<id>
```
Check: artifacts land in `<id>/<benchmark>/self_evolved/<task>/`; `run_0.trace.jsonl` validates;
descriptor + summary.csv include the system; trace shows plan → spawn → turn 0 → audit →
(revise) → turn 1 → finalize.

## Risks / notes

1. **Orchestrator is code, not a 4th LLM agent** — required by the termination invariant; cognitive
   work remains in the LLM coordinator node. Deviation from the doc's framing, flagged above.
2. **Playbook learning across runs breaks run independence** — mitigated by in-memory short-term
   updates plus a post-hoc persistent updater; per-experiment playbook copies possible later.
3. **In-run audit has no ground truth** — process-signal heuristics only; `llm_judge` mode refines
   at token cost.
4. **Cross-backend C\* metrics aren't directly comparable** (SDK runs an internal loop; cumulative
   tokens, higher latency) — fine since backend is a config dimension; note it in README.
5. **Cost ceiling** — mutation can ~double agents; guarded by `max_total_agents` and `max_turns=2`.
6. **Private-method reuse** from `langgraph_engine.py` — acceptable inside the `MAS` package;
   optional `stage_exec.py` extraction in Phase 1 if a cleaner seam is preferred.

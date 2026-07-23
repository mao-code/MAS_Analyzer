import json
import re
import unittest
from dataclasses import dataclass, field
from typing import Any

from benchmark.math500 import Math500Benchmark
from descriptor.schema import validate_trace_events
from MAS.config import SelfEvolvedConfig
from MAS.langgraph_engine import ExperimentSpec
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.relay import build_layout
from MAS.self_evolved.engine import SelfEvolvedEngine
from MAS.self_evolved.executor import _READ_FIRST_DIRECTIVE, TurnExecutor, state_changing_tool_names
from MAS.self_evolved.spec import GroupSpec, MutationOp, TopologyMutation, spec_from_layout

BLOCKED_TEXT = "Insufficient evidence: the answer cannot be determined."

_INITIAL_PLAN = json.dumps(
    {
        "rationale": "Star with three branches for a decomposable task.",
        "pattern": "star",
        "num_agents": 3,
        "expansions": [],
    }
)

# The doc's example repair: one leaf becomes a star-orchestrator, the other a
# fully-linked debate group.
_MUTATION_PLAN = json.dumps(
    {
        "rationale": "Branch agent_1 lacked decomposition; branch agent_2 needs verification.",
        "ops": [
            {
                "op": "expand_agent_to_group",
                "agent_id": "agent_1",
                "pattern": "star",
                "num_subagents": 2,
            },
            {
                "op": "expand_agent_to_group",
                "agent_id": "agent_2",
                "pattern": "debate",
                "num_subagents": 2,
            },
        ],
    }
)


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def test_voting_members_receive_distinct_independence_routes() -> None:
    group = GroupSpec(
        group_id="g_root",
        pattern="voting",
        member_ids=("agent_0", "agent_1", "agent_2", "agent_3"),
    )

    directives = [
        TurnExecutor._independent_route_directive(group, agent_id) for agent_id in group.member_ids
    ]

    assert all("Do not treat likely peer agreement as evidence" in item for item in directives)
    assert len(set(directives)) == 4
    assert "first principles" in directives[0]
    assert "falsify" in directives[2]


def test_retrieval_directive_expands_ranges_and_diversifies_queries() -> None:
    assert "call constraint_search first" in _READ_FIRST_DIRECTIVE
    assert "one value at a time" in _READ_FIRST_DIRECTIVE
    assert "Copy rare modifiers" in _READ_FIRST_DIRECTIVE
    assert "high-information conjunction" in _READ_FIRST_DIRECTIVE
    assert "do not repeat a peer's query" in _READ_FIRST_DIRECTIVE


def test_state_changing_tools_are_identified_without_hiding_reads() -> None:
    tools = [
        {"name": "calendar.search_events"},
        {"name": "calendar.get_event_information_by_id"},
        {"name": "calendar.create_event"},
        {"name": "customer_relationship_manager.update_customer"},
    ]

    assert state_changing_tool_names(tools) == {
        "calendar.create_event",
        "customer_relationship_manager.update_customer",
    }


def test_group_vote_recovers_majority_from_malformed_structured_wrappers() -> None:
    artifacts = [
        {
            "agent_id": "agent_0",
            "answer": r"\boxed{2-(3+2\sqrt{2})i}",
            "confidence": 1.0,
            "evidence_summary": [],
            "tool_records": [],
        },
        {
            "agent_id": "agent_1",
            "answer": r'{"answer_artifact":"derivation with invalid \i ... \boxed{6-5i}"}',
            "confidence": 1.0,
            "evidence_summary": [],
            "tool_records": [],
        },
        {
            "agent_id": "agent_2",
            "answer": r"Independent derivation. \boxed{6-5i}",
            "confidence": 1.0,
            "evidence_summary": [],
            "tool_records": [],
        },
        {
            "agent_id": "agent_3",
            "answer": r'{"answer_artifact":"another invalid \p wrapper ... \boxed{6-5i}"}',
            "confidence": 1.0,
            "evidence_summary": [],
            "tool_records": [],
        },
    ]

    selected = TurnExecutor._select_group_output(artifacts)

    assert selected is not None
    assert selected["agent_id"] in {"agent_1", "agent_2", "agent_3"}
    assert selected["answer"] == r"\boxed{6-5i}"
    assert TurnExecutor._decision_answer_signature(str(selected["answer"])) == "6 5i"


class _ScriptedLLM(OpenRouterLLMClient):
    """Planner gets scripted responses; workers are blocked in round 0 and
    answer directly in round 1; the coordinator always answers directly."""

    def __init__(self, planner_responses: list[str], *, blocked_round0: bool = True) -> None:
        self._planner_responses = list(planner_responses)
        self._blocked_round0 = blocked_round0
        self.calls: list[str] = []
        self.planner_prompts: list[str] = []
        self.worker_prompts: list[str] = []

    def generate(
        self,
        *,
        prompt,
        agent_type,
        task_id,
        run_index,
        agent_id,
        tools=None,
        max_tool_iterations=8,
        temperature=0.0,
    ) -> LLMResult:
        self.calls.append(agent_id)
        prompt_text = (
            prompt
            if isinstance(prompt, str)
            else " ".join(str(m.get("content", "")) for m in prompt)
        )
        if agent_id == "topology_planner":
            self.planner_prompts.append(prompt_text)
            text = self._planner_responses.pop(0) if self._planner_responses else "{}"
        elif agent_id == "agent_0":
            self.worker_prompts.append(prompt_text)
            text = "The final answer is 42."
        elif self._blocked_round0 and '"round_index": 0' in prompt_text:
            self.worker_prompts.append(prompt_text)
            text = BLOCKED_TEXT
        else:
            self.worker_prompts.append(prompt_text)
            text = "The answer is 42."
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="scripted",
            mock_used=False,
            metadata={},
        )


class _ConstraintSynthesisLLM(_ScriptedLLM):
    def generate(self, **kwargs) -> LLMResult:
        agent_id = str(kwargs.get("agent_id", ""))
        if agent_id == "final_synthesizer":
            self.calls.append(agent_id)
            self.worker_prompts.append(
                " ".join(str(item.get("content", "")) for item in kwargs.get("prompt", []))
            )
            text = json.dumps(
                {
                    "answer_artifact": (
                        "Boeing 737 was retrieved, but passenger capacity and range were not "
                        "provided."
                    ),
                    "summary": "The row is incomplete.",
                    "critique": "Requested fields are missing.",
                    "revision_request": "Preserve the row schema.",
                    "confidence": 0.4,
                    "unresolved_issues": ["capacity and range"],
                    "evidence_summary": ["Boeing 737 was returned by the tool."],
                }
            )
        elif agent_id == "constraint_reconciler":
            self.calls.append(agent_id)
            self.worker_prompts.append(
                " ".join(str(item.get("content", "")) for item in kwargs.get("prompt", []))
            )
            text = json.dumps(
                {
                    "answer_artifact": (
                        "Boeing 737 — passenger capacity: unknown; range: unknown. "
                        "The source record establishes the aircraft name but omits both fields."
                    ),
                    "summary": "Every requested row field is preserved.",
                    "critique": "",
                    "revision_request": "",
                    "confidence": 0.5,
                    "unresolved_issues": [],
                    "evidence_summary": ["Boeing 737 was returned by the tool."],
                }
            )
        else:
            return super().generate(**kwargs)
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="scripted",
            mock_used=False,
            metadata={},
        )


class _ProofFalsificationLLM(_ScriptedLLM):
    def generate(self, **kwargs) -> LLMResult:
        agent_id = str(kwargs.get("agent_id", ""))
        if agent_id in {"final_synthesizer", "proof_falsifier"}:
            self.calls.append(agent_id)
            self.worker_prompts.append(
                " ".join(str(item.get("content", "")) for item in kwargs.get("prompt", []))
            )
            answer = r"\boxed{62}" if agent_id == "final_synthesizer" else r"\boxed{28}"
            text = json.dumps(
                {
                    "answer_artifact": answer,
                    "summary": "The decisive angle relation was independently checked.",
                    "critique": (
                        "The draft confused an interior angle with its supplement."
                        if agent_id == "proof_falsifier"
                        else ""
                    ),
                    "revision_request": "",
                    "confidence": 0.9,
                    "unresolved_issues": [],
                    "evidence_summary": ["The candidate derivations disagree."],
                }
            )
        else:
            return super().generate(**kwargs)
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="scripted",
            mock_used=False,
            metadata={},
        )


class _ProofAppealLLM(_ProofFalsificationLLM):
    def generate(self, **kwargs) -> LLMResult:
        agent_id = str(kwargs.get("agent_id", ""))
        if agent_id not in {
            "proof_falsifier",
            "proof_appeal_reviewer",
            "proof_oracle_corrector",
        }:
            return super().generate(**kwargs)
        self.calls.append(agent_id)
        self.worker_prompts.append(
            " ".join(str(item.get("content", "")) for item in kwargs.get("prompt", []))
        )
        answer = r"\boxed{62}"
        summary = (
            r"The ordered rays show that \angle ABC=56 degrees."
            if agent_id in {"proof_appeal_reviewer", "proof_oracle_corrector"}
            else "Ordered vertex rays distinguish the supplementary angles."
        )
        return LLMResult(
            text=json.dumps(
                {
                    "answer_artifact": answer,
                    "summary": summary,
                    "critique": "The unchanged incumbent used AB where the vertex requires BA.",
                    "revision_request": "",
                    "confidence": 0.9,
                    "unresolved_issues": [],
                    "evidence_summary": ["Reversing a ray changes direction by 180 degrees."],
                }
            ),
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="scripted",
            mock_used=False,
            metadata={},
        )


class _OutputContractLLM(OpenRouterLLMClient):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, **kwargs) -> LLMResult:
        agent_id = str(kwargs.get("agent_id", ""))
        self.calls.append(agent_id)
        if agent_id == "topology_planner":
            text = json.dumps(
                {
                    "rationale": "One solver is sufficient for this formatting regression.",
                    "pattern": "singleton",
                    "num_agents": 1,
                    "expansions": [],
                }
            )
        elif agent_id == "output_contract_enforcer":
            text = r"\boxed{\left(3,\frac{\pi}{2}\right)}"
        else:
            text = r"(3, \pi/2)"
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="scripted",
            mock_used=False,
            metadata={},
        )


class _RoundScriptedLLM(_ScriptedLLM):
    """Workers are blocked (with per-round wording, so turns differ) for every
    round in ``blocked_rounds``; the leader's aggregate also varies per round so
    the no_meaningful_change check does not fire before a repair can run."""

    def __init__(self, planner_responses: list[str], *, blocked_rounds: set[int]) -> None:
        super().__init__(planner_responses, blocked_round0=False)
        self._blocked_rounds = set(blocked_rounds)

    def generate(
        self,
        *,
        prompt,
        agent_type,
        task_id,
        run_index,
        agent_id,
        tools=None,
        max_tool_iterations=8,
        temperature=0.0,
    ) -> LLMResult:
        prompt_text = (
            prompt
            if isinstance(prompt, str)
            else " ".join(str(m.get("content", "")) for m in prompt)
        )
        round_index = None
        match = re.search(r'"round_index": (\d+)', prompt_text)
        if match:
            round_index = int(match.group(1))
        if agent_id != "topology_planner" and round_index is not None:
            self.calls.append(agent_id)
            leader_texts = {
                0: "Synthesis attempt one: both branches returned nothing usable yet.",
                1: (
                    "Second synthesis after restructuring: coverage widened but every "
                    "branch report is still marked unresolved, so no aggregate exists."
                ),
            }
            blocked_texts = {
                0: BLOCKED_TEXT,
                1: (
                    "Insufficient evidence: the expanded subgroup likewise failed to "
                    "ground either branch, verification still pending."
                ),
            }
            if agent_id == "agent_0":
                text = leader_texts.get(round_index, "The final answer is 42.")
            elif round_index in self._blocked_rounds:
                text = blocked_texts.get(round_index, BLOCKED_TEXT)
            else:
                text = "The answer is 42."
            return LLMResult(
                text=text,
                token_in=10,
                token_out=5,
                cost_usd=0.0,
                model="scripted",
                mock_used=False,
                metadata={},
            )
        return super().generate(
            prompt=prompt,
            agent_type=agent_type,
            task_id=task_id,
            run_index=run_index,
            agent_id=agent_id,
            tools=tools,
            max_tool_iterations=max_tool_iterations,
            temperature=temperature,
        )


class _OpenSetAuditLLM(_ScriptedLLM):
    """Return one grounded high-risk open-set finding, then a clean audit."""

    def __init__(self, planner_responses: list[str]) -> None:
        super().__init__(planner_responses, blocked_round0=False)
        self.audit_calls = 0

    def generate(self, **kwargs) -> LLMResult:
        if kwargs.get("agent_id") != "trace_auditor":
            return super().generate(**kwargs)
        prompt = kwargs.get("prompt") or []
        prompt_text = " ".join(str(item.get("content", "")) for item in prompt)
        refs = re.findall(r'"artifact_id":\s*"([^"]+)"', prompt_text)
        evidence_ref = next((ref for ref in refs if "agent_1" in ref), "")
        self.audit_calls += 1
        if self.audit_calls == 1:
            text = json.dumps(
                {
                    "repair_recommended": True,
                    "recommendation": "Add an independent critic to challenge correlated agreement.",
                    "new_failure_modes": [
                        {
                            "mode": "correlated_consensus",
                            "severity": "high",
                            "confidence": 0.9,
                            "repairable": True,
                            "agent_ids": ["agent_1", "agent_2"],
                            "evidence": [
                                {
                                    "ref": "task",
                                    "quote": "Question with two weak branches",
                                },
                                {
                                    "ref": evidence_ref,
                                    "quote": "The answer is 42.",
                                },
                            ],
                            "detail": "The agreeing workers repeat the same unsupported rationale.",
                        }
                    ],
                }
            )
        else:
            text = json.dumps(
                {
                    "repair_recommended": False,
                    "recommendation": "The repaired turn is decision-grade.",
                    "new_failure_modes": [],
                }
            )
        return LLMResult(
            text=text,
            token_in=10,
            token_out=5,
            cost_usd=0.0,
            model="scripted",
            mock_used=False,
            metadata={},
        )


def _experiment_spec() -> ExperimentSpec:
    return ExperimentSpec(
        topology="self_evolved",
        num_agents=3,
        rounds=2,
        communication_budget_per_agent=2,
        termination_consensus_mode="lexical",
        final_vote_mode="deterministic",
        benchmark_name="finance_agent",
        enable_dynamic_roles=False,
    )


def _run(engine: SelfEvolvedEngine) -> Any:
    return engine.run(
        task=_Task(task_id="repair-task", prompt="Question with two weak branches"),
        run_index=0,
        seed=3,
        spec=_experiment_spec(),
        agent_types=["general"],
    )


class TestSingleRepairLoop(unittest.TestCase):
    def test_failed_tool_synthesis_allows_labeled_general_background(self) -> None:
        client = _ScriptedLLM([_INITIAL_PLAN])
        engine = SelfEvolvedEngine(client, SelfEvolvedConfig())
        state = {
            "task_id": "limited-tool-task",
            "task_prompt": "Give practical guidance covering parts A and B.",
            "run_index": 0,
            "round_index": 0,
            "discussion_index": 0,
            "dispatch_id": 0,
            "llm_client": client,
            "artifacts": [],
            "interaction_logs": [],
            "trace_events": [],
            "tools": [],
            "tool_records_log": [
                {
                    "tool_name": "guidance.lookup",
                    "arguments": {},
                    "status": "completed",
                    "output": "This endpoint describes the catalog but returns no records.",
                }
            ],
        }

        artifact = engine._maybe_synthesize_final_answer(
            state=state,
            final_answer=BLOCKED_TEXT,
            selected_artifact_id="",
            vote_result={},
        )

        self.assertIsNotNone(artifact)
        prompt = client.worker_prompts[-1]
        self.assertIn("durable, widely established background knowledge", prompt)
        self.assertIn("never claim it was tool-verified", prompt)
        self.assertIn("preserve every available entity", prompt)
        self.assertIn("explicitly approximate background values", prompt)
        self.assertIn("preferable to refusing the whole task", prompt)
        self.assertIn("maps every requested part", prompt)
        self.assertIn("Never end at a bare 'unable to provide'", prompt)
        self.assertIn("Preserve the requested row schema", prompt)
        self.assertIn("distinct observed values", prompt)

    def test_constraint_reconciliation_preserves_missing_row_fields(self) -> None:
        client = _ConstraintSynthesisLLM([_INITIAL_PLAN])
        engine = SelfEvolvedEngine(client, SelfEvolvedConfig())
        state = {
            "task_id": "partial-schema-task",
            "task_prompt": "List each airplane with passenger capacity and range.",
            "run_index": 0,
            "round_index": 0,
            "discussion_index": 0,
            "dispatch_id": 0,
            "llm_client": client,
            "artifacts": [],
            "interaction_logs": [],
            "trace_events": [],
            "tools": [],
            "tool_records_log": [
                {
                    "tool_name": "airplanes.all",
                    "arguments": {},
                    "status": "completed",
                    "output": {"airplanes": [{"name": "Boeing 737"}]},
                }
            ],
        }

        artifact = engine._maybe_synthesize_final_answer(
            state=state,
            final_answer=BLOCKED_TEXT,
            selected_artifact_id="",
            vote_result={},
        )

        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(artifact["agent_id"], "constraint_reconciler")
        self.assertIn("passenger capacity: unknown", artifact["answer"])
        self.assertIn("range: unknown", artifact["answer"])
        self.assertEqual(client.calls[-2:], ["final_synthesizer", "constraint_reconciler"])
        self.assertIn("classification field", client.worker_prompts[-1])
        trace_actors = [event["actor"] for event in state["trace_payloads"]]
        self.assertIn("final_synthesizer", trace_actors)
        self.assertIn("constraint_reconciler", trace_actors)

    def test_constraint_reconciliation_is_not_used_for_complete_synthesis(self) -> None:
        engine = SelfEvolvedEngine(_ScriptedLLM([_INITIAL_PLAN]), SelfEvolvedConfig())

        self.assertFalse(
            engine._needs_constraint_preserving_revision(
                {"answer": "Paris is the capital of France.", "unresolved_issues": []}
            )
        )

    def test_no_tool_candidate_disagreement_requires_proof_reconciliation(self) -> None:
        engine = SelfEvolvedEngine(_ScriptedLLM([_INITIAL_PLAN]), SelfEvolvedConfig())
        state = {
            "tools": [],
            "artifacts": [
                {
                    "agent_id": "agent_0",
                    "stage_role": "worker",
                    "answer": r"\boxed{\frac{11}{6}}",
                    "summary": "The arithmetic gives 14/3.",
                },
                {
                    "agent_id": "agent_1",
                    "stage_role": "worker",
                    "answer": r"The arithmetic gives 14/3. \boxed{\frac{14}{3}}",
                    "summary": "The arithmetic gives 14/3.",
                },
            ],
        }

        self.assertTrue(engine._candidate_answer_disagreement(state))
        self.assertTrue(
            engine._needs_final_synthesis(
                r"\boxed{\frac{11}{6}}",
                vote_result={"tally": {"wrong": 1}},
                state=state,
            )
        )
        evidence = engine._collect_reasoning_candidate_evidence(state)
        self.assertEqual(len(evidence), 2)
        self.assertIn("candidate=agent_1", evidence[1])

    def test_conflicting_reasoning_gets_adversarial_proof_falsification(self) -> None:
        client = _ProofFalsificationLLM([_INITIAL_PLAN])
        engine = SelfEvolvedEngine(client, SelfEvolvedConfig())
        state = {
            "task_id": "supplementary-angle-task",
            "task_prompt": "Solve the diagram and return \\boxed{<answer>}.",
            "run_index": 0,
            "round_index": 0,
            "discussion_index": 0,
            "dispatch_id": 0,
            "llm_client": client,
            "artifacts": [
                {
                    "artifact_id": "a0",
                    "agent_id": "agent_0",
                    "stage_role": "worker",
                    "answer": r"The straight-angle equation gives 28. \boxed{28}",
                    "summary": "124 + x + x = 180.",
                },
                {
                    "artifact_id": "a1",
                    "agent_id": "agent_1",
                    "stage_role": "worker",
                    "answer": r"A supplementary-angle derivation gives 62. \boxed{62}",
                    "summary": "The base angles are 62 degrees.",
                },
            ],
            "interaction_logs": [],
            "trace_events": [],
            "tools": [],
            "tool_records_log": [],
        }

        artifact = engine._maybe_synthesize_final_answer(
            state=state,
            final_answer=r"\boxed{62}",
            selected_artifact_id="a1",
            vote_result={"tally": {"28": 1, "62": 1}},
        )

        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(artifact["agent_id"], "proof_falsifier")
        self.assertEqual(artifact["answer"], r"\boxed{28}")
        self.assertEqual(client.calls[-2:], ["final_synthesizer", "proof_falsifier"])
        verifier_prompt = client.worker_prompts[-1]
        self.assertIn("adversarial proof falsifier", verifier_prompt)
        self.assertIn("vector directions", verifier_prompt)
        self.assertIn("Draft to falsify", verifier_prompt)

    def test_unchanged_proof_with_dissent_gets_one_bounded_appeal(self) -> None:
        client = _ProofAppealLLM([_INITIAL_PLAN])
        engine = SelfEvolvedEngine(client, SelfEvolvedConfig())
        state = {
            "task_id": "ray-direction-task",
            "task_prompt": (
                "Solve the diagram and return \\boxed{<answer>}.\n"
                '[asy] label("$A$",(2,3),N); label("$B$",(4,0),S); '
                'label("$C$",(8,0),S); [/asy]'
            ),
            "run_index": 0,
            "round_index": 0,
            "discussion_index": 0,
            "dispatch_id": 0,
            "llm_client": client,
            "artifacts": [
                {
                    "artifact_id": "a0",
                    "agent_id": "agent_0",
                    "stage_role": "worker",
                    "answer": r"Using ordered rays gives \boxed{28}",
                    "summary": "The angle at B uses BA, not AB.",
                },
                {
                    "artifact_id": "a1",
                    "agent_id": "agent_1",
                    "stage_role": "worker",
                    "answer": r"Using the supplement gives \angle ABC=56 and \boxed{62}",
                    "summary": "The angle at B was treated as 56 degrees.",
                },
                {
                    "artifact_id": "a2",
                    "agent_id": "agent_2",
                    "stage_role": "worker",
                    "answer": r"A second ordered-ray derivation gives \boxed{28}",
                    "summary": "The obtuse angle is 124 degrees.",
                },
                {
                    "artifact_id": "a3",
                    "agent_id": "agent_3",
                    "stage_role": "worker",
                    "answer": r"A third calculation gives \boxed{52}",
                    "summary": "A distinct dissenting result.",
                },
            ],
            "interaction_logs": [],
            "trace_events": [],
            "tools": [],
            "tool_records_log": [],
        }

        artifact = engine._maybe_synthesize_final_answer(
            state=state,
            final_answer=r"\boxed{62}",
            selected_artifact_id="a1",
            vote_result={"tally": {"28": 1, "62": 1}},
        )

        self.assertIsNotNone(artifact)
        assert artifact is not None
        self.assertEqual(artifact["agent_id"], "proof_gate_selector")
        self.assertEqual(TurnExecutor._decision_answer_signature(artifact["answer"]), "28")
        self.assertEqual(
            client.calls[-4:],
            [
                "final_synthesizer",
                "proof_falsifier",
                "proof_appeal_reviewer",
                "proof_oracle_corrector",
            ],
        )
        correction_prompt = client.worker_prompts[-1]
        self.assertIn("Hard proof conflicts", correction_prompt)
        self.assertIn("∠ABC was claimed as 56°", correction_prompt)
        self.assertIn("smaller_angle≈123.690°", correction_prompt)
        trace_actors = [event["actor"] for event in state["trace_payloads"]]
        self.assertIn("proof_falsifier", trace_actors)
        self.assertIn("proof_appeal_reviewer", trace_actors)
        self.assertIn("proof_oracle_corrector", trace_actors)
        self.assertIn("proof_gate_selector", trace_actors)

    def test_coordinate_orientation_oracle_uses_ordered_vertex_rays(self) -> None:
        prompt = r"""
        AB = BC. Find x.
        [asy]
        label("$A$",(2,3),N);
        label("$B$",(4,0),S);
        label("$C$",(8,0),S);
        label("$124^{\circ}$",(2,3),SW);
        label("$x^{\circ}$",(4.5,3),S);
        [/asy]
        """
        checks = SelfEvolvedEngine._coordinate_orientation_checks(
            prompt,
            r"The disputed step claims $\angle ABC=56^\circ$ while another uses 124 degrees.",
        )

        self.assertIn("∠ABC", checks)
        self.assertIn("B->A=(-2.0, 3.0)", checks)
        self.assertIn("B->C=(4.0, 0.0)", checks)
        self.assertIn("dot=-8", checks)
        self.assertIn("smaller_angle≈123.690°", checks)
        self.assertIn("supplement distinctions", checks)
        conflicts = SelfEvolvedEngine._coordinate_claim_conflicts(
            r"The proof asserts that \angle ABC = 56^\circ$.", checks
        )
        self.assertEqual(len(conflicts), 1)
        self.assertIn("acute/obtuse mismatch", conflicts[0])

        engine = SelfEvolvedEngine(_ScriptedLLM([_INITIAL_PLAN]), SelfEvolvedConfig())
        state = {
            "task_prompt": prompt,
            "run_index": 0,
            "round_index": 0,
            "discussion_index": 0,
            "dispatch_id": 0,
            "trace_events": [],
        }
        certificate = engine._deterministic_triangle_angle_certificate(
            state=state,
            evidence=[
                r"Candidate one concludes $x = \angle BCA$ by alternate interior angles.",
                r"Candidate two independently finds x = angle BAC.",
            ],
            rejected_artifact={"artifact_id": "rejected", "answer": r"\boxed{62}"},
        )
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertEqual(certificate["answer"], r"\boxed{28}")
        self.assertEqual(certificate["agent_id"], "geometry_certificate")

    def test_equivalent_worker_final_forms_do_not_trigger_reconciliation(self) -> None:
        engine = SelfEvolvedEngine(_ScriptedLLM([_INITIAL_PLAN]), SelfEvolvedConfig())
        state = {
            "tools": [],
            "artifacts": [
                {
                    "agent_id": "agent_0",
                    "stage_role": "worker",
                    "answer": r"\boxed{(3, \pi/2)}",
                },
                {
                    "agent_id": "agent_1",
                    "stage_role": "worker",
                    "answer": r"\boxed{(3, \frac{\pi}{2})}",
                },
            ],
        }

        self.assertFalse(engine._candidate_answer_disagreement(state))

    def test_explicit_math_answer_canonicalizes_simple_fraction(self) -> None:
        self.assertEqual(
            SelfEvolvedEngine._canonicalize_explicit_math_answer(r"\boxed{(3, \pi/2)}"),
            r"\boxed{(3, \frac{\pi}{2})}",
        )

    def test_unsolicited_structured_identity_is_canonicalized_to_plain_name(self) -> None:
        self.assertEqual(
            SelfEvolvedEngine._canonicalize_structured_identity_answer(
                '{"first_name": "Kiran", "last_name": "Shah"}',
                task_prompt="Identify the person.",
            ),
            "Kiran Shah",
        )

    def test_requested_structured_identity_schema_is_preserved(self) -> None:
        answer = '{"first_name": "Kiran", "last_name": "Shah"}'

        self.assertEqual(
            SelfEvolvedEngine._canonicalize_structured_identity_answer(
                answer,
                task_prompt="Return JSON with first_name and last_name.",
            ),
            answer,
        )

    def test_verified_entity_spelling_alias_is_retained(self) -> None:
        state = {
            "tool_records_log": [
                {
                    "status": "completed",
                    "output": [
                        {
                            "docid": "d1",
                            "snippet": "The formal rendering in this source is Taj-Ul-Masajid.",
                        }
                    ],
                }
            ]
        }

        answer = SelfEvolvedEngine._append_verified_entity_aliases("Taj-ul-Masjid", state=state)

        self.assertEqual(answer, "Taj-ul-Masjid (also rendered Taj-Ul-Masajid)")

    def test_unrelated_document_title_is_not_added_as_alias(self) -> None:
        state = {
            "tool_records_log": [
                {
                    "status": "completed",
                    "output": {"snippet": "--- title: Faisal Mosque ---\nEvidence."},
                }
            ]
        }

        self.assertEqual(
            SelfEvolvedEngine._append_verified_entity_aliases("Taj-ul-Masjid", state=state),
            "Taj-ul-Masjid",
        )

    def test_verified_formal_institution_suffix_replaces_short_name(self) -> None:
        state = {
            "task_prompt": "Which institute matches all of these clues?",
            "tool_records_log": [
                {
                    "status": "completed",
                    "output": {
                        "snippet": (
                            "Article on Lady Shri Ram College for Women, delhilive.com. "
                            "The institution is in South Delhi."
                        )
                    },
                }
            ],
        }

        self.assertEqual(
            SelfEvolvedEngine._append_verified_entity_aliases("Lady Shri Ram College", state=state),
            "Lady Shri Ram College for Women",
        )

    def test_formal_suffix_is_not_added_when_task_requests_a_city(self) -> None:
        state = {
            "task_prompt": "Which city matches the clues?",
            "tool_records_log": [
                {
                    "status": "completed",
                    "output": {"snippet": "Boston University is located in Boston."},
                }
            ],
        }

        self.assertEqual(
            SelfEvolvedEngine._append_verified_entity_aliases("Boston", state=state),
            "Boston",
        )

    def test_stateful_topology_contracts_to_readers_then_one_committer(self) -> None:
        initial = spec_from_layout(build_layout(topology="fully_linked_debate", num_agents=3))

        transactional = SelfEvolvedEngine._transactional_star_spec(initial)

        root = transactional.group(transactional.root_group_id)
        self.assertEqual(root.pattern, "star")
        self.assertEqual(root.leader_id, "agent_0")
        self.assertEqual(set(root.member_ids), {"agent_0", "agent_1", "agent_2"})
        self.assertEqual(len(transactional.groups), 1)

        singleton = spec_from_layout(build_layout(topology="sas", num_agents=1))
        expanded = SelfEvolvedEngine._transactional_star_spec(singleton, minimum_agents=3)
        self.assertEqual(len(expanded.agents), 3)
        self.assertEqual(expanded.group(expanded.root_group_id).leader_id, "agent_0")

    def test_retrieval_repair_contracts_to_one_existing_specialist(self) -> None:
        engine = SelfEvolvedEngine(_ScriptedLLM([_INITIAL_PLAN]), SelfEvolvedConfig())
        initial = spec_from_layout(
            build_layout(topology="orchestrator_no_discussion", num_agents=3)
        )
        state = {
            "run_index": 0,
            "tool_records_log": [
                {"agent_id": "agent_2", "tool_name": "constraint_search"},
                {"agent_id": "agent_2", "tool_name": "get_document"},
            ],
            "trace_payloads": [],
        }

        rescued, payload = engine._apply_retrieval_rescue_mutation(
            state,
            initial,
            {"detected_modes": [{"mode": "insufficient_search_coverage"}]},
        )

        self.assertEqual(len(rescued.agents), 1)
        self.assertEqual(rescued.agents[0].agent_id, "agent_2")
        self.assertEqual(rescued.groups[0].pattern, "singleton")
        self.assertEqual(payload["source"], "resource_adaptive_retrieval_rescue")
        self.assertEqual(payload["added_agents"], [])

    def test_observed_list_recovery_is_non_exhaustive_and_keeps_limitation(self) -> None:
        answer = json.dumps(
            {
                "available_categories": "The categories endpoint failed to return a list.",
                "products": [{"id": 1, "name": "TV", "category": "electronics"}],
            }
        )
        state = {
            "tool_records_log": [
                {
                    "tool_name": "products.by_category",
                    "status": "completed",
                    "output": {
                        "products": [
                            {"id": 1, "name": "TV", "category": "electronics"},
                            {"id": 2, "name": "Radio", "category": "Electronics"},
                        ]
                    },
                }
            ]
        }

        recovered = SelfEvolvedEngine._recover_observed_list_fields(answer, state=state)
        payload = json.loads(recovered)

        self.assertEqual(payload["available_categories"]["observed_values"], ["electronics"])
        self.assertIn("non-exhaustive", payload["available_categories"]["scope"])
        self.assertIn("endpoint failed", payload["available_categories"]["endpoint_limitation"])

    def test_final_synthesis_collects_bounded_failed_tool_evidence(self) -> None:
        engine = SelfEvolvedEngine(_ScriptedLLM([_INITIAL_PLAN]), SelfEvolvedConfig())
        failed_record = {
            "agent_id": "agent_1",
            "round_index": 0,
            "tool_name": "catalog.view_item",
            "arguments": {"item_id": "56789"},
            "status": "error",
            "output": {"error": "service unavailable"},
        }

        evidence = engine._collect_tool_evidence_for_synthesis(
            {
                "tool_records_log": [
                    failed_record,
                    failed_record,
                    {
                        "tool_name": "inter_agent_send",
                        "arguments": {},
                        "status": "ok",
                        "output": "internal relay",
                    },
                ]
            }
        )

        self.assertEqual(len(evidence), 1)
        self.assertIn("tool=catalog.view_item", evidence[0])
        self.assertIn("status=error", evidence[0])
        self.assertIn("service unavailable", evidence[0])

    def test_constraint_search_candidates_feed_read_net_and_synthesis(self) -> None:
        engine = SelfEvolvedEngine(_ScriptedLLM([_INITIAL_PLAN]), SelfEvolvedConfig())
        documents = {
            "noise": {"text": "A broad chronology."},
            "decisive": {"text": "The decisive entity satisfies every clue."},
        }
        state = {
            "tools": [
                {
                    "name": "get_document",
                    "handler": lambda args: documents.get(args["docid"], {}),
                }
            ],
            "tool_records_log": [
                {
                    "tool_name": "constraint_search",
                    "arguments": {"queries": ["formation clue", "release clue"]},
                    "status": "completed",
                    "output": [
                        {"docid": "noise", "snippet": "chronology"},
                        {"docid": "decisive", "snippet": "satisfies release clue"},
                    ],
                }
            ],
        }

        evidence = engine._collect_tool_evidence_for_synthesis(state)
        documents_for_synthesis = engine._read_documents_for_synthesis(state)

        self.assertIn("tool=constraint_search", evidence[0])
        self.assertTrue(any("decisive" in item for item in evidence))
        self.assertTrue(any("decisive entity" in item for item in documents_for_synthesis))

    def test_final_synthesis_projects_distinct_structured_fields(self) -> None:
        engine = SelfEvolvedEngine(_ScriptedLLM([_INITIAL_PLAN]), SelfEvolvedConfig())

        evidence = engine._collect_tool_evidence_for_synthesis(
            {
                "tool_records_log": [
                    {
                        "tool_name": "products.by_category",
                        "arguments": {"category": "Electronics"},
                        "status": "completed",
                        "output": {
                            "products": [
                                {"id": 1, "name": "TV", "category": "electronics"},
                                {"id": 2, "name": "Radio", "category": "Electronics"},
                            ]
                        },
                    }
                ]
            }
        )

        self.assertEqual(len(evidence), 1)
        self.assertIn('"category": ["electronics", "Electronics"]', evidence[0])
        self.assertIn('"name": ["TV", "Radio"]', evidence[0])

    def test_explicit_output_contract_is_repaired_without_reference_answer(self) -> None:
        client = _OutputContractLLM()
        engine = SelfEvolvedEngine(
            client,
            SelfEvolvedConfig(max_turns=1, repair_budget=0, audit_mode="heuristic"),
        )
        task = _Task(
            task_id="format-contract",
            prompt=(
                "Convert (0,3) to polar coordinates. Give your final answer on the last "
                r"line in the form: \boxed{<answer>}."
            ),
            reference_answer=r"\left(3,\frac{\pi}{2}\right)",
        )
        result = engine.run(
            task=task,
            run_index=0,
            seed=1,
            spec=ExperimentSpec(
                topology="self_evolved",
                num_agents=1,
                rounds=1,
                communication_budget_per_agent=0,
                termination_consensus_mode="lexical",
                final_vote_mode="deterministic",
                benchmark_name="math500",
                enable_dynamic_roles=False,
            ),
            agent_types=["general"],
        )

        self.assertTrue(Math500Benchmark().evaluate(task, result.final_answer).success)
        self.assertIn("output_contract_enforcer", client.calls)
        repairs = result.run_metadata["self_evolved"]["contract_repairs"]
        self.assertEqual(len(repairs), 1)
        self.assertTrue(repairs[0]["accepted"])
        contract_log = next(
            item
            for item in result.run_metadata["interaction_logs"]
            if item.get("agent_id") == "output_contract_enforcer"
        )
        self.assertNotIn("reference_answer", json.dumps(contract_log["prompt_messages"]))

    def test_satisfied_output_contract_skips_formatter(self) -> None:
        contract = r"\boxed{<answer>}."
        self.assertTrue(
            SelfEvolvedEngine._output_contract_satisfied(r"Reasoning. \boxed{42}", contract)
        )
        self.assertFalse(SelfEvolvedEngine._output_contract_satisfied("42", contract))

    def test_repair_applied_exactly_once(self) -> None:
        client = _ScriptedLLM([_INITIAL_PLAN, _MUTATION_PLAN])
        engine = SelfEvolvedEngine(client, SelfEvolvedConfig())
        result = _run(engine)
        validate_trace_events(result.trace_events)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(len(meta["topology_spec_versions"]), 2)
        self.assertEqual(len(meta["topology_spec_versions"][0]["agents"]), 3)
        self.assertEqual(len(meta["topology_spec_versions"][1]["agents"]), 7)

        mutation = meta["mutation"]
        self.assertIsNotNone(mutation)
        self.assertEqual(len(mutation["mutation"]["ops"]), 2)
        self.assertEqual(len(mutation["added_agents"]), 4)
        self.assertIn("branch_collapse", mutation["mutation"]["target_failure_modes"])
        self.assertEqual(len(meta["short_term_playbook_entries"]), 2)
        self.assertEqual(meta["short_term_playbook_entries"][0]["scope"], "short_term")
        self.assertEqual(len(meta["context_state_versions"]), 2)
        self.assertEqual(meta["context_state_versions"][0]["reason"], "spawn")
        self.assertEqual(meta["context_state_versions"][1]["reason"], "mutation")
        self.assertIn("turn-level process memory", client.planner_prompts[1])
        self.assertIn("branch_collapse", client.planner_prompts[1])
        self.assertTrue(
            any("TRACE-BACKED REPAIR DIAGNOSIS" in prompt for prompt in client.worker_prompts)
        )

        revise_events = [
            event
            for event in result.trace_events
            if event.actor == "orchestrator"
            and event.event_type == "revise"
            and event.payload.get("node") == "apply_mutation"
        ]
        self.assertEqual(len(revise_events), 1)

        self.assertEqual(result.run_metadata["turns_executed"], 2)
        history = result.run_metadata["termination_history"]
        self.assertFalse(history[0]["should_stop"])
        self.assertTrue(history[-1]["should_stop"])
        # The repaired turn converges, so the loop stops on consensus.
        self.assertEqual(history[-1]["reason"], "consensus_reached")

        # Agents minted by the mutation actually executed in turn 1.
        actors = {event.actor for event in result.trace_events}
        for new_agent in ("agent_3", "agent_4", "agent_5", "agent_6"):
            self.assertIn(new_agent, actors)

        self.assertTrue(str(result.final_answer).strip())
        self.assertIn("42", result.final_answer)

    def test_repeated_decision_signature_suppresses_further_agent_growth(self) -> None:
        client = _ScriptedLLM([_INITIAL_PLAN, _MUTATION_PLAN, _MUTATION_PLAN], blocked_round0=False)
        engine = SelfEvolvedEngine(client, SelfEvolvedConfig())

        class _PersistentChallengeAuditor:
            @staticmethod
            def audit(*_args, **_kwargs) -> dict[str, Any]:
                return {
                    "repair_recommended": True,
                    "challenge_consensus": True,
                    "recommendation": "Add another independent verifier.",
                    "detected_modes": [
                        {
                            "mode": "persistent_correlated_consensus",
                            "detail": "The same decision remains after mutation.",
                        }
                    ],
                }

        engine.auditor = _PersistentChallengeAuditor()
        result = _run(engine)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(result.run_metadata["turns_executed"], 2)
        self.assertEqual(len(meta["mutations"]), 1)
        self.assertEqual(
            result.run_metadata["termination_history"][-1]["repair_suppressed"],
            "repeated_decision_signature",
        )
        self.assertEqual(
            result.run_metadata["termination_history"][-1]["reason"],
            "no_meaningful_change",
        )

    def test_consensus_stops_without_mutation(self) -> None:
        engine = SelfEvolvedEngine(
            _ScriptedLLM([_INITIAL_PLAN, _MUTATION_PLAN], blocked_round0=False),
            SelfEvolvedConfig(),
        )
        result = _run(engine)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(len(meta["topology_spec_versions"]), 1)
        self.assertIsNone(meta["mutation"])
        self.assertEqual(result.run_metadata["turns_executed"], 1)
        history = result.run_metadata["termination_history"]
        self.assertEqual(history[-1]["reason"], "consensus_reached")

        revise_events = [
            event for event in result.trace_events if event.payload.get("node") == "apply_mutation"
        ]
        self.assertEqual(revise_events, [])

    def test_unusable_mutation_response_uses_validated_repair_compiler(self) -> None:
        engine = SelfEvolvedEngine(
            _ScriptedLLM([_INITIAL_PLAN, "no mutation for you"]), SelfEvolvedConfig()
        )
        result = _run(engine)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(len(meta["topology_spec_versions"]), 2)
        self.assertIsNotNone(meta["mutation"])
        self.assertTrue(meta["mutation"]["used_fallback"])
        self.assertIn(
            "deterministic_repair_compiler",
            meta["mutation"]["mutation"]["rationale"],
        )
        skip_events = [
            event
            for event in result.trace_events
            if event.payload.get("node") == "mutation_skipped"
        ]
        self.assertEqual(skip_events, [])
        proposals = meta["mutation_proposals"]
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["fallback_reason"], "invalid_or_unparseable_mutation")

    def test_verifier_stage_alias_compiles_to_critic(self) -> None:
        alias_mutation = json.dumps(
            {
                "rationale": "Add an independent verifier.",
                "ops": [
                    {
                        "op": "add_agent",
                        "group_id": "g_root",
                        "structural_role": "verifier",
                        "stage_role": "verifier",
                    }
                ],
            }
        )
        engine = SelfEvolvedEngine(
            _ScriptedLLM([_INITIAL_PLAN, alias_mutation]), SelfEvolvedConfig()
        )

        result = _run(engine)

        mutation = result.run_metadata["self_evolved"]["mutation"]
        self.assertFalse(mutation["used_fallback"])
        self.assertEqual(
            mutation["mutation"]["ops"][0]["args"]["stage_role"],
            "critic",
        )
        final_spec = result.run_metadata["self_evolved"]["topology_spec_versions"][-1]
        self.assertEqual(final_spec["agents"][-1]["stage_role"], "critic")

    def test_two_repairs_within_default_budget(self) -> None:
        """max_turns=3 / repair_budget=3 defaults allow a second trace-backed repair."""
        second_mutation = json.dumps(
            {
                "rationale": "Coverage still thin; add a verifier to the root group.",
                "ops": [
                    {"op": "add_agent", "group_id": "g_root", "structural_role": "worker"},
                ],
            }
        )
        engine = SelfEvolvedEngine(
            _RoundScriptedLLM(
                [_INITIAL_PLAN, _MUTATION_PLAN, second_mutation], blocked_rounds={0, 1}
            ),
            SelfEvolvedConfig(),
        )
        result = _run(engine)
        validate_trace_events(result.trace_events)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(len(meta["topology_spec_versions"]), 3)
        self.assertEqual(len(meta["mutations"]), 2)
        self.assertIn(" -> ", meta["playbook_update_candidate"]["mutation_summary"])
        self.assertEqual(result.run_metadata["turns_executed"], 3)
        revise_events = [
            event
            for event in result.trace_events
            if event.actor == "orchestrator"
            and event.event_type == "revise"
            and event.payload.get("node") == "apply_mutation"
        ]
        self.assertEqual(len(revise_events), 2)
        history = result.run_metadata["termination_history"]
        self.assertFalse(history[0]["should_stop"])
        self.assertFalse(history[1]["should_stop"])
        self.assertTrue(history[-1]["should_stop"])

    def test_grounded_audit_can_challenge_consensus_and_preserves_turn_incumbents(self) -> None:
        client = _OpenSetAuditLLM([_INITIAL_PLAN, _MUTATION_PLAN])
        engine = SelfEvolvedEngine(client, SelfEvolvedConfig(audit_mode="hybrid"))

        result = _run(engine)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(result.run_metadata["turns_executed"], 2)
        self.assertEqual(result.run_metadata["termination_history"][0]["reason"], "audit_challenge")
        self.assertEqual(len(meta["topology_spec_versions"]), 2)
        self.assertEqual(meta["audit_reports"][0]["novel_modes"][0]["mode"], "correlated_consensus")
        self.assertGreaterEqual(len(meta["temporal_candidates"]), 2)

    def test_repair_budget_zero_disables_repair(self) -> None:
        engine = SelfEvolvedEngine(
            _ScriptedLLM([_INITIAL_PLAN, _MUTATION_PLAN]),
            SelfEvolvedConfig(repair_budget=0),
        )
        result = _run(engine)

        meta = result.run_metadata["self_evolved"]
        self.assertEqual(len(meta["topology_spec_versions"]), 1)
        self.assertIsNone(meta["mutation"])
        self.assertEqual(result.run_metadata["turns_executed"], 1)
        self.assertTrue(result.run_metadata["termination_history"][-1]["should_stop"])

    def test_audit_events_present(self) -> None:
        engine = SelfEvolvedEngine(
            _ScriptedLLM([_INITIAL_PLAN, _MUTATION_PLAN]), SelfEvolvedConfig()
        )
        result = _run(engine)
        audit_events = [event for event in result.trace_events if event.actor == "trace_auditor"]
        self.assertEqual(len(audit_events), 2)
        self.assertTrue(
            audit_events[0].payload["audit"]["repair_recommended"],
        )
        self.assertEqual(len(result.run_metadata["self_evolved"]["audit_reports"]), 2)


class TestRepairBudgetConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = SelfEvolvedConfig()
        self.assertEqual(cfg.max_turns, 5)
        self.assertEqual(cfg.repair_budget, 4)
        self.assertEqual(cfg.audit_mode, "hybrid")
        cfg.validate()

    def test_max_turns_bounds(self) -> None:
        SelfEvolvedConfig(max_turns=10).validate()
        with self.assertRaises(ValueError):
            SelfEvolvedConfig(max_turns=0).validate()
        with self.assertRaises(ValueError):
            SelfEvolvedConfig(max_turns=11).validate()

    def test_negative_repair_budget_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SelfEvolvedConfig(repair_budget=-1).validate()


class TestMutationOps(unittest.TestCase):
    def _star_spec(self):
        return spec_from_layout(build_layout(topology="orchestrator_no_discussion", num_agents=3))

    def test_expand_agent_to_group(self) -> None:
        spec = self._star_spec()
        mutation = TopologyMutation(
            rationale="split branch",
            target_failure_modes=("branch_collapse",),
            ops=(
                MutationOp(
                    op="expand_agent_to_group",
                    args={"agent_id": "agent_1", "pattern": "star", "num_subagents": 3},
                ),
            ),
        )
        mutated = mutation.apply(spec, max_agents=10)
        self.assertEqual(mutated.version, 1)
        self.assertEqual(len(mutated.agents), 6)
        subgroup = mutated.subgroup_of("agent_1")
        self.assertEqual(subgroup.pattern, "star")
        layout = mutated.to_layout()
        self.assertEqual(layout.parent_by_agent["agent_3"], "agent_1")

    def test_expand_rejects_double_expansion(self) -> None:
        spec = self._star_spec()
        op = MutationOp(
            op="expand_agent_to_group",
            args={"agent_id": "agent_1", "pattern": "star", "num_subagents": 2},
        )
        mutation = TopologyMutation(rationale="", target_failure_modes=(), ops=(op,))
        mutated = mutation.apply(spec, max_agents=10)
        with self.assertRaises(ValueError):
            mutation.apply(mutated, max_agents=10)

    def test_expand_rejects_budget_overflow(self) -> None:
        spec = self._star_spec()
        mutation = TopologyMutation(
            rationale="",
            target_failure_modes=(),
            ops=(
                MutationOp(
                    op="expand_agent_to_group",
                    args={"agent_id": "agent_1", "pattern": "voting", "num_subagents": 5},
                ),
            ),
        )
        with self.assertRaises(ValueError):
            mutation.apply(spec, max_agents=5)

    def test_set_group_pattern_adjusts_leader(self) -> None:
        spec = self._star_spec()
        to_debate = TopologyMutation(
            rationale="",
            target_failure_modes=(),
            ops=(
                MutationOp(
                    op="set_group_pattern", args={"group_id": "g_root", "pattern": "debate"}
                ),
            ),
        )
        mutated = to_debate.apply(spec, max_agents=10)
        root = mutated.group("g_root")
        self.assertEqual(root.pattern, "debate")
        self.assertIsNone(root.leader_id)

    def test_add_agent_with_critic_stage(self) -> None:
        spec = self._star_spec()
        mutation = TopologyMutation(
            rationale="add validator",
            target_failure_modes=("missing_validator",),
            ops=(
                MutationOp(
                    op="add_agent",
                    args={
                        "group_id": "g_root",
                        "structural_role": "verifier",
                        "stage_role": "critic",
                    },
                ),
            ),
        )
        mutated = mutation.apply(spec, max_agents=10)
        self.assertEqual(len(mutated.agents), 4)
        new_agent = mutated.agent("agent_3")
        self.assertEqual(new_agent.stage_role, "critic")
        self.assertIn("agent_3", mutated.group("g_root").member_ids)

    def test_add_agent_promotes_singleton_root_to_star(self) -> None:
        spec = spec_from_layout(build_layout(topology="sas", num_agents=1))
        root = spec.group("g_root")
        self.assertEqual(root.pattern, "singleton")
        original_member = root.member_ids[0]
        mutation = TopologyMutation(
            rationale="add capacity",
            target_failure_modes=("premature_consensus",),
            ops=(MutationOp(op="add_agent", args={"group_id": "g_root"}),),
        )
        mutated = mutation.apply(spec, max_agents=10)  # must not raise
        promoted = mutated.group("g_root")
        self.assertEqual(promoted.pattern, "star")
        self.assertEqual(promoted.leader_id, original_member)
        self.assertEqual(len(promoted.member_ids), 2)

    def test_edge_ops_round_trip(self) -> None:
        spec = self._star_spec()
        add = TopologyMutation(
            rationale="",
            target_failure_modes=(),
            ops=(MutationOp(op="add_edge", args={"src": "agent_1", "dst": "agent_2"}),),
        )
        mutated = add.apply(spec, max_agents=10)
        self.assertIn(("agent_1", "agent_2"), mutated.extra_edges)
        self.assertIn("agent_2", mutated.to_layout().adjacency["agent_1"])

        remove = TopologyMutation(
            rationale="",
            target_failure_modes=(),
            ops=(MutationOp(op="remove_edge", args={"src": "agent_2", "dst": "agent_1"}),),
        )
        back = remove.apply(mutated, max_agents=10)
        self.assertEqual(back.extra_edges, ())

    def test_set_context_policy(self) -> None:
        spec = self._star_spec()
        mutation = TopologyMutation(
            rationale="widen evidence",
            target_failure_modes=("evidence_lost_before_synthesis",),
            ops=(
                MutationOp(
                    op="set_context_policy",
                    args={"agent_id": "agent_0", "evidence_access": "global"},
                ),
            ),
        )
        mutated = mutation.apply(spec, max_agents=10)
        self.assertEqual(mutated.agent("agent_0").context.evidence_access, "global")
        # Other agents untouched.
        self.assertEqual(mutated.agent("agent_1").context.evidence_access, "own")


if __name__ == "__main__":
    unittest.main()

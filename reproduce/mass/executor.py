from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .interfaces import BenchmarkExample
from .models import AgentPromptBundle, ExampleExecution, ExecutionTurn, MASSCandidate

ModelCallback = Callable[[str, str, BenchmarkExample, dict[str, Any]], str]
ExecutionCallback = Callable[[str, BenchmarkExample, dict[str, Any]], str]


def default_model_callback(
    role: str,
    prompt_text: str,
    example: BenchmarkExample,
    context: dict[str, Any],
) -> str:
    """Deterministic fallback so the executor skeleton works without an LLM."""

    del prompt_text
    summary = f"{role} response for {example.example_id}"
    if context.get("candidate_answers"):
        summary += f" | peers={len(context['candidate_answers'])}"
    if context.get("execution_feedback"):
        summary += " | with execution feedback"
    return summary


@dataclass
class MASSCandidateExecutor:
    """Paper-shaped execution skeleton for one MASS candidate.

    This is intentionally lightweight. It encodes the order and role semantics
    of the paper's blocks without depending on the production MAS runtime.
    """

    model_callback: ModelCallback = default_model_callback
    execution_callback: ExecutionCallback | None = None

    def run_candidate(
        self, candidate: MASSCandidate, example: BenchmarkExample
    ) -> ExampleExecution:
        turns: list[ExecutionTurn] = []
        context: dict[str, Any] = {
            "candidate_answers": [],
            "example_metadata": dict(example.metadata),
            "execution_feedback": [],
            "reflections": [],
            "debate_history": [],
        }

        for summarize_round in range(candidate.workflow.summarize_rounds):
            if "summarize" not in candidate.prompts:
                break
            summary_text = self._call_block(
                role=f"summarize_{summarize_round}",
                prompt=candidate.prompts["summarize"],
                example=example,
                context=context,
            )
            context["summary"] = summary_text
            turns.append(
                ExecutionTurn(
                    step="summarize",
                    role=f"summarize_{summarize_round}",
                    content=summary_text,
                    metadata={"summarize_round": summarize_round},
                )
            )

        predictor_prompt = candidate.prompts["predictor"]
        candidate_answers: list[str] = []
        predictor_count = candidate.workflow.aggregate_width
        if candidate.workflow.debate_rounds > 0:
            predictor_count = max(predictor_count, 2)
        for predictor_index in range(max(1, predictor_count)):
            answer = self._call_block(
                role=f"predictor_{predictor_index}",
                prompt=predictor_prompt,
                example=example,
                context=context,
            )
            candidate_answers.append(answer)
            turns.append(
                ExecutionTurn(
                    step="predict",
                    role=f"predictor_{predictor_index}",
                    content=answer,
                    metadata={"predictor_index": predictor_index},
                )
            )
        context["candidate_answers"] = list(candidate_answers)

        if candidate.workflow.execute_enabled and "execute" in candidate.prompts:
            execution_feedback, execution_source = self._call_execute_block(
                prompt=candidate.prompts["execute"],
                example=example,
                context=context,
            )
            context["execution_feedback"].append(execution_feedback)
            turns.append(
                ExecutionTurn(
                    step="execute",
                    role="execute",
                    content=execution_feedback,
                    metadata={"execution_source": execution_source},
                )
            )

        if candidate.workflow.reflect_rounds > 0 and "reflect" in candidate.prompts:
            for reflect_round in range(candidate.workflow.reflect_rounds):
                reflected = self._call_block(
                    role=f"reflect_{reflect_round}",
                    prompt=candidate.prompts["reflect"],
                    example=example,
                    context=context,
                )
                context["reflections"].append(reflected)
                turns.append(
                    ExecutionTurn(
                        step="reflect",
                        role=f"reflect_{reflect_round}",
                        content=reflected,
                        metadata={"reflect_round": reflect_round},
                    )
                )
                if self._reflection_accepts_answer(reflected):
                    break
                refined_answers: list[str] = []
                current_answers = list(context["candidate_answers"])
                for answer_index, _answer in enumerate(current_answers):
                    refine_role = f"refine_{reflect_round}"
                    if len(current_answers) > 1:
                        refine_role = f"{refine_role}_agent_{answer_index}"
                    refined = self._call_block(
                        role=refine_role,
                        prompt=predictor_prompt,
                        example=example,
                        context={**context, "refine_answer_index": answer_index},
                    )
                    refined_answers.append(refined)
                    turns.append(
                        ExecutionTurn(
                            step="refine",
                            role=refine_role,
                            content=refined,
                            metadata={
                                "reflect_round": reflect_round,
                                "answer_index": answer_index,
                            },
                        )
                    )
                context["candidate_answers"] = refined_answers

        if candidate.workflow.debate_rounds > 0 and "debate" in candidate.prompts:
            for debate_round in range(candidate.workflow.debate_rounds):
                debated_answers: list[str] = []
                for agent_index, _answer in enumerate(context["candidate_answers"]):
                    debated = self._call_block(
                        role=f"debate_{debate_round}_agent_{agent_index}",
                        prompt=candidate.prompts["debate"],
                        example=example,
                        context={**context, "debate_agent_index": agent_index},
                    )
                    debated_answers.append(debated)
                    turns.append(
                        ExecutionTurn(
                            step="debate",
                            role=f"debate_{debate_round}_agent_{agent_index}",
                            content=debated,
                            metadata={
                                "debate_round": debate_round,
                                "agent_index": agent_index,
                            },
                        )
                    )
                context["debate_history"].append(tuple(debated_answers))
                context["candidate_answers"] = debated_answers

        aggregate_prompt = candidate.prompts.get("aggregate", predictor_prompt)
        aggregate_used = len(context["candidate_answers"]) > 1
        if aggregate_used:
            final_answer = self._call_block(
                role="aggregate",
                prompt=aggregate_prompt,
                example=example,
                context=context,
            )
            turns.append(ExecutionTurn(step="aggregate", role="aggregate", content=final_answer))
        else:
            final_answer = str(context["candidate_answers"][0])
        return ExampleExecution(
            example_id=example.example_id,
            workflow=candidate.workflow,
            final_answer=final_answer,
            turns=tuple(turns),
            metadata={
                "active_blocks": candidate.workflow.active_blocks(),
                "candidate_answer_count": len(context["candidate_answers"]),
                "execution_feedback_count": len(context["execution_feedback"]),
                "execution_feedback_source": context.get("execution_feedback_source", ""),
                "reflection_count": len(context["reflections"]),
                "debate_round_count": len(context["debate_history"]),
                "aggregate_used": aggregate_used,
            },
        )

    def _call_block(
        self,
        *,
        role: str,
        prompt: AgentPromptBundle,
        example: BenchmarkExample,
        context: dict[str, Any],
    ) -> str:
        prompt_text = self._render_prompt_text(prompt, context=context)
        return str(self.model_callback(role, prompt_text, example, context))

    def _call_execute_block(
        self,
        *,
        prompt: AgentPromptBundle,
        example: BenchmarkExample,
        context: dict[str, Any],
    ) -> tuple[str, str]:
        current_answer = str(context.get("candidate_answers", [""])[-1])
        if self.execution_callback is not None:
            feedback = str(self.execution_callback(current_answer, example, context))
            context["execution_feedback_source"] = "execution_callback"
            return feedback, "execution_callback"

        metadata_feedback = self._metadata_execution_feedback(example, current_answer)
        if metadata_feedback:
            context["execution_feedback_source"] = "example_metadata"
            return metadata_feedback, "example_metadata"

        context["execution_feedback_source"] = "model_callback"
        return (
            self._call_block(
                role="execute",
                prompt=prompt,
                example=example,
                context=context,
            ),
            "model_callback",
        )

    def _metadata_execution_feedback(
        self,
        example: BenchmarkExample,
        current_answer: str,
    ) -> str:
        metadata = dict(example.metadata or {})
        sub_steps = metadata.get("sub_steps")
        if isinstance(sub_steps, list) and sub_steps:
            return self._scicode_metadata_feedback(sub_steps, metadata, current_answer)

        public_tests = self._first_present(
            metadata,
            ("public_tests", "test_cases", "unit_tests", "tests", "examples"),
        )
        if public_tests:
            return (
                "External execution feedback from benchmark metadata: "
                f"candidate answer length={len(current_answer)}; "
                f"public test signal={self._short_metadata(public_tests)}"
            )

        tool_definitions = self._first_present(
            metadata,
            ("tool_definitions", "tools", "available_tools", "function_schemas"),
        )
        if tool_definitions:
            return (
                "External tool feedback from benchmark metadata: "
                f"candidate answer length={len(current_answer)}; "
                f"available tools={self._short_metadata(tool_definitions)}"
            )
        return ""

    def _scicode_metadata_feedback(
        self,
        sub_steps: list[Any],
        metadata: dict[str, Any],
        current_answer: str,
    ) -> str:
        total_tests = 0
        headers: list[str] = []
        for step in sub_steps:
            if not isinstance(step, dict):
                continue
            test_cases = step.get("test_cases") or []
            if isinstance(test_cases, list):
                total_tests += len(test_cases)
            elif test_cases:
                total_tests += 1
            header = str(step.get("function_header") or "").strip()
            if header:
                headers.append(header.splitlines()[0])
        dependencies = str(metadata.get("required_dependencies") or "").strip()
        return (
            "External execution feedback from SciCode public tests: "
            f"candidate answer length={len(current_answer)}; "
            f"sub_steps={len(sub_steps)}; public_test_cases={total_tests}; "
            f"function_headers={self._short_metadata(headers)}; "
            f"required_dependencies={self._short_metadata(dependencies or 'none')}"
        )

    def _first_present(self, metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            value = metadata.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    def _short_metadata(self, value: Any, *, limit: int = 500) -> str:
        text = str(value).strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _render_prompt_text(self, prompt: AgentPromptBundle, *, context: dict[str, Any]) -> str:
        parts = [prompt.system_instruction.strip()]
        signature = self._render_prompt_signature(prompt)
        if signature:
            parts.append(signature)
        if prompt.exemplar.strip():
            parts.append(prompt.exemplar.strip())
        if context.get("summary"):
            parts.append(f"Summary context: {context['summary']}")
        if context.get("candidate_answers"):
            parts.append(
                "Peer answers:\n" + "\n".join(str(item) for item in context["candidate_answers"])
            )
        if context.get("debate_history"):
            debate_lines = [
                f"Round {round_index}: " + " | ".join(str(item) for item in answers)
                for round_index, answers in enumerate(context["debate_history"])
            ]
            parts.append("Debate history:\n" + "\n".join(debate_lines))
        if context.get("execution_feedback"):
            parts.append(
                "Execution feedback:\n"
                + "\n".join(str(item) for item in context["execution_feedback"])
            )
        if context.get("reflections"):
            parts.append("Reflections:\n" + "\n".join(str(item) for item in context["reflections"]))
        return "\n\n".join(part for part in parts if part)

    def _render_prompt_signature(self, prompt: AgentPromptBundle) -> str:
        if not prompt.input_fields and not prompt.output_fields and not prompt.output_contract:
            return ""
        lines = ["Follow the following format."]
        for field in prompt.input_fields:
            lines.append(f"{field}: ${{{self._field_token(field)}}}")
        for field in prompt.output_fields:
            lines.append(f"{field}: ${{{self._field_token(field)}}}")
        if prompt.output_contract:
            lines.append(f"Output contract: {prompt.output_contract}")
        return "\n".join(lines)

    def _field_token(self, field: str) -> str:
        return (
            str(field)
            .strip()
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )

    def _reflection_accepts_answer(self, reflection: str) -> bool:
        normalized = reflection.lower()
        positive = ("correct", "true", "valid", "sound")
        negative = ("incorrect", "false", "invalid", "wrong", "revise")
        return any(token in normalized for token in positive) and not any(
            token in normalized for token in negative
        )

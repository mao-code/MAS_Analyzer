from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from benchmark.base import BenchmarkTask
from descriptor.schema import TraceEvent
from MAS.llm import OpenRouterLLMClient
from MAS.runner import MASRunResult

from .executor import MASSCandidateExecutor
from .interfaces import BenchmarkExample
from .models import ExampleExecution, MASSCandidate


class MASSRuntimeRunner:
    """Benchmark-facing runner for one MASS candidate.

    The MASS framework owns topology and prompt search, while repo benchmarks
    own task execution. Benchmarks call this through their normal ``run()``
    method, so benchmark-specific tools, environments, and run metadata stay on
    the same path as the production runners.
    """

    def __init__(
        self,
        *,
        candidate: MASSCandidate,
        llm_client: OpenRouterLLMClient,
        model_agent_type: str = "default",
        temperature: float = 0.7,
    ) -> None:
        self.candidate = candidate
        self.llm_client = llm_client
        self.model_agent_type = model_agent_type
        self.temperature = float(temperature)

    def run_task(
        self,
        task: BenchmarkTask,
        run_index: int,
        seed: int,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = 8,
        benchmark_name: str | None = None,
        **_: Any,
    ) -> MASRunResult:
        trace_events: list[TraceEvent] = []
        llm_results: list[dict[str, Any]] = []

        def model_callback(
            role: str,
            prompt_text: str,
            example: BenchmarkExample,
            context: dict[str, Any],
        ) -> str:
            benchmark_key = str(benchmark_name or "").lower()
            system_content = (
                "You are executing one MASS workflow block. Follow the block role, "
                "use available tools when they help, and return only this block's output."
            )
            if benchmark_key == "browsecomp":
                system_content += (
                    " This is BrowseComp: use retrieval tools for evidence before answering. "
                    "Never refuse because context was not provided; the search tool is the context source. "
                    "Do not finish with 'insufficient evidence'; return the most likely answer string."
                )
            elif benchmark_key == "workbench":
                system_content += (
                    " This is WorkBench: use tools to complete the requested action. "
                    "Do not finish with 'insufficient evidence'; either perform the action with tools "
                    "or give the concrete no-op result required by tool evidence."
                )
            elif benchmark_key == "stabletoolbench":
                system_content += (
                    " This is StableToolBench: call the provided API tools with arguments grounded "
                    "in the user query and the tool schema. If the user omits a required argument "
                    "but the tool schema provides a default or example value, use that value instead "
                    "of asking the user for clarification. Prefer exact schema defaults before "
                    "inventing alternate spellings or identifiers. If one tool call fails, try a "
                    "more exact query-grounded or schema-default call before giving a final answer. "
                    "Do not answer from general knowledge after a tool/cache failure. The final "
                    "Answer must contain the requested result, not just a plan to call tools or a "
                    "request for more input. If any tool call returns a dict, list, table, or "
                    "descriptive text, extract concrete returned fields/items into the final answer. "
                    "Do not say the tool would provide data, that a result set was returned, or that "
                    "details were unavailable without first listing the useful concrete fields that "
                    "were actually returned."
                )
            messages = [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": (
                        f"Block role: {role}\n"
                        f"Task:\n{example.prompt}\n\n"
                        f"MASS block prompt and context:\n{prompt_text}"
                    ),
                },
            ]
            started_wall = time.time()
            started_perf = time.perf_counter()
            result = self.llm_client.generate(
                prompt=messages,
                agent_type=self.model_agent_type,
                task_id=f"{benchmark_name or 'benchmark'}:{example.example_id}:{role}",
                run_index=run_index,
                agent_id=role,
                tools=list(tools or []),
                max_tool_iterations=max(1, int(max_tool_iterations)),
                temperature=self.temperature,
            )
            latency_ms = max((time.perf_counter() - started_perf) * 1000.0, 1.0)
            event = TraceEvent(
                timestamp_start=started_wall,
                timestamp_end=time.time(),
                actor=role,
                event_type="act",
                payload={
                    "node": "mass",
                    "role": role,
                    "workflow": self.candidate.workflow.to_payload(),
                    "text": result.text,
                    "model": result.model,
                    "mock_used": result.mock_used,
                    "llm_metadata": dict(result.metadata),
                    "tool_calls": list(result.tool_calls),
                },
                token_in=int(result.token_in),
                token_out=int(result.token_out),
                latency_ms=latency_ms,
                cost_usd=float(result.cost_usd),
                state_id=f"{benchmark_name or 'benchmark'}:{example.example_id}:{run_index}:{role}:act",
            )
            trace_events.append(event)
            for idx, record in enumerate(result.tool_calls):
                tool_name = str(record.get("tool_name") or "")
                arguments = (
                    record.get("arguments") if isinstance(record.get("arguments"), dict) else {}
                )
                trace_events.append(
                    TraceEvent(
                        timestamp_start=event.timestamp_start,
                        timestamp_end=event.timestamp_start,
                        actor=role,
                        event_type="tool_call",
                        payload={
                            "node": "mass",
                            "role": role,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "status": record.get("status"),
                        },
                        token_in=0,
                        token_out=0,
                        latency_ms=0.001,
                        cost_usd=0.0,
                        state_id=(
                            f"{benchmark_name or 'benchmark'}:{example.example_id}:"
                            f"{run_index}:{role}:tool_call:{idx}"
                        ),
                    )
                )
                trace_events.append(
                    TraceEvent(
                        timestamp_start=event.timestamp_start,
                        timestamp_end=event.timestamp_start,
                        actor="system",
                        event_type="tool_result",
                        payload={
                            "node": "mass",
                            "role": role,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "status": record.get("status"),
                            "error": record.get("error"),
                            "output": record.get("output"),
                        },
                        token_in=0,
                        token_out=0,
                        latency_ms=0.001,
                        cost_usd=0.0,
                        state_id=(
                            f"{benchmark_name or 'benchmark'}:{example.example_id}:"
                            f"{run_index}:{role}:tool_result:{idx}"
                        ),
                    )
                )
            payload = {
                "role": role,
                "model": result.model,
                "token_in": result.token_in,
                "token_out": result.token_out,
                "cost_usd": result.cost_usd,
                "mock_used": result.mock_used,
                "metadata": dict(result.metadata),
                "tool_calls": list(result.tool_calls),
            }
            llm_results.append(payload)
            context.setdefault("llm_calls", []).append(payload)
            return result.text

        example = BenchmarkExample(
            example_id=str(task.task_id),
            prompt=task.prompt,
            reference_answer=task.reference_answer,
            metadata=dict(task.metadata or {}),
        )
        execution = MASSCandidateExecutor(model_callback=model_callback).run_candidate(
            self.candidate,
            example,
        )
        final_answer = self._extract_final_answer(
            execution.final_answer,
            benchmark_name=str(benchmark_name or ""),
        )
        metadata = self._run_metadata(
            task=task,
            run_index=run_index,
            seed=seed,
            tools=tools or [],
            execution=execution,
            trace_events=trace_events,
            llm_results=llm_results,
        )
        metadata["raw_final_answer"] = execution.final_answer
        metadata["final_answer_extracted"] = final_answer != execution.final_answer
        return MASRunResult(
            final_answer=final_answer,
            trace_events=trace_events,
            run_metadata=metadata,
        )

    def _extract_final_answer(self, text: str, *, benchmark_name: str) -> str:
        answer = str(text or "").strip()
        if not answer:
            return answer
        benchmark_key = benchmark_name.lower()
        if benchmark_key == "plancraft":
            action_match = re.search(
                r"(?ims)(?:^|\n)\s*Action\s*:\s*(.+?)(?=\n\s*(?:Observation|Reasoning|Answer|Action)\s*:|\Z)",
                answer,
            )
            if action_match:
                action = self._normalize_plancraft_action(action_match.group(1))
                if action:
                    return action
            action = self._normalize_plancraft_action(answer)
            if action:
                return action
        matches = list(
            re.finditer(
                r"(?ims)(?:^|\n)\s*Answer\s*:\s*(.+?)(?=\n\s*(?:Question|Task|Reasoning|Feedback|Correctness|Evidence|Search plan|Action|Observation)\s*:|\Z)",
                answer,
            )
        )
        if matches:
            extracted = matches[-1].group(1).strip()
            if extracted:
                return extracted
        bracket_match = re.search(r"(?is)<answer>\s*(.+?)\s*</answer>", answer)
        if bracket_match:
            extracted = bracket_match.group(1).strip()
            if extracted:
                return extracted
        return answer

    @staticmethod
    def _normalize_plancraft_action(text: str) -> str:
        answer = str(text or "").strip()
        if not answer:
            return ""
        answer = re.sub(r"(?is)^Action\s*:\s*", "", answer).strip()
        answer = answer.splitlines()[0].strip()
        answer = re.sub(r"\s+\((?:Note|Reasoning|Explanation)\s*:.*\)\s*$", "", answer, flags=re.I)
        answer = re.sub(r"\s+(?:Note|Reasoning|Explanation)\s*:.*$", "", answer, flags=re.I)
        impossible = re.match(r"(?is)^(impossible)(?:\s*:\s*(.*?))?\s*$", answer)
        if impossible:
            reason = (impossible.group(2) or "").strip()
            return f"impossible: {reason}" if reason else "impossible"
        search = re.match(r"(?is)^(search\s*:\s*[a-z0-9_]+)", answer)
        if search:
            return re.sub(r"\s+", " ", search.group(1)).strip()
        move = re.match(
            r"(?is)^(move\s*:\s*from\s*\[[^\]]+\]\s*to\s*\[[^\]]+\]\s*with\s*quantity\s*\d+)",
            answer,
        )
        if move:
            return re.sub(r"\s+", " ", move.group(1)).strip()
        return answer

    def _run_metadata(
        self,
        *,
        task: BenchmarkTask,
        run_index: int,
        seed: int,
        tools: list[dict[str, Any]],
        execution: ExampleExecution,
        trace_events: list[TraceEvent],
        llm_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        tool_counts: Counter[str] = Counter()
        retrieved_docids: set[str] = set()
        for result in llm_results:
            for record in result.get("tool_calls", []):
                if not isinstance(record, dict):
                    continue
                tool_name = str(record.get("tool_name") or "")
                if tool_name:
                    tool_counts[tool_name] += 1
                self._collect_docids(record.get("output"), retrieved_docids)
        return {
            "mass_reproduce": True,
            "task_id": str(task.task_id),
            "run_index": int(run_index),
            "seed": int(seed),
            "workflow": self.candidate.workflow.to_payload(),
            "candidate_stage": self.candidate.stage,
            "prompt_blocks": sorted(self.candidate.prompts.keys()),
            "execution": {
                "example_id": execution.example_id,
                "turn_count": len(execution.turns),
                "metadata": dict(execution.metadata),
                "turns": [
                    {
                        "step": turn.step,
                        "role": turn.role,
                        "content": turn.content,
                        "metadata": dict(turn.metadata),
                    }
                    for turn in execution.turns
                ],
            },
            "tool_definitions": self._tool_definition_names(tools),
            "tool_call_counts": dict(tool_counts),
            "tool_calls_total": int(sum(tool_counts.values())),
            "retrieved_docids": sorted(retrieved_docids),
            "messages_sent_total": len(llm_results),
            "rounds_configured": len(execution.turns),
            "rounds_executed": len(execution.turns),
            "final_reason": "mass_workflow_completed",
            "trace_event_count": len(trace_events),
        }

    def _tool_definition_names(self, tools: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and function.get("name"):
                names.append(str(function["name"]))
            elif isinstance(tool, dict) and tool.get("name"):
                names.append(str(tool["name"]))
        return names

    def _collect_docids(self, value: Any, output: set[str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {"docid", "doc_id", "id"} and item not in (None, ""):
                    output.add(str(item))
                else:
                    self._collect_docids(item, output)
        elif isinstance(value, list):
            for item in value:
                self._collect_docids(item, output)

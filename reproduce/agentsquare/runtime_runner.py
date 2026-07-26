from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any

from benchmark.base import BenchmarkTask
from descriptor.schema import TraceEvent
from MAS.llm import OpenRouterLLMClient
from MAS.runner import MASRunResult

from .models import AgentSquareConfig, AgentSquareSpec


class AgentSquareRuntimeRunner:
    """Benchmark-facing runner for one AgentSquare module combination.

    Benchmarks keep ownership of environments, tools, side effects, and
    evaluation.  This runner implements the AgentSquare four-module workflow
    as prompt-level modules over the repo's existing OpenRouter client.
    """

    def __init__(
        self,
        *,
        spec: AgentSquareSpec,
        llm_client: OpenRouterLLMClient,
        config: AgentSquareConfig | None = None,
    ) -> None:
        self.spec = spec
        self.llm_client = llm_client
        self.config = config or AgentSquareConfig()

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
        memory_text = ""
        plan_text = ""

        if self.spec.memory is not None:
            memory_text = self._call_module(
                module_type="memory",
                instruction=self.spec.memory.prompt,
                task=task,
                run_index=run_index,
                seed=seed,
                benchmark_name=benchmark_name,
                tools=[],
                max_tool_iterations=max_tool_iterations,
                context={"phase": "retrieve"},
                trace_events=trace_events,
                llm_results=llm_results,
            )

        if self.spec.planning is not None:
            plan_text = self._call_module(
                module_type="planning",
                instruction=self.spec.planning.prompt,
                task=task,
                run_index=run_index,
                seed=seed,
                benchmark_name=benchmark_name,
                tools=[],
                max_tool_iterations=max_tool_iterations,
                context={"memory": memory_text},
                trace_events=trace_events,
                llm_results=llm_results,
            )

        final_answer = self._run_reasoning(
            task=task,
            run_index=run_index,
            seed=seed,
            benchmark_name=benchmark_name,
            tools=list(tools or []) if self.spec.tooluse is not None else [],
            max_tool_iterations=max_tool_iterations,
            context={
                "plan": plan_text,
                "memory": memory_text,
                "tooluse": self.spec.tooluse.prompt if self.spec.tooluse else "",
            },
            trace_events=trace_events,
            llm_results=llm_results,
        )
        metadata = self._run_metadata(
            task=task,
            run_index=run_index,
            seed=seed,
            tools=tools or [],
            trace_events=trace_events,
            llm_results=llm_results,
            plan_text=plan_text,
            memory_text=memory_text,
        )
        return MASRunResult(
            final_answer=final_answer, trace_events=trace_events, run_metadata=metadata
        )

    def _run_reasoning(
        self,
        *,
        task: BenchmarkTask,
        run_index: int,
        seed: int,
        benchmark_name: str | None,
        tools: list[dict[str, Any]],
        max_tool_iterations: int,
        context: dict[str, Any],
        trace_events: list[TraceEvent],
        llm_results: list[dict[str, Any]],
    ) -> str:
        sample_count = int(self.spec.reasoning.metadata.get("samples", 1))
        sample_count = max(
            1, min(sample_count, int(self.config.max_reasoning_samples or sample_count))
        )
        raw_answers: list[str] = []
        extracted_answers: list[str] = []
        for sample_index in range(sample_count):
            sample_context = dict(context)
            if sample_count > 1:
                sample_context["self_consistency_sample_index"] = sample_index
                sample_context["self_consistency_sample_count"] = sample_count
            raw_answer = self._call_module(
                module_type="reasoning",
                instruction=self._reasoning_instruction(benchmark_name),
                task=task,
                run_index=run_index,
                seed=seed + sample_index,
                benchmark_name=benchmark_name,
                tools=tools,
                max_tool_iterations=max_tool_iterations,
                context=sample_context,
                trace_events=trace_events,
                llm_results=llm_results,
            )
            raw_answers.append(raw_answer)
            extracted_answers.append(
                self._extract_final_answer(raw_answer, benchmark_name=str(benchmark_name or ""))
            )
        if sample_count == 1:
            return extracted_answers[0]
        return self._majority_answer(extracted_answers, raw_answers)

    def _call_module(
        self,
        *,
        module_type: str,
        instruction: str,
        task: BenchmarkTask,
        run_index: int,
        seed: int,
        benchmark_name: str | None,
        tools: list[dict[str, Any]],
        max_tool_iterations: int,
        context: dict[str, Any],
        trace_events: list[TraceEvent],
        llm_results: list[dict[str, Any]],
    ) -> str:
        module = self.spec.module_for(module_type)
        module_name = module.name if module else "None"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are executing an AgentSquare module. AgentSquare composes "
                    "Planning, Reasoning, Tool Use, and Memory modules with uniform "
                    "interfaces. Follow this module role and return only useful output."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Benchmark: {benchmark_name or 'benchmark'}\n"
                    f"Module type: {module_type}\n"
                    f"Module name: {module_name}\n"
                    f"Instruction:\n{instruction}\n\n"
                    f"Task:\n{task.prompt}\n\n"
                    f"Context:\n{context}"
                ),
            },
        ]
        started_wall = time.time()
        started_perf = time.perf_counter()
        try:
            result = self.llm_client.generate(
                prompt=messages,
                agent_type=self.config.model_agent_type,
                task_id=f"{benchmark_name or 'benchmark'}:{task.task_id}:{module_type}",
                run_index=run_index,
                agent_id=f"agentsquare_{module_type}",
                tools=tools,
                max_tool_iterations=max(1, int(max_tool_iterations)),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception as exc:
            error_text = f"AGENTSQUARE_LLM_ERROR: {type(exc).__name__}: {exc}"
            event = TraceEvent(
                timestamp_start=started_wall,
                timestamp_end=time.time(),
                actor=f"agentsquare_{module_type}",
                event_type="error",
                payload={
                    "node": "agentsquare",
                    "module_type": module_type,
                    "module_name": module_name,
                    "spec": self.spec.to_payload(),
                    "text": error_text,
                    "model": self.config.model_agent_type,
                    "mock_used": False,
                    "llm_metadata": {"error_type": type(exc).__name__},
                    "tool_calls": [],
                    "error": str(exc),
                },
                token_in=0,
                token_out=0,
                latency_ms=max((time.perf_counter() - started_perf) * 1000.0, 1.0),
                cost_usd=0.0,
                state_id=f"{benchmark_name or 'benchmark'}:{task.task_id}:{run_index}:{module_type}:error",
            )
            trace_events.append(event)
            llm_results.append(
                {
                    "module_type": module_type,
                    "module_name": module_name,
                    "model": self.config.model_agent_type,
                    "token_in": 0,
                    "token_out": 0,
                    "cost_usd": 0.0,
                    "mock_used": False,
                    "text": error_text,
                    "tool_call_count": 0,
                    "error": str(exc),
                }
            )
            return error_text
        event = TraceEvent(
            timestamp_start=started_wall,
            timestamp_end=time.time(),
            actor=f"agentsquare_{module_type}",
            event_type="act",
            payload={
                "node": "agentsquare",
                "module_type": module_type,
                "module_name": module_name,
                "spec": self.spec.to_payload(),
                "text": result.text,
                "model": result.model,
                "mock_used": result.mock_used,
                "llm_metadata": dict(result.metadata),
                "tool_calls": list(result.tool_calls),
            },
            token_in=int(result.token_in),
            token_out=int(result.token_out),
            latency_ms=max((time.perf_counter() - started_perf) * 1000.0, 1.0),
            cost_usd=float(result.cost_usd),
            state_id=f"{benchmark_name or 'benchmark'}:{task.task_id}:{run_index}:{module_type}:act",
        )
        trace_events.append(event)
        for idx, record in enumerate(result.tool_calls):
            trace_events.extend(self._tool_trace_events(event, module_type, record, idx))
        llm_results.append(
            {
                "module_type": module_type,
                "module_name": module_name,
                "model": result.model,
                "token_in": result.token_in,
                "token_out": result.token_out,
                "cost_usd": result.cost_usd,
                "mock_used": result.mock_used,
                "metadata": dict(result.metadata),
                "tool_calls": list(result.tool_calls),
            }
        )
        return result.text

    def _reasoning_instruction(self, benchmark_name: str | None) -> str:
        instruction = self.spec.reasoning.prompt
        if self.spec.tooluse is not None:
            instruction += "\n" + self.spec.tooluse.prompt
        key = str(benchmark_name or "").lower()
        if key == "plancraft":
            instruction += "\nReturn the next valid action only, or impossible with a reason."
        elif key == "browsecomp":
            instruction += (
                "\nUse search evidence when available and end with the exact answer string."
            )
        elif key == "workbench":
            instruction += (
                "\nUse tools to complete required side effects; avoid extra side effects."
            )
        elif key == "stabletoolbench":
            instruction += "\nCall the provided API tools and summarize concrete returned fields."
        return instruction

    def _tool_trace_events(
        self,
        parent: TraceEvent,
        module_type: str,
        record: dict[str, Any],
        idx: int,
    ) -> list[TraceEvent]:
        tool_name = str(record.get("tool_name") or "")
        arguments = record.get("arguments") if isinstance(record.get("arguments"), dict) else {}
        base_state = f"{parent.state_id}:tool:{idx}"
        return [
            TraceEvent(
                timestamp_start=parent.timestamp_start,
                timestamp_end=parent.timestamp_start,
                actor=f"agentsquare_{module_type}",
                event_type="tool_call",
                payload={
                    "node": "agentsquare",
                    "module_type": module_type,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "status": record.get("status"),
                },
                token_in=0,
                token_out=0,
                latency_ms=0.001,
                cost_usd=0.0,
                state_id=f"{base_state}:call",
            ),
            TraceEvent(
                timestamp_start=parent.timestamp_start,
                timestamp_end=parent.timestamp_start,
                actor="system",
                event_type="tool_result",
                payload={
                    "node": "agentsquare",
                    "module_type": module_type,
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
                state_id=f"{base_state}:result",
            ),
        ]

    def _extract_final_answer(self, text: str, *, benchmark_name: str) -> str:
        answer = str(text or "").strip()
        if not answer:
            return answer
        if benchmark_name.lower() == "plancraft":
            first_line = answer.splitlines()[0].strip()
            action = re.sub(r"(?is)^Action\s*:\s*", "", first_line).strip()
            return action or first_line
        matches = list(
            re.finditer(
                r"(?ims)(?:^|\n)\s*Answer\s*:\s*(.+?)(?=\n\s*(?:Task|Reasoning|Action|Observation|Feedback)\s*:|\Z)",
                answer,
            )
        )
        if matches:
            extracted = matches[-1].group(1).strip()
            if extracted:
                return extracted
        bracket_match = re.search(r"(?is)<answer>\s*(.+?)\s*</answer>", answer)
        if bracket_match:
            return bracket_match.group(1).strip()
        return answer

    def _majority_answer(self, extracted_answers: list[str], raw_answers: list[str]) -> str:
        normalized = [self._normalize_answer_for_vote(answer) for answer in extracted_answers]
        counts = Counter(value for value in normalized if value)
        if not counts:
            return extracted_answers[0] if extracted_answers else ""
        winner, _ = counts.most_common(1)[0]
        for answer, normalized_answer in zip(extracted_answers, normalized, strict=False):
            if normalized_answer == winner:
                return answer
        return raw_answers[0] if raw_answers else ""

    @staticmethod
    def _normalize_answer_for_vote(answer: str) -> str:
        return re.sub(r"\s+", " ", str(answer or "").strip().lower())

    def _run_metadata(
        self,
        *,
        task: BenchmarkTask,
        run_index: int,
        seed: int,
        tools: list[dict[str, Any]],
        trace_events: list[TraceEvent],
        llm_results: list[dict[str, Any]],
        plan_text: str,
        memory_text: str,
    ) -> dict[str, Any]:
        tool_counts: Counter[str] = Counter()
        for result in llm_results:
            for record in result.get("tool_calls", []):
                name = str(record.get("tool_name") or "")
                if name:
                    tool_counts[name] += 1
        return {
            "agentsquare_reproduce": True,
            "task_id": str(task.task_id),
            "run_index": int(run_index),
            "seed": int(seed),
            "spec": self.spec.to_payload(),
            "plan_text": plan_text,
            "memory_text": memory_text,
            "tool_definitions": self._tool_definition_names(tools),
            "tool_call_counts": dict(tool_counts),
            "tool_calls_total": int(sum(tool_counts.values())),
            "messages_sent_total": len(llm_results),
            "rounds_configured": len(llm_results),
            "rounds_executed": len(llm_results),
            "final_reason": "agentsquare_workflow_completed",
            "trace_event_count": len(trace_events),
        }

    @staticmethod
    def _tool_definition_names(tools: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and function.get("name"):
                names.append(str(function["name"]))
            elif isinstance(tool, dict) and tool.get("name"):
                names.append(str(tool["name"]))
        return names

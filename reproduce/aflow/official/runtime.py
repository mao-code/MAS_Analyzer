from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from descriptor.schema import TraceEvent
from MAS.llm import LLMResult, OpenRouterLLMClient
from MAS.runner import MASRunResult

from . import prompts


@dataclass
class AFlowCallRecord:
    operator: str
    result: LLMResult
    started_wall: float
    latency_ms: float


@dataclass
class AFlowExecutionContext:
    llm_client: OpenRouterLLMClient
    agent_type: str
    task_id: str
    run_index: int
    temperature: float
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tool_iterations: int = 8
    allow_mock: bool = False
    checkpoint_path: Path | None = None
    records: list[AFlowCallRecord] = field(default_factory=list)
    seq: int = 0

    def next_agent_id(self, operator: str) -> str:
        self.seq += 1
        return f"{operator.lower()}_{self.seq}"


class OfficialLLM:
    """Small OpenRouter-backed replacement for the official AsyncLLM."""

    def __init__(self, context: AFlowExecutionContext) -> None:
        self.context = context
        self.total_cost = 0.0

    async def __call__(self, prompt: str, *, operator: str = "Custom") -> str:
        await asyncio.sleep(0)
        started_wall = time.time()
        started_perf = time.perf_counter()
        result = self.context.llm_client.generate(
            prompt=[
                {
                    "role": "system",
                    "content": (
                        "You are executing an AFlow workflow operator. Return only the requested "
                        "operator output."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            agent_type=self.context.agent_type,
            task_id=self.context.task_id,
            run_index=self.context.run_index,
            agent_id=self.context.next_agent_id(operator),
            tools=self.context.tools,
            max_tool_iterations=self.context.max_tool_iterations,
            temperature=self.context.temperature,
        )
        if result.mock_used and not self.context.allow_mock:
            raise RuntimeError(
                f"Live OpenRouter AFlow run expected, but mock fallback was used for "
                f"task_id={self.context.task_id} operator={operator}."
            )
        self.total_cost += float(result.cost_usd)
        self.context.records.append(
            AFlowCallRecord(
                operator=operator,
                result=result,
                started_wall=started_wall,
                latency_ms=max((time.perf_counter() - started_perf) * 1000.0, 1.0),
            )
        )
        write_execution_checkpoint(self.context)
        return result.text

    async def call_xml(self, prompt: str, fields: list[str], *, operator: str) -> dict[str, str]:
        field_examples = "\n".join(f"<{field}>{field}</{field}>" for field in fields)
        text = await self(
            f"{prompt}\n\nReturn XML with exactly these fields:\n{field_examples}",
            operator=operator,
        )
        parsed = parse_xml_fields(text, fields)
        if parsed:
            return parsed
        return {fields[-1]: text}

    def get_usage_summary(self) -> dict[str, Any]:
        token_in = sum(record.result.token_in for record in self.context.records)
        token_out = sum(record.result.token_out for record in self.context.records)
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": token_in,
            "total_output_tokens": token_out,
            "total_tokens": token_in + token_out,
            "call_count": len(self.context.records),
        }


def parse_xml_fields(text: str, fields: list[str]) -> dict[str, str]:
    found = {key: value.strip() for key, value in re.findall(r"<(\w+)>(.*?)</\1>", text, re.DOTALL)}
    return {field: found.get(field, "") for field in fields if found.get(field, "")}


class Operators:
    def __init__(self, llm: OfficialLLM) -> None:
        self.llm = llm

    async def custom(self, input: str, instruction: str) -> dict[str, str]:
        text = await self.llm(f"{instruction}{input}", operator="Custom")
        return {"response": text}

    async def answer_generate(self, input: str) -> dict[str, str]:
        return await self.llm.call_xml(
            prompts.ANSWER_GENERATION_PROMPT.format(input=input),
            ["thought", "answer"],
            operator="AnswerGenerate",
        )

    async def sc_ensemble(self, solutions: list[str], problem: str) -> dict[str, str]:
        if not solutions:
            return {"response": ""}
        answer_mapping: dict[str, int] = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            letter = chr(65 + index)
            answer_mapping[letter] = index
            solution_text += f"{letter}:\n{solution}\n\n"
        response = await self.llm.call_xml(
            prompts.SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text),
            ["thought", "solution_letter"],
            operator="ScEnsemble",
        )
        letter = str(response.get("solution_letter", "A")).strip().upper()[:1] or "A"
        return {"response": solutions[answer_mapping.get(letter, 0)]}

    async def review(self, problem: str, solution: str) -> dict[str, str]:
        return await self.llm.call_xml(
            prompts.REVIEW_PROMPT.format(problem=problem, solution=solution),
            ["thought", "review_result", "feedback"],
            operator="Review",
        )

    async def revise(self, problem: str, solution: str, feedback: str) -> dict[str, str]:
        return await self.llm.call_xml(
            prompts.REVISE_PROMPT.format(problem=problem, solution=solution, feedback=feedback),
            ["thought", "solution"],
            operator="Revise",
        )

    async def format(self, problem: str, solution: str) -> dict[str, str]:
        text = await self.llm(
            prompts.FORMAT_PROMPT.format(problem_description=problem, solution=solution),
            operator="Format",
        )
        return {"solution": text.strip()}


class OfficialWorkflowBase:
    def __init__(
        self,
        *,
        name: str,
        llm: OfficialLLM,
        dataset: str,
        prompt_custom: ModuleType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = llm
        self.prompt_custom = prompt_custom
        self.operator_custom = Operators(llm)
        self.custom = self.operator_custom.custom
        self.answer_generate = self.operator_custom.answer_generate
        self.sc_ensemble = self.operator_custom.sc_ensemble
        self.review = self.operator_custom.review
        self.revise = self.operator_custom.revise
        self.format = self.operator_custom.format

    async def __call__(self, problem: str) -> tuple[str, float]:
        raise NotImplementedError


INITIAL_GRAPH = """class Workflow(OfficialWorkflowBase):
    async def __call__(self, problem: str):
        solution = await self.custom(input=problem, instruction="")
        return solution["response"], self.llm.get_usage_summary()["total_cost"]
"""

INITIAL_PROMPT = """# Custom prompts used by generated AFlow workflows.
"""

OPERATOR_DESCRIPTIONS = {
    "Custom": {
        "description": "Generates anything based on customized input and instruction.",
        "interface": "custom(input: str, instruction: str) -> dict with key 'response' of type str",
    },
    "AnswerGenerate": {
        "description": "Generates a thought and concise answer.",
        "interface": "answer_generate(input: str) -> dict with keys 'thought' and 'answer'",
    },
    "ScEnsemble": {
        "description": "Uses self-consistency to select the best solution from a list.",
        "interface": "sc_ensemble(solutions: list[str], problem: str) -> dict with key 'response'",
    },
    "Review": {
        "description": "Reviews a solution and returns correctness feedback.",
        "interface": "review(problem: str, solution: str) -> dict with keys 'review_result' and 'feedback'",
    },
    "Revise": {
        "description": "Revises a solution using feedback.",
        "interface": "revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'",
    },
    "Format": {
        "description": "Formats a solution into a concise final answer.",
        "interface": "format(problem: str, solution: str) -> dict with key 'solution'",
    },
}


def ensure_initial_workspace(workflows_dir: Path, operators: list[str]) -> None:
    round_dir = workflows_dir / "round_1"
    template_dir = workflows_dir / "template"
    round_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "__init__.py").write_text("", encoding="utf-8")
    if not (round_dir / "graph.py").exists():
        write_graph_file(round_dir / "graph.py", INITIAL_GRAPH)
    if not (round_dir / "prompt.py").exists():
        (round_dir / "prompt.py").write_text(INITIAL_PROMPT, encoding="utf-8")
    description = {
        name: OPERATOR_DESCRIPTIONS[name] for name in operators if name in OPERATOR_DESCRIPTIONS
    }
    (template_dir / "operator.json").write_text(json.dumps(description, indent=2), encoding="utf-8")


def write_graph_file(path: Path, graph_code: str) -> None:
    header = "from reproduce.aflow.official.runtime import OfficialWorkflowBase\n\n"
    text = graph_code.strip()
    if "OfficialWorkflowBase" not in text.split("class Workflow", 1)[0]:
        text = header + text
    path.write_text(text + "\n", encoding="utf-8")


def load_workflow_class(round_dir: Path) -> type[OfficialWorkflowBase]:
    graph_path = round_dir / "graph.py"
    module_name = f"aflow_generated_{abs(hash(str(graph_path.resolve())))}_{round_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, graph_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import generated graph: {graph_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    workflow_class = module.Workflow
    if not issubclass(workflow_class, OfficialWorkflowBase):
        raise TypeError(f"Generated Workflow must subclass OfficialWorkflowBase: {graph_path}")
    return workflow_class


def load_prompt_module(round_dir: Path) -> ModuleType:
    prompt_path = round_dir / "prompt.py"
    module_name = f"aflow_prompt_{abs(hash(str(prompt_path.resolve())))}_{round_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, prompt_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import generated prompt: {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class OfficialAFlowRunnerAdapter:
    def __init__(
        self,
        *,
        workflow_class: type[OfficialWorkflowBase],
        prompt_module: ModuleType,
        llm_client: OpenRouterLLMClient,
        benchmark_name: str,
        agent_type: str,
        temperature: float,
        allow_mock: bool,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.workflow_class = workflow_class
        self.prompt_module = prompt_module
        self.llm_client = llm_client
        self.benchmark_name = benchmark_name
        self.agent_type = agent_type
        self.temperature = temperature
        self.allow_mock = allow_mock
        self.checkpoint_path = checkpoint_path

    def run_task(
        self,
        task: Any,
        run_index: int,
        seed: int,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = 8,
        benchmark_name: str | None = None,
        **_: Any,
    ) -> MASRunResult:
        context = AFlowExecutionContext(
            llm_client=self.llm_client,
            agent_type=self.agent_type,
            task_id=f"{benchmark_name or self.benchmark_name}:{task.task_id}",
            run_index=run_index,
            temperature=self.temperature,
            tools=list(tools or []),
            max_tool_iterations=max_tool_iterations,
            allow_mock=self.allow_mock,
            checkpoint_path=self.checkpoint_path,
        )
        llm = OfficialLLM(context)
        workflow = self.workflow_class(
            name=benchmark_name or self.benchmark_name,
            llm=llm,
            dataset=benchmark_name or self.benchmark_name,
            prompt_custom=self.prompt_module,
        )
        result = asyncio.run(workflow(task.prompt))
        if isinstance(result, tuple):
            final_answer = str(result[0])
        else:
            final_answer = str(result)
        trace_events = trace_events_from_context(context)
        run_metadata = run_metadata_from_context(context)
        return MASRunResult(
            final_answer=final_answer, trace_events=trace_events, run_metadata=run_metadata
        )


def trace_events_from_context(context: AFlowExecutionContext) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for idx, record in enumerate(context.records):
        result = record.result
        node = f"aflow_official:{record.operator}"
        state_prefix = f"{context.task_id}:{context.run_index}:{idx}:{record.operator}"
        events.append(
            TraceEvent(
                timestamp_start=record.started_wall,
                timestamp_end=record.started_wall + record.latency_ms / 1000.0,
                actor=record.operator.lower(),
                event_type="act",
                payload={
                    "node": node,
                    "operator": record.operator,
                    "text": result.text,
                    "model": result.model,
                    "mock_used": result.mock_used,
                    "llm_metadata": dict(result.metadata),
                },
                token_in=int(result.token_in),
                token_out=int(result.token_out),
                latency_ms=float(record.latency_ms),
                cost_usd=float(result.cost_usd),
                state_id=f"{state_prefix}:act",
            )
        )
        for tool_idx, tool_record in enumerate(result.tool_calls):
            tool_name = str(tool_record.get("tool_name") or "")
            if not tool_name:
                continue
            call_start = (
                record.started_wall + record.latency_ms / 1000.0 + (tool_idx * 2 + 1) * 1e-6
            )
            events.append(
                TraceEvent(
                    timestamp_start=call_start,
                    timestamp_end=call_start + 1e-6,
                    actor=record.operator.lower(),
                    event_type="tool_call",
                    payload={
                        "node": node,
                        "tool_name": tool_name,
                        "arguments": tool_record.get("arguments", {}),
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=0.001,
                    cost_usd=0.0,
                    state_id=f"{state_prefix}:tool_call:{tool_idx}",
                )
            )
            events.append(
                TraceEvent(
                    timestamp_start=call_start + 1e-6,
                    timestamp_end=call_start + 2e-6,
                    actor="system",
                    event_type="tool_result",
                    payload={
                        "node": node,
                        "tool_name": tool_name,
                        "status": str(tool_record.get("status", "")),
                        "error": tool_record.get("error"),
                        "output": tool_record.get("output"),
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=0.001,
                    cost_usd=0.0,
                    state_id=f"{state_prefix}:tool_result:{tool_idx}",
                )
            )
    return events


def write_execution_checkpoint(context: AFlowExecutionContext) -> None:
    if context.checkpoint_path is None:
        return
    checkpoint_path = context.checkpoint_path
    trace_path = checkpoint_path.with_suffix(".trace.json")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trace_events = [event.to_dict() for event in trace_events_from_context(context)]
    payload = {
        "status": "running",
        "task_id": context.task_id,
        "run_index": context.run_index,
        "operator_calls_completed": len(context.records),
        "trace_path": str(trace_path.resolve()),
        "run_metadata": run_metadata_from_context(context),
        "calls": [
            {
                "operator": record.operator,
                "text": record.result.text,
                "model": record.result.model,
                "token_in": record.result.token_in,
                "token_out": record.result.token_out,
                "cost_usd": record.result.cost_usd,
                "mock_used": record.result.mock_used,
                "llm_metadata": dict(record.result.metadata),
                "tool_calls": list(record.result.tool_calls),
                "latency_ms": record.latency_ms,
            }
            for record in context.records
        ],
    }
    trace_path.write_text(
        json.dumps({"trace": trace_events}, indent=2, default=str), encoding="utf-8"
    )
    checkpoint_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def run_metadata_from_context(context: AFlowExecutionContext) -> dict[str, Any]:
    tool_call_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    hit_output_limit_count = 0
    for record in context.records:
        for tool_record in record.result.tool_calls:
            tool_name = str(tool_record.get("tool_name") or "")
            if tool_name:
                tool_call_counts[tool_name] += 1
        metadata = dict(record.result.metadata)
        reasons = metadata.get("finish_reasons")
        if not isinstance(reasons, list):
            reasons = [metadata.get("finish_reason")]
        for reason in reasons:
            if reason:
                finish_reason_counts[str(reason)] += 1
        if bool(metadata.get("hit_output_limit")):
            hit_output_limit_count += 1
    return {
        "aflow_reproduce": True,
        "aflow_official_adapter": True,
        "tool_call_counts": dict(tool_call_counts),
        "tool_calls_total": int(sum(tool_call_counts.values())),
        "finish_reason_counts": dict(finish_reason_counts),
        "hit_output_limit_count": hit_output_limit_count,
        "messages_sent_total": len(context.records),
        "rounds_configured": len(context.records),
        "rounds_executed": len(context.records),
        "final_reason": "workflow_completed",
    }


def select_round(items: list[dict[str, Any]], *, sample: int) -> dict[str, Any]:
    if not items:
        raise ValueError("No rounds available for AFlow selection.")
    unique: dict[int, dict[str, Any]] = {}
    for item in sorted(items, key=lambda row: float(row["score"]), reverse=True):
        unique.setdefault(int(item["round"]), item)
    candidates = list(unique.values())[: max(1, sample)]
    if len(candidates) == 1:
        return candidates[0]
    scores = [float(item["score"]) * 100.0 for item in candidates]
    max_score = max(scores)
    weights = [
        0.3 * (1.0 / len(scores)) + 0.7 * pow(2.718281828, 0.2 * (score - max_score))
        for score in scores
    ]
    total = sum(weights)
    return random.choices(candidates, weights=[weight / total for weight in weights], k=1)[0]

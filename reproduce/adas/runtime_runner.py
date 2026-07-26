from __future__ import annotations

import builtins
import json
import random
import re
import string
import threading
import time
from collections import Counter, namedtuple
from typing import Any

from benchmark.base import BenchmarkTask
from descriptor.schema import TraceEvent
from MAS.llm import OpenRouterLLMClient
from MAS.runner import MASRunResult

from .models import ADASConfig, ADASSolution

Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])

FORMAT_INST = lambda request_keys: (
    "Reply EXACTLY with the following JSON format.\n"
    f"{request_keys}\n"
    "Do not miss fields. Return a well-formed JSON object."
)
ROLE_DESC = lambda role: f"You are a {role}."


def random_id(length: int = 4) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


class ADASRuntimeRunner:
    """Executes one Meta Agent Search generated `forward` function on a benchmark task."""

    def __init__(
        self,
        *,
        solution: ADASSolution,
        llm_client: OpenRouterLLMClient,
        config: ADASConfig | None = None,
    ) -> None:
        self.solution = solution
        self.llm_client = llm_client
        self.config = config or ADASConfig()
        self.forward = compile_forward(solution.code)

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
        random.seed(seed)
        trace_events: list[TraceEvent] = []
        llm_results: list[dict[str, Any]] = []
        runtime = _AgentSystemRuntime(
            solution=self.solution,
            llm_client=self.llm_client,
            config=self.config,
            task=task,
            run_index=run_index,
            seed=seed,
            benchmark_name=benchmark_name or "benchmark",
            tools=list(tools or []),
            max_tool_iterations=max_tool_iterations,
            trace_events=trace_events,
            llm_results=llm_results,
        )
        task_info = Info("task", "User", task.prompt, -1)
        try:
            output = self.forward(runtime, task_info)
            final_answer = output.content if isinstance(output, Info) else str(output or "")
        except Exception as exc:
            final_answer = f"ADAS_RUNTIME_ERROR: {type(exc).__name__}: {exc}"
            trace_events.append(
                TraceEvent(
                    timestamp_start=time.time(),
                    timestamp_end=time.time(),
                    actor="adas_runtime",
                    event_type="error",
                    payload={
                        "node": "adas",
                        "solution": self.solution.to_payload(),
                        "text": final_answer,
                        "error": str(exc),
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=0.001,
                    cost_usd=0.0,
                    state_id=f"{benchmark_name or 'benchmark'}:{task.task_id}:{run_index}:adas:error",
                )
            )
        metadata = _metadata(
            solution=self.solution,
            task=task,
            run_index=run_index,
            seed=seed,
            trace_events=trace_events,
            llm_results=llm_results,
            tools=tools or [],
        )
        return MASRunResult(
            final_answer=_extract_final_answer(final_answer, benchmark_name or ""),
            trace_events=trace_events,
            run_metadata=metadata,
        )


class _AgentSystemRuntime:
    def __init__(
        self,
        *,
        solution: ADASSolution,
        llm_client: OpenRouterLLMClient,
        config: ADASConfig,
        task: BenchmarkTask,
        run_index: int,
        seed: int,
        benchmark_name: str,
        tools: list[dict[str, Any]],
        max_tool_iterations: int,
        trace_events: list[TraceEvent],
        llm_results: list[dict[str, Any]],
    ) -> None:
        self.solution = solution
        self.llm_client = llm_client
        self.config = config
        self.task = task
        self.run_index = run_index
        self.seed = seed
        self.benchmark_name = benchmark_name
        self.tools = tools
        self.max_tool_iterations = max_tool_iterations
        self.trace_events = trace_events
        self.llm_results = llm_results


class LLMAgentBase:
    def __init__(
        self,
        output_fields: list[str],
        agent_name: str,
        role: str = "helpful assistant",
        model: str | None = None,
        temperature: float = 0.5,
    ) -> None:
        self.output_fields = list(output_fields)
        self.agent_name = agent_name
        self.role = role
        self.model = model
        self.temperature = float(temperature)
        self.id = random_id()

    def generate_prompt(self, input_infos: list[Any], instruction: str) -> tuple[str, str]:
        descriptions = {
            key: (
                f"Your {key}. Return only the final answer content."
                if "answer" in key.lower()
                else f"Your {key}."
            )
            for key in self.output_fields
        }
        system_prompt = ROLE_DESC(self.role) + "\n\n" + FORMAT_INST(descriptions)
        input_text = ""
        for input_info in input_infos:
            if not isinstance(input_info, Info):
                continue
            field_name, author, content, iteration_idx = input_info
            if author == self.__repr__():
                author += " (yourself)"
            if field_name == "task":
                input_text += f"# Your Task:\n{content}\n\n"
            elif iteration_idx != -1:
                input_text += f"### {field_name} #{iteration_idx + 1} by {author}:\n{content}\n\n"
            else:
                input_text += f"### {field_name} by {author}:\n{content}\n\n"
        return system_prompt, input_text + instruction

    def __call__(self, input_infos: list[Any], instruction: str, iteration_idx: int = -1) -> list[Info]:
        runtime = getattr(_THREAD_LOCAL, "runtime", None)
        if runtime is None:
            raise RuntimeError("LLMAgentBase called outside ADAS runtime")
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        started_wall = time.time()
        started_perf = time.perf_counter()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            result = runtime.llm_client.generate(
                prompt=messages,
                agent_type=runtime.config.model_agent_type,
                task_id=f"{runtime.benchmark_name}:{runtime.task.task_id}:{self.agent_name}",
                run_index=runtime.run_index,
                agent_id=f"adas_{self.agent_name}_{self.id}",
                tools=runtime.tools,
                max_tool_iterations=max(1, int(runtime.max_tool_iterations)),
                temperature=self.temperature,
                max_tokens=runtime.config.max_tokens,
            )
            parsed = _parse_json_fields(result.text, self.output_fields)
            text_for_trace = result.text
            metadata = dict(result.metadata)
            tool_calls = list(result.tool_calls)
            token_in = int(result.token_in)
            token_out = int(result.token_out)
            cost_usd = float(result.cost_usd)
            model = result.model
            mock_used = bool(result.mock_used)
            error = None
        except Exception as exc:
            parsed = {field: "" for field in self.output_fields}
            text_for_trace = f"ADAS_LLM_ERROR: {type(exc).__name__}: {exc}"
            metadata = {"error_type": type(exc).__name__}
            tool_calls = []
            token_in = token_out = 0
            cost_usd = 0.0
            model = runtime.config.model_agent_type
            mock_used = False
            error = str(exc)
        event = TraceEvent(
            timestamp_start=started_wall,
            timestamp_end=time.time(),
            actor=f"adas_{self.agent_name}_{self.id}",
            event_type="act" if error is None else "error",
            payload={
                "node": "adas",
                "solution_name": runtime.solution.name,
                "agent_name": self.agent_name,
                "role": self.role,
                "output_fields": list(self.output_fields),
                "text": text_for_trace,
                "parsed": dict(parsed),
                "model": model,
                "mock_used": mock_used,
                "llm_metadata": metadata,
                "tool_calls": tool_calls,
                "error": error,
            },
            token_in=token_in,
            token_out=token_out,
            latency_ms=max((time.perf_counter() - started_perf) * 1000.0, 1.0),
            cost_usd=cost_usd,
            state_id=(
                f"{runtime.benchmark_name}:{runtime.task.task_id}:{runtime.run_index}:"
                f"adas:{self.agent_name}:{self.id}"
            ),
        )
        runtime.trace_events.append(event)
        runtime.trace_events.extend(_tool_trace_events(event, self.agent_name, tool_calls))
        runtime.llm_results.append(
            {
                "agent_name": self.agent_name,
                "role": self.role,
                "model": model,
                "token_in": token_in,
                "token_out": token_out,
                "cost_usd": cost_usd,
                "mock_used": mock_used,
                "metadata": metadata,
                "tool_calls": tool_calls,
                "text": text_for_trace,
                "parsed": dict(parsed),
                "error": error,
            }
        )
        return [
            Info(field, self.__repr__(), str(parsed.get(field, "")), iteration_idx)
            for field in self.output_fields
        ]

    def __repr__(self) -> str:
        return f"{self.agent_name} {self.id}"


_THREAD_LOCAL = threading.local()


def compile_forward(code: str):
    namespace: dict[str, Any] = {}
    globals_dict = _safe_globals()
    exec(str(code), globals_dict, namespace)
    callables = [value for value in namespace.values() if callable(value)]
    if len(callables) != 1:
        raise AssertionError(f"Expected exactly one callable from generated code, got {len(callables)}")
    return _wrap_forward(callables[0])


def _wrap_forward(func):
    def wrapped(runtime: _AgentSystemRuntime, task_info: Info):
        previous = getattr(_THREAD_LOCAL, "runtime", None)
        _THREAD_LOCAL.runtime = runtime
        try:
            return func(runtime, task_info)
        finally:
            if previous is None:
                try:
                    delattr(_THREAD_LOCAL, "runtime")
                except AttributeError:
                    pass
            else:
                _THREAD_LOCAL.runtime = previous

    return wrapped


def _safe_globals() -> dict[str, Any]:
    allowed_builtins = {
        name: getattr(builtins, name)
        for name in [
            "abs",
            "all",
            "any",
            "bool",
            "dict",
            "enumerate",
            "float",
            "int",
            "len",
            "list",
            "max",
            "min",
            "range",
            "round",
            "set",
            "sorted",
            "str",
            "sum",
            "tuple",
            "zip",
        ]
    }
    allowed_imports = {"collections"}

    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name not in allowed_imports:
            raise ImportError(f"Import '{name}' is not allowed in ADAS generated code")
        return builtins.__import__(name, globals, locals, fromlist, level)

    allowed_builtins["__import__"] = limited_import
    return {
        "__builtins__": allowed_builtins,
        "LLMAgentBase": LLMAgentBase,
        "Info": Info,
        "Counter": Counter,
        "json": json,
    }


def _parse_json_fields(text: str, fields: list[str]) -> dict[str, str]:
    parsed: dict[str, Any] = {}
    stripped = str(text or "").strip()
    try:
        parsed = json.loads(stripped)
    except Exception:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = {}
    result = {field: "" for field in fields}
    if isinstance(parsed, dict):
        for field in fields:
            if field in parsed:
                result[field] = str(parsed[field])
    if not any(result.values()) and fields:
        result[fields[-1]] = stripped
    return result


def _tool_trace_events(parent: TraceEvent, agent_name: str, records: list[dict[str, Any]]) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    for idx, record in enumerate(records):
        tool_name = str(record.get("tool_name") or "")
        arguments = record.get("arguments") if isinstance(record.get("arguments"), dict) else {}
        base_state = f"{parent.state_id}:tool:{idx}"
        events.append(
            TraceEvent(
                timestamp_start=parent.timestamp_start,
                timestamp_end=parent.timestamp_start,
                actor=f"adas_{agent_name}",
                event_type="tool_call",
                payload={
                    "node": "adas",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "status": record.get("status"),
                },
                token_in=0,
                token_out=0,
                latency_ms=0.001,
                cost_usd=0.0,
                state_id=f"{base_state}:call",
            )
        )
        events.append(
            TraceEvent(
                timestamp_start=parent.timestamp_start,
                timestamp_end=parent.timestamp_start,
                actor="system",
                event_type="tool_result",
                payload={
                    "node": "adas",
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
            )
        )
    return events


def _extract_final_answer(text: str, benchmark_name: str) -> str:
    answer = str(text or "").strip()
    if benchmark_name.lower() == "plancraft" and answer:
        return re.sub(r"(?is)^Action\s*:\s*", "", answer.splitlines()[0].strip()).strip()
    match = re.search(r"(?is)<answer>\s*(.+?)\s*</answer>", answer)
    if match:
        return match.group(1).strip()
    matches = list(re.finditer(r"(?ims)(?:^|\n)\s*Answer\s*:\s*(.+?)(?=\n\s*\w+\s*:|\Z)", answer))
    if matches:
        return matches[-1].group(1).strip()
    return answer


def _metadata(
    *,
    solution: ADASSolution,
    task: BenchmarkTask,
    run_index: int,
    seed: int,
    trace_events: list[TraceEvent],
    llm_results: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    tool_counts: Counter[str] = Counter()
    for result in llm_results:
        for record in result.get("tool_calls", []):
            name = str(record.get("tool_name") or "")
            if name:
                tool_counts[name] += 1
    return {
        "adas_reproduce": True,
        "task_id": str(task.task_id),
        "run_index": int(run_index),
        "seed": int(seed),
        "solution": solution.to_payload(),
        "tool_definitions": _tool_definition_names(tools),
        "tool_call_counts": dict(tool_counts),
        "tool_calls_total": int(sum(tool_counts.values())),
        "messages_sent_total": len(llm_results),
        "rounds_executed": len(llm_results),
        "final_reason": "adas_forward_completed",
        "trace_event_count": len(trace_events),
    }


def _tool_definition_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
        elif isinstance(tool, dict) and tool.get("name"):
            names.append(str(tool["name"]))
    return names

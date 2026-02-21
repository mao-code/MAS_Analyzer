from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import random
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.base import BenchmarkEvaluation
from descriptor.experiment import analyze_task_runs, write_run_trace
from descriptor.schema import TraceEvent
from MAS import (
    ExperimentConfig,
    ExperimentRuntimeConfig,
    MASConfig,
    OpenRouterConfig,
    OpenRouterLLMClient,
    load_experiment_config,
)
from MAS.topology import AgentSpec, Topology, build_topology


@dataclass(frozen=True)
class SimpleTask:
    task_id: str
    prompt: str
    reference_answer: str = ""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    usage: str


@dataclass
class SimpleRunResult:
    final_answer: str
    trace_events: list[TraceEvent]
    run_metadata: dict[str, Any] = field(default_factory=dict)


class EventClock:
    def __init__(self) -> None:
        self.cursor = time.time()

    def span(self, latency_ms: float) -> tuple[float, float]:
        duration_s = max(latency_ms / 1000.0, 1e-6)
        start = self.cursor
        end = start + duration_s
        self.cursor = end + 1e-6
        return start, end


class MemoryStore:
    def __init__(self, agent_ids: Sequence[str]) -> None:
        self.shared_notes: list[str] = []
        self.agent_notes: dict[str, list[str]] = {agent_id: [] for agent_id in agent_ids}
        self.agent_inboxes: dict[str, list[dict[str, Any]]] = {
            agent_id: [] for agent_id in agent_ids
        }
        self.tool_history: dict[str, list[str]] = {agent_id: [] for agent_id in agent_ids}

    def add_inbox_message(self, recipient: str, sender: str, turn: int, text: str) -> None:
        self.agent_inboxes[recipient].append({"from": sender, "turn": turn, "text": text[:220]})

    def remember_agent_output(self, agent_id: str, text: str, turn: int) -> None:
        note = f"turn={turn} output={text[:200]}"
        self.agent_notes[agent_id].append(note)
        self.shared_notes.append(f"{agent_id}: {text[:180]}")

    def remember_tool_output(self, agent_id: str, tool_name: str, output: str, turn: int) -> None:
        note = f"turn={turn} {tool_name}: {output[:200]}"
        self.tool_history[agent_id].append(note)
        self.shared_notes.append(f"{agent_id}::{tool_name}: {output[:180]}")

    def prompt_memory_block(self, agent_id: str, *, limit: int = 5) -> str:
        inbox = self.agent_inboxes.get(agent_id, [])[-limit:]
        inbox_text = (
            "\n".join(f"- From {msg['from']} (turn {msg['turn']}): {msg['text']}" for msg in inbox)
            or "- None"
        )

        agent_local = self.agent_notes.get(agent_id, [])[-limit:]
        local_text = "\n".join(f"- {item}" for item in agent_local) or "- None"

        tool_items = self.tool_history.get(agent_id, [])[-limit:]
        tool_text = "\n".join(f"- {item}" for item in tool_items) or "- None"

        shared = self.shared_notes[-limit:]
        shared_text = "\n".join(f"- {item}" for item in shared) or "- None"

        return (
            "Inbox memory:\n"
            f"{inbox_text}\n\n"
            "Local memory:\n"
            f"{local_text}\n\n"
            "Tool memory:\n"
            f"{tool_text}\n\n"
            "Shared memory:\n"
            f"{shared_text}"
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "shared_notes_count": len(self.shared_notes),
            "agent_notes_count": {key: len(value) for key, value in self.agent_notes.items()},
            "inbox_count": {key: len(value) for key, value in self.agent_inboxes.items()},
            "tool_history_count": {key: len(value) for key, value in self.tool_history.items()},
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {
            "calculator": ToolSpec(
                name="calculator",
                description="Evaluate simple arithmetic expressions.",
                usage="TOOL:calculator:<expression>",
            ),
            "lookup": ToolSpec(
                name="lookup",
                description="Lookup short facts from a small local knowledge base.",
                usage="TOOL:lookup:<query>",
            ),
        }
        self._knowledge_base: list[tuple[str, str]] = [
            ("1", "OpenRouter provides an OpenAI-compatible chat completion API endpoint."),
            ("2", "MAS means multi-agent system. SAS means single-agent system."),
            ("3", "This script logs descriptor-compatible trace events in JSONL format."),
            (
                "4",
                "Python dataclasses are used in this repository for configuration and schema objects.",
            ),
        ]
        self._bin_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        self._unary_ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def has_tool(self, name: str) -> bool:
        return name in self._specs

    def execute(self, name: str, argument: str) -> dict[str, Any]:
        if name == "calculator":
            return self._calculator(argument)
        if name == "lookup":
            return self._lookup(argument)
        return {"ok": False, "output": "", "error": f"unknown tool: {name}"}

    def _calculator(self, expression: str) -> dict[str, Any]:
        expression = expression.strip()
        if not expression:
            return {"ok": False, "output": "", "error": "empty expression"}
        try:
            parsed = ast.parse(expression, mode="eval")
            value = self._eval_ast(parsed)
            return {"ok": True, "output": str(value), "error": None}
        except Exception as exc:
            return {"ok": False, "output": "", "error": f"calculator failed: {exc}"}

    def _eval_ast(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self._eval_ast(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in self._bin_ops:
            left = self._eval_ast(node.left)
            right = self._eval_ast(node.right)
            return float(self._bin_ops[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary_ops:
            val = self._eval_ast(node.operand)
            return float(self._unary_ops[type(node.op)](val))
        raise ValueError("unsupported expression")

    def _lookup(self, query: str) -> dict[str, Any]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not query_tokens:
            return {"ok": False, "output": "", "error": "empty query"}

        scored: list[tuple[int, str, str]] = []
        for docid, text in self._knowledge_base:
            doc_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
            score = len(query_tokens & doc_tokens)
            scored.append((score, docid, text))

        scored.sort(reverse=True)
        hits = [(docid, text) for score, docid, text in scored if score > 0][:2]
        if not hits:
            return {"ok": True, "output": "No local facts found.", "error": None}

        rendered = " | ".join(f"[{docid}] {text}" for docid, text in hits)
        return {"ok": True, "output": rendered, "error": None}


class SimpleAgentSystemRunner:
    TOOL_REQUEST_PATTERN = re.compile(r"TOOL\s*:\s*([a-zA-Z_][\w-]*)\s*:\s*(.+)")
    MATH_EXPR_PATTERN = re.compile(r"\b\d+(?:\s*[-+*/]\s*\d+)+\b")

    def __init__(self, config: ExperimentConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        self.tools = ToolRegistry()

    def run_task(self, task: SimpleTask, run_index: int, seed: int) -> SimpleRunResult:
        mas_cfg = self.config.mas
        topology = build_topology(mas_cfg, seed=seed)
        rng = random.Random(seed)
        clock = EventClock()
        events: list[TraceEvent] = []

        agent_ids = [agent.agent_id for agent in topology.agents]
        memory = MemoryStore(agent_ids=agent_ids)
        message_budget = {
            agent.agent_id: mas_cfg.communication_count_internally for agent in topology.agents
        }
        agent_outputs: dict[str, str] = {}
        auto_tool_calls = 0
        llm_tool_calls = 0

        events.append(
            self._event(
                clock,
                actor="system",
                event_type="plan",
                payload={
                    "task_id": task.task_id,
                    "mode": "sas" if mas_cfg.total_agents == 1 else "mas",
                    "levels": mas_cfg.levels,
                    "total_agents": mas_cfg.total_agents,
                    "turn_mode": mas_cfg.turn_mode,
                    "max_turns": mas_cfg.max_turns,
                    "message_budget_per_agent": mas_cfg.communication_count_internally,
                    "tools": [tool.__dict__ for tool in self.tools.specs()],
                    "topology": self._topology_payload(topology),
                },
                token_in=0,
                token_out=0,
                latency_ms=1.0,
                cost_usd=0.0,
                state_id=f"run_{run_index}_plan",
            )
        )

        turns_limit = 1 if mas_cfg.turn_mode == "single_turn" else mas_cfg.max_turns
        turns_executed = 0

        for turn in range(turns_limit):
            if turn > 0 and sum(message_budget.values()) <= 0:
                break

            if turn > 0:
                events.append(
                    self._event(
                        clock,
                        actor="system",
                        event_type="revise",
                        payload={"turn": turn, "reason": "continue coordination"},
                        token_in=0,
                        token_out=0,
                        latency_ms=1.0,
                        cost_usd=0.0,
                        state_id=f"run_{run_index}_turn_{turn}_revise",
                    )
                )

            for spec in topology.agents:
                memory_block = memory.prompt_memory_block(spec.agent_id)
                auto_calls = self._auto_tool_calls(task.prompt, memory_block)
                for tool_name, tool_arg in auto_calls:
                    auto_tool_calls += 1
                    self._run_tool(
                        clock=clock,
                        events=events,
                        memory=memory,
                        actor=spec.agent_id,
                        tool_name=tool_name,
                        tool_arg=tool_arg,
                        turn=turn,
                        state_id_prefix=f"run_{run_index}_turn_{turn}_{spec.agent_id}_auto_tool_{auto_tool_calls}",
                        source="auto",
                    )

                prompt = self._build_agent_prompt(
                    task_prompt=task.prompt,
                    spec=spec,
                    turn=turn,
                    memory=memory,
                )

                t0 = time.perf_counter()
                llm = self.llm_client.generate(
                    prompt=prompt,
                    agent_type=spec.agent_type,
                    task_id=task.task_id,
                    run_index=run_index,
                    agent_id=spec.agent_id,
                    temperature=0.0,
                )
                latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)

                memory.remember_agent_output(spec.agent_id, llm.text, turn)
                agent_outputs[spec.agent_id] = llm.text
                events.append(
                    self._event(
                        clock,
                        actor=spec.agent_id,
                        event_type="act",
                        payload={
                            "turn": turn,
                            "agent_type": spec.agent_type,
                            "model": llm.model,
                            "mock_used": llm.mock_used,
                            "metadata": llm.metadata,
                            "response_preview": llm.text[:240],
                        },
                        token_in=llm.token_in,
                        token_out=llm.token_out,
                        latency_ms=latency_ms,
                        cost_usd=llm.cost_usd,
                        state_id=f"run_{run_index}_turn_{turn}_{spec.agent_id}",
                    )
                )

                llm_call = self._extract_first_tool_request(llm.text)
                if llm_call is not None:
                    llm_tool_calls += 1
                    tool_name, tool_arg = llm_call
                    if self.tools.has_tool(tool_name):
                        self._run_tool(
                            clock=clock,
                            events=events,
                            memory=memory,
                            actor=spec.agent_id,
                            tool_name=tool_name,
                            tool_arg=tool_arg,
                            turn=turn,
                            state_id_prefix=f"run_{run_index}_turn_{turn}_{spec.agent_id}_llm_tool_{llm_tool_calls}",
                            source="llm_request",
                        )

                if message_budget[spec.agent_id] > 0:
                    recipient = self._select_recipient(rng, topology, spec.agent_id)
                    if recipient is not None:
                        message_text = self._message_snippet(llm.text)
                        memory.add_inbox_message(recipient, spec.agent_id, turn, message_text)
                        message_budget[spec.agent_id] -= 1

                        events.append(
                            self._event(
                                clock,
                                actor=spec.agent_id,
                                event_type="tool_call",
                                payload={
                                    "tool_name": "inter_agent_send",
                                    "source": "runtime",
                                    "to": recipient,
                                    "turn": turn,
                                },
                                token_in=0,
                                token_out=0,
                                latency_ms=1.0,
                                cost_usd=0.0,
                                state_id=f"run_{run_index}_turn_{turn}_{spec.agent_id}_send",
                            )
                        )
                        events.append(
                            self._event(
                                clock,
                                actor=recipient,
                                event_type="tool_result",
                                payload={
                                    "tool_name": "inter_agent_send",
                                    "status": "ok",
                                    "from": spec.agent_id,
                                    "turn": turn,
                                },
                                token_in=0,
                                token_out=0,
                                latency_ms=1.0,
                                cost_usd=0.0,
                                state_id=f"run_{run_index}_turn_{turn}_{recipient}_receive",
                            )
                        )

            turns_executed += 1
            active_inboxes = sum(1 for values in memory.agent_inboxes.values() if values)
            events.append(
                self._event(
                    clock,
                    actor="system",
                    event_type="verify",
                    payload={
                        "node": "turn_check",
                        "turn": turn,
                        "active_inboxes": active_inboxes,
                        "memory_snapshot": memory.snapshot(),
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{run_index}_turn_{turn}_verify",
                )
            )

        final_answer = self._final_answer(topology.agents, agent_outputs)
        events.append(
            self._event(
                clock,
                actor="system",
                event_type="verify",
                payload={
                    "node": "evaluation_precheck",
                    "turns_executed": turns_executed,
                    "auto_tool_calls": auto_tool_calls,
                    "llm_tool_calls": llm_tool_calls,
                    "memory_snapshot": memory.snapshot(),
                },
                token_in=0,
                token_out=0,
                latency_ms=1.0,
                cost_usd=0.0,
                state_id=f"run_{run_index}_evaluation_precheck",
            )
        )
        events.append(
            self._event(
                clock,
                actor="system",
                event_type="finalize",
                payload={
                    "status": "completed",
                    "final_answer": final_answer,
                    "retrieved_docids": self._extract_docids(final_answer),
                },
                token_in=0,
                token_out=max(1, len(final_answer.split())),
                latency_ms=1.0,
                cost_usd=0.0,
                state_id=f"run_{run_index}_finalize",
            )
        )

        metadata = {
            "task_id": task.task_id,
            "run_index": run_index,
            "seed": seed,
            "turns_executed": turns_executed,
            "mode": "sas" if mas_cfg.total_agents == 1 else "mas",
            "message_budget_remaining": message_budget,
            "topology": self._topology_payload(topology),
            "agent_outputs": agent_outputs,
            "memory_snapshot": memory.snapshot(),
            "auto_tool_calls": auto_tool_calls,
            "llm_tool_calls": llm_tool_calls,
        }
        return SimpleRunResult(
            final_answer=final_answer,
            trace_events=events,
            run_metadata=metadata,
        )

    def _run_tool(
        self,
        *,
        clock: EventClock,
        events: list[TraceEvent],
        memory: MemoryStore,
        actor: str,
        tool_name: str,
        tool_arg: str,
        turn: int,
        state_id_prefix: str,
        source: str,
    ) -> None:
        events.append(
            self._event(
                clock,
                actor=actor,
                event_type="tool_call",
                payload={
                    "tool_name": tool_name,
                    "input": tool_arg,
                    "source": source,
                    "turn": turn,
                },
                token_in=0,
                token_out=0,
                latency_ms=1.0,
                cost_usd=0.0,
                state_id=f"{state_id_prefix}_call",
            )
        )

        result = self.tools.execute(tool_name, tool_arg)
        output = result["output"] if result.get("ok", False) else result.get("error", "")
        memory.remember_tool_output(actor, tool_name, output, turn)

        events.append(
            self._event(
                clock,
                actor=actor,
                event_type="tool_result",
                payload={
                    "tool_name": tool_name,
                    "status": "ok" if result.get("ok", False) else "error",
                    "source": source,
                    "turn": turn,
                    "output_preview": str(output)[:240],
                    "error": result.get("error"),
                },
                token_in=0,
                token_out=max(1, len(str(output).split())) if output else 0,
                latency_ms=1.0,
                cost_usd=0.0,
                state_id=f"{state_id_prefix}_result",
            )
        )

    def _auto_tool_calls(self, task_prompt: str, memory_block: str) -> list[tuple[str, str]]:
        calls: list[tuple[str, str]] = []
        all_text = f"{task_prompt}\n{memory_block}"
        expression = self.MATH_EXPR_PATTERN.search(all_text)
        if expression:
            calls.append(("calculator", expression.group(0)))

        trigger_words = ("openrouter", "mas", "sas", "trace", "python", "api")
        if any(word in all_text.lower() for word in trigger_words):
            calls.append(("lookup", task_prompt))
        return calls[:2]

    def _build_agent_prompt(
        self,
        *,
        task_prompt: str,
        spec: AgentSpec,
        turn: int,
        memory: MemoryStore,
    ) -> str:
        tool_lines = "\n".join(
            f"- {tool.name}: {tool.description} | usage: {tool.usage}"
            for tool in self.tools.specs()
        )
        memory_block = memory.prompt_memory_block(spec.agent_id)
        return (
            f"Task:\n{task_prompt}\n\n"
            f"Agent: {spec.agent_id} (type={spec.agent_type}, level={spec.level})\n"
            f"Turn: {turn}\n\n"
            "Available tools:\n"
            f"{tool_lines}\n\n"
            "If you need a tool, include one line like TOOL:calculator:2+2.\n"
            "Return concise reasoning followed by the best current answer.\n\n"
            f"{memory_block}"
        )

    def _extract_first_tool_request(self, text: str) -> tuple[str, str] | None:
        for line in text.splitlines():
            match = self.TOOL_REQUEST_PATTERN.search(line.strip())
            if not match:
                continue
            return match.group(1).lower(), match.group(2).strip()[:200]
        return None

    @staticmethod
    def _select_recipient(
        rng: random.Random,
        topology: Topology,
        agent_id: str,
    ) -> str | None:
        neighbors = topology.neighbors(agent_id)
        if not neighbors:
            return None
        return neighbors[rng.randrange(len(neighbors))]

    @staticmethod
    def _final_answer(agents: list[AgentSpec], outputs: dict[str, str]) -> str:
        if not outputs:
            return ""
        final_agent = agents[-1].agent_id
        return outputs.get(final_agent, next(iter(outputs.values())))

    @staticmethod
    def _message_snippet(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())[:220]

    @staticmethod
    def _extract_docids(text: str) -> list[str]:
        single = re.findall(r"\[(\d+)\]", text)
        grouped = re.findall(r"\[([^\[\]]+?)\]", text)
        ids = set(single)
        for group in grouped:
            ids.update(re.findall(r"\d+", group))
        return sorted(ids)

    @staticmethod
    def _topology_payload(topology: Topology) -> dict[str, Any]:
        return {
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "level": agent.level,
                    "agent_type": agent.agent_type,
                }
                for agent in topology.agents
            ],
            "adjacency": topology.adjacency,
        }

    @staticmethod
    def _event(
        clock: EventClock,
        *,
        actor: str,
        event_type: str,
        payload: dict[str, Any],
        token_in: int,
        token_out: int,
        latency_ms: float,
        cost_usd: float,
        state_id: str,
    ) -> TraceEvent:
        start, end = clock.span(latency_ms)
        return TraceEvent(
            timestamp_start=start,
            timestamp_end=end,
            actor=actor,
            event_type=event_type,
            payload=payload,
            token_in=max(0, int(token_in)),
            token_out=max(0, int(token_out)),
            latency_ms=max(0.0, float(latency_ms)),
            cost_usd=max(0.0, float(cost_usd)),
            state_id=state_id,
        )


def evaluate_run(task: SimpleTask, run: SimpleRunResult) -> BenchmarkEvaluation:
    answer = run.final_answer.strip()
    completion = 1.0 if answer else 0.0

    reference = task.reference_answer.strip().lower()
    if reference:
        ref_tokens = set(re.findall(r"[a-z0-9]+", reference))
        ans_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))
        overlap = len(ref_tokens & ans_tokens) / max(1, len(ref_tokens))
    else:
        overlap = 1.0 if answer else 0.0

    tool_results = [
        event
        for event in run.trace_events
        if event.event_type == "tool_result"
        and event.payload.get("tool_name") in {"calculator", "lookup"}
    ]
    tool_bonus = 1.0 if tool_results else 0.0

    act_events = [event for event in run.trace_events if event.event_type == "act"]
    mock_calls = sum(1 for event in act_events if bool(event.payload.get("mock_used", False)))
    mock_ratio = mock_calls / max(1, len(act_events))

    score = 0.35 * completion + 0.45 * overlap + 0.20 * tool_bonus
    success = score >= 0.6

    details = {
        "completion": completion,
        "overlap": overlap,
        "tool_bonus": tool_bonus,
        "tool_results": len(tool_results),
        "mock_ratio": mock_ratio,
        "run_metadata": run.run_metadata,
    }
    return BenchmarkEvaluation(
        task_id=task.task_id,
        score=float(score),
        success=bool(success),
        details=details,
    )


def load_base_config(path: str | None) -> ExperimentConfig:
    if path:
        resolved = Path(path).expanduser().resolve()
        if resolved.exists():
            return load_experiment_config(resolved)
    return default_config()


def default_config() -> ExperimentConfig:
    cfg = ExperimentConfig(
        openrouter=OpenRouterConfig(api_key=os.getenv("OPENROUTER_API_KEY")),
        mas=MASConfig(
            levels=1,
            intra_level_link_ratio=1.0,
            full_linked=True,
            number_of_agents=1,
            agents_per_level=[1],
            agent_types=["general"],
            communication_count_internally=0,
            turn_mode="single_turn",
            max_turns=1,
        ),
        experiment=ExperimentRuntimeConfig(output_dir="outputs", runs_per_task=1, seed=42),
        models={"default": "openai/gpt-4o-mini"},
    )
    cfg.validate()
    return cfg


def build_runtime_config(base: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    agent_types = [item.strip() for item in args.agent_types.split(",") if item.strip()]
    if not agent_types:
        agent_types = ["general"]

    if args.mode == "sas":
        mas_cfg = replace(
            base.mas,
            levels=1,
            intra_level_link_ratio=1.0,
            full_linked=True,
            number_of_agents=1,
            agents_per_level=[1],
            agent_types=[agent_types[0]],
            communication_count_internally=0,
            turn_mode="single_turn",
            max_turns=1,
        )
    else:
        levels = max(1, int(args.levels))
        total_agents = max(levels, int(args.agents))
        mas_cfg = replace(
            base.mas,
            levels=levels,
            intra_level_link_ratio=float(args.intra_level_link_ratio),
            full_linked=bool(args.full_linked),
            number_of_agents=total_agents,
            agents_per_level=None,
            agent_types=agent_types,
            communication_count_internally=max(0, int(args.message_budget)),
            turn_mode=args.turn_mode,
            max_turns=max(1, int(args.max_turns)),
        )

    models = dict(base.models)
    if args.model:
        models["default"] = args.model
        for agent_type in agent_types:
            models[agent_type] = args.model
    elif "default" not in models:
        models["default"] = "openai/gpt-4o-mini"
    for agent_type in agent_types:
        models.setdefault(agent_type, models["default"])

    exp_cfg = replace(
        base.experiment,
        output_dir=str(args.output_dir),
        runs_per_task=max(1, int(args.runs)),
        seed=max(0, int(args.seed)),
    )

    config = ExperimentConfig(
        openrouter=base.openrouter,
        mas=mas_cfg,
        experiment=exp_cfg,
        models=models,
        browsecomp=dict(base.browsecomp),
        finance_agent=dict(base.finance_agent),
    )
    config.validate()
    return config


def now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def write_eval(path: Path, evaluation: BenchmarkEvaluation, prediction: str) -> None:
    payload = {
        "task_id": evaluation.task_id,
        "score": evaluation.score,
        "success": evaluation.success,
        "details": evaluation.details,
        "prediction": prediction,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone SAS or MAS prompt test without benchmark adapters. "
            "Includes memory, tool usage, evaluation, and trace extraction."
        )
    )
    parser.add_argument("--mode", choices=["sas", "mas"], required=True)
    parser.add_argument("--prompt", required=True, help="Test prompt to run")
    parser.add_argument("--reference", default="", help="Optional reference for overlap scoring")
    parser.add_argument("--task-id", default="simple_prompt")
    parser.add_argument("--config", default="config/experiment.toml")
    parser.add_argument("--output-dir", default="outputs/simple_agent_test")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--agent-types", default="planner,researcher")
    parser.add_argument("--message-budget", type=int, default=1)
    parser.add_argument("--turn-mode", choices=["single_turn", "multi_turn"], default="multi_turn")
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--intra-level-link-ratio", type=float, default=0.7)
    parser.add_argument("--full-linked", action="store_true")
    parser.add_argument("--model", default=None, help="Optional model override for all agent types")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_cfg = load_base_config(args.config)
    runtime_cfg = build_runtime_config(base_cfg, args)

    llm_client = OpenRouterLLMClient(runtime_cfg.openrouter, runtime_cfg.models)
    runner = SimpleAgentSystemRunner(runtime_cfg, llm_client)
    task = SimpleTask(
        task_id=args.task_id,
        prompt=args.prompt,
        reference_answer=args.reference,
    )

    timestamp = now_stamp()
    run_root = Path(args.output_dir) / timestamp / args.mode / task.task_id
    run_root.mkdir(parents=True, exist_ok=True)

    run_traces: list[list[TraceEvent]] = []
    evaluations: list[BenchmarkEvaluation] = []
    final_answers: list[str] = []

    for run_index in range(runtime_cfg.experiment.runs_per_task):
        run_seed = runtime_cfg.experiment.seed + run_index
        run = runner.run_task(task=task, run_index=run_index, seed=run_seed)
        final_answers.append(run.final_answer)
        run_traces.append(run.trace_events)

        trace_path = run_root / f"run_{run_index}.trace.jsonl"
        write_run_trace(run.trace_events, trace_path)

        evaluation = evaluate_run(task, run)
        evaluations.append(evaluation)
        eval_path = run_root / f"run_{run_index}.eval.json"
        write_eval(eval_path, evaluation, run.final_answer)

    analysis = analyze_task_runs(
        task_id=task.task_id,
        benchmark_name=f"simple_{args.mode}",
        run_traces=run_traces,
        evaluations=evaluations,
        output_dir=run_root,
    )

    summary = {
        "timestamp": timestamp,
        "mode": args.mode,
        "task_id": task.task_id,
        "prompt": task.prompt,
        "reference": task.reference_answer,
        "runs": runtime_cfg.experiment.runs_per_task,
        "output_dir": str(run_root),
        "final_answers": final_answers,
        "evaluation": analysis["evaluation"],
        "stage_bottleneck": analysis["stage_bottleneck"],
        "descriptor_keys": sorted(list(analysis["descriptor"].keys())),
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Run complete: {run_root}")
    print(f"Summary: {summary_path}")
    if evaluations:
        print(f"Last run score={evaluations[-1].score:.3f} success={evaluations[-1].success}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

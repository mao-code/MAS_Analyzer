from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from descriptor.schema import TraceEvent

from .config import ExperimentConfig
from .llm import OpenRouterLLMClient
from .topology import AgentSpec, Topology, build_topology


@dataclass
class MASRunResult:
    final_answer: str
    trace_events: list[TraceEvent]
    run_metadata: dict[str, Any] = field(default_factory=dict)


class _EventClock:
    def __init__(self) -> None:
        self.cursor = time.time()

    def span(self, latency_ms: float) -> tuple[float, float]:
        duration_s = max(latency_ms / 1000.0, 1e-6)
        start = self.cursor
        end = start + duration_s
        self.cursor = end + 1e-6
        return start, end


class MASRunner:
    """Minimal MAS runtime emitting descriptor-compatible trace events."""

    def __init__(self, config: ExperimentConfig, llm_client: OpenRouterLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client

    def run_task(
        self,
        task: Any,
        run_index: int,
        seed: int,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tool_iterations: int = 8,
    ) -> MASRunResult:
        mas_cfg = self.config.mas
        topology = build_topology(mas_cfg, seed=seed)
        rng = random.Random(seed)
        tools = tools or []

        events: list[TraceEvent] = []
        clock = _EventClock()

        inboxes: dict[str, list[dict[str, Any]]] = {agent.agent_id: [] for agent in topology.agents}
        message_budget = {
            agent.agent_id: mas_cfg.communication_count_internally for agent in topology.agents
        }
        sent_by_agent = {agent.agent_id: 0 for agent in topology.agents}
        agent_outputs: dict[str, str] = {}
        tool_call_counts: dict[str, int] = {}
        retrieved_docids_set: set[str] = set()
        tool_event_index = 0

        events.append(
            self._event(
                clock,
                actor="system",
                event_type="plan",
                payload={
                    "task_id": task.task_id,
                    "levels": mas_cfg.levels,
                    "total_agents": mas_cfg.total_agents,
                    "turn_mode": mas_cfg.turn_mode,
                    "max_turns": mas_cfg.max_turns,
                    "message_budget_per_agent": mas_cfg.communication_count_internally,
                    "available_tools": [str(item.get("name", "")) for item in tools],
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
                        payload={
                            "turn": turn,
                            "reason": "Continuing multi-turn coordination",
                        },
                        token_in=0,
                        token_out=0,
                        latency_ms=1.0,
                        cost_usd=0.0,
                        state_id=f"run_{run_index}_turn_{turn}_revise",
                    )
                )

            for spec in topology.agents:
                prompt = self._build_agent_prompt(task.prompt, spec, turn, inboxes[spec.agent_id])

                t0 = time.perf_counter()
                llm = self.llm_client.generate(
                    prompt=prompt,
                    agent_type=spec.agent_type,
                    task_id=task.task_id,
                    run_index=run_index,
                    agent_id=spec.agent_id,
                    tools=tools,
                    max_tool_iterations=max_tool_iterations,
                    temperature=0.0,
                )
                latency_ms = max((time.perf_counter() - t0) * 1000.0, 1.0)

                tool_records = list(llm.tool_calls)
                if tools and llm.mock_used and not tool_records:
                    tool_records = self._mock_tool_plan(prompt, tools)

                if tool_records:
                    for record in tool_records:
                        tool_name = str(record.get("tool_name") or "")
                        if not tool_name:
                            continue
                        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                        tool_event_index += 1
                        call_state = (
                            f"run_{run_index}_turn_{turn}_{spec.agent_id}_tool_{tool_event_index}_call"
                        )
                        result_state = (
                            f"run_{run_index}_turn_{turn}_{spec.agent_id}_tool_{tool_event_index}_result"
                        )
                        args = record.get("arguments")
                        if not isinstance(args, dict):
                            args = {}
                        status = str(record.get("status") or "completed")
                        error = record.get("error")
                        output = record.get("output")

                        events.append(
                            self._event(
                                clock,
                                actor=spec.agent_id,
                                event_type="tool_call",
                                payload={
                                    "tool_name": tool_name,
                                    "arguments": args,
                                    "status": status,
                                    "turn": turn,
                                },
                                token_in=0,
                                token_out=0,
                                latency_ms=1.0,
                                cost_usd=0.0,
                                state_id=call_state,
                            )
                        )
                        events.append(
                            self._event(
                                clock,
                                actor=spec.agent_id,
                                event_type="tool_result",
                                payload={
                                    "tool_name": tool_name,
                                    "status": status,
                                    "error": error,
                                    "turn": turn,
                                    "output_preview": self._tool_output_preview(output),
                                },
                                token_in=0,
                                token_out=0,
                                latency_ms=1.0,
                                cost_usd=0.0,
                                state_id=result_state,
                            )
                        )

                        if self._is_retrieval_tool(tool_name):
                            retrieved_docids_set.update(
                                self._extract_docids_from_tool_output(output, args, tool_name)
                            )

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

                if message_budget[spec.agent_id] > 0:
                    recipient = self._select_recipient(rng, topology, spec.agent_id)
                    if recipient is not None:
                        message_text = self._message_snippet(llm.text)
                        inboxes[recipient].append(
                            {
                                "from": spec.agent_id,
                                "text": message_text,
                                "turn": turn,
                            }
                        )
                        message_budget[spec.agent_id] -= 1
                        sent_by_agent[spec.agent_id] += 1
                        tool_call_counts["inter_agent_send"] = (
                            tool_call_counts.get("inter_agent_send", 0) + 1
                        )

                        events.append(
                            self._event(
                                clock,
                                actor=spec.agent_id,
                                event_type="tool_call",
                                payload={
                                    "tool_name": "inter_agent_send",
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
            verify_count = sum(1 for inbox in inboxes.values() if inbox)
            events.append(
                self._event(
                    clock,
                    actor="system",
                    event_type="verify",
                    payload={
                        "turn": turn,
                        "active_inboxes": verify_count,
                    },
                    token_in=0,
                    token_out=0,
                    latency_ms=1.0,
                    cost_usd=0.0,
                    state_id=f"run_{run_index}_turn_{turn}_verify",
                )
            )

        final_answer = self._final_answer(topology.agents, agent_outputs)
        retrieved_docids = (
            sorted(retrieved_docids_set)
            if retrieved_docids_set
            else self._extract_docids(final_answer)
        )

        events.append(
            self._event(
                clock,
                actor="system",
                event_type="finalize",
                payload={
                    "status": "completed",
                    "success": True,
                    "final_answer": final_answer,
                    "retrieved_docids": retrieved_docids,
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
            "turn_mode": mas_cfg.turn_mode,
            "messages_sent_total": sum(sent_by_agent.values()),
            "messages_sent_by_agent": sent_by_agent,
            "tool_call_counts": tool_call_counts,
            "tool_calls_total": int(sum(tool_call_counts.values())),
            "remaining_message_budget": message_budget,
            "topology": self._topology_payload(topology),
            "agent_outputs": agent_outputs,
            "retrieved_docids": retrieved_docids,
        }

        return MASRunResult(
            final_answer=final_answer,
            trace_events=events,
            run_metadata=metadata,
        )

    @staticmethod
    def _event(
        clock: _EventClock,
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

    @staticmethod
    def _build_agent_prompt(
        task_prompt: Any,
        spec: AgentSpec,
        turn: int,
        inbox: list[dict[str, Any]],
    ) -> Any:
        inbox_text = ""
        if inbox:
            pieces = []
            for message in inbox[-5:]:
                pieces.append(f"From {message['from']} (turn {message['turn']}): {message['text']}")
            inbox_text = "\n".join(pieces)

        if isinstance(task_prompt, list):
            # Pass the list through natively, but wrap MAS system logic as the final instruction
            mas_instruction = (
                f"Agent: {spec.agent_id} (type={spec.agent_type}, level={spec.level})\n"
                f"Turn: {turn}\n"
                f"Messages:\n{inbox_text or 'None'}\n\n"
                "Produce a concise step toward the final answer."
            )
            return task_prompt + [{"role": "system", "content": mas_instruction}]

        return (
            f"Task:\n{task_prompt}\n\n"
            f"Agent: {spec.agent_id} (type={spec.agent_type}, level={spec.level})\n"
            f"Turn: {turn}\n"
            f"Messages:\n{inbox_text or 'None'}\n\n"
            "Produce a concise step toward the final answer."
        )

    @staticmethod
    def _select_recipient(rng: random.Random, topology: Topology, agent_id: str) -> str | None:
        candidates = topology.neighbors(agent_id)
        if not candidates:
            return None
        return candidates[rng.randrange(len(candidates))]

    @staticmethod
    def _message_snippet(text: str) -> str:
        snippet = re.sub(r"\s+", " ", text.strip())
        return snippet[:220]

    @staticmethod
    def _tool_output_preview(output: Any) -> str:
        if isinstance(output, str):
            text = output
        else:
            try:
                text = json.dumps(output, ensure_ascii=False)
            except Exception:
                text = str(output)
        text = re.sub(r"\s+", " ", text.strip())
        return text[:240]

    @staticmethod
    def _is_retrieval_tool(tool_name: str) -> bool:
        name = tool_name.lower().strip()
        return "search" in name or "retriev" in name or "document" in name

    @classmethod
    def _extract_docids_from_tool_output(
        cls,
        output: Any,
        arguments: dict[str, Any],
        tool_name: str,
    ) -> list[str]:
        out: set[str] = set()
        if "docid" in arguments:
            out.add(str(arguments.get("docid")))

        if isinstance(output, dict):
            if "docid" in output:
                out.add(str(output["docid"]))
            payload = [output]
        elif isinstance(output, list):
            payload = output
        elif isinstance(output, str):
            payload = []
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    payload = [parsed]
                elif isinstance(parsed, list):
                    payload = parsed
            except Exception:
                for docid in re.findall(r'"docid"\s*:\s*"([^"]+)"', output):
                    out.add(str(docid))
                for docid in re.findall(r'"docid"\s*:\s*(\d+)', output):
                    out.add(str(docid))
        else:
            payload = []

        for item in payload:
            if isinstance(item, dict) and "docid" in item:
                out.add(str(item["docid"]))

        if not out and cls._is_retrieval_tool(tool_name):
            text = output if isinstance(output, str) else ""
            out.update(cls._extract_docids(text))
        return sorted(docid for docid in out if docid and docid != "None")

    @classmethod
    def _mock_tool_plan(
        cls,
        prompt: Any,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_name: dict[str, dict[str, Any]] = {}
        for tool in tools:
            name = str(tool.get("name", "")).strip()
            if name and callable(tool.get("handler")):
                by_name[name] = tool
        if not by_name:
            return []

        def _execute(name: str, args: dict[str, Any]) -> dict[str, Any]:
            handler = by_name[name]["handler"]
            status = "completed"
            error = None
            try:
                output = handler(args)
            except Exception as exc:
                status = "error"
                error = str(exc)
                output = {"error": error}
            return {
                "tool_name": name,
                "arguments": args,
                "status": status,
                "error": error,
                "output": output,
            }

        if isinstance(prompt, list):
            prompt_text = " ".join(str(item.get("content", "")) for item in prompt if isinstance(item, dict))
        else:
            prompt_text = str(prompt)
        query = re.sub(r"\s+", " ", prompt_text).strip()[:600]
        if not query:
            query = "Find relevant documents for the task."

        records: list[dict[str, Any]] = []
        if "search" in by_name:
            search_record = _execute("search", {"query": query})
            records.append(search_record)

            candidate_docids = cls._extract_docids_from_tool_output(
                search_record.get("output"),
                {"query": query},
                "search",
            )
            if "get_document" in by_name and candidate_docids:
                records.append(_execute("get_document", {"docid": candidate_docids[0]}))
        else:
            first_name = next(iter(by_name))
            records.append(_execute(first_name, {}))

        return records

    @staticmethod
    def _final_answer(agents: list[AgentSpec], outputs: dict[str, str]) -> str:
        if not outputs:
            return ""
        last_agent = agents[-1].agent_id
        return outputs.get(last_agent, next(iter(outputs.values())))

    @staticmethod
    def _extract_docids(text: str) -> list[str]:
        # Compatible with citation styles like [123], [123, 456], and 【123】.
        single = re.findall(r"\[(\d+)\]", text)
        single_full = re.findall(r"【(\d+)】", text)
        grouped = re.findall(r"\[([^\[\]]+?)\]", text)
        grouped_full = re.findall(r"【([^【】]+?)】", text)

        docids = set(single)
        docids.update(single_full)
        for group in grouped + grouped_full:
            docids.update(re.findall(r"\d+", group))
        return sorted(docids)

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

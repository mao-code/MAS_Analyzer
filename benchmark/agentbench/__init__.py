"""AgentBench benchmark adapter.

Source repo  : https://github.com/THUDM/AgentBench
Paper        : https://arxiv.org/abs/2308.03688

This adapter replaces the official ``AgentClient`` with our ``MASRunner``,
while **faithfully mirroring the HTTP protocol** defined in
``src/client/task.py`` (``TaskClient.run_sample``).

Architecture
~~~~~~~~~~~~
AgentBench uses a distributed Client ↔ Controller ↔ Worker model:

  Controller  – Flask API on ``http://localhost:5000/api``
  Worker      – Docker container running one task environment
  Client      – our adapter (``AgentBenchBenchmark``)

HTTP API (controller endpoints used by the client):

  GET  /get_indices?name=<task>           → List[SampleIndex]
  POST /start_sample  {name, index}       → {session_id, output: TaskOutput}
  POST /interact      {session_id, agent_response: AgentOutput}
                                          → {output: TaskOutput}
  POST /cancel        {session_id}        → 204
  POST /calculate_overall {name, results} → custom metrics dict

Data types (from ``src/typings``):

  SampleStatus = "running" | "completed" | "agent context limit"
                 | "agent validation failed" | "agent invalid action"
                 | "task limit reached" | "unknown" | "task error"

  AgentOutputStatus = "normal" | "cancelled" | "agent context limit"

  AgentOutput       = {status: AgentOutputStatus, content: str|None}
  TaskOutput        = {index, status: SampleStatus, result, history: [ChatHistoryItem]}
  ChatHistoryItem   = {role: "user"|"agent", content: str}

The interact loop mirrors the official ``run_sample`` exactly:
  1. POST /start_sample → get initial TaskOutput
  2. while status == "running":
       content = agent.inference(history)   # ← we use MASRunner here
       POST /interact {session_id, agent_response={status:"normal", content}}
  3. Return final TaskOutput

You must start the AgentBench Task Server before running this benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import requests
from loguru import logger

# Re-use our runner types
from MAS import MASRunner, MASRunResult

from ..base import (
    BenchmarkEvaluation,
    BenchmarkTask,
    init_run_metadata_aggregate,
    merge_step_run_metadata,
)


# ---------------------------------------------------------------------------
# SampleStatus enum values (from src/typings/status.py)
# ---------------------------------------------------------------------------
class _SampleStatus:
    RUNNING = "running"
    COMPLETED = "completed"
    AGENT_CONTEXT_LIMIT = "agent context limit"
    AGENT_VALIDATION_FAILED = "agent validation failed"
    AGENT_INVALID_ACTION = "agent invalid action"
    TASK_LIMIT_REACHED = "task limit reached"
    UNKNOWN = "unknown"
    TASK_ERROR = "task error"


class _AgentOutputStatus:
    NORMAL = "normal"
    CANCELLED = "cancelled"
    AGENT_CONTEXT_LIMIT = "agent context limit"


# ---------------------------------------------------------------------------
# Benchmark adapter
# ---------------------------------------------------------------------------


class AgentBenchBenchmark:
    """AgentBench adapter that communicates with the official Task Server.

    Config keys (all optional):
        controller_address  Base URL of the controller (default localhost:5000/api).
        task_name           AgentBench task id, e.g. "os", "dbbench", "ltp",
                            "alfworld", "webshop", "card_game", "knowledgegraph".
        max_turns           Safety limit on interaction turns (default: 30).
        timeout             HTTP request timeout in seconds (default: 120).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.controller_address: str = str(
            cfg.get("controller_address", "http://localhost:5000/api")
        ).rstrip("/")
        self.task_name: str = str(cfg.get("task_name", "os"))
        self.max_turns: int = int(cfg.get("max_turns", 30))
        self.timeout: int = int(cfg.get("timeout", 120))

    # ------------------------------------------------------------------
    # load_tasks  –  GET /get_indices
    # ------------------------------------------------------------------

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        """Fetch available sample indices from the Task Server."""
        try:
            resp = requests.get(
                f"{self.controller_address}/get_indices",
                params={"name": self.task_name},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            indices = resp.json()
        except Exception as e:
            logger.error(f"Failed to connect to AgentBench controller: {e}")
            raise RuntimeError(
                f"Cannot reach AgentBench Task Server at {self.controller_address}. "
                "Make sure the controller is running:\n"
                "  cd AgentBench && python -m src.start_task -a"
            ) from e

        tasks: list[BenchmarkTask] = []
        for idx in indices:
            tasks.append(
                BenchmarkTask(
                    task_id=f"{self.task_name}_{idx}",
                    prompt="",  # built dynamically from history
                    reference_answer="",
                    metadata={
                        "agentbench_index": idx,
                        "task_name": self.task_name,
                    },
                )
            )
            if task_limit is not None and len(tasks) >= task_limit:
                break

        logger.info(f"Loaded {len(tasks)} AgentBench tasks for '{self.task_name}'")
        return tasks

    # ------------------------------------------------------------------
    # run  –  mirrors official TaskClient.run_sample
    # ------------------------------------------------------------------

    def run(
        self,
        task: BenchmarkTask,
        runner: MASRunner,
        run_index: int,
        seed: int,
    ) -> MASRunResult:
        """Execute one AgentBench sample via the controller API.

        Faithfully mirrors ``src/client/task.py :: TaskClient.run_sample``.
        """
        index = task.metadata["agentbench_index"]
        all_events: list = []
        aggregate_metadata = init_run_metadata_aggregate()

        # 1. POST /start_sample  ------------------------------------------
        start_payload = {"name": self.task_name, "index": index}
        try:
            resp = requests.post(
                f"{self.controller_address}/start_sample",
                json=start_payload,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.error(f"Network error on /start_sample: {e}")
            return MASRunResult(
                final_answer="",
                trace_events=all_events,
                run_metadata={
                    "agentbench_error": "NETWORK_ERROR",
                    "agentbench_info": str(e),
                    **aggregate_metadata,
                },
            )

        if resp.status_code == 406:
            logger.warning(f"Task not available (406): {resp.text}")
            return MASRunResult(
                final_answer="",
                trace_events=all_events,
                run_metadata={
                    "agentbench_error": "NOT_AVAILABLE",
                    "agentbench_info": resp.text,
                    **aggregate_metadata,
                },
            )
        if resp.status_code != 200:
            logger.error(f"start_sample failed ({resp.status_code}): {resp.text}")
            return MASRunResult(
                final_answer="",
                trace_events=all_events,
                run_metadata={
                    "agentbench_error": "START_FAILED",
                    "agentbench_info": resp.text,
                    **aggregate_metadata,
                },
            )

        result = resp.json()
        session_id: int = result["session_id"]
        task_output = result["output"]
        final_answer = ""
        turn = 0

        # 2. Interact loop  -----------------------------------------------
        # Official: while SampleStatus(result["output"]["status"]) == SampleStatus.RUNNING
        while task_output.get("status") == _SampleStatus.RUNNING:
            turn += 1
            if turn > self.max_turns:
                logger.warning(
                    f"Max turns ({self.max_turns}) exceeded for {task.task_id}; cancelling session."
                )
                self._cancel_session(session_id)
                break

            # Official history format: list of ChatHistoryItem {role, content}
            # role is "user" (environment) or "agent" (LLM)
            history: list[dict[str, str]] = task_output.get("history") or []

            # Convert AgentBench history → OpenAI-style messages for MASRunner
            prompt_messages = self._history_to_messages(history)

            # Call MAS runner
            step_task = BenchmarkTask(
                task_id=f"{task.task_id}_turn_{turn}",
                prompt=prompt_messages,
                reference_answer="",
                metadata=task.metadata,
            )

            try:
                mas_result = runner.run_task(task=step_task, run_index=run_index, seed=seed)
                all_events.extend(mas_result.trace_events)
                merge_step_run_metadata(
                    aggregate_metadata,
                    dict(mas_result.run_metadata),
                    outer_step_index=turn - 1,
                    step_task_id=step_task.task_id,
                    final_answer=mas_result.final_answer,
                )
                agent_content = mas_result.final_answer
                agent_response = {
                    "status": _AgentOutputStatus.NORMAL,
                    "content": agent_content,
                }
            except Exception as e:
                # Mirror official: on agent error → cancel session
                logger.error(f"Agent inference error: {e}")
                self._cancel_session(session_id)
                return MASRunResult(
                    final_answer=final_answer,
                    trace_events=all_events,
                    run_metadata={
                        "agentbench_error": "AGENT_FAILED",
                        "agentbench_info": str(e),
                        "agentbench_status": task_output.get("status"),
                        "agentbench_history": history,
                        **aggregate_metadata,
                    },
                )

            final_answer = agent_content

            # 3. POST /interact  -------------------------------------------
            interact_payload = {
                "session_id": session_id,
                "agent_response": agent_response,
            }
            try:
                resp = requests.post(
                    f"{self.controller_address}/interact",
                    json=interact_payload,
                    timeout=self.timeout,
                )
            except Exception as e:
                logger.error(f"Network error on /interact: {e}")
                return MASRunResult(
                    final_answer=final_answer,
                    trace_events=all_events,
                    run_metadata={
                        "agentbench_error": "NETWORK_ERROR",
                        "agentbench_info": str(e),
                        "agentbench_status": task_output.get("status"),
                        "agentbench_history": history,
                        **aggregate_metadata,
                    },
                )

            if resp.status_code != 200:
                logger.error(f"interact failed ({resp.status_code}): {resp.text}")
                self._cancel_session(session_id)
                return MASRunResult(
                    final_answer=final_answer,
                    trace_events=all_events,
                    run_metadata={
                        "agentbench_error": "INTERACT_FAILED",
                        "agentbench_info": resp.text,
                        "agentbench_status": task_output.get("status"),
                        "agentbench_history": history,
                        **aggregate_metadata,
                    },
                )

            result = resp.json()
            task_output = result["output"]

        # 4. Return final result  ------------------------------------------
        final_status = task_output.get("status", _SampleStatus.UNKNOWN)
        final_result = task_output.get("result")
        final_history = task_output.get("history") or []

        return MASRunResult(
            final_answer=final_answer,
            trace_events=all_events,
            run_metadata={
                "agentbench_status": final_status,
                "agentbench_result": final_result,
                "agentbench_history": final_history,
                "agentbench_turns": turn,
                "agentbench_session_id": session_id,
                **aggregate_metadata,
            },
        )

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        """Score based on the SampleStatus returned by the Task Server.

        Official scoring: ``SampleStatus.COMPLETED`` == success.
        """
        run_metadata = run_metadata or {}
        status = str(run_metadata.get("agentbench_status", "")).lower().strip()
        error = run_metadata.get("agentbench_error")

        # In AgentBench, "completed" means the task was solved correctly.
        # All other terminal statuses count as failure.
        success = status == _SampleStatus.COMPLETED and error is None
        score = 1.0 if success else 0.0

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=score,
            success=success,
            details={
                "agentbench_status": status,
                "agentbench_error": error,
                "agentbench_result": run_metadata.get("agentbench_result"),
                "agentbench_turns": run_metadata.get("agentbench_turns", 0),
                "prediction_preview": prediction[:500] if prediction else "",
            },
        )

    # ------------------------------------------------------------------
    # requirements
    # ------------------------------------------------------------------

    def requirements(self) -> dict[str, Any]:
        return {
            "benchmark": "agentbench",
            "paper": "https://arxiv.org/abs/2308.03688",
            "source": "https://github.com/THUDM/AgentBench",
            "task_name": self.task_name,
            "controller_address": self.controller_address,
            "notes": [
                "1. Clone https://github.com/THUDM/AgentBench",
                "2. Install: pip install -r requirements.txt",
                "3. Start controller+workers: python -m src.start_task -a",
                f"4. Or start specific task: python -m src.start_task "
                f"-c configs/tasks/{self.task_name}.yaml",
                "5. Docker is required for most tasks (os, dbbench, alfworld, etc.)",
                "6. Then run: uv run python main.py run "
                "--config test_agentbench.toml --benchmark agentbench",
            ],
            "available_tasks": [
                "os",
                "dbbench",
                "ltp",
                "alfworld",
                "webshop",
                "card_game",
                "knowledgegraph",
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cancel_session(self, session_id: int) -> None:
        """POST /cancel to release the session on the controller."""
        try:
            requests.post(
                f"{self.controller_address}/cancel",
                json={"session_id": session_id},
                timeout=10,
            )
        except Exception:
            logger.warning(f"Failed to cancel session {session_id}", exc_info=True)

    @staticmethod
    def _history_to_messages(
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Convert AgentBench ChatHistoryItem list to OpenAI-style messages.

        AgentBench uses:
            role="user"   → environment observation / prompt
            role="agent"  → previous agent response

        We map to:
            role="user"      → role="user"
            role="agent"     → role="assistant"
        """
        messages: list[dict[str, str]] = []
        for item in history:
            role = item.get("role", "user")
            content = item.get("content", "")
            if role == "agent":
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})
        return messages

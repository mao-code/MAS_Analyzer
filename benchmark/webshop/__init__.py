"""WebShop benchmark adapter.

Source repo : https://github.com/princeton-nlp/WebShop
Paper       : https://arxiv.org/abs/2207.01206

This adapter keeps the official WebShop product data, human-goal construction,
page/action flow, and reward semantics, but runs in-process inside this repo so
it can be driven by the existing MAS runtime and traced like the other
benchmarks.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

from descriptor.schema import TraceEvent

from ..base import (
    BenchmarkEvaluation,
    BenchmarkTask,
    init_run_metadata_aggregate,
    merge_step_run_metadata,
)
from .runtime import WebShopDataStore, parse_action

WEBSHOP_ACTION_SYSTEM_PROMPT = """\
You are interacting with the WebShop benchmark.

Return exactly one action and nothing else.

Valid action formats:
- search[keywords]
- click[value]

Rules:
- Use only one action per turn.
- `search[...]` is only valid when a search bar is available.
- `click[...]` must exactly match one of the available click actions.
- To purchase the current item, use `click[buy now]`.
- Do not explain your choice.
"""


class WebShopBenchmark:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.data_mode = str(cfg.get("data_mode", "small")).strip().lower()
        self.split = str(cfg.get("split", "test")).strip().lower()
        self.auto_download = bool(cfg.get("auto_download", True))
        self.human_goals = bool(cfg.get("human_goals", True))
        self.max_steps = max(1, int(cfg.get("max_steps", 15)))
        self.history_window = max(0, int(cfg.get("history_window", 4)))
        self.show_attrs = bool(cfg.get("show_attrs", False))

        num_products_cfg = cfg.get("num_products")
        if num_products_cfg in (None, "", False):
            self.num_products = 1000 if self.data_mode == "small" else None
        else:
            self.num_products = int(num_products_cfg)

        from pathlib import Path

        self.data_root = Path(str(cfg.get("data_dir", "benchmark/webshop/data"))).expanduser()
        self._store = WebShopDataStore(
            data_root=self.data_root,
            data_mode=self.data_mode,
            auto_download=self.auto_download,
            human_goals=self.human_goals,
            num_products=self.num_products,
            show_attrs=self.show_attrs,
        )

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        goals = self._store.goals_for_split(self.split)
        tasks: list[BenchmarkTask] = []
        for goal in goals:
            tasks.append(
                BenchmarkTask(
                    task_id=f"webshop_{self.split}_{goal.goal_index:04d}",
                    prompt=goal.instruction_text,
                    reference_answer=goal.asin,
                    metadata=goal.to_metadata(),
                )
            )
            if task_limit is not None and len(tasks) >= task_limit:
                break
        return tasks

    def run(
        self,
        task: BenchmarkTask,
        runner: Any,
        run_index: int,
        seed: int,
    ):
        goal_index = int(task.metadata["goal_index"])
        episode = self._store.make_episode(goal_index)

        trace_events: list[TraceEvent] = []
        aggregate_tool_counts: Counter[str] = Counter()
        aggregate_retrieved_docids: set[str] = set()
        total_messages_sent = 0
        action_history: list[dict[str, Any]] = []
        aggregate_metadata = init_run_metadata_aggregate()

        last_raw_output = ""
        last_action = ""

        for step_index in range(self.max_steps):
            observation = episode.observation_text()
            available_actions = episode.available_actions_info()
            prompt = self._build_prompt(
                task=task,
                observation=observation,
                available_actions=available_actions,
                action_history=action_history,
                step_index=step_index,
                current_page_name=episode.page_name,
            )

            step_task = BenchmarkTask(
                task_id=f"{task.task_id}_step_{step_index}",
                prompt=prompt,
                reference_answer="",
                metadata={
                    **dict(task.metadata),
                    "step_index": step_index,
                    "page_name": episode.page_name,
                    "available_actions": available_actions,
                },
            )

            result = runner.run_task(task=step_task, run_index=run_index, seed=seed + step_index, benchmark_name="webshop")
            trace_events.extend(result.trace_events)

            run_metadata = dict(result.run_metadata)
            merge_step_run_metadata(
                aggregate_metadata,
                run_metadata,
                outer_step_index=step_index,
                step_task_id=step_task.task_id,
                final_answer=result.final_answer,
            )
            aggregate_tool_counts.update(
                {str(name): int(count) for name, count in run_metadata.get("tool_call_counts", {}).items()}
            )
            aggregate_retrieved_docids.update(
                str(docid) for docid in run_metadata.get("retrieved_docids", []) if str(docid)
            )
            total_messages_sent += int(run_metadata.get("messages_sent_total", 0))

            last_raw_output = str(result.final_answer or "").strip()
            last_action = self._extract_action(last_raw_output, available_actions)

            next_observation, reward, done, info = episode.step(last_action)
            action_record = {
                "step_index": step_index,
                "page_name": info.get("page_name"),
                "model_output": last_raw_output,
                "parsed_action": last_action,
                "reward": float(reward),
                "paper_score_100": float(info.get("paper_score_100", 0.0)),
                "done": bool(done),
                "available_actions": available_actions,
            }
            action_history.append(action_record)

            trace_events.append(
                self._env_event(
                    trace_events=trace_events,
                    step_index=step_index,
                    payload={
                        "page_name": info.get("page_name"),
                        "parsed_action": last_action,
                        "raw_output": last_raw_output,
                        "reward": float(reward),
                        "paper_score_100": float(info.get("paper_score_100", 0.0)),
                        "done": bool(done),
                        "invalid_actions": int(info.get("invalid_actions", 0)),
                        "next_observation_preview": next_observation[:240],
                    },
                )
            )

            if done:
                break

        from MAS.runner import MASRunResult

        final_info = episode.info()
        return MASRunResult(
            final_answer=last_action or last_raw_output,
            trace_events=trace_events,
            run_metadata={
                "benchmark": "webshop",
                "data_mode": self.data_mode,
                "split": self.split,
                "goal_index": goal_index,
                "goal": dict(task.metadata),
                "steps_taken": len(action_history),
                "max_steps": self.max_steps,
                "terminated": bool(final_info.get("done", False)),
                "step_limit_reached": not bool(final_info.get("done", False)),
                "final_reward": float(final_info.get("reward", 0.0)),
                "paper_score_100": float(final_info.get("paper_score_100", 0.0)),
                "selected_asin": final_info.get("asin"),
                "selected_options": dict(final_info.get("selected_options", {})),
                "invalid_actions": int(final_info.get("invalid_actions", 0)),
                "action_counts": dict(final_info.get("action_counts", {})),
                "tool_call_counts": dict(aggregate_tool_counts),
                "tool_calls_total": int(sum(aggregate_tool_counts.values())),
                "messages_sent_total": total_messages_sent,
                "retrieved_docids": sorted(aggregate_retrieved_docids),
                "reward_info": dict(final_info.get("reward_info", {})),
                "action_history": action_history,
                **aggregate_metadata,
            },
        )

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        run_metadata = run_metadata or {}
        final_reward = float(run_metadata.get("final_reward", 0.0))
        success = math_isclose(final_reward, 1.0)
        reward_info = dict(run_metadata.get("reward_info", {}))

        details = {
            "instruction_text": task.metadata.get("instruction_text", ""),
            "goal_asin": task.metadata.get("asin", ""),
            "selected_asin": run_metadata.get("selected_asin"),
            "selected_options": dict(run_metadata.get("selected_options", {})),
            "prediction": prediction,
            "score": final_reward,
            "paper_score_100": float(run_metadata.get("paper_score_100", final_reward * 100.0)),
            "success": success,
            "reward_info": reward_info,
            "steps_taken": int(run_metadata.get("steps_taken", 0)),
            "invalid_actions": int(run_metadata.get("invalid_actions", 0)),
            "run_metadata": run_metadata,
        }
        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=final_reward,
            success=success,
            details=details,
        )

    def requirements(self) -> dict[str, Any]:
        return {
            "benchmark": "webshop",
            "source_repo": "https://github.com/princeton-nlp/WebShop",
            "dataset": (
                "Official WebShop product and human-instruction assets "
                f"(data_mode={self.data_mode}, split={self.split})"
            ),
            "notes": [
                "Runs WebShop as an interactive action loop through the MAS runtime.",
                "Default script uses the official small asset set for manageable smoke tests.",
                "Switching to full mode downloads the full corpus and is substantially heavier on disk and runtime.",
            ],
        }

    def _build_prompt(
        self,
        *,
        task: BenchmarkTask,
        observation: str,
        available_actions: dict[str, Any],
        action_history: list[dict[str, Any]],
        step_index: int,
        current_page_name: str,
    ) -> list[dict[str, str]]:
        recent_history = action_history[-self.history_window :] if self.history_window > 0 else []
        if recent_history:
            history_lines = [
                (
                    f"Step {item['step_index']}: action={item['parsed_action']} "
                    f"reward={item['reward']:.4f} done={item['done']}"
                )
                for item in recent_history
            ]
            history_text = "\n".join(history_lines)
        else:
            history_text = "None"

        clickables = available_actions.get("clickables", [])
        click_text = ", ".join(clickables) if clickables else "None"
        user_prompt = (
            f"Instruction:\n{task.metadata.get('instruction_text', task.prompt)}\n\n"
            f"Current step: {step_index + 1} / {self.max_steps}\n"
            f"Current page: {current_page_name}\n\n"
            f"Observation:\n{observation}\n\n"
            "Available actions:\n"
            f"- search bar available: {bool(available_actions.get('has_search_bar', False))}\n"
            f"- click actions: {click_text}\n\n"
            f"Recent history:\n{history_text}\n\n"
            "Return exactly one valid action."
        )
        return [
            {"role": "system", "content": WEBSHOP_ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _extract_action(model_output: str, available_actions: dict[str, Any]) -> str:
        text = str(model_output or "").strip()
        action_name, action_arg = parse_action(text)

        if action_name in {"search", "click"} and action_arg is not None:
            if action_name == "search":
                return f"search[{action_arg}]"
            resolved = WebShopBenchmark._resolve_click_value(
                action_arg, list(available_actions.get("clickables", []))
            )
            return f"click[{resolved}]"

        if bool(available_actions.get("has_search_bar", False)):
            query = text
            lowered = query.lower()
            if lowered.startswith("search:"):
                query = query.split(":", 1)[1].strip()
            if lowered.startswith("search "):
                query = query[7:].strip()
            query = query.strip().strip("`").strip('"').strip("'")
            return f"search[{query}]"

        resolved = WebShopBenchmark._resolve_click_value(text, list(available_actions.get("clickables", [])))
        return f"click[{resolved}]"

    @staticmethod
    def _resolve_click_value(candidate: str, clickables: list[str]) -> str:
        candidate_norm = " ".join(str(candidate or "").strip().lower().split())
        if not clickables:
            return candidate_norm

        by_norm = {" ".join(item.strip().lower().split()): item for item in clickables}
        if candidate_norm in by_norm:
            return by_norm[candidate_norm]

        for item in clickables:
            item_norm = " ".join(item.strip().lower().split())
            if item_norm and (item_norm in candidate_norm or candidate_norm in item_norm):
                return item

        lowered = candidate_norm.lower()
        for item in clickables:
            if item.lower() in lowered:
                return item

        return clickables[0]

    @staticmethod
    def _env_event(
        *,
        trace_events: list[TraceEvent],
        step_index: int,
        payload: dict[str, Any],
    ) -> TraceEvent:
        start = time.time()
        if trace_events:
            start = max(start, trace_events[-1].timestamp_end + 1e-6)
        end = start + 1e-6
        return TraceEvent(
            timestamp_start=start,
            timestamp_end=end,
            actor="environment",
            event_type="verify",
            payload={"step_index": step_index, **payload},
            token_in=0,
            token_out=0,
            latency_ms=0.001,
            cost_usd=0.0,
            state_id=f"webshop_env_step_{step_index}",
        )


def math_isclose(value: float, target: float) -> bool:
    return abs(float(value) - float(target)) <= 1e-9

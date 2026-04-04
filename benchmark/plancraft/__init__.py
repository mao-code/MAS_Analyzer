"""PlanCraft benchmark adapter.

Source repo : https://github.com/gautierdag/plancraft
Paper       : https://arxiv.org/abs/2412.21033
Package     : pip install plancraft

PlanCraft is a Minecraft crafting benchmark.
In the current upstream `main` dataset (plancraft 0.4.9):
    - val: 570 examples
    - test: 580 examples
    - additional splits exist (e.g. val.small, test.small, *.easy)
Each example has:
    - A target item to craft (e.g. "cyan_stained_glass_pane")
    - An initial inventory with items in specific slots
    - A recipe type (shaped / shapeless / smelting) and complexity (number of
      sub-steps)
    - Some tasks are intentionally impossible (agent should output "impossible")

The agent interacts via text:
    - Observation: text inventory + target item description
    - Action: "move: from [I2] to [A1] with quantity 3"
            | "smelt: from [I5] to [I6] with quantity 1"
            | "impossible: <reason>"

Evaluation:
    - Success = agent crafts the target item (or correctly identifies impossible)
    - Scored per-step for trace metrics

This adapter handles load_tasks() and evaluate(). The actual multi-step
interaction loop must be driven by the runner.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from benchmark.base import (
    BenchmarkEvaluation,
    BenchmarkTask,
    init_run_metadata_aggregate,
    merge_step_run_metadata,
)

if TYPE_CHECKING:
    from MAS.runner import MASRunResult

logger = logging.getLogger(__name__)


class PlancraftBenchmark:
    """PlanCraft adapter for MAS_Analyzer.

    Config keys (all optional):
        split           Dataset split name (default: "val"), e.g.
                        "val", "test", "val.small", "test.small"
        max_steps       Max steps per episode (default: 30)
        resolution      Image resolution "high" or "low" (default: "high")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.split: str = str(cfg.get("split", "val"))
        self.max_steps: int = int(cfg.get("max_steps", 30))
        self.resolution: str = str(cfg.get("resolution", "high"))

        # Cache examples for get_environment()
        self._examples: list | None = None
        self._system_prompt_text: str = ""

        try:
            from plancraft.environment.actions import (
                ImpossibleActionHandler,
                MoveActionHandler,
                SmeltActionHandler,
            )
            from plancraft.environment.prompts import (
                get_prompt_example,
                get_system_prompt,
            )

            handlers = [MoveActionHandler(), SmeltActionHandler(), ImpossibleActionHandler()]
            self._system_prompt_text = get_system_prompt(handlers)["content"]

            # Retrieve official few-shot examples as dict array
            self._few_shot_messages = get_prompt_example(handlers, use_text_inventory=True)
        except ImportError:
            logger.warning("Failed to import PlanCraft system prompt tools.")
            self._few_shot_messages = []

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        from plancraft.simple import get_plancraft_examples

        examples = get_plancraft_examples(split=self.split)
        self._examples = examples

        tasks: list[BenchmarkTask] = []
        for ex in examples:
            metadata = {
                "source": "plancraft",
                "recipe_type": ex.recipe_type,
                "complexity": ex.complexity,
                "complexity_split": getattr(ex, "complexity_split", ""),
                "optimal_path_length": getattr(ex, "optimal_path_length", None),
                "impossible": ex.impossible,
                "target": ex.target,
                "inventory": dict(ex.slotted_inventory),
            }
            tasks.append(
                BenchmarkTask(
                    task_id=str(ex.id),
                    prompt=f"Craft an item of type: {ex.target}",
                    reference_answer=("impossible" if ex.impossible else ex.target),
                    metadata=metadata,
                )
            )
            if task_limit is not None and len(tasks) >= task_limit:
                break

        return tasks

    def get_environment(self, task: BenchmarkTask):
        """Create a PlanCraft Gym environment for the given task.

        Returns a PlancraftGymWrapper ready for step() calls.
        """
        from plancraft.simple import PlancraftGymWrapper

        if self._examples is None:
            from plancraft.simple import get_plancraft_examples

            self._examples = get_plancraft_examples(split=self.split)

        example = None
        for ex in self._examples:
            if str(ex.id) == task.task_id:
                example = ex
                break

        if example is None:
            raise ValueError(f"PlanCraft example not found for task_id={task.task_id}")

        return PlancraftGymWrapper(
            example=example,
            max_steps=self.max_steps,
            resolution=self.resolution,
            use_text_inventory=True,
        )

    def run(
        self,
        task: BenchmarkTask,
        runner: Any,
        run_index: int,
        seed: int,
    ) -> MASRunResult:
        from MAS.runner import MASRunResult

        env = self.get_environment(task)
        obs, reward, terminated, truncated, info = env.step("")

        all_events = []
        final_action = ""
        done = terminated or truncated
        aggregate_metadata = init_run_metadata_aggregate()
        raw_outputs: list[str] = []

        # Build a persistent history matching official dialogue layout
        dialogue_history = []

        while not done:
            current_obs_text = obs.get("text", "")

            # Construct the identical list of messages expected by official evaluators
            full_prompt: list[dict[str, str]] = [
                {"role": "system", "content": self._system_prompt_text}
            ]
            full_prompt.extend(self._few_shot_messages)
            full_prompt.extend(dialogue_history)
            full_prompt.append({"role": "user", "content": current_obs_text})

            # Create a localized task object for this step's LLM generation
            step_task = BenchmarkTask(
                task_id=task.task_id,
                prompt=full_prompt,
                reference_answer=task.reference_answer,
                metadata=task.metadata,
            )

            # Delegate exactly one step of generation to the MAS
            mas_result = runner.run_task(task=step_task, run_index=run_index, seed=seed, benchmark_name="plancraft")
            all_events.extend(mas_result.trace_events)
            merge_step_run_metadata(
                aggregate_metadata,
                dict(mas_result.run_metadata),
                outer_step_index=len(dialogue_history) // 2,
                step_task_id=step_task.task_id,
                final_answer=mas_result.final_answer,
            )
            raw_outputs.append(str(mas_result.final_answer or ""))

            final_action = mas_result.final_answer.strip()

            # Record the step for context in subsequent turns
            dialogue_history.append({"role": "user", "content": current_obs_text})
            dialogue_history.append({"role": "assistant", "content": final_action})

            obs, reward, terminated, truncated, info = env.step(final_action)
            done = terminated or truncated

        return MASRunResult(
            final_answer=final_action,
            trace_events=all_events,
            run_metadata={
                "reward": reward,
                "num_steps": info.get("steps", 0),
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
                "raw_outputs": raw_outputs,
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

        # Determine success
        reward = float(run_metadata.get("reward", 0.0))
        num_steps = int(run_metadata.get("num_steps", 0))
        terminated = bool(run_metadata.get("terminated", False))
        truncated = bool(run_metadata.get("truncated", False))

        is_impossible = task.metadata.get("impossible", False)

        # Success conditions:
        # 1. Positive reward (crafted the target)
        # 2. Correctly identified impossible task
        if reward > 0 or is_impossible and prediction.strip().lower() == "impossible":
            success = True
        else:
            success = False

        score = 1.0 if success else 0.0

        details: dict[str, Any] = {
            "prediction": prediction,
            "reference_answer": task.reference_answer,
            "reward": reward,
            "num_steps": num_steps,
            "terminated": terminated,
            "truncated": truncated,
            "impossible": is_impossible,
            "recipe_type": task.metadata.get("recipe_type", ""),
            "complexity": task.metadata.get("complexity", 0),
            "complexity_split": task.metadata.get("complexity_split", ""),
            "run_metadata": run_metadata,
        }

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=score,
            success=success,
            details=details,
        )

    def requirements(self) -> dict[str, Any]:
        return {
            "benchmark": "plancraft",
            "version": "0.4.9",
            "package": "pip install plancraft",
            "dataset": f"Built-in ({self.split} split)",
            "metrics": [
                "success (crafted target or correctly identified impossible)",
                "num_steps (steps taken)",
                "per recipe_type and complexity breakdown",
            ],
            "notes": [
                "PlanCraft is a multi-step interactive environment.",
                "The runner must call get_environment() and drive the step loop.",
                "Adapter uses PlancraftGymWrapper + official prompt helpers (not full Evaluator parity).",
                "Agent actions: move, smelt, impossible.",
                "Upstream README (2025-11-07) notes original paper baseline scores were underreported; re-run baselines for fair comparison.",
            ],
        }

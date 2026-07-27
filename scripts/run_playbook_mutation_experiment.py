#!/usr/bin/env python3
"""Run the combined long-term-playbook transfer and mutation-budget experiment.

The experiment has two sequential phases:

1. Learn one isolated long-term playbook per source benchmark from 30 runs,
   reflecting process-only outcomes after every 10 runs.
2. Freeze those playbooks and evaluate every source -> target transfer at each
   configured mutation budget. A frozen copy of the initial skill is included as
   the no-learning control.

Completed task artifacts and reflection state are checkpoints. Re-running ``run``
with the same arguments resumes without repeating completed work.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

TASKS_PER_BENCHMARK = 30
RUNS_PER_TASK = 1
REFLECTION_BATCH_SIZE = 10
DEFAULT_MANIFEST = ROOT / "config" / "manta_ablation_tasks_30_seed42.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "full_experiment"
DEFAULT_EXPERIMENT_ID = "playbook_transfer_mutation_curve_pw_30_seed42_batch10"
DEFAULT_MODEL = "google/gemma-4-31b-it:nitro"
DEFAULT_BENCHMARKS = ("workbench", "plancraft")
DEFAULT_MUTATION_BUDGETS = (0, 1, 2, 3)
SEED_SKILL = ROOT / "config" / "topology_skill.md"
PLAYBOOK = ROOT / "config" / "topology_playbook.json"

BENCHMARK_TOML = {
    "workbench": """
[workbench]
domain = "multi_domain"
tool_selection = "domains"
max_tool_iterations = 20
tasks_version = "v1"
""",
    "plancraft": """
[plancraft]
split = "val.small"
max_steps = 30
resolution = "high"
""",
}

_STOP_REQUESTED = False
_CURRENT_CHILD: subprocess.Popen[str] | None = None


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_names(value: str, *, option: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    invalid = sorted(set(names) - set(BENCHMARK_TOML))
    if not names or invalid:
        raise ValueError(
            f"{option} must be a non-empty comma-separated subset of "
            f"{sorted(BENCHMARK_TOML)}; invalid={invalid}"
        )
    return names


def _parse_budgets(value: str) -> tuple[int, ...]:
    try:
        budgets = tuple(sorted(set(int(part.strip()) for part in value.split(",") if part.strip())))
    except ValueError as exc:
        raise ValueError("--mutation-budgets must contain comma-separated integers") from exc
    if not budgets or budgets[0] < 0 or budgets[-1] > 9:
        raise ValueError(
            "--mutation-budgets must be non-empty and between 0 and 9 "
            "(self_evolved.max_turns is capped at 10)"
        )
    return budgets


def _design(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    sources = _parse_names(args.source_benchmarks, option="--source-benchmarks")
    targets = _parse_names(args.target_benchmarks, option="--target-benchmarks")
    budgets = _parse_budgets(args.mutation_budgets)
    return sources, targets, budgets


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict) or int(payload.get("seed", -1)) != 42:
        raise ValueError(f"Invalid seed-42 task manifest: {path}")
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise ValueError(f"Manifest has no benchmark mapping: {path}")
    for benchmark in BENCHMARK_TOML:
        task_ids = benchmarks.get(benchmark, {}).get("task_ids", [])
        if len(task_ids) != TASKS_PER_BENCHMARK or len(set(map(str, task_ids))) != (
            TASKS_PER_BENCHMARK
        ):
            raise ValueError(
                f"Manifest must contain {TASKS_PER_BENCHMARK} unique {benchmark} task ids"
            )
    return payload


def _experiment_root(args: argparse.Namespace) -> Path:
    return Path(args.output_root).expanduser().resolve() / str(args.experiment_id)


def _state_root(args: argparse.Namespace) -> Path:
    return _experiment_root(args) / "_playbook_mutation_state"


def _training_skill(state_root: Path, source: str) -> Path:
    return state_root / "skills" / "training" / source / "topology_skill.md"


def _frozen_skill(state_root: Path, source: str) -> Path:
    return state_root / "skills" / "frozen" / source / "topology_skill.md"


def _config_path(
    state_root: Path,
    *,
    phase: str,
    source: str,
    target: str,
    budget: int,
) -> Path:
    return state_root / "configs" / phase / source / target / f"mutation_{budget}.toml"


def _system_label(*, phase: str, source: str, budget: int) -> str:
    if phase == "learning":
        return f"learn__{source}"
    return f"eval__pb_{source}__mut_{budget}"


def evaluation_cells(
    sources: tuple[str, ...],
    targets: tuple[str, ...],
    budgets: tuple[int, ...],
) -> list[tuple[str, str, int]]:
    playbook_sources = ("seed", *sources)
    return [
        (source, target, budget)
        for target in targets
        for source in playbook_sources
        for budget in budgets
    ]


def _render_config(
    *,
    model: str,
    benchmark: str,
    skill_path: Path,
    mutation_budget: int,
) -> str:
    max_turns = mutation_budget + 1
    return f"""[openrouter]
api_key = ""
base_url = "https://openrouter.ai/api/v1"
timeout_s = 600

[experiment]
output_dir = "artifacts/benchmark_traces/playbook_mutation"
runs_per_task = {RUNS_PER_TASK}
seed = 42

[models]
default = {json.dumps(model)}

[mas]
levels = 1
intra_level_link_ratio = 1.0
full_linked = true
topology = "self_evolved"
number_of_agents = 5
agent_types = ["general"]
communication_count_internally = 2
turn_mode = {json.dumps("single_turn" if max_turns == 1 else "multi_turn")}
max_turns = {max_turns}
discussion_rounds = 1
minimum_discussion_rounds = 1
termination_consensus_mode = "llm_judge"
final_vote_mode = "llm_judge"
peer_artifact_max_chars = 0
enable_dynamic_roles = true

[self_evolved]
harness_backend = "openrouter"
initial_planner_mode = "task_conditioned"
max_initial_agents = 5
max_total_agents = 10
max_turns = {max_turns}
repair_budget = {mutation_budget}
audit_mode = "hybrid"
playbook_path = {json.dumps(str(PLAYBOOK.resolve()))}
skill_path = {json.dumps(str(skill_path.resolve()))}
playbook_read = true
# This driver owns resumable batch reflection during learning. Evaluation is frozen.
skill_update_batch_size = 0
default_packet_max_chars = 0

{BENCHMARK_TOML[benchmark].strip()}
"""


def prepare(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    sources, targets, budgets = _design(args)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = _load_manifest(manifest_path)
    state_root = _state_root(args)
    state_root.mkdir(parents=True, exist_ok=True)

    training_budget = (
        int(args.training_mutation_budget)
        if args.training_mutation_budget is not None
        else max(budgets)
    )
    if not 0 <= training_budget <= 9:
        raise ValueError("--training-mutation-budget must be between 0 and 9")

    run_manifest_path = state_root / "experiment_manifest.json"
    existing = _read_json(run_manifest_path)
    requested = {
        "experiment_id": str(args.experiment_id),
        "seed": 42,
        "runs_per_task": RUNS_PER_TASK,
        "tasks_per_benchmark": TASKS_PER_BENCHMARK,
        "reflection_batch_size": REFLECTION_BATCH_SIZE,
        "model": str(args.model),
        "source_benchmarks": list(sources),
        "target_benchmarks": list(targets),
        "mutation_budgets": list(budgets),
        "training_mutation_budget": training_budget,
        "seed_control": True,
        "task_manifest_path": str(manifest_path),
        "task_manifest_sha256": _sha256(manifest_path),
        "seed_skill_path": str(SEED_SKILL.resolve()),
        "seed_skill_sha256": _sha256(SEED_SKILL),
        "evaluation_cells": len(evaluation_cells(sources, targets, budgets)),
        "learning_runs": len(sources) * TASKS_PER_BENCHMARK,
        "evaluation_runs": (len(evaluation_cells(sources, targets, budgets)) * TASKS_PER_BENCHMARK),
    }
    if isinstance(existing, dict):
        # Resume against the immutable snapshot copied at experiment creation, even if
        # another experiment subsequently updates the repository's canonical skill.
        requested["seed_skill_path"] = existing.get("seed_skill_path")
        requested["seed_skill_sha256"] = existing.get("seed_skill_sha256")
    if existing is not None and existing != requested:
        raise RuntimeError(
            f"Existing experiment settings differ from this command: {run_manifest_path}. "
            "Use the original arguments or a new --experiment-id."
        )
    if existing is None:
        _write_json(run_manifest_path, requested)

    seed_frozen = _frozen_skill(state_root, "seed")
    if not seed_frozen.exists():
        _copy_atomic(SEED_SKILL, seed_frozen)
    for source in sources:
        skill = _training_skill(state_root, source)
        if not skill.exists():
            _copy_atomic(seed_frozen, skill)
        frozen = _frozen_skill(state_root, source)
        if not frozen.exists():
            # Placeholder only; evaluation starts after _freeze_source replaces it.
            _copy_atomic(seed_frozen, frozen)

        learning_config = _config_path(
            state_root,
            phase="learning",
            source=source,
            target=source,
            budget=training_budget,
        )
        _write_config_once(
            learning_config,
            _render_config(
                model=str(args.model),
                benchmark=source,
                skill_path=skill,
                mutation_budget=training_budget,
            ),
        )

    for source, target, budget in evaluation_cells(sources, targets, budgets):
        config_path = _config_path(
            state_root,
            phase="evaluation",
            source=source,
            target=target,
            budget=budget,
        )
        _write_config_once(
            config_path,
            _render_config(
                model=str(args.model),
                benchmark=target,
                skill_path=_frozen_skill(state_root, source),
                mutation_budget=budget,
            ),
        )

    return manifest, state_root


def _write_config_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Generated config changed for an existing experiment: {path}. "
                "Use a new --experiment-id."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _task_root(
    args: argparse.Namespace,
    *,
    benchmark: str,
    phase: str,
    source: str,
    budget: int,
    task_id: str,
) -> Path:
    return (
        _experiment_root(args)
        / benchmark
        / _system_label(phase=phase, source=source, budget=budget)
        / task_id
    )


def _task_complete(
    args: argparse.Namespace,
    *,
    benchmark: str,
    phase: str,
    source: str,
    budget: int,
    task_id: str,
) -> bool:
    root = _task_root(
        args,
        benchmark=benchmark,
        phase=phase,
        source=source,
        budget=budget,
        task_id=task_id,
    )
    summary = _read_json(root / "task_summary.json")
    metadata = _read_json(root / "run_0.metadata.json")
    evaluation = _read_json(root / "run_0.eval.json")
    return bool(
        isinstance(summary, dict)
        and (bool(args.allow_mock) or not bool(summary.get("needs_rerun", True)))
        and isinstance(metadata, dict)
        and str(metadata.get("run_status", "completed")) == "completed"
        and isinstance(evaluation, dict)
        and evaluation.get("success") is not None
    )


def _reflection_state_path(state_root: Path, source: str) -> Path:
    return state_root / "reflection" / f"{source}.json"


def _load_reflection_state(state_root: Path, source: str) -> dict[str, Any]:
    state = _read_json(_reflection_state_path(state_root, source), default={})
    if not isinstance(state, dict):
        state = {}
    state.setdefault("seen_run_keys", [])
    state.setdefault("pending", [])
    state.setdefault("updates", [])
    state.setdefault("inflight", None)
    return state


def _save_reflection_state(state_root: Path, source: str, state: dict[str, Any]) -> None:
    _write_json(_reflection_state_path(state_root, source), state)


def _record_learning_candidate(
    *,
    args: argparse.Namespace,
    state_root: Path,
    source: str,
    budget: int,
    task_id: str,
) -> None:
    from MAS.self_evolved.skill import summary_from_candidate

    run_key = f"{source}/{task_id}/run_0"
    state = _load_reflection_state(state_root, source)
    if run_key in set(map(str, state["seen_run_keys"])):
        return
    metadata_path = (
        _task_root(
            args,
            benchmark=source,
            phase="learning",
            source=source,
            budget=budget,
            task_id=task_id,
        )
        / "run_0.metadata.json"
    )
    metadata = _read_json(metadata_path)
    candidate = ((metadata or {}).get("self_evolved") or {}).get("playbook_update_candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError(f"Completed learning run has no playbook candidate: {run_key}")
    state["seen_run_keys"].append(run_key)
    state["pending"].append({"run_key": run_key, "summary": summary_from_candidate(candidate)})
    _save_reflection_state(state_root, source, state)


def _recover_inflight_reflection(state_root: Path, source: str) -> None:
    state = _load_reflection_state(state_root, source)
    inflight = state.get("inflight")
    if not isinstance(inflight, dict):
        return
    skill = _training_skill(state_root, source)
    run_keys = list(map(str, inflight.get("run_keys", [])))
    before_sha = str(inflight.get("skill_before_sha256", ""))
    current_sha = _sha256(skill)
    if current_sha != before_sha:
        pending_keys = [str(row.get("run_key", "")) for row in state["pending"]]
        if pending_keys[: len(run_keys)] != run_keys:
            raise RuntimeError(f"Reflection recovery state is inconsistent for source={source}")
        state["pending"] = state["pending"][len(run_keys) :]
        state["updates"].append(
            {
                "run_keys": run_keys,
                "changed": True,
                "reason": "recovered_after_skill_write",
                "skill_sha256": current_sha,
            }
        )
    # If the hash is unchanged, either reflection had no update or it did not finish.
    # Keeping pending intact and clearing inflight makes retry safe.
    state["inflight"] = None
    _save_reflection_state(state_root, source, state)


def _reflect_pending(
    *,
    args: argparse.Namespace,
    state_root: Path,
    source: str,
    config_path: Path,
) -> None:
    from MAS.config import load_experiment_config
    from MAS.llm import OpenRouterLLMClient
    from MAS.self_evolved.skill import SkillReflector, TopologySkill

    _recover_inflight_reflection(state_root, source)
    state = _load_reflection_state(state_root, source)
    while len(state["pending"]) >= REFLECTION_BATCH_SIZE:
        batch = list(state["pending"][:REFLECTION_BATCH_SIZE])
        skill_path = _training_skill(state_root, source)
        state["inflight"] = {
            "run_keys": [str(row["run_key"]) for row in batch],
            "skill_before_sha256": _sha256(skill_path),
        }
        _save_reflection_state(state_root, source, state)

        config = load_experiment_config(config_path)
        client = OpenRouterLLMClient(config.openrouter, config.models)
        result = SkillReflector(client, config.self_evolved).reflect(
            current_skill=TopologySkill.load(skill_path).text,
            run_summaries=[dict(row["summary"]) for row in batch],
        )
        if result.changed:
            TopologySkill.load(skill_path).save(result.skill_markdown)

        state = _load_reflection_state(state_root, source)
        pending_keys = [str(row.get("run_key", "")) for row in state["pending"]]
        batch_keys = [str(row["run_key"]) for row in batch]
        if pending_keys[:REFLECTION_BATCH_SIZE] != batch_keys:
            raise RuntimeError(f"Reflection state changed unexpectedly for source={source}")
        state["pending"] = state["pending"][REFLECTION_BATCH_SIZE:]
        state["updates"].append(
            {
                "run_keys": batch_keys,
                "changed": bool(result.changed),
                "reason": str(result.reason),
                "skill_sha256": _sha256(skill_path),
                "llm": dict(result.llm),
            }
        )
        state["inflight"] = None
        _save_reflection_state(state_root, source, state)
        print(
            f"[reflection] source={source} runs={len(batch)} "
            f"update={len(state['updates'])} changed={result.changed} reason={result.reason}",
            flush=True,
        )


def _freeze_source(state_root: Path, source: str, *, expected_updates: int) -> None:
    state = _load_reflection_state(state_root, source)
    if state["pending"] or state["inflight"] or len(state["updates"]) != expected_updates:
        raise RuntimeError(
            f"Cannot freeze source={source}: pending={len(state['pending'])}, "
            f"inflight={bool(state['inflight'])}, updates={len(state['updates'])}/"
            f"{expected_updates}"
        )
    source_skill = _training_skill(state_root, source)
    frozen_skill = _frozen_skill(state_root, source)
    _copy_atomic(source_skill, frozen_skill)
    _write_json(
        frozen_skill.parent / "freeze.json",
        {
            "source_benchmark": source,
            "learning_runs": TASKS_PER_BENCHMARK,
            "reflection_updates": expected_updates,
            "reflection_batch_size": REFLECTION_BATCH_SIZE,
            "sha256": _sha256(frozen_skill),
        },
    )


def _python() -> str:
    candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate if candidate.exists() else Path(sys.executable))


def _run_command(
    *,
    args: argparse.Namespace,
    config_path: Path,
    benchmark: str,
    phase: str,
    source: str,
    budget: int,
    task_ids: list[str],
) -> list[str]:
    return [
        _python(),
        "main.py",
        "run",
        "--config",
        str(config_path),
        "--benchmark",
        benchmark,
        "--output-dir",
        str(Path(args.output_root).expanduser().resolve()),
        "--output-layout",
        "hierarchical",
        "--experiment-id",
        str(args.experiment_id),
        "--system-label",
        _system_label(phase=phase, source=source, budget=budget),
        "--topology",
        "self_evolved",
        "--agents",
        "5",
        "--mas-rounds",
        str(budget + 1),
        "--discussion-rounds",
        "1",
        "--communication-budget",
        "2",
        "--skill-update-batch-size",
        "0",
        "--task-ids",
        ",".join(task_ids),
        "--runs-per-task",
        str(RUNS_PER_TASK),
        "--seed",
        "42",
    ]


def _run_child(command: list[str], log_path: Path, *, allow_mock: bool) -> int:
    global _CURRENT_CHILD
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if allow_mock:
        env.pop("MAS_REQUIRE_LIVE_LLM", None)
    else:
        env["MAS_REQUIRE_LIVE_LLM"] = "1"
        env.pop("MAS_DISABLE_LIVE_LLM", None)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        _CURRENT_CHILD = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        next_update = started + 60
        while _CURRENT_CHILD.poll() is None:
            if _STOP_REQUESTED:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(_CURRENT_CHILD.pid, signal.SIGTERM)
            now = time.monotonic()
            if now >= next_update:
                print(
                    f"[running] pid={_CURRENT_CHILD.pid} "
                    f"elapsed_min={(now - started) / 60:.1f} log={log_path}",
                    flush=True,
                )
                next_update = now + 60
            time.sleep(1)
        code = int(_CURRENT_CHILD.returncode or 0)
        _CURRENT_CHILD = None
        return code


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\n[stop] received signal={signum}; forwarding it to the active run", flush=True)
    if _CURRENT_CHILD is not None and _CURRENT_CHILD.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(_CURRENT_CHILD.pid, signal.SIGTERM)


def _acquire_runner(state_root: Path) -> Path:
    pid_path = state_root / "runner.pid"
    if pid_path.exists():
        try:
            existing_pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(existing_pid, 0)
        except (OSError, ValueError):
            pid_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Experiment runner is already active with pid={existing_pid}")
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    return pid_path


def _run_learning(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    state_root: Path,
    sources: tuple[str, ...],
    training_budget: int,
) -> int:
    expected_updates = TASKS_PER_BENCHMARK // REFLECTION_BATCH_SIZE
    for source in sources:
        config_path = _config_path(
            state_root,
            phase="learning",
            source=source,
            target=source,
            budget=training_budget,
        )
        task_ids = [str(value) for value in manifest["benchmarks"][source]["task_ids"]]
        for index, task_id in enumerate(task_ids, start=1):
            if not _task_complete(
                args,
                benchmark=source,
                phase="learning",
                source=source,
                budget=training_budget,
                task_id=task_id,
            ):
                command = _run_command(
                    args=args,
                    config_path=config_path,
                    benchmark=source,
                    phase="learning",
                    source=source,
                    budget=training_budget,
                    task_ids=[task_id],
                )
                log_path = state_root / "logs" / "learning" / source / f"{task_id}.log"
                print(
                    f"[learning] source={source} task={index}/{len(task_ids)} id={task_id}",
                    flush=True,
                )
                code = _run_child(command, log_path, allow_mock=bool(args.allow_mock))
                if _STOP_REQUESTED:
                    return 130
                if code != 0:
                    raise RuntimeError(
                        f"Learning run failed for {source}/{task_id}; see {log_path}"
                    )
            _record_learning_candidate(
                args=args,
                state_root=state_root,
                source=source,
                budget=training_budget,
                task_id=task_id,
            )
            if _STOP_REQUESTED:
                return 130
            _reflect_pending(
                args=args,
                state_root=state_root,
                source=source,
                config_path=config_path,
            )
            if _STOP_REQUESTED:
                return 130
        _freeze_source(state_root, source, expected_updates=expected_updates)
        print(f"[learning-complete] source={source}", flush=True)
    return 0


def _run_evaluation(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    state_root: Path,
    cells: list[tuple[str, str, int]],
) -> int:
    for index, (source, target, budget) in enumerate(cells, start=1):
        task_ids = [str(value) for value in manifest["benchmarks"][target]["task_ids"]]
        completed = sum(
            _task_complete(
                args,
                benchmark=target,
                phase="evaluation",
                source=source,
                budget=budget,
                task_id=task_id,
            )
            for task_id in task_ids
        )
        if completed == len(task_ids):
            continue
        config_path = _config_path(
            state_root,
            phase="evaluation",
            source=source,
            target=target,
            budget=budget,
        )
        command = _run_command(
            args=args,
            config_path=config_path,
            benchmark=target,
            phase="evaluation",
            source=source,
            budget=budget,
            task_ids=task_ids,
        )
        log_path = state_root / "logs" / "evaluation" / target / source / f"mutation_{budget}.log"
        print(
            f"[evaluation] cell={index}/{len(cells)} source={source} target={target} "
            f"budget={budget} completed={completed}/{len(task_ids)}",
            flush=True,
        )
        code = _run_child(command, log_path, allow_mock=bool(args.allow_mock))
        if _STOP_REQUESTED:
            return 130
        if code != 0:
            raise RuntimeError(
                f"Evaluation failed for source={source}, target={target}, budget={budget}; "
                f"see {log_path}"
            )
        remaining = [
            task_id
            for task_id in task_ids
            if not _task_complete(
                args,
                benchmark=target,
                phase="evaluation",
                source=source,
                budget=budget,
                task_id=task_id,
            )
        ]
        if remaining:
            raise RuntimeError(
                f"Evaluation cell returned successfully but has incomplete tasks: {remaining[:5]}"
            )
    return 0


def _transfer_type(source: str, target: str) -> str:
    if source == "seed":
        return "seed_control"
    return "in_domain" if source == target else "cross_domain"


def collect_results(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    cells: list[tuple[str, str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target, budget in cells:
        for raw_task_id in manifest["benchmarks"][target]["task_ids"]:
            task_id = str(raw_task_id)
            root = _task_root(
                args,
                benchmark=target,
                phase="evaluation",
                source=source,
                budget=budget,
                task_id=task_id,
            )
            evaluation = _read_json(root / "run_0.eval.json")
            metadata = _read_json(root / "run_0.metadata.json")
            if not isinstance(evaluation, dict) or not isinstance(metadata, dict):
                continue
            self_evolved = metadata.get("self_evolved") or {}
            mutations = self_evolved.get("mutations") or []
            rows.append(
                {
                    "playbook_source": source,
                    "target_benchmark": target,
                    "transfer_type": _transfer_type(source, target),
                    "configured_mutation_budget": budget,
                    "actual_mutations": len(mutations) if isinstance(mutations, list) else 0,
                    "task_id": task_id,
                    "success": int(bool(evaluation.get("success", False))),
                    "score": float(evaluation.get("score", 0.0) or 0.0),
                    "turns_executed": int(metadata.get("turns_executed", 1) or 1),
                }
            )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_results(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_budget: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_actual: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        base = (
            str(row["playbook_source"]),
            str(row["target_benchmark"]),
            str(row["transfer_type"]),
        )
        by_budget[(*base, int(row["configured_mutation_budget"]))].append(row)
        by_actual[(*base, int(row["actual_mutations"]))].append(row)

    configured: list[dict[str, Any]] = []
    for (source, target, transfer, budget), group in sorted(by_budget.items()):
        successes = sum(int(row["success"]) for row in group)
        low, high = _wilson(successes, len(group))
        configured.append(
            {
                "playbook_source": source,
                "target_benchmark": target,
                "transfer_type": transfer,
                "configured_mutation_budget": budget,
                "n": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "success_ci95_low": low,
                "success_ci95_high": high,
                "avg_score": _mean([float(row["score"]) for row in group]),
                "mean_actual_mutations": _mean([float(row["actual_mutations"]) for row in group]),
                "mean_turns_executed": _mean([float(row["turns_executed"]) for row in group]),
            }
        )

    seed_lookup = {
        (
            str(row["target_benchmark"]),
            int(row["configured_mutation_budget"]),
            str(row["task_id"]),
        ): row
        for row in rows
        if row["playbook_source"] == "seed"
    }
    for summary in configured:
        source = str(summary["playbook_source"])
        if source == "seed":
            summary["paired_success_delta_vs_seed"] = 0.0
            summary["paired_score_delta_vs_seed"] = 0.0
            continue
        paired = [
            (
                float(row["success"])
                - float(
                    seed_lookup[
                        (
                            str(row["target_benchmark"]),
                            int(row["configured_mutation_budget"]),
                            str(row["task_id"]),
                        )
                    ]["success"]
                ),
                float(row["score"])
                - float(
                    seed_lookup[
                        (
                            str(row["target_benchmark"]),
                            int(row["configured_mutation_budget"]),
                            str(row["task_id"]),
                        )
                    ]["score"]
                ),
            )
            for row in rows
            if row["playbook_source"] == source
            and row["target_benchmark"] == summary["target_benchmark"]
            and row["configured_mutation_budget"] == summary["configured_mutation_budget"]
            and (
                str(row["target_benchmark"]),
                int(row["configured_mutation_budget"]),
                str(row["task_id"]),
            )
            in seed_lookup
        ]
        summary["paired_success_delta_vs_seed"] = _mean([value[0] for value in paired])
        summary["paired_score_delta_vs_seed"] = _mean([value[1] for value in paired])

    actual: list[dict[str, Any]] = []
    for (source, target, transfer, count), group in sorted(by_actual.items()):
        successes = sum(int(row["success"]) for row in group)
        low, high = _wilson(successes, len(group))
        actual.append(
            {
                "playbook_source": source,
                "target_benchmark": target,
                "transfer_type": transfer,
                "actual_mutations": count,
                "n": len(group),
                "successes": successes,
                "success_rate": successes / len(group),
                "success_ci95_low": low,
                "success_ci95_high": high,
                "avg_score": _mean([float(row["score"]) for row in group]),
            }
        )
    return configured, actual


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_curve(
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    output_path: Path,
    title: str,
    subtitle: str,
) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = sorted({str(row["target_benchmark"]) for row in rows})
    sources = sorted({str(row["playbook_source"]) for row in rows})
    colors = {
        source: color
        for source, color in zip(
            sources,
            ("#334E68", "#D97706", "#557A46", "#B4537A", "#7C3AED"),
            strict=False,
        )
    }
    line_styles = ("-", "--", "-.", ":")
    fig, axes = plt.subplots(
        len(targets),
        1,
        figsize=(9.5, max(3.6, 3.2 * len(targets))),
        sharex=False,
        squeeze=False,
    )
    for axis, target in zip(axes[:, 0], targets, strict=True):
        target_rows = [row for row in rows if row["target_benchmark"] == target]
        for source_index, source in enumerate(sources):
            series = sorted(
                (row for row in target_rows if row["playbook_source"] == source),
                key=lambda row: int(row[x_field]),
            )
            if not series:
                continue
            x = [int(row[x_field]) for row in series]
            y = [float(row["success_rate"]) for row in series]
            yerr = [
                [value - float(row["success_ci95_low"]) for value, row in zip(y, series)],
                [float(row["success_ci95_high"]) - value for value, row in zip(y, series)],
            ]
            label = "seed control" if source == "seed" else source
            if source == target:
                label += " (in-domain)"
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                label=label,
                color=colors[source],
                linestyle=line_styles[source_index % len(line_styles)],
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                capsize=2.5,
            )
        axis.set_title(f"Target: {target}", loc="left", fontsize=11, fontweight="semibold")
        axis.set_ylabel("Success rate")
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(sorted({int(row[x_field]) for row in target_rows}))
        axis.grid(axis="y", color="#D9E2EC", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    axes[-1, 0].set_xlabel(
        "Configured mutation budget"
        if x_field == "configured_mutation_budget"
        else "Actual mutations applied"
    )
    fig.suptitle(title, x=0.08, ha="left", fontsize=14, fontweight="bold", color="#102A43")
    fig.text(0.08, 0.955, subtitle, ha="left", va="top", fontsize=9, color="#486581")
    fig.tight_layout(rect=(0.04, 0.03, 0.99, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def analyze(args: argparse.Namespace) -> int:
    sources, targets, budgets = _design(args)
    manifest, _ = prepare(args)
    cells = evaluation_cells(sources, targets, budgets)
    rows = collect_results(args, manifest, cells)
    configured, actual = summarize_results(rows)
    analysis_root = _experiment_root(args) / "combined_analysis"
    _write_csv(analysis_root / "run_results.csv", rows)
    _write_csv(analysis_root / "mutation_budget_summary.csv", configured)
    _write_csv(analysis_root / "actual_mutation_summary.csv", actual)
    _render_curve(
        configured,
        x_field="configured_mutation_budget",
        output_path=analysis_root / "mutation_budget_success_curve.png",
        title="Success rate by configured mutation budget",
        subtitle=(
            "Points are benchmark evaluation success rates; bars show 95% Wilson intervals. "
            "Each complete point has 30 tasks."
        ),
    )
    _render_curve(
        actual,
        x_field="actual_mutations",
        output_path=analysis_root / "actual_mutation_success_curve.png",
        title="Success rate by actual mutations applied",
        subtitle=(
            "Descriptive grouping of observed runs; sample size can differ by point and "
            "does not identify the causal effect of mutation."
        ),
    )
    _write_json(
        analysis_root / "analysis_manifest.json",
        {
            "complete_evaluation_runs": len(rows),
            "expected_evaluation_runs": len(cells) * TASKS_PER_BENCHMARK,
            "configured_summary_rows": len(configured),
            "actual_mutation_summary_rows": len(actual),
            "primary_metric": "benchmark.evaluate(...).success",
            "notes": [
                "Configured-budget curves compare randomized-identical task sets across arms.",
                "Actual-mutation curves are descriptive because harder runs may consume more mutations.",
            ],
        },
    )
    print(
        f"[analysis] rows={len(rows)}/{len(cells) * TASKS_PER_BENCHMARK} output={analysis_root}",
        flush=True,
    )
    return 0


def run(args: argparse.Namespace) -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    if not args.allow_mock and not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for experiment runs. Use --allow-mock only "
            "for a smoke test, never for reported results."
        )
    sources, targets, budgets = _design(args)
    manifest, state_root = prepare(args)
    training_budget = (
        int(args.training_mutation_budget)
        if args.training_mutation_budget is not None
        else max(budgets)
    )
    cells = evaluation_cells(sources, targets, budgets)
    experiment_manifest = _read_json(state_root / "experiment_manifest.json")
    print(
        "[design] "
        f"learning_runs={experiment_manifest['learning_runs']} "
        f"evaluation_runs={experiment_manifest['evaluation_runs']} "
        f"cells={len(cells)} batch={REFLECTION_BATCH_SIZE}",
        flush=True,
    )
    pid_path = _acquire_runner(state_root)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, _signal_handler)
    try:
        code = _run_learning(args, manifest, state_root, sources, training_budget)
        if code != 0:
            return code
        code = _run_evaluation(args, manifest, state_root, cells)
        if code != 0:
            return code
        analyze(args)
        summary_log = state_root / "logs" / "summarize_experiment.log"
        code = _run_child(
            [
                _python(),
                "main.py",
                "summarize-experiment",
                "--experiment-root",
                str(_experiment_root(args)),
            ],
            summary_log,
            allow_mock=bool(args.allow_mock),
        )
        if code != 0:
            raise RuntimeError(f"Experiment summarization failed; see {summary_log}")
        print(f"[complete] {_experiment_root(args)}", flush=True)
        return 0
    finally:
        pid_path.unlink(missing_ok=True)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)


def status(args: argparse.Namespace) -> int:
    sources, targets, budgets = _design(args)
    manifest, state_root = prepare(args)
    training_budget = (
        int(args.training_mutation_budget)
        if args.training_mutation_budget is not None
        else max(budgets)
    )
    for source in sources:
        task_ids = [str(value) for value in manifest["benchmarks"][source]["task_ids"]]
        completed = sum(
            _task_complete(
                args,
                benchmark=source,
                phase="learning",
                source=source,
                budget=training_budget,
                task_id=task_id,
            )
            for task_id in task_ids
        )
        reflection = _load_reflection_state(state_root, source)
        frozen = (_frozen_skill(state_root, source).parent / "freeze.json").exists()
        print(
            f"learning {source:15s} {completed:2d}/{len(task_ids)} "
            f"updates={len(reflection['updates'])} pending={len(reflection['pending'])} "
            f"frozen={frozen}"
        )
    cells = evaluation_cells(sources, targets, budgets)
    complete_cells = 0
    complete_runs = 0
    for source, target, budget in cells:
        task_ids = [str(value) for value in manifest["benchmarks"][target]["task_ids"]]
        count = sum(
            _task_complete(
                args,
                benchmark=target,
                phase="evaluation",
                source=source,
                budget=budget,
                task_id=task_id,
            )
            for task_id in task_ids
        )
        complete_runs += count
        complete_cells += int(count == len(task_ids))
    print(
        f"evaluation cells={complete_cells}/{len(cells)} "
        f"runs={complete_runs}/{len(cells) * TASKS_PER_BENCHMARK}"
    )
    pid_path = state_root / "runner.pid"
    print(f"runner_pid={pid_path.read_text().strip() if pid_path.exists() else 'not running'}")
    print(f"experiment_root={_experiment_root(args)}")
    return 0


def stop(args: argparse.Namespace) -> int:
    pid_path = _state_root(args) / "runner.pid"
    if not pid_path.exists():
        print("Experiment runner is not running.")
        return 0
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        pid_path.unlink(missing_ok=True)
        print("Removed a stale runner PID file.")
        return 0
    print(f"Stop requested for experiment runner pid={pid}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "run", "status", "stop", "analyze"),
        help="Prepare, run/resume, inspect, stop, or rebuild analysis outputs.",
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--source-benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmarks that each learn a long-term playbook.",
    )
    parser.add_argument(
        "--target-benchmarks",
        default=",".join(DEFAULT_BENCHMARKS),
        help="Comma-separated benchmarks on which frozen playbooks are evaluated.",
    )
    parser.add_argument(
        "--mutation-budgets",
        default=",".join(map(str, DEFAULT_MUTATION_BUDGETS)),
        help="Comma-separated repair-mutation budgets. Each budget B uses max_turns=B+1.",
    )
    parser.add_argument(
        "--training-mutation-budget",
        type=int,
        default=None,
        help="Mutation budget used while learning each playbook (default: largest eval budget).",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Permit deterministic mock LLM calls for smoke tests. Never use for paper results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        _, state_root = prepare(args)
        sources, targets, budgets = _design(args)
        cells = evaluation_cells(sources, targets, budgets)
        print(f"Prepared experiment state under {state_root}")
        print(
            f"Learning runs: {len(sources) * TASKS_PER_BENCHMARK}; "
            f"evaluation cells: {len(cells)}; "
            f"evaluation runs: {len(cells) * TASKS_PER_BENCHMARK}"
        )
        return 0
    if args.command == "run":
        return run(args)
    if args.command == "status":
        return status(args)
    if args.command == "stop":
        return stop(args)
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())

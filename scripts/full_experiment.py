#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SYSTEMS: list[tuple[str, str, int, int, int, int]] = [
    ("sas", "sas", 1, 1, 1, 0),
    ("orchestrator_tree_structure", "orchestrator_tree_structure", 5, 2, 1, 2),
    ("orchestrator_no_discussion", "orchestrator_no_discussion", 4, 2, 1, 2),
    ("orchestrator_with_discussion", "orchestrator_with_discussion", 4, 2, 2, 2),
    ("only_voting", "only_voting", 4, 1, 1, 0),
    ("fully_linked_debate", "fully_linked_debate", 4, 2, 1, 2),
    ("group_chat_debate", "group_chat_debate", 4, 2, 2, 2),
    ("self_evolved", "self_evolved", 5, 2, 1, 2),
]

SYSTEM_ARG_ENV_BY_LABEL: dict[str, str] = {
    "sas": "SAS_ARGS",
    "orchestrator_tree_structure": "ORCHESTRATOR_TREE_STRUCTURE_ARGS",
    "orchestrator_no_discussion": "ORCHESTRATOR_NO_DISCUSSION_ARGS",
    "orchestrator_with_discussion": "ORCHESTRATOR_WITH_DISCUSSION_ARGS",
    "only_voting": "ONLY_VOTING_ARGS",
    "fully_linked_debate": "FULLY_LINKED_DEBATE_ARGS",
    "group_chat_debate": "GROUP_CHAT_DEBATE_ARGS",
    "self_evolved": "SELF_EVOLVED_ARGS",
}


def selected_systems() -> list[tuple[str, str, int, int, int, int]]:
    """SYSTEMS, optionally filtered to a comma-separated allowlist of system labels
    via the ONLY_SYSTEMS env var (e.g. ONLY_SYSTEMS=self_evolved). Default: all."""
    only = os.getenv("ONLY_SYSTEMS", "").strip()
    if not only:
        return SYSTEMS
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    filtered = [row for row in SYSTEMS if row[0] in wanted]
    if not filtered:
        raise ValueError(
            f"ONLY_SYSTEMS={only!r} matched no systems; valid: "
            f"{', '.join(row[0] for row in SYSTEMS)}"
        )
    return filtered


DEFAULT_CONFIG_DIR = ROOT / "config" / "benchmarks"
SERVICES_ROOT = ROOT / ".cache" / "services"
EXTERNAL_ROOT = ROOT / ".cache" / "external"
AGENTBENCH_ROOT = EXTERNAL_ROOT / "AgentBench"
AGENTBENCH_VENV = AGENTBENCH_ROOT / ".venv"
AGENTBENCH_STATE = SERVICES_ROOT / "agentbench"
STABLETOOLBENCH_STATE = SERVICES_ROOT / "stabletoolbench"


class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class BatchOptions:
    experiment_id: str
    output_root: Path
    config_dir: Path
    benchmarks: list[str] | None
    task_limit: int | None
    runs_per_task: int | None
    retry_failures: int
    max_parallel: int
    models: list[str] | None
    final_vote_mode: str | None
    benchmark_eval_judge_model: str | None
    skip_setup: bool
    setup_only: bool
    no_dynamic_roles: bool
    list_benchmarks: bool


@dataclass(frozen=True)
class RunJob:
    benchmark_name: str
    config_path: Path
    system_label: str
    topology: str
    agents: int
    rounds: int
    discussion_rounds: int
    communication_budget: int
    cmd: list[str]
    log_path: Path


@dataclass(frozen=True)
class RunJobResult:
    job: RunJob
    returncode: int
    attempt_count: int
    elapsed_s: float


def info(message: str) -> None:
    print(message, flush=True)


def env_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def env_float(value: str | None, default: float) -> float:
    raw = str(value or "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    return max(parsed, 0.0)


def progress_snapshot_lines(experiment_root: Path) -> list[str]:
    if not experiment_root.exists():
        return [f"[progress] waiting_for_experiment_root={experiment_root}"]

    command = [
        sys.executable,
        "main.py",
        "summarize-experiment",
        "--experiment-root",
        str(experiment_root),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        suffix = f" detail={detail[-1]}" if detail else ""
        return [
            f"[progress] summary_unavailable experiment={experiment_root.name} "
            f"returncode={result.returncode}{suffix}"
        ]

    selected = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("Experiment run-status summary:")
        or line.startswith("  - ")
    ]
    if not selected:
        return [f"[progress] summary_empty experiment={experiment_root.name}"]
    return [f"[progress] experiment={experiment_root.name}", *[f"[progress] {line}" for line in selected]]


def system_summary_is_clean(summary_json: Path, summary_csv: Path) -> bool:
    if not summary_json.exists() or not summary_csv.exists():
        return False
    try:
        payload = json.loads(summary_json.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False

    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return False
    task_count = int(payload.get("task_count", len(tasks)) or 0)
    if len(tasks) < task_count:
        return False

    for task in tasks:
        if not isinstance(task, dict):
            return False
        if bool(task.get("needs_rerun", False)):
            return False
    return True


class ProgressReporter:
    def __init__(self, experiment_roots: list[Path], interval_s: float) -> None:
        self.experiment_roots = list(experiment_roots)
        self.interval_s = max(interval_s, 0.0)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_s <= 0.0 or not self.experiment_roots:
            return
        self.thread = threading.Thread(
            target=self._run,
            name="full-experiment-progress",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)

    def emit_snapshot(self, label: str = "Progress snapshot") -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        info("")
        info(f"=== {label} {timestamp} ===")
        for experiment_root in self.experiment_roots:
            for line in progress_snapshot_lines(experiment_root):
                info(line)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self.emit_snapshot()


def run_command(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    info(f"$ {shlex.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        check=check,
    )


def run_command_capture(
    cmd: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    info(f"$ {shlex.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def python_executable_for_venv(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _model_slug(model: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(model).strip())
    compact = "_".join(part for part in text.split("_") if part)
    return compact or "model"


def _merge_model_override(args: list[str], model: str | None) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token in {"--default-model", "--judge-model"}:
            i += 2
            continue
        if token.startswith("--default-model=") or token.startswith("--judge-model="):
            i += 1
            continue
        merged.append(token)
        i += 1
    if model:
        merged.extend(["--default-model", model, "--judge-model", model])
    return merged


def _find_ignored_model_arg_sources() -> dict[str, list[str]]:
    ignored: dict[str, list[str]] = {}
    env_names = ["MAS_GLOBAL_ARGS", *SYSTEM_ARG_ENV_BY_LABEL.values()]
    for env_name in env_names:
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        tokens = shlex.split(raw)
        found: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"--default-model", "--judge-model"}:
                value = tokens[i + 1] if i + 1 < len(tokens) else ""
                found.append(" ".join(part for part in [token, value] if part))
                i += 2
                continue
            if token.startswith("--default-model=") or token.startswith("--judge-model="):
                found.append(token)
            i += 1
        if found:
            ignored[env_name] = found
    return ignored


def parse_batch_args(argv: list[str]) -> BatchOptions:
    parser = argparse.ArgumentParser(
        description="Bootstrap benchmark environments and run LangGraph batch experiments.",
    )
    parser.add_argument(
        "--experiment-id",
        default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "artifacts" / "full_experiment"),
    )
    parser.add_argument(
        "--config-dir",
        default=str(DEFAULT_CONFIG_DIR),
    )
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Comma-separated benchmark subset. Defaults to every config in config/benchmarks.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Select one benchmark. Repeat to include multiple benchmarks.",
    )
    parser.add_argument(
        "--list-benchmarks",
        action="store_true",
        help="List discovered benchmark configs and exit.",
    )
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--runs-per-task", type=int, default=None)
    parser.add_argument(
        "--retry-failures",
        type=int,
        default=0,
        help="Retry each failed run job this many additional times.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of run jobs to execute concurrently.",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated model routes to run sequentially. "
            "Each model overrides both --default-model and --judge-model for launched jobs."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Select one model route. Repeat to include multiple models. "
            "Each model overrides both --default-model and --judge-model."
        ),
    )
    parser.add_argument(
        "--final-vote-mode",
        choices=["llm_judge", "deterministic"],
        default=None,
        help="Override MAS final answer selection mode for every launched system.",
    )
    parser.add_argument(
        "--benchmark-eval-judge-model",
        default=None,
        help="Override benchmark-side evaluation judge_model for every launched run.",
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument(
        "--no-dynamic-roles",
        dest="no_dynamic_roles",
        action="store_true",
        default=False,
        help="Disable LLM-based dynamic role assignment for every launched system.",
    )
    args = parser.parse_args(argv)

    selected = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    selected.extend(item.strip() for item in args.benchmark if item.strip())
    selected = _dedupe_keep_order(selected)
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    selected_models.extend(item.strip() for item in args.model if item.strip())
    selected_models = _dedupe_keep_order(selected_models)
    return BatchOptions(
        experiment_id=str(args.experiment_id),
        output_root=Path(args.output_root).expanduser().resolve(),
        config_dir=Path(args.config_dir).expanduser().resolve(),
        benchmarks=selected or None,
        task_limit=args.task_limit,
        runs_per_task=args.runs_per_task,
        retry_failures=max(int(args.retry_failures), 0),
        max_parallel=max(int(args.max_parallel), 1),
        models=selected_models or None,
        final_vote_mode=args.final_vote_mode,
        benchmark_eval_judge_model=(
            str(args.benchmark_eval_judge_model)
            if args.benchmark_eval_judge_model is not None
            else None
        ),
        skip_setup=bool(args.skip_setup),
        setup_only=bool(args.setup_only),
        no_dynamic_roles=bool(args.no_dynamic_roles),
        list_benchmarks=bool(args.list_benchmarks),
    )


def load_benchmark_configs(config_dir: Path) -> dict[str, Path]:
    configs: dict[str, Path] = {}
    for path in sorted(config_dir.glob("*.toml")):
        name = path.stem
        if name.endswith("_10"):
            name = name[:-3]
        configs[name] = path
    if not configs:
        raise FileNotFoundError(f"No benchmark configs found under {config_dir}")
    return configs


def list_available_benchmarks(config_dir: Path) -> list[str]:
    return sorted(load_benchmark_configs(config_dir))


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def select_benchmarks(options: BatchOptions) -> dict[str, Path]:
    configs = load_benchmark_configs(options.config_dir)
    if not options.benchmarks:
        return configs

    selected: dict[str, Path] = {}
    for benchmark in options.benchmarks:
        if benchmark not in configs:
            available = ", ".join(sorted(configs))
            raise ValueError(f"Unknown benchmark '{benchmark}'. Available: {available}")
        selected[benchmark] = configs[benchmark]
    return selected


def print_available_benchmarks(config_dir: Path) -> int:
    for benchmark in list_available_benchmarks(config_dir):
        print(benchmark)
    return 0


def configured_runs_per_task(config_path: Path, override: int | None) -> int:
    if override is not None:
        return max(int(override), 0)
    config = load_toml(config_path)
    experiment_cfg = dict(config.get("experiment", {}))
    return max(int(experiment_cfg.get("runs_per_task", 1)), 0)


def warn_non_estimable_metrics(
    selected_configs: dict[str, Path],
    *,
    runs_per_task_override: int | None,
) -> None:
    for benchmark_name, config_path in selected_configs.items():
        runs_per_task = configured_runs_per_task(config_path, runs_per_task_override)
        unavailable: list[str] = []
        if runs_per_task < 2:
            unavailable.extend(["stability", "tokens_cv"])
        for k in (1, 3, 5, 8):
            if runs_per_task < k:
                unavailable.append(f"pass_at_{k}")
        if unavailable:
            info(
                f"WARNING benchmark={benchmark_name} runs_per_task={runs_per_task} "
                f"non_estimable_metrics={','.join(unavailable)}"
            )


def service_pid_path(state_dir: Path, name: str) -> Path:
    return state_dir / f"{name}.pid"


def process_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return str(pid) in str(result.stdout or "")
    try:
        os.kill(pid, 0)
    except Exception:  # noqa: BLE001
        return False
    return True


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def stop_process_group(pid_path: Path, *, timeout_s: float = 10.0) -> None:
    pid = read_pid(pid_path)
    if pid is None:
        return
    if not process_is_running(pid):
        pid_path.unlink(missing_ok=True)
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not process_is_running(pid):
            pid_path.unlink(missing_ok=True)
            return
        time.sleep(0.2)
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        with contextlib.suppress(OSError):
            os.killpg(pid, signal.SIGKILL)
    pid_path.unlink(missing_ok=True)


def stop_stale_process(pid_path: Path) -> None:
    stop_process_group(pid_path)


def spawn_background_process(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    pid_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    stop_stale_process(pid_path)
    ensure_parent(log_path)
    log_handle = log_path.open("a", encoding="utf-8")
    info(f"$ {shlex.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    pid_path.write_text(str(process.pid), encoding="utf-8")
    log_handle.close()
    info(f"Started {name} with pid={process.pid}")


@contextlib.contextmanager
def managed_background_services() -> Any:
    spawned_pid_paths: list[Path] = []
    previous_handlers: dict[int, Any] = {}

    def cleanup() -> None:
        for pid_path in reversed(spawned_pid_paths):
            stop_process_group(pid_path)

    def register(pid_path: Path) -> None:
        if pid_path not in spawned_pid_paths:
            spawned_pid_paths.append(pid_path)

    def handle_signal(signum: int, _frame: Any) -> None:
        info(f"Received signal {signum}; stopping background services.")
        cleanup()
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    try:
        yield register
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        cleanup()


def wait_for_http(
    url: str,
    *,
    timeout_s: float,
    validator: callable | None = None,
) -> requests.Response:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if validator is None or validator(response):
                return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for {url}") from last_error


def wait_for_command_success(
    *,
    timeout_s: float,
    fn: callable,
) -> Any:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise RuntimeError("Timed out waiting for background service readiness") from last_error


def ensure_symlink_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
        info(f"Linked {target} -> {source}")
    except OSError:
        shutil.copy2(source, target)
        info(f"Copied {source} -> {target}")


def warmup_benchmark(name: str, config: dict[str, Any], *, task_limit: int = 1) -> None:
    from benchmark.registry import get_benchmark

    benchmark = get_benchmark(name, config)
    tasks = benchmark.load_tasks(task_limit=task_limit)
    info(f"Prepared benchmark={name} sample_tasks={len(tasks)}")


def prepare_browsecomp(config: dict[str, Any]) -> None:
    from benchmark.browsecomp import BrowseCompBenchmark

    cfg = dict(config)
    decrypted_path_value = str(cfg.get("decrypted_path", "") or "").strip()
    default_repo_path = (
        ROOT / "benchmark" / "browsecomp" / "data" / "browsecomp_plus_decrypted.jsonl"
    )

    if decrypted_path_value:
        requested_path = (ROOT / decrypted_path_value).resolve()
        if not requested_path.exists() and default_repo_path.exists():
            ensure_symlink_or_copy(default_repo_path, requested_path)

    if decrypted_path_value and (ROOT / decrypted_path_value).resolve().exists():
        benchmark = BrowseCompBenchmark(cfg)
        tasks = benchmark.load_tasks(task_limit=1)
        info(f"Prepared benchmark=browsecomp sample_tasks={len(tasks)}")
        return

    bootstrap_cfg = dict(cfg)
    if decrypted_path_value:
        bootstrap_cfg.pop("decrypted_path", None)
        bootstrap_cfg["decrypted_output_path"] = str((ROOT / decrypted_path_value).resolve())
    bootstrap_cfg["auto_download"] = True
    benchmark = BrowseCompBenchmark(bootstrap_cfg)
    tasks = benchmark.load_tasks(task_limit=1)
    info(f"Prepared benchmark=browsecomp sample_tasks={len(tasks)}")


def stabletoolbench_health_url(virtual_server_url: str) -> str:
    parsed = urlparse(virtual_server_url)
    return f"{parsed.scheme}://{parsed.netloc}/healthz"


def verify_local_stabletoolbench_server(
    benchmark: Any,
    *,
    timeout_s: float = 60.0,
) -> None:
    tasks = benchmark.load_tasks(task_limit=1)
    if not tasks:
        raise RuntimeError("StableToolBench task list is empty after setup")
    task_id = tasks[0].task_id
    api_list = benchmark._task_api_lists.get(task_id, [])  # noqa: SLF001
    if not api_list:
        raise RuntimeError("StableToolBench task has no api_list for server verification")
    api_meta = api_list[0]
    tool_input = benchmark._resolved_tool_input(api_meta, {})  # noqa: SLF001
    response = requests.post(
        benchmark.virtual_server_url,
        json={
            "category": str(api_meta.get("category_name", "")),
            "tool_name": str(api_meta.get("tool_name", "")),
            "api_name": str(api_meta.get("api_name", "")),
            "tool_input": json.dumps(tool_input, ensure_ascii=False),
            "strip": benchmark.strip,
            "toolbench_key": benchmark.toolbench_key,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    error = str(payload.get("error", "")).strip()
    if error and "cache miss" not in error.lower():
        raise RuntimeError(f"StableToolBench server self-check failed: {payload['error']}")
    if error:
        info(f"StableToolBench self-check returned a cache miss: {error}")


def prepare_stabletoolbench(
    config: dict[str, Any],
    *,
    register_background_service: callable | None = None,
) -> None:
    from benchmark.stabletoolbench import StableToolBenchBenchmark

    cfg = dict(config)
    cfg["auto_download"] = True
    cfg["auto_download_server_assets"] = True
    benchmark = StableToolBenchBenchmark(cfg)
    benchmark.load_tasks(task_limit=1)

    parsed = urlparse(benchmark.virtual_server_url)
    host = parsed.hostname or "127.0.0.1"
    if host not in {"127.0.0.1", "localhost"}:
        info(
            f"StableToolBench points to non-local server {benchmark.virtual_server_url}; skipping local bootstrap"
        )
        return

    health_url = stabletoolbench_health_url(benchmark.virtual_server_url)
    try:
        wait_for_http(
            health_url,
            timeout_s=3,
            validator=lambda response: response.ok,
        )
        info(f"StableToolBench server already reachable at {benchmark.virtual_server_url}")
    except Exception:  # noqa: BLE001
        STABLETOOLBENCH_STATE.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "stabletoolbench_virtual_server.py"),
            "--host",
            host,
            "--port",
            str(parsed.port or 8080),
            "--path",
            parsed.path or "/virtual",
            "--cache-root",
            str(ROOT / "benchmark" / "stabletoolbench" / "tool_response_cache"),
        ]
        spawn_background_process(
            name="stabletoolbench",
            cmd=cmd,
            cwd=ROOT,
            log_path=STABLETOOLBENCH_STATE / "server.log",
            pid_path=service_pid_path(STABLETOOLBENCH_STATE, "server"),
        )
        if register_background_service is not None:
            register_background_service(service_pid_path(STABLETOOLBENCH_STATE, "server"))
        wait_for_http(
            health_url,
            timeout_s=60,
            validator=lambda response: response.ok,
        )
        info(f"StableToolBench server is ready at {benchmark.virtual_server_url}")

    verify_local_stabletoolbench_server(benchmark)
    info("Prepared benchmark=stabletoolbench")


def docker_ready() -> bool:
    result = subprocess.run(
        ["docker", "info"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def ensure_docker_daemon() -> None:
    if docker_ready():
        info("Docker daemon is ready")
        return

    if sys.platform == "darwin":
        info("Docker daemon not ready; attempting to launch Docker.app")
        subprocess.run(
            ["open", "-a", "Docker"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        deadline = time.time() + 180
        while time.time() < deadline:
            if docker_ready():
                info("Docker daemon is ready")
                return
            time.sleep(5)

    raise RuntimeError(
        "Docker is required for AgentBench. Start Docker Desktop and rerun the script."
    )


def ensure_git_checkout(*, repo_dir: Path, repo_url: str, ref: str) -> None:
    if (repo_dir / ".git").exists():
        info(f"Using existing checkout at {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            ref,
            repo_url,
            str(repo_dir),
        ]
    )


def select_agentbench_python() -> str:
    for candidate in ("python3.10", "python3", "python"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return sys.executable


def ensure_agentbench_venv() -> Path:
    python_path = python_executable_for_venv(AGENTBENCH_VENV)
    if python_path.exists():
        return python_path
    AGENTBENCH_VENV.parent.mkdir(parents=True, exist_ok=True)
    run_command([select_agentbench_python(), "-m", "venv", str(AGENTBENCH_VENV)])
    return python_path


def ensure_agentbench_dependencies(agentbench_python: Path) -> None:
    sentinel = AGENTBENCH_VENV / ".requirements_installed"
    if sentinel.exists():
        return
    run_command(
        [str(agentbench_python), "-m", "pip", "install", "--upgrade", "pip"], cwd=AGENTBENCH_ROOT
    )
    run_command(
        [str(agentbench_python), "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=AGENTBENCH_ROOT,
    )
    sentinel.write_text("ok\n", encoding="utf-8")


def ensure_agentbench_os_images() -> None:
    image_specs = [
        (
            "local-os/default",
            [
                "docker",
                "build",
                "-f",
                "data/os_interaction/res/dockerfiles/default",
                "data/os_interaction/res/dockerfiles",
                "--tag",
                "local-os/default",
            ],
        ),
        (
            "local-os/packages",
            [
                "docker",
                "build",
                "-f",
                "data/os_interaction/res/dockerfiles/packages",
                "data/os_interaction/res/dockerfiles",
                "--tag",
                "local-os/packages",
            ],
        ),
        (
            "local-os/ubuntu",
            [
                "docker",
                "build",
                "-f",
                "data/os_interaction/res/dockerfiles/ubuntu",
                "data/os_interaction/res/dockerfiles",
                "--tag",
                "local-os/ubuntu",
            ],
        ),
    ]

    for tag, build_cmd in image_specs:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            cwd=AGENTBENCH_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            continue
        run_command(build_cmd, cwd=AGENTBENCH_ROOT)


def controller_health_check(controller_address: str, task_name: str) -> Any:
    response = requests.get(
        f"{controller_address.rstrip('/')}/get_indices",
        params={"name": task_name},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def prepare_agentbench(
    config: dict[str, Any],
    *,
    register_background_service: callable | None = None,
) -> None:
    from benchmark.agentbench import AgentBenchBenchmark

    cfg = dict(config)
    controller_address = str(cfg.get("controller_address", "http://localhost:5000/api")).rstrip("/")
    task_name = str(cfg.get("task_name", "os-std")).strip()

    ensure_docker_daemon()
    ensure_git_checkout(
        repo_dir=AGENTBENCH_ROOT,
        repo_url="https://github.com/THUDM/AgentBench",
        ref="v0.2",
    )
    agentbench_python = ensure_agentbench_venv()
    ensure_agentbench_dependencies(agentbench_python)

    if task_name.startswith("os"):
        ensure_agentbench_os_images()

    parsed = urlparse(controller_address)
    if parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise RuntimeError(
            f"AgentBench bootstrap only supports local controller URLs, got {controller_address}"
        )

    AGENTBENCH_STATE.mkdir(parents=True, exist_ok=True)
    controller_pid = service_pid_path(AGENTBENCH_STATE, "controller")
    worker_pid = service_pid_path(AGENTBENCH_STATE, f"worker_{task_name}")
    agentbench_env = dict(os.environ)
    agentbench_env["VIRTUAL_ENV"] = str(AGENTBENCH_VENV)
    agentbench_env["PATH"] = (
        f"{AGENTBENCH_VENV / 'bin'}{os.pathsep}{agentbench_env.get('PATH', '')}"
    )

    try:
        controller_health_check(controller_address, task_name)
        info(f"AgentBench controller already reachable at {controller_address}")
    except Exception:  # noqa: BLE001
        controller_port = str(parsed.port or 5000)
        spawn_background_process(
            name="agentbench-controller",
            cmd=[
                str(agentbench_python),
                "-m",
                "src.server.task_controller",
                "--port",
                controller_port,
            ],
            cwd=AGENTBENCH_ROOT,
            log_path=AGENTBENCH_STATE / "controller.log",
            pid_path=controller_pid,
            env=agentbench_env,
        )
        if register_background_service is not None:
            register_background_service(controller_pid)
        spawn_background_process(
            name="agentbench-worker",
            cmd=[
                str(agentbench_python),
                "-m",
                "src.start_task",
                "--controller",
                controller_address,
                "-s",
                task_name,
                "1",
            ],
            cwd=AGENTBENCH_ROOT,
            log_path=AGENTBENCH_STATE / f"worker_{task_name}.log",
            pid_path=worker_pid,
            env=agentbench_env,
        )
        if register_background_service is not None:
            register_background_service(worker_pid)

    wait_for_command_success(
        timeout_s=300,
        fn=lambda: controller_health_check(controller_address, task_name),
    )

    benchmark = AgentBenchBenchmark(cfg)
    tasks = benchmark.load_tasks(task_limit=1)
    info(f"Prepared benchmark=agentbench sample_tasks={len(tasks)}")


def prepare_benchmark_environment(
    config_path: Path,
    *,
    benchmark_name: str,
    register_background_service: callable | None = None,
) -> None:
    config = load_toml(config_path)
    benchmark_cfg = dict(config.get(benchmark_name, {}))

    if benchmark_name == "browsecomp":
        prepare_browsecomp(benchmark_cfg)
        return
    if benchmark_name == "stabletoolbench":
        prepare_stabletoolbench(
            benchmark_cfg,
            register_background_service=register_background_service,
        )
        return
    if benchmark_name == "agentbench":
        prepare_agentbench(
            benchmark_cfg,
            register_background_service=register_background_service,
        )
        return

    warmup_benchmark(benchmark_name, benchmark_cfg)


def batch_run(options: BatchOptions) -> int:
    selected_configs = select_benchmarks(options)
    live_env = dict(os.environ)
    if env_flag(live_env.get("MAS_DISABLE_LIVE_LLM")):
        live_env["MAS_DISABLE_LIVE_LLM"] = "1"
    else:
        live_env["MAS_REQUIRE_LIVE_LLM"] = "1"
    models = list(options.models or [])
    experiment_ids = (
        [f"{options.experiment_id}__{_model_slug(model)}" for model in models]
        if models
        else [options.experiment_id]
    )
    experiment_roots = [options.output_root / experiment_id for experiment_id in experiment_ids]
    for experiment_root in experiment_roots:
        experiment_root.mkdir(parents=True, exist_ok=True)

    manifest_root = options.output_root / options.experiment_id
    log_path = manifest_root / "batch.log" if models else experiment_roots[0] / "experiment.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_handle:
        stdout_original = sys.stdout
        stderr_original = sys.stderr
        sys.stdout = Tee(stdout_original, log_handle)
        sys.stderr = Tee(stderr_original, log_handle)
        try:
            with managed_background_services() as register_background_service:
                info(f"Batch root: {options.output_root}")
                info(f"Started at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
                info(f"Benchmarks: {', '.join(selected_configs)}")
                if models:
                    info(f"Models: {', '.join(models)}")
                ignored_model_args = _find_ignored_model_arg_sources()
                if ignored_model_args:
                    info(
                        "Ignoring raw --default-model/--judge-model flags from env arg bundles; "
                        "use --models/--model or benchmark config [models] instead."
                    )
                    for env_name, flags in ignored_model_args.items():
                        info(f"  {env_name}: {', '.join(flags)}")
                warn_non_estimable_metrics(
                    selected_configs,
                    runs_per_task_override=options.runs_per_task,
                )

                if not options.skip_setup:
                    for benchmark_name, config_path in selected_configs.items():
                        info("")
                        info(f"=== Preparing benchmark={benchmark_name} ===")
                        prepare_benchmark_environment(
                            config_path,
                            benchmark_name=benchmark_name,
                            register_background_service=register_background_service,
                        )

                if options.setup_only:
                    info("")
                    info(
                        "Setup completed. Batch execution was skipped because --setup-only was set."
                    )
                    return 0

                progress_interval_s = env_float(
                    os.environ.get("FULL_EXPERIMENT_PROGRESS_INTERVAL_S"),
                    300.0,
                )
                progress_reporter = ProgressReporter(experiment_roots, progress_interval_s)
                if progress_interval_s > 0.0:
                    info("")
                    info(
                        "Live progress snapshots enabled: "
                        f"interval_s={progress_interval_s:.0f}"
                    )
                    progress_reporter.emit_snapshot("Initial progress snapshot")
                    progress_reporter.start()

                try:
                    overall_failures: list[str] = []
                    executed_roots: list[str] = []
                    model_runs = models if models else [None]

                    for model in model_runs:
                        experiment_id = (
                            f"{options.experiment_id}__{_model_slug(model)}"
                            if model
                            else options.experiment_id
                        )
                        experiment_root = options.output_root / experiment_id
                        logs_root = experiment_root / "logs"
                        logs_root.mkdir(parents=True, exist_ok=True)
                        executed_roots.append(str(experiment_root))
                        if model:
                            info("")
                            info(
                                f"=== Running model={model} experiment_id={experiment_id} "
                                f"experiment_root={experiment_root} ==="
                            )
                        else:
                            info("")
                            info(f"Experiment root: {experiment_root}")

                        jobs: list[RunJob] = []
                        skipped_jobs = 0
                        for benchmark_name, config_path in selected_configs.items():
                            for (
                                system_label,
                                topology,
                                agents,
                                rounds,
                                discussion_rounds,
                                communication_budget,
                            ) in selected_systems():
                                system_root = experiment_root / benchmark_name / system_label
                                summary_json = system_root / "summary.json"
                                summary_csv = system_root / "summary.csv"
                                if system_summary_is_clean(summary_json, summary_csv):
                                    skipped_jobs += 1
                                    info("")
                                    info(
                                        f"SKIP benchmark={benchmark_name} system={system_label} "
                                        f"(clean summary found)"
                                    )
                                    continue

                                cmd = [
                                    sys.executable,
                                    "main.py",
                                    "run",
                                    "--config",
                                    str(config_path),
                                    "--benchmark",
                                    benchmark_name,
                                    "--output-dir",
                                    str(options.output_root),
                                    "--output-layout",
                                    "hierarchical",
                                    "--experiment-id",
                                    experiment_id,
                                    "--system-label",
                                    system_label,
                                    "--topology",
                                    topology,
                                    "--agents",
                                    str(agents),
                                    "--mas-rounds",
                                    str(rounds),
                                    "--discussion-rounds",
                                    str(discussion_rounds),
                                    "--communication-budget",
                                    str(communication_budget),
                                ]
                                cmd.extend(system_extra_args(system_label, model=model))
                                if options.task_limit is not None:
                                    cmd.extend(["--task-limit", str(options.task_limit)])
                                if options.runs_per_task is not None:
                                    cmd.extend(["--runs-per-task", str(options.runs_per_task)])
                                if options.final_vote_mode is not None:
                                    cmd.extend(["--final-vote-mode", str(options.final_vote_mode)])
                                if options.benchmark_eval_judge_model is not None:
                                    cmd.extend(
                                        [
                                            "--benchmark-eval-judge-model",
                                            str(options.benchmark_eval_judge_model),
                                        ]
                                    )
                                if options.no_dynamic_roles:
                                    cmd.append("--no-dynamic-roles")
                                jobs.append(
                                    RunJob(
                                        benchmark_name=benchmark_name,
                                        config_path=config_path,
                                        system_label=system_label,
                                        topology=topology,
                                        agents=agents,
                                        rounds=rounds,
                                        discussion_rounds=discussion_rounds,
                                        communication_budget=communication_budget,
                                        cmd=cmd,
                                        log_path=logs_root / benchmark_name / f"{system_label}.log",
                                    )
                                )

                        info("")
                        info(
                            "=== Running batch "
                            f"(jobs={len(jobs)}, skipped={skipped_jobs}, "
                            f"max_parallel={options.max_parallel}, retries={options.retry_failures}) ==="
                        )

                        def run_job(job: RunJob) -> RunJobResult:
                            ensure_parent(job.log_path)
                            started = time.perf_counter()
                            attempt_count = 0
                            returncode = 1
                            with job.log_path.open("a", encoding="utf-8") as cell_log:
                                for attempt in range(options.retry_failures + 1):
                                    attempt_count = attempt + 1
                                    cell_log.write(
                                        f"\n=== attempt {attempt_count}/{options.retry_failures + 1} "
                                        f"benchmark={job.benchmark_name} system={job.system_label} ===\n"
                                    )
                                    cell_log.write(f"$ {shlex.join(job.cmd)}\n")
                                    cell_log.flush()
                                    result = subprocess.run(
                                        job.cmd,
                                        cwd=ROOT,
                                        env=live_env,
                                        text=True,
                                        stdout=cell_log,
                                        stderr=subprocess.STDOUT,
                                        check=False,
                                    )
                                    returncode = result.returncode
                                    if returncode == 0:
                                        break
                                    cell_log.write(
                                        f"--- attempt {attempt_count} failed with code {returncode} ---\n"
                                    )
                                    cell_log.flush()
                            return RunJobResult(
                                job=job,
                                returncode=returncode,
                                attempt_count=attempt_count,
                                elapsed_s=max(time.perf_counter() - started, 0.0),
                            )

                        failures: list[str] = []
                        completed = 0
                        total = len(jobs)
                        active: dict[Future[RunJobResult], RunJob] = {}
                        pending_jobs = list(jobs)

                        with ThreadPoolExecutor(max_workers=options.max_parallel) as executor:
                            while pending_jobs or active:
                                while pending_jobs and len(active) < options.max_parallel:
                                    job = pending_jobs.pop(0)
                                    info("")
                                    info(
                                        f"[{completed + len(active) + 1}/{total}] "
                                        f"START benchmark={job.benchmark_name} system={job.system_label} "
                                        f"log={job.log_path}"
                                    )
                                    future = executor.submit(run_job, job)
                                    active[future] = job

                                done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                                for future in done:
                                    job = active.pop(future)
                                    result = future.result()
                                    completed += 1
                                    if result.returncode != 0:
                                        failures.append(f"{job.benchmark_name}::{job.system_label}")
                                        info(
                                            f"[{completed}/{total}] FAIL benchmark={job.benchmark_name} "
                                            f"system={job.system_label} attempts={result.attempt_count} "
                                            f"elapsed_s={result.elapsed_s:.1f} log={job.log_path}"
                                        )
                                    else:
                                        info(
                                            f"[{completed}/{total}] OK benchmark={job.benchmark_name} "
                                            f"system={job.system_label} attempts={result.attempt_count} "
                                            f"elapsed_s={result.elapsed_s:.1f}"
                                        )

                        info("")
                        info("=== Aggregating experiment summary ===")
                        run_command(
                            [
                                sys.executable,
                                "main.py",
                                "summarize-experiment",
                                "--experiment-root",
                                str(experiment_root),
                            ]
                        )

                        info("")
                        if failures:
                            info("Completed with failures:")
                            for failure in failures:
                                info(f"  - {failure}")
                            label = model if model else options.experiment_id
                            overall_failures.extend(f"{label}:{failure}" for failure in failures)
                        else:
                            info(f"Completed successfully: {experiment_root}")

                    if models:
                        manifest_path = manifest_root / "multi_model_manifest.json"
                        manifest_payload = {
                            "batch_experiment_id": options.experiment_id,
                            "models": [
                                {
                                    "model": model,
                                    "experiment_id": f"{options.experiment_id}__{_model_slug(model)}",
                                    "experiment_root": str(
                                        (options.output_root / f"{options.experiment_id}__{_model_slug(model)}").resolve()
                                    ),
                                }
                                for model in models
                            ],
                        }
                        manifest_path.write_text(
                            json.dumps(manifest_payload, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                        info(f"Wrote multi-model manifest: {manifest_path}")

                    if overall_failures:
                        return 1

                    if models:
                        info("Completed successfully:")
                        for root in executed_roots:
                            info(f"  - {root}")
                    return 0
                finally:
                    progress_reporter.stop()
        finally:
            sys.stdout = stdout_original
            sys.stderr = stderr_original


def maybe_run_legacy(argv: list[str]) -> int | None:
    if "--single-run" in argv:
        forwarded = [arg for arg in argv if arg != "--single-run"]
        cmd = [sys.executable, "-m", "MAS.experiment_cli", *forwarded]
        return subprocess.run(cmd, cwd=ROOT, check=False).returncode
    if "--topology" in argv:
        cmd = [sys.executable, "-m", "MAS.experiment_cli", *argv]
        return subprocess.run(cmd, cwd=ROOT, check=False).returncode
    return None


def system_extra_args(system_label: str, *, model: str | None = None) -> list[str]:
    args: list[str] = []
    global_args = os.getenv("MAS_GLOBAL_ARGS", "").strip()
    if global_args:
        args.extend(shlex.split(global_args))
    env_name = SYSTEM_ARG_ENV_BY_LABEL.get(system_label)
    if env_name:
        system_args = os.getenv(env_name, "").strip()
        if system_args:
            args.extend(shlex.split(system_args))
    return _merge_model_override(args, model)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    legacy_exit = maybe_run_legacy(args)
    if legacy_exit is not None:
        return legacy_exit
    options = parse_batch_args(args)
    if options.list_benchmarks:
        return print_available_benchmarks(options.config_dir)
    return batch_run(options)


if __name__ == "__main__":
    raise SystemExit(main())

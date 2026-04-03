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
import time
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
]

SYSTEM_ARG_ENV_BY_LABEL: dict[str, str] = {
    "sas": "SAS_ARGS",
    "orchestrator_tree_structure": "ORCHESTRATOR_TREE_STRUCTURE_ARGS",
    "orchestrator_no_discussion": "ORCHESTRATOR_NO_DISCUSSION_ARGS",
    "orchestrator_with_discussion": "ORCHESTRATOR_WITH_DISCUSSION_ARGS",
    "only_voting": "ONLY_VOTING_ARGS",
    "fully_linked_debate": "FULLY_LINKED_DEBATE_ARGS",
    "group_chat_debate": "GROUP_CHAT_DEBATE_ARGS",
}

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
    final_vote_mode: str | None
    skip_setup: bool
    setup_only: bool


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
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--runs-per-task", type=int, default=None)
    parser.add_argument(
        "--final-vote-mode",
        choices=["llm_judge", "deterministic"],
        default=None,
        help="Override MAS final answer selection mode for every launched system.",
    )
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args(argv)

    selected = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    return BatchOptions(
        experiment_id=str(args.experiment_id),
        output_root=Path(args.output_root).expanduser().resolve(),
        config_dir=Path(args.config_dir).expanduser().resolve(),
        benchmarks=selected or None,
        task_limit=args.task_limit,
        runs_per_task=args.runs_per_task,
        final_vote_mode=args.final_vote_mode,
        skip_setup=bool(args.skip_setup),
        setup_only=bool(args.setup_only),
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


def service_pid_path(state_dir: Path, name: str) -> Path:
    return state_dir / f"{name}.pid"


def process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def stop_stale_process(pid_path: Path) -> None:
    pid = read_pid(pid_path)
    if pid is None:
        return
    if not process_is_running(pid):
        pid_path.unlink(missing_ok=True)
        return
    with contextlib.suppress(OSError):
        os.killpg(pid, signal.SIGTERM)
    pid_path.unlink(missing_ok=True)


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


def prepare_stabletoolbench(config: dict[str, Any]) -> None:
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


def prepare_agentbench(config: dict[str, Any]) -> None:
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

    wait_for_command_success(
        timeout_s=300,
        fn=lambda: controller_health_check(controller_address, task_name),
    )

    benchmark = AgentBenchBenchmark(cfg)
    tasks = benchmark.load_tasks(task_limit=1)
    info(f"Prepared benchmark=agentbench sample_tasks={len(tasks)}")


def prepare_benchmark_environment(config_path: Path, *, benchmark_name: str) -> None:
    config = load_toml(config_path)
    benchmark_cfg = dict(config.get(benchmark_name, {}))

    if benchmark_name == "browsecomp":
        prepare_browsecomp(benchmark_cfg)
        return
    if benchmark_name == "stabletoolbench":
        prepare_stabletoolbench(benchmark_cfg)
        return
    if benchmark_name == "agentbench":
        prepare_agentbench(benchmark_cfg)
        return

    warmup_benchmark(benchmark_name, benchmark_cfg)


def batch_run(options: BatchOptions) -> int:
    selected_configs = select_benchmarks(options)
    experiment_root = options.output_root / options.experiment_id
    experiment_root.mkdir(parents=True, exist_ok=True)
    live_env = dict(os.environ)
    live_env["MAS_REQUIRE_LIVE_LLM"] = "1"

    jobs: list[RunJob] = []
    logs_root = experiment_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    log_path = experiment_root / "experiment.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        stdout_original = sys.stdout
        stderr_original = sys.stderr
        sys.stdout = Tee(stdout_original, log_handle)
        sys.stderr = Tee(stderr_original, log_handle)
        try:
            info(f"Experiment root: {experiment_root}")
            info(f"Started at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

            if not options.skip_setup:
                for benchmark_name, config_path in selected_configs.items():
                    info("")
                    info(f"=== Preparing benchmark={benchmark_name} ===")
                    prepare_benchmark_environment(config_path, benchmark_name=benchmark_name)

            if options.setup_only:
                info("")
                info("Setup completed. Batch execution was skipped because --setup-only was set.")
                return 0

            skipped_jobs = 0
            for benchmark_name, config_path in selected_configs.items():
                for (
                    system_label,
                    topology,
                    agents,
                    rounds,
                    discussion_rounds,
                    communication_budget,
                ) in SYSTEMS:
                    system_root = experiment_root / benchmark_name / system_label
                    summary_json = system_root / "summary.json"
                    summary_csv = system_root / "summary.csv"
                    if summary_json.exists() and summary_csv.exists():
                        skipped_jobs += 1
                        info("")
                        info(
                            f"SKIP benchmark={benchmark_name} system={system_label} "
                            f"(existing summary found)"
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
                        options.experiment_id,
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
                    cmd.extend(system_extra_args(system_label))
                    if options.task_limit is not None:
                        cmd.extend(["--task-limit", str(options.task_limit)])
                    if options.runs_per_task is not None:
                        cmd.extend(["--runs-per-task", str(options.runs_per_task)])
                    if options.final_vote_mode is not None:
                        cmd.extend(["--final-vote-mode", str(options.final_vote_mode)])

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

            if failures:
                info("")
                info("Completed with failures:")
                for failure in failures:
                    info(f"  - {failure}")
                return 1

            info("")
            info(f"Completed successfully: {experiment_root}")
            return 0
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


def system_extra_args(system_label: str) -> list[str]:
    args: list[str] = []
    global_args = os.getenv("MAS_GLOBAL_ARGS", "").strip()
    if global_args:
        args.extend(shlex.split(global_args))
    env_name = SYSTEM_ARG_ENV_BY_LABEL.get(system_label)
    if env_name:
        system_args = os.getenv(env_name, "").strip()
        if system_args:
            args.extend(shlex.split(system_args))
    return args


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    legacy_exit = maybe_run_legacy(args)
    if legacy_exit is not None:
        return legacy_exit
    options = parse_batch_args(args)
    return batch_run(options)


if __name__ == "__main__":
    raise SystemExit(main())

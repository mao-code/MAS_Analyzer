"""StableToolBench benchmark adapter.

Source repo : https://github.com/THUNLP-MT/StableToolBench
Paper       : https://arxiv.org/abs/2403.07714

This adapter uses the StableToolBench solvable query files and the upstream
GPT-based virtual API server. Each query's ``api_list`` is exposed as a set of
OpenAI-compatible tools to the MAS runtime, and each tool call is proxied to
``/virtual`` on the external StableToolBench server.

Evaluation modes:
    1. heuristic   - Cheap local smoke-test grading. Useful for plumbing checks.
    2. llm_judge   - SoPR-style answer-status grading on solvable queries.

Official StableToolBench also reports pairwise SoWR (win rate) via a separate
cross-model evaluation pass. That is not folded into ``main.py`` because the
current benchmark interface scores one run at a time.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from ..base import BenchmarkEvaluation, BenchmarkTask

if TYPE_CHECKING:
    from MAS.runner import MASRunResult

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_QUERY_ROOT = PACKAGE_ROOT / "data" / "solvable_queries"
DEFAULT_DOWNLOAD_BASE_URL = (
    "https://raw.githubusercontent.com/THUNLP-MT/StableToolBench/master/solvable_queries"
)
DEFAULT_SERVER_ASSETS_ROOT = PACKAGE_ROOT
DEFAULT_TOOLS_DIR = DEFAULT_SERVER_ASSETS_ROOT / "tools"
DEFAULT_CACHE_DIR = DEFAULT_SERVER_ASSETS_ROOT / "tool_response_cache"
DEFAULT_HF_TOOLS_REPO = "stabletoolbench/ToolEnv2404"
DEFAULT_HF_TOOLS_FILE = "toolenv2404_filtered.tar.gz"
DEFAULT_HF_CACHE_REPO = "stabletoolbench/Cache"
DEFAULT_HF_CACHE_FILE = "server_cache.zip"
DEFAULT_TASK_SETS = [
    "G1_instruction",
    "G1_category",
    "G1_tool",
    "G2_instruction",
    "G2_category",
    "G3_instruction",
]

TOOL_USAGE_GUIDANCE = """\
You are solving a StableToolBench task with external API tools.

Use the provided tools when they are useful. Tool outputs may contain raw JSON
or plain text. If a tool fails, continue with the best information you have and
explain the failure clearly in the final answer.

Important:
- You do not need a special Finish tool in this environment.
- Return a plain final answer to the user once you have enough information.
- If the task has multiple parts, address all parts in the final answer.
"""

LLM_JUDGE_PROMPT = """\
You are grading a StableToolBench answer on a solvable query.

Return JSON with exactly two keys:
- "answer_status": one of "Solved", "Unsolved", or "Unsure"
- "reason": a short explanation

Judging rules:
1. Mark "Solved" if the answer makes a genuine attempt to satisfy all parts of
   the query.
2. Mark "Unsolved" if the answer is empty, refuses, apologizes instead of
   answering, is clearly irrelevant, or leaves part of a multi-part request
   unaddressed.
3. Mark "Unsure" only if the answer is non-empty but you genuinely cannot tell
   whether it satisfies the whole request.
4. Do not require perfect wording. Focus on whether the information need is
   satisfied.

Query:
{query}

Answer:
{answer}
"""

HEURISTIC_REFUSAL_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "sorry,",
    "cannot help",
    "can't help",
    "unable to",
    "i cannot",
    "i can't",
    "do not have enough information",
)


def _standardize(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z_]", "_", str(value or ""))
    result = re.sub(r"_+", "_", result).strip("_").lower()
    if result and result[0].isdigit():
        result = f"get_{result}"
    return result


def _change_name(name: str) -> str:
    if name in {"from", "class", "return", "false", "true", "id", "and"}:
        return f"is_{name}"
    return name


def _canonical_tool_name(api_meta: dict[str, Any]) -> str:
    api_name = _change_name(_standardize(str(api_meta.get("api_name", ""))))
    tool_name = _standardize(str(api_meta.get("tool_name", "")))
    return f"{api_name}_for_{tool_name}"


def _json_schema_type(raw_type: Any) -> str:
    value = str(raw_type or "string").strip().lower()
    if value in {"int", "integer"}:
        return "integer"
    if value in {"float", "double", "number", "decimal"}:
        return "number"
    if value in {"bool", "boolean"}:
        return "boolean"
    if value in {"array", "list"}:
        return "array"
    if value in {"object", "dict", "json"}:
        return "object"
    return "string"


def _task_sets_from_config(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_TASK_SETS)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError("stabletoolbench.task_sets must be a string or list of strings")


class StableToolBenchBenchmark:
    """StableToolBench adapter backed by the GPT-based virtual server.

    Config keys (all optional):
        query_root          Local root containing test_instruction/ and
                            test_query_ids/ subfolders.
        task_sets           Subsets to load. Defaults to all six official sets.
        auto_download       Download missing query assets from GitHub.
        download_base_url   Raw GitHub base for solvable_queries.
        auto_download_server_assets
                            Download missing local tools/cache assets from
                            Hugging Face.
        server_assets_root  Local root containing tools/ and
                            tool_response_cache/.
        tools_dir           Override tools/ directory path directly.
        tool_cache_dir      Override tool_response_cache/ directory path
                            directly.
        virtual_server_url  StableToolBench server URL (default:
                            http://localhost:8080/virtual).
        toolbench_key       Optional ToolBench key forwarded to the server.
        request_timeout_s   HTTP timeout for virtual tool calls.
        strip               Payload field forwarded to the server.
        enable_tools        Enable virtual API tools for MAS.
        max_tools_per_task  Optional cap on APIs exposed per task.
        max_tool_iterations Max tool-calling rounds for the MAS runtime.
        eval_mode           heuristic | llm_judge | sopr.
        judge_model         OpenAI-compatible model for llm_judge.
        judge_api_key       API key for llm_judge. Falls back to OPENAI_API_KEY.
        judge_api_base      Base URL for llm_judge. Falls back to OPENAI_API_BASE
                            or https://api.openai.com/v1.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}

        self.query_root = Path(str(cfg.get("query_root", DEFAULT_QUERY_ROOT)))
        self.task_sets = _task_sets_from_config(cfg.get("task_sets"))
        self.auto_download = bool(cfg.get("auto_download", True))
        self.download_base_url = str(
            cfg.get("download_base_url", DEFAULT_DOWNLOAD_BASE_URL)
        ).rstrip("/")
        self.auto_download_server_assets = bool(cfg.get("auto_download_server_assets", False))
        self.server_assets_root = Path(
            str(cfg.get("server_assets_root", DEFAULT_SERVER_ASSETS_ROOT))
        )
        self.tools_dir = Path(str(cfg.get("tools_dir", self.server_assets_root / "tools")))
        self.tool_cache_dir = Path(
            str(cfg.get("tool_cache_dir", self.server_assets_root / "tool_response_cache"))
        )
        self.hf_tools_repo = str(cfg.get("hf_tools_repo", DEFAULT_HF_TOOLS_REPO)).strip()
        self.hf_tools_file = str(cfg.get("hf_tools_file", DEFAULT_HF_TOOLS_FILE)).strip()
        self.hf_cache_repo = str(cfg.get("hf_cache_repo", DEFAULT_HF_CACHE_REPO)).strip()
        self.hf_cache_file = str(cfg.get("hf_cache_file", DEFAULT_HF_CACHE_FILE)).strip()

        self.virtual_server_url = str(
            cfg.get(
                "virtual_server_url",
                os.getenv("STABLETOOLBENCH_VIRTUAL_SERVER_URL", "http://localhost:8080/virtual"),
            )
        ).rstrip("/")
        self.toolbench_key = str(cfg.get("toolbench_key", os.getenv("TOOLBENCH_KEY", "")))
        self.request_timeout_s = float(cfg.get("request_timeout_s", 120.0))
        self.strip = str(cfg.get("strip", "truncate"))
        self.enable_tools = bool(cfg.get("enable_tools", True))

        max_tools_per_task = cfg.get("max_tools_per_task")
        if max_tools_per_task in (None, "", False):
            self.max_tools_per_task: int | None = None
        else:
            self.max_tools_per_task = max(1, int(max_tools_per_task))
        self.max_tool_iterations = max(1, int(cfg.get("max_tool_iterations", 8)))

        self.eval_mode = str(cfg.get("eval_mode", "heuristic")).strip().lower()
        self.judge_model = str(cfg.get("judge_model", "gpt-4.1-mini"))
        self.judge_api_key = str(cfg.get("judge_api_key") or os.getenv("OPENAI_API_KEY") or "")
        self.judge_api_base = str(
            cfg.get("judge_api_base") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        )
        self.judge_temperature = float(cfg.get("judge_temperature", 0.0))
        self.judge_max_tokens = int(cfg.get("judge_max_tokens", 256))
        self.judge_timeout_s = float(cfg.get("judge_timeout_s", 60.0))
        self.judge_http_referer = str(cfg.get("judge_http_referer", "")).strip()
        self.judge_x_title = str(cfg.get("judge_x_title", "MAS Analyzer StableToolBench")).strip()

        self._session = requests.Session()
        self._judge_client: Any | None = None
        self._task_api_lists: dict[str, list[dict[str, Any]]] = {}

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        self._ensure_query_assets()
        self._ensure_server_assets()
        self._task_api_lists = {}

        tasks: list[BenchmarkTask] = []
        seen_task_ids: set[str] = set()

        for task_set in self.task_sets:
            query_path = self.query_root / "test_instruction" / f"{task_set}.json"
            ids_path = self.query_root / "test_query_ids" / f"{task_set}.json"

            rows = json.loads(query_path.read_text(encoding="utf-8"))
            id_order = list(json.loads(ids_path.read_text(encoding="utf-8")).keys())
            rows_by_id = {str(item.get("query_id")): item for item in rows}

            for query_id in id_order:
                row = rows_by_id.get(str(query_id))
                if row is None:
                    logger.warning(
                        "StableToolBench query id %s missing from %s", query_id, query_path
                    )
                    continue

                task_id = str(query_id)
                if task_id in seen_task_ids:
                    task_id = f"{task_set}_{task_id}"
                seen_task_ids.add(task_id)

                query = str(row.get("query", "")).strip()
                api_list = list(row.get("api_list") or [])
                tool_names = [_canonical_tool_name(item) for item in api_list]

                self._task_api_lists[task_id] = api_list
                tasks.append(
                    BenchmarkTask(
                        task_id=task_id,
                        prompt=self._build_task_prompt(query),
                        reference_answer="",
                        metadata={
                            "query": query,
                            "query_id": str(query_id),
                            "task_set": task_set,
                            "source": "stabletoolbench",
                            "available_tool_names": tool_names,
                            "relevant_apis": list(row.get("relevant APIs") or []),
                        },
                    )
                )
                if task_limit is not None and len(tasks) >= task_limit:
                    return tasks

        return tasks

    def run(
        self,
        task: BenchmarkTask,
        runner: Any,
        run_index: int,
        seed: int,
    ) -> MASRunResult:
        tools: list[dict[str, Any]] = []
        if self.enable_tools:
            tools = self._build_tools_for_task(task.task_id)
        return runner.run_task(
            task=task,
            run_index=run_index,
            seed=seed,
            tools=tools,
            max_tool_iterations=self.max_tool_iterations,
            benchmark_name="stabletoolbench",
        )

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        if self.eval_mode in {"llm_judge", "sopr", "sopr_llm"}:
            return self._evaluate_llm_judge(task, prediction, run_metadata or {})
        return self._evaluate_heuristic(task, prediction, run_metadata or {})

    def requirements(self) -> dict[str, Any]:
        return {
            "benchmark": "stabletoolbench",
            "version": "1.0",
            "dataset_source": "https://github.com/THUNLP-MT/StableToolBench",
            "metrics": [
                "stabletoolbench_solve_score",
                "stabletoolbench_answer_status",
                "tool_calls_total",
            ],
            "notes": [
                "Uses StableToolBench solvable query files (official six subsets).",
                "Requires the upstream virtual server to be started separately.",
                "GPT-based virtual server uses cache/tools from StableToolBench upstream.",
                "Local tools/cache assets can be auto-downloaded from Hugging Face when enabled.",
                "llm_judge mode approximates official SoPR on the solvable-query split.",
                "Official pairwise SoWR is not integrated into main.py because it is cross-run, cross-model evaluation.",
            ],
        }

    def _ensure_query_assets(self) -> None:
        missing: list[tuple[str, Path]] = []
        for task_set in self.task_sets:
            missing_query = self.query_root / "test_instruction" / f"{task_set}.json"
            missing_ids = self.query_root / "test_query_ids" / f"{task_set}.json"
            if not missing_query.exists():
                missing.append((f"test_instruction/{task_set}.json", missing_query))
            if not missing_ids.exists():
                missing.append((f"test_query_ids/{task_set}.json", missing_ids))

        if not missing:
            return

        if not self.auto_download:
            wanted = ", ".join(str(path) for _, path in missing)
            raise FileNotFoundError(
                "StableToolBench query assets are missing and auto_download=false. "
                f"Missing: {wanted}"
            )

        for relative_path, target_path in missing:
            url = f"{self.download_base_url}/{relative_path}"
            logger.info("Downloading StableToolBench asset %s", url)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            response = self._session.get(url, timeout=self.request_timeout_s)
            response.raise_for_status()
            target_path.write_text(response.text, encoding="utf-8")

    def _ensure_server_assets(self) -> None:
        missing_labels: list[str] = []
        if not self._resolved_tools_dir().exists():
            missing_labels.append(str(self.tools_dir))
        if not self._resolved_tool_cache_dir().exists():
            missing_labels.append(str(self.tool_cache_dir))

        if not missing_labels:
            return

        if not self.auto_download_server_assets:
            logger.info(
                "StableToolBench server assets missing locally. "
                "Set stabletoolbench.auto_download_server_assets=true to fetch them. Missing: %s",
                ", ".join(missing_labels),
            )
            return

        if not self._resolved_tools_dir().exists():
            self._download_and_extract_hf_archive(
                repo_id=self.hf_tools_repo,
                filename=self.hf_tools_file,
                expected_path=self.tools_dir,
                alternate_paths=[
                    self.server_assets_root / "toolenv2404_filtered",
                ],
            )
        if not self._resolved_tool_cache_dir().exists():
            self._download_and_extract_hf_archive(
                repo_id=self.hf_cache_repo,
                filename=self.hf_cache_file,
                expected_path=self.tool_cache_dir,
                alternate_paths=[
                    self.server_assets_root / "server_cache",
                ],
            )

    def _download_and_extract_hf_archive(
        self,
        *,
        repo_id: str,
        filename: str,
        expected_path: Path,
        alternate_paths: Sequence[Path] | None = None,
    ) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise RuntimeError(
                "huggingface_hub is required to auto-download StableToolBench server assets"
            ) from exc

        expected_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading StableToolBench server asset %s/%s", repo_id, filename)
        archive_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
            )
        )

        if filename.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(expected_path.parent)
        elif filename.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as archive:
                extract_kwargs: dict[str, Any] = {}
                if hasattr(tarfile, "data_filter"):
                    extract_kwargs["filter"] = "data"
                archive.extractall(expected_path.parent, **extract_kwargs)
        else:
            raise RuntimeError(f"Unsupported StableToolBench asset archive: {filename}")

        alternate_paths = list(alternate_paths or [])
        if not expected_path.exists():
            for candidate in alternate_paths:
                if candidate.exists():
                    expected_path.parent.mkdir(parents=True, exist_ok=True)
                    if expected_path.exists():
                        break
                    candidate.rename(expected_path)
                    break

        if not expected_path.exists():
            raise RuntimeError(
                "StableToolBench asset download finished but the expected path was not created: "
                f"{expected_path}"
            )

    def _resolved_tools_dir(self) -> Path:
        if self.tools_dir.exists():
            return self.tools_dir
        alternate = self.server_assets_root / "toolenv2404_filtered"
        if alternate.exists():
            return alternate
        return self.tools_dir

    def _resolved_tool_cache_dir(self) -> Path:
        if self.tool_cache_dir.exists():
            return self.tool_cache_dir
        alternate = self.server_assets_root / "server_cache"
        if alternate.exists():
            return alternate
        return self.tool_cache_dir

    @staticmethod
    def _build_task_prompt(query: str) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": f"{TOOL_USAGE_GUIDANCE}\n\nUser query:\n{query}",
            }
        ]

    def _build_tools_for_task(self, task_id: str) -> list[dict[str, Any]]:
        api_list = list(self._task_api_lists.get(task_id, []))
        if self.max_tools_per_task is not None:
            api_list = api_list[: self.max_tools_per_task]

        tools: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for api_meta in api_list:
            name = _canonical_tool_name(api_meta)
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            tools.append(self._build_virtual_tool(api_meta))
        return tools

    def _build_virtual_tool(self, api_meta: dict[str, Any]) -> dict[str, Any]:
        canonical_name = _canonical_tool_name(api_meta)
        tool_name = str(api_meta.get("tool_name", "")).strip()
        api_name = str(api_meta.get("api_name", "")).strip()
        api_description = str(api_meta.get("api_description", "")).strip()
        method = str(api_meta.get("method", "")).strip().upper()

        def _handler(args: dict[str, Any], *, meta: dict[str, Any] = api_meta) -> Any:
            return self._call_virtual_api(meta, args)

        return {
            "name": canonical_name,
            "description": (
                f'This is the subfunction for tool "{tool_name}", you can use this tool.'
                f'The description of this function is: "{api_description}".'
                f" HTTP method: {method or 'GET'}."
            ),
            "parameters": self._json_schema_for_api(api_meta),
            "handler": _handler,
            "stabletoolbench_meta": {
                "category_name": api_meta.get("category_name"),
                "tool_name": tool_name,
                "api_name": api_name,
            },
        }

    @staticmethod
    def _json_schema_for_api(api_meta: dict[str, Any]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for bucket_name, is_required in (
            ("required_parameters", True),
            ("optional_parameters", False),
        ):
            for param in list(api_meta.get(bucket_name) or []):
                name = str(param.get("name", "")).strip()
                if not name:
                    continue
                description = str(param.get("description", "")).strip()
                default = param.get("default")
                if default not in (None, "", []):
                    default_text = json.dumps(default, ensure_ascii=False)
                    if description:
                        description = f"{description} Default: {default_text}"
                    else:
                        description = f"Default: {default_text}"
                properties[name] = {
                    "type": _json_schema_type(param.get("type")),
                    "description": description,
                }
                if is_required:
                    required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": True,
        }

    def _call_virtual_api(self, api_meta: dict[str, Any], args: dict[str, Any]) -> Any:
        tool_input = self._resolved_tool_input(api_meta, args)
        payload = {
            "category": str(api_meta.get("category_name", "")),
            "tool_name": str(api_meta.get("tool_name", "")),
            "api_name": str(api_meta.get("api_name", "")),
            "tool_input": json.dumps(tool_input, ensure_ascii=False),
            "strip": self.strip,
            "toolbench_key": self.toolbench_key,
        }

        try:
            response = self._session.post(
                self.virtual_server_url,
                json=payload,
                timeout=self.request_timeout_s,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Virtual server request failed: {exc}") from exc

        if response.status_code != 200:
            body_preview = response.text.strip()[:300]
            raise RuntimeError(
                f"Virtual server returned HTTP {response.status_code}: {body_preview}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise RuntimeError("Virtual server returned invalid JSON") from exc

        if not isinstance(body, dict):
            return body

        error = str(body.get("error", "") or "").strip()
        if error:
            raise RuntimeError(error)

        response_payload = body.get("response", "")
        if isinstance(response_payload, str):
            trimmed = response_payload.strip()
            if trimmed.startswith("{") or trimmed.startswith("["):
                try:
                    return json.loads(trimmed)
                except ValueError:
                    return response_payload
        return response_payload

    @staticmethod
    def _resolved_tool_input(api_meta: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        provided = dict(args or {})
        consumed_keys: set[str] = set()

        for bucket_name in ("required_parameters", "optional_parameters"):
            for param in list(api_meta.get(bucket_name) or []):
                name = str(param.get("name", "")).strip()
                if not name:
                    continue
                consumed_keys.add(name)
                if name in provided and provided[name] not in (None, ""):
                    resolved[name] = provided[name]
                    continue
                default = param.get("default")
                if default not in (None, "", []):
                    resolved[name] = default

        for key, value in provided.items():
            if key not in consumed_keys and value is not None:
                resolved[key] = value

        return resolved

    def _evaluate_heuristic(
        self,
        task: BenchmarkTask,
        prediction: str,
        run_metadata: dict[str, Any],
    ) -> BenchmarkEvaluation:
        text = str(prediction or "").strip()
        lowered = text.lower()

        if not text:
            answer_status = "Unsolved"
            reason = "Empty final answer."
        elif any(marker in lowered for marker in HEURISTIC_REFUSAL_MARKERS):
            answer_status = "Unsolved"
            reason = "Detected refusal or apology phrasing."
        elif len(re.findall(r"\w+", text)) < 8:
            answer_status = "Unsure"
            reason = "Answer is non-empty but too short for confident grading."
        else:
            answer_status = "Solved"
            reason = "Non-empty answer without refusal markers."

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=self._score_for_status(answer_status),
            success=answer_status == "Solved",
            details={
                "eval_mode": "heuristic",
                "answer_status": answer_status,
                "reason": reason,
                "task_set": task.metadata.get("task_set", ""),
                "query_id": task.metadata.get("query_id", task.task_id),
                "run_metadata": run_metadata,
            },
        )

    def _evaluate_llm_judge(
        self,
        task: BenchmarkTask,
        prediction: str,
        run_metadata: dict[str, Any],
    ) -> BenchmarkEvaluation:
        text = str(prediction or "").strip()
        if not text:
            answer_status = "Unsolved"
            reason = "Empty final answer."
            raw_judge = {"answer_status": answer_status, "reason": reason}
        else:
            raw_judge = self._judge_answer(task.metadata.get("query", ""), text)
            answer_status = raw_judge["answer_status"]
            reason = raw_judge["reason"]

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=self._score_for_status(answer_status),
            success=answer_status == "Solved",
            details={
                "eval_mode": "llm_judge",
                "answer_status": answer_status,
                "reason": reason,
                "judge_model": self.judge_model,
                "judge_raw": raw_judge,
                "task_set": task.metadata.get("task_set", ""),
                "query_id": task.metadata.get("query_id", task.task_id),
                "run_metadata": run_metadata,
            },
        )

    def _judge_answer(self, query: str, answer: str) -> dict[str, str]:
        client = self._judge_client_instance()
        prompt = LLM_JUDGE_PROMPT.format(query=query, answer=answer)
        messages = [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ]
        request_kwargs = {
            "model": self.judge_model,
            "messages": messages,
            "temperature": self.judge_temperature,
            "max_tokens": self.judge_max_tokens,
        }

        try:
            completion = client.chat.completions.create(
                response_format={"type": "json_object"},
                **request_kwargs,
            )
        except Exception:
            completion = client.chat.completions.create(**request_kwargs)

        content = self._extract_message_text(completion)
        return self._parse_judge_payload(content)

    def _judge_client_instance(self) -> Any:
        if self._judge_client is not None:
            return self._judge_client
        if not self.judge_api_key:
            raise RuntimeError(
                "stabletoolbench.eval_mode=llm_judge requires judge_api_key or OPENAI_API_KEY"
            )

        try:
            import openai  # type: ignore
        except Exception as exc:
            raise RuntimeError("openai package is required for llm_judge evaluation") from exc

        headers: dict[str, str] = {}
        if self.judge_http_referer:
            headers["HTTP-Referer"] = self.judge_http_referer
        if self.judge_x_title:
            headers["X-Title"] = self.judge_x_title

        kwargs: dict[str, Any] = {
            "api_key": self.judge_api_key,
            "base_url": self.judge_api_base,
            "timeout": self.judge_timeout_s,
        }
        if headers:
            kwargs["default_headers"] = headers

        self._judge_client = openai.OpenAI(**kwargs)
        return self._judge_client

    @staticmethod
    def _extract_message_text(completion: Any) -> str:
        choice = completion.choices[0] if getattr(completion, "choices", None) else None
        message = getattr(choice, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                else:
                    text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            return "\n".join(parts).strip()
        return str(content or "")

    @classmethod
    def _parse_judge_payload(cls, content: str) -> dict[str, str]:
        parsed: dict[str, Any] = {}
        content = str(content or "").strip()
        if content:
            try:
                parsed = json.loads(content)
            except ValueError:
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except ValueError:
                        parsed = {}

        raw_status = str(parsed.get("answer_status", "") or "").strip()
        normalized_status = cls._normalize_answer_status(raw_status)
        if not normalized_status:
            lowered = content.lower()
            if "unsure" in lowered:
                normalized_status = "Unsure"
            elif "unsolved" in lowered:
                normalized_status = "Unsolved"
            else:
                normalized_status = "Solved"

        reason = str(parsed.get("reason", "") or "").strip()
        if not reason:
            reason = content[:400]

        return {
            "answer_status": normalized_status,
            "reason": reason,
        }

    @staticmethod
    def _normalize_answer_status(value: str) -> str:
        lowered = str(value or "").strip().lower()
        if lowered == "solved":
            return "Solved"
        if lowered == "unsolved":
            return "Unsolved"
        if lowered == "unsure":
            return "Unsure"
        return ""

    @staticmethod
    def _score_for_status(answer_status: str) -> float:
        if answer_status == "Solved":
            return 1.0
        if answer_status == "Unsure":
            return 0.5
        return 0.0

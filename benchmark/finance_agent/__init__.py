"""Vals Finance Agent Benchmark adapter.

Data source: https://github.com/vals-ai/finance-agent
Leaderboard: https://www.vals.ai/benchmarks/finance_agent

CSV schema (public.csv):
    Question, Answer, Question Type, Expert time (mins), Rubric

Rubric is a JSON list of criteria dicts, each with:
    {"operator": "correctness"|"contradiction", "criteria": "<text>"}

Evaluation modes:
    1. llm_judge (default)  - Uses an LLM to judge each criterion against the
       prediction. Mirrors the official Vals evaluation: mode-of-3 LLM calls
       per criterion.
    2. substring            - Simple normalised substring matching (fast, no
       API cost, lower accuracy). Useful for debugging and quick iteration.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import os
import random
import re
import time
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from statistics import mode
from typing import TYPE_CHECKING, Any

from benchmark.base import BenchmarkEvaluation, BenchmarkTask

if TYPE_CHECKING:
    from MAS.runner import MASRunResult

logger = logging.getLogger(__name__)

# Pinned commit so results are reproducible even if upstream main changes.
PUBLIC_CSV_URL = (
    "https://raw.githubusercontent.com/vals-ai/finance-agent/"
    "aad00743ce54b348678a2073aac51fba825ca901/data/public.csv"
)
MAX_END_DATE = "2025-04-07"

# ---------------------------------------------------------------------------
# LLM-as-Judge prompt template
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert financial analyst acting as a judge. Your job is to
determine whether a PREDICTION satisfies a specific evaluation CRITERION
with respect to a REFERENCE ANSWER.

Rules:
- Focus on factual accuracy. Minor formatting or phrasing differences are OK.
- For numerical values: allow up to 2% relative tolerance unless the
  criterion specifies exact figures.  "$3.25 Billion" matches "$3.25B",
  "3,250,000,000", "3.25 billion", etc.
- For percentages: "22.6%" matches "22.60%", "~23%" does NOT match "22.6%".
- Rounding: if the prediction rounds an intermediate number but arrives at
  the same final answer, that is acceptable.
- If the criterion operator is "contradiction", you are checking whether
  the prediction CONTRADICTS the reference.  Return "yes" if there IS a
  contradiction, "no" if there is no contradiction.

Respond with ONLY a JSON object: {"judgment": "yes" | "no"}
"yes" = the criterion is satisfied (correctness met, or contradiction found).
"no"  = the criterion is NOT satisfied.
"""

_JUDGE_USER_TEMPLATE = """\
REFERENCE ANSWER:
{reference_answer}

PREDICTION:
{prediction}

CRITERION ({operator}):
{criteria}
"""

_OFFICIAL_INSTRUCTIONS_PROMPT = """
You are a financial agent. You are given a question and you need to answer it using the tools provided.
You will not be able to interact with the user or ask clarifications, you must answer the question only based on the information provided.

You should answer all questions as if the current date is April 07, 2025.

You will have access to a data storage system. You can use this system to store parsed contents of HTML pages retrieved from the web.
You can then use the retrieve_information tool to apply answer questions or gather information from the stored documents using LLM-based prompts.
This data storage system is designed to help you avoid context window issues.

When you have the final answer, output it directly in your final response.

You should include any necessary step-by-step reasoning, justification, calculations, or explanation in your answer. You will be evaluated both on the accuracy of the final answer, and the correctness of the supporting logic.

When possible, please provide any calculated answers to at least two decimal places (e.g. 18.78% rather than 19%). Please do not round intermediate steps in any calculations - you should only round your final answer.

At the end of your answer, you should provide your sources in a dictionary with the following format:
{{
    "sources": [
        {{
            "url": "https://example.com",
            "name": "Name of the source"
        }},
        ...
    ]
}}

Question:
{question}
"""


class FinanceAgentBenchmark:
    """Adapter for the Vals Finance Agent public benchmark CSV.

    Config keys (all optional):
        dataset_url         URL to the public CSV (default: pinned commit).
        cache_dir           Local cache directory (default: .cache/finance_agent).
        local_csv_path      Path to a pre-downloaded CSV (overrides download).
        success_threshold   Score >= this is considered success (default: 0.5).
        eval_mode           "llm_judge" or "substring" (default: "llm_judge").
        judge_model         Model name for LLM judge (default: "openai/gpt-4o").
        judge_repeats       Number of repeated judge calls; final answer is
                            the mode (default: 3, matching Vals methodology).
        judge_temperature   Temperature for judge calls (default: 0.0).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.dataset_url = str(cfg.get("dataset_url", PUBLIC_CSV_URL))
        self.cache_dir = Path(str(cfg.get("cache_dir", ".cache/finance_agent")))
        self.local_csv_path = cfg.get("local_csv_path")
        self.success_threshold = float(cfg.get("success_threshold", 0.5))

        # Evaluation settings
        self.eval_mode: str = str(cfg.get("eval_mode", "llm_judge"))
        self.judge_model: str = str(cfg.get("judge_model", "openai/gpt-4o"))
        self.judge_repeats: int = int(cfg.get("judge_repeats", 3))
        self.judge_temperature: float = float(cfg.get("judge_temperature", 0.0))
        self.retrieve_model: str = str(cfg.get("retrieve_model", self.judge_model))
        self.retrieve_temperature: float = float(cfg.get("retrieve_temperature", 0.0))

        # Lazily initialised LLM client (only if eval_mode == "llm_judge")
        self._llm_client: Any | None = None
        self._llm_config: dict[str, Any] = dict(cfg.get("openrouter", {}))

        # Tool API keys — fall back to env vars (mirrors official repo)
        self.tavily_api_key: str = str(
            cfg.get("tavily_api_key") or os.environ.get("TAVILY_API_KEY", "")
        )
        self.sec_api_key: str = str(
            cfg.get("sec_api_key") or os.environ.get("SEC_EDGAR_API_KEY", "")
        )
        # Tool options
        self.max_tool_iterations: int = int(cfg.get("max_tool_iterations", 8))
        self.web_search_top_n: int = int(cfg.get("web_search_top_n", 10))
        # In-session HTML store shared across tool calls within one run
        self._html_store: dict[str, str] = {}

    # ------------------------------------------------------------------
    # BenchmarkAdapter interface
    # ------------------------------------------------------------------

    def load_tasks(self, task_limit: int | None = None) -> Sequence[BenchmarkTask]:
        csv_path = self._resolve_csv_path()
        tasks: list[BenchmarkTask] = []

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                task_id = str(idx)
                rubric = self._parse_rubric(row.get("Rubric", ""))
                metadata = {
                    "question_type": row.get("Question Type", ""),
                    "expert_time_mins": self._safe_float(row.get("Expert time (mins)", "")),
                    "rubric": rubric,
                    "source": "finance_agent_public_csv",
                }
                tasks.append(
                    BenchmarkTask(
                        task_id=task_id,
                        prompt=(row.get("Question") or "").strip(),
                        reference_answer=(row.get("Answer") or "").strip(),
                        metadata=metadata,
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
    ) -> MASRunResult:
        """Run with the four official finance-agent tools wired in.

        Tools follow the MAS tool-handler format::

            {"name": str, "description": str, "parameters": {...}, "handler": callable}

        Tool calls are recorded in trace_events (tool_call + tool_result events)
        by the LangGraph engine automatically.
        """
        # Reset per-run state
        self._html_store = {}
        tools = self._build_tools()
        prompt = _OFFICIAL_INSTRUCTIONS_PROMPT.format(question=task.prompt)
        wrapped_task = BenchmarkTask(
            task_id=task.task_id,
            prompt=[{"role": "user", "content": prompt}],
            reference_answer=task.reference_answer,
            metadata=dict(task.metadata),
        )
        return runner.run_task(
            task=wrapped_task,
            run_index=run_index,
            seed=seed,
            tools=tools,
            max_tool_iterations=self.max_tool_iterations,
            benchmark_name="finance_agent",
        )

    def evaluate(
        self,
        task: BenchmarkTask,
        prediction: str,
        *,
        run_metadata: dict[str, Any] | None = None,
    ) -> BenchmarkEvaluation:
        run_metadata = run_metadata or {}
        rubric = list(task.metadata.get("rubric", []))

        if self.eval_mode == "llm_judge":
            return self._evaluate_llm_judge(task, prediction, rubric, run_metadata)
        return self._evaluate_substring(task, prediction, rubric, run_metadata)

    def requirements(self) -> dict[str, Any]:
        notes = [
            "Data: public CSV from vals-ai/finance-agent (50 questions).",
            f"Eval mode: {self.eval_mode}.",
        ]
        if self.eval_mode == "llm_judge":
            notes.append(
                f"LLM judge: {self.judge_model}, mode-of-{self.judge_repeats}. "
                "Requires 'openai' package and a valid API key / OpenRouter key."
            )
        else:
            notes.append(
                "Substring mode: fast but lower accuracy. "
                "Set eval_mode='llm_judge' for higher-fidelity evaluation."
            )
        return {
            "benchmark": "finance_agent",
            "version": "1.1",
            "dataset_source": self.dataset_url,
            "question_types": [
                "Simple retrieval - Quantitative",
                "Simple retrieval - Qualitative",
                "Numerical Reasoning",
                "Complex Retrieval",
                "Adjustments",
                "Beat or Miss",
                "Trends",
                "Financial Modeling",
                "Market Analysis",
            ],
            "notes": notes,
        }

    # ------------------------------------------------------------------
    # Tool builders — mirrors official vals-ai/finance-agent tools.py
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[dict[str, Any]]:
        """Return MAS-compatible tool dicts for the four official tools."""
        tools: list[dict[str, Any]] = []

        # 1. web_search (Tavily)
        tavily_api_key = self.tavily_api_key
        top_n = self.web_search_top_n

        async def web_search(args: dict[str, Any]) -> Any:
            query = str(args.get("search_query", "")).strip()
            start_date = str(args.get("start_date", "")).strip()
            end_date = self._clamp_end_date(str(args.get("end_date", "")).strip())
            if not end_date:
                end_date = MAX_END_DATE
            number_of_results = int(args.get("number_of_results", top_n))
            if not query:
                return {"success": False, "result": "search_query is required"}
            if start_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", start_date):
                return {
                    "success": False,
                    "result": f"Invalid start_date format: '{start_date}'. Expected YYYY-MM-DD.",
                }
            if end_date and not re.match(r"^\d{4}-\d{2}-\d{2}$", end_date):
                return {
                    "success": False,
                    "result": f"Invalid end_date format: '{end_date}'. Expected YYYY-MM-DD.",
                }
            if start_date and start_date > end_date:
                return {
                    "success": False,
                    "result": (
                        f"Parameter start_date '{start_date}' was set to a date that is later than "
                        f"end_date '{end_date}'"
                    ),
                }
            if not tavily_api_key:
                return {
                    "success": False,
                    "result": "TAVILY_API_KEY not configured. Set tavily_api_key in config or TAVILY_API_KEY env var.",
                }
            try:
                payload: dict[str, Any] = {
                    "api_key": tavily_api_key,
                    "query": query,
                    "search_depth": "fast",
                    "max_results": max(1, min(20, number_of_results)),
                    "chunks_per_source": 1,
                    "end_date": end_date,
                }
                if start_date:
                    payload["start_date"] = start_date

                import aiohttp

                for attempt in range(8):
                    try:
                        async with (
                            aiohttp.ClientSession() as session,
                            session.post(
                                "https://api.tavily.com/search",
                                json=payload,
                                timeout=20,
                            ) as response,
                        ):
                            if response.status == 429:
                                raise aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=429,
                                    message="429",
                                    headers=response.headers,
                                )
                            response.raise_for_status()
                            data = await response.json()
                            results = data.get("results", [])
                            return {"success": True, "result": json.dumps(results)}
                    except aiohttp.ClientResponseError as exc:
                        if exc.status == 429 and attempt < 7:
                            time.sleep(min(20.0, (3 * (2**attempt)) + random.uniform(0.0, 1.0)))
                            continue
                        raise
                return {"success": False, "result": "Max retries reached for Tavily API"}
            except Exception as exc:
                return {"success": False, "result": str(exc)}

        tools.append(
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The query to search for",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "(optional) The start date for the search range in the format YYYY-MM-DD",
                        },
                        "end_date": {
                            "type": "string",
                            "description": f"(optional) The end date for the search range in the format YYYY-MM-DD. If later than {MAX_END_DATE}, it will be clamped.",
                        },
                        "number_of_results": {
                            "type": "integer",
                            "description": "(optional) Number of results to return (1-20).",
                        },
                    },
                    "required": ["search_query"],
                },
                "handler": web_search,
            }
        )

        # 2. edgar_search
        sec_key = self.sec_api_key

        async def edgar_search(args: dict[str, Any]) -> Any:
            if not sec_key:
                return {
                    "success": False,
                    "result": "SEC_EDGAR_API_KEY not configured. Set sec_api_key in config or SEC_EDGAR_API_KEY env var.",
                }
            try:
                search_query = str(args.get("search_query", "")).strip()
                if not search_query:
                    return {"success": False, "result": "search_query is required"}
                form_types = args.get("form_types")
                if form_types is not None and not isinstance(form_types, list):
                    return {
                        "success": False,
                        "result": f"The parameter form_types must be a list if provided. Was of type {type(form_types)}",
                    }
                ciks = args.get("ciks")
                if ciks is not None and not isinstance(ciks, list):
                    return {
                        "success": False,
                        "result": f"The parameter ciks must be a list if provided. Was of type {type(ciks)}",
                    }
                start_date = str(args.get("start_date", "1900-01-01")).strip() or "1900-01-01"
                end_date = str(args.get("end_date", MAX_END_DATE)).strip() or MAX_END_DATE
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", start_date):
                    return {
                        "success": False,
                        "result": f"start_date '{start_date}' is not in yyyy-mm-dd format",
                    }
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", end_date):
                    return {
                        "success": False,
                        "result": f"end_date '{end_date}' is not in yyyy-mm-dd format",
                    }
                if start_date > MAX_END_DATE:
                    start_date = MAX_END_DATE
                if end_date > MAX_END_DATE:
                    end_date = MAX_END_DATE
                if start_date > end_date:
                    return {
                        "success": False,
                        "result": (
                            f"Parameter start_date '{start_date}' was set to a date that is later than "
                            f"end_date '{end_date}'"
                        ),
                    }
                payload = {
                    "query": search_query,
                    "startDate": start_date,
                    "endDate": end_date,
                    "page": int(args.get("page", 1) or 1),
                }
                if form_types:
                    payload["formTypes"] = form_types
                if ciks:
                    payload["ciks"] = ciks
                top_n_results = int(args.get("top_n_results", 100))

                import aiohttp

                for attempt in range(8):
                    try:
                        async with (
                            aiohttp.ClientSession() as session,
                            session.post(
                                "https://api.sec-api.io/full-text-search",
                                json=payload,
                                headers={
                                    "Authorization": sec_key,
                                    "Content-Type": "application/json",
                                },
                                timeout=20,
                            ) as response,
                        ):
                            if response.status == 429:
                                raise aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=429,
                                    message="429",
                                    headers=response.headers,
                                )
                            if response.status == 503:
                                raise aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=503,
                                    message="503",
                                    headers=response.headers,
                                )
                            response.raise_for_status()
                            result = await response.json()
                            filings = result.get("filings", [])[:top_n_results]
                            return {"success": True, "result": json.dumps(filings)}
                    except aiohttp.ClientResponseError as exc:
                        if exc.status in {429, 503} and attempt < 7:
                            time.sleep(min(20.0, (3 * (2**attempt)) + random.uniform(0.0, 1.0)))
                            continue
                        raise
                return {"success": False, "result": "Max retries reached for SEC API"}
            except Exception as exc:
                return {"success": False, "result": str(exc)}

        tools.append(
            {
                "name": "edgar_search",
                "description": (
                    "Search the EDGAR Database through the SEC API. "
                    "You should provide a search_query. You can optionally provide form_types, ciks, start_date, end_date, page, and top_n_results. "
                    "The results are returned as a list of dictionaries, each containing the metadata for a filing. It does not contain the full text of the filing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The keyword or phrase to search, such as 'substantial doubt' OR 'material weakness'",
                        },
                        "form_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Limits search to specific SEC form types (e.g., ['8-K', '10-Q']) list of strings. Default is None (all form types)",
                        },
                        "ciks": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filters results to filings by specified CIKs, type list of strings. Default is None (all filers).",
                        },
                        "start_date": {
                            "type": "string",
                            "description": "Start date for the search range in yyyy-mm-dd format. Used with endDate to define the date range. Example: '2024-01-01'. Default is 30 days ago",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date for the search range, in the same format as startDate. Default is today",
                        },
                        "page": {
                            "type": "string",
                            "description": "Pagination for results. Default is '1'",
                        },
                        "top_n_results": {
                            "type": "integer",
                            "description": "The top N results to return after the query. Useful if you are not sure the result you are loooking for is ranked first after your query.",
                        },
                    },
                    "required": ["search_query"],
                },
                "handler": edgar_search,
            }
        )

        # 3. parse_html_page
        html_store = self._html_store

        async def parse_html_page(args: dict[str, Any]) -> Any:
            url = str(args.get("url", "")).strip()
            key = str(args.get("key", "")).strip()
            if not url or not key:
                return {"success": False, "result": "url and key are required"}
            try:
                import aiohttp
                from bs4 import BeautifulSoup

                # Use exact same UA as official repo which SEC typically allows
                headers = {"User-Agent": "ValsAI/antoine@vals.ai"}
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(url, headers=headers, timeout=60) as response,
                ):
                    response.raise_for_status()
                    html = await response.text()

                soup = BeautifulSoup(html, "html.parser")
                # Remove script and style elements
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()

                # Get text
                text = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)
                existed = key in html_store
                html_store[key] = text
                lines_out: list[str] = []
                if existed:
                    lines_out.append(
                        "WARNING: The key already exists in the data storage. The new result overwrites the old one."
                    )
                lines_out.append(
                    f"SUCCESS: The result has been saved to the data storage under the key: {key}."
                )
                keys_list = "\n".join(html_store.keys())
                lines_out.append("The data_storage currently contains the following keys:")
                lines_out.append(keys_list)
                return {"success": True, "result": "\n".join(lines_out)}
            except Exception as exc:
                return {"success": False, "result": str(exc)}

        tools.append(
            {
                "name": "parse_html_page",
                "description": (
                    "Parse an HTML page. This tool is used to parse the HTML content of a page and saves the content outside of the conversation to avoid context window issues. "
                    "You should provide both the URL of the page to parse, as well as the key you want to use to save the result in the agent's data structure. "
                    "The data structure is a dictionary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the HTML page to parse",
                        },
                        "key": {
                            "type": "string",
                            "description": "The key to use when saving the result in the conversation's data structure (dict).",
                        },
                    },
                    "required": ["url", "key"],
                },
                "handler": parse_html_page,
            }
        )

        # 4. retrieve_information
        async def retrieve_information(args: dict[str, Any]) -> Any:
            prompt = str(args.get("prompt", ""))
            if not re.search(r"{{[^{}]+}}", prompt):
                return {
                    "success": False,
                    "result": (
                        "ERROR: Your prompt must include at least one key from data storage "
                        "in the format {{key_name}}. Please try again with the correct format."
                    ),
                }
            input_character_ranges = args.get("input_character_ranges", []) or []
            if not isinstance(input_character_ranges, list):
                return {
                    "success": False,
                    "result": (
                        "ERROR: input_character_ranges must be a list of objects with key/start/end."
                    ),
                }
            ranges_dict: dict[str, tuple[int, int]] = {}
            for range_spec in input_character_ranges:
                if not isinstance(range_spec, dict):
                    return {
                        "success": False,
                        "result": (
                            "ERROR: Each item in input_character_ranges must be an object with "
                            "'key', 'start', and 'end' fields."
                        ),
                    }
                if not {"key", "start", "end"} <= set(range_spec.keys()):
                    return {
                        "success": False,
                        "result": (
                            "ERROR: Each range specification must have 'key', 'start', and 'end' fields."
                        ),
                    }
                ranges_dict[str(range_spec["key"])] = (
                    int(range_spec["start"]),
                    int(range_spec["end"]),
                )
            keys = re.findall(r"{{([^{}]+)}}", prompt)
            keys_set = set(keys)
            for range_key in ranges_dict:
                if range_key not in keys_set:
                    return {
                        "success": False,
                        "result": (
                            f"ERROR: The key '{range_key}' is specified in input_character_ranges but is not referenced in the prompt. "
                            f"Keys in prompt: {', '.join(keys_set) if keys_set else '(none)'}"
                        ),
                    }

            formatted_data: dict[str, str] = {}
            for key in keys:
                if key not in html_store:
                    return {
                        "success": False,
                        "result": (
                            f"ERROR: The key '{key}' was not found in the data storage. "
                            f"Available keys are: {', '.join(html_store.keys())}"
                        ),
                    }
                doc_content = html_store[key]
                if key in ranges_dict:
                    start_idx, end_idx = ranges_dict[key]
                    formatted_data[key] = doc_content[start_idx:end_idx]
                else:
                    formatted_data[key] = doc_content

            formatted_prompt = re.sub(r"{{([^{}]+)}}", r"{\1}", prompt)
            try:
                model_prompt = formatted_prompt.format(**formatted_data)
            except KeyError as exc:
                return {
                    "success": False,
                    "result": (
                        f"ERROR: The key {str(exc)} was not found in the data storage. "
                        f"Available keys are: {', '.join(html_store.keys())}"
                    ),
                }

            try:
                client = self._get_llm_client()
                response = client.chat.completions.create(
                    model=self.retrieve_model,
                    temperature=self.retrieve_temperature,
                    messages=[{"role": "user", "content": model_prompt}],
                )
                content = response.choices[0].message.content or ""
                usage_payload = self._extract_usage_payload(getattr(response, "usage", None))
                return {"success": True, "result": content, "usage": usage_payload}
            except Exception as exc:
                return {"success": False, "result": str(exc)}

        tools.append(
            {
                "name": "retrieve_information",
                "description": """Retrieve information from the conversation's data structure (dict) and allow character range extraction.

IMPORTANT: Your prompt MUST include at least one key from the data storage using the exact format: {{key_name}}

For example, if you want to analyze data stored under the key "financial_report", your prompt should look like:
"Analyze the following financial report and extract the revenue figures: {{financial_report}}"

The {{key_name}} will be replaced with the actual content stored under that key before being sent to the LLM.
If you don't use this exact format with double braces, the tool will fail to retrieve the information.

You can optionally specify character ranges for each document key to extract only portions of documents. That can be useful to avoid token limit errors or improve efficiency by selecting only part of the document.
For example, if "financial_report" contains "Annual Report 2023" and you specify a range [1, 5] for that key,
only "nnual" will be inserted into the prompt.

The output is the result from the LLM that receives the prompt with the inserted data.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": (
                                "The prompt that will be passed to the LLM. You MUST include at least one data storage key in the format {{key_name}} "
                                "- for example: 'Summarize this 10-K filing: {{company_10k}}'. The content stored under each key will replace the {{key_name}} placeholder."
                            ),
                        },
                        "input_character_ranges": {
                            "type": "object",
                            "description": (
                                "A dictionary mapping document keys to their character ranges. Each range should be an array where the first element is the start index "
                                "and the second element is the end index. Can be used to only read portions of documents. By default, the full document is used. "
                                "To use the full document, set the range to an empty list []."
                            ),
                            "additionalProperties": {"type": "array", "items": {"type": "integer"}},
                        },
                    },
                    "required": ["prompt"],
                },
                "handler": retrieve_information,
            }
        )

        return tools

    # ------------------------------------------------------------------
    # Evaluation: LLM-as-Judge (mirrors Vals methodology)
    # ------------------------------------------------------------------

    def _evaluate_llm_judge(
        self,
        task: BenchmarkTask,
        prediction: str,
        rubric: list[dict[str, Any]],
        run_metadata: dict[str, Any],
    ) -> BenchmarkEvaluation:
        correctness_total = 0
        contradiction_total = 0
        correctness_hits = 0
        contradiction_hits = 0

        criterion_results: list[dict[str, Any]] = []

        for criterion in rubric:
            operator = str(criterion.get("operator", "")).strip().lower()
            text = str(criterion.get("criteria", "")).strip()
            if not operator or not text:
                continue

            matched = self._judge_criterion(
                prediction=prediction,
                reference_answer=task.reference_answer,
                operator=operator,
                criteria=text,
            )

            if operator == "correctness":
                correctness_total += 1
                correctness_hits += int(matched)
            elif operator == "contradiction":
                contradiction_total += 1
                contradiction_hits += int(matched)

            criterion_results.append(
                {
                    "operator": operator,
                    "criteria": text,
                    "matched": matched,
                    "eval_mode": "llm_judge",
                }
            )

        correctness_ratio = correctness_hits / correctness_total if correctness_total > 0 else 0.0
        contradiction_ratio = (
            contradiction_hits / contradiction_total if contradiction_total > 0 else 0.0
        )

        score = max(0.0, min(1.0, correctness_ratio - contradiction_ratio))
        success = score >= self.success_threshold

        details: dict[str, Any] = {
            "eval_mode": "llm_judge",
            "judge_model": self.judge_model,
            "judge_repeats": self.judge_repeats,
            "correctness_hits": correctness_hits,
            "correctness_total": correctness_total,
            "contradiction_hits": contradiction_hits,
            "contradiction_total": contradiction_total,
            "correctness_ratio": correctness_ratio,
            "contradiction_ratio": contradiction_ratio,
            "success_threshold": self.success_threshold,
            "criterion_results": criterion_results,
            "prediction": prediction,
            "reference_answer": task.reference_answer,
            "question_type": task.metadata.get("question_type", ""),
            "run_metadata": run_metadata,
        }

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=score,
            success=success,
            details=details,
        )

    def _judge_criterion(
        self,
        *,
        prediction: str,
        reference_answer: str,
        operator: str,
        criteria: str,
    ) -> bool:
        """Call the LLM judge `judge_repeats` times and return mode result."""
        client = self._get_llm_client()

        user_msg = _JUDGE_USER_TEMPLATE.format(
            reference_answer=reference_answer,
            prediction=prediction,
            operator=operator,
            criteria=criteria,
        )

        votes: list[bool] = []
        for _ in range(self.judge_repeats):
            try:
                response = client.chat.completions.create(
                    model=self.judge_model,
                    temperature=self.judge_temperature,
                    messages=[
                        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                judgment = json.loads(content).get("judgment", "no")
                votes.append(judgment.strip().lower() == "yes")
            except Exception:
                logger.warning(
                    "LLM judge call failed for criterion: %s",
                    criteria[:80],
                    exc_info=True,
                )
                votes.append(False)

        # Mode-of-N: majority vote (matches Vals methodology)
        try:
            return mode(votes)
        except Exception:
            # If no clear mode (shouldn't happen with odd N), default False
            return sum(votes) > len(votes) / 2

    def _get_llm_client(self) -> Any:
        """Lazily create an OpenAI-compatible client."""
        if self._llm_client is not None:
            return self._llm_client

        import openai  # type: ignore

        kwargs: dict[str, Any] = {}

        # Support OpenRouter if configured
        base_url = self._llm_config.get("base_url")
        api_key = self._llm_config.get("api_key")

        if base_url:
            kwargs["base_url"] = str(base_url)
        if api_key:
            kwargs["api_key"] = str(api_key)

        self._llm_client = openai.OpenAI(**kwargs)
        return self._llm_client

    # ------------------------------------------------------------------
    # Evaluation: Substring fallback (fast, no API cost)
    # ------------------------------------------------------------------

    def _evaluate_substring(
        self,
        task: BenchmarkTask,
        prediction: str,
        rubric: list[dict[str, Any]],
        run_metadata: dict[str, Any],
    ) -> BenchmarkEvaluation:
        pred_norm = self._normalize_text(prediction)

        correctness_total = 0
        contradiction_total = 0
        correctness_hits = 0
        contradiction_hits = 0

        criterion_results: list[dict[str, Any]] = []
        for criterion in rubric:
            operator = str(criterion.get("operator", "")).strip().lower()
            text = str(criterion.get("criteria", "")).strip()
            if not operator or not text:
                continue

            matched = self._substring_match(pred_norm, text)

            if operator == "correctness":
                correctness_total += 1
                correctness_hits += int(matched)
            elif operator == "contradiction":
                contradiction_total += 1
                contradiction_hits += int(matched)

            criterion_results.append(
                {
                    "operator": operator,
                    "criteria": text,
                    "matched": matched,
                    "eval_mode": "substring",
                }
            )

        correctness_ratio = correctness_hits / correctness_total if correctness_total > 0 else 0.0
        contradiction_ratio = (
            contradiction_hits / contradiction_total if contradiction_total > 0 else 0.0
        )

        score = max(0.0, min(1.0, correctness_ratio - contradiction_ratio))
        success = score >= self.success_threshold

        details: dict[str, Any] = {
            "eval_mode": "substring",
            "correctness_hits": correctness_hits,
            "correctness_total": correctness_total,
            "contradiction_hits": contradiction_hits,
            "contradiction_total": contradiction_total,
            "correctness_ratio": correctness_ratio,
            "contradiction_ratio": contradiction_ratio,
            "success_threshold": self.success_threshold,
            "criterion_results": criterion_results,
            "prediction": prediction,
            "reference_answer": task.reference_answer,
            "question_type": task.metadata.get("question_type", ""),
            "run_metadata": run_metadata,
        }

        return BenchmarkEvaluation(
            task_id=task.task_id,
            score=score,
            success=success,
            details=details,
        )

    # ------------------------------------------------------------------
    # Matching helpers
    # ------------------------------------------------------------------

    @classmethod
    def _substring_match(cls, pred_norm: str, criteria_text: str) -> bool:
        """Check if normalised criteria appears in normalised prediction.

        Handles common financial formatting variants:
          - "$3.25 Billion" ↔ "$3.25B" ↔ "3.25 billion"
          - "22.6%" ↔ "22.60%"
          - "2,865,507" ↔ "2865507"
        """
        criteria_norm = cls._normalize_text(criteria_text)

        # Direct substring
        if criteria_norm and criteria_norm in pred_norm:
            return True

        # Try numeric extraction: if criteria is a single number, check
        # if it appears in prediction within 2% tolerance.
        criteria_numbers = cls._extract_numbers(criteria_text)
        if criteria_numbers:
            pred_numbers = cls._extract_numbers(pred_norm)
            for ref_num in criteria_numbers:
                for pred_num in pred_numbers:
                    if cls._is_close(pred_num, ref_num, rel_tol=0.02):
                        return True

        return False

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        """Extract numbers from text, handling commas and common suffixes."""
        # Remove commas in numbers (e.g. "2,865,507" -> "2865507")
        cleaned = re.sub(r"(\d),(\d)", r"\1\2", text)
        numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        out: list[float] = []
        for value in numbers:
            try:
                out.append(float(value))
            except ValueError:
                continue
        return out

    @staticmethod
    def _is_close(a: float, b: float, *, rel_tol: float = 0.02) -> bool:
        """Check if two numbers are within relative tolerance."""
        if b == 0:
            return abs(a) < 1e-9
        return abs(a - b) / abs(b) <= rel_tol

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _resolve_csv_path(self) -> Path:
        if self.local_csv_path:
            local = Path(str(self.local_csv_path)).expanduser().resolve()
            if not local.exists():
                raise FileNotFoundError(f"FinanceAgent CSV not found: {local}")
            return local

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self.cache_dir / "public.csv"
        if cached.exists():
            return cached

        logger.info("Downloading FinanceAgent CSV from %s", self.dataset_url)
        with urllib.request.urlopen(self.dataset_url, timeout=30) as response:
            data = response.read()
        cached.write_bytes(data)
        return cached

    @staticmethod
    def _clamp_end_date(end_date: str) -> str:
        if end_date and end_date > MAX_END_DATE:
            return MAX_END_DATE
        return end_date

    @staticmethod
    def _extract_usage_payload(usage: Any) -> dict[str, Any]:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": {"total": 0.0},
        }

    @staticmethod
    def _parse_list_arg(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text.replace("'", '"'))
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except json.JSONDecodeError:
                    items = [item.strip(" \"'") for item in text[1:-1].split(",")]
                    return [item for item in items if item]
        return [str(value)]

    @staticmethod
    def _parse_rubric(value: str) -> list[dict[str, Any]]:
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                out: list[dict[str, Any]] = []
                for item in parsed:
                    if isinstance(item, dict):
                        out.append(dict(item))
                return out
        except (SyntaxError, ValueError):
            return []
        return []

    @staticmethod
    def _normalize_text(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text.strip().lower())
        return collapsed

    @staticmethod
    def _safe_float(value: str) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

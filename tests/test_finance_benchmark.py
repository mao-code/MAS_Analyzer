import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from benchmark.finance_agent import FinanceAgentBenchmark


class _AiohttpResponse:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        status: int = 200,
        json_started: asyncio.Event | None = None,
        json_release: asyncio.Event | None = None,
    ) -> None:
        self.status = status
        self.request_info = None
        self.history: tuple[object, ...] = ()
        self.headers: dict[str, str] = {}
        self._payload = payload
        self._json_started = json_started
        self._json_release = json_release

    async def __aenter__(self) -> "_AiohttpResponse":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict[str, object]:
        if self._json_started is not None:
            self._json_started.set()
        if self._json_release is not None:
            await self._json_release.wait()
        return self._payload


class _AiohttpSessionFactory:
    def __init__(self, responses: list[_AiohttpResponse]) -> None:
        self.responses = responses
        self.posts: list[dict[str, object]] = []

    def __call__(self) -> "_AiohttpSession":
        return _AiohttpSession(self)


class _AiohttpSession:
    def __init__(self, factory: _AiohttpSessionFactory) -> None:
        self._factory = factory

    async def __aenter__(self) -> "_AiohttpSession":
        return self

    async def __aexit__(self, *_args: object) -> bool:
        return False

    def post(self, url: str, **kwargs: object) -> _AiohttpResponse:
        self._factory.posts.append({"url": url, **kwargs})
        return self._factory.responses.pop(0)


class TestFinanceBenchmark(unittest.TestCase):
    def test_csv_load_and_rubric_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "public.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Question",
                        "Answer",
                        "Question Type",
                        "Expert time (mins)",
                        "Rubric",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Question": "What is X?",
                        "Answer": "X is 42",
                        "Question Type": "Quantitative Retrieval",
                        "Expert time (mins)": "2",
                        "Rubric": (
                            '[{"operator": "correctness", "criteria": "42"}, '
                            '{"operator": "contradiction", "criteria": "13"}]'
                        ),
                    }
                )

            benchmark = FinanceAgentBenchmark(
                {
                    "local_csv_path": str(path),
                    "success_threshold": 0.5,
                    "eval_mode": "substring",
                }
            )
            tasks = benchmark.load_tasks()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].prompt, "What is X?")

            good_eval = benchmark.evaluate(tasks[0], "The answer is 42")
            self.assertGreaterEqual(good_eval.score, 0.5)
            self.assertTrue(good_eval.success)

            bad_eval = benchmark.evaluate(tasks[0], "The answer is 13")
            self.assertLess(bad_eval.score, 0.5)
            self.assertFalse(bad_eval.success)

    def test_edgar_search_reuses_successful_identical_requests(self) -> None:
        benchmark = FinanceAgentBenchmark({"sec_api_key": "sec-token"})
        handler = self._edgar_handler(benchmark)
        args = self._edgar_args()
        sessions = _AiohttpSessionFactory(
            [
                _AiohttpResponse(
                    {"filings": [{"accessionNo": "one"}, {"accessionNo": "two"}]}
                )
            ]
        )

        async def run() -> tuple[dict[str, object], dict[str, object]]:
            with patch("aiohttp.ClientSession", new=sessions):
                first = await handler(args)
                second = await handler(dict(args))
            return first, second

        first, second = asyncio.run(run())

        expected = {
            "success": True,
            "result": json.dumps([{"accessionNo": "one"}]),
        }
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)
        self.assertIsNot(first, second)
        self.assertEqual(len(sessions.posts), 1)

    def test_edgar_search_deduplicates_concurrent_identical_requests(self) -> None:
        benchmark = FinanceAgentBenchmark({"sec_api_key": "sec-token"})
        handler = self._edgar_handler(benchmark)
        args = self._edgar_args()

        async def run() -> tuple[list[dict[str, object]], _AiohttpSessionFactory]:
            json_started = asyncio.Event()
            json_release = asyncio.Event()
            sessions = _AiohttpSessionFactory(
                [
                    _AiohttpResponse(
                        {"filings": [{"accessionNo": "one"}]},
                        json_started=json_started,
                        json_release=json_release,
                    )
                ]
            )
            with patch("aiohttp.ClientSession", new=sessions):
                first_task = asyncio.create_task(handler(args))
                await json_started.wait()
                second_task = asyncio.create_task(handler(dict(args)))
                await asyncio.sleep(0)
                json_release.set()
                results = await asyncio.gather(first_task, second_task)
            return results, sessions

        results, sessions = asyncio.run(run())

        self.assertEqual(results[0], results[1])
        self.assertEqual(len(sessions.posts), 1)

    def test_edgar_search_uses_async_sleep_for_429_retry(self) -> None:
        benchmark = FinanceAgentBenchmark({"sec_api_key": "sec-token"})
        handler = self._edgar_handler(benchmark)
        sessions = _AiohttpSessionFactory(
            [
                _AiohttpResponse({}, status=429),
                _AiohttpResponse({"filings": [{"accessionNo": "one"}]}),
            ]
        )
        sleep = AsyncMock(return_value=None)

        async def run() -> dict[str, object]:
            with (
                patch("aiohttp.ClientSession", new=sessions),
                patch("benchmark.finance_agent.asyncio.sleep", new=sleep),
                patch("benchmark.finance_agent.random.uniform", return_value=0.0),
            ):
                return await handler(self._edgar_args())

        result = asyncio.run(run())

        self.assertTrue(result["success"])
        self.assertEqual(len(sessions.posts), 2)
        sleep.assert_awaited_once_with(3.0)

    @staticmethod
    def _edgar_handler(benchmark: FinanceAgentBenchmark):
        tools = benchmark._build_tools()
        return next(tool["handler"] for tool in tools if tool["name"] == "edgar_search")

    @staticmethod
    def _edgar_args() -> dict[str, object]:
        return {
            "query": "material weakness",
            "form_types": ["10-K"],
            "ciks": ["0000320193"],
            "start_date": "2024-01-01",
            "end_date": "2025-04-07",
            "page": "1",
            "top_n_results": 1,
        }


if __name__ == "__main__":
    unittest.main()

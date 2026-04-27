import unittest
import time
import json
from types import SimpleNamespace
from unittest.mock import patch

from MAS.config import OpenRouterConfig
from MAS.llm import OpenRouterLLMClient


def _make_completion(
    *,
    content: str = "",
    tool_calls: list | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" in kwargs:
            tool_call = SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="search_tool", arguments='{"query":"x"}'),
            )
            return _make_completion(tool_calls=[tool_call], prompt_tokens=10, completion_tokens=2)
        return _make_completion(content="FINAL ANSWER: done", prompt_tokens=5, completion_tokens=4)


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


class _SlowCompletions:
    def create(self, **kwargs):
        time.sleep(0.2)
        return _make_completion(content="late answer", prompt_tokens=3, completion_tokens=2)


class _SlowClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_SlowCompletions())


class _CompactingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tool_call_count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return _make_completion(content="FINAL ANSWER: compacted", prompt_tokens=6, completion_tokens=4)
        self.tool_call_count += 1
        tool_call = SimpleNamespace(
            id=f"call_{self.tool_call_count}",
            type="function",
            function=SimpleNamespace(
                name="search_tool",
                arguments=json.dumps({"query": f"q{self.tool_call_count}"}),
            ),
        )
        return _make_completion(tool_calls=[tool_call], prompt_tokens=20, completion_tokens=2)


class _CompactingClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_CompactingCompletions())


class _StagnatingCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tool_call_count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return _make_completion(
                content="FINAL ANSWER: insufficient evidence",
                prompt_tokens=7,
                completion_tokens=5,
            )
        self.tool_call_count += 1
        tool_call = SimpleNamespace(
            id=f"call_{self.tool_call_count}",
            type="function",
            function=SimpleNamespace(
                name="search_tool",
                arguments=json.dumps({"query": f"q{self.tool_call_count}"}),
            ),
        )
        return _make_completion(tool_calls=[tool_call], prompt_tokens=18, completion_tokens=2)


class _StagnatingClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_StagnatingCompletions())


class _ForceFinalTimeoutCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            time.sleep(0.2)
            return _make_completion(content="late answer", prompt_tokens=5, completion_tokens=4)
        tool_call = SimpleNamespace(
            id="call_timeout",
            type="function",
            function=SimpleNamespace(name="search_tool", arguments='{"query":"q1"}'),
        )
        return _make_completion(tool_calls=[tool_call], prompt_tokens=12, completion_tokens=2)


class _ForceFinalTimeoutClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_ForceFinalTimeoutCompletions())


class _InitialToolTimeoutCompletions:
    def create(self, **kwargs):
        time.sleep(0.2)
        return _make_completion(content="late tool answer", prompt_tokens=3, completion_tokens=2)


class _InitialToolTimeoutClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_InitialToolTimeoutCompletions())


class _MidToolTimeoutCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.tool_call_count = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if "tools" not in kwargs:
            return _make_completion(content="FINAL ANSWER: should not be used", prompt_tokens=5, completion_tokens=4)
        self.tool_call_count += 1
        if self.tool_call_count == 1:
            tool_call = SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="search_tool", arguments='{"query":"q1"}'),
            )
            return _make_completion(tool_calls=[tool_call], prompt_tokens=12, completion_tokens=2)
        time.sleep(0.2)
        return _make_completion(content="late tool answer", prompt_tokens=3, completion_tokens=2)


class _MidToolTimeoutClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_MidToolTimeoutCompletions())


class TestLLMToolLoop(unittest.TestCase):
    def test_disable_live_override_forces_mock_even_with_api_key(self) -> None:
        with patch.dict("os.environ", {"MAS_DISABLE_LIVE_LLM": "1"}, clear=False):
            client = OpenRouterLLMClient(
                OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
            )

        self.assertIsNone(client.client)

        result = client.generate(
            prompt="solve this",
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
        )

        self.assertTrue(result.mock_used)

    def test_forces_final_answer_after_tool_limit(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _FakeClient()

        result = client.generate(
            prompt=[{"role": "user", "content": "solve this"}],
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "search.tool",
                    "description": "Search tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": lambda args: {"ok": True},
                }
            ],
            max_tool_iterations=1,
        )

        self.assertFalse(result.mock_used)
        self.assertEqual(result.text, "FINAL ANSWER: done")
        self.assertTrue(result.metadata.get("tool_loop_forced_final_answer"))
        self.assertEqual(result.tool_calls[0]["tool_name"], "search.tool")

    def test_generate_falls_back_to_mock_on_hard_timeout(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _SlowClient()

        result = client.generate(
            prompt="solve this",
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
        )

        self.assertTrue(result.mock_used)
        self.assertIn("timeout", str(result.metadata.get("fallback_reason", "")).lower())

    def test_compacts_prompt_facing_tool_output_and_summarizes_older_turns(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _CompactingClient()

        raw_outputs = {
            "q1": "FIRST_FULL_PAYLOAD::" + ("A" * 1200) + "::END1",
            "q2": "SECOND_FULL_PAYLOAD::" + ("B" * 1200) + "::END2",
            "q3": "THIRD_FULL_PAYLOAD::" + ("C" * 1200) + "::END3",
        }

        with patch.dict(
            "os.environ",
            {
                "MAS_TOOL_CONTEXT_RAW_TURNS": "1",
                "MAS_TOOL_CONTEXT_PREVIEW_CHARS": "160",
                "MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS": "1800",
            },
            clear=False,
        ):
            result = client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search.tool",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: [
                            {
                                "docid": f"doc-{args['query']}",
                                "score": 42.123456,
                                "snippet": raw_outputs[args["query"]],
                            }
                        ],
                    }
                ],
                max_tool_iterations=3,
            )

        completions = client.client.chat.completions
        third_tool_call_messages = completions.calls[2]["messages"]
        joined_content = "\n".join(str(message.get("content", "")) for message in third_tool_call_messages)

        self.assertEqual(result.text, "FINAL ANSWER: compacted")
        self.assertTrue(result.metadata.get("tool_context_compacted"))
        self.assertEqual(result.metadata.get("tool_context_summarized_turns"), 2)
        self.assertIn("Tool-memory summary from earlier tool iterations.", joined_content)
        self.assertIn('"docid": "doc-q2"', joined_content)
        self.assertIn('"snippet_preview"', joined_content)
        self.assertNotIn(raw_outputs["q1"], joined_content)
        self.assertNotIn(raw_outputs["q2"], joined_content)
        self.assertEqual(len(result.tool_calls), 3)

    def test_bounds_older_tool_turn_summary_size(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _CompactingClient()

        with patch.dict(
            "os.environ",
            {
                "MAS_TOOL_CONTEXT_RAW_TURNS": "1",
                "MAS_TOOL_CONTEXT_PREVIEW_CHARS": "120",
                "MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS": "220",
            },
            clear=False,
        ):
            client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search.tool",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: {
                            "query": args["query"],
                            "payload": "PAYLOAD::" + ("X" * 800),
                        },
                    }
                ],
                max_tool_iterations=3,
            )

        force_final_messages = client.client.chat.completions.calls[3]["messages"]
        summary_messages = [
            str(message.get("content", ""))
            for message in force_final_messages
            if message.get("role") == "system"
            and "Tool-memory summary from earlier tool iterations." in str(message.get("content", ""))
        ]

        self.assertTrue(summary_messages)
        self.assertLessEqual(len(summary_messages[0]), 220)

    def test_recovers_when_force_final_times_out_after_tool_evidence(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _ForceFinalTimeoutClient()

        result = client.generate(
            prompt=[{"role": "user", "content": "solve this"}],
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "search.tool",
                    "description": "Search tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": lambda args: [
                        {
                            "docid": "59188",
                            "score": 50.0,
                            "snippet": f"evidence-for-{args['query']}",
                        }
                    ],
                }
            ],
            max_tool_iterations=1,
        )

        self.assertFalse(result.mock_used)
        self.assertTrue(result.metadata.get("tool_loop_timeout_recovered"))
        self.assertEqual(result.metadata.get("tool_loop_timeout_stage"), "force_final")
        self.assertEqual(
            result.metadata.get("tool_loop_recovery_mode"),
            "best_effort_evidence_fallback",
        )
        self.assertTrue(result.metadata.get("tool_loop_forced_final_answer"))
        self.assertIn("Best-effort conclusion", result.text)
        self.assertIn("Evidence snapshot:", result.text)

    def test_tool_timeout_without_evidence_still_falls_back_to_mock(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _InitialToolTimeoutClient()

        result = client.generate(
            prompt=[{"role": "user", "content": "solve this"}],
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "search.tool",
                    "description": "Search tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": lambda args: {"ok": True},
                }
            ],
            max_tool_iterations=1,
        )

        self.assertTrue(result.mock_used)
        self.assertIn("timeout", str(result.metadata.get("fallback_reason", "")).lower())

    def test_recovers_when_tool_iteration_times_out_after_evidence(self) -> None:
        with patch.dict("os.environ", {"MAS_REQUIRE_LIVE_LLM": "1"}, clear=False):
            client = OpenRouterLLMClient(
                OpenRouterConfig(api_key="test", timeout_s=0.05), {"default": "openai/gpt-4o-mini"}
            )
        client.client = _MidToolTimeoutClient()

        result = client.generate(
            prompt=[{"role": "user", "content": "solve this"}],
            agent_type="general",
            task_id="t1",
            run_index=0,
            agent_id="agent_0",
            tools=[
                {
                    "name": "search.tool",
                    "description": "Search tool",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                    "handler": lambda args: [
                        {
                            "docid": "59188",
                            "score": 50.0,
                            "snippet": f"evidence-for-{args['query']}",
                        }
                    ],
                }
            ],
            max_tool_iterations=3,
        )

        self.assertFalse(result.mock_used)
        self.assertTrue(result.metadata.get("tool_loop_timeout_recovered"))
        self.assertEqual(result.metadata.get("tool_loop_timeout_stage"), "tool_iteration")
        self.assertEqual(result.metadata.get("tool_loop_timeout_iteration"), 2)
        self.assertEqual(
            result.metadata.get("tool_loop_recovery_mode"),
            "best_effort_evidence_fallback",
        )
        self.assertIn("tool-loop iteration timed out", result.metadata.get("tool_loop_stopped_reason", "").lower())
        self.assertIn("Evidence snapshot:", result.text)
        self.assertIn("Best-effort conclusion", result.text)

    def test_stops_early_when_search_results_stagnate(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _StagnatingClient()

        with patch.dict("os.environ", {"MAS_SEARCH_STAGNATION_CONSECUTIVE": "1"}, clear=False):
            result = client.generate(
                prompt=[{"role": "user", "content": "solve this"}],
                agent_type="general",
                task_id="t1",
                run_index=0,
                agent_id="agent_0",
                tools=[
                    {
                        "name": "search.tool",
                        "description": "Search tool",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "handler": lambda args: [
                            {
                                "docid": "59188",
                                "score": 50.0,
                                "snippet": f"same-doc-for-{args['query']}",
                            }
                        ],
                    }
                ],
                max_tool_iterations=8,
            )

        completions = client.client.chat.completions
        force_final_messages = completions.calls[2]["messages"]
        joined_content = "\n".join(str(message.get("content", "")) for message in force_final_messages)

        self.assertEqual(result.text, "FINAL ANSWER: insufficient evidence")
        self.assertEqual(len(result.tool_calls), 2)
        self.assertIn("Search results stagnated", result.metadata.get("tool_loop_stopped_reason", ""))
        self.assertTrue(result.metadata.get("tool_loop_forced_final_answer"))
        self.assertIn("search results have stagnated", joined_content.lower())
        self.assertIn("insufficient-evidence", joined_content.lower())


if __name__ == "__main__":
    unittest.main()

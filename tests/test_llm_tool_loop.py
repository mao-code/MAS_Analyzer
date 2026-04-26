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

    def test_compacts_older_tool_turns_but_keeps_latest_raw_tool_output(self) -> None:
        client = OpenRouterLLMClient(
            OpenRouterConfig(api_key="test"), {"default": "openai/gpt-4o-mini"}
        )
        client.client = _CompactingClient()

        raw_outputs = {
            "q1": "FIRST_FULL_PAYLOAD::" + ("A" * 1200) + "::END1",
            "q2": "SECOND_FULL_PAYLOAD::" + ("B" * 1200) + "::END2",
            "q3": "THIRD_FULL_PAYLOAD::" + ("C" * 1200) + "::END3",
        }

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
                    "handler": lambda args: {"query": args["query"], "payload": raw_outputs[args["query"]]},
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
        self.assertIn(raw_outputs["q2"], joined_content)
        self.assertNotIn(raw_outputs["q1"], joined_content)
        self.assertEqual(len(result.tool_calls), 3)


if __name__ == "__main__":
    unittest.main()

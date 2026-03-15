import unittest
from types import SimpleNamespace

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


class TestLLMToolLoop(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

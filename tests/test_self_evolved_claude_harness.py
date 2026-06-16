import json
import os
import sys
import types
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from descriptor.schema import validate_trace_events
from MAS.artifacts import _extract_json_payload
from MAS.config import OpenRouterConfig, SelfEvolvedConfig
from MAS.langgraph_engine import ExperimentSpec
from MAS.llm import OpenRouterLLMClient
from MAS.self_evolved.claude_harness import ClaudeAgentSDKClient
from MAS.self_evolved.engine import SelfEvolvedEngine
from MAS.self_evolved.harness import build_harness

# A schema-valid structured answer; "42" survives into the extracted answer_artifact.
_DEFAULT_ANSWER_JSON = json.dumps(
    {
        "answer_artifact": "The answer is 42.",
        "summary": "42",
        "critique": "",
        "revision_request": "",
        "confidence": 0.9,
        "unresolved_issues": [],
        "evidence_summary": ["doc1"],
    }
)
# Default usage: input split across raw + cache so token_in == 25 exercises cache summation.
_DEFAULT_USAGE = {"input_tokens": 5, "cache_read_input_tokens": 20, "output_tokens": 9}


@dataclass(frozen=True)
class _Task:
    task_id: str
    prompt: Any
    reference_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# -- fake claude_agent_sdk module (no network, no real dependency) -------------


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _AssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class _ResultMessage:
    def __init__(
        self,
        result: str,
        usage: dict[str, Any],
        total_cost_usd: float,
        num_turns: int = 1,
        *,
        structured_output: Any = None,
        is_error: bool = False,
        subtype: str = "success",
        stop_reason: str | None = None,
    ) -> None:
        self.result = result
        self.usage = usage
        self.total_cost_usd = total_cost_usd
        self.num_turns = num_turns
        self.session_id = "fake-session"
        self.structured_output = structured_output
        self.is_error = is_error
        self.subtype = subtype
        self.stop_reason = stop_reason


class _SdkMcpTool:
    def __init__(self, name: str, description: str, schema: dict, handler: Any) -> None:
        self.name = name
        self.description = description
        self.input_schema = schema
        self.handler = handler


def _install_fake_sdk(
    *,
    answer: str = _DEFAULT_ANSWER_JSON,
    call_tools: bool = True,
    usage: dict[str, Any] | None = None,
    cost: float = 0.0123,
    truncate: bool = False,
    narration: str = "Let me search for the remaining criteria.",
    finalize_answer: str = _DEFAULT_ANSWER_JSON,
) -> types.ModuleType:
    """Fake claude_agent_sdk.

    ``truncate=True`` simulates a tool-using run cut off mid-investigation: the main
    query yields narration only (no JSON) plus an ``is_error`` result, so the harness
    must issue a finalization query (no mcp_servers, output_format set) which returns
    ``finalize_answer`` as ``structured_output``.
    """

    module = types.ModuleType("claude_agent_sdk")
    module.queries = []

    def _tool(name: str, description: str, schema: dict) -> Any:
        def decorator(fn: Any) -> _SdkMcpTool:
            return _SdkMcpTool(name, description, schema, fn)

        return decorator

    def _create_sdk_mcp_server(*, name: str, version: str = "1.0.0", tools: list | None = None):
        return {"name": name, "version": version, "tools": list(tools or [])}

    class _ClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)
            self.__dict__.update(kwargs)

    async def _query(*, prompt: str, options: Any):
        module.queries.append({"prompt": prompt, "options": options})
        servers = getattr(options, "mcp_servers", None) or {}
        is_finalize = getattr(options, "output_format", None) is not None and not servers
        if call_tools and servers:
            for server in servers.values():
                for sdk_tool in server["tools"]:
                    await sdk_tool.handler({"search_query": "needle"})
                break
        if is_finalize:
            try:
                structured = json.loads(finalize_answer)
            except Exception:
                structured = None
            yield _AssistantMessage([_TextBlock(finalize_answer)])
            yield _ResultMessage(
                finalize_answer,
                usage or {"input_tokens": 7, "cache_read_input_tokens": 6, "output_tokens": 4},
                cost,
                structured_output=structured,
            )
            return
        if truncate:
            yield _AssistantMessage([_TextBlock(narration)])
            yield _ResultMessage(
                "",
                usage or _DEFAULT_USAGE,
                cost,
                is_error=True,
                subtype="error_max_turns",
                stop_reason="max_turns",
            )
            return
        yield _AssistantMessage([_TextBlock(answer)])
        yield _ResultMessage(answer, usage or _DEFAULT_USAGE, cost)

    module.tool = _tool
    module.create_sdk_mcp_server = _create_sdk_mcp_server
    module.ClaudeAgentOptions = _ClaudeAgentOptions
    module.query = _query
    module.AssistantMessage = _AssistantMessage
    module.ResultMessage = _ResultMessage
    module.TextBlock = _TextBlock
    return module


def _generate(client: ClaudeAgentSDKClient, **kwargs: Any):
    defaults: dict[str, Any] = {
        "prompt": [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Find the answer."},
        ],
        "agent_type": "general",
        "task_id": "t0",
        "run_index": 0,
        "agent_id": "agent_0",
        "tools": None,
        "max_tool_iterations": 4,
    }
    defaults.update(kwargs)
    return client.generate(**defaults)


class TestClaudeHarness(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
        env.start()
        self.addCleanup(env.stop)

    def test_build_harness_switch(self) -> None:
        mock_client = OpenRouterLLMClient(
            OpenRouterConfig(api_key=None), {"default": "claude-sonnet-4-6"}
        )
        self.assertIs(build_harness("openrouter", mock_client), mock_client)
        with patch.dict(sys.modules, {"claude_agent_sdk": _install_fake_sdk()}):
            harness = build_harness("claude_agent_sdk", mock_client)
            self.assertIsInstance(harness, ClaudeAgentSDKClient)
            self.assertEqual(harness.models, mock_client.models)
        with self.assertRaises(ValueError):
            build_harness("unknown_backend", mock_client)

    def test_tool_bridging_and_result_mapping(self) -> None:
        fake = _install_fake_sdk()
        handler_calls: list[dict[str, Any]] = []

        def handler(args: dict[str, Any]) -> dict[str, Any]:
            handler_calls.append(args)
            return {"success": True, "result": "found it"}

        tools = [
            {
                "name": "google.web_search",
                "description": "Web search",
                "parameters": {
                    "type": "object",
                    "properties": {"search_query": {"type": "string"}},
                    "required": ["search_query"],
                },
                "handler": handler,
            }
        ]
        with patch.dict(sys.modules, {"claude_agent_sdk": fake}):
            client = ClaudeAgentSDKClient({"default": "claude-sonnet-4-6"})
            result = _generate(client, tools=tools)

        # The repo handler actually ran through the MCP bridge.
        self.assertEqual(handler_calls, [{"search_query": "needle"}])
        self.assertEqual(len(result.tool_calls), 1)
        record = result.tool_calls[0]
        self.assertEqual(record["tool_name"], "google.web_search")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["output"], {"success": True, "result": "found it"})

        # The structured JSON answer is captured (status would be "ok" downstream).
        parsed = _extract_json_payload(result.text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["answer_artifact"], "The answer is 42.")
        # JSON present in the main query -> no finalization, single SDK call.
        self.assertFalse(result.metadata["sdk_finalized"])
        self.assertEqual(len(fake.queries), 1)
        self.assertEqual(result.token_in, 25)  # 5 raw + 20 cache_read
        self.assertEqual(result.token_out, 9)
        self.assertAlmostEqual(result.cost_usd, 0.0123)
        self.assertEqual(result.model, "claude-sonnet-4-6")
        self.assertFalse(result.mock_used)
        self.assertEqual(result.metadata["provider"], "claude_agent_sdk")
        self.assertEqual(result.metadata["auth_source"], "api_key")
        self.assertEqual(result.metadata["num_turns"], 1)

        options = fake.queries[0]["options"]
        # Dotted names are sanitized for the MCP server; only bridged tools allowed.
        self.assertEqual(options.allowed_tools, ["mcp__mas__google_web_search"])
        self.assertEqual(options.max_turns, 4)
        self.assertIn("You are an expert.", options.system_prompt)
        self.assertIn("mcp__mas__google_web_search", options.system_prompt)
        self.assertIn("Find the answer.", fake.queries[0]["prompt"])

    def test_tool_handler_error_is_recorded(self) -> None:
        fake = _install_fake_sdk()

        def broken(args: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("upstream 429")

        tools = [{"name": "edgar_search", "description": "", "handler": broken}]
        with patch.dict(sys.modules, {"claude_agent_sdk": fake}):
            client = ClaudeAgentSDKClient({"default": "claude-sonnet-4-6"})
            result = _generate(client, tools=tools)
        record = result.tool_calls[0]
        self.assertEqual(record["status"], "error")
        self.assertIn("upstream 429", record["error"])

    def test_missing_package_raises_actionable_error(self) -> None:
        with patch.dict(sys.modules, {"claude_agent_sdk": None}):
            client = ClaudeAgentSDKClient({"default": "claude-sonnet-4-6"})
            with self.assertRaises(RuntimeError) as ctx:
                _generate(client)
        self.assertIn(".[claude]", str(ctx.exception))

    def test_missing_api_key_uses_sdk_default_auth(self) -> None:
        with (
            patch.dict(sys.modules, {"claude_agent_sdk": _install_fake_sdk()}),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}),
        ):
            client = ClaudeAgentSDKClient({"default": "claude-sonnet-4-6"})
            result = _generate(client)
        self.assertEqual(_extract_json_payload(result.text)["answer_artifact"], "The answer is 42.")
        self.assertEqual(result.metadata["auth_source"], "sdk_default")

    def test_finalization_on_truncated_tool_run(self) -> None:
        # Main tool query is cut off mid-investigation (narration only, no JSON);
        # the harness must run a second no-tools finalization query that yields JSON.
        fake = _install_fake_sdk(truncate=True)
        tools = [{"name": "search", "description": "", "handler": lambda a: {"ok": True}}]
        with patch.dict(sys.modules, {"claude_agent_sdk": fake}):
            client = ClaudeAgentSDKClient({"default": "claude-sonnet-4-6"})
            result = _generate(client, tools=tools)

        self.assertEqual(len(fake.queries), 2)
        # Second query is the finalization: no MCP servers, output_format set, no tools.
        final_opts = fake.queries[1]["options"]
        self.assertFalse(getattr(final_opts, "mcp_servers", None))
        self.assertEqual(final_opts.allowed_tools, [])
        self.assertIsNotNone(getattr(final_opts, "output_format", None))
        self.assertGreaterEqual(final_opts.max_turns, 2)  # >1 so the SDK can emit a result
        # Investigation notes carry the truncated narration into the finalize prompt.
        self.assertIn("Let me search for the remaining criteria.", fake.queries[1]["prompt"])

        parsed = _extract_json_payload(result.text)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["answer_artifact"], "The answer is 42.")
        self.assertTrue(result.metadata["sdk_finalized"])
        self.assertTrue(result.metadata["is_error"])  # main-query run status preserved
        self.assertEqual(result.metadata["stop_reason"], "max_turns")
        # Tokens are summed across the main (25) and finalization (7+6=13) calls.
        self.assertEqual(result.token_in, 25 + 13)
        self.assertEqual(result.token_out, 9 + 4)

    def test_no_finalization_when_json_already_present(self) -> None:
        fake = _install_fake_sdk()  # main answer is already JSON
        tools = [{"name": "search", "description": "", "handler": lambda a: {"ok": True}}]
        with patch.dict(sys.modules, {"claude_agent_sdk": fake}):
            client = ClaudeAgentSDKClient({"default": "claude-sonnet-4-6"})
            result = _generate(client, tools=tools)
        self.assertEqual(len(fake.queries), 1)
        self.assertFalse(result.metadata["sdk_finalized"])

    def test_cache_tokens_counted_in_token_in(self) -> None:
        fake = _install_fake_sdk(
            usage={
                "input_tokens": 12,
                "cache_read_input_tokens": 3000,
                "cache_creation_input_tokens": 500,
                "output_tokens": 40,
            }
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": fake}):
            client = ClaudeAgentSDKClient({"default": "claude-sonnet-4-6"})
            result = _generate(client)
        self.assertEqual(result.token_in, 12 + 3000 + 500)
        self.assertEqual(result.token_out, 40)

    def test_engine_run_through_fake_sdk(self) -> None:
        fake = _install_fake_sdk(call_tools=False)
        mock_client = OpenRouterLLMClient(
            OpenRouterConfig(api_key=None), {"default": "claude-sonnet-4-6"}
        )
        with patch.dict(sys.modules, {"claude_agent_sdk": fake}):
            harness = build_harness("claude_agent_sdk", mock_client)
            engine = SelfEvolvedEngine(
                harness,
                SelfEvolvedConfig(harness_backend="claude_agent_sdk", playbook_read=False),
            )
            spec = ExperimentSpec(
                topology="self_evolved",
                num_agents=3,
                rounds=2,
                communication_budget_per_agent=2,
                termination_consensus_mode="lexical",
                final_vote_mode="deterministic",
                benchmark_name="finance_agent",
                enable_dynamic_roles=False,
            )
            result = engine.run(
                task=_Task(task_id="sdk-task", prompt="What is the answer?"),
                run_index=0,
                seed=5,
                spec=spec,
                agent_types=["general"],
                tools=[],
            )

        validate_trace_events(result.trace_events)
        self.assertIn("42", str(result.final_answer))
        metadata = result.run_metadata
        self.assertEqual(metadata["self_evolved"]["harness_backend"], "claude_agent_sdk")
        # SDK usage/cost ride on the trace events, so C* metrics include them.
        act_events = [
            event
            for event in result.trace_events
            if event.event_type == "act" and event.actor.startswith("agent_")
        ]
        self.assertTrue(act_events)
        self.assertTrue(all(event.token_in == 25 for event in act_events))
        self.assertTrue(all(event.cost_usd > 0 for event in act_events))


if __name__ == "__main__":
    unittest.main()

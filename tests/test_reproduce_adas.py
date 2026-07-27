from __future__ import annotations

from benchmark.base import BenchmarkTask
from reproduce.adas.models import ADASSolution
from reproduce.adas.runtime_runner import (
    ADASRuntimeRunner,
    _parse_json_fields,
    compile_forward,
)


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        agent_id = str(kwargs["agent_id"])
        if "Planner" in agent_id:
            text = '{"plan": "inspect then update"}'
        elif "Verifier" in agent_id:
            text = '{"feedback": "", "status": "VERIFIED"}'
        else:
            text = '{"thinking": "done", "answer": "completed"}'

        class Result:
            token_in = 10
            token_out = 3
            cost_usd = 0.0
            model = "mock"
            mock_used = False
            metadata = {}
            tool_calls = []

            def __init__(self, value: str) -> None:
                self.text = value

        return Result(text)


def test_adas_parser_accepts_wrapped_python_literal_dict() -> None:
    text = "<|channel>thought\n<channel|>```json\n{'thinking': 'work', 'answer': '2220'}\n```"

    assert _parse_json_fields(text, ["thinking", "answer"]) == {
        "thinking": "work",
        "answer": "2220",
    }


def test_adas_safe_runtime_supports_basic_type_checks() -> None:
    solution = ADASSolution(
        name="safe builtins",
        thought="",
        code=(
            "def forward(self, taskInfo):\n"
            "    if isinstance(taskInfo, Info) and hasattr(taskInfo, 'content'):\n"
            "        return taskInfo.content\n"
            "    return ''\n"
        ),
    )

    assert compile_forward(solution.code) is not None


def test_workbench_assigns_tools_only_to_single_execution_agent() -> None:
    solution = ADASSolution(
        name="authority",
        thought="",
        code=(
            "def forward(self, taskInfo):\n"
            "    planner = LLMAgentBase(['plan'], 'Planner Agent')\n"
            "    executor = LLMAgentBase(['thinking', 'answer'], 'Executor Agent')\n"
            "    verifier = LLMAgentBase(['feedback', 'status'], 'Verifier Agent')\n"
            "    plan = planner([taskInfo], 'plan')[0]\n"
            "    thinking, answer = executor([taskInfo, plan], 'execute')\n"
            "    verifier([taskInfo, thinking, answer], 'verify')\n"
            "    return answer\n"
        ),
    )
    llm = RecordingLLM()
    runner = ADASRuntimeRunner(solution=solution, llm_client=llm)  # type: ignore[arg-type]
    tools = [{"type": "function", "function": {"name": "crm_add", "parameters": {}}}]

    result = runner.run_task(
        BenchmarkTask(task_id="w1", prompt="Add a lead", reference_answer=""),
        run_index=0,
        seed=42,
        benchmark_name="workbench",
        tools=tools,
    )

    assert result.final_answer == "completed"
    assert [bool(call["tools"]) for call in llm.calls] == [False, True, False]

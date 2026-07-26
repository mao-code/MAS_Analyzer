from __future__ import annotations

import json
from typing import Any

from .models import ADASSolution


EXAMPLE = {
    "thought": (
        "**Insights:** Your insights on what should be the next interesting agent.\n"
        "**Overall Idea:** Your reasoning and the overall concept behind the agent design.\n"
        "**Implementation:** Describe the implementation step by step."
    ),
    "name": "Name of your proposed agent",
    "code": "def forward(self, taskInfo):\n    # Your code here\n    return answer\n",
}


BASELINE_SOLUTIONS: tuple[ADASSolution, ...] = (
    ADASSolution(
        name="Chain-of-Thought",
        thought="Ask one LLM agent to reason step by step and return an answer.",
        code="""def forward(self, taskInfo):
    cot_instruction = "Please think step by step and then solve the task. Return the final answer in the answer field."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent")
    thinking, answer = cot_agent([taskInfo], cot_instruction)
    return answer
""",
    ),
    ADASSolution(
        name="Self-Consistency with Chain-of-Thought",
        thought="Sample several CoT answers and choose the most common normalized answer.",
        code="""def forward(self, taskInfo):
    from collections import Counter
    cot_instruction = "Please think step by step and then solve the task. Return the final answer in the answer field."
    agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", temperature=0.8) for _ in range(5)]
    answers = []
    for agent in agents:
        thinking, answer = agent([taskInfo], cot_instruction)
        answers.append(answer.content)
    return Counter(answers).most_common(1)[0][0]
""",
    ),
    ADASSolution(
        name="Self-Refine",
        thought="Generate an answer, ask a critic for feedback, and refine the answer.",
        code="""def forward(self, taskInfo):
    solve_instruction = "Please think step by step and then solve the task. Return the final answer in the answer field."
    refine_instruction = "Given previous attempts and feedback, carefully solve the task better. Return the final answer in the answer field."
    critic_instruction = "Review the answer above. If it is certainly correct, output True in correct; otherwise provide concise feedback."
    solver = LLMAgentBase(["thinking", "answer"], "Solver Agent")
    critic = LLMAgentBase(["feedback", "correct"], "Critic Agent")
    thinking, answer = solver([taskInfo], solve_instruction, 0)
    history = [thinking, answer]
    for i in range(5):
        feedback, correct = critic([taskInfo] + history, critic_instruction, i)
        if str(correct.content).strip().lower() == "true":
            break
        history.extend([feedback])
        thinking, answer = solver([taskInfo] + history, refine_instruction, i + 1)
        history.extend([thinking, answer])
    return answer
""",
    ),
    ADASSolution(
        name="LLM Debate",
        thought="Use several role-conditioned agents and a final decision agent.",
        code="""def forward(self, taskInfo):
    initial_instruction = "Please think step by step and solve the task. Return the final answer in the answer field."
    debate_instruction = "Given other agents' solutions, examine them and return the best final answer in the answer field."
    roles = ["Math Professor", "Grade School Teacher", "Math Enthusiast"]
    agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", role=role, temperature=0.8) for role in roles]
    outputs = []
    for agent in agents:
        thinking, answer = agent([taskInfo], initial_instruction)
        outputs.extend([thinking, answer])
    final_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", temperature=0.2)
    thinking, answer = final_agent([taskInfo] + outputs, debate_instruction)
    return answer
""",
    ),
    ADASSolution(
        name="Step-back Abstraction",
        thought=(
            "Let the LLM first think about the principles involved in solving this task. "
            "Understanding underlying principles can help the model reason through the task."
        ),
        code="""def forward(self, taskInfo):
    principle_instruction = "What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
    cot_instruction = "Given the question and the involved principle behind the question, think step by step and then solve the task. Return the final answer in the answer field."
    principle_agent = LLMAgentBase(["thinking", "principle"], "Principle Agent")
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent")
    thinking, principle = principle_agent([taskInfo], principle_instruction)
    thinking, answer = cot_agent([taskInfo, thinking, principle], cot_instruction)
    return answer
""",
    ),
    ADASSolution(
        name="Quality-Diversity",
        thought=(
            "Generate multiple diverse interesting solutions, then use a final decision agent "
            "to reason over the collected attempts."
        ),
        code="""def forward(self, taskInfo):
    cot_initial_instruction = "Please think step by step and then solve the task. Return the final answer in the answer field."
    qd_instruction = "Given previous attempts, try to come up with another interesting way to solve the task. Return the final answer in the answer field."
    final_decision_instruction = "Given all the above solutions, reason over them carefully and provide a final answer in the answer field."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent")
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", temperature=0.1)
    cot_inputs = [taskInfo]
    possible_answers = []
    thinking, answer = cot_agent(cot_inputs, cot_initial_instruction, 0)
    possible_answers.extend([thinking, answer])
    for i in range(3):
        cot_inputs.extend([thinking, answer])
        thinking, answer = cot_agent(cot_inputs, qd_instruction, i + 1)
        possible_answers.extend([thinking, answer])
    thinking, answer = final_decision_agent([taskInfo] + possible_answers, final_decision_instruction)
    return answer
""",
    ),
    ADASSolution(
        name="Dynamic Assignment of Roles",
        thought=(
            "Use dynamic control flow to route the task to a role-conditioned expert agent, "
            "similar to expert prompting and Auto-GPT style role assignment."
        ),
        code="""def forward(self, taskInfo):
    cot_instruction = "Please think step by step and then solve the task. Return the final answer in the answer field."
    expert_agents = [LLMAgentBase(["thinking", "answer"], "Expert Agent", role=role) for role in ["Math Professor", "Grade School Teacher", "Math Enthusiast", "Helpful Assistant"]]
    routing_instruction = "Given the task, please choose an Expert to answer the question. Choose from: Math Professor, Grade School Teacher, Math Enthusiast."
    routing_agent = LLMAgentBase(["choice"], "Routing agent")
    choice = routing_agent([taskInfo], routing_instruction)[0]
    if "professor" in choice.content.lower():
        expert_id = 0
    elif "teacher" in choice.content.lower():
        expert_id = 1
    elif "enthusiast" in choice.content.lower():
        expert_id = 2
    else:
        expert_id = 3
    thinking, answer = expert_agents[expert_id]([taskInfo], cot_instruction)
    return answer
""",
    ),
)


SYSTEM_PROMPT = "You are a helpful assistant. Make sure to return a well-formed JSON object."


def build_search_prompt(
    *,
    archive: list[ADASSolution],
    benchmark_name: str,
    benchmark_description: str,
) -> tuple[str, str]:
    archive_payload = [solution.to_payload() for solution in archive]
    user_prompt = f"""# Overview
You are an expert machine learning researcher testing various agentic systems.
Your objective is to design building blocks such as prompts and control flows within these systems.
Your aim is to design an optimal agent for the benchmark below.

# Target benchmark
Benchmark name: {benchmark_name}
Benchmark description:
{benchmark_description}

# Utility code available to your generated function
You must output one Python function with this exact interface:

```python
def forward(self, taskInfo):
    ...
    return answer
```

Inside the function, you may use:
- `LLMAgentBase(output_fields, agent_name, role="helpful assistant", model=None, temperature=...)`
- `Info(name, author, content, iteration_idx)`
- standard Python control flow and small helper functions
- `collections.Counter`

`taskInfo` is an Info object containing the task prompt. `LLMAgentBase(...)` returns Info objects,
and each Info has `.content`. Return either an Info object or a string. Always return the best answer
you can produce. Do not print, read files, write files, call subprocesses, or import network libraries.

# Discovered architecture archive
The archive stores previous agents and their validation fitness. Your goal is to maximize fitness.

{json.dumps(archive_payload, indent=2)}

# Output instruction and example
Reply exactly as JSON with keys `thought`, `name`, and `code`.
The `code` value must contain a complete `forward(self, taskInfo)` function only.

Example:
{json.dumps(EXAMPLE, indent=2)}

# Wrong implementation examples
- Do not create fake final answers like "Error" or "No answer generated".
- Do not return debug logs.
- Do not use filesystem, subprocess, sockets, requests, or arbitrary external imports.
- Do not forget to pass taskInfo to LLM agents that need the task.

# Your task
Observe the archive carefully. Propose an interestingly new agent architecture for this benchmark.
Be creative but keep the implementation robust and concise.
"""
    return SYSTEM_PROMPT, user_prompt


def build_reflexion_prompts(previous: ADASSolution | None) -> tuple[str, str]:
    prev = (
        "Here is the previous agent you tried:\n"
        + json.dumps(previous.to_payload(), indent=2)
        + "\n\n"
        if previous
        else ""
    )
    prompt1 = f"""{prev}Carefully review the proposed new architecture and reflect on:
1. Interestingness compared with the archive.
2. Implementation mistakes in the code.
3. Improvements that increase effectiveness without making the code fragile.

Reply exactly as JSON with keys `reflection`, `thought`, `name`, and `code`.
The `code` value must contain the corrected complete `forward(self, taskInfo)` function only.
"""
    prompt2 = """Revise the code one more time for robustness.
Check the exact function interface, avoid forbidden filesystem/subprocess/network behavior, and make sure the function always returns the best answer available.
Reply exactly as JSON with keys `reflection`, `thought`, `name`, and `code`.
"""
    return prompt1, prompt2

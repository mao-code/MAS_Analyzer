from __future__ import annotations

from .models import AgentSquareModule, AgentSquareSpec

PLANNING_NONE = AgentSquareModule(
    name="None",
    module_type="planning",
    thought="No explicit planning module.",
)

PLANNING_IO = AgentSquareModule(
    name="IO",
    module_type="planning",
    thought="Divide the task into a concise ordered list of subtasks.",
    prompt=(
        "Decompose the task into a short ordered list of actionable subtasks. "
        "For each subtask include the reasoning goal and any tool-use instruction. "
        "Return only the list."
    ),
)

PLANNING_DEPS = AgentSquareModule(
    name="DEPS",
    module_type="planning",
    thought="Generate subgoals with explicit temporal or evidence dependencies.",
    prompt=(
        "Break the task into subgoals and explicitly state dependencies between "
        "steps. Keep the plan minimal and executable."
    ),
)

REASONING_IO = AgentSquareModule(
    name="IO",
    module_type="reasoning",
    thought="Direct input-output reasoning.",
    prompt="Solve the task directly and return the required final answer/action.",
)

REASONING_COT = AgentSquareModule(
    name="COT",
    module_type="reasoning",
    thought="Step-by-step reasoning.",
    prompt=(
        "Solve the task step by step. Use the plan, memory, and tool information "
        "if provided. End with the required final answer/action."
    ),
)

REASONING_COT_SC = AgentSquareModule(
    name="COT-SC",
    module_type="reasoning",
    thought="Self-consistency over multiple reasoning samples.",
    prompt=(
        "Solve the task step by step. Produce a complete answer that can be "
        "compared with alternative samples for consistency."
    ),
    metadata={"samples": 3},
)

TOOLUSE_NONE = AgentSquareModule(
    name="None",
    module_type="tooluse",
    thought="No explicit tool-use module.",
)

TOOLUSE_IO = AgentSquareModule(
    name="IO",
    module_type="tooluse",
    thought="Select and call available tools when they are needed.",
    prompt=(
        "Use available tools when evidence, environment state, or side effects "
        "are required. Ground each tool call in the task and tool schema."
    ),
)

MEMORY_NONE = AgentSquareModule(
    name="None",
    module_type="memory",
    thought="No explicit memory module.",
)

MEMORY_SUMMARY = AgentSquareModule(
    name="Summary",
    module_type="memory",
    thought="Carry a compact summary of previous plan/tool/answer state.",
    prompt=(
        "Maintain a compact memory of useful observations, failed attempts, and "
        "constraints from earlier steps."
    ),
)


DEFAULT_MODULE_POOLS: dict[str, tuple[AgentSquareModule, ...]] = {
    "planning": (PLANNING_NONE, PLANNING_IO, PLANNING_DEPS),
    "reasoning": (REASONING_IO, REASONING_COT, REASONING_COT_SC),
    "tooluse": (TOOLUSE_NONE, TOOLUSE_IO),
    "memory": (MEMORY_NONE, MEMORY_SUMMARY),
}


def default_spec() -> AgentSquareSpec:
    """Upstream initial agent: planning=None, reasoning=IO, tooluse=None, memory=None."""

    return AgentSquareSpec(
        planning=None,
        reasoning=REASONING_IO,
        tooluse=None,
        memory=None,
    )


def spec_from_names(
    *,
    planning: str = "None",
    reasoning: str = "IO",
    tooluse: str = "None",
    memory: str = "None",
    extra_modules: dict[str, dict[str, AgentSquareModule]] | None = None,
) -> AgentSquareSpec:
    def find(module_type: str, name: str) -> AgentSquareModule | None:
        if name.lower() == "none":
            return None
        if extra_modules:
            for module_name, module in extra_modules.get(module_type, {}).items():
                if module_name.lower() == name.lower():
                    return module
        for module in DEFAULT_MODULE_POOLS[module_type]:
            if module.name.lower() == name.lower():
                return module
        raise ValueError(f"Unknown AgentSquare {module_type} module: {name}")

    reasoning_module = find("reasoning", reasoning)
    if reasoning_module is None:
        raise ValueError("AgentSquare requires a reasoning module")
    return AgentSquareSpec(
        planning=find("planning", planning),
        reasoning=reasoning_module,
        tooluse=find("tooluse", tooluse),
        memory=find("memory", memory),
    )

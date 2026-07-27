"""Paper-faithful per-role prompt templates (MASS App. D/E style).

The MASS appendix defines prompt templates as DSPy-style signatures: an
instruction, fixed input/output fields, and examples. The optimizer may search
over instructions/demos, but the field contract defines the agentic module and
must not drift during LLM proposal. These templates encode that contract for
paper task families and adapt the closest matching family to this repo's
benchmarks:

- ``math_reasoning``   -> MATH templates
- ``discrete_reasoning`` -> DROP templates
- ``long_context``     -> HotpotQA / LongBench templates
- ``coding`` / ``tool_or_web`` -> MBPP / HumanEval templates

Block names map to paper roles: ``predictor`` -> Predictor, ``reflect`` ->
Reflector, ``debate`` -> Debator, ``summarize`` -> Summarizer, ``execute`` ->
Executor, ``aggregate`` -> the self-consistency / synthesis aggregator. The
``refine`` step in the executor reuses the predictor prompt, matching the paper.

These are the base search-space templates, not the authors' hidden discovered
best prompts. Callers may still override any block via
``MASSConfig.prompt_templates``.
"""

from __future__ import annotations

from .models import AgentPromptBundle


def _bundle(
    instruction: str,
    *,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    output_contract: str = "",
) -> AgentPromptBundle:
    return AgentPromptBundle(
        system_instruction=instruction,
        input_fields=inputs,
        output_fields=outputs,
        output_contract=output_contract,
    )


_QUESTION_INPUT = ("Question",)
_QUESTION_CONTEXT_INPUT = ("Question", "Context")
_PLAIN_ANSWER = "Return only the final answer unless the role explicitly asks for rationale fields."

# Shared aggregator instruction (self-consistency / majority across parallel
# predictor chains, as defined by the Aggregate block in Sec. 2.2).
_AGGREGATOR = (
    "These are candidate answers to the question produced by parallel agents. "
    "Examine the candidates, select the most consistent and well-supported answer, "
    "and return a single final answer. Do not average incompatible answers."
)


_MATH_TEMPLATES: dict[str, AgentPromptBundle] = {
    "predictor": _bundle(
        "Let's think step by step.",
        inputs=_QUESTION_INPUT,
        outputs=("Reasoning", "Answer"),
    ),
    "reflect": _bundle(
        (
            "Please review the answer above and criticize on where might be wrong. "
            "If you are absolutely sure it is correct, output 'True' in 'correctness'."
        ),
        inputs=("Question", "Text"),
        outputs=("Reasoning", "Feedback", "Correctness"),
    ),
    "debate": _bundle(
        (
            "These are the solutions to the question from other agents. Examine the "
            "solutions from other agents in your rationale, finish by giving an updated "
            "answer. Show your final answer bracketed between <answer> and </answer> at the end."
        ),
        inputs=("Question", "Solutions"),
        outputs=("Reasoning", "Answer"),
    ),
    "aggregate": _bundle(_AGGREGATOR, inputs=("Question", "Solutions"), outputs=("Answer",)),
}


_DROP_TEMPLATES: dict[str, AgentPromptBundle] = {
    "predictor": _bundle(
        (
            "Please think step by step and then solve the task. Please answer the "
            "following question based on the given context. Directly answer the "
            "question. Keep it very concise."
        ),
        inputs=_QUESTION_CONTEXT_INPUT,
        outputs=("Thinking", "Answer"),
        output_contract=_PLAIN_ANSWER,
    ),
    "reflect": _bundle(
        (
            "Verify that the answer is based on the provided context. Give your "
            "reflection in the rationale."
        ),
        inputs=("Question", "Context", "Text"),
        outputs=("Reasoning", "Correctness"),
    ),
    "debate": _bundle(
        (
            "These are the solutions to the question from other agents. Based on the "
            "context, examine the solutions from other agents in your rationale, finish "
            "by giving an updated answer."
        ),
        inputs=("Question", "Context", "Solutions"),
        outputs=("Reasoning", "Answer"),
    ),
    "aggregate": _bundle(_AGGREGATOR, inputs=("Question", "Context", "Solutions"), outputs=("Answer",)),
}


_LONG_CONTEXT_TEMPLATES: dict[str, AgentPromptBundle] = {
    "predictor": _bundle(
        (
            "Answer the following question based only on the provided context. Do not "
            "use external knowledge. Directly answer the question and keep it very concise."
        ),
        inputs=_QUESTION_CONTEXT_INPUT,
        outputs=("Answer",),
        output_contract="Only return the answer. Do not output any other words.",
    ),
    "summarize": _bundle(
        (
            "Based on the question, retrieve relevant information from context that is "
            "ONLY helpful in answering the question. Include all key information. Do not "
            "repeat context. Start with 'Summary:'."
        ),
        inputs=_QUESTION_CONTEXT_INPUT,
        outputs=("Summary",),
    ),
    "reflect": _bundle(
        "Verify that the answer is based on the provided context.",
        inputs=("Question", "Context", "Text"),
        outputs=("Reasoning", "Correctness"),
    ),
    "debate": _bundle(
        (
            "These are the solutions to the question from other agents. Based on the "
            "context, examine the solutions from other agents in your rationale, finish "
            "by giving an updated answer."
        ),
        inputs=("Question", "Context", "Solutions"),
        outputs=("Reasoning", "Answer"),
    ),
    "aggregate": _bundle(_AGGREGATOR, inputs=("Question", "Context", "Solutions"), outputs=("Answer",)),
}


_CODING_TEMPLATES: dict[str, AgentPromptBundle] = {
    "predictor": _bundle(
        (
            "Let's think step by step. Provide a complete and correct code implementation "
            "in python. Output only the code implementation. Do not include example usage "
            "or explanations."
        ),
        inputs=_QUESTION_INPUT,
        outputs=("Thinking", "Answer"),
        output_contract="Only output the code implementation.",
    ),
    "reflect": _bundle(
        (
            "Please determine the correctness of the solution in passing all test cases. "
            "If it fails, based on the error message and traceback, think step by step and "
            "carefully propose an updated solution with a correct code implementation in python."
        ),
        inputs=("Question", "Previous solution", "Traceback"),
        outputs=("Correctness", "Thinking", "Answer"),
    ),
    "execute": _bundle(
        (
            "Run the candidate solution against the available test cases. If there is an "
            "executive output in the traceback, parse the output into an assertion given the "
            "executive output. Output 'True'/'False' based on the correctness of the "
            "executive feedback; if there is an error message, output 'False'."
        ),
        inputs=("Question", "Previous solution", "Traceback"),
        outputs=("Correctness", "Feedback"),
    ),
    "debate": _bundle(
        (
            "These are the solutions to the question from other agents. Examine the "
            "solutions from other agents in your rationale, finish by giving an updated "
            "answer. Let's think step by step. Provide a complete and correct code "
            "implementation in python."
        ),
        inputs=("Question", "Solutions"),
        outputs=("Reasoning", "Answer"),
    ),
    "aggregate": _bundle(_AGGREGATOR, inputs=("Question", "Solutions"), outputs=("Answer",)),
}


_PLANCraft_CONTRACT = (
    "Return exactly one valid PlanCraft action for the current observation: "
    "`move: from [slot] to [slot] with quantity N`, "
    "`smelt: from [slot] to [slot] with quantity N`, `search: <recipe name>` when the recipe "
    "or accepted ingredient alternatives are uncertain, or `impossible` only after recipe search "
    "confirms that the task cannot be completed from the available inventory. When a recipe needs "
    "a missing intermediate item, search for that intermediate item's recipe before declaring "
    "the task impossible; continue recursively until a valid inventory action or a genuinely "
    "unobtainable ingredient is identified. "
    "Do not answer with the target item name, do not compare against a reference answer, "
    "and do not use match/mismatch wording."
)

_PLANCraft_TEMPLATES: dict[str, AgentPromptBundle] = {
    "predictor": _bundle(
        "Use the current inventory observation and crafting rules to choose the next valid PlanCraft action.",
        inputs=("Observation", "Target item"),
        outputs=("Reasoning", "Action"),
        output_contract=_PLANCraft_CONTRACT,
    ),
    "reflect": _bundle(
        "Check whether the proposed PlanCraft action is valid for the current inventory and target.",
        inputs=("Observation", "Target item", "Text"),
        outputs=("Reasoning", "Feedback", "Correctness"),
        output_contract=_PLANCraft_CONTRACT,
    ),
    "debate": _bundle(
        "Compare peer PlanCraft action proposals and finish with the single best next action.",
        inputs=("Observation", "Target item", "Solutions"),
        outputs=("Reasoning", "Action"),
        output_contract=_PLANCraft_CONTRACT,
    ),
    "aggregate": _bundle(
        "Select the best valid PlanCraft action from candidate actions.",
        inputs=("Observation", "Target item", "Solutions"),
        outputs=("Action",),
        output_contract=_PLANCraft_CONTRACT,
    ),
    "execute": _bundle(
        "Use environment feedback to identify concrete invalid actions and propose the next valid PlanCraft action.",
        inputs=("Observation", "Target item", "Previous action", "Execution feedback"),
        outputs=("Feedback", "Action"),
        output_contract=_PLANCraft_CONTRACT,
    ),
}


_TOOL_CONTRACT = (
    "Use provided tools when needed. Return a concise final answer that satisfies every part "
    "of the user request; do not invent tool results. The Answer field must contain the actual "
    "answer requested by the user, not only a plan, reasoning, an insufficient-evidence refusal, "
    "or an empty placeholder. If a tool schema supplies default/example values for required "
    "arguments omitted by the user, use those schema values rather than asking for clarification. "
    "Do not replace failed tool results with general-knowledge fallback content. When a tool returns "
    "a dict, list, table, or descriptive text, copy the concrete returned fields/items into the "
    "Answer and directly address each requested part. Never answer only that a tool can provide "
    "the data, that a result set was returned, or that the user should provide more input when "
    "usable tool output is already present."
)

_TOOL_TEMPLATES: dict[str, AgentPromptBundle] = {
    "predictor": _bundle(
        "Solve the task using the available tools when tool evidence or state changes are required.",
        inputs=("Task", "Available tools"),
        outputs=("Reasoning", "Answer"),
        output_contract=_TOOL_CONTRACT,
    ),
    "reflect": _bundle(
        "Check whether the tool-using answer satisfies the user request and identify missing tool evidence or state changes.",
        inputs=("Task", "Available tools", "Text"),
        outputs=("Reasoning", "Feedback", "Correctness"),
        output_contract=_TOOL_CONTRACT,
    ),
    "execute": _bundle(
        "Use available tool feedback to ground the solution and identify concrete failures or next actions.",
        inputs=("Task", "Available tools", "Previous answer", "Execution feedback"),
        outputs=("Feedback", "Answer"),
        output_contract=_TOOL_CONTRACT,
    ),
    "debate": _bundle(
        "Compare peer tool-using solutions, check tool evidence, and finish with the best final answer.",
        inputs=("Task", "Available tools", "Solutions"),
        outputs=("Reasoning", "Answer"),
        output_contract=_TOOL_CONTRACT,
    ),
    "aggregate": _bundle(
        "Select the best final answer from candidate tool-using solutions.",
        inputs=("Task", "Available tools", "Solutions"),
        outputs=("Answer",),
        output_contract=_TOOL_CONTRACT,
    ),
}

_BROWSECOMP_CONTRACT = (
    "This is a BrowseComp retrieval task. Use the search tool to gather evidence before "
    "answering. Do not say that context is missing. Do not ask for context. If search "
    "results are weak, search again with a different query. Return the final answer "
    "concisely, with enough supporting reasoning for the judge to identify the answer. "
    "Always give the most likely answer string; never finish with insufficient-evidence text."
)

_BROWSECOMP_TEMPLATES: dict[str, AgentPromptBundle] = {
    "predictor": _bundle(
        (
            "Solve the BrowseComp question by searching the local corpus, reading useful "
            "documents when needed, and deriving the answer from retrieved evidence."
        ),
        inputs=("Question", "Available tools"),
        outputs=("Search plan", "Evidence", "Answer"),
        output_contract=_BROWSECOMP_CONTRACT,
    ),
    "summarize": _bundle(
        (
            "Based on the BrowseComp question, retrieve relevant information from tool "
            "results and summarize only evidence that helps answer the question. If "
            "evidence is missing, identify the next search query instead of refusing."
        ),
        inputs=("Question", "Retrieved evidence"),
        outputs=("Summary",),
        output_contract=_BROWSECOMP_CONTRACT,
    ),
    "reflect": _bundle(
        (
            "Check whether the proposed BrowseComp answer is supported by retrieved "
            "evidence. If support is missing, say what evidence or search query is needed."
        ),
        inputs=("Question", "Available tools", "Text"),
        outputs=("Reasoning", "Feedback", "Correctness"),
        output_contract=_BROWSECOMP_CONTRACT,
    ),
    "debate": _bundle(
        (
            "Compare peer BrowseComp answers against their retrieved evidence and finish "
            "with the best supported answer."
        ),
        inputs=("Question", "Available tools", "Solutions"),
        outputs=("Reasoning", "Answer"),
        output_contract=_BROWSECOMP_CONTRACT,
    ),
    "aggregate": _bundle(
        (
            "Select the best supported BrowseComp answer from candidate solutions. Prefer "
            "answers backed by retrieved evidence over refusals or context-missing claims."
        ),
        inputs=("Question", "Available tools", "Solutions"),
        outputs=("Answer",),
        output_contract=_BROWSECOMP_CONTRACT,
    ),
}


_TEMPLATES_BY_FAMILY: dict[str, dict[str, AgentPromptBundle]] = {
    "math_reasoning": _MATH_TEMPLATES,
    "discrete_reasoning": _DROP_TEMPLATES,
    "long_context": _LONG_CONTEXT_TEMPLATES,
    "coding": _CODING_TEMPLATES,
    "tool_or_web": _TOOL_TEMPLATES,
}


def family_prompt_templates(
    family: str, *, benchmark_name: str | None = None
) -> dict[str, AgentPromptBundle]:
    """Return paper App. D role templates for a benchmark family.

    Unknown / general families return an empty dict so the framework keeps its
    generic default prompts.
    """

    if str(benchmark_name or "").lower() == "plancraft":
        return dict(_PLANCraft_TEMPLATES)
    if str(benchmark_name or "").lower() == "browsecomp":
        return dict(_BROWSECOMP_TEMPLATES)
    return dict(_TEMPLATES_BY_FAMILY.get(family, {}))

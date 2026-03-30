# MAS Runtime README

## Prompt Construction

The per-agent prompt is assembled in:

- `MAS/langgraph_engine.py::_build_agent_prompt`
- `MAS/langgraph_engine.py::_execute_agent_stage`
- `MAS/llm.py::OpenRouterLLMClient.generate`
- `MAS/llm.py::_generate_with_tools`

## Message / Tool Format Used Today

This runtime does **not** use LangChain `ChatPromptTemplate`, `HumanMessage`, `AIMessage`,
`ToolMessage`, or `model.bind_tools()`.

Instead, it uses the provider-native OpenAI-compatible chat format directly:

```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
```

Tools are attached through the OpenAI-compatible `tools=[...]` parameter in
`client.chat.completions.create(...)`.

This is not the LangChain abstraction layer, but it is a standard provider-side tool-calling
pattern and maps cleanly to the LangChain message model.

## Standard Agent Prompt Template

Each agent receives the following message stack.

### 1. System Message

```text
You are one agent in a deterministic multi-agent workflow.
Agent ID: {agent_id}
Agent Role: {role}
Stage Role: {stage_role}

Use only the task messages, the prior artifact, and the visible packets provided in this conversation.
Do not invent hidden context.
Return exactly one JSON object and do not wrap it in markdown.
Required JSON keys: answer_artifact, summary, critique, revision_request, confidence, unresolved_issues, evidence_summary.
If a field is unknown, use an empty string, an empty list, or a conservative confidence score.
```

### 2. Task Messages

If the benchmark already supplies a chat history, those messages are appended as-is.

Otherwise the runtime appends one user message:

```text
Task:
{task_prompt}
```

### 3. Stage Context Message

The runtime then appends one final user message with the structured execution state:

```text
Stage context follows as JSON. Treat it as the authoritative runtime state for this step.

{
  "agent_id": "{agent_id}",
  "agent_role": "{role}",
  "stage_role": "{stage_role}",
  "directive": "{directive}",
  "round_index": {round_index},
  "discussion_index": {discussion_index},
  "prior_artifact": {prior_artifact_json_or_null},
  "visible_packets": [{visible_packet_json}, ...]
}
```

## Tool-Calling Turn Shape

If tools are enabled, the model call is executed with `tools=[...]` and the message history is
extended in the standard OpenAI-compatible loop:

```python
messages = [
    *prompt_messages,
    {
        "role": "assistant",
        "content": "<assistant text>",
        "tool_calls": [
            {
                "id": "<tool_call_id>",
                "type": "function",
                "function": {
                    "name": "<tool_name>",
                    "arguments": "<json string>"
                }
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "<tool_call_id>",
        "name": "<tool_name>",
        "content": "<json-encoded tool output>"
    },
]
```

This is the provider-native equivalent of:

- LangChain `AIMessage(tool_calls=[...])`
- LangChain `ToolMessage(...)`

## Standard LangChain Equivalent

If this runtime were expressed with LangChain abstractions, the equivalent shape would be:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("placeholder", "{task_messages}"),
        ("human", STAGE_CONTEXT_TEMPLATE),
    ]
)

messages = prompt.invoke(
    {
        "task_messages": task_messages,
        "agent_id": agent_id,
        "role": role,
        "stage_role": stage_role,
        "directive": directive,
        "round_index": round_index,
        "discussion_index": discussion_index,
        "prior_artifact_json": prior_artifact_json,
        "visible_packets_json": visible_packets_json,
    }
)

model_with_tools = model.bind_tools(tools)
response = model_with_tools.invoke(messages)
```

We currently keep the lower-level provider-native format because it is explicit, reproducible in
traces, and easy to keep aligned across OpenAI-compatible backends.

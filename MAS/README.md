# MAS Runtime README

## Runtime Model

This runtime keeps the custom LangGraph topology engine and the provider-native OpenAI-compatible tool loop. It does not use LangChain message classes at runtime, but it follows the same supervisor/subagent design principles:

- structural stage roles control behavior
- dynamic domain personas specialize behavior
- tool-enabled answer stages must use tools honestly
- judges prefer direct, evidence-backed answers over blocked or planning outputs

Primary references:

- OpenAI, *A practical guide to building agents*: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- LangChain, *Subagents*: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
- LangChain, *Handoffs*: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
- LangChain, *Deep Agents overview*: https://docs.langchain.com/oss/python/deepagents/index

## Prompt Contract

Per-agent prompts are assembled in:

- `MAS/langgraph_engine.py::_build_agent_prompt`
- `MAS/langgraph_engine.py::_execute_agent_stage`
- `MAS/llm.py::OpenRouterLLMClient.generate`
- `MAS/llm.py::_generate_with_tools`

Instruction priority is explicit:

1. structural stage contract
2. tool-use contract
3. task and benchmark instructions
4. domain persona

That means personas never override stage behavior. A `Web Search Strategist` acting as a `worker` is still bound by the worker/tool/output contract, and an `aggregator` persona is still required to synthesize into one supported answer instead of free-form brainstorming.

## Stage Roles

Every agent returns one JSON object with:

- `answer_artifact`
- `summary`
- `critique`
- `revision_request`
- `confidence`
- `unresolved_issues`
- `evidence_summary`

Stage-specific rules:

- `planner`: produce a bounded work plan or task package, not a speculative final answer
- `worker`: gather/apply evidence and state the best supported answer
- `critic`: challenge weak claims, verify against evidence, and revise toward the strongest answer
- `aggregator`: synthesize peer outputs into one supported answer; do not just restate unresolved disagreement

For non-planner stages, `answer_artifact` must be either:

- a direct answer
- a concise blocked-status explanation

It must not be:

- a search plan
- a tool list
- a sub-question list
- a generic “I need to search” status update

## Tool Use

If tools are enabled, the runtime uses the standard provider-native loop:

```python
{"role": "assistant", "content": "...", "tool_calls": [...]}
{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
```

Tools are attached through the OpenAI-compatible `tools=[...]` parameter in `client.chat.completions.create(...)`.

Behavioral rules:

- answer-producing stages must call tools when evidence is missing or weak
- the runtime never fabricates tool calls after the model response
- `evidence_summary` must reflect actual tool outputs, visible packets, or the prior artifact
- blocked/no-evidence/planning outputs are treated as non-substantive for voting and termination

The runtime also applies one bounded corrective retry for tool-enabled answer stages that return a blocked or planning answer with zero tool calls. The retry reminder is short and operational; if the model still makes no tool calls, the run records an honest zero-tool result.

## Dynamic Personas

Dynamic role assignment remains first-class:

- `MAS/role_pools.py` defines benchmark-specific role pools
- `MAS/role_assigner.py` performs one pre-workflow LLM assignment pass
- `WorkflowState.domain_personas` stores the chosen role/persona per agent

Persona injection is additive. The prompt includes:

- `Agent Role`
- `Stage Role`
- optional `Domain Role`
- optional `Persona`

This matches the intended layering from the external guidance: the workflow decides what an agent is allowed to do; the persona shapes how it approaches that work.

## Final Answer And Termination

Termination and final selection are separate.

- termination judges decide whether another loop is likely to improve correctness
- final judges choose the best answer among the latest artifacts

Both now consider:

- answer text
- answer mode (`direct`, `blocked`, `plan`, `empty`)
- summary
- evidence summary
- confidence
- tool usage

Finalization guarantees:

- direct answers outrank blocked or planning outputs
- evidence-backed direct answers outrank unsupported direct answers when counts are otherwise tied
- no topology finalizes to an empty string
- if no supported direct answer exists, the runtime falls back to the best recent non-empty artifact text, then to a stable explicit failure string

## Discussion Rounds

`mas.minimum_discussion_rounds` applies only to discussion/debate controllers.

Outer collaboration loops use `rounds` only. In particular:

- `rounds=1` means one outer pass
- discussion controllers may still enforce their own minimum discussion count
- outer cycle termination nodes do not borrow `minimum_discussion_rounds`

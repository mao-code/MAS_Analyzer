# Agent prompting and tool-use design

How agent prompts are layered, what a stage is allowed to do, and why the runtime
never fabricates tool calls.

The MAS runtime follows a supervisor/subagent design while keeping the current custom topology engine and provider-native OpenAI-compatible tool loop.

- Structural workflow roles remain authoritative: planner/orchestrator/worker/critic/aggregator roles determine routing, visibility, and output contract.
- Dynamic personas specialize the agent within that structural role. They do not override stage rules or tool requirements.
- Tool-enabled answer-producing stages are expected to call tools when evidence is missing or weak. The runtime does not fabricate tool calls or synthetic retrieval after the fact.
- Final judges and deterministic fallbacks prefer direct, evidence-backed answers over blocked-status, planning-only, or "no evidence" outputs.
- Context sharing is explicit. Agents see only task messages, selected relay packets, their prior artifact, and the tool outputs they actually received.

This design aligns with current primary-source guidance on agent systems:

- OpenAI, *A practical guide to building agents*: start with clear instructions, explicit tool loops, and manager-pattern orchestration when specialization is useful.
- LangChain, *Subagents*: the main agent should see concise subagent outputs and treat tool/subagent descriptions as routing levers.
- LangChain, *Handoffs*: explicit context engineering matters; malformed or overly broad context degrades multi-agent behavior.
- LangChain, *Deep Agents overview*: keep the main context clean and isolate specialized work into bounded subagent contexts.

References:

- https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
- https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
- https://docs.langchain.com/oss/python/deepagents/index

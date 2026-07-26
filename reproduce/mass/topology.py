from __future__ import annotations

from dataclasses import replace

from .models import SearchSpace, WorkflowSpec


def build_initial_workflow(search_space: SearchSpace) -> WorkflowSpec:
    """Construct the initial single-agent workflow a0 used in Stage 1."""

    return WorkflowSpec(
        summarize_rounds=0,
        aggregate_width=1,
        reflect_rounds=0,
        debate_rounds=0,
        execute_enabled=False,
        order=search_space.topology_order,
    )


def build_block_workflow(search_space: SearchSpace, block_name: str) -> WorkflowSpec:
    """Construct the minimum topology used to warm up one building block."""

    workflow = build_initial_workflow(search_space)
    if block_name == "aggregate":
        positive = [value for value in search_space.aggregate if int(value) > 1]
        width = min(positive) if positive else max(1, search_space.aggregate_minimum_width)
        return replace(
            workflow, aggregate_width=max(int(width), search_space.aggregate_minimum_width)
        )
    if block_name == "summarize":
        positive = [value for value in search_space.summarize if int(value) > 0]
        rounds = min(positive) if positive else search_space.summarize_minimum_rounds
        return replace(
            workflow, summarize_rounds=max(int(rounds), search_space.summarize_minimum_rounds)
        )
    if block_name == "reflect":
        positive = [value for value in search_space.reflect if int(value) > 0]
        rounds = min(positive) if positive else search_space.reflect_minimum_rounds
        return replace(
            workflow, reflect_rounds=max(int(rounds), search_space.reflect_minimum_rounds)
        )
    if block_name == "debate":
        positive = [value for value in search_space.debate if int(value) > 0]
        rounds = min(positive) if positive else search_space.debate_minimum_rounds
        return replace(
            workflow,
            aggregate_width=max(workflow.aggregate_width, search_space.debate_minimum_width),
            debate_rounds=max(int(rounds), search_space.debate_minimum_rounds),
        )
    if block_name == "execute":
        return replace(
            workflow,
            execute_enabled=True,
            reflect_rounds=max(1, search_space.reflect_minimum_rounds),
        )
    raise ValueError(f"Unsupported block name: {block_name}")


def enumerate_workflows(search_space: SearchSpace) -> list[WorkflowSpec]:
    """Enumerate all valid workflows under the provided budget."""

    summarize_values = search_space.summarize if search_space.block_enabled("summarize") else (0,)
    reflect_values = search_space.reflect if search_space.block_enabled("reflect") else (0,)
    debate_values = search_space.debate if search_space.block_enabled("debate") else (0,)
    execute_values = search_space.execute if search_space.block_enabled("execute") else (False,)

    workflows: list[WorkflowSpec] = []
    for summarize_rounds in summarize_values:
        for aggregate_width in search_space.aggregate:
            for reflect_rounds in reflect_values:
                for debate_rounds in debate_values:
                    for execute_enabled in execute_values:
                        workflow = WorkflowSpec(
                            summarize_rounds=int(summarize_rounds),
                            aggregate_width=int(aggregate_width),
                            reflect_rounds=int(reflect_rounds),
                            debate_rounds=int(debate_rounds),
                            execute_enabled=bool(execute_enabled),
                            order=search_space.topology_order,
                        )
                        if workflow.estimated_agent_count <= search_space.max_agent_budget:
                            workflows.append(workflow)
    return workflows

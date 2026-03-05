from __future__ import annotations

from typing import Any

from .agentbench import AgentBenchBenchmark
from .browsecomp import BrowseCompBenchmark
from .finance_agent import FinanceAgentBenchmark
from .plancraft import PlancraftBenchmark
from .scicode import SciCodeBenchmark

BENCHMARK_REGISTRY = {
    "agentbench": AgentBenchBenchmark,
    "finance_agent": FinanceAgentBenchmark,
    "browsecomp": BrowseCompBenchmark,
    "plancraft": PlancraftBenchmark,
    "scicode": SciCodeBenchmark,
}


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARK_REGISTRY.keys())


def get_benchmark(name: str, config: dict[str, Any] | None = None):
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(list_benchmarks())
        raise ValueError(f"Unknown benchmark '{name}'. Available benchmarks: {available}")
    return BENCHMARK_REGISTRY[name](config=config)

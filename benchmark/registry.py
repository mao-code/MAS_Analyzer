from __future__ import annotations

from typing import Any, Dict

from .browsecomp import BrowseCompBenchmark
from .finance_agent import FinanceAgentBenchmark

BENCHMARK_REGISTRY = {
    "finance_agent": FinanceAgentBenchmark,
    "browsecomp": BrowseCompBenchmark,
}


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARK_REGISTRY.keys())


def get_benchmark(name: str, config: Dict[str, Any] | None = None):
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(list_benchmarks())
        raise ValueError(f"Unknown benchmark '{name}'. Available benchmarks: {available}")
    return BENCHMARK_REGISTRY[name](config=config)

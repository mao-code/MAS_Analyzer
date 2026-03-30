from __future__ import annotations

import importlib
from typing import Any

BENCHMARK_REGISTRY = {
    "agentbench": ("benchmark.agentbench", "AgentBenchBenchmark"),
    "finance_agent": ("benchmark.finance_agent", "FinanceAgentBenchmark"),
    "browsecomp": ("benchmark.browsecomp", "BrowseCompBenchmark"),
    "stabletoolbench": ("benchmark.stabletoolbench", "StableToolBenchBenchmark"),
    "plancraft": ("benchmark.plancraft", "PlancraftBenchmark"),
    "scicode": ("benchmark.scicode", "SciCodeBenchmark"),
    "webshop": ("benchmark.webshop", "WebShopBenchmark"),
    "workbench": ("benchmark.workbench", "WorkBenchBenchmark"),
}


def list_benchmarks() -> list[str]:
    return sorted(BENCHMARK_REGISTRY.keys())


def get_benchmark(name: str, config: dict[str, Any] | None = None):
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(list_benchmarks())
        raise ValueError(f"Unknown benchmark '{name}'. Available benchmarks: {available}")

    module_name, class_name = BENCHMARK_REGISTRY[name]
    module = importlib.import_module(module_name)
    benchmark_cls = getattr(module, class_name)
    return benchmark_cls(config=config)

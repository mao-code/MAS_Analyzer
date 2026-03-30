from __future__ import annotations

import importlib

from .base import BenchmarkAdapter, BenchmarkEvaluation, BenchmarkTask
from .registry import get_benchmark, list_benchmarks

_LAZY_EXPORTS = {
    "BrowseCompBenchmark": ("benchmark.browsecomp", "BrowseCompBenchmark"),
    "FinanceAgentBenchmark": ("benchmark.finance_agent", "FinanceAgentBenchmark"),
    "StableToolBenchBenchmark": ("benchmark.stabletoolbench", "StableToolBenchBenchmark"),
    "WebShopBenchmark": ("benchmark.webshop", "WebShopBenchmark"),
    "WorkBenchBenchmark": ("benchmark.workbench", "WorkBenchBenchmark"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'benchmark' has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BenchmarkAdapter",
    "BenchmarkEvaluation",
    "BenchmarkTask",
    "BrowseCompBenchmark",
    "FinanceAgentBenchmark",
    "StableToolBenchBenchmark",
    "WebShopBenchmark",
    "WorkBenchBenchmark",
    "get_benchmark",
    "list_benchmarks",
]

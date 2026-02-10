"""Compatibility shim: use the canonical `benchmark` package."""

from __future__ import annotations

import importlib
import sys

from benchmark import *  # noqa: F401,F403
from benchmark import __all__ as _benchmark_all

for _name in ("base", "registry", "finance_agent", "browsecomp"):
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"benchmark.{_name}")

__all__ = list(_benchmark_all)

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import (
    ExtensionOptions,
    RunOutcome,
    aggregate_failure_modes,
    compute_run_metrics,
    compute_task_metrics,
)
from .schema import TraceEvent


@dataclass
class DescriptorResult:
    """Descriptor for a task built from repeated runs."""

    metrics: dict[str, Any]
    per_run: list[dict[str, Any]]
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        data = dict(self.metrics)
        for key, value in self.extensions.items():
            if isinstance(value, dict):
                data[key] = json.dumps(value, sort_keys=True)
            else:
                data[key] = value
        return data

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_flat_dict()])


def compute_descriptor_from_runs(
    runs: Sequence[Sequence[TraceEvent]],
    *,
    run_outcomes: Sequence[RunOutcome | None] | None = None,
    extensions: ExtensionOptions | None = None,
) -> DescriptorResult:
    if extensions is None:
        extensions = ExtensionOptions()
    if run_outcomes is None:
        run_outcomes = [None] * len(runs)
    if len(run_outcomes) != len(runs):
        raise ValueError("run_outcomes must have the same length as runs")

    run_metrics = [
        compute_run_metrics(run, outcome=outcome, extensions=extensions)
        for run, outcome in zip(runs, run_outcomes, strict=True)
    ]
    metrics = compute_task_metrics(run_metrics)
    extensions_out: dict[str, Any] = {}

    if extensions.failure_mode_hist:
        extensions_out["failure_mode_hist"] = aggregate_failure_modes(run_metrics)

    return DescriptorResult(metrics=metrics, per_run=run_metrics, extensions=extensions_out)


def write_descriptor_json(result: DescriptorResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": result.metrics,
        "per_run": result.per_run,
        "extensions": result.extensions,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_descriptor_csv(result: DescriptorResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = result.to_dataframe()
    df.to_csv(path, index=False)

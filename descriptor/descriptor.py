from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .metrics import ExtensionOptions, aggregate_failure_modes, compute_run_metrics, compute_task_metrics
from .schema import TraceEvent


@dataclass
class DescriptorResult:
    """Descriptor for a task built from repeated runs."""

    metrics: Dict[str, Any]
    per_run: List[Dict[str, Any]]
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> Dict[str, Any]:
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
    evaluator: Optional[Any] = None,
    extensions: Optional[ExtensionOptions] = None,
) -> DescriptorResult:
    if extensions is None:
        extensions = ExtensionOptions()

    run_metrics = [
        compute_run_metrics(run, evaluator=evaluator, extensions=extensions) for run in runs
    ]
    metrics = compute_task_metrics(run_metrics)
    extensions_out: Dict[str, Any] = {}

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

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


DEFAULT_EXPERIMENTS = {
    "GPT-OSS-120B": Path(
        "20260427T134706Z__openai_gpt_oss_120b/20260427T134706Z__openai_gpt_oss_120b"
    ),
    "Gemma-4-31B-IT": Path(
        "20260427T134706Z__google_gemma_4_31b_it_nitro/"
        "20260427T134706Z__google_gemma_4_31b_it_nitro"
    ),
    "Qwen3-32B": Path(
        "20260426_qwen3_32b_paper_run__qwen_qwen3_32b_nitro/"
        "20260426_qwen3_32b_paper_run__qwen_qwen3_32b_nitro"
    ),
}
MODEL_ORDER = list(DEFAULT_EXPERIMENTS)
QUALITY_METRICS = ["success", "score", "completion"]
COST_METRICS = ["tokens_total", "tool_calls_total", "communication_count", "handoff_count"]
TEST_METRICS = QUALITY_METRICS + ["tokens_total"]

FAILURE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("rate_limit", re.compile(r"\b(rate.?limit|429|too many requests)\b", re.I)),
    ("timeout", re.compile(r"\b(timeout|timed out|deadline|read timed out)\b", re.I)),
    ("network_error", re.compile(r"\b(connectionerror|connecterror|network error|dns error|ssl error|socket error|temporar(?:y|ily) unavailable)\b", re.I)),
    ("api_error", re.compile(r"\b(apierror|api error|bad gateway|502|503|504|server error|internal server error|service unavailable)\b", re.I)),
    ("malformed_response", re.compile(r"\b(jsondecode|invalid json|malformed|parse error|schema validation)\b", re.I)),
    ("missing_output", re.compile(r"\b(no final answer|empty response|missing final|no response)\b", re.I)),
    ("evaluation_parse_error", re.compile(r"\b(evaluation failed|eval.*parse|grader.*error)\b", re.I)),
]
EXCLUDABLE_REASONS = {
    "missing_trace_metrics",
    "missing_trajectory",
    "zero_token_incomplete",
    "api_error",
    "rate_limit",
    "timeout",
    "network_error",
    "malformed_response",
    "missing_output",
    "evaluation_parse_error",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def _read_text_sample(paths: Iterable[Path], *, limit_per_file: int = 250_000) -> str:
    chunks: list[str] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[:limit_per_file])
        except Exception:
            continue
    return "\n".join(chunks)


def _number(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return np.nan
    return _number(value)


def _run_index_from_name(path: Path) -> int | None:
    match = re.match(r"run_(\d+)\.", path.name)
    return int(match.group(1)) if match else None


def _classify_failure(
    *,
    has_trace_metrics: bool,
    has_trajectory: bool,
    metrics: dict[str, Any],
    evaluation: dict[str, Any],
    text_blob: str,
) -> tuple[str, bool]:
    if not has_trace_metrics:
        return "missing_trace_metrics", True
    if not has_trajectory:
        return "missing_trajectory", True

    tokens_total = _number(metrics.get("tokens_total", metrics.get("token_total")), 0.0)
    completion = metrics.get("completion", evaluation.get("completion"))
    success = metrics.get("success", evaluation.get("success"))
    if tokens_total <= 0 and completion is False:
        return "zero_token_incomplete", True

    for reason, pattern in FAILURE_PATTERNS:
        if pattern.search(text_blob):
            return reason, reason in EXCLUDABLE_REASONS

    if completion is False and pd.isna(_bool_number(success)):
        return "unknown_incomplete", False
    if success is False or _bool_number(success) == 0.0:
        return "normal_task_failure", False
    return "none", False


def collect_run_audit(experiments: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, root in experiments.items():
        root = root.resolve()
        for benchmark_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if benchmark_dir.name in {"Plot", "logs", "__MACOSX"}:
                continue
            benchmark = benchmark_dir.name
            for system_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
                system_label = system_dir.name
                for task_dir in sorted(path for path in system_dir.iterdir() if path.is_dir()):
                    run_indices = {
                        index
                        for path in task_dir.glob("run_*.*")
                        if (index := _run_index_from_name(path)) is not None
                    }
                    for run_index in sorted(run_indices):
                        trace_metrics_path = task_dir / f"run_{run_index}.trace_metrics.json"
                        trajectory_path = task_dir / f"run_{run_index}.trajectory.json"
                        trace_jsonl_path = task_dir / f"run_{run_index}.trace.jsonl"
                        eval_path = task_dir / f"run_{run_index}.eval.json"
                        result_path = task_dir / f"run_{run_index}.result.json"
                        metadata_path = task_dir / f"run_{run_index}.metadata.json"
                        trace_payload = _read_json(trace_metrics_path) if trace_metrics_path.exists() else {}
                        eval_payload = _read_json(eval_path) if eval_path.exists() else {}
                        metrics = trace_payload.get("metrics", {})
                        evaluation = trace_payload.get("evaluation", eval_payload)
                        completion_value = metrics.get("completion", evaluation.get("completion") if isinstance(evaluation, dict) else None)
                        tokens_value = _number(metrics.get("tokens_total", metrics.get("token_total")), 0.0)
                        needs_text_audit = (
                            not trace_metrics_path.exists()
                            or not trajectory_path.exists()
                            or completion_value is False
                            or tokens_value <= 0
                        )
                        text_blob = (
                            _read_text_sample([trajectory_path, trace_jsonl_path, result_path])
                            if needs_text_audit
                            else ""
                        )
                        reason, exclude_candidate = _classify_failure(
                            has_trace_metrics=trace_metrics_path.exists(),
                            has_trajectory=trajectory_path.exists(),
                            metrics=metrics,
                            evaluation=evaluation if isinstance(evaluation, dict) else {},
                            text_blob=text_blob,
                        )
                        rows.append(
                            {
                                "model": model,
                                "experiment_root": str(root),
                                "benchmark": benchmark,
                                "system_label": system_label,
                                "task_id": task_dir.name,
                                "run_index": run_index,
                                "has_trace_metrics": trace_metrics_path.exists(),
                                "has_trajectory": trajectory_path.exists(),
                                "has_eval": eval_path.exists(),
                                "has_result": result_path.exists(),
                                "success": _bool_number(metrics.get("success", evaluation.get("success") if isinstance(evaluation, dict) else None)),
                                "score": _number(metrics.get("score", evaluation.get("score") if isinstance(evaluation, dict) else np.nan)),
                                "completion": _bool_number(metrics.get("completion", evaluation.get("completion") if isinstance(evaluation, dict) else None)),
                                "tokens_total": _number(metrics.get("tokens_total", metrics.get("token_total"))),
                                "tool_calls_total": _number(metrics.get("tool_calls_total")),
                                "tool_error_count": _number(metrics.get("tool_fail_total", metrics.get("tool_error_count")), 0.0),
                                "communication_count": _number(metrics.get("communication_count")),
                                "handoff_count": _number(metrics.get("handoff_count")),
                                "latency_e2e": _number(metrics.get("latency_e2e", metrics.get("latency_total"))),
                                "failure_reason": reason,
                                "exclude_candidate": exclude_candidate,
                                "trace_metrics_path": str(trace_metrics_path),
                                "trajectory_path": str(trajectory_path),
                            }
                        )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    m = len(valid)
    running = 0.0
    for rank, (idx, value) in enumerate(valid.items(), start=1):
        running = max(running, min(1.0, value * (m - rank + 1)))
        adjusted.loc[idx] = running
    return adjusted


def _bootstrap_ci(values: pd.Series, *, seed: int = 1234, reps: int = 2000) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return np.nan, np.nan
    if numeric.size == 1:
        return float(numeric[0]), float(numeric[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(numeric, size=(reps, numeric.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _rank_biserial_from_diff(diff: pd.Series) -> float:
    values = pd.to_numeric(diff, errors="coerce").dropna()
    values = values[values != 0]
    if values.empty:
        return np.nan
    ranks = stats.rankdata(np.abs(values))
    pos = float(ranks[values.to_numpy() > 0].sum())
    neg = float(ranks[values.to_numpy() < 0].sum())
    denom = pos + neg
    return (pos - neg) / denom if denom else np.nan


def _task_level_from_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "success": "mean",
        "score": "mean",
        "completion": "mean",
        "tokens_total": "mean",
        "tool_calls_total": "mean",
        "tool_error_count": "mean",
        "communication_count": "mean",
        "handoff_count": "mean",
        "latency_e2e": "mean",
        "run_index": "count",
    }
    frame = (
        run_df.groupby(["model", "benchmark", "system_label", "task_id"], as_index=False)
        .agg({key: value for key, value in agg.items() if key in run_df.columns})
        .rename(columns={"run_index": "runs"})
    )
    return frame


def _descriptive_summary(task_df: pd.DataFrame) -> pd.DataFrame:
    return (
        task_df.groupby(["model", "benchmark", "system_label"], as_index=False)
        .agg(
            task_count=("task_id", "nunique"),
            runs=("runs", "sum"),
            avg_success=("success", "mean"),
            avg_score=("score", "mean"),
            avg_completion=("completion", "mean"),
            avg_tokens=("tokens_total", "mean"),
            avg_tool_calls=("tool_calls_total", "mean"),
            avg_tool_errors=("tool_error_count", "mean"),
            avg_communication=("communication_count", "mean"),
            avg_handoffs=("handoff_count", "mean"),
        )
        .sort_values(["model", "benchmark", "avg_success", "avg_tokens"], ascending=[True, True, False, True])
    )


def _pairwise_vs_sas(task_df: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in TEST_METRICS:
        for (model, benchmark, system), group in task_df.groupby(
            ["model", "benchmark", "system_label"], observed=True
        ):
            if system == "sas":
                continue
            sas = task_df[
                (task_df["model"] == model)
                & (task_df["benchmark"] == benchmark)
                & (task_df["system_label"] == "sas")
            ][["task_id", metric]].rename(columns={metric: "sas"})
            merged = group[["task_id", metric]].rename(columns={metric: "mas"}).merge(
                sas, on="task_id", how="inner"
            )
            merged = merged.dropna(subset=["mas", "sas"])
            if len(merged) < 2:
                continue
            diff = merged["mas"] - merged["sas"]
            alternative = "greater" if metric in QUALITY_METRICS else "two-sided"
            try:
                stat, p_value = stats.wilcoxon(diff, alternative=alternative, zero_method="zsplit")
            except ValueError:
                stat, p_value = np.nan, 1.0 if np.allclose(diff, 0) else np.nan
            ci_low, ci_high = _bootstrap_ci(diff)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "model": model,
                    "benchmark": benchmark,
                    "system_label": system,
                    "n_shared_tasks": len(merged),
                    "mean_sas": merged["sas"].mean(),
                    "mean_mas": merged["mas"].mean(),
                    "mean_delta": diff.mean(),
                    "median_delta": diff.median(),
                    "ci95_mean_delta_low": ci_low,
                    "ci95_mean_delta_high": ci_high,
                    "wilcoxon_stat": stat,
                    "p_value": p_value,
                    "alternative": alternative,
                    "rank_biserial": _rank_biserial_from_diff(diff),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_by_metric"] = out.groupby(["dataset", "metric"])["p_value"].transform(_holm_adjust)
    return out


def _model_pairwise(task_df: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_pairs = [("GPT-OSS-120B", "Gemma-4-31B-IT"), ("GPT-OSS-120B", "Qwen3-32B"), ("Gemma-4-31B-IT", "Qwen3-32B")]
    for metric in TEST_METRICS:
        for (benchmark, system), group in task_df.groupby(["benchmark", "system_label"], observed=True):
            for model_a, model_b in model_pairs:
                a = group[group["model"] == model_a][["task_id", metric]].rename(columns={metric: "a"})
                b = group[group["model"] == model_b][["task_id", metric]].rename(columns={metric: "b"})
                merged = a.merge(b, on="task_id", how="inner").dropna(subset=["a", "b"])
                if len(merged) < 2:
                    continue
                diff = merged["a"] - merged["b"]
                try:
                    stat, p_value = stats.wilcoxon(diff, alternative="two-sided", zero_method="zsplit")
                except ValueError:
                    stat, p_value = np.nan, 1.0 if np.allclose(diff, 0) else np.nan
                ci_low, ci_high = _bootstrap_ci(diff)
                rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "benchmark": benchmark,
                        "system_label": system,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_shared_tasks": len(merged),
                        "mean_a": merged["a"].mean(),
                        "mean_b": merged["b"].mean(),
                        "mean_delta_a_minus_b": diff.mean(),
                        "median_delta_a_minus_b": diff.median(),
                        "ci95_mean_delta_low": ci_low,
                        "ci95_mean_delta_high": ci_high,
                        "wilcoxon_stat": stat,
                        "p_value": p_value,
                        "rank_biserial": _rank_biserial_from_diff(diff),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_by_metric"] = out.groupby(["dataset", "metric"])["p_value"].transform(_holm_adjust)
    return out


def _friedman_topology_tests(task_df: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in TEST_METRICS:
        for (model, benchmark), group in task_df.groupby(["model", "benchmark"], observed=True):
            pivot = group.pivot_table(index="task_id", columns="system_label", values=metric, aggfunc="mean")
            pivot = pivot.dropna(axis=1, how="all").dropna(axis=0, how="any")
            if pivot.shape[0] < 2 or pivot.shape[1] < 3:
                continue
            try:
                stat, p_value = stats.friedmanchisquare(*(pivot[col] for col in pivot.columns))
            except ValueError:
                stat, p_value = np.nan, np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "model": model,
                    "benchmark": benchmark,
                    "n_shared_tasks": pivot.shape[0],
                    "n_topologies": pivot.shape[1],
                    "topologies": "|".join(map(str, pivot.columns)),
                    "friedman_chi_square": stat,
                    "p_value": p_value,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm_by_metric"] = out.groupby(["dataset", "metric"])["p_value"].transform(_holm_adjust)
    return out


def _write_report(
    *,
    failure_dir: Path,
    stats_dir: Path,
    run_audit: pd.DataFrame,
    exclusion_summary: pd.DataFrame,
    desc_raw: pd.DataFrame,
    desc_cleaned: pd.DataFrame,
    vs_sas: pd.DataFrame,
) -> None:
    total = len(run_audit)
    excluded = int(run_audit["exclude_candidate"].sum())
    lines = [
        "# Failure Analysis Report",
        "",
        f"- Total runs audited: `{total}`",
        f"- Exclusion candidates: `{excluded}` ({excluded / total:.2%} of runs)" if total else "- Exclusion candidates: `0`",
        "",
        "## Failure Reasons",
        "",
        "```text",
        exclusion_summary.to_string(index=False),
        "```",
        "",
        "## Interpretation Rule",
        "",
        "- Exclude candidates are infrastructure/API/evaluation failures, not ordinary wrong answers.",
        "- `normal_task_failure` means the model ran and produced an evaluable wrong answer; these are retained.",
    ]
    (failure_dir / "failure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sig = vs_sas[(vs_sas["metric"] == "success") & (vs_sas["p_holm_by_metric"] < 0.05)].copy()
    stat_lines = [
        "# Statistical Analysis Report",
        "",
        "Two datasets are reported:",
        "",
        "- `raw`: all audited runs with metrics.",
        "- `cleaned`: excludes failure candidates from the failure audit.",
        "",
        "## Files",
        "",
        "- `descriptive_summary_raw.csv`",
        "- `descriptive_summary_cleaned.csv`",
        "- `pairwise_vs_sas_tests.csv`",
        "- `model_pairwise_tests.csv`",
        "- `friedman_topology_tests.csv`",
        "",
        "## Significant Success Lifts vs SAS After Holm Correction",
        "",
    ]
    if sig.empty:
        stat_lines.append("_None at p < 0.05._")
    else:
        display = sig[
            [
                "dataset",
                "model",
                "benchmark",
                "system_label",
                "n_shared_tasks",
                "mean_delta",
                "ci95_mean_delta_low",
                "ci95_mean_delta_high",
                "p_holm_by_metric",
                "rank_biserial",
            ]
        ].sort_values(["dataset", "model", "benchmark", "p_holm_by_metric"])
        stat_lines.extend(["```text", display.round(4).to_string(index=False), "```"])
    stat_lines.extend(
        [
            "",
            "## Raw vs Cleaned Row Counts",
            "",
            f"- Raw summary rows: `{len(desc_raw)}`",
            f"- Cleaned summary rows: `{len(desc_cleaned)}`",
        ]
    )
    (stats_dir / "statistical_report.md").write_text("\n".join(stat_lines) + "\n", encoding="utf-8")


def analyze(experiments: dict[str, Path], failure_dir: Path, stats_dir: Path) -> None:
    failure_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    resolved = {model: path.expanduser().resolve() for model, path in experiments.items()}
    run_audit = collect_run_audit(resolved)
    run_audit.to_csv(failure_dir / "run_failure_audit.csv", index=False)
    candidates = run_audit[run_audit["exclude_candidate"]].copy()
    candidates.to_csv(failure_dir / "exclusion_candidates.csv", index=False)
    exclusion_summary = (
        run_audit.groupby(["model", "benchmark", "system_label", "failure_reason"], as_index=False)
        .agg(runs=("run_index", "count"), exclude_candidates=("exclude_candidate", "sum"))
        .sort_values(["exclude_candidates", "runs"], ascending=[False, False])
    )
    exclusion_summary.to_csv(
        failure_dir / "exclusion_summary_by_model_benchmark_topology.csv",
        index=False,
    )
    reason_summary = (
        run_audit.groupby(["failure_reason"], as_index=False)
        .agg(runs=("run_index", "count"), exclude_candidates=("exclude_candidate", "sum"))
        .sort_values(["exclude_candidates", "runs"], ascending=[False, False])
    )
    reason_summary.to_csv(failure_dir / "exclusion_summary_by_reason.csv", index=False)

    metric_rows = run_audit[run_audit["has_trace_metrics"]].copy()
    cleaned_rows = metric_rows[~metric_rows["exclude_candidate"]].copy()
    metric_rows.to_csv(stats_dir / "run_level_raw.csv", index=False)
    cleaned_rows.to_csv(stats_dir / "run_level_cleaned.csv", index=False)

    task_raw = _task_level_from_runs(metric_rows)
    task_cleaned = _task_level_from_runs(cleaned_rows)
    task_raw.to_csv(stats_dir / "task_level_raw.csv", index=False)
    task_cleaned.to_csv(stats_dir / "task_level_cleaned.csv", index=False)

    desc_raw = _descriptive_summary(task_raw)
    desc_cleaned = _descriptive_summary(task_cleaned)
    desc_raw.to_csv(stats_dir / "descriptive_summary_raw.csv", index=False)
    desc_cleaned.to_csv(stats_dir / "descriptive_summary_cleaned.csv", index=False)

    tests = []
    for label, task_df in [("raw", task_raw), ("cleaned", task_cleaned)]:
        tests.append(_pairwise_vs_sas(task_df, dataset=label))
    vs_sas = pd.concat([frame for frame in tests if not frame.empty], ignore_index=True)
    vs_sas.to_csv(stats_dir / "pairwise_vs_sas_tests.csv", index=False)

    model_tests = pd.concat(
        [
            _model_pairwise(task_raw, dataset="raw"),
            _model_pairwise(task_cleaned, dataset="cleaned"),
        ],
        ignore_index=True,
    )
    model_tests.to_csv(stats_dir / "model_pairwise_tests.csv", index=False)

    friedman_tests = pd.concat(
        [
            _friedman_topology_tests(task_raw, dataset="raw"),
            _friedman_topology_tests(task_cleaned, dataset="cleaned"),
        ],
        ignore_index=True,
    )
    friedman_tests.to_csv(stats_dir / "friedman_topology_tests.csv", index=False)

    _write_report(
        failure_dir=failure_dir,
        stats_dir=stats_dir,
        run_audit=run_audit,
        exclusion_summary=reason_summary,
        desc_raw=desc_raw,
        desc_cleaned=desc_cleaned,
        vs_sas=vs_sas,
    )

    manifest = {
        "experiments": {model: str(path) for model, path in resolved.items()},
        "outputs": {
            "failure_dir": str(failure_dir.resolve()),
            "stats_dir": str(stats_dir.resolve()),
        },
        "run_count": int(len(run_audit)),
        "exclusion_candidate_count": int(run_audit["exclude_candidate"].sum()),
    }
    (stats_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit API/infra failures and run statistical tests across the three model experiments."
    )
    parser.add_argument("--failure-dir", default="outputs/failure_analyses")
    parser.add_argument("--stats-dir", default="outputs/stats")
    args = parser.parse_args()
    analyze(DEFAULT_EXPERIMENTS, Path(args.failure_dir), Path(args.stats_dir))
    print(Path(args.failure_dir).resolve())
    print(Path(args.stats_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

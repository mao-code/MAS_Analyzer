from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .distances import covariance_inverse, mahalanobis_distance, pairwise_mahalanobis
from .embeddings import pca_2d, umap_2d
from .pareto import ideal_point_distance, pareto_frontier
from .scaling import robust_scale

DEFAULT_DESCRIPTOR_COLUMNS = [
    "Q1_success_rate",
    "Q2_completion_rate",
    "C1_latency_p95",
    "C2_tokens_total",
    "C3_cost_total",
    "C4_tool_calls_total",
    "C5_tool_error_rate",
    "R1_success_var",
    "R2_latency_var",
    "R3_tokens_var",
    "P1_steps_total",
    "P2_backtrack_rate",
    "P3_loop_score",
    "P4_verification_density",
]

DEFAULT_OBJECTIVES = {
    "eval_avg_score": "max",
    "Q1_success_rate": "max",
    "C2_tokens_total": "min",
    "C1_latency_p95": "min",
    "P3_loop_score": "min",
}


@dataclass(frozen=True)
class TopologyRun:
    topology: str
    run_timestamp: str
    run_dir: Path
    summary_csv: Path
    experiment_settings: Path | None


@dataclass
class TopologyAnalysisResult:
    experiment_root: Path
    output_dir: Path
    task_metrics: pd.DataFrame
    topology_metrics: pd.DataFrame
    pareto_frontier: pd.DataFrame
    embeddings_pca: pd.DataFrame
    embeddings_umap: pd.DataFrame | None
    artifacts: dict[str, str]


def discover_topology_runs(
    experiment_root: str | Path,
    *,
    latest_only: bool = True,
) -> list[TopologyRun]:
    root = Path(experiment_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")

    runs: list[TopologyRun] = []
    for topology_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if topology_dir.name in {"configs", "topology_analysis"}:
            continue
        candidates = sorted(
            [
                path
                for path in topology_dir.iterdir()
                if path.is_dir() and (path / "summary.csv").exists()
            ],
            key=lambda path: path.name,
        )
        if not candidates:
            continue
        if latest_only:
            candidates = [candidates[-1]]

        for run_dir in candidates:
            runs.append(
                TopologyRun(
                    topology=topology_dir.name,
                    run_timestamp=run_dir.name,
                    run_dir=run_dir,
                    summary_csv=run_dir / "summary.csv",
                    experiment_settings=(
                        run_dir / "experiment_settings.json"
                        if (run_dir / "experiment_settings.json").exists()
                        else None
                    ),
                )
            )

    if not runs:
        raise ValueError(f"No topology run directories with summary.csv found under: {root}")
    return runs


def load_task_metrics(runs: list[TopologyRun]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        frame = pd.read_csv(run.summary_csv)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["topology"] = run.topology
        frame["run_timestamp"] = run.run_timestamp
        frame["run_dir"] = str(run.run_dir)
        frames.append(frame)

    if not frames:
        raise ValueError("No task-level records loaded from summary.csv files")
    return pd.concat(frames, ignore_index=True)


def aggregate_topology_metrics(task_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = task_df.select_dtypes(include=[np.number]).columns.tolist()
    if "task_id" in numeric_cols:
        numeric_cols.remove("task_id")
    grouped = task_df.groupby("topology", dropna=False)

    agg = grouped[numeric_cols].mean()
    agg["n_tasks"] = grouped.size().astype(float)
    agg["run_count"] = grouped["run_timestamp"].nunique().astype(float)
    agg = agg.sort_index()
    return agg


def analyze_topology_experiment(
    experiment_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    descriptor_columns: list[str] | None = None,
    objectives: dict[str, str] | None = None,
    objective_weights: dict[str, float] | None = None,
    include_umap: bool = False,
    latest_only: bool = True,
) -> TopologyAnalysisResult:
    runs = discover_topology_runs(experiment_root, latest_only=latest_only)
    root = Path(experiment_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "topology_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    task_df = load_task_metrics(runs)
    topology_df = aggregate_topology_metrics(task_df)

    descriptor_columns = descriptor_columns or list(DEFAULT_DESCRIPTOR_COLUMNS)
    descriptor_cols = [col for col in descriptor_columns if col in topology_df.columns]
    if not descriptor_cols:
        raise ValueError(
            "No descriptor columns found in topology metrics. "
            f"Requested columns: {descriptor_columns}"
        )

    feature_df = topology_df[descriptor_cols].astype(float)
    feature_df = feature_df.fillna(feature_df.mean()).fillna(0.0)
    X = feature_df.to_numpy(dtype=float)

    X_scaled, scaler = robust_scale(X, feature_names=descriptor_cols)
    cov, cov_inv = covariance_inverse(X_scaled, regularization=1e-6)
    centroid = np.mean(X_scaled, axis=0)

    topology_df = topology_df.copy()
    topology_df["distance_mahalanobis_to_centroid"] = pairwise_mahalanobis(
        X_scaled, centroid, cov_inv
    )

    if "sas" in topology_df.index:
        sas_idx = topology_df.index.tolist().index("sas")
        sas_vector = X_scaled[sas_idx]
        topology_df["distance_l2_to_sas"] = np.linalg.norm(X_scaled - sas_vector, axis=1)
        topology_df["distance_mahalanobis_to_sas"] = [
            mahalanobis_distance(row, sas_vector, cov_inv) for row in X_scaled
        ]
    else:
        topology_df["distance_l2_to_sas"] = np.nan
        topology_df["distance_mahalanobis_to_sas"] = np.nan

    objectives = objectives or dict(DEFAULT_OBJECTIVES)
    missing_objectives = [key for key in objectives if key not in topology_df.columns]
    if missing_objectives:
        raise ValueError(f"Objective columns missing from topology metrics: {missing_objectives}")

    pareto_mask = pareto_frontier(topology_df, objectives, return_mask=True)
    topology_df["pareto_frontier"] = pareto_mask

    d_ideal, ideal_point, normalized_objectives = ideal_point_distance(
        topology_df,
        objectives,
        weights=objective_weights,
    )
    topology_df["d_ideal"] = d_ideal
    topology_df["selection_rank"] = (
        topology_df["d_ideal"].rank(method="dense", ascending=True).astype(int)
    )

    frontier_df = (
        topology_df[topology_df["pareto_frontier"]]
        .sort_values("d_ideal", ascending=True)
        .copy()
    )

    embedding_pca, pca_model = pca_2d(X_scaled)
    pca_df = pd.DataFrame(
        {
            "topology": topology_df.index.tolist(),
            "pca_1": embedding_pca[:, 0],
            "pca_2": embedding_pca[:, 1],
            "pareto_frontier": topology_df["pareto_frontier"].to_numpy(dtype=bool),
            "d_ideal": topology_df["d_ideal"].to_numpy(dtype=float),
            "selection_rank": topology_df["selection_rank"].to_numpy(dtype=int),
        }
    )

    umap_df: pd.DataFrame | None = None
    umap_note: str | None = None
    if include_umap:
        try:
            embedding_umap, _ = umap_2d(X_scaled, random_state=42)
            umap_df = pd.DataFrame(
                {
                    "topology": topology_df.index.tolist(),
                    "umap_1": embedding_umap[:, 0],
                    "umap_2": embedding_umap[:, 1],
                    "pareto_frontier": topology_df["pareto_frontier"].to_numpy(dtype=bool),
                }
            )
        except Exception as exc:
            umap_note = f"UMAP unavailable: {exc}"

    task_csv = out_dir / "task_metrics.csv"
    topology_csv = out_dir / "topology_metrics.csv"
    frontier_csv = out_dir / "pareto_frontier.csv"
    scaled_csv = out_dir / "scaled_descriptor_matrix.csv"
    pca_csv = out_dir / "embedding_pca.csv"
    report_json = out_dir / "topology_analysis.json"
    report_md = out_dir / "report.md"

    task_df.to_csv(task_csv, index=False)
    topology_df.reset_index().rename(columns={"index": "topology"}).to_csv(topology_csv, index=False)
    frontier_df.reset_index().rename(columns={"index": "topology"}).to_csv(frontier_csv, index=False)
    pd.DataFrame(X_scaled, columns=descriptor_cols, index=topology_df.index).reset_index().rename(
        columns={"index": "topology"}
    ).to_csv(scaled_csv, index=False)
    pca_df.to_csv(pca_csv, index=False)
    if umap_df is not None:
        umap_df.to_csv(out_dir / "embedding_umap.csv", index=False)

    plot_paths = _write_visualizations(
        out_dir=out_dir,
        topology_df=topology_df,
        pca_df=pca_df,
        objective_x="C2_tokens_total" if "C2_tokens_total" in topology_df.columns else None,
        objective_y="eval_avg_score" if "eval_avg_score" in topology_df.columns else None,
    )

    payload = {
        "experiment_root": str(root),
        "latest_only": latest_only,
        "runs_discovered": [
            {
                "topology": run.topology,
                "run_timestamp": run.run_timestamp,
                "run_dir": str(run.run_dir),
                "summary_csv": str(run.summary_csv),
            }
            for run in runs
        ],
        "descriptor_columns": descriptor_cols,
        "objectives": objectives,
        "objective_weights": objective_weights or {},
        "scaler": scaler.to_dict(),
        "covariance_matrix": cov.tolist(),
        "covariance_inverse": cov_inv.tolist(),
        "centroid": centroid.tolist(),
        "ideal_point": ideal_point.tolist(),
        "pca_explained_variance_ratio": pca_model.explained_variance_ratio_.tolist(),
        "topology_ranking": (
            topology_df.sort_values("selection_rank")
            .reset_index()
            .rename(columns={"index": "topology"})[
                [
                    "topology",
                    "selection_rank",
                    "d_ideal",
                    "pareto_frontier",
                    "eval_avg_score",
                    "C2_tokens_total",
                    "C1_latency_p95",
                    "P3_loop_score",
                    "distance_mahalanobis_to_centroid",
                    "distance_mahalanobis_to_sas",
                ]
            ]
            .to_dict(orient="records")
        ),
        "umap_note": umap_note,
        "artifacts": {
            "task_metrics_csv": str(task_csv),
            "topology_metrics_csv": str(topology_csv),
            "pareto_frontier_csv": str(frontier_csv),
            "scaled_descriptor_matrix_csv": str(scaled_csv),
            "embedding_pca_csv": str(pca_csv),
            "analysis_json": str(report_json),
            "report_md": str(report_md),
            **plot_paths,
        },
    }
    report_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _write_markdown_report(
        path=report_md,
        topology_df=topology_df,
        frontier_df=frontier_df,
        objectives=objectives,
        plots=plot_paths,
        umap_note=umap_note,
    )

    return TopologyAnalysisResult(
        experiment_root=root,
        output_dir=out_dir,
        task_metrics=task_df,
        topology_metrics=topology_df,
        pareto_frontier=frontier_df,
        embeddings_pca=pca_df,
        embeddings_umap=umap_df,
        artifacts={key: str(value) for key, value in payload["artifacts"].items()},
    )


def _write_visualizations(
    *,
    out_dir: Path,
    topology_df: pd.DataFrame,
    pca_df: pd.DataFrame,
    objective_x: str | None,
    objective_y: str | None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        paths["plot_note"] = f"matplotlib unavailable: {exc}"
        return paths

    pca_plot = out_dir / "pca_frontier.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in pca_df.iterrows():
        color = "#d62728" if bool(row["pareto_frontier"]) else "#1f77b4"
        marker = "D" if bool(row["pareto_frontier"]) else "o"
        ax.scatter(float(row["pca_1"]), float(row["pca_2"]), color=color, marker=marker, s=80)
        ax.text(float(row["pca_1"]) + 0.02, float(row["pca_2"]) + 0.02, str(row["topology"]), fontsize=8)
    ax.set_title("Topology Embedding (PCA) with Pareto Frontier")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(pca_plot, dpi=180)
    plt.close(fig)
    paths["pca_frontier_png"] = str(pca_plot)

    if objective_x and objective_y and objective_x in topology_df.columns and objective_y in topology_df.columns:
        frontier_plot = out_dir / "pareto_tradeoff.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        for topology, row in topology_df.iterrows():
            color = "#d62728" if bool(row["pareto_frontier"]) else "#1f77b4"
            marker = "D" if bool(row["pareto_frontier"]) else "o"
            ax.scatter(float(row[objective_x]), float(row[objective_y]), color=color, marker=marker, s=80)
            ax.text(
                float(row[objective_x]) * 1.01,
                float(row[objective_y]) + 0.01,
                str(topology),
                fontsize=8,
            )
        ax.set_title(f"Trade-off: {objective_y} vs {objective_x}")
        ax.set_xlabel(objective_x)
        ax.set_ylabel(objective_y)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(frontier_plot, dpi=180)
        plt.close(fig)
        paths["pareto_tradeoff_png"] = str(frontier_plot)

    dist_plot = out_dir / "distance_mahalanobis.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_df = topology_df.sort_values("distance_mahalanobis_to_centroid")
    ax.bar(
        sorted_df.index.tolist(),
        sorted_df["distance_mahalanobis_to_centroid"].astype(float).to_numpy(),
        color=["#d62728" if bool(v) else "#1f77b4" for v in sorted_df["pareto_frontier"].tolist()],
    )
    ax.set_title("Mahalanobis Distance to Descriptor Centroid")
    ax.set_ylabel("Distance")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(dist_plot, dpi=180)
    plt.close(fig)
    paths["mahalanobis_distance_png"] = str(dist_plot)
    return paths


def _write_markdown_report(
    *,
    path: Path,
    topology_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    objectives: dict[str, str],
    plots: dict[str, str],
    umap_note: str | None,
) -> None:
    ranked = topology_df.sort_values("selection_rank")
    lines = [
        "# Topology Analysis Report",
        "",
        "## Objectives",
        "",
    ]
    for key, direction in objectives.items():
        lines.append(f"- `{key}`: `{direction}`")

    lines.extend(
        [
            "",
            "## Pareto Frontier",
            "",
            *(f"- `{item}`" for item in frontier_df.index.tolist()),
            "",
            "## Ranking (Ideal Point Distance)",
            "",
        ]
    )
    for topology, row in ranked.iterrows():
        lines.append(
            f"- `{int(row['selection_rank'])}. {topology}` "
            f"(d_ideal={float(row['d_ideal']):.4f}, frontier={bool(row['pareto_frontier'])})"
        )

    if plots:
        lines.extend(["", "## Generated Plots", ""])
        for key, value in plots.items():
            lines.append(f"- `{key}`: `{value}`")

    if umap_note:
        lines.extend(["", "## UMAP Note", "", f"- {umap_note}"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_objectives(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(
                f"Invalid objective specification '{piece}'. Expected format metric:max|metric:min."
            )
        key, direction = piece.split(":", 1)
        key = key.strip()
        direction = direction.strip().lower()
        if direction not in {"max", "min"}:
            raise ValueError(f"Invalid direction '{direction}' for objective '{key}'")
        out[key] = direction
    return out


def _parse_weights(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid weight specification '{piece}'. Expected metric=weight.")
        key, value = piece.split("=", 1)
        out[key.strip()] = float(value.strip())
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze topology experiment outputs (distance, Pareto frontier, embeddings)."
    )
    parser.add_argument("--input-root", required=True, help="Path to topology experiment root")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis artifacts (default: <input-root>/topology_analysis)",
    )
    parser.add_argument(
        "--descriptor-columns",
        default=",".join(DEFAULT_DESCRIPTOR_COLUMNS),
        help="Comma-separated descriptor columns for scaling/distances/embedding",
    )
    parser.add_argument(
        "--objectives",
        default=",".join(f"{key}:{value}" for key, value in DEFAULT_OBJECTIVES.items()),
        help="Comma-separated objective directions, e.g. eval_avg_score:max,C2_tokens_total:min",
    )
    parser.add_argument(
        "--objective-weights",
        default="",
        help="Optional objective weights, e.g. eval_avg_score=2,C2_tokens_total=1",
    )
    parser.add_argument("--include-umap", action="store_true", help="Also compute UMAP embedding")
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Use all timestamped runs per topology (default: latest run only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    descriptor_columns = [item.strip() for item in args.descriptor_columns.split(",") if item.strip()]
    objectives = _parse_objectives(args.objectives)
    weights = _parse_weights(args.objective_weights) if args.objective_weights else None

    result = analyze_topology_experiment(
        args.input_root,
        output_dir=args.output_dir,
        descriptor_columns=descriptor_columns,
        objectives=objectives,
        objective_weights=weights,
        include_umap=bool(args.include_umap),
        latest_only=not bool(args.all_runs),
    )
    print(result.artifacts["analysis_json"])
    print(result.artifacts["report_md"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

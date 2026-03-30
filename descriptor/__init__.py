from .descriptor import (
    DescriptorResult,
    compute_descriptor_from_runs,
    write_descriptor_csv,
    write_descriptor_json,
)
from .distances import covariance_inverse, mahalanobis_distance, pairwise_mahalanobis
from .embeddings import pca_2d, umap_2d
from .experiment import analyze_task_runs, write_run_trace
from .io import TraceLogger, read_trace_jsonl, write_trace_jsonl
from .metrics import ExtensionOptions, compute_run_metrics, compute_task_metrics
from .metrics import (
    COST_METRICS,
    DIAGNOSTIC_METRICS,
    PROCESS_METRICS,
    QUALITY_METRICS,
    RELIABILITY_METRICS,
    RunOutcome,
    resolve_run_outcome,
)
from .pareto import ideal_point_distance, normalize_objectives, pareto_frontier
from .scaling import RobustScaler, robust_scale
from .schema import EVENT_TYPES, TraceEvent, validate_event_dict, validate_trace_events
from .stages import EVENT_STAGE_MAP, STAGES, compute_stage_metrics, stage_for_event
from .topology_analysis import (
    TopologyAnalysisResult,
    TopologyRun,
    aggregate_topology_metrics,
    analyze_topology_experiment,
    discover_topology_runs,
    load_task_metrics,
)

__all__ = [
    "DescriptorResult",
    "compute_descriptor_from_runs",
    "write_descriptor_csv",
    "write_descriptor_json",
    "covariance_inverse",
    "mahalanobis_distance",
    "pairwise_mahalanobis",
    "pca_2d",
    "umap_2d",
    "analyze_task_runs",
    "write_run_trace",
    "TraceLogger",
    "read_trace_jsonl",
    "write_trace_jsonl",
    "ExtensionOptions",
    "RunOutcome",
    "compute_run_metrics",
    "compute_task_metrics",
    "resolve_run_outcome",
    "QUALITY_METRICS",
    "COST_METRICS",
    "DIAGNOSTIC_METRICS",
    "RELIABILITY_METRICS",
    "PROCESS_METRICS",
    "ideal_point_distance",
    "normalize_objectives",
    "pareto_frontier",
    "RobustScaler",
    "robust_scale",
    "EVENT_TYPES",
    "TraceEvent",
    "validate_event_dict",
    "validate_trace_events",
    "STAGES",
    "EVENT_STAGE_MAP",
    "compute_stage_metrics",
    "stage_for_event",
    "TopologyRun",
    "TopologyAnalysisResult",
    "discover_topology_runs",
    "load_task_metrics",
    "aggregate_topology_metrics",
    "analyze_topology_experiment",
]

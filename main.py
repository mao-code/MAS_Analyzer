"""Command-line entrypoint for the MAS_Analyzer experiment harness.

Subcommands: ``run``, ``list-benchmarks``, ``benchmark-info``, ``summarize-experiment``.

The implementation lives in the :mod:`cli` package. The re-exports below keep
``main.<helper>`` working for the scripts and tests that have long imported
internals from this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from benchmark import list_benchmarks
from cli.artifacts import (  # noqa: F401
    _classify_run_exception,
    _failed_run_result,
    _run_progress_summary,
    _summary_row_from_analysis,
    _summary_task_entry_from_payload,
    _task_manifest_payload,
    _write_eval,
    _write_raw_output,
    _write_run_artifacts,
    _write_task_checkpoint,
)
from cli.commands import (  # noqa: F401
    _build_skill_updater,
    benchmark_info_command,
    list_benchmarks_command,
    run_command,
    summarize_experiment_command,
)
from cli.common import (  # noqa: F401
    OutputPaths,
    _append_markdown_fence,
    _env_truthy,
    _log_progress,
    _mean,
    _normalized_int,
    _now_stamp,
    _parse_int_list,
    _parse_str_list,
    _prompt_preview,
    _redact_secrets,
    _text_preview,
    _write_json,
    _write_summary_csv,
)
from cli.graphs import (  # noqa: F401
    _matplotlib_positions,
    _write_matplotlib_graph_png,
    _write_system_graph_artifact,
    _write_workflow_matplotlib_graph_png,
)
from cli.resume import (  # noqa: F401
    _llm_payload_needs_rerun,
    _load_completed_run_resume,
    _load_completed_task_resume,
    _metadata_needs_rerun,
    _run_artifact_paths,
    _task_payload_needs_rerun,
)
from cli.settings import (  # noqa: F401
    _apply_benchmark_overrides,
    _apply_mas_overrides,
    _benchmark_section_config,
    _default_system_label,
    _experiment_settings_payload,
    _mas_mode_label,
    _resolve_output_paths,
    _runtime_tools,
    _write_experiment_settings,
)
from cli.trajectory import (  # noqa: F401
    _build_prompt_catalog,
    _collect_message_catalog,
    _compact_run_metadata,
    _fallback_interaction_logs,
    _render_trajectory_markdown,
    _stage_metric_payload,
    _trace_metrics_payload,
    _trajectory_payload,
)
from MAS.prompting_baselines import (
    BASELINE_DIRECT,
    PROMPTING_BASELINES,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MAS experiments against benchmark adapters and descriptor analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("--config", required=True, help="Path to experiment TOML config")
    run_parser.add_argument(
        "--benchmark",
        required=True,
        choices=list_benchmarks(),
        help="Benchmark adapter to run",
    )
    run_parser.add_argument("--task-limit", type=int, default=None)
    run_parser.add_argument(
        "--task-offset",
        type=int,
        default=0,
        help="Skip the first N tasks (in deterministic load order). Enables sharding a task "
        "set across parallel runs: shard i uses --task-offset i*L --task-limit L.",
    )
    run_parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task ids to run (e.g. '791,796'). Overrides --task-offset/"
        "--task-limit: loads all tasks and keeps only the listed ids, in load order. "
        "Useful for re-running specific hard/failed examples.",
    )
    run_parser.add_argument("--runs-per-task", type=int, default=None)
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument(
        "--output-layout",
        choices=["legacy", "hierarchical"],
        default="legacy",
        help="Output folder layout. 'hierarchical' writes experiment/benchmark/system/task.",
    )
    run_parser.add_argument("--experiment-id", default=None)
    run_parser.add_argument("--system-label", default=None)
    run_parser.add_argument(
        "--prompting-baseline",
        choices=sorted(PROMPTING_BASELINES),
        default=BASELINE_DIRECT,
        help=(
            "Fixed prompting baseline to run on top of the configured SAS runtime. "
            "Use cot, self_consistency, or self_refine for the paper table prompting rows."
        ),
    )
    run_parser.add_argument(
        "--self-consistency-samples",
        type=int,
        default=3,
        help="Number of internal samples for --prompting-baseline self_consistency.",
    )
    run_parser.add_argument(
        "--self-refine-rounds",
        type=int,
        default=3,
        help="Number of feedback/revision rounds for --prompting-baseline self_refine.",
    )
    run_parser.add_argument("--topology", default=None)
    run_parser.add_argument("--agents", type=int, default=None)
    run_parser.add_argument("--mas-rounds", type=int, default=None)
    run_parser.add_argument("--discussion-rounds", type=int, default=None)
    run_parser.add_argument("--communication-budget", type=int, default=None)
    run_parser.add_argument(
        "--termination-consensus-mode",
        choices=["llm_judge", "lexical"],
        default=None,
    )
    run_parser.add_argument(
        "--final-vote-mode",
        choices=["llm_judge", "deterministic"],
        default=None,
    )
    run_parser.add_argument("--default-model", default=None)
    run_parser.add_argument("--judge-model", default=None)
    run_parser.add_argument(
        "--benchmark-eval-judge-model",
        default=None,
        help="Override benchmark-side evaluation judge_model without changing the MAS internal judge model.",
    )
    run_parser.add_argument("--peer-artifact-max-chars", type=int, default=None)
    run_parser.add_argument("--agents-per-level", default=None)
    run_parser.add_argument("--group-sizes", default=None)
    run_parser.add_argument("--agent-types", default=None)
    run_parser.add_argument(
        "--no-dynamic-roles",
        dest="no_dynamic_roles",
        action="store_true",
        default=False,
        help="Disable LLM-based dynamic role assignment and use only structural roles.",
    )
    run_parser.add_argument(
        "--skill-update-batch-size",
        type=int,
        default=None,
        help=(
            "Self-evolved only: reflect the long-term skill online every N freshly executed "
            "runs (default from config = 12). 0 disables online updates (offline reflection "
            "only). Online updates rewrite a shared file mid-experiment, so the enclosing "
            "experiment must be a single sequential process."
        ),
    )
    run_parser.set_defaults(func=run_command)

    list_parser = subparsers.add_parser("list-benchmarks", help="List available benchmarks")
    list_parser.set_defaults(func=list_benchmarks_command)

    info_parser = subparsers.add_parser(
        "benchmark-info", help="Show benchmark requirements and setup notes"
    )
    info_parser.add_argument(
        "--benchmark",
        required=True,
        choices=list_benchmarks(),
    )
    info_parser.add_argument("--config", default=None)
    info_parser.set_defaults(func=benchmark_info_command)

    summarize_parser = subparsers.add_parser(
        "summarize-experiment",
        help="Aggregate hierarchical experiment outputs into benchmark and experiment summaries",
    )
    summarize_parser.add_argument("--experiment-root", required=True)
    summarize_parser.set_defaults(func=summarize_experiment_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

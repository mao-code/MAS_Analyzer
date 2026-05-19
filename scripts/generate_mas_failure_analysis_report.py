from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARKS = ("browsecomp", "plancraft", "stabletoolbench", "workbench")
EXCLUDED_BENCHMARKS = {"finance_agent"}
MODEL_ROOTS = (
    "artifacts/full_experiment/20260427T134706Z__google_gemma_4_31b_it_nitro",
    "artifacts/full_experiment/20260427T134706Z__openai_gpt_oss_120b",
)

MODEL_LABELS = {
    "20260427T134706Z__google_gemma_4_31b_it_nitro": "Gemma 4 31B IT Nitro",
    "20260427T134706Z__openai_gpt_oss_120b": "GPT-OSS 120B",
}

SYSTEM_LABELS = {
    "sas": "SAS baseline",
    "only_voting": "Only voting",
    "orchestrator_no_discussion": "Orchestrator, no discussion",
    "orchestrator_with_discussion": "Orchestrator with discussion",
    "orchestrator_tree_structure": "Orchestrator tree",
    "fully_linked_debate": "Fully linked debate",
    "group_chat_debate": "Group chat debate",
}

FAILURE_MODE_DEFS = {
    "retrieval_synthesis_gap": {
        "label": "Retrieved evidence, wrong synthesis",
        "kind": "run",
        "definition": (
            "BrowseComp runs where gold or evidence documents were retrieved, but the final answer "
            "was still incorrect."
        ),
    },
    "premature_consensus": {
        "label": "Premature consensus",
        "kind": "run",
        "definition": "Incorrect runs that stopped with consensus_reached.",
    },
    "stalled_discussion": {
        "label": "Stalled discussion",
        "kind": "run",
        "definition": "Incorrect runs that stopped because the controller judged no meaningful change.",
    },
    "branch_collapse": {
        "label": "Branch collapse",
        "kind": "run",
        "definition": "Incorrect runs whose topology stopped with invalid_or_failed_branch.",
    },
    "round_budget_exhausted": {
        "label": "Round budget exhausted",
        "kind": "run",
        "definition": "Incorrect runs that reached max_rounds before a correct final answer.",
    },
    "aggregation_lock_in": {
        "label": "Aggregation lock-in",
        "kind": "run",
        "definition": "Incorrect voting or tie-break outcomes that selected a shared wrong answer.",
    },
    "tool_error_cascade": {
        "label": "Tool error cascade",
        "kind": "run",
        "definition": "Incorrect runs with one or more recorded tool failures.",
    },
    "false_impossible_state": {
        "label": "False impossible/state error",
        "kind": "run",
        "definition": "PlanCraft runs where the task metadata says possible but the answer says impossible.",
    },
    "tool_selection_blind_spot": {
        "label": "Tool selection blind spot",
        "kind": "run",
        "definition": "Workbench runs where required domain tools were absent from the predicted function calls.",
    },
    "format_contract_leak": {
        "label": "Format/contract leak",
        "kind": "run",
        "definition": "Final answers that leaked analysis or planning text instead of the requested answer.",
    },
    "wrong_answer_or_incomplete_execution": {
        "label": "Wrong answer or incomplete execution",
        "kind": "run",
        "definition": "Incorrect completed runs not captured by a narrower heuristic.",
    },
    "coordination_cost_dominance": {
        "label": "Coordination cost dominates gain",
        "kind": "system",
        "definition": (
            "System-level MAS rows with at least 5x SAS tokens and <=3 point success-rate lift."
        ),
    },
}

TOPOLOGY_ANALYSIS_TEXT = {
    "only_voting": {
        "evidence": (
            "Independent workers are cheap relative to debate topologies and have zero communication, "
            "but majority vote can lock in a shared misconception."
        ),
        "risk": "No worker can see or repair another worker's evidence gap before aggregation.",
        "solution": "Use voting as the default low-cost MAS, but escalate to discussion only when voters disagree or confidence is low.",
    },
    "orchestrator_no_discussion": {
        "evidence": (
            "Specialists report to one coordinator, keeping communication bounded and avoiding peer chatter."
        ),
        "risk": "The orchestrator is a single synthesis bottleneck; specialist evidence can be compressed away.",
        "solution": "Require the merge step to carry source-grounded claims and unresolved issues forward explicitly.",
    },
    "orchestrator_with_discussion": {
        "evidence": (
            "This topology often adds the largest communication and handoff budget, but is not consistently the best system."
        ),
        "risk": "Discussion can amplify repeated tool failures or repeated weak claims rather than adding new evidence.",
        "solution": "Gate additional discussion rounds on new evidence, not merely semantic disagreement.",
    },
    "orchestrator_tree_structure": {
        "evidence": (
            "Tree routing enforces hierarchy and can help divide work, but root and manager reducers become evidence bottlenecks."
        ),
        "risk": "Leaf findings must survive multiple summarization hops before final selection.",
        "solution": "Attach compact citations, tool outputs, and branch confidence to every upward packet.",
    },
    "fully_linked_debate": {
        "evidence": (
            "All-to-all peer visibility is the strongest GPT-OSS topology on three of four non-finance benchmarks."
        ),
        "risk": "It buys quality with high token and handoff cost, and still stops incorrectly on some consensus/no-progress cases.",
        "solution": "Use for high-value ambiguous tasks after cheaper modes fail or disagree.",
    },
    "group_chat_debate": {
        "evidence": (
            "Local group debate plus representative exchange is strong on some retrieval tasks, especially BrowseComp under GPT-OSS."
        ),
        "risk": "Representative summaries can carry a group-level false negative into final consensus.",
        "solution": "Require representatives to pass unresolved evidence and dissent, not just the dominant local answer.",
    },
}

BENCHMARK_TEXT = {
    "browsecomp": {
        "label": "BrowseComp",
        "evidence": (
            "Retrieval-heavy multi-hop questions show many completed wrong answers; some failures retrieved gold evidence "
            "but failed to synthesize the final entity."
        ),
        "risk": "Search recall and final synthesis are separate bottlenecks.",
        "solution": "Use retrieval verification checklists and force final answers to cite which criteria are satisfied.",
    },
    "plancraft": {
        "label": "PlanCraft",
        "evidence": (
            "Failures are often short action/state mistakes, including false impossible decisions on possible tasks."
        ),
        "risk": "Debate can be overkill when one grammar-valid next action is needed.",
        "solution": "Prefer deterministic recipe/state validators or LLM-as-tool checks before invoking large MAS discussion.",
    },
    "stabletoolbench": {
        "label": "StableToolBench",
        "evidence": (
            "Most rows solve successfully, but failures cluster around tool/API failures and repeated unsuccessful calls."
        ),
        "risk": "Agents can spend large budgets retrying the same failing endpoint variants.",
        "solution": "Add tool-error memory, call deduplication, and early fallback once cache/API errors repeat.",
    },
    "workbench": {
        "label": "WorkBench",
        "evidence": (
            "Cross-domain tasks expose tool-selection and side-effect sequencing failures; low overall success persists even with MAS."
        ),
        "risk": "The correct answer is often a precise sequence of state-changing calls across domains.",
        "solution": "Route by required domain coverage, then verify that the planned function set includes every ground-truth domain.",
    },
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value_float):
        return default
    return value_float


def as_bool_success(data: dict[str, Any]) -> bool | None:
    candidates = (
        data.get("outcome", {}).get("success"),
        data.get("success"),
        data.get("evaluation", {}).get("success"),
    )
    for candidate in candidates:
        if isinstance(candidate, bool):
            return candidate
    return None


def as_score(data: dict[str, Any]) -> float:
    for candidate in (
        data.get("outcome", {}).get("score"),
        data.get("score"),
        data.get("evaluation", {}).get("score"),
    ):
        if candidate is not None:
            return as_float(candidate)
    return 0.0


def truncate(text: Any, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def prompt_text(task: dict[str, Any]) -> str:
    prompt = task.get("prompt")
    if isinstance(prompt, list):
        return " ".join(str(item.get("content", "")) for item in prompt if isinstance(item, dict))
    return str(prompt or task.get("metadata", {}).get("query") or "")


def first_tool_error_preview(trace_path: Path) -> str:
    if not trace_path.exists():
        return ""
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = event.get("payload", {})
            if event.get("event_type") == "tool_result" and payload.get("status") == "error":
                return truncate(payload.get("output_preview", ""), 360)
    return ""


def source_paths(eval_path: Path) -> dict[str, str]:
    run_prefix = eval_path.name.replace(".eval.json", "")
    task_dir = eval_path.parent
    return {
        "eval": rel(eval_path),
        "trace_metrics": rel(task_dir / f"{run_prefix}.trace_metrics.json"),
        "trajectory": rel(task_dir / f"{run_prefix}.trajectory.md"),
        "task": rel(task_dir / "task.json"),
    }


def load_system_rows(model_roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in model_roots:
        model_id = root.name
        csv_path = root / "Plot" / "system_level_metrics.csv"
        for row in read_csv(csv_path):
            if row.get("benchmark") in EXCLUDED_BENCHMARKS:
                continue
            record = {
                "model_id": model_id,
                "model": MODEL_LABELS.get(model_id, model_id),
                "benchmark": row.get("benchmark", ""),
                "system_label": row.get("system_label", ""),
                "system": SYSTEM_LABELS.get(row.get("system_label", ""), row.get("system_label", "")),
                "task_count": int(as_float(row.get("task_count"))),
                "avg_success_rate": as_float(row.get("avg_success_rate")),
                "avg_eval_score": as_float(row.get("avg_eval_score")),
                "avg_stability": as_float(row.get("avg_stability")),
                "avg_tokens_total": as_float(row.get("avg_tokens_total")),
                "avg_cost_per_success": as_float(row.get("avg_cost_per_success")),
                "avg_tool_calls_total": as_float(row.get("avg_tool_calls_total")),
                "avg_tool_error_rate": as_float(row.get("avg_tool_error_rate")),
                "avg_communication_count": as_float(row.get("avg_communication_count")),
                "avg_handoff_count": as_float(row.get("avg_handoff_count")),
                "avg_latency_e2e": as_float(row.get("avg_latency_e2e")),
                "avg_pass_at_1": as_float(row.get("avg_pass_at_1")),
                "avg_pass_at_3": as_float(row.get("avg_pass_at_3")),
            }
            rows.append(record)
    return rows


def with_vs_sas(system_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sas = {
        (row["model_id"], row["benchmark"]): row
        for row in system_rows
        if row["system_label"] == "sas"
    }
    out: list[dict[str, Any]] = []
    for row in system_rows:
        baseline = sas.get((row["model_id"], row["benchmark"]))
        record = dict(row)
        if baseline and row["system_label"] != "sas":
            sas_tokens = baseline["avg_tokens_total"]
            record["success_delta_vs_sas"] = row["avg_success_rate"] - baseline["avg_success_rate"]
            record["tokens_delta_vs_sas"] = row["avg_tokens_total"] - sas_tokens
            record["token_multiplier_vs_sas"] = (
                row["avg_tokens_total"] / sas_tokens if sas_tokens > 0 else None
            )
        else:
            record["success_delta_vs_sas"] = None
            record["tokens_delta_vs_sas"] = None
            record["token_multiplier_vs_sas"] = None
        out.append(record)
    return out


def load_runs(model_roots: list[Path]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for root in model_roots:
        model_id = root.name
        for eval_path in root.glob("*/*/*/run_*.eval.json"):
            rel_parts = eval_path.relative_to(root).parts
            if len(rel_parts) < 4:
                continue
            benchmark, topology, task_id, filename = rel_parts[:4]
            if benchmark in EXCLUDED_BENCHMARKS or benchmark not in BENCHMARKS:
                continue
            run_index_match = re.search(r"run_(\d+)\.eval\.json$", filename)
            run_index = int(run_index_match.group(1)) if run_index_match else 0
            eval_data = read_json(eval_path)
            task_path = eval_path.parent / "task.json"
            task = read_json(task_path) if task_path.exists() else {}
            trace_path = eval_path.with_name(filename.replace(".eval.json", ".trace_metrics.json"))
            trace = read_json(trace_path) if trace_path.exists() else {}
            details = eval_data.get("details") or eval_data.get("evaluation", {}).get("details", {})
            runtime = (
                details.get("run_metadata_summary")
                or details.get("run_metadata")
                or trace.get("runtime")
                or {}
            )
            metrics = trace.get("metrics") or {}
            task_metadata = task.get("metadata", {})
            retrieval = details.get("retrieval", {})
            prediction_calls = details.get("prediction") if isinstance(details.get("prediction"), list) else []
            run = {
                "model_id": model_id,
                "model": MODEL_LABELS.get(model_id, model_id),
                "benchmark": benchmark,
                "topology": topology,
                "system_label": topology,
                "system": SYSTEM_LABELS.get(topology, topology),
                "task_id": task_id,
                "run_index": run_index,
                "success": as_bool_success(eval_data),
                "score": as_score(eval_data),
                "completion": bool(eval_data.get("outcome", {}).get("completion", eval_data.get("completion", False))),
                "prediction": eval_data.get("prediction") or details.get("prediction") or "",
                "reference_answer": (
                    details.get("reference_answer")
                    or task.get("reference_answer")
                    or details.get("ground_truth")
                    or task_metadata.get("answer")
                    or ""
                ),
                "ground_truth": details.get("ground_truth") or task_metadata.get("answer") or [],
                "prompt": prompt_text(task),
                "final_reason": runtime.get("final_reason", ""),
                "vote_tally": runtime.get("vote_tally", {}),
                "messages_sent_total": as_float(runtime.get("messages_sent_total")),
                "tool_call_counts": runtime.get("tool_call_counts", {}),
                "function_calls": runtime.get("function_calls", []),
                "predicted_calls": prediction_calls,
                "tool_calls_total": as_float(metrics.get("tool_calls_total", runtime.get("tool_calls_total"))),
                "tool_fail_total": as_float(metrics.get("tool_fail_total")),
                "tokens_total": as_float(metrics.get("tokens_total", metrics.get("token_total"))),
                "steps_total": as_float(metrics.get("steps_total")),
                "communication_count": as_float(metrics.get("communication_count")),
                "handoff_count": as_float(metrics.get("handoff_count")),
                "loop_score": as_float(metrics.get("loop_score")),
                "verification_density": as_float(metrics.get("verification_density")),
                "recall_gold": as_float(retrieval.get("recall_gold")),
                "recall_evidence": as_float(retrieval.get("recall_evidence")),
                "retrieved_docids": retrieval.get("retrieved_docids") or runtime.get("retrieved_docids") or [],
                "gold_docids": task_metadata.get("gold_docids") or details.get("task_metadata", {}).get("gold_docids") or [],
                "evidence_docids": task_metadata.get("evidence_docids")
                or details.get("task_metadata", {}).get("evidence_docids")
                or [],
                "task_metadata": task_metadata,
                "eval_path": rel(eval_path),
                "trace_metrics_path": rel(trace_path),
                "trajectory_path": rel(eval_path.with_name(filename.replace(".eval.json", ".trajectory.md"))),
                "task_path": rel(task_path),
                "absolute_eval_path": str(eval_path),
                "absolute_trace_jsonl_path": str(eval_path.with_name(filename.replace(".eval.json", ".trace.jsonl"))),
                "termination": trace.get("termination", {}),
            }
            run["failure_modes"] = classify_failure(run)
            runs.append(run)
    return runs


def classify_failure(run: dict[str, Any]) -> list[str]:
    if run.get("success") is not False:
        return []
    modes: list[str] = []
    final_reason = str(run.get("final_reason") or "")
    prediction = str(run.get("prediction") or "")
    prediction_lower = prediction.lower()

    if "consensus_reached" in final_reason:
        modes.append("premature_consensus")
    if "no_meaningful_change" in final_reason:
        modes.append("stalled_discussion")
    if "invalid_or_failed_branch" in final_reason:
        modes.append("branch_collapse")
    if "max_rounds_reached" in final_reason:
        modes.append("round_budget_exhausted")
    if "majority_vote" in final_reason or "deterministic_tiebreak" in final_reason:
        modes.append("aggregation_lock_in")
    if as_float(run.get("tool_fail_total")) > 0:
        modes.append("tool_error_cascade")

    retrieved = set(map(str, run.get("retrieved_docids") or []))
    gold = set(map(str, run.get("gold_docids") or []))
    evidence = set(map(str, run.get("evidence_docids") or []))
    if run.get("benchmark") == "browsecomp" and (
        as_float(run.get("recall_gold")) > 0
        or bool(retrieved.intersection(gold))
        or bool(retrieved.intersection(evidence))
    ):
        modes.append("retrieval_synthesis_gap")

    if prediction_lower.startswith("analysis") or prediction_lower.startswith("we need to"):
        modes.append("format_contract_leak")

    if (
        run.get("benchmark") == "plancraft"
        and "impossible" in prediction_lower
        and run.get("task_metadata", {}).get("impossible") is False
    ):
        modes.append("false_impossible_state")

    if run.get("benchmark") == "workbench":
        ground_truth = " ".join(map(str, run.get("ground_truth") or []))
        predicted = " ".join(map(str, run.get("predicted_calls") or run.get("function_calls") or []))
        if "customer_relationship_manager" in ground_truth and "customer_relationship_manager" not in predicted:
            modes.append("tool_selection_blind_spot")
        if "calendar.create_event" in ground_truth and "calendar.create_event" not in predicted:
            modes.append("tool_selection_blind_spot")

    if not modes:
        modes.append("wrong_answer_or_incomplete_execution")
    return modes


def build_failure_mode_summary(
    runs: list[dict[str, Any]], system_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    failed_runs = [run for run in runs if run.get("success") is False]
    counter: Counter[str] = Counter()
    by_mode_examples: dict[str, dict[str, Any]] = {}
    for run in failed_runs:
        for mode in run.get("failure_modes", []):
            counter[mode] += 1
            by_mode_examples.setdefault(mode, run)

    high_overhead = [
        row
        for row in system_rows
        if row["system_label"] != "sas"
        and row.get("token_multiplier_vs_sas") is not None
        and row["token_multiplier_vs_sas"] >= 5.0
        and row["success_delta_vs_sas"] <= 0.03
    ]
    counter["coordination_cost_dominance"] = len(high_overhead)
    if high_overhead:
        by_mode_examples["coordination_cost_dominance"] = max(
            high_overhead, key=lambda row: row["token_multiplier_vs_sas"] or 0
        )

    summaries = []
    for mode, meta in FAILURE_MODE_DEFS.items():
        example = by_mode_examples.get(mode, {})
        example_text = ""
        if meta["kind"] == "run" and example:
            example_text = (
                f"{example.get('model')} / {example.get('benchmark')} / "
                f"{example.get('system')} / {example.get('task_id')} run {example.get('run_index')}"
            )
        elif meta["kind"] == "system" and example:
            example_text = (
                f"{example.get('model')} / {example.get('benchmark')} / "
                f"{example.get('system')}: {example.get('token_multiplier_vs_sas'):.1f}x SAS tokens, "
                f"{example.get('success_delta_vs_sas'):+.3f} success delta"
            )
        summaries.append(
            {
                "id": mode,
                "label": meta["label"],
                "kind": meta["kind"],
                "count": counter.get(mode, 0),
                "definition": meta["definition"],
                "example": example_text,
            }
        )
    return summaries


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def build_topology_profiles(system_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_topology: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures_by_topology: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in system_rows:
        if row["system_label"] != "sas":
            rows_by_topology[row["system_label"]].append(row)
    for run in runs:
        if run.get("success") is False and run["system_label"] != "sas":
            failures_by_topology[run["system_label"]].append(run)

    profiles = []
    for topology, rows in sorted(rows_by_topology.items()):
        if not rows:
            continue
        deltas = [row["success_delta_vs_sas"] for row in rows if row.get("success_delta_vs_sas") is not None]
        multipliers = [
            row["token_multiplier_vs_sas"] for row in rows if row.get("token_multiplier_vs_sas") is not None
        ]
        best_count = 0
        for model_id in {row["model_id"] for row in rows}:
            for benchmark in BENCHMARKS:
                candidates = [
                    row
                    for row in system_rows
                    if row["model_id"] == model_id
                    and row["benchmark"] == benchmark
                    and row["system_label"] != "sas"
                ]
                if candidates and max(candidates, key=lambda row: row["avg_success_rate"])["system_label"] == topology:
                    best_count += 1
        mode_counts = Counter(
            mode for run in failures_by_topology[topology] for mode in run.get("failure_modes", [])
        )
        text = TOPOLOGY_ANALYSIS_TEXT.get(topology, {})
        profiles.append(
            {
                "topology": topology,
                "label": SYSTEM_LABELS.get(topology, topology),
                "rows": len(rows),
                "best_count": best_count,
                "avg_success": sum(row["avg_success_rate"] for row in rows) / len(rows),
                "avg_success_delta_vs_sas": sum(deltas) / len(deltas) if deltas else None,
                "median_token_multiplier_vs_sas": median(multipliers) if multipliers else None,
                "avg_tool_error_rate": sum(row["avg_tool_error_rate"] for row in rows) / len(rows),
                "avg_communication_count": sum(row["avg_communication_count"] for row in rows) / len(rows),
                "avg_handoff_count": sum(row["avg_handoff_count"] for row in rows) / len(rows),
                "failures": len(failures_by_topology[topology]),
                "top_failure_modes": mode_counts.most_common(3),
                "evidence": text.get("evidence", ""),
                "risk": text.get("risk", ""),
                "solution": text.get("solution", ""),
            }
        )
    return profiles


def build_benchmark_profiles(system_rows: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for benchmark in BENCHMARKS:
        rows = [row for row in system_rows if row["benchmark"] == benchmark]
        failed = [run for run in runs if run["benchmark"] == benchmark and run.get("success") is False]
        sas_rows = [row for row in rows if row["system_label"] == "sas"]
        best_rows = []
        for model_id in {row["model_id"] for row in rows}:
            candidates = [row for row in rows if row["model_id"] == model_id and row["system_label"] != "sas"]
            if candidates:
                best_rows.append(max(candidates, key=lambda row: row["avg_success_rate"]))
        mode_counts = Counter(mode for run in failed for mode in run.get("failure_modes", []))
        text = BENCHMARK_TEXT[benchmark]
        profiles.append(
            {
                "benchmark": benchmark,
                "label": text["label"],
                "sas_success_avg": sum(row["avg_success_rate"] for row in sas_rows) / len(sas_rows),
                "best_success_avg": sum(row["avg_success_rate"] for row in best_rows) / len(best_rows),
                "best_gain_avg": sum(row["success_delta_vs_sas"] for row in best_rows) / len(best_rows),
                "best_topologies": [
                    {
                        "model": row["model"],
                        "topology": SYSTEM_LABELS.get(row["system_label"], row["system_label"]),
                        "success": row["avg_success_rate"],
                        "gain": row["success_delta_vs_sas"],
                        "tokens": row["avg_tokens_total"],
                    }
                    for row in best_rows
                ],
                "failed_runs": len(failed),
                "top_failure_modes": mode_counts.most_common(4),
                "evidence": text["evidence"],
                "risk": text["risk"],
                "solution": text["solution"],
            }
        )
    return profiles


def build_model_profiles(system_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for model_id in sorted({row["model_id"] for row in system_rows}):
        rows = [row for row in system_rows if row["model_id"] == model_id]
        best_rows = []
        sas_rows = []
        for benchmark in BENCHMARKS:
            sas_rows.extend(
                row for row in rows if row["benchmark"] == benchmark and row["system_label"] == "sas"
            )
            candidates = [
                row for row in rows if row["benchmark"] == benchmark and row["system_label"] != "sas"
            ]
            if candidates:
                best_rows.append(max(candidates, key=lambda row: row["avg_success_rate"]))
        profiles.append(
            {
                "model_id": model_id,
                "model": MODEL_LABELS.get(model_id, model_id),
                "sas_success_avg": sum(row["avg_success_rate"] for row in sas_rows) / len(sas_rows),
                "best_mas_success_avg": sum(row["avg_success_rate"] for row in best_rows) / len(best_rows),
                "best_gain_avg": sum(row["success_delta_vs_sas"] for row in best_rows) / len(best_rows),
                "median_best_token_multiplier": median(
                    [row["token_multiplier_vs_sas"] for row in best_rows if row["token_multiplier_vs_sas"]]
                ),
                "best_rows": [
                    {
                        "benchmark": row["benchmark"],
                        "topology": SYSTEM_LABELS.get(row["system_label"], row["system_label"]),
                        "success": row["avg_success_rate"],
                        "sas_delta": row["success_delta_vs_sas"],
                        "token_multiplier": row["token_multiplier_vs_sas"],
                    }
                    for row in best_rows
                ],
            }
        )
    return profiles


def find_run(runs: list[dict[str, Any]], suffix: str) -> dict[str, Any]:
    for run in runs:
        if run["eval_path"].endswith(suffix):
            return run
    raise ValueError(f"Could not find run ending with {suffix}")


def build_case_studies(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_specs = [
        {
            "suffix": "20260427T134706Z__openai_gpt_oss_120b/browsecomp/group_chat_debate/769/run_0.eval.json",
            "title": "BrowseComp: consensus around a false negative despite retrieved evidence",
            "mode": "retrieval_synthesis_gap",
            "hypothesis": (
                "The group summaries preserved the lack-of-proof conclusion more strongly than the retrieved document "
                "IDs, so final selection favored caution over the reference entity."
            ),
        },
        {
            "suffix": "20260427T134706Z__openai_gpt_oss_120b/stabletoolbench/orchestrator_with_discussion/1572/run_1.eval.json",
            "title": "StableToolBench: discussion repeated cache-miss tool calls until no progress",
            "mode": "tool_error_cascade",
            "hypothesis": (
                "The orchestrator loop lacked shared negative-result memory, so specialists retried parameter variants "
                "instead of stopping after repeated cache misses."
            ),
        },
        {
            "suffix": "20260427T134706Z__openai_gpt_oss_120b/plancraft/only_voting/VAL0049/run_1.eval.json",
            "title": "PlanCraft: independent voters unanimously chose a false impossible state",
            "mode": "false_impossible_state",
            "hypothesis": (
                "All workers applied a remembered recipe/resource rule without a deterministic state validator, and "
                "majority vote had no dissent to surface."
            ),
        },
        {
            "suffix": "20260427T134706Z__openai_gpt_oss_120b/workbench/only_voting/multi_domain_16/run_0.eval.json",
            "title": "WorkBench: required CRM side effect was missed by tool selection",
            "mode": "tool_selection_blind_spot",
            "hypothesis": (
                "The worker focused on the project-management subgoal and failed to route the second half of the task "
                "to the CRM tool namespace."
            ),
        },
    ]
    cases = []
    for spec in case_specs:
        run = find_run(runs, spec["suffix"])
        trace_jsonl = Path(run["absolute_trace_jsonl_path"])
        task_meta = run.get("task_metadata", {})
        facts = [
            f"Outcome: success={run['success']}, score={run['score']:.1f}, final_reason={run['final_reason'] or 'n/a'}.",
            f"Runtime: {fmt_num(run['tokens_total'], 0)} tokens, {fmt_num(run['tool_calls_total'], 0)} tool calls, {fmt_num(run['tool_fail_total'], 0)} tool failures, {fmt_num(run['messages_sent_total'], 0)} messages.",
        ]
        if run["benchmark"] == "browsecomp":
            facts.append(
                f"Evaluator reference answer: {truncate(run['reference_answer'], 120)}; retrieval recall_gold={run['recall_gold']:.3f}, recall_evidence={run['recall_evidence']:.3f}."
            )
            facts.append(
                f"Retrieved/gold overlap: {len(set(map(str, run['retrieved_docids'])).intersection(map(str, run['gold_docids'])))} gold doc IDs retrieved."
            )
        if run["benchmark"] == "stabletoolbench":
            facts.append(
                f"Task metadata says cache_ready={task_meta.get('cache_ready')} and missing_cache_count={task_meta.get('missing_cache_count')}."
            )
            error = first_tool_error_preview(trace_jsonl)
            if error:
                facts.append(f"First recorded tool error: {error}")
        if run["benchmark"] == "plancraft":
            facts.append(
                f"Task metadata: impossible={task_meta.get('impossible')}, optimal_path_length={task_meta.get('optimal_path_length')}, target={task_meta.get('target')}."
            )
            facts.append(f"Vote tally: {truncate(run['vote_tally'], 180)}")
        if run["benchmark"] == "workbench":
            facts.append(f"Ground truth call: {truncate(run['ground_truth'], 260)}")
            facts.append(f"Predicted calls: {truncate(run['predicted_calls'], 260)}")
        cases.append(
            {
                "title": spec["title"],
                "primary_mode": spec["mode"],
                "failure_modes": run["failure_modes"],
                "model": run["model"],
                "benchmark": run["benchmark"],
                "topology": run["system"],
                "task_id": run["task_id"],
                "run_index": run["run_index"],
                "prompt": truncate(run["prompt"], 520),
                "prediction": truncate(run["prediction"], 520),
                "reference": truncate(run["reference_answer"], 360),
                "facts": facts,
                "hypothesis": spec["hypothesis"],
                "paths": source_paths(Path(run["absolute_eval_path"])),
            }
        )
    return cases


def build_evidence_rows(runs: list[dict[str, Any]], limit: int = 140) -> list[dict[str, Any]]:
    failed = [run for run in runs if run.get("success") is False]
    def score(run: dict[str, Any]) -> float:
        modes = set(run.get("failure_modes", []))
        value = 0.0
        value += min(run["tool_fail_total"], 60.0) / 3.0
        value += min(run["tokens_total"], 350000.0) / 50000.0
        value += 8.0 if "retrieval_synthesis_gap" in modes else 0.0
        value += 6.0 if "tool_selection_blind_spot" in modes else 0.0
        value += 6.0 if "false_impossible_state" in modes else 0.0
        value += 4.0 if "format_contract_leak" in modes else 0.0
        value += 2.0 if "premature_consensus" in modes else 0.0
        return value

    selected = sorted(failed, key=score, reverse=True)[:limit]
    return [
        {
            "model": run["model"],
            "benchmark": run["benchmark"],
            "topology": run["system"],
            "task_id": run["task_id"],
            "run_index": run["run_index"],
            "failure_modes": run["failure_modes"],
            "final_reason": run["final_reason"],
            "score": run["score"],
            "tokens_total": run["tokens_total"],
            "tool_calls_total": run["tool_calls_total"],
            "tool_fail_total": run["tool_fail_total"],
            "prediction": truncate(run["prediction"], 220),
            "reference": truncate(run["reference_answer"], 180),
            "eval_path": run["eval_path"],
            "trajectory_path": run["trajectory_path"],
        }
        for run in selected
    ]


def build_payload(model_roots: list[Path]) -> dict[str, Any]:
    system_rows = with_vs_sas(load_system_rows(model_roots))
    runs = load_runs(model_roots)
    failed_runs = [run for run in runs if run.get("success") is False]
    best_pairs = []
    for model_id in sorted({row["model_id"] for row in system_rows}):
        for benchmark in BENCHMARKS:
            candidates = [
                row
                for row in system_rows
                if row["model_id"] == model_id
                and row["benchmark"] == benchmark
                and row["system_label"] != "sas"
            ]
            if candidates:
                best_pairs.append(max(candidates, key=lambda row: row["avg_success_rate"]))
    best_gains = [row["success_delta_vs_sas"] for row in best_pairs]
    best_multipliers = [row["token_multiplier_vs_sas"] for row in best_pairs if row["token_multiplier_vs_sas"]]

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "source_instruction": "docs/qualitative_analysis.md",
        "requested_instruction": "docs/mas_failure_analysis_goal.md",
        "model_roots": [rel(path) for path in model_roots],
        "excluded_benchmarks": sorted(EXCLUDED_BENCHMARKS),
        "benchmarks": list(BENCHMARKS),
        "counts": {
            "models": len(model_roots),
            "benchmarks": len(BENCHMARKS),
            "system_rows": len(system_rows),
            "runs": len(runs),
            "failed_runs": len(failed_runs),
            "completed_runs": sum(1 for run in runs if run.get("completion")),
            "best_mas_beats_sas_pairs": sum(1 for gain in best_gains if gain and gain > 0),
            "model_benchmark_pairs": len(best_pairs),
            "median_best_gain": median(best_gains) if best_gains else 0.0,
            "median_best_token_multiplier": median(best_multipliers) if best_multipliers else 0.0,
        },
        "system_rows": system_rows,
        "failure_modes": build_failure_mode_summary(runs, system_rows),
        "topology_profiles": build_topology_profiles(system_rows, runs),
        "benchmark_profiles": build_benchmark_profiles(system_rows, runs),
        "model_profiles": build_model_profiles(system_rows),
        "case_studies": build_case_studies(runs),
        "evidence_rows": build_evidence_rows(runs),
    }
    return payload


def render_metric_cards(payload: dict[str, Any]) -> str:
    counts = payload["counts"]
    cards = [
        ("Included runs", f"{counts['runs']:,}", "Non-FinanceAgent run-level eval artifacts"),
        ("Failed runs", f"{counts['failed_runs']:,}", "Completed but benchmark-incorrect or unsolved"),
        (
            "Best MAS beats SAS",
            f"{counts['best_mas_beats_sas_pairs']}/{counts['model_benchmark_pairs']}",
            "Model/benchmark pairs where the best MAS exceeds SAS success",
        ),
        (
            "Median best-MAS lift",
            pct(counts["median_best_gain"]),
            "Median success-rate delta vs SAS among best MAS rows",
        ),
        (
            "Median best-MAS cost",
            f"{counts['median_best_token_multiplier']:.1f}x",
            "Median token multiplier vs SAS among best MAS rows",
        ),
    ]
    return "\n".join(
        f"""
        <article class="metric-card">
          <div class="metric-label">{escape(label)}</div>
          <div class="metric-value">{escape(value)}</div>
          <p>{escape(desc)}</p>
        </article>
        """
        for label, value, desc in cards
    )


def render_static_sections(payload: dict[str, Any]) -> str:
    cards = render_metric_cards(payload)
    return f"""
    <section id="summary" class="section">
      <div class="section-kicker">Executive Summary</div>
      <h2>MAS helps, but failure is usually a process problem rather than a completion problem.</h2>
      <div class="metric-grid">{cards}</div>
      <div class="callout-grid">
        <div class="callout evidence">
          <h3>Evidence-backed findings</h3>
          <ul>
            <li>The best MAS topology beats SAS in {payload['counts']['best_mas_beats_sas_pairs']} of {payload['counts']['model_benchmark_pairs']} non-finance model/benchmark pairs, but the median best-MAS token cost is {payload['counts']['median_best_token_multiplier']:.1f}x the SAS baseline.</li>
            <li>BrowseComp failures include runs where gold or evidence documents were retrieved but final synthesis still missed the answer.</li>
            <li>StableToolBench failures can be dominated by repeated tool errors; the selected case study records 79 tool failures in one run.</li>
            <li>Workbench failures expose tool-routing gaps: some runs call only one domain's tools despite ground truth requiring another domain.</li>
          </ul>
        </div>
        <div class="callout hypothesis">
          <h3>Interpretive hypotheses</h3>
          <ul>
            <li>Bounded relay packets reduce cost, but can drop decisive evidence unless packets carry source-backed claims and unresolved issues.</li>
            <li>Static topology choice is a poor fit: easy deterministic tasks need validators; ambiguous retrieval needs evidence-focused debate; tool failures need early fallback.</li>
            <li>More discussion helps only when it introduces new evidence or checks a concrete failure condition.</li>
          </ul>
        </div>
      </div>
    </section>

    <section id="method" class="section">
      <div class="section-kicker">Methodology</div>
      <h2>What was analyzed</h2>
      <p>This report is generated from the two requested experiment roots and excludes FinanceAgent. It uses system-level CSV summaries for quantitative comparisons, then reads per-run eval, trace metrics, task metadata, and selected trajectories for qualitative evidence. The available goal spec in this checkout is <code>docs/qualitative_analysis.md</code>; it matches the requested report contents, while <code>docs/mas_failure_analysis_goal.md</code> is not present.</p>
      <div class="method-grid">
        <div><strong>Quantitative evidence</strong><span>Plot/system_level_metrics.csv, run trace_metrics.json, eval.json</span></div>
        <div><strong>Qualitative evidence</strong><span>trajectory.md/json, task.json, eval details, tool error previews</span></div>
        <div><strong>Failure labels</strong><span>Heuristic, auditable labels derived from final_reason, evaluator details, tool failures, and task metadata</span></div>
        <div><strong>Constraint</strong><span>No FinanceAgent rows; evidence-backed findings are separated from hypotheses</span></div>
      </div>
    </section>
    """


def render_recommendation_sections() -> str:
    return """
    <section id="why" class="section">
      <div class="section-kicker">Why MAS Fails</div>
      <h2>A causal view from the traces</h2>
      <div class="why-grid">
        <article>
          <h3>1. Evidence does not guarantee synthesis</h3>
          <p>BrowseComp shows a gap between retrieval and answer formation. A run can retrieve relevant or gold documents and still converge on a false negative or unrelated entity.</p>
        </article>
        <article>
          <h3>2. Communication compresses both signal and dissent</h3>
          <p>Topologies pass bounded packets, not full transcripts. This keeps costs manageable, but decisive evidence, uncertainty, or minority objections can be lost before the final judge.</p>
        </article>
        <article>
          <h3>3. Consensus is not correctness</h3>
          <p>Many incorrect runs stop with consensus_reached or majority_vote. Agreement is a workflow-control signal, not an evaluator signal.</p>
        </article>
        <article>
          <h3>4. Tool errors compound under discussion</h3>
          <p>When agents independently retry failing tools, discussion can multiply API/cache failures instead of recovering.</p>
        </article>
        <article>
          <h3>5. Some tasks need validators more than agents</h3>
          <p>PlanCraft failures often involve action grammar, inventory state, and recipe constraints. These are better checked by deterministic validators than by longer debate.</p>
        </article>
        <article>
          <h3>6. Cross-domain workflows need routing guarantees</h3>
          <p>Workbench failures can omit the required domain tool entirely. More agents do not help if none are forced to cover every required side-effect domain.</p>
        </article>
      </div>
    </section>

    <section id="solution" class="section">
      <div class="section-kicker">Proposed General Solution</div>
      <h2>Use dynamic MAS rather than fixed topology selection.</h2>
      <div class="solution-steps">
        <div><strong>Start cheap.</strong><span>Run SAS or only-voting first for deterministic, low-ambiguity tasks.</span></div>
        <div><strong>Detect failure shape.</strong><span>Classify the live run as retrieval gap, tool error, state-action error, low-confidence disagreement, or domain-routing miss.</span></div>
        <div><strong>Spawn specialists on demand.</strong><span>Use LLM-as-tools or agent spawning for bounded subtasks: retrieval verifier, API recovery worker, domain router, or final-answer auditor.</span></div>
        <div><strong>Switch topology by evidence need.</strong><span>Escalate to fully linked or group debate only when agents have conflicting evidence or high-value ambiguity.</span></div>
        <div><strong>Gate rounds on new evidence.</strong><span>Continue discussion only when a round adds new documents, new tool outputs, or a corrected action plan.</span></div>
        <div><strong>Validate before finalization.</strong><span>Run task-specific validators: citation coverage for BrowseComp, action/state validator for PlanCraft, call-set coverage for WorkBench, tool-error dedupe for StableToolBench.</span></div>
      </div>
    </section>

    <section id="limitations" class="section">
      <div class="section-kicker">Limitations</div>
      <h2>What this report does not prove</h2>
      <ul class="plain-list">
        <li>Failure-mode labels are deterministic heuristics over traces, not human adjudication of every run.</li>
        <li>Case studies are representative examples selected for diagnostic value; they are not exhaustive.</li>
        <li>Token costs are trace-derived totals from these artifact runs and should not be generalized to other models or prompts without rerunning.</li>
        <li>Hypotheses about topology switching and dynamic MAS are design recommendations, not experimentally validated interventions in this repository.</li>
      </ul>
    </section>
    """


def render_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    static_sections = render_static_sections(payload)
    recommendation_sections = render_recommendation_sections()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MAS Failure Analysis Report</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --ink: #141922;
      --muted: #667085;
      --line: #d7dde6;
      --accent: #156f73;
      --accent-2: #9b4d20;
      --accent-3: #6d5bd0;
      --danger: #b42318;
      --warn: #946200;
      --good: #116329;
      --shadow: 0 1px 2px rgba(20, 25, 34, 0.05);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    html {{
      max-width: 100%;
      overflow-x: hidden;
    }}
    body {{
      margin: 0;
      max-width: 100%;
      overflow-x: hidden;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.5;
    }}
    a {{ color: var(--accent); }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
      background: #eef2f6;
      padding: 0.1rem 0.28rem;
      border-radius: 4px;
      overflow-wrap: anywhere;
    }}
    .layout {{
      min-height: 100vh;
    }}
    nav {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 14px clamp(16px, 4vw, 44px);
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(10px);
    }}
    nav .brand {{
      display: flex;
      gap: 10px;
      align-items: center;
      font-weight: 800;
      letter-spacing: 0;
      min-width: max-content;
    }}
    .brand-mark {{
      width: 30px;
      height: 30px;
      border-radius: 6px;
      background: var(--accent);
    }}
    nav .links {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 4px;
    }}
    nav a {{
      display: block;
      padding: 6px 9px;
      color: var(--muted);
      text-decoration: none;
      border-radius: 6px;
      font-size: 0.84rem;
      font-weight: 700;
      white-space: nowrap;
    }}
    nav a:hover, nav a.active {{
      color: var(--ink);
      background: #edf4f4;
    }}
    main {{
      min-width: 0;
      max-width: 1180px;
      margin: 0 auto;
      padding: 34px 20px 56px;
    }}
    .hero {{
      padding: 16px 0 14px;
    }}
    h1 {{
      margin: 0 0 14px;
      max-width: 920px;
      overflow-wrap: break-word;
      font-size: clamp(2.1rem, 5vw, 3.25rem);
      line-height: 1.08;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 18px;
      font-size: 1.62rem;
      line-height: 1.22;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 1.04rem;
      line-height: 1.25;
      letter-spacing: 0;
    }}
    p {{ color: var(--muted); margin: 0 0 12px; }}
    h1, h2, h3, p, li, summary {{
      overflow-wrap: anywhere;
    }}
    .lede {{
      max-width: 880px;
      font-size: 1.08rem;
      color: #354152;
      overflow-wrap: break-word;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      gap: 10px;
      margin-top: 20px;
    }}
    .tag {{
      flex: 0 0 auto;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 6px 10px;
      color: #445064;
      font-size: 0.86rem;
    }}
    .section {{
      min-width: 0;
      margin: 22px 0;
      padding: 26px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .section-kicker {{
      margin-bottom: 8px;
      color: var(--accent);
      font-size: 0.78rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 180px), 1fr));
      gap: 12px;
      margin: 22px 0;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #f9fafb;
      min-width: 0;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 0.82rem;
      font-weight: 700;
    }}
    .metric-value {{
      margin: 6px 0;
      font-size: 1.65rem;
      font-weight: 800;
      line-height: 1;
    }}
    .metric-card p {{ font-size: 0.86rem; margin: 0; }}
    .callout-grid, .why-grid, .profile-grid, .case-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));
      gap: 16px;
    }}
    .why-grid {{
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
    }}
    .callout, .why-grid article, .profile-card, .case-card, .taxonomy-card {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      background: #fff;
    }}
    .callout.evidence {{ border-top: 4px solid var(--accent); }}
    .callout.hypothesis {{ border-top: 4px solid var(--accent-2); }}
    ul {{ margin: 0; padding-left: 18px; }}
    li {{ margin: 6px 0; color: #354152; }}
    .method-grid, .solution-steps {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
      gap: 12px;
    }}
    .method-grid div, .solution-steps div {{
      min-width: 0;
      border-left: 3px solid var(--accent);
      background: #f6faf9;
      padding: 12px 14px;
      border-radius: 6px;
    }}
    .method-grid strong, .solution-steps strong {{
      display: block;
      margin-bottom: 4px;
    }}
    .method-grid span, .solution-steps span {{
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 190px), 1fr));
      gap: 12px;
      margin: 14px 0 18px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 0.84rem;
      font-weight: 700;
    }}
    select, input {{
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 0.95rem;
    }}
    .table-wrap {{
      width: 100%;
      max-width: 100%;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1040px;
      table-layout: fixed;
      font-size: 0.9rem;
    }}
    #evidenceTable {{ min-width: 1320px; }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    td code {{
      display: block;
      max-width: 100%;
      white-space: normal;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f1f4f8;
      color: #344054;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      z-index: 1;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .bar-cell {{
      display: flex;
      gap: 8px;
      align-items: center;
      min-width: 0;
    }}
    .bar {{
      position: relative;
      height: 8px;
      flex: 1;
      min-width: 46px;
      border-radius: 999px;
      background: #e8edf3;
      overflow: hidden;
    }}
    .bar span {{
      position: absolute;
      inset: 0 auto 0 0;
      background: var(--accent);
      border-radius: inherit;
    }}
    .delta.pos {{ color: var(--good); font-weight: 800; }}
    .delta.neg {{ color: var(--danger); font-weight: 800; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 6px; min-width: 0; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 0.78rem;
      color: #334155;
      background: #f8fafc;
      white-space: normal;
    }}
    .taxonomy-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
      gap: 14px;
    }}
    .taxonomy-card .count {{
      font-size: 1.65rem;
      font-weight: 800;
      margin: 4px 0;
    }}
    .muted {{ color: var(--muted); }}
    .case-card {{
      display: grid;
      gap: 10px;
    }}
    details {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 800;
    }}
    .fact-list {{
      margin-top: 10px;
      padding-left: 18px;
    }}
    .path-list {{
      display: grid;
      gap: 5px;
      margin-top: 10px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.78rem;
      overflow-wrap: anywhere;
    }}
    .chart {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .chart-row {{
      display: grid;
      grid-template-columns: minmax(120px, 180px) minmax(150px, 1fr) 70px minmax(180px, 0.8fr);
      gap: 10px;
      align-items: center;
      font-size: 0.9rem;
    }}
    .chart-track {{
      height: 12px;
      background: #e8edf3;
      border-radius: 999px;
      overflow: hidden;
    }}
    .chart-fill {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--accent-3));
    }}
    .plain-list {{ max-width: 860px; }}
    @media (max-width: 980px) {{
      nav {{
        position: static;
        align-items: flex-start;
        flex-direction: column;
        gap: 10px;
      }}
      nav .links {{
        justify-content: flex-start;
      }}
      main {{ padding-top: 24px; }}
    }}
    @media (max-width: 680px) {{
      nav {{ padding: 12px; }}
      nav .links {{
        width: 100%;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
        gap: 2px;
      }}
      nav a {{ font-size: 0.8rem; padding: 5px 7px; white-space: normal; }}
      main {{ padding: 20px 12px 36px; }}
      .section {{ padding: 18px 14px; margin: 16px 0; }}
      h1 {{ font-size: 2rem; line-height: 1.12; }}
      h2 {{ font-size: 1.32rem; }}
      .metric-grid, .callout-grid, .why-grid, .profile-grid, .taxonomy-grid, .case-grid, .method-grid, .solution-steps, .controls {{
        grid-template-columns: 1fr;
      }}
      .chart-row {{ grid-template-columns: 1fr; gap: 5px; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <nav aria-label="Report navigation">
      <div class="brand"><span class="brand-mark"></span><span>MAS Failure Analysis</span></div>
      <div class="links">
        <a href="#summary">Summary</a>
        <a href="#method">Method</a>
        <a href="#quant">Quantitative</a>
        <a href="#taxonomy">Taxonomy</a>
        <a href="#topologies">Topologies</a>
        <a href="#benchmarks">Benchmarks</a>
        <a href="#models">Models</a>
        <a href="#cases">Case Studies</a>
        <a href="#why">Why MAS Fails</a>
        <a href="#solution">Solution</a>
        <a href="#limitations">Limitations</a>
        <a href="#appendix">Appendix</a>
      </div>
    </nav>
    <main>
      <header class="hero">
        <h1>Qualitative failure-mode analysis of MAS topologies</h1>
        <p class="lede">A self-contained interactive report connecting benchmark outcomes, trace-derived coordination metrics, tool-use behavior, and trajectory evidence across Gemma and GPT-OSS experiments.</p>
        <div class="meta-row">
          <span class="tag">Generated {escape(payload['generated_at_utc'])}</span>
          <span class="tag">FinanceAgent excluded</span>
          <span class="tag">{payload['counts']['runs']:,} run artifacts</span>
          <span class="tag">{payload['counts']['system_rows']} system rows</span>
        </div>
      </header>

      {static_sections}

      <section id="quant" class="section">
        <div class="section-kicker">Quantitative Overview</div>
        <h2>Success, cost, and coordination by system</h2>
        <div class="controls">
          <label>Model<select id="quantModel"></select></label>
          <label>Benchmark<select id="quantBenchmark"></select></label>
          <label>Topology<select id="quantTopology"></select></label>
          <label>Search<input id="quantSearch" placeholder="Filter rows"></label>
        </div>
        <div id="quantChart" class="chart" aria-label="Best success rates by benchmark"></div>
        <div class="table-wrap"><table id="quantTable"></table></div>
      </section>

      <section id="taxonomy" class="section">
        <div class="section-kicker">Failure-Mode Taxonomy</div>
        <h2>Auditable labels derived from trace and eval artifacts</h2>
        <div id="taxonomyGrid" class="taxonomy-grid"></div>
      </section>

      <section id="topologies" class="section">
        <div class="section-kicker">Topology-Specific Analysis</div>
        <h2>Where each topology tends to help or fail</h2>
        <div id="topologyGrid" class="profile-grid"></div>
      </section>

      <section id="benchmarks" class="section">
        <div class="section-kicker">Benchmark-Specific Analysis</div>
        <h2>Different tasks expose different MAS failure surfaces</h2>
        <div id="benchmarkGrid" class="profile-grid"></div>
      </section>

      <section id="models" class="section">
        <div class="section-kicker">Cross-Model Comparison</div>
        <h2>The best topology changes with model and benchmark</h2>
        <div id="modelGrid" class="profile-grid"></div>
      </section>

      <section id="cases" class="section">
        <div class="section-kicker">Representative Trajectory Case Studies</div>
        <h2>Trace-backed examples of why MAS failed</h2>
        <div id="caseGrid" class="case-grid"></div>
      </section>

      {recommendation_sections}

      <section id="appendix" class="section">
        <div class="section-kicker">Appendix / Evidence Table</div>
        <h2>Filtered failure evidence rows</h2>
        <p>The table shows high-signal failed runs selected from all non-finance failures. Source paths point to the exact eval and trajectory artifacts used by the generator.</p>
        <div class="controls">
          <label>Model<select id="evidenceModel"></select></label>
          <label>Benchmark<select id="evidenceBenchmark"></select></label>
          <label>Topology<select id="evidenceTopology"></select></label>
          <label>Failure mode<select id="evidenceMode"></select></label>
        </div>
        <div class="table-wrap"><table id="evidenceTable"></table></div>
      </section>
    </main>
  </div>

  <script id="report-data" type="application/json">{payload_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('report-data').textContent);
    const pct = value => value === null || value === undefined ? 'n/a' : `${{(value * 100).toFixed(1)}}%`;
    const num = (value, digits = 1) => value === null || value === undefined ? 'n/a' : Number(value).toLocaleString(undefined, {{maximumFractionDigits: digits, minimumFractionDigits: digits}});
    const short = (text, limit = 150) => {{
      const value = String(text ?? '').replace(/\\s+/g, ' ').trim();
      return value.length > limit ? value.slice(0, limit - 1).trim() + '...' : value;
    }};
    const modeLabel = id => (DATA.failure_modes.find(mode => mode.id === id) || {{label: id}}).label;
    const unique = (rows, key) => [...new Set(rows.map(row => row[key]).filter(Boolean))].sort();
    function fillSelect(id, values, allLabel = 'All') {{
      const select = document.getElementById(id);
      select.innerHTML = `<option value="">${{allLabel}}</option>` + values.map(value => `<option value="${{value}}">${{value}}</option>`).join('');
    }}
    function deltaClass(value) {{ return value >= 0 ? 'delta pos' : 'delta neg'; }}
    function bar(value, max = 1) {{
      const width = Math.max(0, Math.min(100, (Number(value) / max) * 100));
      return `<div class="bar-cell"><div class="bar"><span style="width:${{width}}%"></span></div><span>${{pct(value)}}</span></div>`;
    }}
    function renderTable(el, headers, rows) {{
      const table = document.getElementById(el);
      table.innerHTML = `<thead><tr>${{headers.map(h => `<th>${{h.label}}</th>`).join('')}}</tr></thead><tbody>${{rows.map(row => `<tr>${{headers.map(h => `<td>${{h.render ? h.render(row) : (row[h.key] ?? '')}}</td>`).join('')}}</tr>`).join('')}}</tbody>`;
    }}
    function renderQuant() {{
      const model = document.getElementById('quantModel').value;
      const benchmark = document.getElementById('quantBenchmark').value;
      const topology = document.getElementById('quantTopology').value;
      const search = document.getElementById('quantSearch').value.toLowerCase();
      let rows = DATA.system_rows.filter(row =>
        (!model || row.model === model) &&
        (!benchmark || row.benchmark === benchmark) &&
        (!topology || row.system === topology) &&
        (!search || JSON.stringify(row).toLowerCase().includes(search))
      );
      rows.sort((a, b) => b.avg_success_rate - a.avg_success_rate || a.avg_tokens_total - b.avg_tokens_total);
      renderTable('quantTable', [
        {{label: 'Model', render: row => row.model}},
        {{label: 'Benchmark', render: row => row.benchmark}},
        {{label: 'System', render: row => row.system}},
        {{label: 'Success', render: row => bar(row.avg_success_rate)}},
        {{label: 'Delta vs SAS', render: row => row.success_delta_vs_sas === null ? 'baseline' : `<span class="${{deltaClass(row.success_delta_vs_sas)}}">${{pct(row.success_delta_vs_sas)}}</span>`}},
        {{label: 'Tokens', render: row => num(row.avg_tokens_total, 0)}},
        {{label: 'Token x SAS', render: row => row.token_multiplier_vs_sas ? `${{row.token_multiplier_vs_sas.toFixed(1)}}x` : 'baseline'}},
        {{label: 'Tool err', render: row => pct(row.avg_tool_error_rate)}},
        {{label: 'Comm', render: row => num(row.avg_communication_count, 1)}},
        {{label: 'Handoff', render: row => num(row.avg_handoff_count, 1)}}
      ], rows);
      renderQuantChart(rows);
    }}
    function renderQuantChart(rows) {{
      const chart = document.getElementById('quantChart');
      const best = [];
      for (const benchmark of DATA.benchmarks) {{
        const candidates = rows.filter(row => row.benchmark === benchmark);
        if (candidates.length) best.push(candidates.sort((a, b) => b.avg_success_rate - a.avg_success_rate)[0]);
      }}
      chart.innerHTML = best.map(row => `
        <div class="chart-row">
          <strong>${{row.benchmark}}</strong>
          <div class="chart-track"><span class="chart-fill" style="width:${{Math.max(2, row.avg_success_rate * 100)}}%"></span></div>
          <span>${{pct(row.avg_success_rate)}}</span>
          <span class="muted">${{row.model}} / ${{row.system}}</span>
        </div>`).join('');
    }}
    function renderTaxonomy() {{
      document.getElementById('taxonomyGrid').innerHTML = DATA.failure_modes.map(mode => `
        <article class="taxonomy-card">
          <h3>${{mode.label}}</h3>
          <div class="count">${{mode.count.toLocaleString()}}</div>
          <p>${{mode.definition}}</p>
          <p class="muted"><strong>Example:</strong> ${{mode.example || 'No matching rows in this artifact set.'}}</p>
        </article>`).join('');
    }}
    function renderProfiles() {{
      document.getElementById('topologyGrid').innerHTML = DATA.topology_profiles.map(row => `
        <article class="profile-card">
          <h3>${{row.label}}</h3>
          <p>${{row.evidence}}</p>
          <div class="pill-row">
            <span class="pill">avg success ${{pct(row.avg_success)}}</span>
            <span class="pill">avg lift ${{pct(row.avg_success_delta_vs_sas)}}</span>
            <span class="pill">median cost ${{row.median_token_multiplier_vs_sas ? row.median_token_multiplier_vs_sas.toFixed(1) : 'n/a'}}x</span>
            <span class="pill">best in ${{row.best_count}} pairs</span>
          </div>
          <p><strong>Risk:</strong> ${{row.risk}}</p>
          <p><strong>Design response:</strong> ${{row.solution}}</p>
          <p class="muted">Top failed-run labels: ${{row.top_failure_modes.map(([mode, count]) => `${{modeLabel(mode)}} (${{count}})`).join(', ') || 'none'}}</p>
        </article>`).join('');
      document.getElementById('benchmarkGrid').innerHTML = DATA.benchmark_profiles.map(row => `
        <article class="profile-card">
          <h3>${{row.label}}</h3>
          <p>${{row.evidence}}</p>
          <div class="pill-row">
            <span class="pill">SAS avg ${{pct(row.sas_success_avg)}}</span>
            <span class="pill">best MAS avg ${{pct(row.best_success_avg)}}</span>
            <span class="pill">best lift ${{pct(row.best_gain_avg)}}</span>
            <span class="pill">${{row.failed_runs.toLocaleString()}} failed runs</span>
          </div>
          <p><strong>Risk:</strong> ${{row.risk}}</p>
          <p><strong>Design response:</strong> ${{row.solution}}</p>
          <details>
            <summary>Best topology by model</summary>
            <ul class="fact-list">${{row.best_topologies.map(item => `<li>${{item.model}}: ${{item.topology}}, success ${{pct(item.success)}}, lift ${{pct(item.gain)}}, tokens ${{num(item.tokens, 0)}}</li>`).join('')}}</ul>
          </details>
        </article>`).join('');
      document.getElementById('modelGrid').innerHTML = DATA.model_profiles.map(row => `
        <article class="profile-card">
          <h3>${{row.model}}</h3>
          <div class="pill-row">
            <span class="pill">SAS avg ${{pct(row.sas_success_avg)}}</span>
            <span class="pill">best MAS avg ${{pct(row.best_mas_success_avg)}}</span>
            <span class="pill">avg lift ${{pct(row.best_gain_avg)}}</span>
            <span class="pill">median cost ${{row.median_best_token_multiplier.toFixed(1)}}x</span>
          </div>
          <details open>
            <summary>Benchmark winners</summary>
            <ul class="fact-list">${{row.best_rows.map(item => `<li>${{item.benchmark}}: ${{item.topology}}, success ${{pct(item.success)}}, lift ${{pct(item.sas_delta)}}, token x SAS ${{item.token_multiplier.toFixed(1)}}x</li>`).join('')}}</ul>
          </details>
        </article>`).join('');
    }}
    function renderCases() {{
      document.getElementById('caseGrid').innerHTML = DATA.case_studies.map(row => `
        <article class="case-card">
          <div class="pill-row">${{row.failure_modes.map(mode => `<span class="pill">${{modeLabel(mode)}}</span>`).join('')}}</div>
          <h3>${{row.title}}</h3>
          <p><strong>Context:</strong> ${{row.model}} / ${{row.benchmark}} / ${{row.topology}} / ${{row.task_id}} run ${{row.run_index}}</p>
          <details open>
            <summary>Evidence</summary>
            <ul class="fact-list">${{row.facts.map(fact => `<li>${{fact}}</li>`).join('')}}</ul>
            <p><strong>Prompt:</strong> ${{row.prompt}}</p>
            <p><strong>Final answer:</strong> ${{row.prediction}}</p>
            <p><strong>Reference or expected call:</strong> ${{row.reference || 'See facts above.'}}</p>
          </details>
          <details>
            <summary>Hypothesis and source paths</summary>
            <p><strong>Hypothesis:</strong> ${{row.hypothesis}}</p>
            <div class="path-list">${{Object.entries(row.paths).map(([k, v]) => `<span>${{k}}: ${{v}}</span>`).join('')}}</div>
          </details>
        </article>`).join('');
    }}
    function renderEvidence() {{
      const model = document.getElementById('evidenceModel').value;
      const benchmark = document.getElementById('evidenceBenchmark').value;
      const topology = document.getElementById('evidenceTopology').value;
      const mode = document.getElementById('evidenceMode').value;
      let rows = DATA.evidence_rows.filter(row =>
        (!model || row.model === model) &&
        (!benchmark || row.benchmark === benchmark) &&
        (!topology || row.topology === topology) &&
        (!mode || row.failure_modes.includes(mode))
      );
      renderTable('evidenceTable', [
        {{label: 'Model', render: row => row.model}},
        {{label: 'Benchmark', render: row => row.benchmark}},
        {{label: 'Topology', render: row => row.topology}},
        {{label: 'Task/run', render: row => `${{row.task_id}} / ${{row.run_index}}`}},
        {{label: 'Modes', render: row => `<div class="pill-row">${{row.failure_modes.map(mode => `<span class="pill">${{modeLabel(mode)}}</span>`).join('')}}</div>`}},
        {{label: 'Reason', render: row => row.final_reason || 'n/a'}},
        {{label: 'Tokens', render: row => num(row.tokens_total, 0)}},
        {{label: 'Tool fails', render: row => `${{num(row.tool_fail_total, 0)}} / ${{num(row.tool_calls_total, 0)}}`}},
        {{label: 'Prediction', render: row => short(row.prediction, 160)}},
        {{label: 'Source', render: row => `<code>${{row.eval_path}}</code>`}}
      ], rows);
    }}
    function initFilters() {{
      fillSelect('quantModel', unique(DATA.system_rows, 'model'));
      fillSelect('quantBenchmark', unique(DATA.system_rows, 'benchmark'));
      fillSelect('quantTopology', unique(DATA.system_rows, 'system'));
      fillSelect('evidenceModel', unique(DATA.evidence_rows, 'model'));
      fillSelect('evidenceBenchmark', unique(DATA.evidence_rows, 'benchmark'));
      fillSelect('evidenceTopology', unique(DATA.evidence_rows, 'topology'));
      const modeOptions = DATA.failure_modes.filter(mode => mode.kind === 'run').map(mode => mode.id);
      const modeSelect = document.getElementById('evidenceMode');
      modeSelect.innerHTML = '<option value="">All modes</option>' + modeOptions.map(id => `<option value="${{id}}">${{modeLabel(id)}}</option>`).join('');
      ['quantModel','quantBenchmark','quantTopology','quantSearch'].forEach(id => document.getElementById(id).addEventListener('input', renderQuant));
      ['evidenceModel','evidenceBenchmark','evidenceTopology','evidenceMode'].forEach(id => document.getElementById(id).addEventListener('input', renderEvidence));
    }}
    initFilters();
    renderQuant();
    renderTaxonomy();
    renderProfiles();
    renderCases();
    renderEvidence();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "mas_failure_analysis_report.html",
        help="Standalone HTML report path.",
    )
    parser.add_argument(
        "--model-root",
        action="append",
        type=Path,
        help="Experiment model root. Defaults to the two full_experiment model folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = []
    for raw_path in args.model_root or MODEL_ROOTS:
        path = Path(raw_path)
        roots.append(path if path.is_absolute() else PROJECT_ROOT / path)
    missing = [path for path in roots if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model roots: {missing}")
    payload = build_payload(roots)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.write_text(render_html(payload), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

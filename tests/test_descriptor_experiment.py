from types import SimpleNamespace

from descriptor.experiment import _summarize_runs
from descriptor.metrics import RunOutcome


def test_summarize_runs_excludes_rerun_fallback_from_scores() -> None:
    evaluations = [
        SimpleNamespace(
            task_id="t1",
            score=0.0,
            success=False,
            details={"run_failed": True, "fallback": True, "needs_rerun": True},
        ),
        SimpleNamespace(task_id="t1", score=1.0, success=True, details={}),
    ]
    outcomes = [
        RunOutcome(
            success=False,
            completion=False,
            score=0.0,
            success_source="run_fallback",
            completion_source="run_fallback",
        ),
        RunOutcome(
            success=True,
            completion=True,
            score=1.0,
            success_source="benchmark_evaluation",
            completion_source="final_answer",
        ),
    ]

    summary = _summarize_runs(evaluations, outcomes)

    assert summary["count"] == 2
    assert summary["valid_count"] == 1
    assert summary["excluded_count"] == 1
    assert summary["avg_score"] == 1.0
    assert summary["runs"][0]["included_in_score"] is False
    assert summary["runs"][1]["included_in_score"] is True

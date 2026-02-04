import unittest

from descriptor.metrics import ExtensionOptions, compute_run_metrics, compute_task_metrics
from descriptor.schema import TraceEvent


class TestMetrics(unittest.TestCase):
    def _make_event(
        self,
        event_type: str,
        token_in: int,
        token_out: int,
        latency_ms: float,
        cost_usd: float,
        payload: dict | None = None,
        state_id: str | None = None,
    ) -> TraceEvent:
        return TraceEvent(
            timestamp_start=0.0,
            timestamp_end=1.0,
            actor="agent",
            event_type=event_type,
            payload=payload or {},
            token_in=token_in,
            token_out=token_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            state_id=state_id,
        )

    def test_run_and_task_metrics(self) -> None:
        events = [
            self._make_event("plan", 1, 1, 10.0, 0.01, state_id="s1"),
            self._make_event("act", 2, 2, 20.0, 0.02, state_id="s1"),
            self._make_event("tool_call", 1, 1, 5.0, 0.005, {"tool_name": "calc"}),
            self._make_event(
                "tool_result",
                1,
                1,
                5.0,
                0.005,
                {"status": "error", "error_code": "FAIL"},
            ),
            self._make_event("verify", 1, 1, 3.0, 0.003, state_id="s2"),
            self._make_event("revise", 1, 1, 4.0, 0.004, {"redo": True}),
            self._make_event("finalize", 1, 1, 2.0, 0.002, {"success": True}),
        ]

        run_metrics = compute_run_metrics(events, extensions=ExtensionOptions(include_stage_metrics=True))
        self.assertTrue(run_metrics["success"])
        self.assertTrue(run_metrics["completion"])
        self.assertEqual(run_metrics["tool_calls_total"], 1.0)
        self.assertEqual(run_metrics["tool_fail_total"], 1.0)
        self.assertAlmostEqual(run_metrics["backtrack_rate"], 1.0 / 7.0, places=6)
        self.assertAlmostEqual(run_metrics["loop_score"], 1.0 / 3.0, places=6)
        self.assertEqual(run_metrics["stage_plan_events"], 1.0)

        task_metrics = compute_task_metrics([run_metrics])
        self.assertEqual(task_metrics["Q1_success_rate"], 1.0)
        self.assertEqual(task_metrics["Q2_completion_rate"], 1.0)
        self.assertEqual(task_metrics["C5_tool_error_rate"], 1.0)
        self.assertEqual(task_metrics["P1_steps_total"], 7.0)
        self.assertAlmostEqual(task_metrics["P4_verification_density"], 1.0 / 7.0, places=6)


if __name__ == "__main__":
    unittest.main()

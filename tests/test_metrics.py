import unittest
from types import SimpleNamespace

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
        actor: str = "agent",
    ) -> TraceEvent:
        return TraceEvent(
            timestamp_start=0.0,
            timestamp_end=1.0,
            actor=actor,
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
            self._make_event(
                "tool_call",
                0,
                0,
                1.0,
                0.0,
                {"tool_name": "inter_agent_send", "to": ["agent_b", "agent_c"]},
                actor="agent_a",
            ),
            self._make_event(
                "tool_result",
                0,
                0,
                1.0,
                0.0,
                {"tool_name": "inter_agent_send", "status": "ok"},
                actor="system",
            ),
            self._make_event(
                "tool_call",
                0,
                0,
                1.0,
                0.0,
                {"tool_name": "inter_agent_send", "to": ["agent_rep"]},
                actor="system",
            ),
            self._make_event("act", 1, 1, 2.0, 0.0, actor="agent_a"),
            self._make_event("act", 1, 1, 2.0, 0.0, actor="agent_b"),
            self._make_event("verify", 1, 1, 3.0, 0.003, state_id="s2"),
            self._make_event("revise", 1, 1, 4.0, 0.004, {"redo": True}),
            self._make_event("finalize", 1, 1, 2.0, 0.002, {"success": True}),
        ]

        run_metrics = compute_run_metrics(
            events, extensions=ExtensionOptions(include_stage_metrics=True)
        )
        self.assertTrue(run_metrics["success"])
        self.assertTrue(run_metrics["completion"])
        self.assertEqual(run_metrics["success_source"], "trace_inference")
        self.assertEqual(run_metrics["tool_calls_total"], 3.0)
        self.assertEqual(run_metrics["tool_fail_total"], 1.0)
        self.assertAlmostEqual(run_metrics["backtrack_rate"], 1.0 / 12.0, places=6)
        self.assertEqual(run_metrics["communication_count"], 3.0)
        self.assertEqual(run_metrics["communication_count_agent_to_agent"], 2.0)
        self.assertEqual(run_metrics["communication_count_system_mediated"], 1.0)
        self.assertEqual(run_metrics["handoff_count"], 3.0)
        self.assertEqual(run_metrics["stage_plan_events"], 1.0)

        task_metrics = compute_task_metrics([run_metrics])
        self.assertEqual(task_metrics["Q1_success_rate"], 1.0)
        self.assertEqual(task_metrics["Q2_completion_rate"], 1.0)
        self.assertEqual(task_metrics["D1_tool_error_rate"], 1.0 / 3.0)
        self.assertEqual(task_metrics["D2_communication_count"], 3.0)
        self.assertEqual(task_metrics["D2_agent_to_agent_communication_count"], 2.0)
        self.assertEqual(task_metrics["D2_system_mediated_communication_count"], 1.0)
        self.assertEqual(task_metrics["D3_handoff_count"], 3.0)
        self.assertEqual(task_metrics["P1_steps_total"], 12.0)
        self.assertAlmostEqual(task_metrics["P4_verification_density"], 1.0 / 12.0, places=6)

    def test_benchmark_evaluation_overrides_trace_success(self) -> None:
        events = [
            self._make_event("act", 2, 3, 10.0, 0.0),
            self._make_event("finalize", 0, 5, 1.0, 0.0, {"status": "completed"}),
        ]

        evaluation = SimpleNamespace(score=0.0, success=False)
        run_metrics = compute_run_metrics(
            events,
            evaluation=evaluation,
            final_answer="wrong answer",
            extensions=ExtensionOptions(include_stage_metrics=False),
        )

        self.assertFalse(run_metrics["success"])
        self.assertTrue(run_metrics["completion"])
        self.assertEqual(run_metrics["success_source"], "benchmark_evaluation")
        self.assertEqual(run_metrics["completion_source"], "final_answer")


if __name__ == "__main__":
    unittest.main()

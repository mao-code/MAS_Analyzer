import unittest

from benchmark.base import init_run_metadata_aggregate, merge_step_run_metadata


class TestBenchmarkBase(unittest.TestCase):
    def test_merge_step_run_metadata_preserves_runtime_fields(self) -> None:
        aggregate = init_run_metadata_aggregate()
        step_metadata = {
            "task_id": "task-1",
            "run_index": 0,
            "seed": 7,
            "topology": "fully_linked_debate",
            "rounds_configured": 2,
            "discussion_rounds": 1,
            "turns_executed": 2,
            "messages_sent_total": 3,
            "messages_sent_by_agent": {"agent_0": 2, "agent_1": 1},
            "tool_calls_total": 1,
            "tool_call_counts": {"inter_agent_send": 1},
            "tool_definitions": [{"name": "inter_agent_send"}],
            "interaction_logs": [{"agent_id": "agent_0"}],
            "phase_history": [{"phase": "debate"}],
            "relay_messages": [{"message_id": "m_1", "sender": "agent_0"}],
            "message_views": [{"viewer": "agent_1"}],
            "termination_history": [{"dispatch_id": 1, "stage_name": "debate_controller"}],
            "agent_outputs": {"agent_0": "42"},
            "vote_tally": {"42": 3},
            "final_reason": "fully_linked_debate:judge_vote",
        }

        merge_step_run_metadata(
            aggregate,
            step_metadata,
            outer_step_index=4,
            step_task_id="task-1_turn_5",
            final_answer="42",
        )

        self.assertEqual(aggregate["task_id"], "task-1")
        self.assertEqual(aggregate["run_index"], 0)
        self.assertEqual(aggregate["seed"], 7)
        self.assertEqual(aggregate["topology"], "fully_linked_debate")
        self.assertEqual(aggregate["turns_executed"], 2)
        self.assertEqual(aggregate["messages_sent_by_agent"]["agent_0"], 2)
        self.assertEqual(aggregate["messages_sent_total"], 3)
        self.assertEqual(aggregate["tool_calls_total"], 1)
        self.assertEqual(aggregate["agent_outputs"]["agent_0"], "42")
        self.assertEqual(aggregate["vote_tally"]["42"], 3)
        self.assertEqual(aggregate["final_reason"], "fully_linked_debate:judge_vote")
        self.assertEqual(aggregate["interaction_logs"][0]["outer_step_index"], 4)
        self.assertEqual(aggregate["relay_messages"][0]["outer_step_index"], 4)
        self.assertEqual(aggregate["termination_history"][0]["outer_step_index"], 4)


if __name__ == "__main__":
    unittest.main()

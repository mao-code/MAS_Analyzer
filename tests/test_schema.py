import unittest

from descriptor.schema import TraceEvent, validate_event_dict


class TestTraceSchema(unittest.TestCase):
    def test_valid_event_roundtrip(self) -> None:
        data = {
            "timestamp_start": 0.0,
            "timestamp_end": 1.0,
            "actor": "agent",
            "event_type": "plan",
            "payload": {"summary": "plan"},
            "token_in": 1,
            "token_out": 2,
            "latency_ms": 100.0,
            "cost_usd": 0.01,
            "state_id": "s1",
        }
        validate_event_dict(data)
        event = TraceEvent.from_dict(data)
        self.assertEqual(event.event_type, "plan")
        roundtrip = event.to_dict()
        self.assertEqual(roundtrip["actor"], "agent")
        self.assertEqual(roundtrip["state_id"], "s1")

    def test_invalid_event_type(self) -> None:
        data = {
            "timestamp_start": 0.0,
            "timestamp_end": 1.0,
            "actor": "agent",
            "event_type": "invalid",
            "payload": {},
            "token_in": 1,
            "token_out": 2,
            "latency_ms": 10.0,
            "cost_usd": 0.01,
        }
        with self.assertRaises(ValueError):
            TraceEvent.from_dict(data)

    def test_invalid_timestamps(self) -> None:
        data = {
            "timestamp_start": 2.0,
            "timestamp_end": 1.0,
            "actor": "agent",
            "event_type": "act",
            "payload": {},
            "token_in": 1,
            "token_out": 2,
            "latency_ms": 10.0,
            "cost_usd": 0.01,
        }
        with self.assertRaises(ValueError):
            TraceEvent.from_dict(data)


if __name__ == "__main__":
    unittest.main()

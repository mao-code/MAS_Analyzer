"""Structural relay-packet compaction: full fidelity by default, boundary-aware
reduction under a budget (never a mid-token chop of the answer)."""

import unittest

from MAS.artifacts import (
    _sentence_bounded_text,
    compact_packet_payload,
    packet_content,
    packet_payload_from_artifact,
)


class TestSentenceBoundedText(unittest.TestCase):
    def test_zero_budget_returns_normalized_full_text(self) -> None:
        self.assertEqual(_sentence_bounded_text("  alpha   beta\tgamma  ", max_chars=0), "alpha beta gamma")

    def test_cuts_at_sentence_boundary_and_appends_ellipsis(self) -> None:
        text = "First sentence here. Second sentence runs on and on and on and on."
        out = _sentence_bounded_text(text, max_chars=30)
        self.assertLessEqual(len(out), 30)
        self.assertTrue(out.endswith("..."))
        self.assertIn("First sentence here", out)

    def test_no_boundary_hard_cut_still_bounded(self) -> None:
        out = _sentence_bounded_text("x" * 500, max_chars=48)
        self.assertLessEqual(len(out), 48)
        self.assertTrue(out.endswith("..."))


class TestCompactPacketPayload(unittest.TestCase):
    def _payload(self, **over):
        base = {
            "artifact_id": "a1",
            "summary": "A concise summary of the finding.",
            "answer_artifact": "L" * 4000,
            "critique": "Some critique text.",
            "revision_request": "Please revise X.",
            "confidence": 0.7,
            "evidence_summary": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "unresolved_issues": ["u1", "u2"],
        }
        base.update(over)
        return base

    def test_zero_budget_is_identity(self) -> None:
        payload = self._payload()
        self.assertEqual(compact_packet_payload(payload, max_chars=0), payload)

    def test_over_budget_drops_low_priority_and_prefers_summary(self) -> None:
        out = compact_packet_payload(self._payload(), max_chars=200)
        self.assertEqual(out["revision_request"], "")
        self.assertEqual(out["critique"], "")
        self.assertEqual(out["answer_artifact"], "")
        self.assertEqual(out["summary"], "A concise summary of the finding.")
        self.assertEqual(out["confidence"], 0.7)

    def test_answer_never_mid_token_chopped(self) -> None:
        # No summary -> the long answer is bounded at a word boundary, never raw-sliced.
        payload = self._payload(summary="", answer_artifact="alpha beta gamma " * 100)
        out = compact_packet_payload(payload, max_chars=60)
        body = out["answer_artifact"]
        self.assertLessEqual(len(body), 60)
        self.assertTrue(body.endswith("..."))
        prefix = body[:-3].rstrip()
        normalized = " ".join(("alpha beta gamma " * 100).split())
        self.assertTrue(normalized.startswith(prefix))
        # The cut landed on a word boundary (next source char is a space / end).
        self.assertTrue(len(prefix) == len(normalized) or normalized[len(prefix)] == " ")

    def test_trims_lists_under_budget(self) -> None:
        payload = self._payload(
            answer_artifact="",
            critique="",
            revision_request="",
            summary="short",
            evidence_summary=["e" * 30 for _ in range(6)],
            unresolved_issues=["u" * 30 for _ in range(6)],
        )
        out = compact_packet_payload(payload, max_chars=120)
        self.assertLessEqual(len(out["evidence_summary"]), 4)
        self.assertLessEqual(len(out["unresolved_issues"]), 4)


class TestPacketPayloadFromArtifact(unittest.TestCase):
    def test_default_is_full_fidelity(self) -> None:
        artifact = {
            "artifact_id": "a1",
            "summary": "s",
            "answer": "A" * 5000,
            "evidence_summary": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "unresolved_issues": ["u1", "u2", "u3", "u4", "u5"],
            "confidence": 0.6,
        }
        payload = packet_payload_from_artifact(artifact)
        self.assertEqual(len(payload["answer_artifact"]), 5000)
        self.assertEqual(len(payload["evidence_summary"]), 6)
        self.assertEqual(len(payload["unresolved_issues"]), 5)

    def test_budget_triggers_structural_compaction(self) -> None:
        artifact = {
            "artifact_id": "a1",
            "summary": "Short supported summary.",
            "answer": "A" * 5000,
            "confidence": 0.6,
        }
        payload = packet_payload_from_artifact(artifact, max_chars=200)
        self.assertEqual(payload["answer_artifact"], "")
        self.assertEqual(payload["summary"], "Short supported summary.")
        self.assertLessEqual(len(packet_content(payload, max_chars=200)), 200)


if __name__ == "__main__":
    unittest.main()

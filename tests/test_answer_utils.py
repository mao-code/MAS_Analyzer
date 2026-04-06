import unittest

from answer_utils import classify_answer_mode, extract_substantive_answer, has_substantive_answer


class TestAnswerUtils(unittest.TestCase):
    def test_blocked_answer_is_non_substantive(self) -> None:
        text = "The requested information cannot be determined because no evidence has been retrieved."

        self.assertEqual(classify_answer_mode(text), "blocked")
        self.assertEqual(extract_substantive_answer(text), "")
        self.assertFalse(has_substantive_answer(text))

    def test_plan_answer_is_non_substantive(self) -> None:
        text = "Plan: search for the institution, verify the graduation date, then confirm the city."

        self.assertEqual(classify_answer_mode(text), "plan")
        self.assertEqual(extract_substantive_answer(text), "")

    def test_direct_answer_remains_substantive(self) -> None:
        text = "Queen Arwa University"

        self.assertEqual(classify_answer_mode(text), "direct")
        self.assertEqual(extract_substantive_answer(text), "Queen Arwa University")

    def test_structured_blocked_payload_is_classified_correctly(self) -> None:
        text = (
            '{"answer_artifact":"","summary":"The task remains blocked because no search has been performed.",'
            '"critique":"","revision_request":"","confidence":0.0,"unresolved_issues":[],"evidence_summary":["None."]}'
        )

        self.assertEqual(classify_answer_mode(text), "blocked")
        self.assertEqual(extract_substantive_answer(text), "")

    def test_progress_status_answer_is_classified_as_blocked(self) -> None:
        text = (
            "I am currently investigating the identity of the learning institution. "
            "I have initiated a search but have not yet identified the institution."
        )

        self.assertEqual(classify_answer_mode(text), "blocked")
        self.assertEqual(extract_substantive_answer(text), "")


if __name__ == "__main__":
    unittest.main()

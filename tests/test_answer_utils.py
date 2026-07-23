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

    def test_structured_future_tool_call_is_classified_as_plan(self) -> None:
        text = (
            '{"answer_artifact":"I need to first retrieve the available categories and then '
            'call the product endpoint.","summary":"Retrieval plan.","confidence":0.5}'
        )

        self.assertEqual(classify_answer_mode(text), "plan")
        self.assertEqual(extract_substantive_answer(text), "")

    def test_pending_unexecuted_tool_plan_is_classified_as_blocked(self) -> None:
        text = (
            "The request is currently blocked because no tool calls have been executed; "
            "actual data retrieval is pending."
        )

        self.assertEqual(classify_answer_mode(text), "blocked")
        self.assertEqual(extract_substantive_answer(text), "")

    def test_future_fetch_payload_is_classified_as_plan(self) -> None:
        text = (
            '{"answer_artifact":"I need to fetch the quote and driver photo using the '
            'provided IDs.","summary":"I have identified the necessary tools.",'
            '"confidence":0.5}'
        )

        self.assertEqual(classify_answer_mode(text), "plan")
        self.assertEqual(extract_substantive_answer(text), "")

    def test_discoverable_identifier_clarification_is_non_substantive(self) -> None:
        text = (
            "Please provide the match ID or team names so I can locate the event and "
            "retrieve its incidents."
        )

        self.assertEqual(classify_answer_mode(text), "plan")
        self.assertEqual(extract_substantive_answer(text), "")

    def test_completed_tool_answer_with_future_wording_remains_direct(self) -> None:
        text = "The lookup returned Daniel Ricciardo's quote; I will display it at the party."

        self.assertEqual(classify_answer_mode(text), "direct")
        self.assertEqual(extract_substantive_answer(text), text)

    def test_domain_shaped_structured_answer_is_direct(self) -> None:
        text = (
            '{"categories":["Electronics"],"electronics_products":'
            '[{"id":893292,"name":"TV","details":"unknown"}]}'
        )

        self.assertEqual(classify_answer_mode(text), "direct")
        extracted = extract_substantive_answer(text)
        self.assertIn('"categories": ["Electronics"]', extracted)
        self.assertIn('"electronics_products"', extracted)


if __name__ == "__main__":
    unittest.main()

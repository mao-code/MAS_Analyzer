import csv
import tempfile
import unittest
from pathlib import Path

from benchmark.base import BenchmarkTask
from benchmark.workbench import WorkBenchBenchmark


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestWorkBenchBenchmark(unittest.TestCase):
    def _build_fixture(self, base: Path) -> Path:
        _write_csv(
            base / "data/processed/calendar_events.csv",
            ["event_id", "event_name", "participant_email", "event_start", "duration"],
            [
                {
                    "event_id": "00000001",
                    "event_name": "Sprint Planning",
                    "participant_email": "alex@company.com",
                    "event_start": "2023-11-30 10:00:00",
                    "duration": "30",
                }
            ],
        )
        _write_csv(
            base / "data/processed/emails.csv",
            ["email_id", "sender/recipient", "subject", "sent_datetime", "body", "status"],
            [
                {
                    "email_id": "1",
                    "sender/recipient": "alex@company.com",
                    "subject": "Hello",
                    "sent_datetime": "2023-11-29 09:00:00",
                    "body": "Test body",
                    "status": "inbox",
                }
            ],
        )
        _write_csv(
            base / "data/processed/analytics_data.csv",
            [
                "visitor_id",
                "date_of_visit",
                "session_duration_seconds",
                "user_engaged",
                "traffic_source",
            ],
            [
                {
                    "visitor_id": "v1",
                    "date_of_visit": "2023-11-29",
                    "session_duration_seconds": "120",
                    "user_engaged": "True",
                    "traffic_source": "search",
                }
            ],
        )
        _write_csv(
            base / "data/processed/project_tasks.csv",
            ["task_id", "task_name", "assigned_to_email", "list_name", "due_date", "board"],
            [
                {
                    "task_id": "00000001",
                    "task_name": "API cleanup",
                    "assigned_to_email": "alex@company.com",
                    "list_name": "Backlog",
                    "due_date": "2023-12-10",
                    "board": "Back end",
                }
            ],
        )
        _write_csv(
            base / "data/processed/customer_relationship_manager_data.csv",
            [
                "customer_id",
                "customer_name",
                "customer_email",
                "customer_phone",
                "last_contact_date",
                "product_interest",
                "status",
                "assigned_to_email",
                "notes",
                "follow_up_by",
            ],
            [
                {
                    "customer_id": "00000001",
                    "customer_name": "Taylor",
                    "customer_email": "taylor@example.com",
                    "customer_phone": "123",
                    "last_contact_date": "2023-11-20",
                    "product_interest": "Software",
                    "status": "Lead",
                    "assigned_to_email": "alex@company.com",
                    "notes": "Interested",
                    "follow_up_by": "2023-12-05",
                }
            ],
        )
        directory_path = base / "data/raw/email_addresses.csv"
        directory_path.parent.mkdir(parents=True, exist_ok=True)
        directory_path.write_text("alex@company.com\nsam@company.com\n", encoding="utf-8")
        _write_csv(
            base / "data/processed/queries_and_answers/calendar_queries_and_answers.csv",
            ["query", "answer", "domains", "base_template", "chosen_template"],
            [
                {
                    "query": "Delete the sprint planning meeting.",
                    "answer": "['calendar.delete_event.func(event_id=\"00000001\")']",
                    "domains": "['calendar']",
                    "base_template": "calendar_base",
                    "chosen_template": "calendar_delete",
                }
            ],
        )
        return base

    def test_load_tasks_and_company_directory_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = self._build_fixture(Path(tmpdir))
            benchmark = WorkBenchBenchmark({"domain": "calendar", "data_dir": str(data_dir)})

            tasks = benchmark.load_tasks(task_limit=1)

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].prompt, "Delete the sprint planning meeting.")
            self.assertEqual(tasks[0].metadata["domains"], ["calendar"])

            tools = benchmark._build_tools(benchmark_module_sandbox(data_dir), ["calendar"])
            tool_names = {tool["name"] for tool in tools}
            self.assertIn("calendar.delete_event", tool_names)
            self.assertIn("company_directory.find_email_address", tool_names)
            self.assertNotIn("email.send_email", tool_names)

    def test_evaluate_marks_correct_state_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = self._build_fixture(Path(tmpdir))
            benchmark = WorkBenchBenchmark({"domain": "calendar", "data_dir": str(data_dir)})
            task = benchmark.load_tasks(task_limit=1)[0]

            result = benchmark.evaluate(
                task,
                "",
                run_metadata={
                    "function_calls": ['calendar.delete_event.func(event_id="00000001")'],
                    "error": "",
                },
            )

            self.assertTrue(result.success)
            self.assertEqual(result.score, 1.0)
            self.assertTrue(result.details["exact_match"])
            self.assertFalse(result.details["unwanted_side_effects"])

    def test_evaluate_flags_wrong_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = self._build_fixture(Path(tmpdir))
            benchmark = WorkBenchBenchmark({"domain": "calendar", "data_dir": str(data_dir)})
            task = benchmark.load_tasks(task_limit=1)[0]

            result = benchmark.evaluate(
                task,
                "",
                run_metadata={
                    "function_calls": [
                        'calendar.create_event.func(event_name="New meeting", participant_email="alex@company.com", event_start="2023-11-30 11:00:00", duration="30")'
                    ],
                    "error": "",
                },
            )

            self.assertFalse(result.success)
            self.assertEqual(result.score, 0.0)
            self.assertFalse(result.details["exact_match"])
            self.assertTrue(result.details["unwanted_side_effects"])

    def test_evaluate_noop_task_requires_substantive_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = self._build_fixture(Path(tmpdir))
            benchmark = WorkBenchBenchmark({"domain": "calendar", "data_dir": str(data_dir)})
            task = BenchmarkTask(
                task_id="noop_0",
                prompt="If needed, book a meeting with Jessie Thomas.",
                reference_answer="",
                metadata={
                    "query": "If needed, book a meeting with Jessie Thomas.",
                    "answer": [],
                    "domains": ["calendar", "crm"],
                },
            )

            result = benchmark.evaluate(
                task,
                "To complete the request, I will execute the following plan: 1. Search CRM. 2. Find the assignee.",
                run_metadata={"function_calls": [], "error": ""},
            )

            self.assertFalse(result.success)
            self.assertEqual(result.score, 0.0)
            self.assertFalse(result.details["exact_match"])
            self.assertFalse(result.details["correct"])
            self.assertFalse(result.details["prediction_substantive"])

    def test_evaluate_noop_task_accepts_substantive_blocked_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = self._build_fixture(Path(tmpdir))
            benchmark = WorkBenchBenchmark({"domain": "calendar", "data_dir": str(data_dir)})
            task = BenchmarkTask(
                task_id="noop_1",
                prompt="If needed, book a meeting with Jessie Thomas.",
                reference_answer="",
                metadata={
                    "query": "If needed, book a meeting with Jessie Thomas.",
                    "answer": [],
                    "domains": ["calendar", "crm"],
                },
            )

            result = benchmark.evaluate(
                task,
                "I could not book the meeting because no assigned contact for Jessie Thomas was found.",
                run_metadata={"function_calls": [], "error": ""},
            )

            self.assertTrue(result.success)
            self.assertEqual(result.score, 1.0)
            self.assertTrue(result.details["exact_match"])
            self.assertTrue(result.details["correct"])
            self.assertTrue(result.details["prediction_substantive"])


def benchmark_module_sandbox(data_dir: Path):
    from benchmark.workbench import WorkBenchSandbox

    return WorkBenchSandbox(data_dir)


if __name__ == "__main__":
    unittest.main()

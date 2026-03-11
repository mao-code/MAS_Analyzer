import json
import tarfile
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import main as main_module


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict[str, object]:
        return self._payload


def _fake_virtual_post(self, url, **kwargs):  # type: ignore[no-untyped-def]
    payload = kwargs.get("json", {})
    api_name = payload.get("api_name", "")
    tool_input = payload.get("tool_input", "{}")
    return _FakeResponse(
        {
            "error": "",
            "response": {
                "api_name": api_name,
                "tool_input": tool_input,
                "result": "ok",
            },
        }
    )


class TestMainStableToolBenchSmoke(unittest.TestCase):
    def test_load_tasks_can_auto_download_server_assets(self) -> None:
        from benchmark.stabletoolbench import StableToolBenchBenchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            query_root = base / "solvable_queries"
            assets_root = base / "server_assets"
            (query_root / "test_instruction").mkdir(parents=True)
            (query_root / "test_query_ids").mkdir(parents=True)

            query_rows = [
                {
                    "query_id": "q1",
                    "query": "Look up title 123.",
                    "relevant APIs": [],
                    "api_list": [],
                }
            ]
            query_ids = {"q1": 0}
            (query_root / "test_instruction" / "G1_instruction.json").write_text(
                json.dumps(query_rows),
                encoding="utf-8",
            )
            (query_root / "test_query_ids" / "G1_instruction.json").write_text(
                json.dumps(query_ids),
                encoding="utf-8",
            )

            archive_root = base / "archives"
            archive_root.mkdir()
            tools_src = archive_root / "tools"
            cache_src = archive_root / "tool_response_cache"
            (tools_src / "Advertising").mkdir(parents=True)
            (cache_src / "Advertising").mkdir(parents=True)
            (tools_src / "Advertising" / "demo.json").write_text("{}", encoding="utf-8")
            (cache_src / "Advertising" / "demo.json").write_text("{}", encoding="utf-8")

            tools_archive = archive_root / "toolenv2404_filtered.tar.gz"
            with tarfile.open(tools_archive, "w:gz") as handle:
                handle.add(tools_src, arcname="tools")

            cache_archive = archive_root / "server_cache.zip"
            with zipfile.ZipFile(cache_archive, "w") as handle:
                for path in cache_src.rglob("*"):
                    if path.is_file():
                        handle.write(path, arcname=path.relative_to(archive_root))

            def _fake_hf_download(*, repo_id, filename, repo_type):  # type: ignore[no-untyped-def]
                self.assertEqual(repo_type, "dataset")
                if filename == "toolenv2404_filtered.tar.gz":
                    return str(tools_archive)
                if filename == "server_cache.zip":
                    return str(cache_archive)
                raise AssertionError(f"Unexpected asset request: {repo_id}/{filename}")

            benchmark = StableToolBenchBenchmark(
                config={
                    "query_root": str(query_root),
                    "task_sets": ["G1_instruction"],
                    "auto_download": False,
                    "auto_download_server_assets": True,
                    "server_assets_root": str(assets_root),
                }
            )

            with patch("huggingface_hub.hf_hub_download", new=_fake_hf_download):
                tasks = benchmark.load_tasks(task_limit=1)

            self.assertEqual(len(tasks), 1)
            self.assertTrue((assets_root / "tools" / "Advertising" / "demo.json").exists())
            self.assertTrue(
                (assets_root / "tool_response_cache" / "Advertising" / "demo.json").exists()
            )

    def test_run_stabletoolbench_one_task_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            query_root = base / "solvable_queries"
            (query_root / "test_instruction").mkdir(parents=True)
            (query_root / "test_query_ids").mkdir(parents=True)
            cfg_path = base / "experiment.toml"
            out_dir = base / "outputs"

            query_rows = [
                {
                    "query_id": "q1",
                    "query": (
                        "Find the featured title for id 123 and tell me whether it is family friendly."
                    ),
                    "relevant APIs": [],
                    "api_list": [
                        {
                            "category_name": "Entertainment",
                            "tool_name": "DemoTool",
                            "api_name": "Lookup Title",
                            "api_description": "Fetches a title by id.",
                            "method": "GET",
                            "required_parameters": [
                                {
                                    "name": "title_id",
                                    "type": "string",
                                    "description": "Title identifier.",
                                    "default": "123",
                                }
                            ],
                            "optional_parameters": [],
                        }
                    ],
                }
            ]
            query_ids = {"q1": 0}

            (query_root / "test_instruction" / "G1_instruction.json").write_text(
                json.dumps(query_rows, indent=2),
                encoding="utf-8",
            )
            (query_root / "test_query_ids" / "G1_instruction.json").write_text(
                json.dumps(query_ids, indent=2),
                encoding="utf-8",
            )

            cfg_path.write_text(
                textwrap.dedent(
                    f"""
                    [openrouter]
                    api_key = ""

                    [experiment]
                    output_dir = "{out_dir.as_posix()}"
                    runs_per_task = 1
                    seed = 42

                    [mas]
                    levels = 1
                    intra_level_link_ratio = 1.0
                    full_linked = true
                    number_of_agents = 1
                    agent_types = ["general"]
                    communication_count_internally = 0
                    turn_mode = "single_turn"
                    max_turns = 1

                    [models]
                    default = "gpt-4.1-mini"

                    [stabletoolbench]
                    query_root = "{query_root.as_posix()}"
                    task_sets = ["G1_instruction"]
                    auto_download = false
                    virtual_server_url = "http://localhost:8080/virtual"
                    eval_mode = "heuristic"
                    enable_tools = true
                    max_tool_iterations = 4
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with patch("requests.sessions.Session.post", new=_fake_virtual_post):
                exit_code = main_module.main(
                    [
                        "run",
                        "--config",
                        str(cfg_path),
                        "--benchmark",
                        "stabletoolbench",
                        "--task-limit",
                        "1",
                        "--runs-per-task",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)

            run_dirs = [item for item in out_dir.iterdir() if item.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            root = run_dirs[0]

            task_dir = root / "stabletoolbench" / "q1"
            self.assertTrue((task_dir / "run_0.trace.jsonl").exists())
            self.assertTrue((task_dir / "run_0.eval.json").exists())
            self.assertTrue((task_dir / "descriptor.json").exists())
            self.assertTrue((task_dir / "descriptor.csv").exists())
            self.assertTrue((task_dir / "analysis.json").exists())

            analysis = json.loads((task_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertGreater(analysis["descriptor"]["C4_tool_calls_total"], 0.0)

            eval_payload = json.loads((task_dir / "run_0.eval.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_payload["details"]["eval_mode"], "heuristic")
            run_metadata = eval_payload["details"]["run_metadata"]
            self.assertIn("lookup_title_for_demotool", run_metadata.get("tool_call_counts", {}))

            settings = json.loads((root / "experiment_settings.json").read_text(encoding="utf-8"))
            self.assertEqual(settings["benchmark"]["name"], "stabletoolbench")
            self.assertIn("stabletoolbench_virtual_api", settings["tools"]["agent_runtime_tools"])
            self.assertTrue((root / "summary.json").exists())
            self.assertTrue((root / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()

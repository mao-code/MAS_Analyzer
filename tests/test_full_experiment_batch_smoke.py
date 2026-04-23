import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import scripts.full_experiment as batch_module


class TestFullExperimentBatchSmoke(unittest.TestCase):
    def test_system_extra_args_ignores_raw_model_flags(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "MAS_GLOBAL_ARGS": (
                    "--default-model global-default "
                    "--judge-model global-judge "
                    "--peer-artifact-max-chars 123"
                ),
                "SAS_ARGS": "--judge-model sas-judge --communication-budget 9",
            },
            clear=False,
        ):
            args = batch_module.system_extra_args("sas", model=None)

        self.assertEqual(
            args,
            ["--peer-artifact-max-chars", "123", "--communication-budget", "9"],
        )

    def test_batch_run_honors_disable_live_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config_dir = base / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            decrypted_path = base / "browsecomp_decrypted.jsonl"
            qrel_evidence = base / "qrel_evidence.txt"
            qrel_golds = base / "qrel_golds.txt"
            output_root = base / "artifacts"

            row = {
                "query_id": "q1",
                "query": "Which city is called the Eternal City?",
                "answer": "Rome",
                "gold_docs": [{"docid": "100", "text": "Rome is known as Eternal City.", "url": "u"}],
                "evidence_docs": [{"docid": "200", "text": "Evidence text", "url": "u"}],
                "negative_docs": [],
            }
            decrypted_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qrel_evidence.write_text("q1 Q0 200 1\n", encoding="utf-8")
            qrel_golds.write_text("q1 Q0 100 1\n", encoding="utf-8")

            (config_dir / "browsecomp.toml").write_text(
                textwrap.dedent(
                    f"""
                    [openrouter]
                    api_key = ""

                    [experiment]
                    output_dir = "{output_root.as_posix()}"
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
                    default = "openai/gpt-4o-mini"

                    [browsecomp]
                    decrypted_path = "{decrypted_path.as_posix()}"
                    qrel_evidence_path = "{qrel_evidence.as_posix()}"
                    qrel_golds_path = "{qrel_golds.as_posix()}"
                    eval_mode = "substring"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            options = batch_module.BatchOptions(
                experiment_id="batch_smoke",
                output_root=output_root,
                config_dir=config_dir,
                benchmarks=["browsecomp"],
                task_limit=1,
                runs_per_task=1,
                retry_failures=0,
                max_parallel=1,
                models=None,
                final_vote_mode=None,
                benchmark_eval_judge_model=None,
                skip_setup=True,
                setup_only=False,
                no_dynamic_roles=False,
                list_benchmarks=False,
            )

            with patch.object(batch_module, "SYSTEMS", [("sas", "sas", 1, 1, 1, 0)]):
                with patch.dict(
                    "os.environ",
                    {
                        "MAS_DISABLE_LIVE_LLM": "1",
                        "MAS_REQUIRE_LIVE_LLM": "1",
                        "OPENROUTER_API_KEY": "",
                    },
                    clear=False,
                ):
                    exit_code = batch_module.batch_run(options)

            self.assertEqual(exit_code, 0)
            summary_path = output_root / "batch_smoke" / "browsecomp" / "sas" / "summary.json"
            self.assertTrue(summary_path.exists())

    def test_batch_run_creates_one_experiment_root_per_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            config_dir = base / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            decrypted_path = base / "browsecomp_decrypted.jsonl"
            qrel_evidence = base / "qrel_evidence.txt"
            qrel_golds = base / "qrel_golds.txt"
            output_root = base / "artifacts"

            row = {
                "query_id": "q1",
                "query": "Which city is called the Eternal City?",
                "answer": "Rome",
                "gold_docs": [{"docid": "100", "text": "Rome is known as Eternal City.", "url": "u"}],
                "evidence_docs": [{"docid": "200", "text": "Evidence text", "url": "u"}],
                "negative_docs": [],
            }
            decrypted_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            qrel_evidence.write_text("q1 Q0 200 1\n", encoding="utf-8")
            qrel_golds.write_text("q1 Q0 100 1\n", encoding="utf-8")

            (config_dir / "browsecomp.toml").write_text(
                textwrap.dedent(
                    f"""
                    [openrouter]
                    api_key = ""

                    [experiment]
                    output_dir = "{output_root.as_posix()}"
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
                    default = "openai/gpt-4o-mini"

                    [browsecomp]
                    decrypted_path = "{decrypted_path.as_posix()}"
                    qrel_evidence_path = "{qrel_evidence.as_posix()}"
                    qrel_golds_path = "{qrel_golds.as_posix()}"
                    eval_mode = "substring"
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            models = ["google/gemma-4-31b-it", "openai/gpt-oss-120b"]
            options = batch_module.BatchOptions(
                experiment_id="batch_multi_model",
                output_root=output_root,
                config_dir=config_dir,
                benchmarks=["browsecomp"],
                task_limit=1,
                runs_per_task=1,
                retry_failures=0,
                max_parallel=1,
                models=models,
                final_vote_mode=None,
                benchmark_eval_judge_model=None,
                skip_setup=True,
                setup_only=False,
                no_dynamic_roles=False,
                list_benchmarks=False,
            )

            with patch.object(batch_module, "SYSTEMS", [("sas", "sas", 1, 1, 1, 0)]):
                with patch.dict(
                    "os.environ",
                    {
                        "MAS_DISABLE_LIVE_LLM": "1",
                        "MAS_REQUIRE_LIVE_LLM": "1",
                        "OPENROUTER_API_KEY": "",
                    },
                    clear=False,
                ):
                    exit_code = batch_module.batch_run(options)

            self.assertEqual(exit_code, 0)
            for model in models:
                experiment_id = f"batch_multi_model__{batch_module._model_slug(model)}"
                summary_path = output_root / experiment_id / "browsecomp" / "sas" / "summary.json"
                self.assertTrue(summary_path.exists())
            manifest_path = output_root / "batch_multi_model" / "multi_model_manifest.json"
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()

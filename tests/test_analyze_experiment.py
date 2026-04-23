import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_experiment import analyze_experiment


class TestAnalyzeExperiment(unittest.TestCase):
    def _write_summary_csv(self, system_dir: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = sorted({key for row in rows for key in row})
        with (system_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_trace_metrics(
        self,
        system_dir: Path,
        *,
        task_id: str,
        run_index: int,
        success: float,
        accuracy: float,
        latency_e2e: float,
        token_total: float,
        communication_count: float,
        handoff_count: float,
        tool_calls_total: float,
        tool_error_count: float,
    ) -> None:
        task_dir = system_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": task_id,
            "run_index": run_index,
            "evaluation": {"score": accuracy, "success": bool(success), "details": {}},
            "metrics": {
                "success": bool(success),
                "completion": True,
                "accuracy": accuracy,
                "score": accuracy,
                "latency_total": latency_e2e,
                "latency_e2e": latency_e2e,
                "tokens_total": token_total,
                "token_total": token_total,
                "tool_calls_total": tool_calls_total,
                "tool_fail_total": tool_error_count,
                "communication_count": communication_count,
                "handoff_count": handoff_count,
            },
            "runtime": {"topology": system_dir.name, "run_index": run_index},
        }
        (task_dir / f"run_{run_index}.trace_metrics.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_generates_paper_aligned_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            experiment_root = base / "exp1"
            sas_dir = experiment_root / "browsecomp" / "sas"
            mas_dir = experiment_root / "browsecomp" / "only_voting"
            sas_dir.mkdir(parents=True, exist_ok=True)
            mas_dir.mkdir(parents=True, exist_ok=True)

            common = {
                "benchmark": "browsecomp",
                "runs": 4,
                "tool_error_rate": 0.0,
                "tool_calls_total": 4.0,
            }
            self._write_summary_csv(
                sas_dir,
                [
                    {
                        **common,
                        "system_label": "sas",
                        "topology": "sas",
                        "task_id": "q1",
                        "eval_avg_score": 0.25,
                        "success_rate": 0.25,
                        "pass_at_1": 0.25,
                        "pass_at_3": 0.75,
                        "pass_at_5": "",
                        "pass_at_8": "",
                        "stability": 0.75,
                        "tokens_total": 100.0,
                        "cost_per_success": 400.0,
                        "tokens_cv": 0.2,
                        "communication_count": 0.0,
                        "handoff_count": 1.0,
                        "agent_to_agent_communication_count": 0.0,
                        "system_mediated_communication_count": 0.0,
                    },
                    {
                        **common,
                        "system_label": "sas",
                        "topology": "sas",
                        "task_id": "q2",
                        "eval_avg_score": 0.50,
                        "success_rate": 0.50,
                        "pass_at_1": 0.50,
                        "pass_at_3": 1.0,
                        "pass_at_5": "",
                        "pass_at_8": "",
                        "stability": 0.50,
                        "tokens_total": 120.0,
                        "cost_per_success": 240.0,
                        "tokens_cv": 0.1,
                        "communication_count": 0.0,
                        "handoff_count": 1.0,
                        "agent_to_agent_communication_count": 0.0,
                        "system_mediated_communication_count": 0.0,
                    },
                ],
            )
            for run_index, (task_id, success, accuracy, latency_e2e, token_total) in enumerate(
                [
                    ("q1", 1.0, 0.25, 10.0, 100.0),
                    ("q1", 0.0, 0.25, 12.0, 110.0),
                    ("q2", 1.0, 0.50, 14.0, 120.0),
                    ("q2", 0.0, 0.50, 16.0, 130.0),
                ]
            ):
                self._write_trace_metrics(
                    sas_dir,
                    task_id=task_id,
                    run_index=run_index,
                    success=success,
                    accuracy=accuracy,
                    latency_e2e=latency_e2e,
                    token_total=token_total,
                    communication_count=0.0,
                    handoff_count=1.0,
                    tool_calls_total=4.0,
                    tool_error_count=0.0,
                )
            self._write_summary_csv(
                mas_dir,
                [
                    {
                        **common,
                        "system_label": "only_voting",
                        "topology": "only_voting",
                        "task_id": "q1",
                        "eval_avg_score": 0.75,
                        "success_rate": 0.75,
                        "pass_at_1": 0.75,
                        "pass_at_3": 1.0,
                        "pass_at_5": "",
                        "pass_at_8": "",
                        "stability": 0.80,
                        "tokens_total": 180.0,
                        "cost_per_success": 240.0,
                        "tokens_cv": 0.3,
                        "communication_count": 6.0,
                        "handoff_count": 4.0,
                        "agent_to_agent_communication_count": 5.0,
                        "system_mediated_communication_count": 1.0,
                    },
                    {
                        **common,
                        "system_label": "only_voting",
                        "topology": "only_voting",
                        "task_id": "q2",
                        "eval_avg_score": 0.75,
                        "success_rate": 0.75,
                        "pass_at_1": 0.75,
                        "pass_at_3": 1.0,
                        "pass_at_5": "",
                        "pass_at_8": "",
                        "stability": 0.90,
                        "tokens_total": 160.0,
                        "cost_per_success": 213.33,
                        "tokens_cv": 0.25,
                        "communication_count": 5.0,
                        "handoff_count": 3.0,
                        "agent_to_agent_communication_count": 4.0,
                        "system_mediated_communication_count": 1.0,
                    },
                ],
            )
            for run_index, (task_id, success, accuracy, latency_e2e, token_total) in enumerate(
                [
                    ("q1", 1.0, 0.75, 20.0, 180.0),
                    ("q1", 1.0, 0.75, 22.0, 190.0),
                    ("q2", 1.0, 0.75, 24.0, 160.0),
                    ("q2", 0.0, 0.75, 26.0, 170.0),
                ]
            ):
                self._write_trace_metrics(
                    mas_dir,
                    task_id=task_id,
                    run_index=run_index,
                    success=success,
                    accuracy=accuracy,
                    latency_e2e=latency_e2e,
                    token_total=token_total,
                    communication_count=5.5,
                    handoff_count=3.5,
                    tool_calls_total=4.0,
                    tool_error_count=0.0,
                )

            output_dir = experiment_root / "analysis"
            analysis = analyze_experiment(experiment_root, output_dir)

            self.assertTrue((output_dir / "task_level_metrics.csv").exists())
            self.assertTrue((output_dir / "system_level_metrics.csv").exists())
            self.assertTrue((output_dir / "report.md").exists())
            self.assertTrue((output_dir / "analysis.json").exists())
            self.assertIn("headline", analysis)

            plot_paths = analysis["artifacts"]["plots"]["browsecomp"]
            self.assertTrue(any(path.endswith("browsecomp_system_scorecard.png") for path in plot_paths))
            self.assertTrue(any("browsecomp_pass_at_k" in path for path in plot_paths))
            self.assertTrue(any("success_vs_tokens_frontier" in path for path in plot_paths))
            self.assertTrue(any("vs_sas_tradeoff" in path for path in plot_paths))
            self.assertTrue(any("coordination_breakdown" in path for path in plot_paths))
            self.assertFalse(any("task_score_heatmap" in path for path in plot_paths))
            self.assertFalse(any("vs_sas_delta_heatmap" in path for path in plot_paths))
            self.assertFalse(any("cost_predictability" in path for path in plot_paths))


if __name__ == "__main__":
    unittest.main()

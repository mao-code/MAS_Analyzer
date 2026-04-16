from __future__ import annotations

import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.econ_eval.run_pipeline import PipelineConfig, run_economic_pipeline


if __name__ == "__main__":
    experiment_root = project_root / "20260411T145457Z"
    output_dir = experiment_root / "analysis" / "econ_eval_browsecomp"

    payload = run_economic_pipeline(
        experiment_root=experiment_root,
        output_dir=output_dir,
        benchmarks=["browsecomp"],
        config=PipelineConfig(
            pass_k_values=(1, 3, 5, 8),
            primary_method="arithmetic",
            methods=("arithmetic", "geometric", "mahalanobis", "topsis"),
        ),
    )

    print("Done:", payload["output_dir"])

from __future__ import annotations

import pandas as pd

from .regime_classification import classify_regime


def compare_sas_vs_mas(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute utility deltas and gain/cost decomposition against SAS baseline."""
    sas = summary_df[summary_df["system_type"] == "SAS"][
        ["benchmark", "Q", "C", "U", "success_rate", "completion_rate"]
    ].rename(
        columns={
            "Q": "Q_sas",
            "C": "C_sas",
            "U": "U_sas",
            "success_rate": "success_rate_sas",
            "completion_rate": "completion_rate_sas",
        }
    )

    mas = summary_df[summary_df["system_type"] == "MAS"].copy()
    merged = mas.merge(sas, on="benchmark", how="left")
    merged = merged.dropna(subset=["U_sas"])

    merged["Q_mas"] = merged["Q"]
    merged["C_mas"] = merged["C"]
    merged["U_mas"] = merged["U"]

    merged["G"] = merged["Q_mas"] - merged["Q_sas"]
    merged["K"] = merged["C_mas"] - merged["C_sas"]
    merged["delta_U"] = merged["U_mas"] - merged["U_sas"]

    merged["collaboration_regime"] = merged.apply(
        lambda row: classify_regime(float(row["G"]), float(row["K"])),
        axis=1,
    )

    keep = [
        "benchmark",
        "system_name",
        "aggregation_method",
        "U_sas",
        "U_mas",
        "delta_U",
        "G",
        "K",
        "collaboration_regime",
        "success_rate_sas",
        "completion_rate_sas",
        "success_rate",
        "completion_rate",
    ]
    return merged[keep].sort_values(["benchmark", "delta_U"], ascending=[True, False])

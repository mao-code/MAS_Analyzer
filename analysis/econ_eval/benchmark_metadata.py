from __future__ import annotations

import pandas as pd


BENCHMARK_METADATA = {
    "finance_agent": {
        "capability_focus": "financial reasoning and tool use",
        "workflow_property": "tool-intensive workflows",
    },
    "browsecomp": {
        "capability_focus": "open-web retrieval and evidence synthesis",
        "workflow_property": "open-ended search",
    },
    "plancraft": {
        "capability_focus": "long-horizon planning",
        "workflow_property": "sequential planning",
    },
    "workbench": {
        "capability_focus": "task decomposition and coordination",
        "workflow_property": "decomposable tasks",
    },
    "agentbench": {
        "capability_focus": "general agentic problem solving",
        "workflow_property": "mixed interactive workflows",
    },
    "stabletoolbench": {
        "capability_focus": "API/tool orchestration robustness",
        "workflow_property": "tool-intensive workflows",
    },
    "webshop": {
        "capability_focus": "shopping decision sequences",
        "workflow_property": "sequential planning",
    },
    "scicode": {
        "capability_focus": "program synthesis and verification",
        "workflow_property": "program synthesis / deterministic verification",
    },
}


def attach_benchmark_metadata(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["benchmark_key"] = frame["benchmark"].astype(str).str.lower()
    frame["capability_focus"] = frame["benchmark_key"].map(
        {k: v["capability_focus"] for k, v in BENCHMARK_METADATA.items()}
    )
    frame["workflow_property"] = frame["benchmark_key"].map(
        {k: v["workflow_property"] for k, v in BENCHMARK_METADATA.items()}
    )
    frame = frame.drop(columns=["benchmark_key"])
    return frame

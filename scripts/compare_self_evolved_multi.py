#!/usr/bin/env python3
"""Multi-benchmark comparison: self-evolved (dynamic) MAS vs the static MAS / SAS
baselines, across several benchmarks at once.

For each benchmark it reuses ``compare_self_evolved_vs_static.run_one`` (per-benchmark
tables + figures), then draws one cross-benchmark overview figure (grouped success-rate
bars) and writes a top-level INDEX.md. The static runs are NOT re-executed.

Usage:
  python scripts/compare_self_evolved_multi.py \
    --static-experiment artifacts/full_experiment/20260427T134706Z__google_gemma_4_31b_it_nitro \
    --self-evolved-experiment artifacts/full_experiment/smoke_selfevo__google_gemma_4_31b_it_nitro \
    --benchmarks browsecomp,plancraft,stabletoolbench,workbench \
    --model gemma-4-31b-it \
    --out-dir artifacts/full_experiment/smoke_selfevo__google_gemma_4_31b_it_nitro/comparison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.compare_self_evolved_vs_static import (  # noqa: E402
    PRETTY,
    SELF_EVOLVED,
    run_one,
)


def fig_overview(results: dict[str, dict], out: Path, model: str) -> None:
    """Grouped bars: success rate per benchmark, one bar per system (common task set)."""
    benchmarks = list(results)
    # Union of systems across benchmarks, preserving the per-benchmark display order.
    systems: list[str] = []
    for r in results.values():
        for s in r["systems"]:
            if s not in systems:
                systems.append(s)

    import numpy as np

    n_b, n_s = len(benchmarks), len(systems)
    width = 0.8 / max(n_s, 1)
    fig, ax = plt.subplots(figsize=(2.6 * n_b + 3, 6))
    x = np.arange(n_b)
    for i, s in enumerate(systems):
        vals = [results[b]["aggs_common"].get(s, {}).get("success") for b in benchmarks]
        vals = [v if v is not None else 0.0 for v in vals]
        is_se = s == SELF_EVOLVED
        ax.bar(
            x + i * width,
            vals,
            width,
            label=PRETTY.get(s, s),
            color="#d62728" if is_se else None,
            edgecolor="black" if is_se else "none",
            linewidth=1.2 if is_se else 0,
        )
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(benchmarks, fontsize=10)
    ax.set_ylabel("Success rate (common task set)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Self-evolved (dynamic) vs static MAS / SAS — {model}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-experiment", required=True,
                    help="Reference experiment root holding <benchmark>/<system>/summary.csv")
    ap.add_argument("--self-evolved-experiment", required=True,
                    help="Smoke experiment root holding <benchmark>/self_evolved/summary.csv")
    ap.add_argument("--benchmarks", required=True, help="Comma-separated benchmark names")
    ap.add_argument("--model", default="gemma-4-31b-it", help="Model label (titles only)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    static_exp = Path(args.static_experiment)
    se_exp = Path(args.self_evolved_experiment)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    results: dict[str, dict] = {}
    for b in benchmarks:
        res = run_one(
            static_root=static_exp / b,
            self_evolved_summary=se_exp / b / SELF_EVOLVED / "summary.csv",
            out_dir=out_dir / b,
            benchmark=b,
            model=args.model,
            verbose=False,
        )
        if res is not None:
            results[b] = res

    if not results:
        print("No benchmarks produced a comparison (self_evolved summaries missing/empty).")
        return 1

    fig_overview(results, out_dir / "fig_overview_success.png", args.model)

    # Top-level index + a compact success-rate table.
    lines = [
        f"# Self-evolved (dynamic) MAS vs static MAS / SAS — {args.model}",
        "",
        "Per-benchmark tables and figures are in each benchmark subdirectory; "
        "`fig_overview_success.png` is the cross-benchmark success-rate summary.",
        "",
        "## Success rate on the common (apples-to-apples) task set",
        "",
    ]
    systems: list[str] = []
    for r in results.values():
        for s in r["systems"]:
            if s not in systems:
                systems.append(s)
    header = "| Benchmark (n) | " + " | ".join(PRETTY.get(s, s) for s in systems) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(systems) + 1))
    for b, r in results.items():
        n = len(r["common"])
        cells = []
        for s in systems:
            v = r["aggs_common"].get(s, {}).get("success")
            cells.append("—" if v is None else f"{v:.2f}")
        lines.append(f"| {b} ({n}) | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"- **Benchmarks compared:** {', '.join(results)}",
        f"- **Static/SAS baselines:** `{static_exp}` (3 runs/task, not re-run)",
        f"- **Self-evolved smoke:** `{se_exp}` (custom OpenRouter harness)",
        "",
        "> Smoke uses a tiny task count per benchmark — success deltas here are "
        "plumbing/sanity signals, not statistically powered claims.",
    ]
    (out_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote multi-benchmark comparison to {out_dir}")
    print((out_dir / "INDEX.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

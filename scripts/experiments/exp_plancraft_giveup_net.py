#!/usr/bin/env python3
"""EXPERIMENTAL / throwaway: PlanCraft MANTA with a give-up verification net in _finalize.

Hypothesis
----------
MANTA's PlanCraft losses are almost entirely confident FALSE "impossible" give-ups (27/90 baseline
runs answer "impossible", 0 correct; these runs look process-clean so no auditor/repair/reflection
mechanism can see them). A give-up answer should have to survive one adversarial refutation attempt
before it is allowed to finalize:

  - if the final answer is give-up-shaped (starts with "impossible", UNSUPPORTED, empty/blocked),
    ask one refuter agent to produce a concrete next action instead;
  - if the refuter produces a substantive action -> adopt it (wrongly acting costs one recoverable
    env step; wrongly stopping forfeits the task);
  - if the refuter concurs -> keep the give-up (genuinely-impossible tasks stay correct).

The trigger is the ANSWER'S SHAPE, never the benchmark name — substantive answers are a strict
no-op, so other benchmarks are unaffected by construction.

This script does NOT edit core code: it wraps ``SelfEvolvedEngine._finalize`` at runtime and runs
the same 15 tasks (seed 42) as the max_turns=5 experiment, with the BASELINE config otherwise
(max_turns=2), so the net is the only lever. Results go to a new experiment id.

Usage
-----
    .venv/bin/python scripts/experiments/exp_plancraft_giveup_net.py --dry-run     # patch + detector self-test
    .venv/bin/python scripts/experiments/exp_plancraft_giveup_net.py               # run 15 tasks x 1 run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from rerun_plancraft_runs12 import render_config_toml  # noqa: E402

BASELINE = ROOT / "artifacts/full_experiment/plancraft_bf16_skill_learning_fixed/plancraft/self_evolved"
NEW_EXPERIMENT_ID = "plancraft_giveup_net"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-maoxunhuang-Desktop-MAS-Analyzer/"
    "2d78b8b0-e7dd-4ab4-802e-41a597f345a1/scratchpad"
)
BF16_ORDER = "WandB,Novita,Venice"
BF16_IGNORE = "Chutes,DeepInfra,SiliconFlow,Parasail,Phala,ModelRun,Together,SambaNova,Cerebras"

_GIVE_UP_PREFIXES = ("impossible", "unsupported")


def is_give_up(text: str) -> bool:
    """Answer-shape give-up detector (benchmark-agnostic)."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in _GIVE_UP_PREFIXES):
        return True
    from answer_utils import classify_answer_mode

    return classify_answer_mode(stripped) in {"blocked", "empty"}


def _refuter_messages(task_prompt: str, give_up_answer: str) -> list[dict[str, str]]:
    system_msg = (
        "You are an adversarial verifier. A multi-agent system is about to conclude that a task "
        "cannot be completed. Impossibility claims are frequently based on a hallucinated missing "
        "prerequisite, so your job is to try to REFUTE the claim.\n"
        "- If ANY concrete progress is possible, respond with EXACTLY one next action in the "
        "task's required output format (nothing else).\n"
        "- Only if you verify that no action can make progress, respond with a single line "
        "starting with 'impossible: ' naming the exact missing prerequisite.\n"
        "Do not explain. Output one line only."
    )
    user_msg = (
        f"TASK (full context and required output format):\n{task_prompt}\n\n"
        f"CLAIM TO REFUTE — the system wants to stop with:\n{give_up_answer}\n\n"
        "Your one-line response:"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def apply_patch() -> None:
    import MAS.self_evolved.engine as engmod

    original_finalize = engmod.SelfEvolvedEngine._finalize
    if getattr(engmod.SelfEvolvedEngine, "_give_up_net_patched", False):
        raise SystemExit("PATCH ABORT: _finalize already patched in this process.")

    def patched_finalize(self, state, result, decision):  # noqa: ANN001
        answer = original_finalize(self, state, result, decision)
        if not is_give_up(answer):
            return answer

        outcome = "kept_give_up"
        new_answer = answer
        try:
            res = self.llm_client.generate(
                prompt=_refuter_messages(str(state.get("task_prompt", "")), str(answer)),
                agent_type="general",
                task_id=str(state.get("task_id", "")),
                run_index=int(state.get("run_index", 0)),
                agent_id="give_up_refuter",
                temperature=0.0,
            )
            text = str(getattr(res, "text", "") or "").strip()
            if not bool(getattr(res, "mock_used", False)) and text and not is_give_up(text):
                # Refuter produced a concrete action: overturn the give-up.
                new_answer = text.splitlines()[0].strip()
                state["final_answer"] = new_answer
                outcome = "overturned"
        except Exception:  # net must never break finalize; keep the give-up on any error
            outcome = "refuter_error"

        try:
            self._emit_meta_event(
                state,
                actor="orchestrator",
                event_type="verify",
                node_name="give_up_verification",
                payload={
                    "outcome": outcome,
                    "give_up_answer": str(answer)[:300],
                    "replacement": (new_answer[:300] if outcome == "overturned" else ""),
                },
            )
        except Exception:
            pass
        return new_answer

    engmod.SelfEvolvedEngine._finalize = patched_finalize
    engmod.SelfEvolvedEngine._give_up_net_patched = True
    print("[patch] SelfEvolvedEngine._finalize wrapped with give-up verification net  ✓")


def self_test() -> None:
    cases = [
        ("impossible: missing fuel for smelting", True),
        ("Impossible: no glazer block", True),
        ("", True),
        ("move: from [I14] to [B1] with quantity 1", False),
        ("smelt: from [I3] to [O1] with quantity 4", False),
    ]
    for text, expected in cases:
        got = is_give_up(text)
        assert got == expected, f"is_give_up({text!r}) = {got}, expected {expected}"
    print("[self-test] give-up detector: 5/5 cases pass  ✓")


def build_config(task_limit: int, runs_per_task: int) -> Path:
    settings = json.loads((BASELINE / "experiment_settings.json").read_text())
    snap = settings["raw_config_snapshot"]
    snap.setdefault("experiment", {})
    snap["experiment"]["task_limit"] = task_limit
    snap["experiment"]["runs_per_task"] = runs_per_task
    # Baseline settings otherwise: max_turns stays 2 so the net is the only lever.
    snap.setdefault("self_evolved", {})["skill_update_batch_size"] = 0
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cfg = SCRATCH / "exp_giveup_net_self_evolved.toml"
    cfg.write_text(render_config_toml(snap))
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-limit", type=int, default=15)
    ap.add_argument("--runs-per-task", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not (BASELINE / "experiment_settings.json").exists():
        raise SystemExit(f"Baseline settings not found under {BASELINE}")

    apply_patch()
    self_test()
    cfg = build_config(args.task_limit, args.runs_per_task)
    print(f"[config] wrote {cfg} (max_turns stays 2 — the net is the only lever)")

    if args.dry_run:
        print("[dry-run] patch + detector verified; not executing the experiment.")
        return 0

    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (put it in .env) — would fall back to mock.")

    os.environ["MAS_OPENROUTER_PROVIDER_ORDER"] = BF16_ORDER
    os.environ["MAS_OPENROUTER_IGNORE_PROVIDERS"] = BF16_IGNORE

    argv = [
        "run",
        "--config", str(cfg),
        "--benchmark", "plancraft",
        "--output-dir", str(ROOT / "artifacts" / "full_experiment"),
        "--output-layout", "hierarchical",
        "--experiment-id", NEW_EXPERIMENT_ID,
        "--system-label", "self_evolved",
        "--runs-per-task", str(args.runs_per_task),
        "--seed", str(args.seed),
        "--task-limit", str(args.task_limit),
        "--skill-update-batch-size", "0",
    ]
    print("[run] main.py " + " ".join(argv))

    import main as cli  # imported after the patch; runner picks up the patched engine

    rc = cli.main(argv)
    if rc != 0:
        raise SystemExit(f"main.py run exited {rc}")

    print("\n[summarize]")
    cli.main(["summarize-experiment", "--experiment-root", str(ROOT / "artifacts" / "full_experiment" / NEW_EXPERIMENT_ID)])
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

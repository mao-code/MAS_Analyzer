#!/usr/bin/env python3
"""EXPERIMENTAL: give-up verification net applied UNIFORMLY to ALL systems on PlanCraft.

Why
---
PlanCraft losses are ~100% confident false "impossible" give-ups for EVERY system (sas 35/90,
only_voting 34, self_evolved 33, debate systems 23-25 on bf16-pinned gemma; 0 of them correct —
val.small has no truly-impossible tasks). Giving only MANTA a give-up verification net would
confound the topology comparison, so this script wraps the ONE finalize chokepoint every system
shares — ``LangGraphMASEngine._finalize_node`` (static topologies compute their final answer in it;
MANTA's ``SelfEvolvedEngine._finalize`` also passes through it) — so all 8 systems get the
IDENTICAL net and the comparison stays unbiased.

The net (answer-shape triggered, benchmark-agnostic):
  - if the final answer is give-up-shaped ("impossible...", UNSUPPORTED, empty/blocked), one
    refuter call must try to overturn it via STRUCTURED refutation: check each claimed missing
    prerequisite against the task's own context (available actions, inventory, evidence) —
    hallucinated prerequisites often contradict the prompt itself;
  - refuter ends with "ACTION: <one concrete next action>" -> adopt it (wrongly acting costs one
    recoverable env step; wrongly stopping forfeits the task);
  - refuter ends with "CONFIRM: impossible: ..." (or anything else) -> keep the give-up.

No core code is edited: the patch is runtime-only. Configs are reconstructed verbatim from each
system's ``plancraft_bf16_pinned`` snapshot (same model/judge/bf16 provider pinning). Results go to
a NEW experiment id: ``plancraft_giveup_net_all``. Resume-safe (main.py skips existing runs).

Usage
-----
    # verify patch + detector only
    .venv/bin/python scripts/experiments/exp_plancraft_giveup_net_all.py --dry-run

    # all 8 systems, 15 tasks x 1 run (same seed-42 slice as prior experiments)
    .venv/bin/python scripts/experiments/exp_plancraft_giveup_net_all.py

    # full unbiased evaluation (matches the bf16_pinned baseline scale)
    .venv/bin/python scripts/experiments/exp_plancraft_giveup_net_all.py --task-limit 30 --runs-per-task 3

    # subset of systems ('manta' = self_evolved)
    .venv/bin/python scripts/experiments/exp_plancraft_giveup_net_all.py --systems sas,manta
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

PINNED = ROOT / "artifacts/full_experiment/plancraft_bf16_pinned/plancraft"
NEW_EXPERIMENT_ID = "plancraft_giveup_net_all"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-maoxunhuang-Desktop-MAS-Analyzer/"
    "2d78b8b0-e7dd-4ab4-802e-41a597f345a1/scratchpad"
)
BF16_ORDER = "WandB,Novita,Venice"
BF16_IGNORE = "Chutes,DeepInfra,SiliconFlow,Parasail,Phala,ModelRun,Together,SambaNova,Cerebras"
ALL_SYSTEMS = [
    "sas",
    "only_voting",
    "group_chat_debate",
    "fully_linked_debate",
    "orchestrator_no_discussion",
    "orchestrator_with_discussion",
    "orchestrator_tree_structure",
    "self_evolved",
]
ALIAS = {"manta": "self_evolved"}

# Online long-term skill learning for MANTA during the sweep (runs-per-update). 12 = ~4 tasks at
# 3 runs/task, matching config.py's default. Only self_evolved can learn; statics ignore it.
SKILL_BATCH = 12
# The evolving skill is written to an ISOLATED copy (seeded from the canonical skill) so the sweep
# never clobbers config/topology_skill.md. Promote it afterward with a cp if you want to keep it.
CANONICAL_SKILL = ROOT / "config" / "topology_skill.md"
EVOLVING_SKILL = SCRATCH / "giveup_net_all_evolving_skill.md"

_GIVE_UP_PREFIXES = ("impossible", "unsupported")


def _seed_evolving_skill() -> Path:
    """Seed the isolated evolving-skill copy from the canonical skill if it doesn't exist yet.

    Seed-if-absent keeps learned state across a kill/resume; delete the copy to start fresh.
    """
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not EVOLVING_SKILL.exists():
        EVOLVING_SKILL.write_text(
            CANONICAL_SKILL.read_text(encoding="utf-8") if CANONICAL_SKILL.exists() else ""
        )
    return EVOLVING_SKILL


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
        "You are an adversarial verifier. A multi-agent system is about to stop, claiming a task "
        "is impossible. Such claims frequently cite a hallucinated missing prerequisite, so your "
        "job is to try to REFUTE the claim by checking it against the task's OWN context.\n\n"
        "Procedure:\n"
        "1. List each prerequisite the claim says is missing.\n"
        "2. Check each one against the task context: the ACTUAL available actions, the ACTUAL "
        "inventory/state, and any stated rules. Quote the contradicting context if you find one "
        "(e.g. the claim requires a capability that an available action already provides).\n"
        "3. Decide.\n\n"
        "End your response with EXACTLY ONE final line:\n"
        "ACTION: <one concrete next action in the task's required output format>\n"
        "  (if any claimed prerequisite is contradicted by the context, or any progress is "
        "possible)\n"
        "CONFIRM: impossible: <the exact verified missing prerequisite>\n"
        "  (only if every claimed prerequisite survives your check)"
    )
    user_msg = (
        f"TASK CONTEXT (includes available actions and required output format):\n{task_prompt}\n\n"
        f"CLAIM TO REFUTE — the system wants to stop with:\n{give_up_answer}\n\n"
        "Run the procedure and end with your single ACTION:/CONFIRM: line."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _parse_refutation(text: str) -> str | None:
    """Return the overturning action, or None to keep the give-up."""
    for line in reversed([ln.strip() for ln in (text or "").splitlines() if ln.strip()]):
        if line.upper().startswith("ACTION:"):
            action = line[len("ACTION:") :].strip()
            return action if action and not is_give_up(action) else None
        if line.upper().startswith("CONFIRM:"):
            return None
    return None


def apply_patch() -> None:
    """Wrap the single finalize chokepoint shared by ALL systems (static + MANTA)."""
    import MAS.langgraph_engine as lgmod

    if getattr(lgmod.LangGraphMASEngine, "_give_up_net_patched", False):
        raise SystemExit("PATCH ABORT: _finalize_node already patched in this process.")
    original_node = lgmod.LangGraphMASEngine._finalize_node

    def patched_finalize_node(self, state):  # noqa: ANN001
        updates = original_node(self, state)
        answer = str(updates.get("final_answer", "") or "")
        if not is_give_up(answer):
            return updates

        task_id = str(state.get("task_id", ""))
        outcome = "kept_give_up"
        try:
            res = self.llm_client.generate(
                prompt=_refuter_messages(str(state.get("task_prompt", "")), answer),
                agent_type="general",
                task_id=task_id,
                run_index=int(state.get("run_index", 0)),
                agent_id="give_up_refuter",
                temperature=0.0,
            )
            if not bool(getattr(res, "mock_used", False)):
                action = _parse_refutation(str(getattr(res, "text", "") or ""))
                if action:
                    outcome = "overturned"
                    updates["final_answer"] = action
                    # Keep the finalize trace event consistent with the shipped answer.
                    for payload in updates.get("trace_payloads", []) or []:
                        event_payload = payload.get("payload")
                        if isinstance(event_payload, dict) and "final_answer" in event_payload:
                            event_payload["final_answer"] = action
                            event_payload["give_up_net"] = {
                                "outcome": outcome,
                                "overturned_give_up": answer[:300],
                            }
        except Exception:  # the net must never break finalize
            outcome = "refuter_error"

        print(
            f"[give-up-net] task={task_id} outcome={outcome} give_up={answer[:120]!r}"
            + (f" action={updates['final_answer'][:120]!r}" if outcome == "overturned" else ""),
            flush=True,
        )
        return updates

    lgmod.LangGraphMASEngine._finalize_node = patched_finalize_node
    lgmod.LangGraphMASEngine._give_up_net_patched = True
    print("[patch] LangGraphMASEngine._finalize_node wrapped (all 8 systems, one code path)  ✓")


def self_test() -> None:
    detector = [
        ("impossible: missing fuel", True),
        ("UNSUPPORTED", True),
        ("", True),
        ("move: from [I14] to [B1] with quantity 1", False),
    ]
    for text, expected in detector:
        assert is_give_up(text) == expected, f"is_give_up({text!r}) != {expected}"
    parser = [
        ("blah\nACTION: smelt: from [I3] to [O1] with quantity 1", "smelt: from [I3] to [O1] with quantity 1"),
        ("checked all\nCONFIRM: impossible: no source of sand", None),
        ("ACTION: impossible: still stuck", None),  # give-up smuggled into ACTION -> rejected
        ("no final line at all", None),
    ]
    for text, expected in parser:
        got = _parse_refutation(text)
        assert got == expected, f"_parse_refutation({text!r}) = {got!r}, expected {expected!r}"
    print("[self-test] detector 4/4, refutation parser 4/4  ✓")


def build_config(label: str, task_limit: int, runs_per_task: int) -> Path:
    settings = json.loads((PINNED / label / "experiment_settings.json").read_text())
    snap = settings["raw_config_snapshot"]
    snap.setdefault("experiment", {})
    snap["experiment"]["task_limit"] = task_limit
    snap["experiment"]["runs_per_task"] = runs_per_task
    if "self_evolved" in snap:
        se = snap["self_evolved"]
        # MANTA self-evolves online during the sweep; write to an isolated skill copy.
        se["skill_update_batch_size"] = SKILL_BATCH
        se["skill_path"] = str(_seed_evolving_skill())
    cfg_dir = SCRATCH / "giveup_net_all_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg_dir / f"{label}.toml"
    cfg.write_text(render_config_toml(snap))
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS), help="Comma list; 'manta'=self_evolved.")
    ap.add_argument("--task-limit", type=int, default=15)
    ap.add_argument("--runs-per-task", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-summarize", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labels = [ALIAS.get(s.strip().lower(), s.strip()) for s in args.systems.split(",") if s.strip()]
    for label in labels:
        if not (PINNED / label / "experiment_settings.json").exists():
            raise SystemExit(f"No plancraft_bf16_pinned snapshot for system {label!r}")

    apply_patch()
    self_test()

    if args.dry_run:
        for label in labels:
            cfg = build_config(label, args.task_limit, args.runs_per_task)
            print(f"[config] {label}: {cfg}")
        print("[dry-run] patch + tests + configs verified; not executing.")
        return 0

    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (put it in .env) — would fall back to mock.")

    os.environ["MAS_OPENROUTER_PROVIDER_ORDER"] = BF16_ORDER
    os.environ["MAS_OPENROUTER_IGNORE_PROVIDERS"] = BF16_IGNORE

    import main as cli  # imported after the patch

    for label in labels:
        cfg = build_config(label, args.task_limit, args.runs_per_task)
        argv = [
            "run",
            "--config", str(cfg),
            "--benchmark", "plancraft",
            "--output-dir", str(ROOT / "artifacts" / "full_experiment"),
            "--output-layout", "hierarchical",
            "--experiment-id", NEW_EXPERIMENT_ID,
            "--system-label", label,
            "--runs-per-task", str(args.runs_per_task),
            "--seed", str(args.seed),
            "--task-limit", str(args.task_limit),
        ]
        if label == "self_evolved":
            argv += ["--skill-update-batch-size", str(SKILL_BATCH)]
        print(f"\n[RUN] {label} (give-up net, bf16-pinned, resume-safe)", flush=True)
        rc = cli.main(argv)
        if rc != 0:
            raise SystemExit(f"{label}: main.py run exited {rc}")

    if not args.no_summarize:
        print("\n[summarize]")
        cli.main([
            "summarize-experiment",
            "--experiment-root", str(ROOT / "artifacts" / "full_experiment" / NEW_EXPERIMENT_ID),
        ])
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

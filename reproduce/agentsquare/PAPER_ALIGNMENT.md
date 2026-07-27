# AgentSquare Reproduction Alignment

This note records how the local AgentSquare reproduction maps to the official
`tsinghua-fib-lab/AgentSquare` implementation and what must be reported as a
methodological difference.

## Official Control Flow

The official search entrypoint is `search/agent_search.py`. Its loop is:

1. Load module archives for `planning`, `reasoning`, `tooluse`, and `memory`.
2. Start from the initial agent:
   `{planning: None, reasoning: IO, tooluse: None, memory: None}`.
3. Run LLM module evolution to generate one new module for each module type.
4. Construct evolved agents by replacing one slot with the generated module.
5. Test evolved agents on ALFWorld validation runs.
6. Run module recombination over the current candidate archive.
7. Use an LLM performance predictor to rank recombined agents.
8. Test the best predicted agent and update the current/best agent.
9. Save current agent, tested cases, module candidates, and archives.

The official code writes generated Python class code into task module files and
imports/runs that code during validation.

## Local Mapping

Local implementation:

- Runner: `reproduce/agentsquare/run_existing_benchmarks.py`
- Module definitions: `reproduce/agentsquare/modules.py`
- Runtime adapter: `reproduce/agentsquare/runtime_runner.py`
- Result summarizer: `reproduce/agentsquare/summarize_results.py`
- Status checker: `reproduce/agentsquare/status.py`
- Formal launcher: `scripts/baselines/run_agentsquare_formal.sh`

Implemented mapping:

| Official AgentSquare concept | Local implementation |
|---|---|
| Four module slots | `AgentSquareSpec(planning, reasoning, tooluse, memory)` |
| Initial agent | `planning=None, reasoning=IO, tooluse=None, memory=None` |
| Module archive | Base module pool plus generated prompt modules saved in `module_archive` |
| LLM module evolution | `agentsquare_module_evolution` call per search iteration |
| Evolved agents | One-slot replacement specs from generated modules and archive modules |
| Recombination | Grid-style recombination over base plus generated module pools |
| LLM predictor | `agentsquare_predictor` call per search iteration |
| Validation testing | Existing benchmark adapters over validation split |
| Best/current agent update | Search loop updates from validation score |
| Trace and artifacts | Per-run JSON, `predictor_rankings.json`, `iteration_result.json`, `search_results.json` |

## Deliberate Safety Difference

The official code executes generated Python module classes. The local
reproduction does not execute arbitrary LLM-generated Python code.

Instead:

- generated `name`, `thought`, and `prompt` are validated and converted into
  prompt-level `AgentSquareModule` definitions;
- generated `code` is preserved in module metadata as `code_audit_only`;
- benchmark environments, tools, side effects, and evaluators remain owned by
  existing repository adapters.

This is a safety constraint, not a silent equivalence claim. In reports, call
this method "AgentSquare-style safe reproduction" or explicitly state:

> We reproduce AgentSquare's modular search loop with LLM-generated prompt
> modules, recombination, LLM performance prediction, and validation-based
> selection. We do not execute arbitrary LLM-generated Python module code;
> generated code is retained only for audit.

## Verified Behavior

Current verified checks:

- Unit tests: `uv run pytest tests/test_reproduce_agentsquare.py -q`
- Live MATH mini-smoke:
  `outputs_agentsquare_reproduce/live_agentsquare_minismoke_math500_20260719T164553`
- All-benchmark mock pipeline:
  `/tmp/agentsquare_smoke/smoke_all_benches_mock`
- Resume smoke: rerunning the same mock `run-id` with `--resume` caused no LLM
  calls and reused search plus final artifacts.

The live mini-smoke verified that:

- LLM module evolution returned generated modules;
- LLM predictor selected a generated reasoning module;
- validation ran the generated module;
- final test loaded the selected generated module from `module_archive`.

## Formal Run

The launcher keeps the published table contract of 10 validation tasks, 30 final
test tasks, and 3 runs per final task. To keep the adapted reproduction
tractable with Gemma/OpenRouter, its default search budget is 3 search
iterations with at most 3 candidates per iteration. Override
`AGENTSQUARE_SEARCH_ITERATIONS` and `AGENTSQUARE_MAX_SEARCH_CANDIDATES` only if a
larger search budget is explicitly desired.

Launch:

```bash
tmux new -s agentsquare 'bash scripts/baselines/run_agentsquare_formal.sh agentsquare_gemma_10val_30test_T1'
```

Status:

```bash
uv run python -m reproduce.agentsquare.status \
  --run-root outputs_agentsquare_reproduce/agentsquare_gemma_10val_30test_T1
```

Summarize:

```bash
uv run python -m reproduce.agentsquare.summarize_results \
  --run-root outputs_agentsquare_reproduce/agentsquare_gemma_10val_30test_T1 \
  --output outputs_agentsquare_reproduce/agentsquare_gemma_10val_30test_T1/agentsquare_summary.json
```

#!/usr/bin/env bash
# =============================================================================
# Full self_evolved MAS experiment runner (Gemma4, custom OpenRouter harness).
#
# Design goals (per request):
#   * Optimized recommended config: 30 tasks x 3 runs x 4 benchmarks, matching the
#     static/SAS reference baseline so the comparison is apples-to-apples.
#   * Efficient: memory-LIGHT benchmarks run their tasks concurrently (async pool).
#   * Resumable: work is sharded into per-task UNITS; a completed unit drops a marker
#     and is skipped on a re-run, so resuming never re-runs finished tasks.
#   * Resource-safe: the heaviest benchmark (browsecomp — biggest contexts/longest tool
#     loops) runs LAST and ALONE/sequentially so it has the whole machine's RAM and isn't
#     competing for the slow LLM provider. Two distinct failure modes are handled:
#       - per-task TIMEOUT (rc 124, OUR wall): the dominant failure here — Gemma4
#         (:nitro) calls can take 2-7 min each and stall a task past its budget. We
#         ISOLATE it (no marker) and continue; a resume retries just that task.
#       - external KILL (rc 137/143, SIGTERM/SIGKILL — jetsam OOM or an outside signal):
#         the run STOPS and prints a resume command instead of thrashing.
#     Either way completed tasks are saved and you resume, never re-run from scratch.
#
# NOTE: empirically the kills on this host are slow-provider TIMEOUTS, not memory OOM
# (worker RSS stayed ~20-60MB). The real throughput lever is provider routing / the
# per-call timeout, not RAM — see PER_TASK_TIMEOUT_S and MAS_OPENROUTER_PROVIDER_ORDER.
#
# The resumable UNIT is one `main.py run` process for a single task (--task-offset i
# --task-limit 1) doing all --runs-per-task runs. (Runs can't be split into separate
# processes: they'd collide on run_0 with the same seed — there is no --run-offset.)
#
# Usage:
#   bash scripts/run_full_selfevo.sh                 # run / resume the full experiment
#   RESET=1 bash scripts/run_full_selfevo.sh         # wipe resume markers, start fresh
#   SAMPLES=10 bash scripts/run_full_selfevo.sh      # smaller run (10 tasks/benchmark)
#   PARALLEL_LIGHT=1 bash scripts/run_full_selfevo.sh# serialize light benchmarks too
# =============================================================================
set -uo pipefail   # NOT -e: we inspect every exit code ourselves (OOM vs timeout vs ok).

# ---------------------------------------------------------------------------
# 0. Repo root + secrets. Load .env (OPENROUTER_API_KEY etc.) without echoing it.
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${OPENROUTER_API_KEY:-}" && -f ".env" ]]; then set -a; . ./.env; set +a; fi
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (or put it in .env).}"

# ---------------------------------------------------------------------------
# 1. Recommended configuration (all overridable via env).
# ---------------------------------------------------------------------------
PY="${PY:-/Users/maoxunhuang/miniconda3/envs/agents/bin/python}"   # conda 'agents' env
[[ -x "$PY" ]] || PY="$ROOT/.venv/bin/python"                      # fallback to .venv
MODEL="${MODEL:-google/gemma-4-31b-it:nitro}"
EXPERIMENT_ID="${EXPERIMENT_ID:-full_selfevo__google_gemma_4_31b_it_nitro}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/full_experiment}"
SAMPLES="${SAMPLES:-30}"                 # tasks/benchmark (reference baseline = 30)
RUNS_PER_TASK="${RUNS_PER_TASK:-3}"      # runs/task (matches reference; enables pass@k)
PARALLEL_LIGHT="${PARALLEL_LIGHT:-3}"    # concurrent UNITS in the light phase (lower if RAM tight)
PER_TASK_TIMEOUT_S="${PER_TASK_TIMEOUT_S:-1800}"  # kill a hung task after 30 min (!= OOM)
HEAVY_PAUSE_S="${HEAVY_PAUSE_S:-0}"      # optional pause before the heavy phase (close apps / free RAM)
SERVER_PORT="${SERVER_PORT:-8080}"       # StableToolBench virtual server
CACHE_ROOT="${CACHE_ROOT:-$ROOT/benchmark/stabletoolbench/tool_response_cache}"

# Benchmark ordering: LIGHT ones first (run concurrently), then the memory-HEAVY one
# LAST and alone so it gets maximal RAM. Override the arrays to change scope.
LIGHT_BENCHMARKS=(${LIGHT_BENCHMARKS:-plancraft workbench stabletoolbench})
HEAVY_BENCHMARKS=(${HEAVY_BENCHMARKS:-browsecomp})

# ---------------------------------------------------------------------------
# 2. Sampling / retry / provider-routing env — identical to the seq runners so
#    self_evolved is compared on the same footing as the static baselines.
# ---------------------------------------------------------------------------
export OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-medium}"
export OPENROUTER_TEMPERATURE="${OPENROUTER_TEMPERATURE:-1.0}"
export OPENROUTER_TOP_P="${OPENROUTER_TOP_P:-1.0}"
export OPENROUTER_TOP_K="${OPENROUTER_TOP_K:-0}"
# Per-call timeout: 150s (not the 600s default). Gemma4 calls are normally 3-17s; a
# call still running at 150s means a transient slow provider, so we abort and let the
# retry loop RE-ROUTE to a faster upstream instead of blocking up to 600s (the cause of
# the earlier task-level timeouts). Only set here -> static baselines are unaffected.
export MAS_OPENROUTER_TIMEOUT_S="${MAS_OPENROUTER_TIMEOUT_S:-150}"
export MAS_LLM_RETRY_ATTEMPTS="${MAS_LLM_RETRY_ATTEMPTS:-5}"
export MAS_LLM_TIMEOUT_RETRY_ATTEMPTS="${MAS_LLM_TIMEOUT_RETRY_ATTEMPTS:-2}"
export MAS_LLM_RETRY_BACKOFF_S="${MAS_LLM_RETRY_BACKOFF_S:-8}"
export MAS_LLM_RETRY_MAX_BACKOFF_S="${MAS_LLM_RETRY_MAX_BACKOFF_S:-180}"
export MAS_OPENROUTER_PROVIDER_ORDER="${MAS_OPENROUTER_PROVIDER_ORDER:-DeepInfra,Chutes,Novita,Together}"
export MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER="${MAS_OPENROUTER_SHUFFLE_PROVIDER_ORDER:-1}"
export MAS_TRANSCRIPT_COMPACTION_ENABLED="${MAS_TRANSCRIPT_COMPACTION_ENABLED:-1}"
export MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER="${MAS_SEARCH_TOOL_FAILURE_CIRCUIT_BREAKER:-2}"
export MAS_TOOL_CONTEXT_RAW_TURNS="${MAS_TOOL_CONTEXT_RAW_TURNS:-2}"
export MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS="${MAS_TOOL_CONTEXT_SUMMARY_MAX_CHARS:-6000}"
export MAS_TOOL_CONTEXT_PREVIEW_CHARS="${MAS_TOOL_CONTEXT_PREVIEW_CHARS:-160}"
export MAS_TOOL_CONTEXT_DOCUMENT_CHARS="${MAS_TOOL_CONTEXT_DOCUMENT_CHARS:-12000}"

# ---------------------------------------------------------------------------
# 3. Bookkeeping paths. Markers live OUTSIDE the experiment root so they never
#    interfere with summarize-experiment, which walks <root>/<bench>/<system>/.
# ---------------------------------------------------------------------------
LOGDIR="artifacts/_smoke_logs"; mkdir -p "$LOGDIR"
MARKER_DIR="artifacts/_markers/$EXPERIMENT_ID"
ABORT_FLAG="$MARKER_DIR/.abort"                 # created (with reason) on an OOM kill
[[ "${RESET:-0}" == "1" ]] && { echo "RESET=1: clearing resume markers under $MARKER_DIR"; rm -rf "$MARKER_DIR"; }
mkdir -p "$MARKER_DIR"; rm -f "$ABORT_FLAG"     # a stale abort flag must not block a fresh start

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# 4. Portable per-process timeout (macOS has no coreutils `timeout`). Returns the
#    child's real exit code normally, but 124 when WE killed it for timing out — so
#    the caller can tell a timeout (124) apart from an external OOM kill (137/143).
# ---------------------------------------------------------------------------
run_with_timeout() {
  local secs="$1"; shift
  local to_flag; to_flag="$(mktemp -u)"          # a path that exists ONLY if timeout fires
  "$@" & local pid=$!
  ( sleep "$secs"; : >"$to_flag"; kill -TERM "$pid" 2>/dev/null; sleep 8; kill -KILL "$pid" 2>/dev/null ) &
  local watcher=$!
  wait "$pid" 2>/dev/null; local rc=$?
  kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
  if [[ -e "$to_flag" ]]; then rc=124; rm -f "$to_flag"; fi
  return "$rc"
}

# ---------------------------------------------------------------------------
# 5. run_unit: execute ONE task (all RUNS_PER_TASK runs) as an isolated process.
#    - Skips instantly if already marked done (resume) or if an abort was triggered.
#    - rc 0          -> drop a done-marker.
#    - rc 124        -> OUR per-task timeout (slow provider / hung call) -> no marker,
#                       continue; a later resume retries just this task. (Most common.)
#    - rc 137/143    -> external SIGKILL/SIGTERM (jetsam OOM or outside signal) -> raise
#                       the global abort flag and stop everything (resume to continue).
#    - other         -> ordinary failure -> no marker, will resume next time.
# ---------------------------------------------------------------------------
run_unit() {
  local bench="$1" off="$2"
  local marker="$MARKER_DIR/$bench/offset_${off}.done"
  [[ -f "$marker"     ]] && { log "SKIP    $bench offset=$off (already done)"; return 0; }
  [[ -f "$ABORT_FLAG" ]] && return 0
  local logf="$LOGDIR/${bench}_off${off}.log"
  run_with_timeout "$PER_TASK_TIMEOUT_S" \
    "$PY" main.py run \
      --config "config/benchmarks/${bench}_10.toml" \
      --benchmark "$bench" \
      --output-dir "$OUTPUT_DIR" \
      --output-layout hierarchical \
      --experiment-id "$EXPERIMENT_ID" \
      --system-label self_evolved \
      --topology self_evolved \
      --agents 5 --mas-rounds 2 --discussion-rounds 1 --communication-budget 2 \
      --default-model "$MODEL" --judge-model "$MODEL" \
      --task-limit 1 --task-offset "$off" --runs-per-task "$RUNS_PER_TASK" \
      >"$logf" 2>&1
  local rc=$?
  case "$rc" in
    0)        mkdir -p "$(dirname "$marker")"; : >"$marker"; log "OK      $bench offset=$off" ;;
    137|143)  log "OOM     $bench offset=$off (rc=$rc) -> STOPPING (resume to continue)"
              echo "$bench offset=$off rc=$rc" >>"$ABORT_FLAG" ;;
    124)      log "TIMEOUT $bench offset=$off (>${PER_TASK_TIMEOUT_S}s) -> will resume (log: $logf)" ;;
    *)        log "FAIL    $bench offset=$off (rc=$rc) -> will resume (log: $logf)" ;;
  esac
  return "$rc"
}

# Block until fewer than $1 background UNITS are running (portable async pool).
# Excludes the StableToolBench server's background job ($SERVER_PID) from the count —
# otherwise the long-lived server would permanently occupy one concurrency slot and
# silently halve the real parallelism.
running_units() { jobs -rp | grep -vxF "${SERVER_PID:-__none__}" | grep -c . ; }
wait_for_slot() { local cap="$1"; while (( $(running_units) >= cap )); do sleep 1; done; }

# ---------------------------------------------------------------------------
# 6. StableToolBench virtual server: start once if any selected benchmark needs it,
#    and always tear it down on exit.
# ---------------------------------------------------------------------------
SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT
if printf '%s\n' ${LIGHT_BENCHMARKS[@]+"${LIGHT_BENCHMARKS[@]}"} ${HEAVY_BENCHMARKS[@]+"${HEAVY_BENCHMARKS[@]}"} | grep -qx stabletoolbench; then
  log "starting StableToolBench virtual server on :$SERVER_PORT"
  "$PY" scripts/stabletoolbench_virtual_server.py --host localhost --port "$SERVER_PORT" \
    --path /virtual --cache-root "$CACHE_ROOT" >"$LOGDIR/stb_server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 30); do grep -qi "ready" "$LOGDIR/stb_server.log" 2>/dev/null && break; sleep 1; done
fi

log "experiment=$EXPERIMENT_ID model=$MODEL samples=$SAMPLES runs/task=$RUNS_PER_TASK"
log "light=[${LIGHT_BENCHMARKS[*]:-}] parallel=$PARALLEL_LIGHT  heavy(last)=[${HEAVY_BENCHMARKS[*]:-}] sequential"

# ---------------------------------------------------------------------------
# 7. PHASE 1 — LIGHT benchmarks, run CONCURRENTLY. All (benchmark, task) units share
#    one global pool of PARALLEL_LIGHT slots, so a unit waiting on the network lets
#    another make progress. Stops dispatching the moment an OOM raises the abort flag.
# ---------------------------------------------------------------------------
log "=== PHASE 1: light benchmarks (concurrent) ==="
for bench in ${LIGHT_BENCHMARKS[@]+"${LIGHT_BENCHMARKS[@]}"}; do
  for (( off=0; off<SAMPLES; off++ )); do
    [[ -f "$ABORT_FLAG" ]] && break 2          # an OOM somewhere -> stop launching
    wait_for_slot "$PARALLEL_LIGHT"            # respect the concurrency cap
    run_unit "$bench" "$off" &                 # dispatch async
  done
done
while (( $(running_units) > 0 )); do sleep 1; done   # drain task units only (NOT the STB server)

# ---------------------------------------------------------------------------
# 8. PHASE 2 — HEAVY benchmark(s), run LAST and SEQUENTIALLY (one task at a time) so
#    each gets the whole machine's RAM. Optional pause lets you free memory first.
#    On an OOM kill the abort flag is set and we stop immediately (no thrashing).
# ---------------------------------------------------------------------------
if [[ ! -f "$ABORT_FLAG" && ${#HEAVY_BENCHMARKS[@]} -gt 0 ]]; then
  log "=== PHASE 2: heavy benchmarks (sequential, last) ==="
  if (( HEAVY_PAUSE_S > 0 )); then
    log "pausing ${HEAVY_PAUSE_S}s before heavy phase — close other apps to free RAM"
    sleep "$HEAVY_PAUSE_S"
  fi
  for bench in ${HEAVY_BENCHMARKS[@]+"${HEAVY_BENCHMARKS[@]}"}; do
    for (( off=0; off<SAMPLES; off++ )); do
      [[ -f "$ABORT_FLAG" ]] && break 2        # OOM on a prior task -> stop + notify
      run_unit "$bench" "$off"                 # foreground = one heavy task at a time
    done
  done
fi

# ---------------------------------------------------------------------------
# 9. Outcome. On OOM-abort: tell the user exactly how to RESUME (markers make the
#    re-run skip everything already finished). Otherwise summarize for the comparison.
# ---------------------------------------------------------------------------
if [[ -f "$ABORT_FLAG" ]]; then
  echo
  log "!!! STOPPED: a task was externally killed (SIGKILL/SIGTERM — jetsam OOM or outside signal). Completed tasks are saved; nothing is lost."
  log "Offending unit(s):"; sed 's/^/    /' "$ABORT_FLAG"
  echo
  log "To RESUME (skips all finished tasks, continues where it stopped):"
  echo "    EXPERIMENT_ID=$EXPERIMENT_ID bash scripts/run_full_selfevo.sh"
  log "Tip: free RAM first (close other apps), or lower PARALLEL_LIGHT, or set HEAVY_PAUSE_S=30."
  exit 1
fi

log "=== all units complete — summarizing experiment ==="
"$PY" main.py summarize-experiment --experiment-root "$OUTPUT_DIR/$EXPERIMENT_ID" \
  >"$LOGDIR/full_selfevo_summarize.log" 2>&1
log "done. summary -> $OUTPUT_DIR/$EXPERIMENT_ID  (log: $LOGDIR/full_selfevo_summarize.log)"

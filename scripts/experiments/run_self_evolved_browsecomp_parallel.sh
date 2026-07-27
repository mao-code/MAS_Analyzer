#!/usr/bin/env bash
# Run the self-evolved (dynamic topology) MAS on BrowseComp in parallel shards.
# Each shard is an independent `main.py run` over a disjoint task slice (via the
# new --task-offset flag), writing to its own experiment root so per-system files
# never race. Afterwards the shard outputs are merged into one self_evolved/ dir
# and the per-task summary.csv rows are concatenated.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Load OpenRouter key from .env if not already exported.
if [[ -z "${OPENROUTER_API_KEY:-}" && -f ".env" ]]; then
  set -a; . ./.env; set +a
fi
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY (or put it in .env).}"

CONFIG_PATH="${CONFIG_PATH:-config/selfevo_browsecomp_gptoss.toml}"
SAMPLES="${SAMPLES:-10}"
SHARD_SIZE="${SHARD_SIZE:-2}"          # tasks per shard -> SAMPLES/SHARD_SIZE shards
RUNS_PER_TASK="${RUNS_PER_TASK:-1}"
OUT_BASE="${OUT_BASE:-artifacts/selfevo_browsecomp_gptoss/run_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ID="${EXPERIMENT_ID:-se}"
SYSTEM_LABEL="${SYSTEM_LABEL:-self_evolved}"

# medium reasoning effort for GPT-OSS (OpenRouter reasoning.effort).
export OPENROUTER_REASONING_EFFORT="${OPENROUTER_REASONING_EFFORT:-medium}"

mkdir -p "$OUT_BASE/logs"
echo "Config:        $CONFIG_PATH"
echo "Out base:      $OUT_BASE"
echo "Samples:       $SAMPLES (shard size $SHARD_SIZE)"
echo "Runs/task:     $RUNS_PER_TASK"
echo "Reasoning:     $OPENROUTER_REASONING_EFFORT"
echo

pids=()
labels=()
offset=0
shard=0
while (( offset < SAMPLES )); do
  limit=$SHARD_SIZE
  (( offset + limit > SAMPLES )) && limit=$(( SAMPLES - offset ))
  shard_out="$OUT_BASE/shard_${shard}"
  log="$OUT_BASE/logs/shard_${shard}.log"
  echo "=== shard $shard: offset=$offset limit=$limit -> $shard_out ==="
  uv run python main.py run \
    --config "$CONFIG_PATH" \
    --benchmark browsecomp \
    --task-limit "$limit" \
    --task-offset "$offset" \
    --runs-per-task "$RUNS_PER_TASK" \
    --output-dir "$shard_out" \
    --output-layout hierarchical \
    --experiment-id "$EXPERIMENT_ID" \
    --system-label "$SYSTEM_LABEL" \
    >"$log" 2>&1 &
  pids+=($!)
  labels+=("shard_${shard}")
  offset=$(( offset + limit ))
  shard=$(( shard + 1 ))
done

echo
echo "Launched ${#pids[@]} shards; waiting..."
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "OK   ${labels[$i]}"
  else
    echo "FAIL ${labels[$i]} (see $OUT_BASE/logs/${labels[$i]}.log)"
    fail=1
  fi
done

# ---- Merge shard outputs into one self_evolved system dir ----
MERGED="$OUT_BASE/merged/browsecomp/$SYSTEM_LABEL"
mkdir -p "$MERGED"
header_written=0
: > "$MERGED/summary.csv"
for d in "$OUT_BASE"/shard_*/"$EXPERIMENT_ID"/browsecomp/"$SYSTEM_LABEL"; do
  [[ -d "$d" ]] || continue
  # copy per-task dirs (named by task id)
  for taskdir in "$d"/*/; do
    [[ -f "${taskdir}task_summary.json" ]] || continue
    cp -R "$taskdir" "$MERGED/"
  done
  # concatenate per-task summary.csv rows (header once)
  if [[ -f "$d/summary.csv" ]]; then
    if [[ "$header_written" -eq 0 ]]; then
      cat "$d/summary.csv" >> "$MERGED/summary.csv"
      header_written=1
    else
      tail -n +2 "$d/summary.csv" >> "$MERGED/summary.csv"
    fi
  fi
  # keep one copy of settings/summary json for downstream summarize-experiment
  [[ -f "$d/experiment_settings.json" && ! -f "$MERGED/experiment_settings.json" ]] && \
    cp "$d/experiment_settings.json" "$MERGED/experiment_settings.json"
done

echo
echo "Merged self_evolved dir: $MERGED"
echo "Per-task rows: $(($(wc -l < "$MERGED/summary.csv") - 1))"
echo "Done (fail=$fail)."
exit $fail

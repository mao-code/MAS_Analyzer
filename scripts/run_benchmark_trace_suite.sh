#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/artifacts/benchmark_traces/logs"
mkdir -p "$LOG_DIR"

benchmarks=(
  "finance_agent"
  "browsecomp"
  "plancraft"
  "scicode"
  "workbench"
  "stabletoolbench"
  "agentbench"
)

for benchmark in "${benchmarks[@]}"; do
  config_path="$ROOT_DIR/config/benchmarks/${benchmark}_10.toml"
  log_path="$LOG_DIR/${benchmark}.log"
  echo "=== Running ${benchmark} with ${config_path} ==="
  if uv run python "$ROOT_DIR/main.py" run --config "$config_path" --benchmark "$benchmark" >"$log_path" 2>&1; then
    echo "OK: ${benchmark}"
  else
    status=$?
    echo "FAIL(${status}): ${benchmark} -- see ${log_path}"
  fi
done

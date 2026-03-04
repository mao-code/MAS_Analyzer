#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  conda run -n agents python -m MAS.experiment_cli "$@"
else
  python -m MAS.experiment_cli "$@"
fi

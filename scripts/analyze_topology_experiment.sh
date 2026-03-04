#!/usr/bin/env bash
set -euo pipefail

ENTRYPOINT='from descriptor.topology_analysis import main; raise SystemExit(main())'

if command -v conda >/dev/null 2>&1; then
  conda run -n agents python -c "${ENTRYPOINT}" "$@"
else
  python -c "${ENTRYPOINT}" "$@"
fi

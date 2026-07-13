#!/usr/bin/env bash
# Thin wrapper: print LF source-profile timing/memory metrics, one table per model.
# Usage: show_metrics.sh [PROFILING_DIR]   (default: profiling_both)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ASYM_PY:-${DIR}/../../.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3
exec "${PY}" "${DIR}/show_metrics.py" "$@"

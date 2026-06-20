#!/usr/bin/env bash
# Thin wrapper: print LF profiling run status, one table per model.
# Usage: show_status.sh [PROFILING_DIR]   (default: profiling_both)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ASYM_PY:-${DIR}/../../.venv/bin/python}"
[[ -x "${PY}" ]] || PY=python3
exec "${PY}" "${DIR}/show_status.py" "$@"

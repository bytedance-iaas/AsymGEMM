#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_DIR="$ROOT_DIR/tests"
cd "$ROOT_DIR"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export PYTHONPATH="$ROOT_DIR:$TEST_DIR${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob
for test_file in tests/test*.py; do
  python "$test_file"
done
shopt -u nullglob

#!/bin/bash
# Build wheel and install into the current environment.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$SCRIPT_DIR"

rm -rf build dist *.egg-info

echo "Building wheel..."
uv build --wheel

echo "Installing..."
uv pip install dist/*.whl --reinstall

echo "Done!"

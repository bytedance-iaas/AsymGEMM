#!/bin/bash
# Build wheel and install into the current environment.
set -euo pipefail

# =============================================================================
# User Parameters
# =============================================================================
UV_BIN=${UV_BIN:-uv}

# =============================================================================
# Derived Parameters
# =============================================================================
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# =============================================================================
# Main Logic
# =============================================================================
cd "$ROOT_DIR"

rm -rf build dist *.egg-info

echo "Building wheel..."
"${UV_BIN}" build --wheel

echo "Installing..."
"${UV_BIN}" pip install dist/*.whl --reinstall

echo "Done!"

#!/usr/bin/env bash
# Re-sync third-party/cpu_gemm from an upstream working tree (maintainer use).
#
# Usage: scripts/sync_cpu_gemm.sh [PATH_TO_UPSTREAM_CPU_GEMM]
# Default upstream path: /workspace/cpu_gemm
set -euo pipefail

UPSTREAM=${1:-/workspace/cpu_gemm}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/third-party/cpu_gemm"

if [[ ! -d "$UPSTREAM" ]]; then
  echo "[FATAL] Upstream cpu_gemm tree not found at: $UPSTREAM" >&2
  exit 1
fi

echo "Syncing cpu_gemm:  $UPSTREAM  →  $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"
# `cp -r` over rsync because rsync isn't always present in CI containers.
( cd "$UPSTREAM" && \
  find . -mindepth 1 -maxdepth 1 \
    -not -name build -not -name .git \
    -exec cp -r {} "$DEST"/ \; )

# Record provenance — best-effort SHA capture.
SHA=$(git -C "$UPSTREAM" rev-parse HEAD 2>/dev/null || echo "unknown")
DATE=$(date -u +%Y-%m-%d)
cat > "$DEST/UPSTREAM.md" <<EOF
# cpu_gemm — vendored snapshot

| Field      | Value                                                             |
|------------|--------------------------------------------------------------------|
| Source     | $UPSTREAM                                                          |
| Snapshot   | $DATE                                                              |
| SHA        | $SHA                                                               |
| License    | See \`LICENSE\` in this directory.                                 |
| Re-sync    | \`bash ../../scripts/sync_cpu_gemm.sh [PATH_TO_UPSTREAM]\`         |

Excluded from the snapshot: \`build/\`, \`.git/\`.
EOF

echo "Synced. Next: run scripts/test_cpu_gemm.sh to confirm the snapshot builds."

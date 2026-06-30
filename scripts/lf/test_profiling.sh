#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep this as a short smoke-test preset while sharing the real implementation
# with profile_lora_lf_test_source.sh.
export GPU_POOL="${GPU_POOL:-3}"
export BACKEND_SPECS="${BACKEND_SPECS:-zero3|norecomp}"
export WORKLOADS="${WORKLOADS:-1024|1|1}"
export MAX_STEPS="${MAX_STEPS:-10000000}"
export RUN_POST=false

exec "${SCRIPT_DIR}/profile_lora_lf_test_source.sh" "$@"

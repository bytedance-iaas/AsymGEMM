#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Keep this as a short smoke-test preset while sharing the real implementation
# with profile_lora_lf.sh.
export GPU_POOL="${GPU_POOL:-3}"
export BACKEND_SPECS="${BACKEND_SPECS:-zero3|norecomp}"
export SEQ_LENS="${SEQ_LENS:-1024}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export MAX_STEPS="${MAX_STEPS:-10000000}"
export RUN_POST=false

exec "${SCRIPT_DIR}/profile_lora_lf.sh" "$@"

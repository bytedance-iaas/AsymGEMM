#!/bin/bash
# Post-merge long-seq validation: q3.5-35b-a3b @ 120000x8, flagship asym row —
# the workload in flight pre-merge (runs.log 07-15) that post-merge testing
# stopped short of. This is where V1/V2 (host-read ordering) and pin fallbacks
# are most likely to bite: afterwards, read the three
# pin_fallback_calls_module_global counters plus the loss band.
# Pre-merge reference (SFT tree, memory_mode.md 120k dial): loss parity, no OOM.
set -uo pipefail
export RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 120000|8|1 ; none|false|true|false|false|false'
export GPU_POOL=0
export PROFILERS=source
export PLOT=false
export OVERWRITE=true
export OUTPUT_ROOT=/workspace/qwen35_local/profiling_postmerge_120k
export ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0
export ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1
export ASYM_LINEAR_ATTENTION_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1
bash scripts/lf/profile_lora_lf_test_both.sh

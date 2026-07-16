#!/bin/bash
# Validation of the wait_cpu_ready_host fix (30 host-reader sites): 45k flagship row.
# Reference (post-merge, pre-fix, 2026-07-15): 291.1 s/step, 31.4 alloc / 34.2 reserved,
# loss 0.89-0.95. The host syncs are true data dependencies; this run checks they do
# not cost measurable latency (host run-ahead loss) and change nothing else.
set -uo pipefail
export RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 45000|8|1 ; none|false|true|false|false|false'
export GPU_POOL=0
export PROFILERS=source
export PLOT=false
export OVERWRITE=true
export OUTPUT_ROOT=/workspace/qwen35_local/profiling_postmerge_45k_hostsync
export ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0
export ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1
export ASYM_LINEAR_ATTENTION_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1
bash scripts/lf/profile_lora_lf_test_both.sh

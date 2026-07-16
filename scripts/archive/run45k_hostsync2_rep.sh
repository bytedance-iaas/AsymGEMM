#!/bin/bash
# T9b — reproducibility confirmation of T9 (identical config, fresh output root).
# Purpose: T9's flagship losses shifted ~0.05 vs T2/T8. By elimination that is F14
# (the attention LoRA-A cpu_left forward host wait, 160 calls) deterministically
# correcting a firing host-read race. This run checks the shift is STABLE
# (reproduces) and not flaky corruption. Same working tree as T9.
set -uo pipefail
export RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 45000|8|1 ; none|false|true|false|false|false'
export GPU_POOL=0
export PROFILERS=source
export PLOT=false
export OVERWRITE=true
export OUTPUT_ROOT=/workspace/qwen35_local/profiling_postmerge_45k_hostsync2_rep
export ASYM_CPU_ADAMW_ASYNC_GRAD_OFFLOAD=0
export ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1
export ASYM_LINEAR_ATTENTION_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1
bash scripts/lf/profile_lora_lf_test_both.sh

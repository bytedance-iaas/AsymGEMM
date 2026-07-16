#!/bin/bash
# Post-merge cross-model gate, lean form: qwen3-30b s20000 control on the canonical
# stack (recorded band: loss 1.775 +/- 0.05, grad_norm ~0.49). Bare defaults on
# purpose — the band was recorded without the qwen3.5 tuned env; ker101 is the 30b
# auto-default. This is the only untested combination left after reasoning:
# RELEASE_FUSED_HOME (default ON) firing on a non-qwen3.5 MoE model.
set -uo pipefail
export RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 20000|8|1 ; none|false|false|false|false|false'
export GPU_POOL=0
export PROFILERS=source
export PLOT=false
export OVERWRITE=true
export OUTPUT_ROOT=/workspace/qwen35_local/profiling_postmerge_q30b
bash scripts/lf/profile_lora_lf_test_both.sh

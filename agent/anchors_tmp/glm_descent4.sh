#!/bin/bash
# Descent round 4: instrumented Flash repro in PLAIN mode (copy-free, dynamic
# compile) with the host sampler running — get DATA on where host memory goes.
# Gated on DESCENT3-DONE (let the Air cells of round 3 finish first).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
export TORCHINDUCTOR_COMPILE_THREADS=1
for i in $(seq 1 1440); do grep -q "DESCENT3-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "DESCENT3-DONE" "$S" || { echo "DESCENT4-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "DESCENT4 begin $(date +%H:%M)" >> "$S"

bash /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/host_sampler.sh "$LOGD/host_sampler_q24p.log" &
SAMPLER=$!
export ASYMM_ATTN_QCHUNK_ROWS=24000 ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_ROWS=24000
export ASYMM_ATTN_QCHUNK_MODE=plain ASYM_GEMM_LF_CONFIG_ASYMM_ATTN_QCHUNK_MODE=plain
run_cell l47q24p glm4.7-flash "$T3TOK" 192000 "5"
kill "$SAMPLER" 2>/dev/null || true
echo "DESCENT4-DONE $(date +%H:%M)" >> "$S"

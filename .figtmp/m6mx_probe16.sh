#!/bin/bash
# Round 15 (INSIDE container): the bank-clone unlock at 416K.
# ASYM_EXACT_PINNED_FORCE_CLONE=1 (new, host_weight.py): the 270-GB expert
# bank clones out of its loader-owned mmap alias and exact-registers,
# reclaiming the ~74-110 GiB standing pow2 tax on BOTH rungs. Walls shift
# right ~80-115K; T3's GPU ceiling is ~434K, so the corridor lands ~400-432K.
# T3 recipe: fix2+slab infra + ohbm56 (1 root; GPU ~97% ceiling at 416K)
# + skip-bwd + cpu-act cap + block-experts + reuse + threads64. NO dual-DA.
set -uo pipefail
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
guard() {
  sleep 30
  local apps
  for _ in $(seq 1 21); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -z "$apps" ] && return 0
    echo "!!! GUARD: GPU busy before $1 (pids: $apps) — waiting"; sleep 30
  done
  echo "!!! GUARD FAIL before $1 — aborting chain"; exit 9
}
XPC="ASYM_EXACT_PINNED=1 ASYM_EXACT_PINNED_ROOTS=1 ASYM_EXACT_PINNED_SAVED=1 ASYM_EXACT_PINNED_FORCE_CLONE=1"
set +e
guard T2B416
echo "=== [1] T2B(fix2+slab) @416K capacity attempt (w1+m1) $(date -u +%H:%M:%S)"
env $XPC WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxg2t2b416 "asym_cpuadamwds|T2B|ligerloss1" 416000 1
T2B416=$?
echo "R16_T2B416_EXIT=$T2B416"
guard T3416
echo "=== [2] T3(fix2+slab,ohbm56,blockexp,cap,skip) @416K (w1+m1) $(date -u +%H:%M:%S)"
env $XPC \
ASYM_ATTN_SAVED_TENSOR_OFFLOAD_SKIP_IN_BACKWARD=1 \
ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS=500000 \
ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 \
ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=1 \
ASYMM_QWEN3_MOE_FG_DA_GPU=1 \
ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 \
ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1 \
ASYM_CPU_OPS_THREADS=64 \
ASYM_PLACEMENT_POLICY=1 \
ASYMM_MOE_FG_DIRECT_REUSE=1 \
WARMUP_STEPS=1 MAX_STEPS=1 MAX_SAMPLES=512 \
  bash scripts/lf/tp_probe.sh mixtral-8x22b m6mxg2t3416 "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm56|ligerloss1" 416000 1
T3416=$?
echo "R16_T3416_EXIT=$T3416"
if [ $T3416 -eq 0 ] && [ $T2B416 -ne 0 ]; then echo "PROBE16_WINNER=416000"; exit 0; fi
echo "PROBE16_STATE=t2b=$T2B416,t3=$T3416"
exit 3

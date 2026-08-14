#!/bin/bash
# v1d_air_fix.sh — post-v1c follow-ups from the air 64k trio finding:
# idle DEVICE rings cost -2..-3% tok/s + full ring size in resv even at
# 60%-HBM cells (spl 3871/110.4G vs sp2 3990/97.9G vs sdp 3951/97.9G),
# while sp2's equally-large HOST slots cost nothing. T1 recipes are
# structurally hook-dead (expert GEMMs never reach _dispatch_grouped_nt's
# hook), so hook-dead cells run SLOT_ROWS=8192 ALWAYS.
#  1. a_spl64t  — air 64k dev retry with tiny rings (the V1 dev-cell row)
#  2. a_splt2   — air T2-class armed-regime probe (arena 400, big rings,
#     NO_SNAP + high MPE): the REAL wide-bank win test that a_bo16 (T1,
#     hook-dead) cannot deliver. Fresh sp2/sdp T2-class twins for the pair.
# NOTE ordering: a_splt2 trio runs FIRST if v1c's bo16 pair already showed
# what we expect (zero gates); cells are independent either way.
set -uo pipefail
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/v1d_status.log"

guard() {
  rm -f /dev/shm/asym_fabric_* /dev/shm/asym_seprobe_* 2>/dev/null || true
  for i in $(seq 1 60); do
    live=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && live="$live $p"
    done
    [ -z "$live" ] && return 0
    [ "$i" -ge 3 ] && for p in $live; do kill -9 "$p" 2>/dev/null; done
    sleep 10
  done
  echo "GUARD-TIMEOUT $(date +%H:%M)" >> "$S"; return 1
}

cell() { # cell TAG MODEL BACKEND TIER SEQ B [extra_env...]
  local tag="$1" model="$2" be="$3" tier="$4" seq="$5" b="$6"; shift 6
  guard || return 1
  echo "CELL-START $tag $(date +%m-%d_%H:%M)" >> "$S"
  ( for kv in "$@"; do export "$kv"; done
    RUN_NAME="$tag" RUNS="${model}|2 ; ${be}|${tier}|ligerloss1 ; ${seq}|${b}|1 ; none|false|false|false|false|false" \
      bash scripts/lf/profile_lora_lf_test_source.sh ) > "$L/vc_${tag}.log" 2>&1
  echo "CELL-END $tag rc=$? $(date +%m-%d_%H:%M)" >> "$S"
}

SPL=asym_sepplanlink2_cpuadamwds; SP2=asym_sepplan2_cpuadamwds; SDP=asym_sdp2_cpuadamwds
echo "V1D begin $(date +%m-%d_%H:%M)" >> "$S"
# ---- 1. air 64k dev retry, tiny rings (T1 hook-dead -> rings idle) ----
cell a_spl64t glm4.5-air "$SPL" T1 64000 2 ASYM_ARENA_SHM_CAP_GB=240 ASYM_EP_SEP_SLOT_KMAX=4096 ASYM_EP_SEP_SLOT_ROWS=8192
# ---- 2. air T2-class armed-regime probe trio (arena 400) ----
cell a_splt2 glm4.5-air "$SPL" T2 64000 2 ASYM_ARENA_SHM_CAP_GB=400 ASYM_EP_SEP_SLOT_KMAX=4096 ASYM_EP_SEP_SLOT_ROWS=262144 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
cell a_sp2t2 glm4.5-air "$SP2" T2 64000 2 ASYM_ARENA_SHM_CAP_GB=400
cell a_sdpt2 glm4.5-air "$SDP" T2 64000 2 ASYM_ARENA_SHM_CAP_GB=400
guard || true
echo "V1D-DONE $(date +%m-%d_%H:%M)" >> "$S"

#!/bin/bash
# v1_guarded.sh — V1 stage cells with a REAL guard (fix_dynamic_ep.md ops:
# V-chains previously lacked the tpfig GPU-idle guard; a wedged cell held
# both GPUs at 186 GiB and poisoned every later cell, incl. plain sdp2).
set -uo pipefail
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/v1_status.log"

guard() {
  rm -f /dev/shm/asym_fabric_* 2>/dev/null || true
  for i in $(seq 1 60); do
    live=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && live="$live $p"
    done
    [ -z "$live" ] && return 0
    # kill zombified holders after 3 polls (they are never legitimate here:
    # this chain owns the node serially)
    [ "$i" -ge 3 ] && for p in $live; do kill -9 "$p" 2>/dev/null; done
    sleep 10
  done
  echo "GUARD-TIMEOUT $(date +%H:%M)" >> "$S"; return 1
}

cell() { # cell TAG MODEL BACKEND TIER SEQ B [extra_env...]
  local tag="$1" model="$2" be="$3" tier="$4" seq="$5" b="$6"; shift 6
  guard || return 1
  echo "CELL-START $tag $(date +%H:%M)" >> "$S"
  ( for kv in "$@"; do export "$kv"; done
    RUN_NAME="$tag" RUNS="${model}|2 ; ${be}|${tier}|ligerloss1 ; ${seq}|${b}|1 ; none|false|false|false|false|false" \
      bash scripts/lf/profile_lora_lf_test_source.sh ) > "$L/vc_${tag}.log" 2>&1
  echo "CELL-END $tag rc=$? $(date +%H:%M)" >> "$S"
}

SPL=asym_sepplanlink2_cpuadamwds; SP2=asym_sepplan2_cpuadamwds; SDP=asym_sdp2_cpuadamwds
echo "V1G begin $(date +%H:%M)" >> "$S"
# ---- Flash follow-ups ----
cell f_spl192r glm4.7-flash "$SPL" T1 192000 4
cell f_sdp192r glm4.7-flash "$SDP" T1 192000 4
cell f_spl960  glm4.7-flash "$SPL" T2 960000 1
cell f_sdp960  glm4.7-flash "$SDP" T2 960000 1
cell f_sp2960  glm4.7-flash "$SP2" T2 960000 1
# ---- Air (full retry) ----
export ASYM_ARENA_SHM_CAP_GB=240 ASYM_EP_SEP_SLOT_KMAX=4096
cell a_spl64  glm4.5-air "$SPL" T1 64000 2
cell a_sp264  glm4.5-air "$SP2" T1 64000 2
cell a_sdp64  glm4.5-air "$SDP" T1 64000 2
cell a_bo16   glm4.5-air "$SPL" T1 16000 1 ASYM_EP_SEP_SLOT_ROWS=262144 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
cell a_sdp16  glm4.5-air "$SDP" T1 16000 1
cell a_spl320 glm4.5-air "$SPL" T1 320000 1
cell a_sp2320 glm4.5-air "$SP2" T1 320000 1
cell a_sdp320 glm4.5-air "$SDP" T1 320000 1
guard || true
echo "V1G-DONE $(date +%H:%M)" >> "$S"

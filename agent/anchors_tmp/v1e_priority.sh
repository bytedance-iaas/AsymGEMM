#!/bin/bash
# v1e_priority.sh — Kevin (08-11): "stop flash, only do the important ones
# first". Dropped: f_sp2960 twin (killed mid-run; campaign sp2=317 suffices),
# f_spl1024 crown, a_bo16/a_sdp16 (T1 hook-dead -> probe moot).
# Order: win-case probes first (T2-class cells where the hook is alive),
# then the §5-required air cells, then mixtral dev entry.
# Ring sizing: air k<=4096 (KMAX 4096); air T2 64k b1 fg rows ~256k+pad ->
# ROWS 327680 (20 GiB/rank rings, fits dev-cell headroom). Mixtral gate/up
# k=6144 -> KMAX 6144; probe ROWS 131072 (12 GiB/rank; down-proj k=16384
# capacity-declines by design — probe answers gate/up arming only).
# T1 cells are structurally hook-dead -> spl runs ROWS=8192 there, always.
set -uo pipefail
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/v1e_status.log"

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
echo "V1E begin $(date +%m-%d_%H:%M)" >> "$S"
# ---- 1. AIR T2-class armed-probe trio (THE win case; hook alive on T2) ----
cell a_splt2 glm4.5-air "$SPL" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=400 ASYM_EP_SEP_SLOT_KMAX=4096 ASYM_EP_SEP_SLOT_ROWS=327680 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
cell a_sp2t2 glm4.5-air "$SP2" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=400
cell a_sdpt2 glm4.5-air "$SDP" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=400
# ---- 2. air 64k T1 dev retry with tiny rings (V1 dev-cell row fix) ----
cell a_spl64t glm4.5-air "$SPL" T1 64000 2 ASYM_ARENA_SHM_CAP_GB=240 ASYM_EP_SEP_SLOT_KMAX=4096 ASYM_EP_SEP_SLOT_ROWS=8192
# ---- 3. air 320k ceiling trio (§5 air ceiling; spl tiny rings) ----
cell a_spl320 glm4.5-air "$SPL" T1 320000 1 ASYM_ARENA_SHM_CAP_GB=240 ASYM_EP_SEP_SLOT_KMAX=4096 ASYM_EP_SEP_SLOT_ROWS=8192
cell a_sp2320 glm4.5-air "$SP2" T1 320000 1 ASYM_ARENA_SHM_CAP_GB=240 ASYM_EP_SEP_SLOT_KMAX=4096
cell a_sdp320 glm4.5-air "$SDP" T1 320000 1 ASYM_ARENA_SHM_CAP_GB=240
# ---- 4. mixtral dev entry: T1 floor trio + T2 armed-probe trio ----
cell m_spl64 mixtral-8x22b "$SPL" T1 64000 1 ASYM_ARENA_SHM_CAP_GB=285 ASYM_EP_SEP_SLOT_KMAX=6144 ASYM_EP_SEP_SLOT_ROWS=8192
cell m_sp264 mixtral-8x22b "$SP2" T1 64000 1 ASYM_ARENA_SHM_CAP_GB=285
cell m_sdp64 mixtral-8x22b "$SDP" T1 64000 1 ASYM_ARENA_SHM_CAP_GB=285
cell m_splt2 mixtral-8x22b "$SPL" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=285 ASYM_EP_SEP_SLOT_KMAX=6144 ASYM_EP_SEP_SLOT_ROWS=131072 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
cell m_sp2t2 mixtral-8x22b "$SP2" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=285
cell m_sdpt2 mixtral-8x22b "$SDP" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=285
guard || true
echo "V1E-DONE $(date +%m-%d_%H:%M)" >> "$S"

#!/bin/bash
# v1c_resume.sh — continuation of v1_guarded.sh after the 04:0x session kill
# (f_sdp960 died mid-cell; f_sp2960 + all air cells never ran).
# Changes vs v1_guarded.sh:
#  (1) ceiling spl cells get ASYM_EP_SEP_SLOT_ROWS=8192 (tiny rings): the
#      default 163840x2048 rings = ~5.1 GiB/rank sank f_spl192r to -6.3% at
#      the 97%-HBM cell (backward-phase allocator pressure; fwd identical).
#      D1 policy: pure-floor cells run tiny rings to hold the +3% criterion.
#  (2) skip-reason counters (ep_sep.count_skip) now expose silent
#      _direct_grouped_bf16_reason bails — 192k T1 had armed=0 declined=0
#      with 4212 launches through the hook site.
#  (3) air env applied PER-CELL (the old chain-level export would have
#      leaked ARENA_CAP=240 + KMAX=4096 into later flash cells).
#  (4) appended f_spl1024 crown (twins exist: campaign sp2=297, banked sdp2=294).
set -uo pipefail
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/v1c_status.log"

guard() {
  rm -f /dev/shm/asym_fabric_* /dev/shm/asym_seprobe_* 2>/dev/null || true
  for i in $(seq 1 60); do
    live=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && live="$live $p"
    done
    [ -z "$live" ] && return 0
    # kill zombified holders after 3 polls (never legitimate: solo+serial node)
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
AIRENV="ASYM_ARENA_SHM_CAP_GB=240 ASYM_EP_SEP_SLOT_KMAX=4096"
echo "V1C begin $(date +%m-%d_%H:%M)" >> "$S"
# ---- 1. flash 192k T1 retry with tiny rings (floor-fix check, ~1.1h) ----
cell f_spl192t glm4.7-flash "$SPL" T1 192000 4 ASYM_EP_SEP_SLOT_ROWS=8192
# ---- 2. air dev trio (fast; reveals air T1 skip reasons early) ----
cell a_spl64 glm4.5-air "$SPL" T1 64000 2 $AIRENV
cell a_sp264 glm4.5-air "$SP2" T1 64000 2 $AIRENV
cell a_sdp64 glm4.5-air "$SDP" T1 64000 2 ASYM_ARENA_SHM_CAP_GB=240
# ---- 3. flash 960k controls (long, ~5.4h each) ----
cell f_sdp960r glm4.7-flash "$SDP" T2 960000 1
cell f_sp2960 glm4.7-flash "$SP2" T2 960000 1
# ---- 4. air bank-once probe pair (wide-bank win hypothesis) ----
cell a_bo16 glm4.5-air "$SPL" T1 16000 1 $AIRENV ASYM_EP_SEP_SLOT_ROWS=262144 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
cell a_sdp16 glm4.5-air "$SDP" T1 16000 1 ASYM_ARENA_SHM_CAP_GB=240
# ---- 5. air ceiling trio (98% HBM -> spl runs tiny rings) ----
cell a_spl320 glm4.5-air "$SPL" T1 320000 1 $AIRENV ASYM_EP_SEP_SLOT_ROWS=8192
cell a_sp2320 glm4.5-air "$SP2" T1 320000 1 $AIRENV
cell a_sdp320 glm4.5-air "$SDP" T1 320000 1 ASYM_ARENA_SHM_CAP_GB=240
# ---- 6. flash crown (1.024M, ~6.5h; twins banked) ----
cell f_spl1024 glm4.7-flash "$SPL" T2 1024000 1 ASYM_EP_SEP_SLOT_ROWS=8192
guard || true
echo "V1C-DONE $(date +%m-%d_%H:%M)" >> "$S"

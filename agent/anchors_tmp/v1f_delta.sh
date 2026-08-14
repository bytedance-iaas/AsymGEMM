#!/bin/bash
# v1f_delta.sh — remaining v1e work after the 08-11 session handoff:
# v1e wedged 4min into its first cell (a_splt2 SIGINT'd); the 320k twins +
# 16k pair were re-banked by the v1_guarded rerun; a_spl64t is in flight as
# an orphan cell. This chain = v1e minus everything already banked/in-flight:
#   1. air T2-class trio (THE win probe; hook alive on T2)
#   2. a_spl320t = air 320k ceiling SPL retry with tiny rings (never yet run;
#      big-ring attempts OOM'd twice on the model's own 19.5GiB whole-layer
#      tensor with only ~19.2GiB free)
#   3. mixtral dev entry: T1 floor trio + T2 armed-probe trio
# All cells run the gate-diagnosis build (frozen_linear counts gate_* skips).
set -uo pipefail
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/v1f_status.log"

guard() {
  rm -f /dev/shm/asym_fabric_* /dev/shm/asym_seprobe_* 2>/dev/null || true
  for i in $(seq 1 360); do
    live=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && live="$live $p"
    done
    # model load runs minutes with no CUDA context — GPU-free is NOT idle.
    # Also wait out any external launcher (orphan cells from other chains).
    ext=$(pgrep -f 'run_lf_profiled_train.py|build_lf_sft_eval_pair.py' 2>/dev/null | wc -l)
    [ -z "$live" ] && [ "${ext:-0}" -eq 0 ] && return 0
    sleep 20
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
  rm -f /dev/shm/asym_fabric_* /dev/shm/asym_seprobe_* 2>/dev/null || true
}

SPL=asym_sepplanlink2_cpuadamwds; SP2=asym_sepplan2_cpuadamwds; SDP=asym_sdp2_cpuadamwds
echo "V1F begin $(date +%m-%d_%H:%M)" >> "$S"
# NOTE: first guard also waits out the in-flight orphan a_spl64t cell.
# ---- 1. AIR T2-class armed-probe trio (win case; arena 400) ----
cell a_splt2 glm4.5-air "$SPL" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=400 ASYM_EP_SEP_SLOT_KMAX=4096 ASYM_EP_SEP_SLOT_ROWS=327680 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
cell a_sp2t2 glm4.5-air "$SP2" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=400
cell a_sdpt2 glm4.5-air "$SDP" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=400
# ---- 2. air 320k ceiling SPL retry, tiny rings (twins banked: 980/981) ----
cell a_spl320t glm4.5-air "$SPL" T1 320000 1 ASYM_ARENA_SHM_CAP_GB=240 ASYM_EP_SEP_SLOT_KMAX=4096 ASYM_EP_SEP_SLOT_ROWS=8192
# ---- 3. mixtral dev entry: T1 floor trio + T2 armed-probe trio ----
cell m_spl64 mixtral-8x22b "$SPL" T1 64000 1 ASYM_ARENA_SHM_CAP_GB=285 ASYM_EP_SEP_SLOT_KMAX=6144 ASYM_EP_SEP_SLOT_ROWS=8192
cell m_sp264 mixtral-8x22b "$SP2" T1 64000 1 ASYM_ARENA_SHM_CAP_GB=285
cell m_sdp64 mixtral-8x22b "$SDP" T1 64000 1 ASYM_ARENA_SHM_CAP_GB=285
cell m_splt2 mixtral-8x22b "$SPL" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=285 ASYM_EP_SEP_SLOT_KMAX=6144 ASYM_EP_SEP_SLOT_ROWS=131072 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
cell m_sp2t2 mixtral-8x22b "$SP2" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=285
cell m_sdpt2 mixtral-8x22b "$SDP" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=285
guard || true
echo "V1F-DONE $(date +%m-%d_%H:%M)" >> "$S"

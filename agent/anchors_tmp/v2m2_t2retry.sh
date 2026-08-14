#!/bin/bash
# (KMAX=6144 covers gate/up k=6144; down k=16384 capacity-declines by design).
set -uo pipefail
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/v2m2_status.log"

guard() {
  rm -f /dev/shm/asym_fabric_* /dev/shm/asym_seprobe_* 2>/dev/null || true
  for i in $(seq 1 360); do
    live=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && live="$live $p"
    done
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
echo "V2M2 begin $(date +%m-%d_%H:%M)" >> "$S"
cell m_splt2 mixtral-8x22b "$SPL" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=360 ASYM_EP_SEP_SLOT_KMAX=6144 ASYM_EP_SEP_SLOT_ROWS=131072 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
cell m_sp2t2 mixtral-8x22b "$SP2" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=360
cell m_sdpt2 mixtral-8x22b "$SDP" T2 64000 1 ASYM_ARENA_SHM_CAP_GB=360
guard || true
echo "V2M2-DONE $(date +%m-%d_%H:%M)" >> "$S"

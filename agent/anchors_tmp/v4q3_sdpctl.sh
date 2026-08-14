#!/bin/bash
# v3h_hunyuan.sh — V3 Hunyuan-A13B ladder (fix_dynamic_ep.md §5: dev 64k b2,
# ceiling 320k b1, both T2B — banked ceiling 1375). T2B is fg-class => the
# hook is ALIVE with default knobs; dev spl runs production posture (organic
# arm-or-decline), h_splt2b is the forced-arm correctness probe, ceiling spl
# runs minimal rings (floor posture; lazy-open makes peer state zero anyway).
# KMAX=6144 covers gate_up n=2*3072; arena 320 (T2B pins ~1.5x the ~160G
# weights per the mixtral T2 lesson; /dev/shm 479G).
set -uo pipefail
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES=512
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL=0,1 DDP_TIMEOUT=1500
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/v4q3_status.log"

guard() {
  rm -f /dev/shm/asym_fabric_* /dev/shm/asym_seprobe_* 2>/dev/null || true
  for i in $(seq 1 360); do
    live=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && live="$live $p"
    done
    ext=$(pgrep -f 'run_lf_profiled_[t]rain.py|build_lf_sft_[e]val_pair.py' 2>/dev/null | wc -l)
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
echo "V4Q3 begin $(date +%m-%d_%H:%M)" >> "$S"
# ---- dev 64k b2 T2B trio (spl production posture: default MPE gate) ----
# ---- forced-arm correctness probe (armed>0 requirement of §5) ----
# ---- ceiling 320k b1 T2B trio (spl minimal rings) ----
cell q122_sdp336 q3.5-122b-a10b "$SDP" T1 336000 1 ASYM_ARENA_SHM_CAP_GB=400
guard || true
echo "V4Q3-DONE $(date +%m-%d_%H:%M)" >> "$S"

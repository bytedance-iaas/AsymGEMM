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
S="$L/v3h6_status.log"

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
echo "V3H6 begin $(date +%m-%d_%H:%M)" >> "$S"
rm -rf profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/h_spl64__* profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/h_sp264__* profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/h_sdp64__* profiling_results/profiling/asym_long_sft_smoke__lora__lf__bf16/h_splt2b__*
# ---- dev 64k b2 T2B trio (spl production posture: default MPE gate) ----
cell h_spl64 hunyuan-a13b "$SPL" T2B 64000 2 ASYM_ARENA_SHM_CAP_GB=320 TRUST_REMOTE_CODE=false ASYM_EP_SEP_SLOT_KMAX=6144 ASYM_EP_SEP_SLOT_ROWS=65536
cell h_sp264 hunyuan-a13b "$SP2" T2B 64000 2 ASYM_ARENA_SHM_CAP_GB=320 TRUST_REMOTE_CODE=false
cell h_sdp64 hunyuan-a13b "$SDP" T2B 64000 2 ASYM_ARENA_SHM_CAP_GB=320 TRUST_REMOTE_CODE=false
# ---- forced-arm correctness probe (armed>0 requirement of §5) ----
cell h_splt2b hunyuan-a13b "$SPL" T2B 64000 2 ASYM_ARENA_SHM_CAP_GB=320 TRUST_REMOTE_CODE=false ASYM_EP_SEP_SLOT_KMAX=6144 ASYM_EP_SEP_SLOT_ROWS=262144 ASYM_EP_SEP_NO_SNAP=1 ASYM_EP_SEP_MAX_MPE=999999
# ---- ceiling 320k b1 T2B trio (spl minimal rings) ----
cell h_spl320 hunyuan-a13b "$SPL" T2B 320000 1 ASYM_ARENA_SHM_CAP_GB=320 TRUST_REMOTE_CODE=false ASYM_EP_SEP_SLOT_KMAX=6144 ASYM_EP_SEP_SLOT_ROWS=512
cell h_sp2320 hunyuan-a13b "$SP2" T2B 320000 1 ASYM_ARENA_SHM_CAP_GB=320 TRUST_REMOTE_CODE=false
cell h_sdp320 hunyuan-a13b "$SDP" T2B 320000 1 ASYM_ARENA_SHM_CAP_GB=320 TRUST_REMOTE_CODE=false
guard || true
echo "V3H6-DONE $(date +%m-%d_%H:%M)" >> "$S"

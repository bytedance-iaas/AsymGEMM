#!/bin/bash
# fig11_chain.sh — the 8 replay cells for the component-memory ablation
# (fig 11): asym-b1 + asym_torch(middle-row) cells, serial on GPU0.
set -uo pipefail
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/fig11t3f_status.log"
guard() {
  rm -f /dev/shm/asym_fabric_* 2>/dev/null || true
  for i in $(seq 1 360); do
    live=""
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      p=${p//,/}; [ -d "/proc/$p" ] && live="$live $p"
    done
    ext=$(pgrep -f 'run_lf_profiled_[t]rain.py|build_lf_sft_[e]val_pair.py|run_lf_lora_[s]ft.sh' 2>/dev/null | wc -l)
    [ -z "$live" ] && [ "${ext:-0}" -eq 0 ] && return 0
    sleep 20
  done
  echo "GUARD-TIMEOUT $(date +%H:%M)" >> "$S"; return 1
}
run_cell() {
  local tag="$1"
  guard || return 1
  echo "CELL-START $tag $(date +%m-%d_%H:%M)" >> "$S"
  bash "$L/fig11_cells/${tag}.sh" > "$L/vc_${tag}.log" 2>&1
  echo "CELL-END $tag rc=$? $(date +%m-%d_%H:%M)" >> "$S"
  rm -f /dev/shm/asym_fabric_* 2>/dev/null || true
}
echo "FIG11T3F begin $(date +%m-%d_%H:%M)" >> "$S"
cell_t3() { # tag model seq maxsamples
  local tag="$1" model="$2" seq="$3" ms="$4"
  guard || return 1
  echo "CELL-START $tag $(date +%m-%d_%H:%M)" >> "$S"
  ( export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
    export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
    export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES="$ms" CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
    RUN_NAME="$tag" RUNS="${model}|1 ; asym_cpuadamwds|T3|ligerloss1 ; ${seq}|1|1 ; none|false|false|false|false|false" \
      bash scripts/lf/profile_lora_lf_test_source.sh ) > "$L/vc_${tag}.log" 2>&1
  echo "CELL-END $tag rc=$? $(date +%m-%d_%H:%M)" >> "$S"
  rm -f /dev/shm/asym_fabric_* 2>/dev/null || true
}
cell_raw() { # tag model seq maxsamples recompute_token
  local tag="$1" model="$2" seq="$3" ms="$4" tok="$5"
  guard || return 1
  echo "CELL-START $tag $(date +%m-%d_%H:%M)" >> "$S"
  ( export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
    export NUMACTL_ENABLE=1 NUMACTL_MODE=membind NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1
    export PROFILERS=source MAX_STEPS=2 WARMUP_STEPS=1 MAX_SAMPLES="$ms" CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
    RUN_NAME="$tag" RUNS="${model}|1 ; asym_cpuadamwds|${tok}|ligerloss1 ; ${seq}|1|1 ; none|false|false|false|false|false" \
      bash scripts/lf/profile_lora_lf_test_source.sh ) > "$L/vc_${tag}.log" 2>&1
  echo "CELL-END $tag rc=$? $(date +%m-%d_%H:%M)" >> "$S"
  rm -f /dev/shm/asym_fabric_* 2>/dev/null || true
}
cell_raw q35so_256 q3.5-35b-a3b 256000 512 recomp
cell_raw q35uo_256 q3.5-35b-a3b 256000 512 unsloth-off-ohbm0
cell_t3 q35t3_256 q3.5-35b-a3b 256000 512
cell_raw q35uo_896 q3.5-35b-a3b 896000 512 unsloth-off-ohbm0
cell_t3 q35t3_896 q3.5-35b-a3b 896000 512
cell_t3 q35t3_1152 q3.5-35b-a3b 1152000 512
guard || true
echo "FIG11T3F-DONE $(date +%m-%d_%H:%M)" >> "$S"

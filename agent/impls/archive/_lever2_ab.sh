#!/bin/bash
# Lever-2 A/B (ON configs only; compare s8192 vs the existing OFF baseline 752/805 GiB, 193.78 s/it).
# MAX_STEPS=1 -> 5 warmup + 1 measured = 6 steps. Captures peak RSS + a steady s/it.
set -u
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
R=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/impls/_lever2_results
mkdir -p "$R"; : > "$R/summary.txt"

metrics () {  # $1=label $2=seq
  local label="$1" seq="$2" dir stage rss1 rss2 peak hbm sit
  dir=$(find profiling -path "*qwen3-30b-a3b*asym_cpuadamwds*b8_s${seq}*" -name process_memory.csv -newermt '-40 min' -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
  stage=$(python3 -c "import json;print(json.load(open('$dir/lf_run/heartbeat.latest.json'))['stage'])" 2>/dev/null || echo NA)
  rss1=$(awk -F, 'NR>1&&$5+0>m{m=$5} END{print m+0}' "$dir/process_memory.csv" 2>/dev/null)
  rss2=$(awk -F'|' '/optimizer_step_(after|before|start)/{gsub(/ /,"",$3); if($3+0>m)m=$3} END{print m+0}' "$dir/summary.md" 2>/dev/null)
  peak=$(awk -v a="${rss1:-0}" -v b="${rss2:-0}" 'BEGIN{m=(a>b?a:b); printf "%.1f", m/1073741824}')
  hbm=$(awk -F'|' '/peak_reserved_hbm_bytes/{gsub(/ /,"",$3);print $3}' "$dir/memory.md" 2>/dev/null)
  sit=$(grep -oE '[0-9.]+s/it' "$R/${label}.log" 2>/dev/null | tail -1)
  { echo "## $label (seq=$seq)  stage=$stage"
    echo "   CPU_rss_peak_GiB=$peak   peak_reserved_HBM_MiB=$hbm   step_time=${sit:-NA}"
    echo "   dir=$dir"; echo
  } | tee -a "$R/summary.txt"
}

run () {  # $1=label $2=seq ; rest=flags
  local label="$1" seq="$2"; shift 2
  echo ">>> $(date +%H:%M:%S) RUN $label seq=$seq flags=[$*]" | tee -a "$R/summary.txt"
  env "$@" NUMACTL_MODE=membind BACKEND_SPECS="asym_cpuadamwds|norecomp" SEQ_LENS="$seq" GPU_POOL=0 \
    PER_DEVICE_TRAIN_BATCH_SIZE=8 MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" WARMUP_STEPS=5 MAX_STEPS=1 \
    bash scripts/lf/profile_lora_lf.sh > "$R/${label}.log" 2>&1
  metrics "$label" "$seq"
}

run 2A_s8192    8192  ASYM_OFFLOAD_ACT_RECOMPUTE=1
run 2B_s8192    8192  ASYM_OFFLOAD_X_UNPACKED=1
run BOTH_s8192  8192  ASYM_OFFLOAD_ACT_RECOMPUTE=1 ASYM_OFFLOAD_X_UNPACKED=1
run OFF_s10240  10240
run BOTH_s10240 10240 ASYM_OFFLOAD_ACT_RECOMPUTE=1 ASYM_OFFLOAD_X_UNPACKED=1
echo ">>> $(date +%H:%M:%S) DONE  (OFF_s8192 baseline = 805 GiB peak, 193.78 s/it)" | tee -a "$R/summary.txt"

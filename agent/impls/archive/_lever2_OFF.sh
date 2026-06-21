#!/bin/bash
set -u
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
R=/workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/impls/_lever2_results
echo ">>> $(date +%H:%M:%S) OFF_s8192 start" | tee "$R/OFF_summary.txt"
OVERWRITE=true NUMACTL_MODE=membind BACKEND_SPECS="asym_cpuadamwds|norecomp" SEQ_LENS=8192 GPU_POOL=0 \
  PER_DEVICE_TRAIN_BATCH_SIZE=8 MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" WARMUP_STEPS=5 MAX_STEPS=1 \
  bash scripts/lf/profile_lora_lf.sh > "$R/OFF_s8192.log" 2>&1
dir=$(find profiling -path "*qwen3-30b-a3b*asym_cpuadamwds*b8_s8192*" -name process_memory.csv -newermt '-40 min' -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
stage=$(python3 -c "import json;print(json.load(open('$dir/lf_run/heartbeat.latest.json'))['stage'])" 2>/dev/null || echo NA)
rss1=$(awk -F, 'NR>1&&$5+0>m{m=$5} END{print m+0}' "$dir/process_memory.csv" 2>/dev/null)
rss2=$(awk -F'|' '/optimizer_step_(after|before|start)/{gsub(/ /,"",$3); if($3+0>m)m=$3} END{print m+0}' "$dir/summary.md" 2>/dev/null)
peak=$(awk -v a="${rss1:-0}" -v b="${rss2:-0}" 'BEGIN{m=(a>b?a:b); printf "%.1f", m/1073741824}')
sit=$(grep -oE '[0-9.]+s/it' "$R/OFF_s8192.log" 2>/dev/null | tail -1)
echo "## OFF_s8192 (fresh, OVERWRITE)  stage=$stage  CPU_rss_peak_GiB=$peak  step_time=${sit:-NA}" | tee -a "$R/OFF_summary.txt"
echo ">>> $(date +%H:%M:%S) OFF_s8192 done" | tee -a "$R/OFF_summary.txt"

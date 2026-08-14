#!/bin/bash
# fig11_chain.sh — the 8 replay cells for the component-memory ablation
# (fig 11): asym-b1 + asym_torch(middle-row) cells, serial on GPU0.
set -uo pipefail
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
L=/workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp
S="$L/fig11b_status.log"
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
echo "FIG11B begin $(date +%m-%d_%H:%M)" >> "$S"
run_cell fm_mid256
run_cell qm_mid320
run_cell fm_mid640
run_cell qm_mid1100
run_cell fm_mid1152
run_cell qm_mid1600
guard || true
echo "FIG11B-DONE $(date +%m-%d_%H:%M)" >> "$S"

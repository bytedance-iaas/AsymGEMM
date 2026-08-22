#!/bin/bash
# stdtps96_occ.sh — run ONE HBM occupier inside its own container instance
# (launcher passes NVIDIA_VISIBLE_DEVICES=<one physical gpu>; inside = dev 0).
# Appends "<host-pid> phys=<tag>" to the shared pids file, then stays up.
cd /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM
PHYS="${1:?usage: stdtps96_occ.sh <phys-gpu-label>}"
# the container sees exactly ONE GPU (NVIDIA_VISIBLE_DEVICES=<phys>) which is
# inside-index 0 — the launcher's CUDA_VISIBLE_DEVICES=<phys> would filter it
# out for phys!=0, so force the inside-index here.
export CUDA_VISIBLE_DEVICES=0
PIDF=agent/anchors_tmp/stdtps96_occupier.pids
S=agent/anchors_tmp/stdtps96_status.log
.venv/bin/python agent/anchors_tmp/hbm96_occupy.py --device 0 &
P=$!
echo "$P" >> "$PIDF"
echo "STDTPS96-OCCUPIER phys=$PHYS pid=$P up $(date '+%F %H:%M:%S')" >> "$S"
wait $P
echo "STDTPS96-OCCUPIER phys=$PHYS pid=$P EXITED $(date '+%F %H:%M:%S')" >> "$S"

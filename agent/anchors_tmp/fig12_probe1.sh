#!/bin/bash
# fig12 probe 1 — Qwen3-30B-A3B @ 96k b8, T3 preset, GPU0, serial.
# Arms: A0 shipped T3 | B0 shipped+reaim | A1 streamed-legs T3 | B1 streamed+reaim.
# (A/B interleaved same-day per house A/B rule.)
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-600}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POL="none|false|true|false|false|false"   # attnact only (m6/anchor T3 shape)
STREAM_ENV="ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=0 ASYMM_QWEN3_MOE_FG_DA_GPU=0"

SYS="asym_cpuadamwds|T3"
echo "PROBE1 begin $(date +%H:%M)" >> "$S"
ARM_ENV=""                                     run_cell kfa0 q3-30b-a3b "$SYS" 96000 "8" "$POL" 1
ARM_ENV="ASYMM_LORA_KERNELS=reaim"             run_cell kfb0 q3-30b-a3b "$SYS" 96000 "8" "$POL" 1
ARM_ENV="$STREAM_ENV"                          run_cell kfa1 q3-30b-a3b "$SYS" 96000 "8" "$POL" 1
ARM_ENV="$STREAM_ENV ASYMM_LORA_KERNELS=reaim" run_cell kfb1 q3-30b-a3b "$SYS" 96000 "8" "$POL" 1
echo "PROBE1-DONE $(date +%H:%M)" >> "$S"

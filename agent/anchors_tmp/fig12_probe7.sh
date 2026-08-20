#!/bin/bash
# fig12 probe 7 — finite-gain hunt: naive AsymGEMM INTEGRATION (raw token =
# no tier recipe env: no placement policy, no CPU stack, library-default
# dispatch, standard unsloth-GC) vs shipped AsymLoRA, same cell, same day.
# Plus reaim (X.cpu@A) pairs on the fat-leg models (122B) at short lengths.
# All 1 GPU, serial, W1+M2. Short lengths = cheap cells.
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-500}" GPU_POOL="${GPU:-0}"
. /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
POLA="none|false|true|false|false|false"   # shipped arms: attnact on (T3-class)
POLN="none|false|false|false|false|false"  # naive arms: nothing

echo "PROBE7 begin $(date +%H:%M)" >> "$S"
# ---- 30B @96k b8: naive-integration vs shipped (fresh same-day pair) ----
ARM_ENV="" run_cell p7n96 q3-30b-a3b "asym_cpuadamwds|unsloth" 96000 "8" "$POLN" 1
ARM_ENV="" run_cell p7a96 q3-30b-a3b "asym_cpuadamwds|T3" 96000 "8" "$POLA" 1
# ---- 30B @320k b1 pair (the old figure's length) ----
ARM_ENV="" run_cell p7n320 q3-30b-a3b "asym_cpuadamwds|unsloth" 320000 "1" "$POLN" 1
ARM_ENV="" run_cell p7a320 q3-30b-a3b "asym_cpuadamwds|T2" 320000 "1" "$POLA" 1
# ---- 122B @32k b12: naive vs shipped-T1(recipe) + reaim on shipped ----
ARM_ENV="" run_cell p7n122_32 q3.5-122b-a10b "asym_cpuadamwds|unsloth" 32000 "12 8" "$POLN" 1
ARM_ENV="" run_cell p7a122_32 q3.5-122b-a10b "asym_cpuadamwds|T1" 32000 "12 8" "$POLN" 1
ARM_ENV="ASYMM_LORA_KERNELS=reaim" run_cell p7r122_32 q3.5-122b-a10b "asym_cpuadamwds|T1" 32000 "12 8" "$POLN" 1
# ---- 122B @128k b3 pair ----
ARM_ENV="" run_cell p7n122_128 q3.5-122b-a10b "asym_cpuadamwds|unsloth" 128000 "3" "$POLN" 1
ARM_ENV="" run_cell p7a122_128 q3.5-122b-a10b "asym_cpuadamwds|T1" 128000 "3" "$POLN" 1
# ---- Mixtral @64k b2 pair ----
ARM_ENV="" run_cell p7nmx64 mixtral-8x22b "asym_cpuadamwds|unsloth" 64000 "2" "$POLN" 1
ARM_ENV="" run_cell p7amx64 mixtral-8x22b "asym_cpuadamwds|T1" 64000 "2" "$POLN" 1
echo "PROBE7-DONE $(date +%H:%M)" >> "$S"

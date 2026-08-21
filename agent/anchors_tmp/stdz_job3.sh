#!/bin/bash
# stdz_job3.sh — TP x-axis standardization, Job 3: GLM-4.5-Air + GLM-4.7-Flash.
# Launch with stdz_launch.sh on a 2-GPU set. Cell lists (standardize_tps.md):
#  Air-1r : ZERO runs — grid 128/192/256/320/384/448K all measured.
#  Air-2r : THE CAP GATE — asym-2r@448 then @384 (sdp2-T1, arena 240 per GLM
#           precedent). 448 fits -> grid 128-448K; only 384 -> grid 64-384K;
#           neither -> STOP, report (grid redesign). Baselines beyond their
#           measured walls (uns (192,256], rc (128,160]) -> OOM free.
#  Flash-1r: asym@512 (T1), asym@768 (T1->T2 walk), asym@896 (T2->T3 walk);
#           uns@512 REUSED from banked off-render measurement (310, un512);
#           other new-column baselines beyond measured walls -> OOM free.
#  Flash-2r: new col 384K -> asym (sdp2-T1), uns (fits, 416 banked 754),
#           rc + fsdp2 probes (bracket (320,416]); zero3 est-derives off rc.
set -u
export GPU="0,1" HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500
S="$LOGD/stdz_status.log"
POL="none|false|false|false|false|false"
UNS="superoffload_mem|unsloth"
RC="superoffload_mem|recomp"
F2="fsdp2_offload|recomp"
SDP="asym_sdp2_cpuadamwds"
echo "== stdz job3 (GLMs) 2r-gate start $(date +%F_%H:%M) ==" >> "$S"

# ---- Air cap gate (2r first — it decides the grid)
v448=$(ARM_ENV="ASYM_ARENA_SHM_CAP_GB=240" run_cell s2aair_448 glm4.5-air "$SDP|T1" 448000 "1" "$POL" 2)
[ "$v448" != "TRAINED" ] && ARM_ENV="ASYM_ARENA_SHM_CAP_GB=240" run_cell s2aair_448b glm4.5-air "$SDP|T2" 448000 "1" "$POL" 2 >/dev/null
v384=$(ARM_ENV="ASYM_ARENA_SHM_CAP_GB=240" run_cell s2aair_384 glm4.5-air "$SDP|T1" 384000 "1" "$POL" 2)
[ "$v384" != "TRAINED" ] && ARM_ENV="ASYM_ARENA_SHM_CAP_GB=240" run_cell s2aair_384b glm4.5-air "$SDP|T2" 384000 "1" "$POL" 2 >/dev/null
echo "GATE air-2r: 448=$v448 384=$v384 (see CELL lines for T2 fallbacks)" >> "$S"

# ---- Flash 2r new column 384K
ARM_ENV="ASYM_ARENA_SHM_CAP_GB=240" run_cell s2afl_384 glm4.7-flash "$SDP|T1" 384000 "1" "$POL" 2
ARM_ENV="" run_cell s2ufl_384 glm4.7-flash "$UNS" 384000 "1" "$POL" 2
ARM_ENV="" run_cell s2rfl_384 glm4.7-flash "$RC" 384000 "1" "$POL" 2
ARM_ENV="" run_cell s2ffl_384 glm4.7-flash "$F2" 384000 "1" "$POL" 2

# ---- Flash 1r fills (single GPU)
echo "== stdz job3 1r start $(date +%F_%H:%M) ==" >> "$S"
export GPU=0 CUDA_VISIBLE_DEVICES=0 HOSTFLOOR=600; unset GPU_POOL DDP_TIMEOUT || true
ARM_ENV="" run_cell s1afl_512 glm4.7-flash "asym_cpuadamwds|T1" 512000 "1" "$POL" 1
v=$(ARM_ENV="" run_cell s1afl_768 glm4.7-flash "asym_cpuadamwds|T1" 768000 "1" "$POL" 1)
[ "$v" != "TRAINED" ] && ARM_ENV="" run_cell s1afl_768b glm4.7-flash "asym_cpuadamwds|T2" 768000 "1" "$POL" 1
v=$(ARM_ENV="" run_cell s1afl_896 glm4.7-flash "asym_cpuadamwds|T2" 896000 "1" "$POL" 1)
[ "$v" != "TRAINED" ] && ARM_ENV="" run_cell s1afl_896b glm4.7-flash "asym_cpuadamwds|T3" 896000 "1" "$POL" 1
echo "== stdz job3 DONE $(date +%F_%H:%M) ==" >> "$S"

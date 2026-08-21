#!/bin/bash
# OBSOLETE (2026-08-20 17:1x) — never run / superseded: Agent 4 done by Session B (c11); Flash-1r asym cells run by B; Air 384k block ran inside stdtp_a3_air.sh. Kept as campaign record (STDTP_LOG.md).
# stdtp_a3_flash1r.sh — Agent 3 / phase 3 (standardize_tps.md): GLM-4.7-Flash
# 1-rank asym cells at the three missing grid rungs 512k/768k/896k.
# Banked ladder context: T1 448k=354, T1 640k=248 @148.6GiB; 1024k=T2 150.
# 512k: T1 b1 (fits by slope). 768k: T1 b1 -> T2 on GOOM (wall pin).
# 896k: start from 768k's landing tier (monotonicity covers T1 if its wall
# was pinned at 768k); walk T2 -> T2B -> T3 on OOM.
# Baseline rows at these rungs are beyond measured walls (rc/fd/uns/zero3/mega)
# -> OOM by monotonicity; uns@512k reuses the banked off-render 310.
# uo@512k/768k: x1e artifacts lost -> est via house fit (main-dropped series).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0" HOSTFLOOR=500
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtp_lib.sh
unset GPU_POOL DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
python3 /workspace/AsymGEMM-SFT-38/.repair_dataset_info.py >> "$S" 2>&1 || echo "repair-dataset-info FAILED" >> "$S"
echo "== STDTP-A3-FLASH1R begin (1r, phys GPU 3 -> inside 0) $(date '+%m-%d %H:%M')" >> "$S"
BE=asym_cpuadamwds
P="none|false|false|false|false|false"

run_cell f1t1512 glm4.7-flash "${BE}|T1" 512000 "1" "$P" 1 >/dev/null

v768=$(run_cell f1t1768 glm4.7-flash "${BE}|T1" 768000 "1" "$P" 1)
t896_start=T1
if [ "$v768" != "TRAINED" ]; then
  ooms "$v768" || { echo "ABORT-A3-F1R: f1t1768 infra $v768" >> "$S"; exit 1; }
  t896_start=T2
  v768=$(run_cell f1t2768 glm4.7-flash "${BE}|T2" 768000 "1" "$P" 1)
  [ "$v768" != "TRAINED" ] && v768=$(run_cell f1t2b768 glm4.7-flash "${BE}|T2B" 768000 "1" "$P" 1)
  [ "$v768" != "TRAINED" ] && run_cell f1t3768 glm4.7-flash "${BE}|T3" 768000 "1" "$P" 1 >/dev/null
fi

if [ "$t896_start" = "T1" ]; then
  v896=$(run_cell f1t1896 glm4.7-flash "${BE}|T1" 896000 "1" "$P" 1)
  [ "$v896" = "TRAINED" ] || t896_start=T2
else
  v896=GOOM
fi
if [ "$t896_start" = "T2" ] && [ "${v896:-GOOM}" != "TRAINED" ]; then
  v896=$(run_cell f1t2896 glm4.7-flash "${BE}|T2" 896000 "1" "$P" 1)
  [ "$v896" != "TRAINED" ] && v896=$(run_cell f1t2b896 glm4.7-flash "${BE}|T2B" 896000 "1" "$P" 1)
  [ "$v896" != "TRAINED" ] && run_cell f1t3896 glm4.7-flash "${BE}|T3" 896000 "1" "$P" 1 >/dev/null
fi
echo "== STDTP-A3-FLASH1R-DONE $(date '+%m-%d %H:%M')" >> "$S"

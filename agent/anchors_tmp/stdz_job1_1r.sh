#!/bin/bash
# stdz_job1_1r.sh — TP x-axis standardization campaign, Job 1 rank-1 cells
# (standardize_tps.md): Qwen3-30B-A3B grid 384/512/640/768/896/1024K and
# Qwen3.5-122B-A10B grid 160/192/224/256/288/320K. Only MISSING cells run;
# measured cells reused per the doc. Runs INSIDE the container, single GPU
# (inside id 0). Sourced protocol from fig12_lib.sh (w1+m2, PROFILERS=source,
# serial guard). Verdicts -> stdz_status.log; tok/s harvested after.
set -u
export GPU=0 HOSTFLOOR=500
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
S="$LOGD/stdz_status.log"
POL="none|false|false|false|false|false"
UNS="superoffload_mem|unsloth"
RC="superoffload_mem|recomp"
Z3="zero3_offload_mem|recomp"
F2="fsdp2_offload|recomp"
echo "== stdz job1 1r start $(date +%F_%H:%M) host=$(hostname) ==" >> "$S"

# ---- 122B rank-1 fills: asym+uns fit everywhere on-grid <288k (uns wall
# (288,320]); rc bracket (128,288] -> descend and record; zero3 only where rc fit.
ARM_ENV="" run_cell s1u122_160 q3.5-122b-a10b "$UNS" 160000 "2 1" "$POL" 1
ARM_ENV="" run_cell s1a122_160 q3.5-122b-a10b "asym_cpuadamwds|T1" 160000 "2 1" "$POL" 1
ARM_ENV="" run_cell s1u122_192 q3.5-122b-a10b "$UNS" 192000 "2 1" "$POL" 1
ARM_ENV="" run_cell s1a122_192 q3.5-122b-a10b "asym_cpuadamwds|T1" 192000 "2 1" "$POL" 1
ARM_ENV="" run_cell s1u122_224 q3.5-122b-a10b "$UNS" 224000 "1" "$POL" 1
ARM_ENV="" run_cell s1a122_224 q3.5-122b-a10b "asym_cpuadamwds|T1" 224000 "1" "$POL" 1
ARM_ENV="" run_cell s1u122_256 q3.5-122b-a10b "$UNS" 256000 "1" "$POL" 1
ARM_ENV="" run_cell s1a122_256 q3.5-122b-a10b "asym_cpuadamwds|T1" 256000 "1" "$POL" 1
declare -A RCV
for sq in 256 224 192 160; do
  v=$(ARM_ENV="" run_cell "s1r122_${sq}" q3.5-122b-a10b "$RC" "${sq}000" "1" "$POL" 1)
  RCV[$sq]="$v"
done
for sq in 256 224 192 160; do
  if [ "${RCV[$sq]}" = "TRAINED" ]; then
    ARM_ENV="" run_cell "s1z122_${sq}" q3.5-122b-a10b "$Z3" "${sq}000" "1" "$POL" 1
  else
    echo "CELL s1z122_${sq} zero3 s=${sq}000 -> SKIP-OOM (rc ${RCV[$sq]}, bracket-identical)" >> "$S"
  fi
done

# ---- 30B rank-1 fills. Walls: rc (392,400] -> rc fits 384; uns (640,660] ->
# 768+ OOM free; fsdp2 bracket (320,480] -> 384 is a genuine probe.
ARM_ENV="" run_cell s1u30_384 q3-30b-a3b "$UNS" 384000 "1" "$POL" 1
ARM_ENV="" run_cell s1a30_384 q3-30b-a3b "asym_cpuadamwds|T2" 384000 "1" "$POL" 1
v=$(ARM_ENV="" run_cell s1r30_384 q3-30b-a3b "$RC" 384000 "1" "$POL" 1)
if [ "$v" = "TRAINED" ]; then
  ARM_ENV="" run_cell s1z30_384 q3-30b-a3b "$Z3" 384000 "1" "$POL" 1
else
  echo "CELL s1z30_384 zero3 s=384000 -> SKIP-OOM (rc $v, bracket-identical)" >> "$S"
fi
ARM_ENV="" run_cell s1f30_384 q3-30b-a3b "$F2" 384000 "1" "$POL" 1
ARM_ENV="" run_cell s1u30_512 q3-30b-a3b "$UNS" 512000 "1" "$POL" 1
ARM_ENV="" run_cell s1a30_512 q3-30b-a3b "asym_cpuadamwds|T2" 512000 "1" "$POL" 1
ARM_ENV="" run_cell s1a30_768 q3-30b-a3b "asym_cpuadamwds|T2" 768000 "1" "$POL" 1
export HOSTFLOOR=700
v=$(ARM_ENV="" run_cell s1a30_896 q3-30b-a3b "asym_cpuadamwds|T2" 896000 "1" "$POL" 1)
if [ "$v" != "TRAINED" ]; then
  ARM_ENV="" run_cell s1a30_896b q3-30b-a3b "asym_cpuadamwds|T2B" 896000 "1" "$POL" 1
fi
v=$(ARM_ENV="" run_cell s1a30_1024 q3-30b-a3b "asym_cpuadamwds|T2B" 1024000 "1" "$POL" 1)
if [ "$v" != "TRAINED" ]; then
  ARM_ENV="" run_cell s1a30_1024c q3-30b-a3b "asym_cpuadamwds|T3" 1024000 "1" "$POL" 1
fi
echo "== stdz job1 1r DONE $(date +%F_%H:%M) ==" >> "$S"

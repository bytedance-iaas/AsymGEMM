#!/bin/bash
# stdtps46_a2.sh — AGENT 2 (Qwen3.5-35B-A3B + Mixtral-8x22B) of the TP-figure
# x-axis standardization campaign, c18 / SFT-46 lane (Session D). Serial,
# in-container, one run at a time on the node. Phases per the LIVE CLAIMS
# entry in agent/impls/s04-p1-dgx-02-c06/standardize_tps.md:
#   A  35B-2r asym sEP-T2 @768k (the only required 35B cell)
#   B  Mixtral-1r: 288k probes (un, asym T1->T2), 160k column, 224k column
#   C  Mixtral-2r: 160k column, 224k column, fsdp2@128k probe
#   D  35B upgrades: 2r 256k best-over-batch, 2r asym@384k b2, 1r T1@640k
#   E  lean-only fills (uns-OFF)
# Idempotent: a tag+batch with a banked `ok` jobs.tsv is skipped.
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps46_lib.sh
POL="none|false|false|false|false|false"
Q35=q3.5-35b-a3b
MX=mixtral-8x22b
RC="superoffload_mem|recomp"
UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
Z3="zero3_offload_mem|recomp"
FD="fsdp2_offload|recomp"
echo "=== STDTPS46-A2 BEGIN $(date '+%F %H:%M:%S') node=$(hostname) ===" >> "$S"

# ---------------- Phase A: 35B-2r asym sEP-T2 @768k (REQUIRED) ----------------
echo "PHASE-A begin $(date +%H:%M)" >> "$S"
v=$(run_cell s2q35sep768 $Q35 "asym_sepplan2_cpuadamwds|T2" 768000 "1" "$POL" 2)
echo "PHASE-A sep-T2@768k -> $v" >> "$S"
if [ "$v" != "TRAINED" ]; then
  # tier ladder on OOM (T2 -> T2B), exactly as the 1r ladder did
  v2=$(run_cell s2q35sep768b $Q35 "asym_sepplan2_cpuadamwds|T2B" 768000 "1" "$POL" 2)
  echo "PHASE-A sep-T2B@768k -> $v2" >> "$S"
fi
echo "PHASE-A-DONE $(date +%H:%M)" >> "$S"

# ---------------- Phase B: Mixtral-1r ----------------
echo "PHASE-B begin $(date +%H:%M)" >> "$S"
# B1 288k: the doc-mandated uns probe (bracket (256k,320k]) + asym tier ladder
vu=$(run_cell s2mxun288 $MX "$UN" 288000 "1" "$POL" 1);  echo "PHASE-B un@288k -> $vu" >> "$S"
vt=$(run_cell s2mxt1288 $MX "asym_cpuadamwds|T1" 288000 "1" "$POL" 1); echo "PHASE-B T1@288k -> $vt" >> "$S"
if [ "$vt" != "TRAINED" ]; then
  vt2=$(run_cell s2mxt2288 $MX "asym_cpuadamwds|T2" 288000 "1" "$POL" 1); echo "PHASE-B T2@288k -> $vt2" >> "$S"
  if [ "$vt2" != "TRAINED" ]; then
    vt3=$(run_cell s2mxt2b288 $MX "asym_cpuadamwds|T2B" 288000 "1" "$POL" 1); echo "PHASE-B T2B@288k -> $vt3" >> "$S"
  fi
fi
# B2 160k column (rc inside its (128k,192k] bracket; un/asym fit by 192k; zero3 rc-class)
v=$(run_cell s2mxrc160 $MX "$RC" 160000 "1" "$POL" 1);  echo "PHASE-B rc@160k -> $v" >> "$S"
v=$(run_cell s2mxun160 $MX "$UN" 160000 "1" "$POL" 1);  echo "PHASE-B un@160k -> $v" >> "$S"
v=$(run_cell s2mxt1160 $MX "asym_cpuadamwds|T1" 160000 "1" "$POL" 1); echo "PHASE-B T1@160k -> $v" >> "$S"
v=$(run_cell s2mxz3160 $MX "$Z3" 160000 "1" "$POL" 1);  echo "PHASE-B z3@160k -> $v" >> "$S"
v=$(run_cell s2mxz3128 $MX "$Z3" 128000 "1" "$POL" 1);  echo "PHASE-B z3@128k -> $v" >> "$S"
# B3 224k column (un/asym fit by 256k)
v=$(run_cell s2mxun224 $MX "$UN" 224000 "1" "$POL" 1);  echo "PHASE-B un@224k -> $v" >> "$S"
v=$(run_cell s2mxt1224 $MX "asym_cpuadamwds|T1" 224000 "1" "$POL" 1); echo "PHASE-B T1@224k -> $v" >> "$S"
echo "PHASE-B-DONE $(date +%H:%M)" >> "$S"

# ---------------- Phase C: Mixtral-2r (GPUs 0+1, sdp2 shared fabric, arena 285) ----------------
echo "PHASE-C begin $(date +%H:%M)" >> "$S"
MXARM="ASYM_ARENA_SHM_CAP_GB=285"
v=$(run_cell s2mx2rc160 $MX "$RC" 160000 "1" "$POL" 2);      echo "PHASE-C rc@160k -> $v" >> "$S"
v=$(run_cell s2mx2un160 $MX "$UN" 160000 "2 1" "$POL" 2);    echo "PHASE-C un@160k -> $v" >> "$S"
v=$(run_cell s2mx2z3160 $MX "$Z3" 160000 "1" "$POL" 2);      echo "PHASE-C z3@160k -> $v" >> "$S"
v=$(ARM_ENV="$MXARM" run_cell s2mx2t1160 $MX "asym_sdp2_cpuadamwds|T1" 160000 "2 1" "$POL" 2); echo "PHASE-C sdp2-T1@160k -> $v" >> "$S"
v=$(run_cell s2mx2un224 $MX "$UN" 224000 "1" "$POL" 2);      echo "PHASE-C un@224k -> $v" >> "$S"
v=$(ARM_ENV="$MXARM" run_cell s2mx2t1224 $MX "asym_sdp2_cpuadamwds|T1" 224000 "1" "$POL" 2); echo "PHASE-C sdp2-T1@224k -> $v" >> "$S"
# fsdp2 2r row is derived-est today: probe 128k; if it fits, walk the rc-class bracket
vf=$(run_cell s2mx2fd128 $MX "$FD" 128000 "1" "$POL" 2);     echo "PHASE-C fsdp2@128k -> $vf" >> "$S"
if [ "$vf" = "TRAINED" ]; then
  vf=$(run_cell s2mx2fd160 $MX "$FD" 160000 "1" "$POL" 2);   echo "PHASE-C fsdp2@160k -> $vf" >> "$S"
  if [ "$vf" = "TRAINED" ]; then
    vf=$(run_cell s2mx2fd192 $MX "$FD" 192000 "1" "$POL" 2); echo "PHASE-C fsdp2@192k -> $vf" >> "$S"
    if [ "$vf" = "TRAINED" ]; then
      vf=$(run_cell s2mx2fd224 $MX "$FD" 224000 "1" "$POL" 2); echo "PHASE-C fsdp2@224k -> $vf" >> "$S"
      [ "$vf" = "TRAINED" ] && { vf=$(run_cell s2mx2fd256 $MX "$FD" 256000 "1" "$POL" 2); echo "PHASE-C fsdp2@256k -> $vf" >> "$S"; }
      [ "$vf" = "TRAINED" ] && { vf=$(run_cell s2mx2fd288 $MX "$FD" 288000 "1" "$POL" 2); echo "PHASE-C fsdp2@288k -> $vf" >> "$S"; }
    fi
  fi
fi
echo "PHASE-C-DONE $(date +%H:%M)" >> "$S"

# ---------------- Phase D: 35B upgrades ----------------
echo "PHASE-D begin $(date +%H:%M)" >> "$S"
# D1 2r 256k best-over-batch (b1 fill cells sit at 43/51/25% HBM)
v=$(run_cell s2q35sep256 $Q35 "asym_sepplan2_cpuadamwds|T2" 256000 "4 3 2" "$POL" 2); echo "PHASE-D sep-T2@256k walk -> $v" >> "$S"
v=$(run_cell s2q35un256 $Q35 "$UN" 256000 "2" "$POL" 2);  echo "PHASE-D un@256k b2 -> $v" >> "$S"
v=$(run_cell s2q35uo256 $Q35 "$UO" 256000 "2" "$POL" 2);  echo "PHASE-D uo@256k b2 -> $v" >> "$S"
# D2 2r asym T2@384k b2 up-probe (56% HBM at b1)
v=$(run_cell s2q35sep384 $Q35 "asym_sepplan2_cpuadamwds|T2" 384000 "2" "$POL" 2); echo "PHASE-D sep-T2@384k b2 -> $v" >> "$S"
# D3 1r T1@640k (fastest-fitting-tier check; T1 never probed past 576k)
v=$(run_cell s2q35t1640 $Q35 "asym_cpuadamwds|T1" 640000 "1" "$POL" 1); echo "PHASE-D T1@640k -> $v" >> "$S"
if [ "$v" = "TRAINED" ]; then
  v=$(run_cell s2q35t1768 $Q35 "asym_cpuadamwds|T1" 768000 "1" "$POL" 1); echo "PHASE-D T1@768k -> $v" >> "$S"
fi
echo "PHASE-D-DONE $(date +%H:%M)" >> "$S"

# ---------------- Phase E: lean-only fills (uns-OFF) ----------------
echo "PHASE-E begin $(date +%H:%M)" >> "$S"
v=$(run_cell s2q35uo640 $Q35 "$UO" 640000 "1" "$POL" 1);  echo "PHASE-E 35B uo@640k -> $v" >> "$S"
v=$(run_cell s2q35uo768 $Q35 "$UO" 768000 "1" "$POL" 1);  echo "PHASE-E 35B uo@768k -> $v" >> "$S"
v=$(run_cell s2mxuo160 $MX "$UO" 160000 "1" "$POL" 1);    echo "PHASE-E MX uo@160k 1r -> $v" >> "$S"
v=$(run_cell s2mx2uo160 $MX "$UO" 160000 "1" "$POL" 2);   echo "PHASE-E MX uo@160k 2r -> $v" >> "$S"
echo "PHASE-E-DONE $(date +%H:%M)" >> "$S"
echo "=== STDTPS46-A2 ALL DONE $(date '+%F %H:%M:%S') ===" >> "$S"

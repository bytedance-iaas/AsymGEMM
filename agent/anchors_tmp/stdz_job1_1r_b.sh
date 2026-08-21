#!/bin/bash
# stdz_job1_1r_b.sh — Job-1 rank-1 FOLLOW-UP: batch-ceiling probes the main
# chain's b1-only lists left open (max-TP-over-batch rule). The measured b2
# fits at 122B 160k/192k refute the banked "b1 beyond 128k" belief, so the
# new mid rungs need their b2 tried. GOOM verdicts just bracket the batch
# wall (b1 cells stand); TRAINED verdicts upgrade the cell.
set -u
export GPU=0 HOSTFLOOR=500
source /workspace/AsymGEMM-SFT/third_party/AsymGEMM/agent/anchors_tmp/fig12_lib.sh
S="$LOGD/stdz_status.log"
POL="none|false|false|false|false|false"
echo "== stdz job1 1r-b (batch probes) start $(date +%F_%H:%M) ==" >> "$S"
ARM_ENV="" run_cell s1a122_224p q3.5-122b-a10b "asym_cpuadamwds|T1" 224000 "2" "$POL" 1
ARM_ENV="" run_cell s1a122_256p q3.5-122b-a10b "asym_cpuadamwds|T1" 256000 "2" "$POL" 1
ARM_ENV="" run_cell s1a30_384p q3-30b-a3b "asym_cpuadamwds|T2" 384000 "2" "$POL" 1
# T2@896k fit (529, s1a30_896) => T2 wall now (896k,1.1M] spans the 1024k
# rung — max-TP demands trying T2 there against the main chain's T2B.
ARM_ENV="" run_cell s1a30_1024t2 q3-30b-a3b "asym_cpuadamwds|T2" 1024000 "1" "$POL" 1
echo "== stdz job1 1r-b DONE $(date +%F_%H:%M) ==" >> "$S"

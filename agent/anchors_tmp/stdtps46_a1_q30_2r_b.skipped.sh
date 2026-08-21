#!/bin/bash
# stdtps46_a1_q30_2r_b.sh — follow-up to stdtps46_a1_q30_2r.sh: 30B-2r asym
# up-probes where the b1 cell sits far below HBM and b2 equals a banked FIT
# footprint (512k b1 = 63%; b2 = 1.02M tokens/rank ~ the banked 1.04M-b1 fit).
set -uo pipefail
. /workspace/AsymGEMM-SFT-46/third_party/AsymGEMM/agent/anchors_tmp/stdtps46_lib.sh
POL="none|false|false|false|false|false"
M=q3-30b-a3b
SEP="asym_sepplan2_cpuadamwds|T2"
echo "=== STDTPS46-A1-Q30-2R-B BEGIN $(date '+%F %H:%M:%S') ===" >> "$S"
v=$(run_cell s1q30sep512 $M "$SEP" 512000 "2" "$POL" 2); echo "A1-Q30 sEP-T2@512k b2 up-probe -> $v" >> "$S"
echo "=== STDTPS46-A1-Q30-2R-B DONE $(date '+%F %H:%M:%S') ===" >> "$S"

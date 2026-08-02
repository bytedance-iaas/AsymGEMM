#!/bin/bash
# MERGE VALIDATION QUEUE (merge_progress.md Phase 3, 2026-08-01):
# near-capacity no-regression cells for the 39⟵{46,SFT} merged tree, serial GPU0.
# References (recorded):
#  V1 q3-30b T2 120k·b8   — CPU-matrix row7: 2726 tok/s · 176.7 GiB · RSS ~539 (c06); sched row7 new: 2762 · 165.7
#  V2 air   T3TOK 128k·b3 — model_integration: 176.4 GiB (96%) · RSS 892 · losses 1.355/1.427/1.406 (c14!)
#  V3 flash T3TOK 192k·b5 — model_integration: 158.8 GiB (86%) · RSS 723 · losses 1.322/1.226/1.235 (c14!)
#  V4 q3.5-35b T2 896k·b1 — fix_qwen3.5_tp §7: 600.4 s/it · 1492 tok/s · 175.6 GiB (95%) · RSS 469 (c18)
#  V5 q3-30b T2B 1.1M·b1  — CPU-matrix row9: 380 tok/s · 144.5 GiB · RSS ~906 (c06); sched row9 new: 385 · 152.9 · 906
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
T3TOK="asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0"
echo "MRG-VAL-QUEUE begin $(date +%H:%M)" >> "$S"

# V1: row7 KA-dial state via T2 preset (n1024 dataset; qwen: no jitter var needed but harmless)
MAX_SAMPLES=1024 run_cell mgv1r7 q3-30b-a3b "asym_cpuadamwds|T2" 120000 "8"
echo "MRG-V1-DONE $(date +%H:%M)" >> "$S"

# V2: Air T3 128k·b3 (near-capacity 96% cell; exact tpfig env: MAX_SAMPLES=512 jitter0 trc=false)
run_cell mgv2a45 glm4.5-air "$T3TOK" 128000 "3"
echo "MRG-V2-DONE $(date +%H:%M)" >> "$S"

# V3: Flash T3 192k·b5 (86% cell)
run_cell mgv3f47 glm4.7-flash "$T3TOK" 192000 "5"
echo "MRG-V3-DONE $(date +%H:%M)" >> "$S"

# V4: q3.5-35b T2 896k·b1 (95% cell on c18 record; n1024 dataset)
MAX_SAMPLES=1024 run_cell mgv4q35 q3.5-35b-a3b "asym_cpuadamwds|T2" 896000 "1"
echo "MRG-V4-DONE $(date +%H:%M)" >> "$S"

# V5: row9 shed 1.1M·b1 via T2B preset (host-RSS-bound capacity row; n512)
MAX_SAMPLES=512 run_cell mgv5r9 q3-30b-a3b "asym_cpuadamwds|T2B" 1100000 "1"
echo "MRG-V5-DONE $(date +%H:%M)" >> "$S"

echo "MRG-VAL-QUEUE-DONE ALL-CELLS-COMPLETE $(date +%H:%M)" >> "$S"

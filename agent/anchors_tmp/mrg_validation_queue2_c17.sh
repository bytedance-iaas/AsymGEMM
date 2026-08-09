#!/bin/bash
# MERGE VALIDATION QUEUE 2 — c17 lane (merge_progress.md Phase 3-ext, 2026-08-02).
# Runs the cross-node-band cells from the 38 checkout (code byte-identical to the
# merged 39 tree) on idle c17 while the c14 session drives the c14-referenced cells.
# X1 (clean-node replication) llama3.3-70b T2 192k·b2 — refs 548/545/543 tok/s · 171.1 · RSS 963-982 (c06/c12 band).
#   On c14 this cell COOMed twice (host watchdog, floor 50; min-avail ~6 GiB below cpum5 ref trajectory).
#   c17 is freshly idle: COOM here => real merged-tree host-RSS regression; TRAINED => c14 node residue.
# X7 q3.5-122b T2 448k·b1 — ref c18 §8: 520.2 s/it · 861 tok/s · 171.3 GiB (93%) · RSS 846 (band)
# X8 q3.5-122b T2 480k·b1 — ref c18 §8: 563.6 s/it · 852 tok/s · 177.8 GiB (96% EDGE) · RSS 849 (band)
set -uo pipefail
export GPU=0 HOSTFLOOR=1300
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
echo "MRG-VAL2-C17 begin $(date +%H:%M)" >> "$S"

MAX_SAMPLES=1024 run_cell x1c17 llama3.3-70b "asym_cpuadamwds|T2" 192000 "2"
echo "MRG-X1C17-DONE $(date +%H:%M)" >> "$S"
run_cell x7q122 q3.5-122b-a10b "asym_cpuadamwds|T2" 448000 "1"
echo "MRG-X7-DONE $(date +%H:%M)" >> "$S"
run_cell x8q122 q3.5-122b-a10b "asym_cpuadamwds|T2" 480000 "1"
echo "MRG-X8-DONE $(date +%H:%M)" >> "$S"
echo "MRG-VAL2-C17-DONE $(date +%H:%M)" >> "$S"

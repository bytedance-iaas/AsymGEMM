#!/bin/bash
# FINAL PATCH: sweeps every fit-less rung across A1/A2 after everything else.
# Self-selecting: for each (tag-prefix, blist) below, runs only if no TRAINED
# line exists for the prefix. Gated on GLMTP-ALL-DONE; emits FINALPATCH-DONE.
# Covers: A1 rc b1 gaps + all A2 host-shifted gaps (need() skips satisfied).
set -uo pipefail
export GPU="0,1" HOSTFLOOR=1200 GPU_POOL="0,1" TORCHINDUCTOR_COMPILE_THREADS=1 DDP_TIMEOUT=1500
export ASYM_ARENA_SHM_CAP_GB=240
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "GLMTP-ALL-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLMTP-ALL-DONE" "$S" || { echo "FINALPATCH-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "FINAL-PATCH begin $(date +%H:%M)" >> "$S"
need() { ! grep -Eq "CELL $1[^a-z0-9_].* -> TRAINED" "$S"; }
# --- A1 gaps (1-rank, GPU0-only is fine but keep both) ---
need a1rc96 && run_cell fp_a1rc96 glm4.5-air "superoffload_mem|recomp" 96000 "1" "none|false|false|false|false|false" 1
need a1rc48 && run_cell fp_a1rc48 glm4.5-air "superoffload_mem|recomp" 48000 "2 1" "none|false|false|false|false|false" 1
# --- A2 gaps (2-rank; b-lists one below the exhausted ones) ---
need a2b_un96 && run_cell fp_a2un96 glm4.5-air "superoffload_mem|unsloth" 96000 "1" "none|false|false|false|false|false" 2
need a2b_t196 && run_cell fp_a2t196 glm4.5-air "asym_sdp2_cpuadamwds|T1" 96000 "2 1" "none|false|false|false|false|false" 2
need a2b_rc64 && run_cell fp_a2rc64 glm4.5-air "superoffload_mem|recomp" 64000 "1" "none|false|false|false|false|false" 2
need a2b_un64 && run_cell fp_a2un64 glm4.5-air "superoffload_mem|unsloth" 64000 "2 1" "none|false|false|false|false|false" 2
need a2b_uo64 && run_cell fp_a2uo64 glm4.5-air "superoffload_mem|unsloth-off-ohbm0" 64000 "1" "none|false|false|false|false|false" 2
need a2b_t164 && run_cell fp_a2t164 glm4.5-air "asym_sdp2_cpuadamwds|T1" 64000 "3 2" "none|false|false|false|false|false" 2
need a2b_rc48 && run_cell fp_a2rc48 glm4.5-air "superoffload_mem|recomp" 48000 "1" "none|false|false|false|false|false" 2
need a2b_un48 && run_cell fp_a2un48 glm4.5-air "superoffload_mem|unsloth" 48000 "2 1" "none|false|false|false|false|false" 2
need a2b_uo48 && run_cell fp_a2uo48 glm4.5-air "superoffload_mem|unsloth-off-ohbm0" 48000 "1" "none|false|false|false|false|false" 2
need a2b_t148 && run_cell fp_a2t148 glm4.5-air "asym_sdp2_cpuadamwds|T1" 48000 "4 2" "none|false|false|false|false|false" 2
need a2b_rc32 && run_cell fp_a2rc32 glm4.5-air "superoffload_mem|recomp" 32000 "2 1" "none|false|false|false|false|false" 2
need a2b_un32 && run_cell fp_a2un32 glm4.5-air "superoffload_mem|unsloth" 32000 "4 2" "none|false|false|false|false|false" 2
need a2b_uo32 && run_cell fp_a2uo32 glm4.5-air "superoffload_mem|unsloth-off-ohbm0" 32000 "2" "none|false|false|false|false|false" 2
need a2b_t132 && run_cell fp_a2t132 glm4.5-air "asym_sdp2_cpuadamwds|T1" 32000 "6 4" "none|false|false|false|false|false" 2
need a2b_rc16 && run_cell fp_a2rc16 glm4.5-air "superoffload_mem|recomp" 16000 "6 4" "none|false|false|false|false|false" 2
need a2b_un16 && run_cell fp_a2un16 glm4.5-air "superoffload_mem|unsloth" 16000 "8 6" "none|false|false|false|false|false" 2
need a2b_uo16 && run_cell fp_a2uo16 glm4.5-air "superoffload_mem|unsloth-off-ohbm0" 16000 "8 4" "none|false|false|false|false|false" 2
need a2b_t116 && run_cell fp_a2t116 glm4.5-air "asym_sdp2_cpuadamwds|T1" 16000 "12 8" "none|false|false|false|false|false" 2
echo "FINALPATCH-DONE EVERYTHING-MEASURED $(date +%H:%M)" >> "$S"

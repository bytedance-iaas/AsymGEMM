#!/bin/bash
# A1 patch: b1 cells for rungs where the fat Air baselines exhausted their
# batch lists (blists assumed b2 fits; Air rc/uns need b1 from 96k down to
# possibly 64k). Gated on GLMTP-ALL-DONE (A2 shares the a_serial process
# with A1 — no safe slot before it; panel data is order-independent).
set -uo pipefail
export GPU=0 HOSTFLOOR=1300 TORCHINDUCTOR_COMPILE_THREADS=1
. /workspace/AsymGEMM-SFT-39/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib.sh
for i in $(seq 1 2880); do grep -q "GLMTP-ALL-DONE" "$S" 2>/dev/null && break; sleep 30; done
grep -q "GLMTP-ALL-DONE" "$S" || { echo "A1PATCH-ABORT $(date +%H:%M)" >> "$S"; exit 1; }
echo "A1-PATCH begin $(date +%H:%M)" >> "$S"
# run only cells whose rung had no fit: check status log for a TRAINED line per tag prefix
need() { ! grep -q "CELL $1.* -> TRAINED" "$S"; }
need a1rc96 && run_cell a1p_rc96 glm4.5-air "superoffload_mem|recomp"  96000 "1"
need a1un96 && run_cell a1p_un96 glm4.5-air "superoffload_mem|unsloth" 96000 "1"
need a1rc64 && run_cell a1p_rc64 glm4.5-air "superoffload_mem|recomp"  64000 "1"
need a1un64 && run_cell a1p_un64 glm4.5-air "superoffload_mem|unsloth" 64000 "1"
need a1rc48 && run_cell a1p_rc48 glm4.5-air "superoffload_mem|recomp"  48000 "2 1"
need a1un48 && run_cell a1p_un48 glm4.5-air "superoffload_mem|unsloth" 48000 "2 1"
need a1rc32 && run_cell a1p_rc32 glm4.5-air "superoffload_mem|recomp"  32000 "2 1"
need a1un32 && run_cell a1p_un32 glm4.5-air "superoffload_mem|unsloth" 32000 "2 1"
need a1rc16 && run_cell a1p_rc16 glm4.5-air "superoffload_mem|recomp"  16000 "4 2"
need a1un16 && run_cell a1p_un16 glm4.5-air "superoffload_mem|unsloth" 16000 "4 2"
echo "A1-PATCH-DONE $(date +%H:%M)" >> "$S"

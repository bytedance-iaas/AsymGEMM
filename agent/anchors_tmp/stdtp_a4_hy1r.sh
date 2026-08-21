#!/bin/bash
# OBSOLETE (2026-08-20 17:1x) — never run / superseded: Agent 4 done by Session B (c11); Flash-1r asym cells run by B; Air 384k block ran inside stdtp_a3_air.sh. Kept as campaign record (STDTP_LOG.md).
# stdtp_a4_hy1r.sh — Agent 4 / phase 1 (standardize_tps.md): Hunyuan-A13B
# 1-rank missing rungs 160k/224k/288k. Walls (HY_CAMPAIGN): rc (192k,256k] ->
# 224k probe; uns (256k,320k] -> 288k probe (the doc's sole-rung decider);
# uo fits thru 512k; asym T1 carries thru 320k. Batch seeds from nearest
# banked rungs (128k: asy b2, rc/un/uo b1; 192k: all b1).
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU=0 HOSTFLOOR=1100
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtp_lib.sh
export GPU_POOL=0; unset DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
python3 /workspace/AsymGEMM-SFT-38/.repair_dataset_info.py >> "$S" 2>&1 || echo "repair-dataset-info FAILED" >> "$S"
echo "== STDTP-A4-HY1R begin (1r, phys GPU 3 -> inside 0) $(date '+%m-%d %H:%M')" >> "$S"
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"; AS="asym_cpuadamwds|T1"
P="none|false|false|false|false|false"

run_cell h1rc160 hunyuan-a13b "$RC" 160000 "1"   "$P" 1 >/dev/null
run_cell h1un160 hunyuan-a13b "$UN" 160000 "1"   "$P" 1 >/dev/null
run_cell h1uo160 hunyuan-a13b "$UO" 160000 "1"   "$P" 1 >/dev/null
run_cell h1as160 hunyuan-a13b "$AS" 160000 "2 1" "$P" 1 >/dev/null

run_cell h1rc224 hunyuan-a13b "$RC" 224000 "1"   "$P" 1 >/dev/null   # wall-bracket probe
run_cell h1un224 hunyuan-a13b "$UN" 224000 "1"   "$P" 1 >/dev/null
run_cell h1uo224 hunyuan-a13b "$UO" 224000 "1"   "$P" 1 >/dev/null
run_cell h1as224 hunyuan-a13b "$AS" 224000 "1"   "$P" 1 >/dev/null

run_cell h1un288 hunyuan-a13b "$UN" 288000 "1"   "$P" 1 >/dev/null   # THE doc probe (sole-rung decider)
run_cell h1uo288 hunyuan-a13b "$UO" 288000 "1"   "$P" 1 >/dev/null
run_cell h1as288 hunyuan-a13b "$AS" 288000 "1"   "$P" 1 >/dev/null
echo "== STDTP-A4-HY1R-DONE $(date '+%m-%d %H:%M')" >> "$S"

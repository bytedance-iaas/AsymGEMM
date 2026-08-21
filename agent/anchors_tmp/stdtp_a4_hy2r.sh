#!/bin/bash
# OBSOLETE (2026-08-20 17:1x) — never run / superseded: Agent 4 done by Session B (c11); Flash-1r asym cells run by B; Air 384k block ran inside stdtp_a3_air.sh. Kept as campaign record (STDTP_LOG.md).
# stdtp_a4_hy2r.sh — Agent 4 / phase 2: Hunyuan-A13B 2-rank missing rungs
# 160k/224k. Walls (2r): uo (128k,192k] C -> 160k probe (192k OOM measured ->
# 224k monotone); rc (192k,256k] -> 224k probe; uns fits 256k. asym sdp2-T1
# (default arena; cap 320 is a T2B-only need). Seeds: 128k asy b2 -> 160k "2 1".
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=1100
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtp_lib.sh
export GPU_POOL="0,1"; unset DDP_TIMEOUT ASYM_ARENA_SHM_CAP_GB || true
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
python3 /workspace/AsymGEMM-SFT-38/.repair_dataset_info.py >> "$S" 2>&1 || echo "repair-dataset-info FAILED" >> "$S"
echo "== STDTP-A4-HY2R begin (2r, phys GPUs 1,3 -> inside 0,1) $(date '+%m-%d %H:%M')" >> "$S"
RC="superoffload_mem|recomp"; UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"; AS="asym_sdp2_cpuadamwds|T1"
P="none|false|false|false|false|false"

run_cell h2rc160 hunyuan-a13b "$RC" 160000 "1"   "$P" 2 >/dev/null
run_cell h2un160 hunyuan-a13b "$UN" 160000 "1"   "$P" 2 >/dev/null
run_cell h2uo160 hunyuan-a13b "$UO" 160000 "1"   "$P" 2 >/dev/null   # wall-bracket probe
run_cell h2as160 hunyuan-a13b "$AS" 160000 "2 1" "$P" 2 >/dev/null

run_cell h2rc224 hunyuan-a13b "$RC" 224000 "1"   "$P" 2 >/dev/null   # wall-bracket probe
run_cell h2un224 hunyuan-a13b "$UN" 224000 "1"   "$P" 2 >/dev/null
run_cell h2as224 hunyuan-a13b "$AS" 224000 "1"   "$P" 2 >/dev/null
echo "== STDTP-A4-HY2R-DONE $(date '+%m-%d %H:%M')" >> "$S"

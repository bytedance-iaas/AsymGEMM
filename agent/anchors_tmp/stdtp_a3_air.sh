#!/bin/bash
# stdtp_a3_air.sh — Agent 3 / phase 1 (standardize_tps.md): GLM-4.5-Air 2-rank
# cap gate. Grid needs 384k+448k asym-2r cells (every baseline is beyond its
# measured wall there -> OOM by monotonicity, no runs).
# Gate order: T2@448k first (2r-T1@320k banked at 98% HBM -> T1@448k is
# physically excluded; its OOM label comes by monotonicity from the T1@384k
# wall-pin probe below). Tier ladder walks ONLY on GOOM/COOM; infra FAIL
# aborts the chain. 384k block runs in every grid outcome.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=1200
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtp_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=450
# cap 450: fg-family tiers (T2/T2B/T3) put the weight bank AND the fg bases
# into the ONE shm fabric (HY_CAMPAIGN precedent: "bank + fg bases share ONE
# shm fabric") — Air: 199.3G bank + ~199.3G fg + ~14G other ≈ 410-427G.
# July's banked Air-2r cells were all T1 (plain bank ~214G -> cap 240 ok);
# caps 240/320 died at inserts 00443/00501 en route (02:56, 03:07).
# /dev/shm = 479G total, so 450 is the practical ceiling.
python3 /workspace/AsymGEMM-SFT-38/.repair_dataset_info.py >> "$S" 2>&1 || echo "repair-dataset-info FAILED" >> "$S"
echo "== STDTP-A3-AIR begin v2 (2r, phys GPUs 1,3 -> inside 0,1) $(date '+%m-%d %H:%M')" >> "$S"
BE=asym_sdp2_cpuadamwds
P="none|false|false|false|false|false"
die() { echo "ABORT-A3-AIR: $1 $(date '+%m-%d %H:%M')" >> "$S"; exit 1; }

v448=$(run_cell a2t2448 glm4.5-air "${BE}|T2" 448000 "1" "$P" 2)
if [ "$v448" != "TRAINED" ]; then
  ooms "$v448" || die "a2t2448 infra verdict $v448"
  v448=$(run_cell a2t2b448 glm4.5-air "${BE}|T2B" 448000 "1" "$P" 2)
  if [ "$v448" != "TRAINED" ]; then
    ooms "$v448" || die "a2t2b448 infra verdict $v448"
    v448=$(run_cell a2t3448 glm4.5-air "${BE}|T3" 448000 "1" "$P" 2)
    [ "$v448" = "TRAINED" ] || ooms "$v448" || die "a2t3448 infra verdict $v448"
  fi
fi
echo "GATE-448K verdict: $v448 $(date +%H:%M)" >> "$S"

v384t1=$(run_cell a2t1384 glm4.5-air "${BE}|T1" 384000 "1" "$P" 2)
if [ "$v384t1" = "TRAINED" ]; then
  run_cell a2t1448 glm4.5-air "${BE}|T1" 448000 "1" "$P" 2 >/dev/null
elif ooms "$v384t1"; then
  v384=$(run_cell a2t2384 glm4.5-air "${BE}|T2" 384000 "1" "$P" 2)
  if [ "$v384" != "TRAINED" ]; then
    ooms "$v384" || die "a2t2384 infra verdict $v384"
    v384=$(run_cell a2t2b384 glm4.5-air "${BE}|T2B" 384000 "1" "$P" 2)
    if [ "$v384" != "TRAINED" ]; then
      ooms "$v384" || die "a2t2b384 infra verdict $v384"
      run_cell a2t3384 glm4.5-air "${BE}|T3" 384000 "1" "$P" 2 >/dev/null
    fi
  fi
else
  die "a2t1384 infra verdict $v384t1"
fi
echo "== STDTP-A3-AIR-DONE $(date '+%m-%d %H:%M')" >> "$S"

#!/bin/bash
# OBSOLETE (2026-08-20 17:1x) — never run / superseded: Agent 4 done by Session B (c11); Flash-1r asym cells run by B; Air 384k block ran inside stdtp_a3_air.sh. Kept as campaign record (STDTP_LOG.md).
# stdtp_a3_air384.sh — Agent 3 / Air gate CONTINUATION: the 384k block alone
# (used when the 448k ladder ends in a non-TRAINED terminal state — incl. the
# expected a2t3448 FAIL: moe|T3 = ker101 is config-rejected for GLM families,
# so GLM's ladder effectively ends at T2B; record, don't abort, per
# HY_CAMPAIGN "config-rejected ... recorded, not a wall").
# T1@384k pins the 2r T1 wall (banked T1@320k = 98%); then T2 -> T2B.
set -uo pipefail
export TORCHINDUCTOR_COMPILE_THREADS=1
export GPU="0,1" HOSTFLOOR=1200
. /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/stdtp_lib.sh
export GPU_POOL="0,1" DDP_TIMEOUT=1500 ASYM_ARENA_SHM_CAP_GB=450
echo "== STDTP-A3-AIR384 begin (continuation) $(date '+%m-%d %H:%M')" >> "$S"
BE=asym_sdp2_cpuadamwds
P="none|false|false|false|false|false"
die() { echo "ABORT-A3-AIR384: $1 $(date '+%m-%d %H:%M')" >> "$S"; exit 1; }

v384t1=$(run_cell b2t1384 glm4.5-air "${BE}|T1" 384000 "1" "$P" 2)
if [ "$v384t1" = "TRAINED" ]; then
  run_cell b2t1448 glm4.5-air "${BE}|T1" 448000 "1" "$P" 2 >/dev/null
elif ooms "$v384t1"; then
  v384=$(run_cell b2t2384 glm4.5-air "${BE}|T2" 384000 "1" "$P" 2)
  if [ "$v384" != "TRAINED" ]; then
    ooms "$v384" || die "b2t2384 infra verdict $v384"
    v384=$(run_cell b2t2b384 glm4.5-air "${BE}|T2B" 384000 "1" "$P" 2)
    [ "$v384" = "TRAINED" ] || ooms "$v384" || die "b2t2b384 infra verdict $v384"
  fi
  echo "AIR-384K terminal: $v384 $(date +%H:%M)" >> "$S"
else
  die "b2t1384 infra verdict $v384t1"
fi
echo "== STDTP-A3-AIR384-DONE $(date '+%m-%d %H:%M')" >> "$S"

#!/bin/bash
# Hunyuan-A13B throughput campaign — phase 0: weights, phase 1: SMOKE GATE
# (asym-T1 + SO-recomp at 16k, 1 rank; abort everything on FAIL), phase 2:
# 1-rank ladders (recomp/uns/uns-off/asym-T1), phase 3: 2-rank ladders.
# House protocol via tpfig_lib_c17.sh. Status -> hy_status.log.
set -uo pipefail
export GPU="${GPU:-0}" HOSTFLOOR="${HOSTFLOOR:-1100}"
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_status.log"
POL="none|false|false|false|false|false"
# hunyuan ties embed/lm_head (one storage); the asym offload stage rejects the
# tied pair, so asym cells run all-minus-embeddings (~1.05 GB stays HBM-resident
# — disclosed in the DATA comment). SO backends ignore this env.
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
# (router also excluded: HunYuanMoEV1Gate is a wrapper module, kept intact on GPU
#  like the glm DS-bias gates — ~0.5 MB/layer)
ASY1="asym_cpuadamwds|T1"        # 1-rank asym backend
ASY2="asym_sdp2_cpuadamwds|T1"   # 2-rank (sdp2 requires |2)
RC="superoffload_mem|recomp"
UN="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"

cell() { # $1 tag $2 systok $3 seq $4 blist $5 ranks
  local v
  v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "$4" "$POL" "$5")
  echo "HY CELL $1 r$5 ${2%%|*} s=$3 -> $v $(date +%H:%M:%S)" >> "$S2"
  echo "$v"
}

# ── phase 0: weights ──
echo "HY PHASE0 dl $(date +%H:%M:%S)" >> "$S2"
/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python - <<'EOF' >> /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_dl.log 2>&1
from huggingface_hub import snapshot_download
print("DL", snapshot_download("tencent/Hunyuan-A13B-Instruct", max_workers=8), flush=True)
EOF
echo "HY PHASE0 done rc=$? $(date +%H:%M:%S)" >> "$S2"

# ── phase 1: smoke gate (1 rank, 16k, b1) ──
export GPU=0 CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
v1=$(cell hysmk3_asy "$ASY1" 16000 "1" 1)
v2=$(cell hysmk_rc "$RC" 16000 "1" 1)
if [ "$v1" != "TRAINED" ] || [ "$v2" != "TRAINED" ]; then
  echo "HY SMOKE GATE FAILED (asym=$v1 rc=$v2) — ladders aborted" >> "$S2"
  exit 1
fi
echo "HY SMOKE GATE PASSED $(date +%H:%M:%S)" >> "$S2"

run_ladder() { # $1 rankcount (1|2)  — walks all four systems over the rungs
  local r="$1" pfx sys tag v
  if [ "$r" = "2" ]; then export GPU="0,1" CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"; else export GPU=0 CUDA_VISIBLE_DEVICES=0 GPU_POOL=0; fi
  declare -A DEAD=()
  for s in 32000 64000 128000 192000 256000 320000 384000 448000; do
    case $s in
      32000)  BA="8 4 2"; BB="4 2";;
      64000)  BA="4 2";   BB="2 1";;
      128000) BA="2 1";   BB="1";;
      *)      BA="1";     BB="1";;
    esac
    for sys in RC UN UO ASY; do
      [ "${DEAD[$sys]:-0}" = "1" ] && continue
      local tok bl
      case $sys in
        RC) tok="$RC"; bl="$BB";;
        UN) tok="$UN"; bl="$BB";;
        UO) tok="$UO"; bl="$BB";;
        ASY) if [ "$r" = "2" ]; then tok="$ASY2"; else tok="$ASY1"; fi; bl="$BA";;
      esac
      pfx=$(echo "${sys}" | tr 'A-Z' 'a-z')
      tag="hy${r}r_${pfx}$((s/1000))"
      v=$(cell "$tag" "$tok" "$s" "$bl" "$r")
      if [ "$v" != "TRAINED" ]; then
        DEAD[$sys]=1
        echo "HY r$r ${sys} wall at $s ($v)" >> "$S2"
      fi
    done
    # stop the whole ladder once asym has walled (nothing left standing)
    [ "${DEAD[ASY]:-0}" = "1" ] && break
  done
}

# ── phase 2: 1-rank ladders ──
echo "HY PHASE2 1-rank ladders $(date +%H:%M:%S)" >> "$S2"
run_ladder 1
echo "HY PHASE2 done $(date +%H:%M:%S)" >> "$S2"

# ── phase 3: 2-rank ladders ──
echo "HY PHASE3 2-rank ladders $(date +%H:%M:%S)" >> "$S2"
run_ladder 2
echo "HY PHASE3 done $(date +%H:%M:%S)" >> "$S2"
echo "HY_ALL_DONE $(date +%H:%M:%S)" >> "$S2"

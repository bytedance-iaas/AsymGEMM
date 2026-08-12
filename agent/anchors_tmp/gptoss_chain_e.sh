#!/bin/bash
# gpt-oss-20b campaign — CHAIN E: 2-rank tp ladder on the SEPPLANLINK backend
# (asym_sepplanlink2_cpuadamwds, fix_dynamic_ep port). Gate: ep_sep_probe
# host+nvlink (mode plan) PR5_PASS on GPUs 0,1. Then per rung rc->un->uo->
# sepplanlink T1 (promote T2B -> T3). /dev/shm/asym_fabric_* cleaned before
# every 2r cell; ep_sep exit stats harvested per asym cell (own-engine note:
# experts don't dispatch grouped kernels -> armed=0 expected; rings+DP stack
# still the measured system).
set -uo pipefail
export GPU="${GPU:-0,1}" HOSTFLOOR="${HOSTFLOOR:-600}"
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
S2="$LOGD/gptoss_status.log"
POL="none|false|false|false|false|false"
MODEL=gpt-oss-20b
export CUDA_VISIBLE_DEVICES=0,1 GPU_POOL="0,1" DDP_TIMEOUT=1500
export ASYM_OFFLOAD_MODULES=all
RC="superoffload_mem|recomp"
UNS="superoffload_mem|unsloth-ohbm0"
UO="superoffload_mem|unsloth-off-ohbm0"
SEP="asym_sepplanlink2_cpuadamwds"
T3TOK="${SEP}|recomp-off-full-fg-ker000-ceil0000-ohbm0"
T3ENV=(ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1
       ASYMM_QWEN3_MOE_DOWN_SCATTER_BLOCK_EXPERTS=0 ASYMM_QWEN3_MOE_FG_DA_GPU=1
       ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1 ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1
       ASYM_CPU_OPS_THREADS=48 ASYM_PLACEMENT_POLICY=1)
note() { echo "[$(date +%H:%M:%S)] $*" >> "$S2"; }
harv() { /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python \
           agent/anchors_tmp/gptoss_harvest.py "$1" "${2:-2}" 2>/dev/null | tee -a "$S2"; }
shmclean() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . || rm -f /dev/shm/asym_fabric_* /dev/shm/asym_sep_ipc_rank* 2>/dev/null; }

note "CHAIN-E begin (2r sepplanlink ladder)"
# ── probe gate: ported sepplanlink code must pass PR5 on THIS node ──────────
for tr in host nvlink; do
  out=$(/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/.venv/bin/python \
        scripts/testing/ep_sep_probe.py --mode plan --transport "$tr" --gpus 0,1 2>&1 | tail -4)
  note "PROBE $tr: $(echo "$out" | tr '\n' ' | ')"
  echo "$out" | grep -q "PR5_PASS" || { note "PROBE $tr FAILED — ABORT CHAIN-E"; exit 1; }
done
note "PROBE GATE PASSED (host+nvlink plan PR5)"

dead_rc=0; dead_un=0; dead_uo=0
asym_tier=T1

blist_for() { local s=$1
  if   [ "$s" -le 64000 ];  then echo "8 6 4 2 1"
  elif [ "$s" -le 128000 ]; then echo "6 4 2 1"
  elif [ "$s" -le 256000 ]; then echo "2 1"
  else echo "1"; fi; }

run_sys() { local v; shmclean
  v=$(run_cell "$1" "$MODEL" "$2" "$3" "$4" "$POL" 2)
  note "CELL $1 ${2%%|*} s=$3 r2 -> $v"; harv "$1" 2 >/dev/null; echo "$v"; }
run_t3() { local v; shmclean
  v=$( (export "${T3ENV[@]}"; run_cell "$1" "$MODEL" "$T3TOK" "$3" "$4" "$POL" 2) )
  note "CELL $1 SEP-T3 s=$3 r2 -> $v"; harv "$1" 2 >/dev/null; echo "$v"; }

for seq in 32000 64000 128000 192000 256000 320000 384000 512000 640000 768000 896000 1024000; do
  sk=$((seq/1000)); bl=$(blist_for "$seq")
  note "RUNG2 ${sk}k begin (asym_tier=$asym_tier)"
  if [ "$dead_rc" = 0 ]; then
    v=$(run_sys "e2rc${sk}" "$RC" "$seq" "$bl"); [ "$v" = "TRAINED" ] || dead_rc=1
  fi
  if [ "$dead_un" = 0 ]; then
    v=$(run_sys "e2un${sk}" "$UNS" "$seq" "$bl"); [ "$v" = "TRAINED" ] || dead_un=1
  fi
  if [ "$dead_uo" = 0 ]; then
    v=$(run_sys "e2uo${sk}" "$UO" "$seq" "$bl"); [ "$v" = "TRAINED" ] || dead_uo=1
  fi
  va=FAIL
  if [ "$asym_tier" = "T1" ]; then
    va=$(run_sys "e2t1${sk}" "${SEP}|T1" "$seq" "$bl")
    [ "$va" = "TRAINED" ] || { asym_tier=T2; note "PROMOTE 2r T1->T2 at ${sk}k"; }
  fi
  if [ "$va" != "TRAINED" ] && [ "$asym_tier" = "T2" ]; then
    va=$(run_sys "e2t2${sk}" "${SEP}|T2" "$seq" "$bl")
    [ "$va" = "TRAINED" ] || { asym_tier=T2B; note "PROMOTE 2r T2->T2B at ${sk}k"; }
  fi
  if [ "$va" != "TRAINED" ] && [ "$asym_tier" = "T2B" ]; then
    va=$(run_sys "e2a2b${sk}" "${SEP}|T2B" "$seq" "$bl")
    [ "$va" = "TRAINED" ] || { asym_tier=T3; note "PROMOTE 2r T2B->T3 at ${sk}k"; }
  fi
  if [ "$va" != "TRAINED" ] && [ "$asym_tier" = "T3" ]; then
    va=$(run_t3 "e2t3${sk}" x "$seq" "$bl")
    if [ "$va" != "TRAINED" ]; then
      note "2R ASYM WALL at ${sk}k — CEILING BRACKETED, ladder ends"
      break
    fi
  fi
  note "RUNG2 ${sk}k done"
done
note "CHAIN-E COMPLETE"

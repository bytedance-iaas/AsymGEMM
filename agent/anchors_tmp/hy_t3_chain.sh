#!/bin/bash
# HUNYUAN T3 enablement chain — waits for the FA4 ladders to release the GPUs.
# Gate 0: numeric probe at hunyuan shapes (fg101 vs fp32 reference, fwd+grads).
# Gate 1: real-model correctness pair — T3 vs T2B @16k b1 same seed (loss
#         parity + route counters >0 in T3, ==0 in T2B).
# Gate 2: throughput A/B @384k 1r (T2B banked 589).
# Gate 3: ceiling — 1r 544k/576k/608k T3; 2r arena320 T3 320k/384k.
set -uo pipefail
S="/workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/hy_t3_status.log"
until grep -q 'FA4LADDERS_DONE' /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/nemo_fa4_probe_status.log 2>/dev/null; do sleep 120; done

export GPU=0 HOSTFLOOR=1100
source /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM/agent/anchors_tmp/tpfig_lib_c17.sh
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
export CUDA_VISIBLE_DEVICES=0 GPU_POOL=0
export ASYM_OFFLOAD_MODULES="routed_experts,shared_experts,attention,norms,mlp_dense"
POL="none|false|false|false|false|false"

echo "T3 GATE0 numeric probe $(date +%H:%M:%S)" >> "$S"
.venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --hunyuan --tokens 4096 \
  > agent/anchors_tmp/hy_t3_numeric.log 2>&1
rc=$?
.venv/bin/python scripts/testing/qwen35_fg_numeric_probe.py --hunyuan --tokens 4096 --zero-b \
  >> agent/anchors_tmp/hy_t3_numeric.log 2>&1
rc2=$?
echo "T3 GATE0 rc=${rc}/${rc2} $(tail -1 agent/anchors_tmp/hy_t3_numeric.log | head -c 120)" >> "$S"
if [ $rc -ne 0 ] || [ $rc2 -ne 0 ]; then echo "T3 GATE0 FAILED — aborting" >> "$S"; exit 1; fi

cell() { # $1 tag $2 systok $3 seq $4 ranks
  local gpus="0"; [ "$4" = "2" ] && { gpus="0,1"; export GPU="0,1" CUDA_VISIBLE_DEVICES="0,1" GPU_POOL="0,1"; } || { export GPU=0 CUDA_VISIBLE_DEVICES=0 GPU_POOL=0; }
  local v
  v=$(run_cell "$1" hunyuan-a13b "$2" "$3" "1" "$POL" "$4")
  echo "T3 CELL $1 r$4 s=$3 -> $v $(date +%H:%M:%S)" >> "$S"
  echo "$v"
}

# Gate 1: correctness pair @16k (fresh tags; T2B rerun for same-day A/B loss)
echo "T3 GATE1 correctness pair $(date +%H:%M:%S)" >> "$S"
cell t3cor_t3 "asym_cpuadamwds|T3" 16000 1
cell t3cor_t2b "asym_cpuadamwds|T2B" 16000 1

# Gate 2: throughput A/B @384k 1r
echo "T3 GATE2 tput 384k $(date +%H:%M:%S)" >> "$S"
cell t3tp384 "asym_cpuadamwds|T3" 384000 1

# Gate 3a: 1r ceiling — climb past the T2B wall
for s in 512000 544000 576000 608000; do
  v=$(cell "t3c$((s/1000))" "asym_cpuadamwds|T3" "$s" 1)
  [ "$v" = "TRAINED" ] || { echo "T3 r1 wall at $s ($v)" >> "$S"; break; }
done

# Gate 3b: 2r ceiling (arena for the shared bank+fg)
export ASYM_ARENA_SHM_CAP_GB=320
rm -rf /dev/shm/asym_fabric_* 2>/dev/null || true
for s in 320000 384000; do
  v=$(cell "t3c2r$((s/1000))" "asym_sdp2_cpuadamwds|T3" "$s" 2)
  [ "$v" = "TRAINED" ] || { echo "T3 r2 wall at $s ($v)" >> "$S"; break; }
  rm -rf /dev/shm/asym_fabric_* 2>/dev/null || true
done
echo "HY_T3_DONE $(date +%H:%M:%S)" >> "$S"

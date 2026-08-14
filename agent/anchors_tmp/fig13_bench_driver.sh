#!/bin/bash
# Fig13 wall measurement driver — runs inside asym_sft_39 (via skew_in39.sh).
#   usage: fig13_bench_driver.sh <gpupair 0,1|2,3> <tag> <model...>
# Per model x {math,code,science}: placed(owned,plan) + oracle(owned) +
# contig(owned), all MoE layers, scope=experts, natural counts.
set -uo pipefail
PAIR="$1"; TAG="$2"; shift 2
H=profiling_results/ep_skew_deep/fig13
PY=.venv/bin/python

geom_of()  { case "$1" in qwen3-30b) echo "128,768,2048";; qwen3.5-122b) echo "256,1024,3072";; glm4.7-flash) echo "64,1536,2048";; esac; }
topk_of()  { case "$1" in glm4.7-flash) echo 4;; *) echo 8;; esac; }
shn_of()   { case "$1" in qwen3-30b) echo 0;; qwen3.5-122b) echo 1024;; glm4.7-flash) echo 1536;; esac; }
mtot_of()  { case "$1" in qwen3-30b) echo 6144000;; *) echo 5120000;; esac; }
nlay_of()  { case "$1" in glm4.7-flash) echo 46;; *) echo 48;; esac; }

for MODEL in "$@"; do
  GEOM=$(geom_of "$MODEL"); TOPK=$(topk_of "$MODEL"); SHN=$(shn_of "$MODEL")
  MTOT=$(mtot_of "$MODEL"); NL=$(nlay_of "$MODEL")
  LAYERS=$($PY - <<EOF
print(','.join(f'L{l:02d}' for l in range($NL)))
EOF
)
  for SET in math code science; do
    for VAR in placed oracle contig; do
      OUT=$H/walls_${MODEL}_${SET}_${VAR}.json
      [[ -s "$OUT" ]] && { echo "[skip] $OUT exists"; continue; }
      MODES=owned; [[ "$VAR" == placed ]] && MODES=owned,plan
      rm -f /dev/shm/asym_epbench_${TAG}*
      echo "[run] $MODEL $SET $VAR ($MODES)"
      $PY scripts/testing/ep_balance_bench.py \
        --hist $H/fig13_${MODEL}_${SET}_${VAR}.json \
        --layers "$LAYERS" --alphas natural --modes $MODES \
        --scope experts --geom "$GEOM" --topk "$TOPK" --shared-n "$SHN" \
        --m-total "$MTOT" --reps 3 --gpus "$PAIR" --tag "$TAG" \
        --out "$OUT" | tail -1
      rm -f /dev/shm/asym_epbench_${TAG}*
    done
  done
done
echo "DRIVER_DONE_$TAG"

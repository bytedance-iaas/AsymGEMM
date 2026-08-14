#!/bin/bash
# Walls for the share-vs-length curve: math-domain placed hist (owned,plan)
# at each length's true b1 launch size. Runs inside asym_sft_39 after the
# anchor cells free the GPUs.
set -uo pipefail
H=profiling_results/ep_skew_deep/fig13
PY=.venv/bin/python

geom_of() { case "$1" in qwen3-30b) echo "128,768,2048";; qwen3.5-122b) echo "256,1024,3072";; glm4.7-flash) echo "64,1536,2048";; esac; }
topk_of() { case "$1" in glm4.7-flash) echo 4;; *) echo 8;; esac; }
shn_of()  { case "$1" in qwen3-30b) echo 0;; qwen3.5-122b) echo 1024;; glm4.7-flash) echo 1536;; esac; }
nlay_of() { case "$1" in glm4.7-flash) echo 46;; *) echo 48;; esac; }

for MODEL in qwen3-30b qwen3.5-122b glm4.7-flash; do
  GEOM=$(geom_of "$MODEL"); TOPK=$(topk_of "$MODEL"); SHN=$(shn_of "$MODEL"); NL=$(nlay_of "$MODEL")
  LAYERS=$($PY - <<EOF
print(','.join(f'L{l:02d}' for l in range($NL)))
EOF
)
  for SEQ in 48000 64000 80000 100000; do
    MTOT=$((2*SEQ*TOPK))
    OUT=$H/wallsS${SEQ}_${MODEL}_math_placed.json
    [[ -s "$OUT" ]] && { echo "[skip] $OUT"; continue; }
    echo "[run] $MODEL s=$SEQ m=$MTOT"
    $PY scripts/testing/ep_balance_bench.py \
      --hist $H/fig13_${MODEL}_math_placed.json \
      --layers "$LAYERS" --alphas natural --modes owned,plan \
      --scope experts --geom "$GEOM" --topk "$TOPK" --shared-n "$SHN" \
      --m-total "$MTOT" --reps 3 --gpus 0,1 --tag "ls${SEQ}" \
      --out "$OUT" | tail -1
    rm -f /dev/shm/asym_epbench_* 2>/dev/null
  done
done
echo "LEN_WALLS_DONE"

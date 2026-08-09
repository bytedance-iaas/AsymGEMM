#!/bin/bash
# NeMo baseline — additional MoEs: q3.5-122b-a10b, glm4.7-flash, glm4.5-air
# (+ mixtral bridge probe). Downloads first, then strictly serial 2-rank
# chains, then 1-rank cells. Verdicts -> nemo_moes_status.log.
set -uo pipefail
export XDG_CACHE_HOME=/scratch_local/user_data/shutian/kevin/cache
export HF_HOME=/scratch_local/user_data/shutian/kevin/cache/huggingface
cd /workspace/AsymGEMM-SFT-38/third_party/AsymGEMM
S="agent/anchors_tmp/nemo_moes_status.log"
PY=.venv-nemo/bin/python

echo "PHASE downloads $(date +%H:%M:%S)" >> "$S"
"$PY" - <<'EOF' >> "agent/anchors_tmp/nemo_moes_dl.log" 2>&1
from huggingface_hub import snapshot_download
for repo in ["zai-org/GLM-4.5-Air", "zai-org/GLM-4.7-Flash"]:
    print("DL", repo, snapshot_download(repo, max_workers=8), flush=True)
print("CONFIG-ONLY mixtral:",
      snapshot_download("mistralai/Mixtral-8x22B-v0.1", allow_patterns=["*.json", "tokenizer*"]), flush=True)
print("ALL_DL_DONE", flush=True)
EOF
echo "PHASE downloads done rc=$? $(date +%H:%M:%S)" >> "$S"

# Mixtral bridge probe (CPU only): expected "no registered bridge".
"$PY" - <<'EOF' >> "agent/anchors_tmp/nemo_mixtral_probe.log" 2>&1
from huggingface_hub import snapshot_download
p = snapshot_download("mistralai/Mixtral-8x22B-v0.1", allow_patterns=["*.json"], local_files_only=True)
import torch
from megatron.bridge import AutoBridge
try:
    AutoBridge.from_hf_pretrained(p, trust_remote_code=False)
    print("MIXTRAL_BRIDGE_OK (unexpected)", flush=True)
except Exception as exc:
    print("MIXTRAL_BRIDGE_UNSUPPORTED:", type(exc).__name__, str(exc)[:600], flush=True)
EOF
grep -o 'MIXTRAL_BRIDGE_[A-Z_]*' agent/anchors_tmp/nemo_mixtral_probe.log | tail -1 >> "$S"

run_cell() { # $1 tag $2 model $3 ranks $4 arm $5 seq
  local gpus="0"; [ "$3" = "2" ] && gpus="0,1"
  echo "START $1 $2 r$3 $4 s=$5 $(date +%H:%M:%S)" >> "$S"
  NEMO_DEBUG_BATCH=1 \
  RUNS="$2|$3 ; nemo|$4|ligerloss1 ; $5|1|1 ; none|false|false|false|false|false" \
  RUN_NAME=nemomoe GPU_POOL="$gpus" RUN_TIMEOUT_SECONDS=5400 OVERWRITE=false \
    bash scripts/lf/profile_lora_nemo.sh >> "agent/anchors_tmp/nemomoe_${1}.log" 2>&1
  v=$(grep -o 'VERDICT=[A-Z]*' "agent/anchors_tmp/nemomoe_${1}.log" | tail -1)
  echo "CELL $1 $2 r$3 $4 s=$5 -> ${v:-NONE} $(date +%H:%M:%S)" >> "$S"
  echo "${v#VERDICT=}"
}

# ── q3.5-122b-a10b, 2 ranks EP2 (~122 GB/rank weights + unfused gated attn) ──
for s in 4000 8000 16000 32000; do
  v=$(run_cell "q122r2rc$((s/1000))" q3.5-122b-a10b 2 recomp "$s")
  [ "$v" = "TRAINED" ] || { echo "q122 r2 recomp wall at $s ($v)" >> "$S"; break; }
done
run_cell q122r2rc128 q3.5-122b-a10b 2 recomp 128000   # tp2r rung 1
run_cell q122r2ao4 q3.5-122b-a10b 2 actoff 4000

# ── glm4.7-flash, 2 ranks EP2 (tp2r rungs 32k..192k) ──
for s in 16000 32000 64000 96000 128000 160000 192000; do
  v=$(run_cell "g47r2rc$((s/1000))" glm4.7-flash 2 recomp "$s")
  [ "$v" = "TRAINED" ] || { echo "glm4.7 r2 recomp wall at $s ($v)" >> "$S"; break; }
done
run_cell g47r2ao32 glm4.7-flash 2 actoff 32000

# ── glm4.5-air, 2 ranks EP2 (~106 GB/rank weights; tp2r rungs 16k..128k) ──
for s in 8000 16000 32000 48000 64000; do
  v=$(run_cell "g45r2rc$((s/1000))" glm4.5-air 2 recomp "$s")
  [ "$v" = "TRAINED" ] || { echo "glm4.5 r2 recomp wall at $s ($v)" >> "$S"; break; }
done
run_cell g45r2ao16 glm4.5-air 2 actoff 16000

# ── 1-rank: glm4.7 ladder (weights fit); load-probes for the >184GB models ──
for s in 16000 32000 64000 96000 128000; do
  v=$(run_cell "g47r1rc$((s/1000))" glm4.7-flash 1 recomp "$s")
  [ "$v" = "TRAINED" ] || { echo "glm4.7 r1 recomp wall at $s ($v)" >> "$S"; break; }
done
run_cell g45r1load glm4.5-air 1 recomp 8000        # 212GB weights vs 184GB HBM
run_cell q122r1load q3.5-122b-a10b 1 recomp 4000   # 244GB weights vs 184GB HBM

echo MOES_DONE >> "$S"

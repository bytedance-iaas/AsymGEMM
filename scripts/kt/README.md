# KT Scripts

This directory contains thin entrypoint wrappers for the KT-isolated LF scripts.
The canonical editable copies live under `agent/kt/scripts/` so KT profiling can
stay isolated from the shared LF scripts in `scripts/lf/`.

- `profile_lora_lf_kt.sh` -> `agent/kt/scripts/profile_lora_lf_kt.sh`
- `run_lf_lora_sft_kt.sh` -> `agent/kt/scripts/run_lf_lora_sft_kt.sh`

Do not edit `scripts/lf/run_lf_lora_sft.sh` or `scripts/lf/profile_lora_lf.sh`
for KT ARM BF16 work. Use GPU 1 first and GPU 2 only as fallback.
The kt_armbf16 production path is grouped packed SVE/BF16 forward and grouped
dropout-0 backward. Legacy scalar/backend selector envs are rejected instead of
selecting fallback code.
Optional tuning knobs are `KT_ARM_SFT_BACKWARD_GRAD_M_TILE`,
`KT_ARM_SFT_BACKWARD_GRAD_K_TILE`, and `KT_ARM_SFT_BACKWARD_LORA_R_TILE`.
Unset values preserve the accepted v5 default tile behavior.

Small KT source smoke:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
ART=profiling_kt_codex_smoke/kt_smoke_qwen3_s64_b1_r8_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT \
  ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  PROFILE_NSYS_GPU_METRICS_DEVICES=1 BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=1 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

/workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python \
  agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/source_profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 --expected-batch 1 --expected-rank 8 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms \
  --require-native-kv backward_base_kernel=grouped_sve_tile \
  --require-native-kv backward_lora_kernel=grouped_sve_tile_dropout0
```

Use `profiling_kt_codex_smoke/v5_*` artifact directories for this KT work.
Use physical GPU 1 first, physical GPU 2 only as fallback, and never GPU 0 or 3.

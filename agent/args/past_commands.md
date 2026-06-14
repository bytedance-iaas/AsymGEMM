# Past Commands

## 2026-06-13: Qwen3 Expert Policy / Expact Comparison

Config:

| field | value |
| --- | --- |
| dataset | `asym_long_sft_smoke` |
| model | `Qwen/Qwen3-30B-A3B` |
| template | `qwen3_nothink` |
| backend spec | `asym_cpuadamwds|norecomp` |
| global recompute | `GRADIENT_CHECKPOINTING=false` |
| router mode | `whole` |
| precision | `bf16` |
| GPU | `3` |
| num GPUs | `1` |
| seq len | `6144` |
| per-device batch | `4` |
| logical qlen | `24576` |
| max steps | `10` |
| warmup steps | `5` |
| total steps | `15` |
| LoRA rank | `64` |
| LoRA alpha | `16` |
| LoRA dropout | `0.00` |
| Asym offload modules | `all` |
| optimizer | AsymGEMM CPU AdamW, DeepSpeed backend |
| profiler | `source` |

Command: combined expact / gc-exp run.

```bash
OUTPUT_ROOT="$PWD/outputs/expact_vs_gc_exp_b4s6144_drop000_20260613T074927Z" \
RUN_POST=false \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
PREPARE_DATASETS=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_EXTERNAL_MEMORY=false \
bash scripts/lf/profile_lora_lf.sh --gpus 3
```

Command: rerun missing `gc-exp|false` / `none|false` rows in same output root.

```bash
OUTPUT_ROOT="$PWD/outputs/expact_vs_gc_exp_b4s6144_drop000_20260613T074927Z" \
RUN_POST=false \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="gc-exp|false,none|false" \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
PREPARE_DATASETS=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_EXTERNAL_MEMORY=false \
bash scripts/lf/profile_lora_lf.sh --gpus 3
```

Command: `tok-ge1|false` same workload.

```bash
OUTPUT_ROOT="$PWD/outputs/tok_ge1_vs_gc_exp_b4s6144_drop000_20260613T082151Z" \
RUN_POST=false \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="tok-ge1|false" \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
PREPARE_DATASETS=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_EXTERNAL_MEMORY=false \
bash scripts/lf/profile_lora_lf.sh --gpus 3
```

Metrics:

| expert policy | expact | impl | status | peak alloc | peak reserved | avg step | avg fwd | avg bwd | asym fwd calls | asym dx calls | fallback |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | `true` | activation offload | complete | 143.368 GiB | 158.055 GiB | 42.842s | 9.323s | 33.430s | 5055 | 4290 | 0 |
| `gc-exp` | `false` | torch checkpoint | complete | 167.462 GiB | 181.883 GiB | 3.649s | 1.464s | 2.135s | 6495 | 4290 | 0 |
| `tok-ge1` | `false` | custom | complete | 169.721 GiB | 179.266 GiB | 3.466s | 1.488s | 1.927s | 5775 | 4290 | 0 |
| `none` | `false` | none | OOM before measured step | 182.748 GiB | 182.848 GiB | n/a | n/a | n/a | 334 | 0 | 0 |

Profile paths:

```text
outputs/expact_vs_gc_exp_b4s6144_drop000_20260613T074927Z/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s6144_w5_s10_r64_a16_drop000/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact1/b4_s6144/source_profile.json
outputs/expact_vs_gc_exp_b4s6144_drop000_20260613T074927Z/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s6144_w5_s10_r64_a16_drop000/asym_cpuadamwds__source__norecomp__polgc-exp__routerwhole__expact0/b4_s6144/source_profile.json
outputs/tok_ge1_vs_gc_exp_b4s6144_drop000_20260613T082151Z/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s6144_w5_s10_r64_a16_drop000/asym_cpuadamwds__source__norecomp__poltok-ge1__routerwhole__expact0/b4_s6144/source_profile.json
outputs/expact_vs_gc_exp_b4s6144_drop000_20260613T074927Z/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s6144_w5_s10_r64_a16_drop000/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact0/b4_s6144/source_profile.partial.json
```

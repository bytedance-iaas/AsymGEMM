# ARM BF16 SFT Current Implementation

This document supersedes the earlier ARM BF16 SFT v2 planning notes. The old
experimental path switches and alternate cache mode have been removed from the
implementation and from public benchmark surfaces.

## Production Path

- `ARMBF16_SFT` always uses the packed forward path.
- Backward uses saved input, packed route metadata, and deterministic route
  recomputation for gradients.
- Positive LoRA dropout is supported by deterministic counter-based masks for
  gate, up, and down LoRA branches.
- `small_qlen_threshold` remains only as a task-dispatch tuning parameter for
  direct execution versus worker/OpenMP dispatch. It does not select another
  algorithm.
- SVE BF16 dot-product support is used when compiled and available. The
  reported `base_projection_kernel` should be `sve_bfdot` on the validated ARM
  host.

## Removed Surfaces

- No public cache-mode selector exists for ARM BF16 SFT.
- No benchmark flag exists to request alternate forward or backward paths.
- No cache-down diagnostic fields are reported by the ARM BF16 SFT wrapper or
  benches.
- Old path-control environment variables are hard rejected before native work is
  submitted, so stale shell environments fail loudly instead of changing
  behavior.

## Active Files

- Native implementation:
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
- Shared SFT config:
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/operators/common.hpp`
- Python wrapper:
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/arm.py`
- Factory integration:
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/experts.py`
- LlamaFactory integration:
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/wrapper.py`
- Benchmarks:
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/bench/bench_arm_sft_compare.py`
- Focused tests:
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py`
  `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py`

## Validation Gates

Run these after any future ARM BF16 SFT change:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
python setup.py build_ext --inplace
PYTHONPATH=python:. python -m pytest test/per_commit/test_armbf16_sft_reference.py test/per_commit/test_sft_lora_dropout.py -q
```

Smoke the public benchmark surface:

```bash
PYTHONPATH=python:. python bench/bench_armbf16_sft.py \
  --backend arm --qlen 1 8 --experts 4 --topk 2 --hidden 64 \
  --intermediate 32 --rank 4 --alpha 8 --lora-dropout 0.10 \
  --small-qlen-threshold 16 --warmup 1 --iters 1 --threads 1 \
  --skip-correctness --output-json /tmp/armbf16_strict_smoke.json
```

Expected smoke properties:

- every ARM result has `last_forward_path == "packed"`;
- `base_projection_kernel == "sve_bfdot"` on the validated ARM host;
- no removed path selector or cache-down fields appear in JSON output.

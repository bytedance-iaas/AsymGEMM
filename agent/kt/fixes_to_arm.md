# ARM BF16 SFT Fixes To ARM

This file records the final current design after the ARM BF16 SFT cleanup. Older
experimental notes have been removed from this file because they described paths
that no longer exist.

## Current Design

- The ARM BF16 SFT implementation is packed-only for forward execution.
- Backward gradients are computed from saved inputs and packed route metadata.
- Route recomputation is implemented in the native ARM file and is part of the
  optimized backward path.
- Public Python wrappers do not expose a cache-mode selector.
- Benchmarks do not expose removed path selector flags.
- The native extension rejects stale path-control environment variables.

## Files That Own The Behavior

- `kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - packed forward, route packing, SVE BF16 kernels, route recomputation,
    backward accumulation, pool diagnostics, and stale-environment rejection.
- `kt-kernel/operators/common.hpp`
  - shared SFT config fields that are still live.
- `kt-kernel/ext_bindings.cpp`
  - pybind exposure for live ARM BF16 SFT diagnostics only.
- `kt-kernel/python/sft/arm.py`
  - Python wrapper, native config creation, and pre-submit stale-environment
    rejection.
- `kt-kernel/python/experts.py`
  - SFT wrapper factory wiring.
- `kt-kernel/python/sft/wrapper.py`
  - LlamaFactory integration wiring.
- `kt-kernel/bench/bench_armbf16_sft.py`
  - focused ARM/Torch SFT benchmark.
- `kt-kernel/bench/bench_arm_sft_compare.py`
  - multi-layer comparison benchmark.
- `scripts/testing/validate_kt_arm_sft_optimizations.py`
  - validation gates for current diagnostics.

## Required Validation

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
python setup.py build_ext --inplace
PYTHONPATH=python:. python -m pytest test/per_commit/test_armbf16_sft_reference.py test/per_commit/test_sft_lora_dropout.py -q
```

Run a smoke benchmark and validator:

```bash
PYTHONPATH=python:. python bench/bench_armbf16_sft.py \
  --backend arm --qlen 1 8 --experts 4 --topk 2 --hidden 64 \
  --intermediate 32 --rank 4 --alpha 8 --lora-dropout 0.10 \
  --small-qlen-threshold 16 --warmup 1 --iters 1 --threads 1 \
  --skip-correctness --output-json /tmp/armbf16_strict_smoke.json

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
python scripts/testing/validate_kt_arm_sft_optimizations.py \
  --stage strict-smoke \
  --candidate-json /tmp/armbf16_strict_smoke.json \
  --backend ARMBF16_SFT \
  --qlen 8 \
  --require-forward-path packed \
  --require-kernel sve_bfdot \
  --require-warmup-ran \
  --require-lora-warmup \
  --require-aligned-weights \
  --require-route-metadata \
  --require-route-skew-fields \
  --require-pool-backed
```

## Non-Goals

- Do not restore alternate ARM BF16 SFT path selectors.
- Do not add another cache-mode selector without proving it is faster and
  maintaining the same correctness gates.
- Do not use historical artifact JSON files as current command references.

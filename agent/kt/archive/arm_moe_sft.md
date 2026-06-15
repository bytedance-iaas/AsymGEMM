# ARM BF16 SFT Current Implementation

This file records the current supported ARM BF16 SFT path. It replaces the old
staged design notes so future agents do not reintroduce removed algorithm
branches or disabled optimization switches.

## Production Path

- Backend: `ARMBF16_SFT`.
- Native implementation: `third_party/ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`.
- Python wrapper: `third_party/ktransformers/kt-kernel/python/sft/arm.py`.
- Public factory and trainer wiring:
  - `third_party/ktransformers/kt-kernel/python/experts.py`
  - `third_party/ktransformers/kt-kernel/python/sft/config.py`
  - `third_party/ktransformers/kt-kernel/python/sft/wrapper.py`
- Binding: `third_party/ktransformers/kt-kernel/ext_bindings.cpp`.

The only production algorithm is packed forward plus input-only recompute
backward. The forward path reported by profiling and benchmark JSON must be
`packed`.

## Removed Paths

These paths and controls are intentionally not supported:

- scalar forward/backward production branches;
- intermediate-activation cache policies;
- route-only or phased backward compatibility branches;
- cache-down diagnostic outputs;
- prefetch env/CLI toggles that did not affect computation;
- synchronous default backward repack.

Old path-selection environment variables are rejected loudly by the Python wrapper
and native backend so stale shells fail before submitting work.

## Runtime Behavior

- Forward stores only the input, top-k weights, expert ids, route metadata, and
  deterministic dropout state needed for backward.
- Backward pops exactly one input-only cache entry and recomputes per-route
  gate/up activation/down values.
- LoRA dropout supports any `0 <= p < 1` and uses deterministic counter-based
  masks so forward recompute and backward agree.
- Backward weight repack is always asynchronous when submitted through
  `submit_backward_repack`; callers synchronize through `wait_backward_repack`.
- Direct ARM wrappers default `share_backward_bb` to true, matching the
  production `KTConfig` default used by LlamaFactory wiring.
- SVE BF16 dot kernels are required for this backend; builds without the required
  ARM features fail instead of selecting an unoptimized projection path.

## Validation Commands

Run these from `third_party/ktransformers/kt-kernel` after native changes:

```bash
python setup.py build_ext --inplace
PYTHONPATH=python:. python -m pytest \
  test/per_commit/test_armbf16_sft_reference.py \
  test/per_commit/test_sft_lora_dropout.py -q
```

Run this smoke benchmark and validator from `third_party/AsymGEMM`:

```bash
PYTHONPATH=/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python:/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel \
python /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --qlen 1 8 --experts 4 --topk 2 --hidden 64 --intermediate 32 \
  --rank 4 --alpha 8 --lora-dropout 0.10 --small-qlen-threshold 16 \
  --warmup 1 --iters 1 --threads 1 --skip-correctness \
  --output-json /tmp/armbf16_strict_smoke.json

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
  --require-pool-backed \
  --require-async-repack-enabled 1
```

The benchmark CLI should reject removed flags for old path selection and removed
optimization switches. If any removed option is accepted again, the implementation
has regressed.

## Convergence Rule

Before declaring this area converged, run a fresh code and docs search for stale
path controls, rebuild the native extension if C++ changed, run the focused
pytest suite, run the smoke benchmark, and validate that benchmark JSON still
reports `last_forward_path=packed`.

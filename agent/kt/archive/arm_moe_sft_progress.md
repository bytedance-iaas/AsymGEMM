# KT ARM MoE SFT Progress

Date: 2026-06-04

## Metrics / Results

All KT ARM SFT work is isolated under
`/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers-arm`.

### LF LoRA-SFT 4k E2E

Artifact root:
`third_party/ktransformers-arm/profiling_kt/lf_three_way_s4096_all_lora_steps1_apples_20260604/`

Workload: `Qwen/Qwen3-30B-A3B`, `cutoff_len=4096`, `max_samples=1`,
`max_steps=1`, batch `1`, grad accumulation `1`, `lora_target=all`,
rank `8`, alpha `16`, dropout `0`, seed `42`, pure BF16,
gradient checkpointing off, GPU `CUDA_VISIBLE_DEVICES=3`.

Execution placement:

| Backend label | Dense LF model / attention | Routed MoE expert compute | Persistent expert storage |
|---|---|---|---|
| LF torch GPU | CUDA | CUDA torch/HF path, no KT wrapper | CUDA/HF Qwen expert |
| LF KT TORCHBF16 on CUDA | CUDA | KT wrapper active, but `TORCHBF16_SFT` computes with torch on CUDA via `KT_TORCHBF16_SFT_DEVICE=cuda` | expert base weights on CUDA; fused expert LoRA params/grads remain CPU KT buffers with transient CUDA copies |
| LF KT ARMBF16 on CPU | CUDA | KT wrapper active, native `ARMBF16_SFT` computes on ARM CPU | CPU KT buffers |

`KT` here means the LLaMA-Factory MoE layer is routed through the KT wrapper.
It does not by itself imply CPU execution. CPU vs CUDA is selected by the
specific KT backend and, for `TORCHBF16_SFT`, by `KT_TORCHBF16_SFT_DEVICE`.
The CUDA TORCHBF16 row still saves HBM versus LF torch GPU because it uses the
KT memory layout: original HF expert parameters are replaced by tiny CPU
placeholders, fused expert LoRA params/grads are KT-managed CPU buffers, and
the routed expert branch is hidden behind `KTMoEFunction` instead of retaining
the ordinary HF/PEFT expert autograd graph for all layers.

| Backend | Runtime | Train loss | Delta vs torch GPU | Peak HBM | Expert LoRA | KT calls |
|---|---:|---:|---:|---:|---:|---:|
| LF torch GPU | `17.919 s` | `1.366307` | `0.000000` | `106.173 GiB` | `415,236,096` Qwen expert | `0 fw / 0 bw` |
| LF KT TORCHBF16 on CUDA | `20.630 s` | `1.364013` | `-0.002294` | `78.202 GiB` | `415,236,096` KT fused | `48 fw / 48 bw` |
| LF KT ARMBF16 on CPU | `160.461 s` | `1.367199` | `+0.000891` | `23.437 GiB` | `415,236,096` KT fused | `48 fw / 48 bw` |

Profiler wall time including setup/model load: torch GPU `33.418 s`,
KT TORCHBF16 on CUDA `43.358 s`, KT ARMBF16 on CPU `185.026 s`.

Coverage audit: all rows report `421,920,768` trainable params,
`753,079,381,131,264` FLOPs, same 4k dataset sample, and no runtime error.
KT rows wrapped all 48 MoE layers. The measured LF `TORCHBF16_SFT` row is not
CPU torch; it is the CUDA opt-in reference row. A separate LF
`TORCHBF16_SFT` CPU row has not been run.

### Ops Kernel

Artifact:
`third_party/ktransformers-arm/profiling_kt/bench/arm_sft_ops_qwenlike_s16_t16_cuda_batched.json`

Shape: Qwen-like routed expert LoRA math, `qlen=16`, `layers=1`,
`experts=128`, `topk=8`, `hidden=2048`, `intermediate=768`, rank `8`,
alpha `16`, threads `16`, seed `20260604`.

| Backend | Latency | Output rel L2 | Grad-input rel L2 | Max LoRA-grad rel L2 |
|---|---:|---:|---:|---:|
| torch CPU batched | `656.929 ms` | `6.072e-05` | `7.574e-05` | `9.072e-07` |
| torch GPU batched | `5.970 ms` | `6.232e-05` | `9.825e-05` | `1.124e-06` |
| KT TORCHBF16 CPU wrapper | `1070.404 ms` | `6.348e-05` | `9.254e-05` | `1.037e-06` |
| KT ARMBF16 CPU native | `338.325 ms` | `5.780e-05` | `1.034e-04` | `2.241e-03` |

### Standalone LoRA E2E

Artifact:
`third_party/ktransformers-arm/profiling_kt/bench/arm_sft_lora_e2e_qwenlike_s8_l2_t16_cuda_batched.json`

Shape: same Qwen-like expert LoRA coverage, `qlen=8`, `layers=2`,
`experts=128`, `topk=8`, `hidden=2048`, `intermediate=768`, rank `8`,
alpha `16`, threads `16`, seed `20260604`.

| Backend | Latency | Output rel L2 | Grad-input rel L2 | Max LoRA-grad rel L2 |
|---|---:|---:|---:|---:|
| torch CPU batched | `872.001 ms` | `4.404e-04` | `8.144e-04` | `4.302e-04` |
| torch GPU batched | `9.596 ms` | `4.754e-04` | `9.230e-04` | `4.545e-04` |
| KT TORCHBF16 CPU wrapper | `1480.437 ms` | `4.349e-04` | `7.857e-04` | `4.222e-04` |
| KT ARMBF16 CPU native | `340.055 ms` | `4.440e-04` | `8.617e-04` | `1.949e-03` |

### Validation

Focused SFT tests passed:

```text
12 passed in 7.52s
```

Command:

```bash
PYTHONPATH=third_party/ktransformers-arm/kt-kernel \
KT_KERNEL_ALLOW_PY_FALLBACK=0 \
third_party/ktransformers-arm/.venv/bin/python -m pytest \
  third_party/ktransformers-arm/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  third_party/ktransformers-arm/kt-kernel/test/per_commit/test_torchbf16_sft_reference.py \
  third_party/ktransformers-arm/kt-kernel/test/per_commit/test_torchbf16_sft_wrapper_lifecycle.py -q
```

## Progress

- Built isolated lab under `third_party/ktransformers-arm` with separate venv,
  cloned/patched LlamaFactory integration, and kept production repos untouched.
- Implemented KT SFT wrapper path for LF: fused expert LoRA tensors are handled
  through KT MoE wrappers while attention LoRA remains in PEFT. `lora_target=all`
  therefore appears as attention modules in PEFT plus `415,236,096` fused expert
  LoRA params in LF/KT profiling.
- Implemented TORCHBF16 SFT reference backend. Default remains CPU for unit
  tests and ops microbenchmarks; the LF 4k comparison uses the CUDA opt-in via
  `KT_TORCHBF16_SFT_DEVICE=cuda`, so its HBM is not expected to match ARMBF16.
- Implemented native ARM BF16 SFT backend and added OpenMP token parallelism.
  It is correct enough for the current tests and LF one-step validation, but it
  is still scalar/OpenMP and much slower than the full GPU path.
- Added/updated profiling entry points under
  `third_party/ktransformers-arm/scripts/lf/kt/` and recorded outputs under
  `third_party/ktransformers-arm/profiling_kt/`.

## Current Conclusion

- Correctness is acceptable for the measured BF16 SFT path: one-step 4k LF
  losses are close across torch GPU, KT TORCHBF16 on CUDA, and KT ARMBF16 on
  CPU.
- KT ARMBF16 on CPU achieves the intended HBM reduction at 4k
  (`106.173 GiB -> 23.437 GiB`) by moving the expert path to CPU.
- The current ARM BF16 kernel is not performance-ready: 4k LF runtime is
  `8.95x` slower than full GPU torch. Next work should optimize the ARM expert
  kernel before treating this as a production SFT backend.
- Full LF torch CPU was not run because it changes the entire training
  placement and is not apples-to-apples with the single-GPU LF rows; CPU
  comparisons are covered at ops/standalone-kernel level.

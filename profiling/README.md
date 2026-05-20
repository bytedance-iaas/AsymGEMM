# Profiling Results Layout

Each workload has its own directory.  The top-level `table.md` and
`profile.json` are the Nsight/profile-mode GPU timeline results and are the
only truth profiling tables.  Source-label coverage artifacts are debug-only
and are kept as `debug.md` / `debug.json`.

| Directory | Workload | Contents |
|---|---|---|
| `matrix_1b/` | Fundamental single 1.073B-parameter frozen matrix | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `mlp_1b/` | Fundamental 1.073B-parameter two-layer MLP | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `mlp/` | M4.1 MLP toy | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `dense_llm/` | M4.2 dense LLM toy | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `moe_contiguous/` | M4.3 MoE toy, contiguous routing | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `moe_masked/` | M4.3 MoE toy, masked routing | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `qwen3_14b/` | Qwen3-14B config-matched profile | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `qwen3_30b_a3b/` | Qwen3-30B-A3B config-matched profile | Nsight GPU timeline/kernel/memcpy/no-kernel table |
| `*/debug.md`, `*/debug.json` | Matching source-label report | Python/source range coverage only; not performance truth |
| `*/nsight/` | Matching Nsight report | Same data as the workload top-level table |
| `*/ncu/` | Matching Nsight Compute report when present | Kernel-internal metrics for AsymGEMM kernels |

Use `--timing-mode profile` plus Nsight Systems for real GPU bubble analysis.
Use `--timing-mode debug_sync` only for source label coverage debugging.
Use Nsight Compute only for kernel-internal diagnosis; it replays kernels and
must not be used for end-to-end wall-time claims.
Use `scripts/profile_nsys_cpu_gaps.py` for CPU-root-cause debug captures of
GPU no-kernel gaps.  It writes `*/cpu_gaps/table.md` and keeps CUDA/NVTX/OSRT
CPU sample/context-switch attribution separate from the low-overhead truth
tables.  This debug mode has non-negligible profiling overhead and should not
be used for paper timing claims.

The fundamental selectors are:

```bash
python scripts/profile_lora.py --workload matrix_1b --timing-mode profile
python scripts/profile_lora.py --workload mlp_1b --timing-mode profile
```

LoRA-SFT workflow comparisons can run `asym_only` and `torch_only` over the
same workload list.  `asym_only` is the direct AsymGEMM host-weight path.
`torch_only` keeps the same host-weight wrapper and uses the PyTorch fallback
path; it is useful for sanity checks, but it is not a GPU-resident
LLaMA-Factory baseline.

```bash
python scripts/profile_lora_driver.py \
  --workloads qwen3_14b qwen3_30b_a3b \
  --backends asym_only torch_only \
  --cuda-devices 2,3,4,5,6,7 \
  --profile-layers 1 --batch-size 1 --seq-len 64 \
  --lora-rank 64 --lora-alpha 128
```

The fundamental NCU selectors are:

```bash
python scripts/profile_ncu_asymgemm.py --workload matrix_1b --preset paper
python scripts/profile_ncu_asymgemm.py --workload mlp_1b --preset paper
```

CPU gap debug example:

```bash
python scripts/profile_nsys_cpu_gaps.py --workload qwen3_30b_a3b --device cuda:2
```

PyTorch stack debug trace example:

```bash
python scripts/profile_lora.py --workload mlp_1b --device cuda:2 --export-torch-trace --torch-profiler-with-stack
```

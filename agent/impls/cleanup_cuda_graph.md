# Cleanup LF CUDA Graph Experiment

Goal: remove the rejected LF `ASYM_CUDA_GRAPH=compile` / HF
`torch_compile reduce-overhead` path so ordinary profiling behaves as if this
experiment never existed.

## Runtime Cleanup

- Remove LF script interface and env plumbing:
  - `scripts/lf/profile_lora_lf.sh`
  - `scripts/lf/run_lf_lora_sft.sh`
- Remove profile metadata/reporting for compile health:
  - `scripts/lf/run_lf_profiled_train.py`
- Remove compile-only trace-range control:
  - `asym_gemm/profiling/lf_trace.py`
  - `asym_gemm/training/profile_ranges.py`
- Remove Dynamo wrappers and compile-only profile-name short-circuits:
  - `asym_gemm/training/frozen_linear.py`
  - `asym_gemm/training/lora.py`
  - `asym_gemm/training/qwen3_moe.py`
- Delete rejected comparator/test files:
  - `scripts/lf/compare_cuda_graph_profiles.py`
  - `tests/lf/test_compare_cuda_graph_profiles.py`

## Keep

- `initialize_asym_cuda_graph_state(...)` and
  `initialize_asym_single_group_launch_tensors(...)` in
  `asym_gemm/training/frozen_linear.py`. These are low-level static launch
  helpers, not the LF compile experiment.
- Operator microbenchmark CUDA graph support in `scripts/lora/profile_lora_ops.py`
  and `scripts/lora/profile_lora_ops.sh`.
- `torch.cuda.is_current_stream_capturing()` allocation guards.
- Qwen3/Llama4 activation-offload and packed MoE code.

## Acceptance

The cleanup is accepted only if:

- `rg` finds no LF SFT `ASYM_CUDA_GRAPH`, `torch_compile`, `_cgcompile`,
  compile-health, or trace-range plumbing in `scripts/lf`, `asym_gemm`, or
  active tests.
- Normal dry-run command generation emits no CUDA graph envs and no
  `--torch_compile` args.
- Syntax and focused tests pass.
- The normal LF smoke still completes.

Validation commands:

```bash
bash -n scripts/lf/profile_lora_lf.sh scripts/lf/run_lf_lora_sft.sh
.venv/bin/python -m py_compile \
  scripts/lf/run_lf_profiled_train.py \
  asym_gemm/profiling/lf_trace.py \
  asym_gemm/training/frozen_linear.py \
  asym_gemm/training/lora.py \
  asym_gemm/training/profile_ranges.py \
  asym_gemm/training/qwen3_moe.py
.venv/bin/python -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/test_lf_memory_breakdown.py \
  tests/training/test_cpu_resident_frozen_base.py
bash scripts/lf/profile_lora_lf_test.sh
```

# Port SM100 BF16 kernel improvements to ASyGEMM-SFT

## Audit result

Source:

```text
/home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM
```

Destination:

```text
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
```

Both nested AsymGEMM repos are at:

```text
b43695ee545f7cbc27cd7ef290d1fb2fbf1f9d41
```

Destination has unrelated local changes, but none are in the kernel/runtime files below. Do not touch destination-local LF script/integration edits.

## Port these runtime files

### 1. CPU-left BF16 forward kernel

Copy the source changes for exactly these files:

```text
asym_gemm/include/asym_gemm/impls/sm100_bf16_cpu_left_asym_gemm.cuh
csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp
csrc/apis/gemm.hpp
asym_gemm/__init__.py
asym_gemm/training/cpu_left.py
asym_gemm/training/exp_act_offload_lora.py
```

Required behavior:

- Add opt-in compact M-grid support:
  - Python env: `DG_BF16_CPU_LEFT_COMPACT_GRID=1`
  - Python computes `compact_m_blocks` from padded group sizes and active kernel `BLOCK_M`.
  - Existing binding accepts optional trailing `compact_m_blocks`.
  - JIT launcher uses `grid.x = compact_m_blocks` when nonzero.
  - Kernel maps compact block index with `scheduler.m_start + blockIdx.x`.

- Add opt-in native gate/up pair forward:
  - Python env: `ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=1`
  - New native binding:

```text
sm100_m_grouped_bf16_cpu_left_pair_asym_gemm_nt_contiguous
```

  - New Python helper: `grouped_expert_lora_pair_cpu_left`.
  - The kernel receives second B/D TMA descriptors and a template flag for pair output.
  - It keeps one pinned-CPU A load stream and loops over two logical N-output halves:
    - half 0: gate B/D
    - half 1: up B/D
  - No CUDA mirror/staging of the CPU activation is introduced.

- Keep old behavior as default:
  - if `DG_BF16_CPU_LEFT_COMPACT_GRID` is unset, launch the original full M grid.
  - if `ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE` is unset, use the old two-call gate/up path.

Important small details:

- Remove the CPU-left launcher assert that forced `config.block_m == config.block_n`.
- Add the pair binding to top-level `asym_gemm.__init__`.
- Preserve the existing `ASYMM_CPU_LEFT_LORA_A_PAIR_CAT=1` fallback only as a fallback/debug path; do not use it when validating the native pair port.

Runtime env for this part:

```bash
export ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=1
export DG_BF16_CPU_LEFT_COMPACT_GRID=1
```

### 2. LoRA-A grad CPU-right kernel

Copy the source changes for exactly this file:

```text
csrc/exp_act_offload/exp_act_offload_kernels.cu
```

Required behavior:

- Add tiled, atomic-free LoRA-A weight-gradient kernel:
  - one CTA per `(group, K-tile)`
  - stages CUDA `dS` and pinned CPU `source_cpu` into shared memory
  - reduces over group rows inside the CTA
  - writes directly to BF16 grad output
  - avoids global atomics, FP32 scratch tensors, and the separate FP32-to-BF16 cast in the default path

- Apply it to both:

```text
sm100_grouped_lora_a_grad_bf16_cpu_right
sm100_grouped_lora_a_pair_grad_bf16_cpu_right
```

- Keep legacy atomic kernel only behind:

```bash
ASYMM_LORA_A_GRAD_ATOMIC=1
```

No new build registration files are needed for this. `csrc/apis/exp_act_offload.hpp` is unchanged in the source.

## Test-only updates

Runtime port does not require tests, but these are the only relevant test updates:

```text
tests/training/test_cpu_left_lora.py
```

Use it to cover:

- env-off default two-call path
- `ASYMM_CPU_LEFT_LORA_A_PAIR_CAT=1`
- `ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=1`
- compact-grid optional arg compatibility with fake bindings

Optional test hunk:

```text
tests/training/test_lf_qwen3_asym_backend.py
```

Only port the native-pair call-count assertion hunk if you need that Qwen3 backend test to pass under `ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=1`. Do not copy its `ASYMM_EXPERT_SILU_BWD_GPU` stats hunk unless also porting the non-kernel Qwen3 change below.

## Do not port for this kernel-only task

Skip these source changes/files:

```text
asym_gemm/training/qwen3_moe.py
scripts/lf/profile_lora_lf_short.sh
agent/impls/improve_sm100_bf16*.md
scripts/lora/analyze_nsys_asym.py
scripts/lora/microbench_lora_a_grad.py
```

Reasons:

- `qwen3_moe.py` is v14 GPU SwiGLU backward. It is a large E2E Python scheduling optimization, not a small SM100 BF16 kernel port.
- `scripts/lf/profile_lora_lf_short.sh` only changes profiling defaults and conflicts with destination-local script edits.
- `agent/impls/improve_*` are notes.
- `scripts/lora/*.py` are diagnostics, not runtime.

## No changes needed

I checked the source diff: these files have no source changes for this port:

```text
CMakeLists.txt
setup.py
pyproject.toml
csrc/python_api.cpp
csrc/apis/exp_act_offload.hpp
asym_gemm/training/__init__.py
```

Do not edit them.

## Patch order

1. Port `asym_gemm/include/asym_gemm/impls/sm100_bf16_cpu_left_asym_gemm.cuh`.
2. Port `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp`.
3. Port `csrc/apis/gemm.hpp`.
4. Port `asym_gemm/__init__.py`.
5. Port `asym_gemm/training/cpu_left.py`.
6. Port `asym_gemm/training/exp_act_offload_lora.py`.
7. Port `csrc/exp_act_offload/exp_act_offload_kernels.cu`.
8. Port only the test hunks you need from the test-only section.

Because source and destination share the same base commit, the runtime files above can be ported by applying the corresponding source diffs. Do not copy entire directories.

## Validation

Rebuild destination:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m pip install -e . --no-build-isolation -q
```

Use a fresh JIT cache after changing the generated JIT stub or included `.cuh`.

Default env-off validation:

```bash
unset ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE
unset ASYMM_CPU_LEFT_LORA_A_PAIR_CAT
unset DG_BF16_CPU_LEFT_COMPACT_GRID
unset ASYMM_LORA_A_GRAD_ATOMIC
DG_JIT_CACHE_DIR="$PWD/profiling/sm100_bf16_cpu_left/port_default_jit" \
.venv/bin/python -m pytest -q \
  tests/m_grouped/test_sm100_bf16_cpu_left.py \
  tests/training/test_cpu_left_lora.py \
  tests/training/test_exp_act_offload_native.py
```

Native pair + compact grid validation:

```bash
ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=1 \
DG_BF16_CPU_LEFT_COMPACT_GRID=1 \
DG_JIT_CACHE_DIR="$PWD/profiling/sm100_bf16_cpu_left/port_native_pair_jit" \
.venv/bin/python -m pytest -q \
  tests/m_grouped/test_sm100_bf16_cpu_left.py \
  tests/training/test_cpu_left_lora.py \
  tests/training/test_exp_act_offload_native.py
```

Atomic fallback validation:

```bash
ASYMM_LORA_A_GRAD_ATOMIC=1 \
.venv/bin/python -m pytest -q tests/training/test_exp_act_offload_native.py
```

Expected source-side reference:

- native pair/compact tests: `37 passed, 1 skipped`
- tiled LoRA-A grad default and `ASYMM_LORA_A_GRAD_ATOMIC=1`: `tests/training/test_exp_act_offload_native.py` passes

## Enabling in the long profiling sweep

`scripts/lf/profile_lora_lf_long.sh` carries two toggles near the top (1=on, 0=off), both forwarded to training via `run_env`:

- `ASYMM_EXPERT_SILU_BWD_GPU` (default 1) — v14 expert SwiGLU backward on GPU.
- `DG_BF16_CPU_LEFT_COMPACT_GRID` (default 1) — compact CPU-left forward M-grid.

Notes:

- `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD` stays `hbm`, so the native gate/up pair forward (`ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE`) is intentionally NOT exercised. Compact grid's multi-group win is the cpu-left *expert* forward (needs `cpu`); under `hbm` it only touches the attention act-offload cpu-left path (~single group), so expect little effect there.
- The tiled atomic-free LoRA-A grad kernel is the default (atomic only via `ASYMM_LORA_A_GRAD_ATOMIC=1`), so it is measured without any toggle.
- These two vars are not part of the profiler's recorded config, so `profile.json` will not label them.

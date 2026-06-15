# KT ARM SFT Merge-Back Integration Plan

This document is the integration plan for taking the proven isolated
implementation in:

```text
/workspace/AsymGEMM-SFT/third_party/ktransformers-arm
```

and merging the required source changes back into the normal repo trees:

```text
/workspace/AsymGEMM-SFT/third_party/ktransformers
/workspace/AsymGEMM-SFT/third_party/LlamaFactory
/workspace/AsymGEMM-SFT/third_party/AsymGEMM
```

This is not a rewrite. The ARM KT implementation already works in
`third_party/ktransformers-arm`; the job here is to port the minimum required
features into the normal KT/LF integration so ARM KT can be selected as a backend
from this repo.

Public user surface after merge:

```yaml
use_kt: true
kt_backend: TORCHBF16   # portable KT SFT oracle
```

or:

```yaml
use_kt: true
kt_backend: ARMBF16     # native ARM CPU BF16 SFT backend
```

Do not add a separate `use_kt_arm` switch. ARM is a KT backend choice under the
existing `use_kt` path.

## Current Proof State

The isolated implementation under `third_party/ktransformers-arm` proved:

- `TORCHBF16_SFT` works as a portable KT SFT correctness backend.
- `ARMBF16_SFT` works as a native ARM CPU BF16 SFT backend.
- LLaMA-Factory Qwen3-30B-A3B LoRA-SFT at 4k runs with all 48 MoE layers wrapped.
- KT wrapper counters are nonzero: 48 forward calls and 48 backward calls for the
  one-step 4k validation.
- Loss matches the LF torch GPU baseline closely enough for a one-step BF16 SFT
  validation.
- ARM BF16 CPU offload sharply reduces HBM, but is currently much slower than GPU
  torch and should not be treated as performance-complete.

Reference metrics from
`third_party/AsymGEMM/agent/kt/arm_moe_sft_progress.md`:

| Backend | Runtime | Train loss | Delta vs torch GPU | Peak HBM | Expert LoRA | KT calls |
|---|---:|---:|---:|---:|---:|---:|
| LF torch GPU | `17.919 s` | `1.366307` | `0.000000` | `106.173 GiB` | `415,236,096` Qwen expert | `0 fw / 0 bw` |
| LF KT TORCHBF16 on CUDA | `20.630 s` | `1.364013` | `-0.002294` | `78.202 GiB` | `415,236,096` KT fused | `48 fw / 48 bw` |
| LF KT ARMBF16 on CPU | `160.461 s` | `1.367199` | `+0.000891` | `23.437 GiB` | `415,236,096` KT fused | `48 fw / 48 bw` |

Note: `LF KT TORCHBF16 on CUDA` is a CUDA reference row selected by
`KT_TORCHBF16_SFT_DEVICE=cuda`. The default `TORCHBF16_SFT` backend remains the
CPU/offload correctness oracle.

## Hard Guardrails

1. Do not copy the isolated `LlamaFactory` tree wholesale. It was cloned before
   the current AsymGEMM LF changes and would delete existing AsymGEMM support.
2. Do not copy runtime artifacts:
   - `.venv`
   - `profiling_kt`
   - `__pycache__`
   - `.pytest_cache`
   - `*.pyc`
   - built `*.so`
   - `*.egg-info`
3. Preserve all current AsymGEMM behavior:
   - `use_asym_gemm`
   - AsymGEMM parser checks
   - AsymGEMM launcher behavior
   - AsymGEMM adapter/save path
   - memory attribution and profiling scripts
4. Keep KT and AsymGEMM mutually exclusive in LF for now.
5. Single-GPU KT ARM is the integration target. FSDP/DDP is a later extension.
6. `lora_target all` must still train expert LoRA for all fused MoE experts.
7. Do not make `ARMBF16` silently fall back to `TORCHBF16`; failed native import
   or missing binding must produce a clear error.

## Source Of Truth

Before editing, inspect and freeze the source patch stack:

```bash
cd /workspace/AsymGEMM-SFT

git -C third_party/ktransformers-arm status --short
git -C third_party/ktransformers-arm/LlamaFactory status --short

diff -qr \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='*.pyc' \
  --exclude='*.so' \
  --exclude='*.egg-info' \
  --exclude='kt_kernel' \
  third_party/ktransformers/kt-kernel \
  third_party/ktransformers-arm/kt-kernel

diff -qr \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  third_party/LlamaFactory/src/llamafactory \
  third_party/ktransformers-arm/LlamaFactory/src/llamafactory
```

Use these diffs as the source, but merge selectively.

## Phase 1: Merge KT Kernel Source

Target tree:

```text
third_party/ktransformers/kt-kernel
```

Source tree:

```text
third_party/ktransformers-arm/kt-kernel
```

### 1.1 Build And CPU Detection

Port these files:

```text
kt-kernel/CMakeLists.txt
kt-kernel/setup.py
kt-kernel/python/_cpu_detect.py
```

Required behavior:

- Detect ARM machines from `platform.machine()` as `aarch64`, `arm64`, or
  `armv*`.
- Support ARM variants:
  - `arm_svebf16`
  - `arm_sve`
  - `arm_generic`
- Keep x86 variants unchanged.
- Keep x86 fallback chain x86-only. Do not fall back to `avx2` on ARM.
- Add ARM feature reporting in `setup.py`: `NEON`, `SVE`, `SVE2`, `SVE_BF16`,
  `BF16`, `I8MM`.
- Keep the CMake ARM `-mfp16-format=ieee` check as a real boolean check:
  `if(COMPILER_SUPPORTS_FP16_FORMAT_I3E)`.
- Preserve optional Python fallback only for pure Python `TORCHBF16_SFT` use.
  Native validation should run with `KT_KERNEL_ALLOW_PY_FALLBACK=0`.

Validation:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
PYTHONPATH=python KT_KERNEL_DEBUG=1 python - <<'PY'
import kt_kernel
print("variant", kt_kernel.__cpu_variant__)
PY
```

Expected on this ARM machine: an ARM variant, not `avx2`.

### 1.2 Native ARM C++ Operators And Bindings

Port these source files:

```text
kt-kernel/operators/arm/bf16_sft_moe.hpp
kt-kernel/operators/arm/int8_moe.hpp
kt-kernel/ext_bindings.cpp
```

Required binding behavior:

- Under `#if defined(__aarch64__)`, include the ARM headers.
- Bind `kt_kernel_ext.moe.ARMBF16_SFT_MOE`.
- Bind ARM INT8 as `Int8_KERNEL_MOE` under ARM.
- Prevent duplicate generic `Int8_KERNEL_MOE` binding on ARM by guarding the
  generic `USE_MOE_KERNEL` binding with `!defined(__aarch64__)`.
- Keep all x86/AMX bindings unchanged.

Do not port unrelated comment-only diffs such as the CUDA top-k comment unless
they appear in a required patch hunk.

Validation after build:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
python -m pip install -v .
KT_KERNEL_ALLOW_PY_FALLBACK=0 PYTHONPATH=python python - <<'PY'
import kt_kernel_ext
print("has ARMBF16", hasattr(kt_kernel_ext.moe, "ARMBF16_SFT_MOE"))
print("has INT8", hasattr(kt_kernel_ext.moe, "Int8_KERNEL_MOE"))
assert hasattr(kt_kernel_ext.moe, "ARMBF16_SFT_MOE")
assert hasattr(kt_kernel_ext.moe, "Int8_KERNEL_MOE")
PY
```

### 1.3 Python SFT Backends And Dispatch

Port/add these files:

```text
kt-kernel/python/sft/torch_backend.py
kt-kernel/python/sft/arm.py
kt-kernel/python/sft/__init__.py
kt-kernel/python/experts.py
kt-kernel/python/utils/moe_kernel.py
```

Required behavior:

- Add SFT methods:
  - `TORCHBF16_SFT`
  - `ARMBF16_SFT`
- Add inference method:
  - `ARMINT8`
- Dispatch:
  - `TORCHBF16_SFT` -> `TorchBF16SFTMoEWrapper`
  - `ARMBF16_SFT` -> `ArmBF16SFTMoEWrapper`
  - `ARMINT8` -> `GeneralMoEWrapper`, with method normalized to `MOE_INT8`.
- Preserve current production `experts.py` behavior for existing x86 features,
  especially MXFP4 and `swiglu_limit` validation.
- `ArmBF16SFTMoEWrapper` must require native `ARMBF16_SFT_MOE` and must not
  silently run the Python backend if native binding is missing.
- `TorchBF16SFTMoEWrapper` is the correctness oracle. Default placement is CPU;
  `KT_TORCHBF16_SFT_DEVICE=cuda` is only a reference/profiling mode.
- `GeneralMoEWrapper.load_weights_from_tensors()` must validate shapes and keep
  CPU BF16 contiguous tensors alive for native pointer lifetime.

Validation:

```bash
cd /workspace/AsymGEMM-SFT
PYTHONPATH=third_party/ktransformers/kt-kernel python - <<'PY'
from kt_kernel.experts import KTMoEWrapper
print(KTMoEWrapper)
PY
```

### 1.4 Shared SFT Correctness Fixes

Port these files carefully:

```text
kt-kernel/python/sft/base.py
kt-kernel/python/sft/layer.py
kt-kernel/python/sft/lora.py
kt-kernel/python/sft/wrapper.py
```

Required `base.py` behavior:

- Shared SFT buffer reuse must validate all static dimensions, not just `qlen`.
- Add operation lock around forward/backward and async paths.
- Track forward cache qlens.
- Reject backward qlen mismatch.
- Enforce `max_cache_depth`.
- Validate hidden and grad shapes.

Required `layer.py` behavior:

- Keep router under `torch.no_grad()`.
- Protect fused expert LoRA params from `_apply()` device/dtype moves.
- Rebind KT grad buffers before train-time forward, because optimizer or Trainer
  `zero_grad(set_to_none=True)` can clear or replace `.grad`.
- Support fused LoRA containers as `ModuleDict` / `ParameterDict`, not just lists.

Required `lora.py` behavior:

- Create named fused LoRA parameters:
  - `gate_lora_a`
  - `gate_lora_b`
  - `up_lora_a`
  - `up_lora_b`
  - `down_lora_a`
  - `down_lora_b`
- Store `_kt_lora_buffers` and `_kt_lora_grad_buffers`.
- Save fused expert LoRA to `fused_expert_lora.safetensors`.
- Load fused expert LoRA back into the matching named params.
- Return trainable KT LoRA params from all supported containers.

Required `wrapper.py` behavior:

- Extend backend map:
  - `TORCHBF16` -> `TORCHBF16_SFT`
  - `TORCHBF16_SFT` -> `TORCHBF16_SFT`
  - `ARMBF16` -> `ARMBF16_SFT`
  - `ARMBF16_SFT` -> `ARMBF16_SFT`
  - `KT_ARM` -> `ARMBF16_SFT`
- Do not clear `wrapper.gate_proj`, `wrapper.up_proj`, and `wrapper.down_proj`
  for `TORCHBF16_SFT`; the torch backend owns those tensors.
- Keep the no-op fallback for missing `accelerate.utils.dataclasses.KTransformersPlugin`
  so local ARM backends do not require `accelerate-kt`.
- Keep the no-op fallback for missing `transformers.integrations.kt` so local
  ARM backends do not require `transformers-kt`.
- Use `model_revision`, not a stale `revision` attribute, when forwarding model
  revision.
- Move non-expert modules to `cuda:0` when CUDA exists, otherwise CPU.

Validation:

```bash
cd /workspace/AsymGEMM-SFT
PYTHONPATH=third_party/ktransformers/kt-kernel \
python -m pytest \
  third_party/ktransformers/kt-kernel/test/per_commit/test_torchbf16_sft_reference.py \
  third_party/ktransformers/kt-kernel/test/per_commit/test_torchbf16_sft_wrapper_lifecycle.py \
  third_party/ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  third_party/ktransformers/kt-kernel/test/per_commit/test_armint8_inference_reference.py \
  third_party/ktransformers/kt-kernel/test/per_commit/test_basic_cpu.py \
  -q
```

Pass condition: no fallback to x86, no illegal instruction, all tests pass.

### 1.5 Tests And Benchmarks

Port source tests and benchmark scripts only:

```text
kt-kernel/bench/bench_arm_sft_compare.py
kt-kernel/bench/bench_armbf16_sft.py
kt-kernel/bench/bench_armint8_inference.py
kt-kernel/test/per_commit/test_armbf16_sft_reference.py
kt-kernel/test/per_commit/test_armint8_inference_reference.py
kt-kernel/test/per_commit/test_torchbf16_sft_reference.py
kt-kernel/test/per_commit/test_torchbf16_sft_wrapper_lifecycle.py
```

Also merge the small `test_basic_cpu.py` import marker change if still needed.

Do not port generated benchmark JSON or profile output.

## Phase 2: Merge LLaMA-Factory KT ARM Integration

Target tree:

```text
third_party/LlamaFactory
```

Source tree:

```text
third_party/ktransformers-arm/LlamaFactory
```

Do not copy this tree wholesale. Merge only the KT ARM changes while preserving
all current AsymGEMM code.

### 2.1 HParams

Target file:

```text
third_party/LlamaFactory/src/llamafactory/hparams/model_args.py
```

Required changes:

- Keep `AsymGEMMArguments`.
- Keep `ModelArguments(..., KTransformersArguments, AsymGEMMArguments, ...)`.
- Extend `KTransformersArguments` with:
  - `kt_backend`
  - `kt_num_threads`
  - `kt_tp_enabled`
  - `kt_threadpool_count`
  - `kt_num_gpu_experts`
  - `kt_max_cache_depth`
- Add local backend values:
  - `TORCHBF16`
  - `TORCHBF16_SFT`
  - `ARMBF16`
  - `ARMBF16_SFT`
  - `KT_ARM`
- Export those fields through `get_kt_config_dict()`.
- Map them to `ACCELERATE_KT_*` env vars in `apply_kt_config()`.

Validation:

```bash
cd /workspace/AsymGEMM-SFT
PYTHONPATH=third_party/LlamaFactory/src:third_party/ktransformers/kt-kernel \
python - <<'PY'
from llamafactory.hparams import get_train_args
args = {
  "model_name_or_path": "dummy",
  "stage": "sft",
  "do_train": True,
  "finetuning_type": "lora",
  "lora_rank": 8,
  "lora_dropout": 0.0,
  "lora_target": "all",
  "dataset": "dummy",
  "template": "qwen3",
  "cutoff_len": 128,
  "output_dir": "dummy_out",
  "overwrite_output_dir": True,
  "use_kt": True,
  "kt_backend": "TORCHBF16",
  "kt_num_threads": 4,
  "kt_max_cache_depth": 2,
}
model_args, *_ = get_train_args(args)
assert model_args.kt_backend == "TORCHBF16"
print("ok")
PY
```

### 2.2 Parser Guards

Target file:

```text
third_party/LlamaFactory/src/llamafactory/hparams/parser.py
```

Required changes:

- Add `_LOCAL_KT_BACKENDS = {"TORCHBF16", "TORCHBF16_SFT", "ARMBF16", "ARMBF16_SFT", "KT_ARM"}`.
- Add `_is_local_kt_backend(model_args)`.
- Add `_check_local_kt_import()`.
- For local KT backends:
  - require importable `kt_kernel`
  - do not require `transformers-kt`
  - do not require `accelerate-kt`
- For non-local KT backends, keep existing dependency checks.
- Keep `check_version("asym_gemm", mandatory=True)` for `use_asym_gemm`.
- Keep the existing AsymGEMM validation block.
- Keep KT and AsymGEMM mutually exclusive.
- For local KT ARM backends, enforce:
  - stage is SFT
  - training is enabled
  - finetuning type is LoRA
  - `lora_dropout == 0`
  - no DeepSpeed
  - no distributed launch in Phase 1

Validation:

```bash
cd /workspace/AsymGEMM-SFT
PYTHONPATH=third_party/LlamaFactory/src:third_party/ktransformers/kt-kernel \
python -m pytest third_party/ktransformers-arm/LlamaFactory/tests/hparams/test_kt_arm_backend.py -q
```

The test may be copied into `third_party/LlamaFactory/tests/hparams/` before
running from the target tree.

### 2.3 Model Loading

Target file:

```text
third_party/LlamaFactory/src/llamafactory/model/loader.py
```

Required merge:

- Preserve `_use_asym_cpu_first_load()` and `_move_asym_cpu_first_model_to_device()`.
- When `model_args.use_kt`:
  - require trainable SFT path
  - reject `mixture_of_depths`
  - call `kt_kernel.sft.load_kt_model(...)`
  - pass `model_revision`
  - pass `torch_dtype=model_args.compute_dtype or torch.bfloat16`
- After `init_adapter(...)`, when `model_args.use_kt and is_trainable`:
  - call `kt_adapt_peft_lora(model)`
  - if adapter path is present, call `load_kt_moe_from_adapter(model, adapter_path)`
- Preserve AsymGEMM CPU-first move after adapter init.

Validation:

```bash
cd /workspace/AsymGEMM-SFT
PYTHONPATH=third_party/AsymGEMM:third_party/LlamaFactory/src:third_party/ktransformers/kt-kernel \
CUDA_VISIBLE_DEVICES=3 \
third_party/LlamaFactory/.venv/bin/python - <<'PY'
import kt_kernel
from kt_kernel.sft import load_kt_model, kt_adapt_peft_lora
print("kt ok", kt_kernel.__cpu_variant__)
PY
```

### 2.4 Adapter And Fused Expert LoRA Baseline

Target files:

```text
third_party/LlamaFactory/src/llamafactory/model/adapter.py
third_party/LlamaFactory/src/llamafactory/model/model_utils/fused_moe_lora.py
```

Required changes:

- Add `model_utils/fused_moe_lora.py` from isolated LF.
- Preserve the current AsymGEMM branch in `adapter.py`.
- Preserve the current KT branch that creates normal PEFT LoRA.
- Add Qwen expert LoRA only for the normal LF torch baseline path:
  - `finetuning_type == "lora"`
  - `lora_target == ["all"]`
  - not `use_kt`
  - not `use_asym_gemm`
  - not `use_unsloth`
- This keeps the torch GPU baseline apples-to-apples with KT fused expert LoRA
  parameter count for Qwen3 fused experts.

Validation:

- For Qwen3-30B-A3B full LoRA:
  - LF torch GPU row should report `415,236,096` Qwen expert LoRA params.
  - KT rows should report `415,236,096` KT fused expert LoRA params.

### 2.5 Trainer Save Path

Target file:

```text
third_party/LlamaFactory/src/llamafactory/train/sft/trainer.py
```

Required merge:

- Do not replace the current AsymGEMM `_save()` override.
- Extend it so:
  - If `use_asym_gemm`, current AsymGEMM save path runs unchanged.
  - Otherwise call `super()._save(...)`.
  - If `use_kt` and rank 0, unwrap the model and call
    `kt_kernel.sft.save_kt_moe_to_adapter(model, output_dir)`.
- Do not copy the isolated workflow behavior that always calls
  `trainer.save_model()` even when `save_strategy=no`.
- Preserve current `workflow.py` behavior that skips final save/state when
  `save_strategy=no`.

Validation:

- Short save-enabled KT run writes:
  - `adapter_model.safetensors`
  - `adapter_config.json`
  - `fused_expert_lora.safetensors`
- Resume/load test proves `fused_expert_lora.safetensors` is loaded.

### 2.6 Files Not To Merge From Isolated LF

Do not apply isolated diffs that delete current AsymGEMM support:

```text
src/llamafactory/extras/misc.py
src/llamafactory/extras/packages.py
src/llamafactory/launcher.py
src/llamafactory/train/trainer_utils.py
src/llamafactory/v1/plugins/trainer_plugins/distributed/fsdp2.py
```

Only touch these if a targeted KT ARM change is required, and preserve all
existing AsymGEMM behavior.

Do not copy isolated LF datasets into production unless a test explicitly needs a
tiny fixture. The current AsymGEMM dataset builder should remain the normal path.

## Phase 3: Merge AsymGEMM Profiling/Runner Support

Target tree:

```text
third_party/AsymGEMM/scripts/lf
```

Production script entry points:

```text
scripts/lf/run_lf_lora_sft.sh
scripts/lf/profile_lora_lf.sh
```

Do not replace these with the isolated scripts. The current scripts keep the
full profiling, memory attribution, and postprocess support while adding KT as a
sibling backend to AsymGEMM.

Current production behavior required for this entry point:

- `profile_lora_lf.sh` calls `scripts/lf/run_lf_lora_sft.sh` for all
  proof-sweep rows.
- `profile_lora_lf.sh::backend_label()` preserves first-class `torch`,
  `kt_torchbf16`, and `kt_armbf16` row labels.
- `profile_lora_lf.sh::backend_gpu_count()` keeps torch at the model GPU
  count and forces KT backends to one GPU for this phase.
- `profile_lora_lf.sh` selects only public backend labels. The single-run
  runner derives the internal LF/KT enum from `BACKEND`:
  - `kt_torchbf16` -> internal `--kt_backend TORCHBF16`
  - `kt_armbf16` -> internal `--kt_backend ARMBF16`
  - `kt_armint8` stays future/optional for SFT and must not be enabled by the
    default SFT sweep.
- `profile_lora_lf.sh` defaults to
  `$(dirname "${KT_KERNEL_DIR}")/profiling_kt`, so production KT proof artifacts
  live under `third_party/ktransformers/profiling_kt`.
- Loss comparison compares `torch` baseline against `kt_torchbf16` and
  `kt_armbf16` without relabeling KT rows as AsymGEMM rows.
- `run_lf_lora_sft.sh` inserts production `KT_KERNEL_DIR` into `PYTHONPATH`,
  applies ARM OpenMP defaults, and records `KT_TORCHBF16_SFT_DEVICE`.
- `run_lf_profiled_train.py` emits KT wrapper counters and LF/KT fused expert
  LoRA counters.
- `postprocess_lf_profile_artifacts.py` currently does not write
  `kt_counters.csv`, `lora_counters.csv`, or summary rows for KT/Lora proof.

After these script changes, `profile_lora_lf.sh` is the production proof
entry point for KT ARM SFT. It must prove that the normal repo stack
`third_party/ktransformers` + `third_party/LlamaFactory` works; it must not
import from `third_party/ktransformers-arm` except as a source reference while
implementing patches.

### 3.1 Shell Env Wiring

Target file:

```text
third_party/AsymGEMM/scripts/lf/run_lf_lora_sft.sh
```

Required changes:

- Make this script the runner used by `profile_lora_lf.sh` for all
  `kt_*` backend rows.
- Add:

```bash
KT_KERNEL_DIR=${KT_KERNEL_DIR:-/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel}
```

- Include KT kernel source in `PYTHONPATH`:

```bash
KT_TOOLS_DIR=${KT_TOOLS_DIR:-${ROOT}}
PYTHONPATH="${KT_TOOLS_DIR}:${KT_KERNEL_DIR}:${LF_DIR}/src:${PYTHONPATH:-}"
```

- For `BACKEND=kt_torchbf16`, allow:

```bash
KT_TORCHBF16_SFT_DEVICE=${KT_TORCHBF16_SFT_DEVICE:-cuda}
```

and record it into the profile config. Use `cuda` for the apples-to-apples CUDA
reference row and `cpu` only when explicitly running the portable CPU oracle.

- For `BACKEND=kt_armbf16`, add ARM OpenMP defaults:

```bash
OMP_NUM_THREADS=${KT_ARM_OMP_NUM_THREADS:-64}
OMP_PROC_BIND=${KT_ARM_OMP_PROC_BIND:-close}
OMP_PLACES=${KT_ARM_OMP_PLACES:-cores}
```

and record them into the profile config.

- Record `KT_KERNEL_DIR`, the internally resolved KT backend enum,
  `KT_TORCHBF16_SFT_DEVICE`, and the ARM OpenMP settings in both the log and
  source-profile config so later artifacts clearly show what ran on CPU and
  what ran on CUDA.
- Add a preflight import check:

```bash
env "${RUN_ENV[@]}" "${ENV_PYTHON}" - <<'PY'
import kt_kernel
import torch
print("kt_kernel_variant", kt_kernel.__cpu_variant__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY
```

Validation:

```bash
cd /workspace/AsymGEMM-SFT
ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
KT_KERNEL_DIR=/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel \
BACKEND=kt_armbf16 GPU_ID=3 PROFILE=0 MAX_STEPS=0 \
third_party/AsymGEMM/scripts/lf/run_lf_lora_sft.sh
```

The preflight must import the integrated `third_party/ktransformers/kt-kernel`,
not the isolated `third_party/ktransformers-arm` tree.

### 3.2 Source Profile KT Counters

Target file:

```text
third_party/AsymGEMM/scripts/lf/run_lf_profiled_train.py
```

Merge only the KT counter ideas from:

```text
third_party/ktransformers-arm/scripts/lf/kt/run_lf_profiled_train_kt.py
```

Required additions:

- Capture the loaded LF model by wrapping `llamafactory.model.load_model` and
  `llamafactory.train.sft.workflow.load_model`.
- Add `_kt_counters_from_model()`:
  - wrapper count
  - per-layer method
  - per-layer forward calls
  - per-layer backward calls
  - total forward/backward calls
  - LoRA initialization flag
- Add `_lora_counters_from_model()`:
  - trainable params
  - PEFT LoRA params
  - Qwen MoE expert LoRA params
  - KT fused expert LoRA params
- Include `"kt"` and `"lora"` sections in the existing source profile JSON.
- Preserve existing memory attribution, memory breakdown, stage timing, and
  AsymGEMM profile range support.

Do not replace `run_lf_profiled_train.py` with the isolated simpler profiler.

Validation:

After a KT sweep run, `source_profile.json` must contain:

```json
{
  "kt": {
    "wrapper_count": 48,
    "total_forward_calls": 48,
    "total_backward_calls": 48
  },
  "lora": {
    "kt_fused_expert_lora_parameters": 415236096
  }
}
```

For LF torch GPU baseline, `kt.wrapper_count` may be missing/zero, but
`lora.qwen_moe_expert_lora_parameters` should be populated for Qwen3 split expert LoRA.

### 3.3 Postprocess Output

Target file:

```text
third_party/AsymGEMM/scripts/lf/postprocess_lf_profile_artifacts.py
```

Required additions:

- If source profile has `"kt"`, write `kt_counters.csv`.
- If source profile has `"lora"`, write `lora_counters.csv`.
- Add KT/Lora rows to `summary.md`:
  - KT backend
  - KT wrappers
  - KT forward calls
  - KT backward calls
  - Qwen MoE expert LoRA params
  - KT fused expert LoRA params
- Preserve existing latency, memory, loss comparison, memory attribution, and
  memory breakdown outputs.

### 3.4 Sweep Script

Target file:

```text
third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Required changes:

- Keep one canonical sweep runner:

```bash
RUN_LF_SCRIPT="${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh"
```

  Do not add a permanent `run_lf_lora_sft_kt.sh` dependency. KT is selected by
  `BACKEND=kt_torchbf16` or `BACKEND=kt_armbf16`, not by choosing a different
  runner script.

- Extend `backend_label()` to preserve KT rows using only canonical backend
  labels:
  - `torch` -> `torch`
  - `asym` -> `asym`
  - `kt_torchbf16` -> `kt_torchbf16`
  - `kt_armbf16` -> `kt_armbf16`
  - reject aliases such as `kt`, `kt_torch`, `torchbf16`, `kt_arm`,
    `armbf16`, and `*_sft`
- Extend `backend_gpu_count()`:
  - `torch` -> model GPU count
  - `kt_torchbf16` -> `1`
  - `kt_armbf16` -> `1`
- When launching a KT job, pass `BACKEND` and KT-specific envs through to
  `run_lf_lora_sft.sh`; do not expose `KT_BACKEND` as a user/sweep variable.
- `kt_torchbf16` -> internal `--kt_backend TORCHBF16`
- `kt_armbf16` -> internal `--kt_backend ARMBF16`
- `kt_armint8` remains optional/future for SFT; do not enable by default.
- Default GPU pool can remain configurable, but validation commands on this
  machine must use `GPU_POOL=3`.
- Default output root for this KT script must be the KT production repo:
  `third_party/ktransformers/profiling_kt`. KT is a sibling backend to
  AsymGEMM, so production KT profiling artifacts must not be written under
  `third_party/AsymGEMM/profiling_kt`. `third_party/AsymGEMM/scripts/lf/*` is
  only the reusable helper-script location.
- For KT rows, require `LORA_DROPOUT=0.00` in this first production proof path.
- Keep `lora_target all` hardwired through the runner so the same fused expert
  LoRA surface is exercised for torch and KT rows.
- Extend loss-comparison grouping so a single three-way sweep can compare:
  - `torch` baseline vs `kt_torchbf16`
  - `torch` baseline vs `kt_armbf16`
  without renaming KT rows or treating them as AsymGEMM rows.
- Comparison groups must compare only apples-to-apples runs:
  - same model
  - same dataset rows
  - same cutoff length
  - same LoRA rank/alpha/dropout
  - same total measured steps
  - same `lora_target all`

Production proof command after implementation:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT

GPU_POOL=3 \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='torch|norecompute,kt_torchbf16|norecompute,kt_armbf16|norecompute' \
SEQ_LENS=4096 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
MAX_SAMPLES=64 \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.00 \
PROFILE_LEVEL=op \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
ROOT=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory \
KT_KERNEL_DIR=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/ktransformers/kt-kernel \
KT_TORCHBF16_SFT_DEVICE=cuda \
third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Required production proof artifacts:

- `source_profile.json` for every row.
- `summary.md`, `lat.md`, `memory.md`, and `profile.json` for every row.
- `kt_counters.csv` for KT rows, showing 48 wrappers and positive forward /
  backward calls for Qwen3-30B-A3B.
- `lora_counters.csv` for every row, showing matching fused expert LoRA
  parameter counts where applicable.
- Combined comparison table under
  `third_party/ktransformers/profiling_kt/.../combined/` with torch,
  `kt_torchbf16`, and `kt_armbf16` rows.
- Logs proving the imported KT kernel path is
  `third_party/ktransformers/kt-kernel`, not `third_party/ktransformers-arm`.

## Phase 4: Validation Gates

Run these gates in order. Do not move to the next gate until the current one is
clean.

### Gate A: KT Import And Native Binding

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
python -m pip install -v .

cd /workspace/AsymGEMM-SFT
KT_KERNEL_ALLOW_PY_FALLBACK=0 \
PYTHONPATH=third_party/ktransformers/kt-kernel \
python - <<'PY'
import kt_kernel
import kt_kernel_ext
print("variant", kt_kernel.__cpu_variant__)
assert hasattr(kt_kernel_ext.moe, "ARMBF16_SFT_MOE")
assert hasattr(kt_kernel_ext.moe, "Int8_KERNEL_MOE")
PY
```

Pass condition: ARM variant, native ARM bindings present.

### Gate B: KT Unit Tests

```bash
cd /workspace/AsymGEMM-SFT
PYTHONPATH=third_party/ktransformers/kt-kernel \
python -m pytest \
  third_party/ktransformers/kt-kernel/test/per_commit/test_torchbf16_sft_reference.py \
  third_party/ktransformers/kt-kernel/test/per_commit/test_torchbf16_sft_wrapper_lifecycle.py \
  third_party/ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  third_party/ktransformers/kt-kernel/test/per_commit/test_armint8_inference_reference.py \
  third_party/ktransformers/kt-kernel/test/per_commit/test_basic_cpu.py \
  -q
```

Pass condition: all selected tests pass.

### Gate C: LF Local Backend Args

```bash
cd /workspace/AsymGEMM-SFT
PYTHONPATH=third_party/LlamaFactory/src:third_party/ktransformers/kt-kernel \
python -m pytest third_party/LlamaFactory/tests/hparams/test_kt_arm_backend.py -q
```

Pass condition:

- `TORCHBF16`, `ARMBF16`, and `ARMBF16_SFT` are accepted.
- Local KT backends set `ACCELERATE_KT_*`.
- Local KT rejects nonzero LoRA dropout.
- Local KT does not require `transformers-kt` / `accelerate-kt`.

### Gate D: Tiny LF Smoke

Use a small sequence first to catch wiring errors:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
GPU_ID=3 \
BACKEND=kt_armbf16 \
CUTOFF_LEN=64 \
MAX_SAMPLES=4 \
MAX_STEPS=1 \
PROFILE=1 \
scripts/lf/run_lf_lora_sft.sh
```

Pass condition:

- LF loads model through KT path.
- `kt_adapt_peft_lora` runs.
- 48 KT wrappers are reported for Qwen3-30B-A3B.
- Forward/backward KT calls are positive.
- No missing `fused_expert_lora` grad binding.

### Gate E: 4k Three-Way LF SFT Comparison

Run the apples-to-apples 4k comparison:

```bash
cd /workspace/AsymGEMM-SFT
ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
KT_KERNEL_DIR=/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel \
GPU_POOL=3 \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='torch|norecompute,kt_torchbf16|norecompute,kt_armbf16|norecompute' \
SEQ_LENS=4096 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
MAX_SAMPLES=64 \
LORA_RANK=8 \
LORA_ALPHA=16 \
LORA_DROPOUT=0.00 \
PROFILE_LEVEL=op \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

For the CUDA reference row:

```bash
KT_TORCHBF16_SFT_DEVICE=cuda
```

For the CPU oracle row, unset it or set:

```bash
KT_TORCHBF16_SFT_DEVICE=cpu
```

Pass condition:

- LF torch GPU completes.
- LF KT TORCHBF16 completes.
- LF KT ARMBF16 completes.
- All runs use `lora_target all`.
- Expert LoRA param counts match across applicable rows:
  `415,236,096` for Qwen3-30B-A3B with rank 8.
- KT rows report 48 wrappers and positive forward/backward calls.
- Loss deltas are in the same range as isolated proof unless an intentional
  change explains the difference.
- `ARMBF16` peak HBM is much lower than LF torch GPU.
- `ARMBF16` runtime is expected to be slower; do not block the merge on speed as
  long as correctness and HBM behavior are proven.

## Expected Integration Result

After successful merge:

- `third_party/ktransformers` contains ARM-capable KT SFT source.
- `third_party/LlamaFactory` can select local KT ARM backends without
  `transformers-kt` / `accelerate-kt`.
- Existing AsymGEMM LF behavior remains intact.
- `third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh` can produce the same
  kind of 4k KT ARM validation tables that were produced in the isolated tree.
- `third_party/ktransformers-arm` remains useful as a reference/proof tree, but
  the normal integration no longer depends on importing from it.

## Known Risks

1. **Whole-tree LF copy risk**: would delete current AsymGEMM changes. Avoid by
   file-level patching only.
2. **`TORCHBF16_SFT` tensor lifetime risk**: do not clear base tensors for
   `TORCHBF16_SFT`.
3. **LoRA grad rebinding risk**: if Trainer clears `.grad`, KT grad buffers must
   be rebound before each train forward.
4. **Native fallback risk**: `ARMBF16` must fail loudly if native binding is not
   available.
5. **Apples-to-apples risk**: LF torch baseline must include fused expert LoRA
   for Qwen3 fused experts, otherwise memory/loss/param comparisons are
   misleading.
6. **Profile regression risk**: do not replace the current AsymGEMM source
   profiler with the isolated KT profiler; merge only counters.
7. **Performance expectation risk**: current `ARMBF16_SFT` is correctness/HBM
   proof, not a tuned kernel. Optimization is a later phase.

## Implementation Checklist

- [ ] Port KT ARM build/detect changes.
- [ ] Port ARM C++ operator files and bindings.
- [ ] Port `TORCHBF16_SFT` and `ARMBF16_SFT` Python wrappers.
- [ ] Port SFT lifecycle/LoRA correctness fixes.
- [ ] Port KT tests and source-only benchmarks.
- [ ] Extend LF `KTransformersArguments`.
- [ ] Add LF local KT backend dependency guard.
- [ ] Add LF direct KT model loading.
- [ ] Add Qwen MoE LoRA baseline for torch.
- [ ] Add KT sidecar save/load integration while preserving AsymGEMM save path.
- [ ] Add `KT_KERNEL_DIR` and backend placement envs to KT runner.
- [ ] Add KT/Lora counters to current source profiler.
- [ ] Add KT/Lora summary artifacts to postprocess.
- [ ] Run Gate A through Gate E.
- [ ] Record final metrics in `agent/kt/arm_moe_sft_progress.md`.

# FlashAttention-4 LLaMA-Factory LoRA-SFT Integration

Status: design and implementation plan. This document owns dense
FlashAttention-4 enablement for LLaMA-Factory LoRA-SFT only.

## Goal

Enable dense FlashAttention-4 as an explicit LLaMA-Factory SFT attention
backend:

```yaml
flash_attn: fa4
```

or:

```bash
--flash_attn fa4
```

The first deliverable is a validated isolated lab integration in exactly this
checkout:

```text
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4
```

That checkout must be self-contained: its own source tree, its own conda env,
its own FlashAttention-4 install/build, its own datasets, its own outputs, and
its own validation reports. Do not modify the current production LLaMA-Factory
checkout at `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory` and
do not modify the current AsymGEMM source tree while figuring this out. The lab
clone must pass the validation gates in this document before any production
integration is attempted.

Everything outside
`/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4` is read-only
reference material for this project. It is fine to inspect existing
LLaMA-Factory, AsymGEMM, profiling scripts, reports, and prior validation
artifacts to understand behavior. It is not fine to edit, install into, generate
outputs under, or otherwise mutate those paths while doing FA4 lab work. The
only exception is updating this planning document itself.

The target behavior is ordinary LoRA-SFT using Hugging Face / Transformers model
attention dispatch with `config._attn_implementation == "flash_attention_4"`.
This is a dense attention backend replacement only. Sparse attention, BLASST,
TensorRT-LLM serving, FlashInfer serving, KV-cache compression, and decode-only
optimizations are explicitly out of scope.

## Current State

Local LLaMA-Factory state:

- `src/llamafactory/extras/constants.py`
  - `AttentionFunction` has `AUTO`, `DISABLED`, `SDPA`, `FA2`, `FA3`.
  - There is no `FA4 = "fa4"`.
- `src/llamafactory/model/model_utils/attention.py`
  - `sdpa` maps to `requested_attn_implementation = "sdpa"`.
  - `fa2` maps to `requested_attn_implementation = "flash_attention_2"`.
  - `fa3` is only forced for `model_type == "gpt_oss"` through the hub kernel
    `kernels-community/vllm-flash-attn3`.
  - Normal non-`gpt_oss` `fa3` is not implemented by this LF wrapper.
  - There is no `fa4` mapping.
- `src/llamafactory/data/collator.py`
  - The SFT collator type annotation only lists
    `Literal["eager", "sdpa", "flash_attention_2"]`.
  - The neat-packing branch has `# FIXME compatibility fa3/fa4`.
- `README.md`
  - Documents FlashAttention-2 support, not FA4.

Local environment state from previous probes:

- GPU: GB200 / SM100-class.
- PyTorch: CUDA 13 build in the existing LF environment.
- Transformers: `5.6.0`.
- Transformers attention registry already contains:
  - `flash_attention_4`
  - `flash_attention_3`
  - `flash_attention_2`
  - `sdpa`
- Transformers FA4 loader imports:
  - `from flash_attn.cute import flash_attn_func, flash_attn_varlen_func`
- `is_flash_attn_4_available()` is currently false because FA4 is not installed.
- Current `profiling_ccurr_most` traces use PyTorch SDPA / cuDNN native SM100
  flash kernels, not external Dao FlashAttention-4.

Dao FlashAttention current README state:

- FlashAttention-4 is written in CuTeDSL.
- It is optimized for Hopper and Blackwell GPUs.
- Base install command:
  - `pip install flash-attn-4`
- CUDA 13 recommended install command:
  - `pip install "flash-attn-4[cu13]"`

## Source Map

Primary LLaMA-Factory attention paths:

- `src/llamafactory/hparams/model_args.py`
  - `flash_attn: AttentionFunction`
  - Public user argument that should accept `fa4` after integration.
- `src/llamafactory/extras/constants.py`
  - Add `FA4 = "fa4"` to `AttentionFunction`.
- `src/llamafactory/model/model_utils/attention.py`
  - Main LF backend switch.
  - Add availability check and mapping from `AttentionFunction.FA4` to
    Transformers `flash_attention_4`.
  - Add explicit logging for FlashAttention-4.
- `src/llamafactory/data/collator.py`
  - SFT attention-mask and neat-packing behavior.
  - Must not silently use FA2-specific packing logic for FA4 unless verified.
- `src/llamafactory/model/patcher.py`
  - Calls `configure_attn_implementation(config, model_args)`.
  - Has model-specific checks for Qwen / Qwen3.5 with FA2. Audit for any
    accidental assumptions that `flash_attn == "fa2"` means "all flash".
- `src/llamafactory/model/model_utils/longlora.py`
  - Patches `LlamaFlashAttention2.forward`.
  - Treat LongLoRA / shift-attention as separate compatibility work. Do not
    include it in the first FA4 deliverable unless it is already enabled in the
    target run.
- `src/llamafactory/train/sft/workflow.py`
  - Passes `attn_implementation=getattr(model.config, "_attn_implementation",
    None)` into `SFTDataCollatorWith4DAttentionMask`.

Primary Transformers paths in the isolated env:

- `transformers/modeling_utils.py`
  - `ALL_ATTENTION_FUNCTIONS` includes `flash_attention_4`.
  - `get_correct_attn_implementation` validates requested attention backend.
- `transformers/modeling_flash_attention_utils.py`
  - FA4 availability and loader logic.
  - FA4 package check must pass through `is_flash_attn_4_available()`.
- `transformers/integrations/flash_attention.py`
  - Shared flash attention wrapper used by FA2 / FA3 / FA4 registry entries.
  - Transposes Q/K/V, handles dtype, mask, varlen, and calls
    `_flash_attention_forward`.

Primary FlashAttention paths:

- `flash_attn.cute.flash_attn_func`
- `flash_attn.cute.flash_attn_varlen_func`
- Direct package metadata must identify `flash-attn-4`, not only an FA2 package
  with a `flash_attn` module.

Primary AsymGEMM / LF driver paths for later rollout only:

- `scripts/lf/profile_lora_lf.sh`
  - Current production profiling driver.
  - Do not change during lab work.
- `scripts/lf/run_lf_lora_sft.sh`
  - Current production LF SFT launcher.
  - Does not currently pass `--flash_attn`.
  - Do not change during lab work.

## Isolation Rules

All implementation and testing must happen in the separate lab checkout
`/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4` and a
separate conda environment stored under that checkout.

Canonical lab paths:

```bash
LAB_LF=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4
LAB_ROOT=${LAB_LF}
LAB_FA=${LAB_LF}/third_party/flash-attention-fa4
LAB_ENV=${LAB_LF}/.conda-lf-fa4
LAB_CONDA_PKGS=${LAB_LF}/.conda_pkgs
LAB_PIP_CACHE=${LAB_LF}/.pip_cache
LAB_TMP=${LAB_LF}/.tmp
LAB_RUNS=${LAB_LF}/fa4_runs
LAB_REPORTS=${LAB_LF}/fa4_reports
LAB_PROBES=${LAB_LF}/fa4_probes
LAB_DATA=${LAB_LF}/fa4_data
LAB_GPU=2
export CUDA_VISIBLE_DEVICES=${LAB_GPU}
export CONDA_PKGS_DIRS=${LAB_CONDA_PKGS}
export PIP_CACHE_DIR=${LAB_PIP_CACHE}
export TMPDIR=${LAB_TMP}
export HF_HOME=${LAB_LF}/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
```

Required guardrails:

- Do not edit `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory`.
- Do not edit `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM` while
  developing the FA4 LF patch, except for this planning document.
- Treat all files outside `${LAB_LF}` as read-only references. Read them as much
  as needed, but do not write, patch, install, cache, or create outputs there.
- Do not install FA4 into the current LF `.venv`.
- Do not point `PYTHONPATH` at the current LF checkout when running lab tests.
- Set `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`, `TMPDIR`, `HF_HOME`,
  `HF_HUB_CACHE`, and `HF_DATASETS_CACHE` to paths under `${LAB_LF}` before any
  package install, model load, tokenizer load, or dataset load.
- Keep all lab training outputs under `${LAB_RUNS}`.
- Keep all validation reports under `${LAB_REPORTS}`.
- Keep all generated validation datasets under `${LAB_DATA}`.
- Do not write smoke datasets into `${LAB_LF}/data`; that source data tree is a
  reference for examples, not the validation output location.
- Use GPU 2 for lab validation:
  - `export CUDA_VISIBLE_DEVICES=2`
  - Inside the process, this GPU appears as `cuda:0`; logs must record both the
    physical selection (`CUDA_VISIBLE_DEVICES=2`) and the visible device name.
- Before any lab run, print:
  - `which python`
  - `python -c 'import sys; print(sys.executable)'`
  - `python -c 'import transformers; print(transformers.__file__)'`
  - `python -c 'import llamafactory; print(llamafactory.__file__)'`
  - `echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"`
- Every validation report must record the LF git commit, FA git commit or wheel
  version, Transformers version, PyTorch version, CUDA version, GPU name, and
  attention implementation selected by the model config.

Do not copy unvalidated patches into the production LF checkout. The final
rollout must be an explicit patch/cherry-pick step after validation.

## Repository and Script Layout

This section is the canonical lab layout. If later sections mention a path, it
should resolve to one of these locations.

Repos to clone:

```bash
set -euo pipefail

test -d /home/shutianluo/kevin/AsymGEMM-SFT/third_party || \
  { echo "Parent third_party directory is missing. Do not create paths outside LlamaFactory-fa4 in this workflow."; exit 2; }

test ! -e /home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4 || \
  { echo "LlamaFactory-fa4 already exists. Inspect it before continuing."; exit 2; }

git clone https://github.com/hiyouga/LLaMA-Factory.git \
  /home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4

mkdir -p /home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4/third_party

git clone https://github.com/Dao-AILab/flash-attention.git \
  /home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4/third_party/flash-attention-fa4
```

Do not clone or edit a second AsymGEMM tree for this task. The current
AsymGEMM repo is only the place that stores this plan. The FA4 implementation
and all validation scripts live inside `LlamaFactory-fa4`.

Required lab tree after setup:

```text
/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4/
  .conda-lf-fa4/                         # isolated conda env
  .conda_pkgs/                           # lab-local conda package cache
  .hf_cache/                             # lab-local HF model/tokenizer/dataset cache
  .pip_cache/                            # lab-local pip cache
  .tmp/                                  # lab-local build/temp files
  fa4_probes/                            # validation scripts, created by us
    fa4_env_probe.py
    fa4_direct_probe.py
    transformers_fa4_probe.py
    prepare_fa4_smoke_dataset.py
    validate_lf_fa4.py
  fa4_reports/                           # JSON/MD reports and raw logs
  fa4_data/                              # generated smoke dataset + dataset_info.json
  fa4_runs/                              # LF training output dirs
  third_party/
    flash-attention-fa4/                 # Dao-AILab/flash-attention clone
  src/llamafactory/                      # lab LF source to patch
```

Validation scripts to prepare:

- `${LAB_PROBES}/fa4_env_probe.py`
  - Verifies isolation and environment only.
  - Fails if `llamafactory.__file__` is not under `${LAB_LF}`.
  - Fails if `transformers.__file__` is not under `${LAB_ENV}`.
  - Fails if `CUDA_VISIBLE_DEVICES` is not exactly `2`.
  - Fails if any package, temp, HF, or datasets cache env var resolves outside
    `${LAB_LF}`.
  - Fails if `is_flash_attn_4_available()` is false.
  - Fails if `from flash_attn.cute import flash_attn_func` does not work.
- `${LAB_PROBES}/fa4_direct_probe.py`
  - Calls FA4 functions directly.
  - Tests BF16 forward/backward, no-padding and varlen paths.
  - Compares outputs roughly against SDPA and checks finite gradients.
- `${LAB_PROBES}/transformers_fa4_probe.py`
  - Uses a tiny in-memory Transformers model/config.
  - Forces `attn_implementation="flash_attention_4"`.
  - Runs forward/backward with labels.
  - Applies PEFT LoRA and checks finite nonzero LoRA gradients.
- `${LAB_PROBES}/prepare_fa4_smoke_dataset.py`
  - Writes a tiny SFT dataset into `${LAB_DATA}`.
  - Writes a lab-local `${LAB_DATA}/dataset_info.json`.
  - Does not write anything to the production LF data directory or the lab LF
    source `${LAB_LF}/data` directory.
- `${LAB_PROBES}/validate_lf_fa4.py`
  - Orchestrates every validation step.
  - Runs `fa4_env_probe.py` first.
  - Creates the smoke dataset.
  - Runs the LF CLI smoke with `flash_attn: fa4`.
  - Runs the paired `sdpa` smoke when `--run-sdpa-pair` is provided; this gate
    is required before rollout.
  - Runs Nsight or runtime-hook verification when `--run-nsys` is provided;
    one of those dispatch-evidence paths is required before rollout.
  - Writes `report.json`, `report.md`, logs, and environment manifests under
    `${LAB_REPORTS}`.

No validation script may import helper code from the current production
LLaMA-Factory checkout. No validation script may write under production
`/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory`.

## Backend Naming and Semantics

Add exactly one new LF public attention value:

```python
FA4 = "fa4"
```

Map it to exactly one Transformers implementation:

```python
requested_attn_implementation = "flash_attention_4"
```

Do not overload `fa2`, `fa3`, or `auto`.

`auto` should remain conservative. The first implementation should not make
`auto` choose FA4 automatically. Requiring explicit `flash_attn: fa4` prevents
surprising baseline changes in existing LF runs.

Fallback policy:

- `flash_attn: sdpa`
  - Use PyTorch SDPA if supported.
- `flash_attn: fa2`
  - Existing behavior.
- `flash_attn: fa4`
  - If `is_flash_attn_4_available()` is false, fail or warn-and-return in a way
    that the validation script treats as failure.
  - Do not silently fall back to SDPA for a claimed FA4 validation run.

Preferred behavior for lab validation is hard failure:

```text
FlashAttention-4 requested but flash-attn-4 is not installed or unavailable.
```

If LF upstream style prefers warning and return, the validation script must
detect that `_attn_implementation != "flash_attention_4"` and fail the run.

## Implementation Phases

### Phase 0: Lab Checkout and Environment

Start every shell for this work by defining the canonical lab variables:

```bash
set -euo pipefail

LAB_LF=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4
LAB_ROOT=${LAB_LF}
LAB_FA=${LAB_LF}/third_party/flash-attention-fa4
LAB_ENV=${LAB_LF}/.conda-lf-fa4
LAB_CONDA_PKGS=${LAB_LF}/.conda_pkgs
LAB_PIP_CACHE=${LAB_LF}/.pip_cache
LAB_TMP=${LAB_LF}/.tmp
LAB_RUNS=${LAB_LF}/fa4_runs
LAB_REPORTS=${LAB_LF}/fa4_reports
LAB_PROBES=${LAB_LF}/fa4_probes
LAB_DATA=${LAB_LF}/fa4_data
LAB_GPU=2
export CUDA_VISIBLE_DEVICES=${LAB_GPU}
export CONDA_PKGS_DIRS=${LAB_CONDA_PKGS}
export PIP_CACHE_DIR=${LAB_PIP_CACHE}
export TMPDIR=${LAB_TMP}
export HF_HOME=${LAB_LF}/.hf_cache
export HF_HUB_CACHE=${HF_HOME}/hub
export HF_DATASETS_CACHE=${HF_HOME}/datasets
```

Create the isolated workspace:

```bash
LAB_PARENT=$(dirname "${LAB_LF}")
test -d "${LAB_PARENT}" || { echo "Parent directory is missing: ${LAB_PARENT}. Stop instead of creating paths outside LAB_LF."; exit 2; }
test ! -e "${LAB_LF}" || { echo "LAB_LF already exists: ${LAB_LF}. Inspect it before continuing."; exit 2; }
git clone https://github.com/hiyouga/LLaMA-Factory.git "${LAB_LF}"
mkdir -p \
  "${LAB_LF}/third_party" \
  "${LAB_RUNS}" \
  "${LAB_REPORTS}" \
  "${LAB_PROBES}" \
  "${LAB_DATA}" \
  "${LAB_CONDA_PKGS}" \
  "${LAB_PIP_CACHE}" \
  "${LAB_TMP}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}"
git clone https://github.com/Dao-AILab/flash-attention.git "${LAB_FA}"
cat >> "${LAB_LF}/.git/info/exclude" <<'EOF'
.conda-lf-fa4/
.conda_pkgs/
.hf_cache/
.pip_cache/
.tmp/
fa4_data/
fa4_runs/
fa4_reports/
third_party/flash-attention-fa4/
EOF
```

Create a separate conda environment:

```bash
conda create -y -p "${LAB_ENV}" python=3.11
conda run -p "${LAB_ENV}" python -m pip install -U pip setuptools wheel
```

Install PyTorch and base LF dependencies in the lab env. Prefer matching the
known-good CUDA 13 / PyTorch family from the current environment. Record exact
versions in the validation report.

Install LLaMA-Factory editable from the lab checkout:

```bash
cd "${LAB_LF}"
conda run -p "${LAB_ENV}" python -m pip install -e ".[torch,metrics]"
```

Install FlashAttention-4:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -p "${LAB_ENV}" python -m pip install "flash-attn-4[cu13]"
```

If the wheel path fails, use the cloned FlashAttention source only inside the
lab:

```bash
cd "${LAB_FA}"
conda run -p "${LAB_ENV}" python -m pip install packaging psutil ninja
CUDA_VISIBLE_DEVICES=2 MAX_JOBS=16 conda run -p "${LAB_ENV}" python -m pip install -v --no-build-isolation .
```

The source-install fallback is acceptable only if the Phase 0 validation still
reports `is_flash_attn_4_available() == True` and package metadata identifies
the installed distribution as `flash-attn-4`. If source install produces only
FA2-style package metadata, stop and fix the FA4 installation instead of
continuing.

Phase 0 validation:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -p "${LAB_ENV}" python - <<'PY'
import importlib.metadata as md
import importlib.util
import os
import torch
import transformers
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import is_flash_attn_4_available

print("python ok")
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__, transformers.__file__)
print("attn keys", list(ALL_ATTENTION_FUNCTIONS.valid_keys()))
print("flash_attn spec", importlib.util.find_spec("flash_attn"))
print("flash-attn-4 available", is_flash_attn_4_available())
print("flash-attn dist", md.version("flash-attn-4"))
from flash_attn.cute import flash_attn_func, flash_attn_varlen_func
print("fa4 funcs", flash_attn_func, flash_attn_varlen_func)
PY
```

Gate:

- `is_flash_attn_4_available()` must be true.
- `flash_attn.cute.flash_attn_func` must import.
- GPU capability must be Hopper-or-newer, with GB200/SM100 preferred for the
  target runs.

### Phase 1: Direct FA4 Kernel Probe

Before touching LF, prove the FA4 package works in isolation.

Create a lab-only probe, for example:

```text
${LAB_PROBES}/fa4_direct_probe.py
```

Required checks:

- No-padding forward:
  - Q/K/V shape: `[batch, seqlen, heads, head_dim]`.
  - BF16 inputs on CUDA.
  - Causal attention.
  - Compare to `torch.nn.functional.scaled_dot_product_attention`.
- No-padding backward:
  - `loss = out.float().square().mean()`.
  - Backward must produce finite gradients for Q/K/V.
- Varlen forward/backward:
  - Use `flash_attn_varlen_func`.
  - Include at least two sequences with different lengths.
  - Compare each sequence against SDPA reference.
- Head-dim coverage:
  - At minimum test `head_dim=64` and `head_dim=128`.
  - If target model uses another head dim, add it.
- GQA/MQA shape coverage:
  - Test query heads greater than KV heads if FA4 API supports it through the
    selected wrapper.
  - If direct FA4 API cannot express a shape used by target LF models, stop and
    resolve before LF integration.

Initial tolerance policy:

- BF16 forward max absolute difference should be recorded, not only asserted.
- Use a relaxed gate first, for example `rtol=5e-2`, `atol=5e-2`, then tighten
  after observing real differences.
- Backward comparison should check finite gradients and rough agreement to SDPA.
  Exact bitwise agreement is not expected.

Gate:

- Probe exits 0.
- No NaN/Inf in outputs or gradients.
- Difference report is saved to `${LAB_REPORTS}/fa4_direct_probe.json`.

### Phase 2: Transformers-Level Dense FA4 Probe

Before changing LF, prove local Transformers can dispatch a model attention
module through FA4.

Create a lab-only probe:

```text
${LAB_PROBES}/transformers_fa4_probe.py
```

Required checks:

- Instantiate a tiny in-memory model config, no large download required.
- Prefer a Qwen3-style config because target LF runs include Qwen3 MoE and Qwen3
  attention inherits the modern Transformers attention interface.
- Also test a Llama-style config if target rollout includes Llama-4 Scout.
- Force:

```python
attn_implementation="flash_attention_4"
```

or set:

```python
config._attn_implementation = "flash_attention_4"
```

depending on the exact model constructor path.

Required assertions:

- `model.config._attn_implementation == "flash_attention_4"`.
- `ALL_ATTENTION_FUNCTIONS.get_interface("flash_attention_4", ...)` is selected
  by the attention layer.
- Forward works with `labels=input_ids`.
- Backward works.
- Gradients are finite.
- A small LoRA-wrapped version also works:
  - Apply PEFT LoRA to attention and MLP linear modules.
  - Run one forward/backward.
  - Assert at least one LoRA parameter has finite nonzero grad.
  - Assert frozen base parameters remain frozen if configured that way.

Early kernel check before Phase 6:

- Run under `nsys` or PyTorch profiler.
- Confirm FA4/CuTe/flash kernels appear.
- Confirm the run is not using `native_sdpa_sm100` as the attention kernel.
This check can be skipped only if Phase 6 runtime-hook or Nsight verification is
still planned and remains a required gate before rollout.

Gate:

- Dense Transformers FA4 dispatch works before LF changes start.

### Phase 3: LLaMA-Factory FA4 Backend Plumbing

All edits in this phase are made only in `${LAB_LF}`.

Implementation steps:

1. Add enum value:

```python
class AttentionFunction(StrEnum):
    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"
    FA3 = "fa3"
    FA4 = "fa4"
```

2. Update `configure_attn_implementation`.

Add import:

```python
from transformers.utils import is_flash_attn_2_available, is_flash_attn_4_available
```

Add mapping:

```python
elif model_args.flash_attn == AttentionFunction.FA4:
    if not is_flash_attn_4_available():
        raise ImportError("FlashAttention-4 is requested but flash-attn-4 is not available.")
    requested_attn_implementation = "flash_attention_4"
```

If using LF's warning style instead of raising, the validation script must fail
unless `_attn_implementation` is exactly `"flash_attention_4"`.

3. Update logging:

```python
elif attn_implementation == "flash_attention_4":
    logger.info_rank0("Using FlashAttention-4 for faster training and inference.")
```

4. Update collator typing and packing guard.

First milestone:

- Add `"flash_attention_4"` to the `attn_implementation` type annotation.
- Do not treat FA4 as FA2 for neat-packing until explicitly validated.
- If `neat_packing` is enabled with FA4, prefer an explicit error in the lab
  branch:

```text
Neat packing with FlashAttention-4 has not been validated yet.
```

Rationale:

- The target profiling runs do not require neat packing.
- Silent reuse of the FA2 branch is the highest-risk correctness mistake.

Later milestone after the first FA4 smoke passes:

- Validate `flash_attention_4` with `neat_packing`.
- If correct, generalize the FA2 branch to a set:

```python
FLASH_ATTN_IMPLS = {"flash_attention_2", "flash_attention_4"}
```

5. Update documentation and examples in the lab checkout only.

Add a short example YAML:

```yaml
flash_attn: fa4
```

Document requirements:

- `flash-attn-4` installed.
- CUDA-capable Hopper/Blackwell GPU.
- For this project, CUDA 13 + GB200 is the target.

6. Do not change `auto`.

The user should have to request `fa4` explicitly.

Phase 3 local validation:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -p "${LAB_ENV}" python - <<'PY'
from llamafactory.extras.constants import AttentionFunction
print([x.value for x in AttentionFunction])
assert AttentionFunction.FA4.value == "fa4"
PY
```

### Phase 3.5: Smoke Dataset Preparation

Data decision:

- Use a generated, deterministic, lab-local Alpaca-format SFT dataset for FA4
  validation.
- This dataset is only for backend, preprocessing, and LoRA-gradient smoke
  validation. It is not a quality benchmark and must not be used to claim model
  accuracy or production behavior.
- Do not download an external training dataset for the first smoke. External
  model/tokenizer downloads are allowed only with `HF_HOME`, `HF_HUB_CACHE`, and
  `HF_DATASETS_CACHE` under `${LAB_LF}`.
- LF supports arbitrary `dataset_dir` directories that contain their own
  `dataset_info.json`. Therefore the validation dataset lives in `${LAB_DATA}`,
  not `${LAB_LF}/data`.

Required generated files:

```text
${LAB_DATA}/dataset_info.json
${LAB_DATA}/fa4_smoke.jsonl
${LAB_DATA}/fa4_smoke_eval.jsonl          # optional; not used by the first smoke
${LAB_REPORTS}/data/fa4_smoke_manifest.json
${LAB_REPORTS}/data/fa4_smoke_token_stats.json
${LAB_REPORTS}/data/fa4_smoke_preview.json
```

Required `${LAB_DATA}/dataset_info.json` entry:

```json
{
  "fa4_smoke": {
    "file_name": "fa4_smoke.jsonl",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

Required row schema for `${LAB_DATA}/fa4_smoke.jsonl`:

```json
{"instruction": "Summarize the following note.", "input": "A short deterministic note.", "output": "The note is short and deterministic."}
```

Rules for the generated rows:

- Generate 8 rows by default.
- Every row must have string fields `instruction`, `input`, and `output`.
- `instruction` and `output` must be non-empty after stripping whitespace.
- Use text only. No tools, images, videos, audios, system prompts, or
  ShareGPT/OpenAI message formats for the first smoke.
- Include a mix of short and medium rows plus one near-cutoff row after LF
  template tokenization. For `cutoff_len=256`, near-cutoff means 192 to 240
  tokens after template tokenization.
- For `cutoff_len=256`, short means `<= 64` tokens and medium means 96 to 160
  tokens after template tokenization.
- Because the one-step LF smoke uses `max_samples: 4`, the first four rows must
  include at least one short row, one medium row, and the near-cutoff row.
- Do not rely on truncation for the first validation dataset. If any generated
  example would be truncated by LF preprocessing, fail data preparation and
  regenerate shorter text.
- Keep the content deterministic and seed-independent so SDPA and FA4 runs use
  identical data.

Required data-prep command:

```bash
CUDA_VISIBLE_DEVICES=2 "${LAB_ENV}/bin/python" "${LAB_PROBES}/prepare_fa4_smoke_dataset.py" \
  --lf-dir "${LAB_LF}" \
  --dataset-dir "${LAB_DATA}" \
  --report-dir "${LAB_REPORTS}/data" \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --template qwen3_nothink \
  --dataset-name fa4_smoke \
  --rows 8 \
  --cutoff-len 256 \
  --overwrite
```

`prepare_fa4_smoke_dataset.py` must:

- Resolve all paths with `realpath`.
- Fail unless `--lf-dir` is exactly `${LAB_LF}`.
- Fail unless `--dataset-dir` is under `${LAB_LF}` and its basename is
  `fa4_data`.
- Fail unless `--report-dir` is under `${LAB_LF}`.
- Fail if the current production LF path appears in `sys.path`.
- Import `llamafactory` from `${LAB_LF}` only.
- Write only under `--dataset-dir` and `--report-dir`.
- Write `dataset_info.json` atomically.
- Write JSONL with sorted keys and a trailing newline.
- Compute and report SHA256 for `dataset_info.json` and `fa4_smoke.jsonl`.
- Load the tokenizer/template with the lab LF package and `qwen3_nothink`.
- Run an LF preprocessing dry-run through the lab LF data path, preferably using
  `llamafactory.data.loader.get_dataset`.
- Assert the preprocessed train dataset has exactly the requested number of
  valid samples.
- Assert every sample has `input_ids`, `attention_mask`, and `labels`.
- Assert every sample length is `<= cutoff_len`.
- Assert every sample has at least one label token not equal to `IGNORE_INDEX`.
- Assert the first four rows contain short, medium, and near-cutoff token-length
  cases.
- Assert no sample is empty and no row was dropped by the SFT processor.
- Save min/max/mean token lengths and non-ignored label-token counts.

Efficiency settings for the smoke:

- `streaming: false`
- `packing: false`
- `neat_packing: false`
- `overwrite_cache: true`
- `preprocessing_num_workers: 1`
- `preprocessing_batch_size: 8`
- `dataloader_num_workers: 0`
- `max_samples: 4` for the one-step CLI smoke, even though the generated file
  contains 8 rows.

Fairness rule for SDPA vs FA4 pairs:

- Generate the dataset once.
- Record file hashes once.
- Use the same `${LAB_DATA}/dataset_info.json` and
  `${LAB_DATA}/fa4_smoke.jsonl` for both backends.
- Do not rewrite the data between the SDPA and FA4 pair unless hashes are
  recorded again and confirmed identical.

### Phase 4: LLaMA-Factory CLI Smoke Validation

Create a lab validation script:

```text
${LAB_PROBES}/validate_lf_fa4.py
```

The script should be self-contained and should not import from the production LF
checkout.

Required CLI:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -p "${LAB_ENV}" python "${LAB_PROBES}/validate_lf_fa4.py" \
  --lf-dir "${LAB_LF}" \
  --env-python "${LAB_ENV}/bin/python" \
  --output-dir "${LAB_REPORTS}/validate_lf_fa4" \
  --model-name-or-path Qwen/Qwen3-0.6B \
  --dataset-name fa4_smoke \
  --dataset-dir "${LAB_DATA}" \
  --cutoff-len 256 \
  --max-steps 1 \
  --run-env-probe \
  --run-direct-probe \
  --run-transformers-probe \
  --run-lf-smoke
```

Validation script responsibilities:

- Verify isolation:
  - `llamafactory.__file__` must be under `${LAB_LF}`.
  - `transformers.__file__` must be under `${LAB_ENV}`.
  - Current production LF path must not appear in `sys.path`.
- Verify environment:
  - `is_flash_attn_4_available() == True`.
  - `flash_attn.cute` imports.
  - GPU is available.
  - `CONDA_PKGS_DIRS`, `PIP_CACHE_DIR`, `TMPDIR`, `HF_HOME`,
    `HF_HUB_CACHE`, and `HF_DATASETS_CACHE` all resolve under `${LAB_LF}`.
- Verify LF parser accepts `flash_attn: fa4`.
- Run direct FA4 probe from Phase 1.
- Run Transformers FA4 probe from Phase 2.
- Prepare and validate the smoke dataset from Phase 3.5.
- Run LF CLI smoke.
- Save:
  - `report.json`
  - `report.md`
  - raw train logs
  - environment manifest
  - smoke dataset hashes and preprocessing stats

LF CLI smoke model progression:

1. Tiny dry parser smoke:
   - Use a tiny model or no-train parser path if available.
   - Goal: prove LF accepts the new arg and builds config.

2. Small real dense model smoke:
   - Candidate: `Qwen/Qwen3-0.6B` or another small Qwen3 dense model.
   - LoRA rank: 4 or 8.
   - `max_steps: 1`
   - `cutoff_len: 128` or `256`
   - `max_samples: 4`
   - `lora_dropout: 0.0`
   - `flash_attn: fa4`
   - `neat_packing: false`

3. Target-family short smoke:
   - Qwen3 MoE or Llama-4-family if available and feasible.
   - Keep sequence length low first, for example `cutoff_len: 512`.
   - One training step is enough for backend validation.

Required LF smoke assertions:

- Log contains the explicit FA4 message.
- `model.config._attn_implementation` is recorded as `flash_attention_4`.
- First forward/backward completes.
- Loss is finite.
- At least one LoRA parameter has finite nonzero gradient.
- No base parameter unexpectedly becomes trainable.
- CUDA peak memory is recorded.
- The run does not log "Using torch SDPA" or "Using vanilla attention" for the
  FA4 smoke.

LF smoke YAML template:

The validation script must render `${LAB_DATA}`, `${LAB_RUNS}`, and
`${HF_HUB_CACHE}` into absolute paths before writing the YAML file. Do not pass
literal `${...}` strings to LLaMA-Factory.

```yaml
model_name_or_path: Qwen/Qwen3-0.6B
cache_dir: ${HF_HUB_CACHE}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 4
lora_alpha: 8
lora_dropout: 0.0
lora_target: all
dataset: fa4_smoke
dataset_dir: ${LAB_DATA}
template: qwen3_nothink
cutoff_len: 256
max_samples: 4
overwrite_cache: true
preprocessing_num_workers: 1
preprocessing_batch_size: 8
output_dir: ${LAB_RUNS}/qwen3_0p6b_fa4_smoke
logging_steps: 1
seed: 42
data_seed: 42
save_strategy: "no"
eval_strategy: "no"
report_to: none
overwrite_output_dir: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
dataloader_num_workers: 0
learning_rate: 1.0e-4
max_steps: 1
bf16: true
pure_bf16: true
gradient_checkpointing: false
flash_attn: fa4
packing: false
neat_packing: false
```

Create and validate the tiny dataset inside `${LAB_DATA}` before this training
YAML is launched. Do not create it in production LF data or in `${LAB_LF}/data`.

### Phase 5: Correctness and Parity Validation

After FA4 LF smoke passes, run paired SDPA vs FA4 validation.

Run pairs:

- Same LF lab checkout.
- Same conda env.
- Same model.
- Same dataset.
- Same seed.
- Same LoRA config.
- Same batch and cutoff length.
- `lora_dropout: 0.0`.
- One run with `flash_attn: sdpa`.
- One run with `flash_attn: fa4`.

Record:

- First-step loss.
- Final loss after a small fixed number of steps, for example 3 to 5.
- LoRA grad norms.
- CUDA peak allocated.
- CUDA peak reserved.
- Step time.
- Attention implementation in config.

Acceptance gates:

- Both losses finite.
- FA4 loss is numerically close enough for BF16 attention backend differences.
  Start with relative tolerance `<= 2e-2` for first-step loss, then tighten if
  observed differences are smaller.
- LoRA grad norms are finite and nonzero.
- No unexpected trainable base params.
- FA4 peak memory and time are recorded. They do not need to beat SDPA for
  correctness gate, but regressions must be documented.

Do not claim memory savings from FA4 as our method. FA4 comparison is only to
understand the attention backend denominator for later AsymGEMM profiling.

### Phase 6: Nsight Kernel Verification

Run one FA4 smoke under Nsight Systems if available. If Nsight is not available,
the runtime-hook verification described below is required and must be treated as
the dispatch evidence gate.

Required outputs:

- `${LAB_REPORTS}/nsys/trace.sqlite`
- `${LAB_REPORTS}/nsys/kernel_summary.txt`
- kernel-name evidence in `${LAB_REPORTS}/validate_lf_fa4/report.json`

Validation logic:

- Positive evidence:
  - FA4/CuTe/flash-attention kernel names appear in the trace, or the loaded
    implementation is otherwise attributable to `flash_attn.cute`.
- Negative evidence:
  - The FA4 run should not have attention dominated by
    `cudnn_generated_fort_native_sdpa_sm100_flash_*`.

The exact FA4 kernel names may differ by package version. The parser should
therefore report raw matching kernel names instead of relying on one hard-coded
substring. Kernel search terms:

```text
flash
attention
cute
fa4
flash_attn
```

If the trace is ambiguous, use a stronger Python-side runtime hook:

- Monkeypatch or wrap `transformers.modeling_flash_attention_utils.lazy_import_flash_attention`
  in the validation script to record the loaded implementation.
- Assert it was called with `flash_attention_4`.

### Phase 7: Target LF LoRA-SFT Profile Smoke

Only after phases 0 through 6 pass, run a small target-family SFT smoke in the
lab clone.

Target runs:

- Qwen3 MoE short smoke.
- Llama-4 Scout short smoke, if access and memory permit.

Initial settings:

- `cutoff_len: 512`
- `max_steps: 1`
- `max_samples: 4`
- `lora_rank: 8`
- `lora_dropout: 0.0`
- `flash_attn: fa4`
- `neat_packing: false`
- No AsymGEMM at first.

Then run SDPA pair for comparison.

Gate:

- FA4 target-family run completes.
- Loss finite.
- LoRA grads finite.
- Attention implementation recorded as FA4.
- Kernel evidence captured or runtime hook confirms FA4.

Only after this passes should we consider combining FA4 with AsymGEMM in the
production research path.

### Phase 8: Production Rollout Plan

Rollout into the current LF checkout happens only after lab validation.
This phase is a future procedure, not part of the isolated FA4 lab work. Do not
execute any Phase 8 command or write to the production LF checkout unless the
user explicitly asks for production rollout after the lab validation report has
passed.

Recommended rollout procedure:

1. Create a branch in the production LF checkout:

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory
git status --short
git checkout -b fa4-lf-sft
```

2. Apply the minimal validated patch from `${LAB_LF}`.

3. Do not copy lab env files or training outputs.

4. Re-run the validation script against production LF, still using a separate
   FA4 environment.

5. Only after production LF validation passes, update AsymGEMM profiling scripts
   to make FA4 optional.

Production profiling script behavior after rollout:

```bash
FLASH_ATTN=${FLASH_ATTN:-auto}
```

and only pass:

```bash
--flash_attn "${FLASH_ATTN}"
```

when explicitly set.

Do not make FA4 the default in the production profiling scripts.

## Validation Script Contract

The final validation script should support:

```bash
CUDA_VISIBLE_DEVICES=2 "${LAB_ENV}/bin/python" "${LAB_PROBES}/validate_lf_fa4.py" \
  --lf-dir "${LAB_LF}" \
  --env-python "${LAB_ENV}/bin/python" \
  --output-dir "${LAB_REPORTS}/validate_lf_fa4" \
  --model-name-or-path MODEL \
  --dataset-name NAME \
  --dataset-dir "${LAB_DATA}" \
  --cutoff-len 256 \
  --max-steps 1 \
  --run-env-probe \
  --run-direct-probe \
  --run-transformers-probe \
  --run-lf-smoke \
  --run-sdpa-pair \
  --run-nsys
```

Required JSON fields:

```json
{
  "status": "pass|fail",
  "env": {
    "python": "...",
    "torch": "...",
    "cuda": "...",
    "cuda_visible_devices": "2",
    "gpu": "...",
    "transformers": "...",
    "llamafactory_path": "...",
    "conda_pkgs_dirs": "...",
    "pip_cache_dir": "...",
    "tmpdir": "...",
    "hf_home": "...",
    "hf_hub_cache": "...",
    "hf_datasets_cache": "...",
    "flash_attn_4_available": true
  },
  "env_probe": {
    "status": "pass|fail",
    "isolation_ok": true,
    "cuda_visible_devices": "2",
    "flash_attn_cute_import_ok": true
  },
  "direct_probe": {
    "status": "pass|fail",
    "forward_max_abs_diff": 0.0,
    "backward_grad_finite": true
  },
  "transformers_probe": {
    "status": "pass|fail",
    "attn_implementation": "flash_attention_4",
    "loss_finite": true,
    "lora_grad_nonzero": true
  },
  "data_prep": {
    "status": "pass|fail",
    "dataset_dir": "...",
    "dataset_info_path": "...",
    "dataset_jsonl_path": "...",
    "dataset_info_sha256": "...",
    "dataset_jsonl_sha256": "...",
    "template": "qwen3_nothink",
    "cutoff_len": 256,
    "num_rows": 8,
    "num_valid_preprocessed_samples": 8,
    "max_input_tokens": 0,
    "min_non_ignore_label_tokens": 0,
    "first_four_length_coverage_ok": true,
    "no_truncation": true
  },
  "lf_smoke": {
    "status": "pass|fail",
    "attn_implementation": "flash_attention_4",
    "loss": 0.0,
    "peak_allocated_mib": 0.0,
    "log_path": "..."
  },
  "sdpa_pair": {
    "enabled": true,
    "sdpa_loss": 0.0,
    "fa4_loss": 0.0,
    "relative_loss_diff": 0.0
  },
  "kernel_evidence": {
    "enabled": true,
    "trace_path": "...",
    "matched_kernel_names": []
  }
}
```

The script should exit nonzero if any required gate fails.

## Fallback Policy

For `flash_attn: fa4`, failure is better than silent fallback.

Do not accept a run as FA4 if any of these are true:

- `is_flash_attn_4_available()` is false.
- `flash_attn.cute` cannot import.
- `model.config._attn_implementation` is missing or not `flash_attention_4`.
- LF logs `Using torch SDPA`.
- LF logs `Using vanilla attention implementation`.
- Nsight/runtime hook shows attention did not route through FA4.

For production profiling, keep `auto` as the default and add FA4 only as an
explicit opt-in after validation.

## Non-Goals

- No sparse attention.
- No BLASST integration.
- No TensorRT-LLM or FlashInfer serving integration.
- No FA4 decode/KV-cache optimization work.
- No FA2/FA3 cleanup unless required to keep the FA4 patch coherent.
- No KTransformers integration changes.
- No external training dataset download for the first FA4 smoke.
- No edits to current production LF until lab validation is complete.
- No changes to current production LF `.venv`.
- No change to AsymGEMM kernels.
- No claim that FA4 memory savings are our memory savings.

## Risks

- FA4 package availability may depend on exact Python, CUDA, PyTorch, and GPU
  versions.
- `flash-attn-4[cu13]` may install a different dependency set than current LF
  expects.
- Transformers may expose `flash_attention_4`, but a specific model class may
  still reject or fallback depending on `_supports_flash_attn` and compatible
  implementation checks.
- Packed / neat-packing SFT masks are not automatically safe for FA4. Treat this
  as a separate validation gate.
- LongLoRA / shift attention is not part of the first deliverable.
- BF16 numerical differences can make strict loss parity too tight. Use finite
  gradient and reasonable loss tolerance gates.
- Nsight kernel names may be version-specific. Runtime hook evidence should be
  available as a backup.
- FA4 may not improve memory over PyTorch SDPA/cuDNN flash on GB200 for the
  current workload. Correct backend enablement is the first goal.

## Definition of Done

The FA4 LF LoRA-SFT integration is done when all of the following are true:

- A separate lab LF clone exists and the production LF checkout is untouched.
- A separate conda env exists and the production LF `.venv` is untouched.
- Conda, pip, temp, HF hub, and HF datasets caches all resolve under
  `${LAB_LF}` during validation.
- `flash-attn-4` is installed and `is_flash_attn_4_available()` is true.
- Direct FA4 forward/backward probes pass.
- Transformers-level FA4 model forward/backward probes pass.
- The lab smoke dataset is generated under `${LAB_DATA}`, hashed, and validated
  through LF preprocessing with non-empty labels and no unintended truncation.
- LF accepts `flash_attn: fa4`.
- LF sets `model.config._attn_implementation == "flash_attention_4"`.
- LF LoRA-SFT one-step smoke passes with finite loss and finite nonzero LoRA
  gradients.
- SDPA vs FA4 paired smoke report is generated.
- Kernel or runtime-hook evidence confirms FA4 dispatch.
- Validation script produces `report.json` and `report.md`.
- The final production patch is minimal and reviewed before applying to the
  current LF checkout.

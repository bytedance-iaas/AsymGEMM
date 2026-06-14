# Qwen3 Complete Activation Backfetch Plan

This document is the Qwen3-only implementation contract for the remaining
activation memory gap after expert activation offload and attention activation
offload. Llama4, Qwen3.5 shared experts, dense-only MLPs, and multimodal paths
are out of scope for this file.

The target is not "more offload" in the abstract. A change is accepted only if
the LF profiling matrix shows a meaningful HBM reduction and no unacceptable
step-time regression. If peak HBM stays the same while latency rises, reject it.
If HBM drops only trivially, reject it.

External references checked:

```text
PyTorch saved-tensor hooks:
  https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html
PyTorch activation checkpointing:
  https://pytorch.org/docs/stable/checkpoint.html
Hugging Face Qwen3 MoE decoder structure:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
```

Local code is the final source of truth:

```text
asym_gemm/training/qwen3_moe.py
asym_gemm/training/attention_activation_offload.py
asym_gemm/training/attention_checkpoint.py
asym_gemm/integrations/lf.py
scripts/lf/profile_lora_lf.sh
scripts/lf/run_lf_profiled_train.py
```

## Current Qwen3 Ground Truth

Workload:

```text
model: Qwen/Qwen3-30B-A3B
backend: asym_cpuadamwds
seq_len: 4096
per_device_train_batch_size: 4
warmup_steps: 5
max_steps: 10
lora_rank: 64
lora_alpha: 16
lora_dropout: 0.00
dataset: asym_long_sft_smoke
profile: source
```

Measured artifacts:

```text
reports/qwen3_layer_act/stage1_profile_matrix/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000
```

Measured comparison table, using the production four-field policy tuple
`policy|expert_act|attn_act|layer_act`. Every row below has stable losses and
`reference_fallback_count=0`.

| backend spec | policy tuple | implementation | peak allocated HBM | peak reserved HBM | avg step | avg forward | avg backward | forward-end HBM | saved CPU peak | AsymGEMM fwd/dx | loss max/last/train |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `asym_cpuadamwds|norecomp` | `none|false|false|false` | no activation offload/no GC | `170.500 GiB` | `183.055 GiB` | `3.038 s` | `1.431 s` | `1.553 s` | `151.703 GiB` | `0.000 GiB` | `5055/4290` | `2.321/1.262/1.872` |
| `asym_cpuadamwds|norecomp` | `gc-exp|false|false|false` | expert checkpoint baseline | `126.312 GiB` | `131.414 GiB` | `3.822 s` | `1.426 s` | `2.342 s` | `107.765 GiB` | `0.000 GiB` | `6495/4290` | `2.333/1.274/1.876` |
| `asym_cpuadamwds|norecomp` | `gc-attn-exp|false|false|false` | attention + expert checkpoint baseline | `85.238 GiB` | `90.359 GiB` | `4.235 s` | `1.448 s` | `2.734 s` | `66.691 GiB` | `0.000 GiB` | `9375/4290` | `2.324/1.264/1.872` |
| `asym_cpuadamwds|norecomp` | `gc-layer|false|false|false` | Qwen3 decoder-layer checkpoint baseline | `37.422 GiB` | `43.016 GiB` | `4.431 s` | `1.430 s` | `2.949 s` | `18.876 GiB` | `0.000 GiB` | `10095/4290` | `2.333/1.263/1.874` |
| `asym_cpuadamwds|recomp` | `none|false|false|false` | global LF gradient checkpointing | `37.422 GiB` | `43.016 GiB` | `4.614 s` | `1.627 s` | `2.932 s` | `18.876 GiB` | `0.000 GiB` | `10095/4335` | `2.322/1.265/1.872` |
| `asym_cpuadamwds|norecomp` | `none|true|false|false` | expert activation offload | `102.312 GiB` | `107.469 GiB` | `43.822 s` | `10.830 s` | `32.881 s` | `83.765 GiB` | `0.000 GiB` | `5055/4290` | `2.329/1.266/1.874` |
| `asym_cpuadamwds|norecomp` | `none|true|true|false` | expert + attention activation offload | `58.343 GiB` | `63.445 GiB` | `44.602 s` | `11.451 s` | `33.000 s` | `39.796 GiB` | `40.688 GiB attn` | `5055/7170` | `2.331/1.264/1.874` |
| `asym_cpuadamwds|norecomp` | `none|true|true|true` | expert + attention + decoder-layer activation offload | `34.593 GiB` | `39.676 GiB` | `45.861 s` | `11.490 s` | `34.213 s` | `16.046 GiB` | `40.688 GiB attn + 23.750 GiB decoder` | `5055/7170` | `2.326/1.269/1.872` |

Interpretation:

```text
exp+attn activation offload saves 112.157 GiB peak allocated HBM versus no
offload, but still trails global recompute by 20.921 GiB.

The remaining gap is already present at forward end:
  exp+attn forward end: 39.796 GiB
  global recompute forward end: 18.876 GiB
  gap: 20.920 GiB

Decoder-layer saved-tensor offload removes that gap:
  exp+attn+layer peak allocated: 34.593 GiB
  global recompute peak allocated: 37.422 GiB
  delta: -2.829 GiB

The layer wrapper reduces forward-end HBM from 39.796 GiB to 16.046 GiB, with
48 decoder saved-tensor rows and 23.750 GiB summed decoder CPU peak. Step time
rises from 44.602 s to 45.861 s (+2.8%) versus exp+attn offload. This
incremental layer stage is accepted because it removes a large layer-boundary
HBM residency with a small incremental timing cost.

The full activation-offload stack is not a latency replacement for recompute:
global recompute is 4.614 s/step, while exp+attn+layer activation offload is
45.861 s/step. The accepted result is a memory-pressure mode that beats global
recompute peak HBM by 2.829 GiB and reduces forward-end HBM by 2.830 GiB. It is
not the default speed path until expert activation offload latency is reduced.

Qwen3-30B-A3B local config:
  layers=48, hidden=2048, batch*seq=16384, mlp_only_layers=[]

One BF16 hidden [4,4096,2048] is 64 MiB.
One FP32 hidden [4,4096,2048] is 128 MiB.
Across 48 layers, a small number of residual/RMSNorm boundary saves per layer
explains almost exactly the 20.9 GiB gap.
```

Already covered:

```text
q/k/v/o projection base weights and LoRA-A activation operands:
  asym_gemm/training/attention_activation_offload.py

attention-core saved tensors under text attention parents:
  AttentionSavedTensorOffloadWrapper

packed routed expert MLP activations:
  asym_gemm/training/qwen3_moe.py::_ActivationOffloadQwen3ExpertFunction
```

Implemented Qwen3 target:

```text
decoder-layer residual and RMSNorm boundary tensors outside the attention
parent and outside AsymQwen3Experts.
```

Non-targets in this Qwen3 file:

```text
router activation offload:
  AsymQwen3MoeBlock computes routing under no_grad in routerwhole mode. Router
  logits/top-k metadata are not the 20.9 GiB gap.

dense/shared MLP activation offload:
  Qwen3-30B-A3B has mlp_only_layers=[] and no shared expert. Do not add Llama4
  or Qwen3.5 shared-expert work to this Qwen3 plan.

logits/loss/vocab output:
  not layer-local recompute coverage and not currently shown as the gap.
```

## Notation

```text
@    = GEMM
@^L  = AsymGEMM with a CPU left operand
@^R  = AsymGEMM with a CPU right operand
@^L_grp, @^R_grp = grouped forms over active expert groups

offload(Z)   = copy HBM tensor to CPU and save the CPU owner
stage(Z_cpu) = copy CPU tensor to an HBM tensor for immediate use
release(...) = listed tensors or saved handles are no longer live

CPU tensors have suffix _cpu.
Tensors without _cpu are HBM tensors.
Temp means HBM temporary, released immediately after last use.
Grad means trainable LoRA gradient.
```

For all GEMM-consuming leaves, follow `agent/mlp_math.md` and
`agent/attn_math.md`. This file adds only the Qwen3 decoder-layer boundary
schedule around those already-implemented kernels.

## Forward

Qwen3 decoder layer, conceptually:

```text
X0 = hidden_states                                      # [B,T,H] HBM

# ---------------- first residual and input RMSNorm ----------------

X0_cpu = offload(X0)                                    # [B,T,H] CPU, for layer boundary backward
N0 = rmsnorm(X0)                                        # [B,T,H] HBM Temp


# ---------------- attention block ----------------

A = attention(N0)                                       # [B,T,H] HBM Temp
                                                        # q/k/v/o use attn_math.md
                                                        # attention saved tensors use attention hooks
release(N0_if_not_saved_by_autograd)


# ---------------- second residual and post-attention RMSNorm ----------------

X1 = X0 + A                                             # [B,T,H] HBM live
X1_cpu = offload(X1)                                    # [B,T,H] CPU, for layer boundary backward
release(X0_cpu_only_when_first_norm_backward_finishes)
N1 = rmsnorm(X1)                                        # [B,T,H] HBM Temp


# ---------------- routed MoE MLP ----------------

M = qwen3_moe(N1)                                       # [B,T,H] HBM Temp
                                                        # packed expert math uses mlp_math.md
release(N1_if_not_saved_by_autograd)


# ---------------- final residual ----------------

Y = X1 + M                                              # [B,T,H] HBM output
release(A, M, X1_if_not_output_alias)
```

Important lifetime rule:

```text
Do not create q/k/v base outputs or expert gate/up/down wide tensors early and
keep them live. Attention and expert internals already use branch-local
scheduling. The new layer wrapper must not disturb that. It only offloads saved
boundary tensors that PyTorch autograd would otherwise keep in HBM.
```

Implementation form in the first accepted stage:

```text
Use saved_tensors_hooks around the Qwen3 decoder layer forward.

The hook packs only large CUDA saved tensors that:
  dtype is bf16/fp16/fp32,
  tensor.requires_grad is true by default,
  tensor is not a leaf trainable parameter,
  nbytes >= ASYM_DECODER_SAVED_TENSOR_OFFLOAD_MIN_BYTES.

This preserves PyTorch's RMSNorm/residual math while replacing HBM residency
with CPU ownership. It is intentionally not a custom RMSNorm backward yet.
```

## Backward

The backward schedule is:

```text
dY = dL/dY                                              # [B,T,H] HBM

# ---------------- final residual backward ----------------

dX1_from_final = dY                                     # [B,T,H] HBM
dM = dY                                                # [B,T,H] HBM


# ---------------- MoE backward ----------------

dN1 = qwen3_moe_backward(dM)                            # [B,T,H] HBM
                                                        # expert backfetch uses @^L_grp/@^R_grp


# ---------------- post-attention RMSNorm backward ----------------

X1_stage = stage(X1_cpu)                                # [B,T,H] HBM Temp
dX1_from_norm = rmsnorm_backward(dN1, X1_stage)         # [B,T,H] HBM
release(X1_stage, X1_cpu)

dX1 = dX1_from_final + dX1_from_norm                    # [B,T,H] HBM


# ---------------- attention backward ----------------

dA = dX1                                                # [B,T,H] HBM
dX0_from_second_residual = dX1                          # [B,T,H] HBM
dN0 = attention_backward(dA)                            # [B,T,H] HBM
                                                        # q/k/v/o backfetch uses @^L/@^R


# ---------------- input RMSNorm backward ----------------

X0_stage = stage(X0_cpu)                                # [B,T,H] HBM Temp
dX0_from_norm = rmsnorm_backward(dN0, X0_stage)         # [B,T,H] HBM
release(X0_stage, X0_cpu)

dX0 = dX0_from_second_residual + dX0_from_norm          # [B,T,H] HBM
```

The first implementation gets this schedule by saved-tensor pack/unpack rather
than a handwritten decoder-layer autograd function. That is deliberate:

```text
Residual and RMSNorm are elementwise reductions, not GEMMs. They cannot use
@^L/@^R directly. The AsymGEMM wins remain in attention q/k/v/o and packed
experts. The layer wrapper removes the large HBM residency that global
checkpointing removes by recompute, while preserving the existing AsymGEMM
backfetch kernels.
```

If the saved-tensor wrapper reduces HBM but latency is unacceptable, implement a
second-stage custom RMSNorm/residual function. Do not start with that path.

## Stage 0 - Qwen3 Profiling Axis And GC-Layer Baseline

Scope:

```text
Modify:
  scripts/lf/profile_lora_lf.sh
    ASYMM_EXP_ACT_POLICIES default/help
    parse_exp_act_policy_tuple
    expact/attnact/layeract path tags
    job_root_path and existing-profile checks
    run_one_profile environment/config

  scripts/lf/run_lf_lora_sft.sh
    ASYMM_LAYER_ACT_OFFLOAD default and bool normalization
    ASYM_GEMM_LF_CONFIG_ASYMM_LAYER_ACT_OFFLOAD
    ASYM_GEMM_LF_CONFIG_LAYER_GC_ENABLED

  scripts/lf/run_lf_profiled_train.py
    _config_from_args
    _activation_offload_counters_from_model if a layer wrapper stats key needs
    normalization

  scripts/lf/postprocess_lf_profile_artifacts.py
    summary/config rendering for layer activation offload and layer GC

  asym_gemm/training/moe.py
    parse_expert_recompute_policy_spec

  asym_gemm/integrations/lf.py
    LFAsymReport
    apply_lf_asym_lora
    Qwen3 decoder-layer detection and wrapper installation

Add:
  asym_gemm/training/decoder_checkpoint.py
    DecoderCheckpointWrapper
    install_decoder_checkpoint
    decoder_checkpoint_module_names

Tests:
  tests/lf/test_asym_cpu_adamw_args.py
  tests/lf/test_lf_profile_postprocess.py
  tests/training/test_lf_qwen3_asym_backend.py
```

Concrete changes:

```text
1. Extend ASYMM_EXP_ACT_POLICIES to accept either:
     policy|expert_act|attn_act
     policy|expert_act|attn_act|layer_act
   Missing layer_act means false. This keeps every existing command valid.

2. Add layeract path tags:
     __expact1__attnact1__layeract1
   Existing three-field paths may remain valid for legacy artifact discovery,
   but new runs must include layeract.

3. Add policy label gc-layer. It means selective Qwen3 decoder-layer
   checkpointing only, with no global LF gradient_checkpointing.

4. Keep activation-offload tuples mutually exclusive with gc-* policies:
     gc-layer|false|false|false
     none|true|true|true
   Do not allow gc-layer|true|... because the comparison would mix recompute
   and offload.

5. Under BACKEND_SPECS=*|recomp, reject selective gc-layer unless an explicit
   developer override is set. Selective layer GC is not global LF GC.

6. Under BACKEND_SPECS=*|recomp, reject any activation-offload tuple. The
   activation-offload comparisons are norecomp-only because global recompute
   would mix checkpointing with offload and hide the actual memory/timing cost.

7. Record config fields:
     asymm_layer_act_offload
     layer_gc_enabled
     layer_gc_wrapped
     layer_act_offload_wrapped
```

Pseudocode:

```python
def parse_tuple(raw):
    fields = raw.split("|")
    if len(fields) == 3:
        policy, expact, attnact = fields
        layeract = "false"
    elif len(fields) == 4:
        policy, expact, attnact, layeract = fields
    else:
        fail("expected policy|exp_act|attn_act[|layer_act]")

    policy = normalize_expert_policy(policy)
    expact = bool_value(expact)
    attnact = bool_value(attnact)
    layeract = bool_value(layeract)

    if (expact or attnact or layeract) and policy != "none":
        fail("activation offload tuples require policy none")
    return policy, expact, attnact, layeract

def layer_gc_enabled(policy_label):
    return policy_label == "gc-layer"
```

Unresolved risks to watch:

```text
gc-layer wraps the entire decoder layer and therefore recomputes attention and
expert paths too. That is the correct same-scope recompute baseline for layer
activation offload, but it will not isolate residual/RMSNorm alone.
```

Validation before Stage 1:

```bash
bash -n scripts/lf/profile_lora_lf.sh scripts/lf/run_lf_lora_sft.sh
PYTHONPATH=. python -m py_compile \
  asym_gemm/training/moe.py \
  asym_gemm/integrations/lf.py \
  scripts/lf/run_lf_profiled_train.py \
  scripts/lf/postprocess_lf_profile_artifacts.py
PYTHONPATH=. pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/lf/test_lf_profile_postprocess.py::test_profile_config_records_attention_activation_and_gc \
  tests/training/test_lf_qwen3_asym_backend.py::test_parse_expert_recompute_policy_spec

OUTPUT_ROOT=reports/qwen3_layer_act/stage0_dryrun \
SFT_ROOT=/workspace/AsymGEMM-SFT \
ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
GPU_POOL=0 \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='gc-layer|false|false|false,none|true|true|true' \
SEQ_LENS=128 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
PREPARE_DATASETS=false \
DRY_RUN=true \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
scripts/lf/profile_lora_lf.sh --models 'Qwen/Qwen3-30B-A3B|1'

grep -R 'ASYM_EXPERT_RECOMPUTE_POLICY=gc-layer' reports/qwen3_layer_act/stage0_dryrun
grep -R 'ASYMM_LAYER_ACT_OFFLOAD=true' reports/qwen3_layer_act/stage0_dryrun
```

## Stage 1 - Qwen3 Decoder-Layer Saved-Tensor Activation Offload

Scope:

```text
Add:
  asym_gemm/training/decoder_activation_offload.py
    DecoderSavedTensorOffloadWrapper
    install_decoder_saved_tensor_offload
    is_decoder_saved_tensor_offload_wrapper
    decoder_saved_tensor_offload_module_names

Modify:
  asym_gemm/training/__init__.py
  asym_gemm/integrations/lf.py
    _layer_act_offload_enabled
    _is_qwen3_decoder_layer_module_name
    _wrap_qwen3_decoder_saved_tensor_offload_modules
    apply_lf_asym_lora report wiring

  scripts/lf/run_lf_profiled_train.py
    ensure activation_offload rows include DecoderSavedTensorOffloadWrapper

Tests:
  tests/training/test_decoder_activation_offload.py
  tests/training/test_lf_qwen3_asym_backend.py
```

Concrete changes:

```text
1. Implement DecoderSavedTensorOffloadWrapper using torch saved_tensors_hooks.
   It should be structurally similar to AttentionSavedTensorOffloadWrapper but
   use separate env names and stats keys:
     ASYM_DECODER_SAVED_TENSOR_OFFLOAD_MIN_BYTES default 1 MiB
     ASYM_DECODER_SAVED_TENSOR_OFFLOAD_DTYPES default bf16,fp16,fp32
     ASYM_DECODER_SAVED_TENSOR_OFFLOAD_REQUIRE_GRAD default true

2. Install it only on Qwen3 text decoder layers. Detection is strict:
     module has children self_attn, mlp, input_layernorm,
       post_attention_layernorm
     module class or config model_type identifies qwen3_moe when available
     path is not vision/multimodal

3. Do not wrap attention parents or expert leaves with this wrapper. Wrap the
   decoder layer parent once. Nested attention saved-tensor hooks should still
   own attention-core tensors inside self_attn; the decoder wrapper owns
   residual/RMSNorm boundary tensors outside that inner hook.

4. Pack filter:
     skip non-tensor objects
     skip non-CUDA tensors
     skip dtype outside allowlist
     skip tensors below min bytes
     skip leaf trainable parameters
     if require_grad=true, skip tensors with requires_grad=false

5. Unpack recreates the original shape, stride, dtype, and device, then releases
   the CPU owner from live-byte accounting. It must not hold HBM stages past the
   immediate autograd use.

6. Record per-module stats through _last_activation_offload_stats:
     decoder_saved_tensor_offload=true
     offloaded_bytes
     cpu_peak_bytes_live
     max_stage_bytes_live
     num_offloads
     num_stages
     skipped_tensors
     skipped_bytes
     offload_bytes_by_tag
     stage_bytes_by_tag
     dtype_counts
     shape_counts

7. In LF integration, enable layer activation offload only when:
     ASYMM_LAYER_ACT_OFFLOAD=true
     backend == asym
     expert policy == none
     model has Qwen3 decoder layers
     training uses no global gradient checkpointing

8. Recommended first production tuple:
     none|true|true|true
   This layers saved-tensor offload on top of existing expert and attention
   activation offload.
```

Pseudocode:

```python
class DecoderSavedTensorOffloadWrapper:
    def run(self, *args, **kwargs):
        if not self.module.training or not torch.is_grad_enabled():
            return self.original_forward(*args, **kwargs)
        if kwargs.get("use_cache", False):
            self.skipped_cache_calls += 1
            return self.original_forward(*args, **kwargs)
        with saved_tensors_hooks(self._pack, self._unpack):
            return self.original_forward(*args, **kwargs)

    def _pack(self, tensor):
        if not self._should_offload(tensor):
            return tensor
        cpu = empty_strided_cpu_like(tensor, pin_memory=True)
        non_blocking = cpu.is_pinned()
        cpu.copy_(tensor.detach(), non_blocking=non_blocking)
        ready_event = None
        if non_blocking:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(tensor.device))
        return SavedTensorOffloadHandle(
            tensor=cpu,
            original_device=tensor.device,
            original_dtype=tensor.dtype,
            original_shape=tensor.shape,
            original_stride=tensor.stride(),
            tag=self._tag_for(tensor),
            ready_event=ready_event,
        )

    def _unpack(self, handle):
        if not isinstance(handle, SavedTensorOffloadHandle):
            return handle
        if handle.ready_event is not None:
            handle.ready_event.synchronize()
        staged = torch.empty_strided(
            handle.original_shape,
            handle.original_stride,
            device=handle.original_device,
            dtype=handle.original_dtype,
        )
        staged.copy_(handle.tensor, non_blocking=False)
        self._release_cpu_live(handle)
        return staged
```

Correctness rule:

```text
Any nonblocking HBM->CPU copy into a pinned activation buffer must record a CUDA
event and must synchronize that event before the CPU tensor is read or copied
back. This applies to expert activation CPU silu/silu-backward, attention saved
tensors, and decoder saved tensors. Without this readiness edge, LF loss can
look correct in small isolated tests but diverge in the full profiling run.
```

Unresolved risks to watch:

```text
Nested saved_tensors_hooks must be verified with a unit test because attention
already installs an inner hook. The desired behavior is: attention tensors are
handled by the attention wrapper, layer-boundary tensors by the decoder wrapper.

RMSNorm backward stages hidden-size tensors by copy, not AsymGEMM. This can
reduce HBM but add copy latency. The LF profile, not a toy test, decides
acceptance.

The wrapper may offload some tensors from the MoE parent outside
_ActivationOffloadQwen3ExpertFunction, such as scatter/index_add intermediates.
That is allowed if the profile shows a real HBM win and no timing blowup.
```

Validation before Stage 2:

```bash
PYTHONPATH=. pytest -q tests/training/test_decoder_activation_offload.py
PYTHONPATH=. pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_qwen3_decoder_layer_activation_offload_wraps_layers \
  tests/training/test_lf_qwen3_asym_backend.py::test_qwen3_decoder_layer_activation_offload_requires_policy_none \
  tests/training/test_lf_qwen3_asym_backend.py::test_attention_activation_offload_lf_qkv_wrappers_share_parent_context

PYTHONPATH=. python -m py_compile \
  asym_gemm/training/activation_offload.py \
  asym_gemm/training/qwen3_moe.py \
  asym_gemm/training/attention_activation_offload.py \
  asym_gemm/training/decoder_activation_offload.py
```

E2E acceptance profile:

```bash
OUTPUT_ROOT=reports/qwen3_layer_act/stage1_profile_matrix \
SFT_ROOT=/workspace/AsymGEMM-SFT \
ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
HF_HOME=/workspace/AsymGEMM-SFT/third_party/LlamaFactory-fa4/.hf_cache \
GPU_POOL=0 \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
PROFILERS=source \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_DROPOUT=0.00 \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
PROFILE_LEVEL=module \
PROFILE_SYNC=true \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
OVERWRITE=true \
CONTINUE_ON_ERROR=false \
ASYMM_EXP_ACT_POLICIES='none|false|false|false,gc-exp|false|false|false,gc-attn-exp|false|false|false,gc-layer|false|false|false,none|true|false|false,none|true|true|false,none|true|true|true' \
scripts/lf/profile_lora_lf.sh --models 'Qwen/Qwen3-30B-A3B|1'

OUTPUT_ROOT=reports/qwen3_layer_act/stage1_profile_matrix \
SFT_ROOT=/workspace/AsymGEMM-SFT \
ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
HF_HOME=/workspace/AsymGEMM-SFT/third_party/LlamaFactory-fa4/.hf_cache \
GPU_POOL=0 \
BACKEND_SPECS='asym_cpuadamwds|recomp' \
PROFILERS=source \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
DATASET=asym_long_sft_smoke \
PREPARE_DATASETS=true \
DATASET_MIN_TOKENS=4096 \
DATASET_OVERWRITE=true \
LORA_DROPOUT=0.00 \
PROFILE_MEMORY_ATTRIBUTION=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_SNAPSHOT=false \
PROFILE_LEVEL=module \
PROFILE_SYNC=true \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
RUN_POST=false \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
OVERWRITE=true \
CONTINUE_ON_ERROR=false \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
scripts/lf/profile_lora_lf.sh --models 'Qwen/Qwen3-30B-A3B|1'
```

Acceptance gate:

```text
Required:
  none|true|true|true peak allocated HBM is at least 10 GiB below
  none|true|true|false on the same workload.

Reject:
  peak allocated HBM >= 48 GiB, unless stage memory proves a profiler artifact.
  avg step time > none|true|true|false by more than 10%.
  CPU pool growth causes host-memory pressure or allocator churn.
  AsymGEMM/GEMM counts change except for the expected attention saved-tensor
  backward fetches in the attention activation-offload path.

Ultimate target:
  none|true|true|true must move toward, and ideally below, the
  asym_cpuadamwds|recomp global checkpoint peak of 37.422 GiB. If it remains
  higher and much slower than global recompute, it is a memory-pressure mode,
  not the default training mode.
```

## Stage 2 - Qwen3 Custom RMSNorm/Residual Backward Only If Needed

Start this stage only if Stage 1 produces a real HBM reduction but staging
generic saved tensors is too slow.

Scope:

```text
Add:
  asym_gemm/training/qwen3_layer_activation_offload.py
    Qwen3RMSNormResidualOffloadFunction or a narrower RMSNorm function

Modify:
  asym_gemm/integrations/lf.py
    optional swap for Qwen3 input_layernorm and post_attention_layernorm

Tests:
  tests/training/test_qwen3_layer_activation_offload.py
```

Concrete changes:

```text
1. Keep attention and expert wrappers unchanged.
2. Replace only the saved-tensor staging for RMSNorm inputs with a custom
   function that explicitly offloads X and the small inv-rms value.
3. Do not write a full handwritten Qwen3 decoder backward unless the profiler
   proves RMSNorm-only work is insufficient. A full decoder autograd function
   would be high risk because it must preserve cache flags, masks,
   output_attentions, router outputs, residual semantics, and HF return shapes.
```

Math:

```text
X_cpu = offload(X)                                      # [B,T,H] CPU
mu = mean(float(X)^2, dim=-1, keepdim=True)             # [B,T,1] HBM Temp
inv = rsqrt(mu + eps)                                   # [B,T,1] HBM small
inv_cpu = offload(inv)                                  # [B,T,1] CPU
Y = X * inv * gamma                                     # [B,T,H] HBM

dY = dL/dY                                              # [B,T,H] HBM
X_stage = stage(X_cpu)                                  # [B,T,H] HBM Temp
inv_stage = stage(inv_cpu)                              # [B,T,1] HBM Temp
dX = rmsnorm_backward_fused(dY, X_stage, inv_stage, gamma)
dGamma = reduce_sum(dY * X_stage * inv_stage)
release(X_stage, inv_stage, X_cpu, inv_cpu)
```

Unresolved risks to watch:

```text
This is not an AsymGEMM kernel. It is an elementwise/reduction kernel whose only
purpose is tighter lifetime control than generic saved_tensors_hooks. It should
not be accepted if Stage 1 already meets memory and latency targets.
```

Validation:

```bash
PYTHONPATH=. pytest -q tests/training/test_qwen3_layer_activation_offload.py

Run the same Stage 1 LF profile matrix and compare:
  none|true|true|true with saved-tensor layer wrapper
  none|true|true|true with custom RMSNorm/residual path
```

## Stage 3 - Final Acceptance Table

After implementation, always regenerate a clear table from
`scripts/lf/profile_lora_lf.sh` artifacts.

Required rows:

```text
asym_cpuadamwds|norecomp none|false|false|false
asym_cpuadamwds|norecomp gc-exp|false|false|false
asym_cpuadamwds|norecomp gc-attn-exp|false|false|false
asym_cpuadamwds|norecomp gc-layer|false|false|false
asym_cpuadamwds|norecomp none|true|false|false
asym_cpuadamwds|norecomp none|true|true|false
asym_cpuadamwds|norecomp none|true|true|true
asym_cpuadamwds|recomp none|false|false|false
```

Table columns:

```text
backend spec
policy tuple
implementation
peak allocated HBM
peak reserved HBM
avg step
avg forward
avg backward
forward-end HBM
activation offload module counts
AsymGEMM/GEMM count deltas
decision
```

Parser command:

```bash
python - <<'PY'
from pathlib import Path
import csv, json

base = Path('reports/qwen3_layer_act/stage1_profile_matrix/asym_long_sft_smoke__lora__lf__bf16')
for profile in sorted(base.glob('qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000/*/b4_s4096/source_profile.json')):
    d = profile.parent
    data = json.loads(profile.read_text())
    rows = {}
    memory_csv = d / 'memory_by_module.csv'
    if memory_csv.exists():
        with memory_csv.open() as f:
            for row in csv.DictReader(f):
                rows[row['name']] = row

    def sec(name):
        for row in data.get('step', {}).get('rows', []):
            if row.get('name') == name:
                return float(row['milliseconds']) / 1000.0
        return 0.0

    fwd = rows.get('step.forward', {})
    print(
        profile.parent.parent.name,
        f"{data['memory']['peak_allocated_hbm_bytes'] / 2**30:.3f}",
        f"{data['memory']['peak_reserved_hbm_bytes'] / 2**30:.3f}",
        f"{sec('lf.step.total'):.3f}",
        f"{sec('step.forward'):.3f}",
        f"{sec('step.backward'):.3f}",
        f"{float(fwd.get('avg_allocated_end_bytes', 0.0)) / 2**30:.3f}",
    )
PY
```

Acceptance rule:

```text
Keep Qwen3 layer activation offload only if:
  memory reduction is meaningful on the full LF b4/s4096 workload,
  timing does not blow up versus none|true|true|false,
  CPU AdamW still updates only CUDA LoRA params,
  selective gc-layer and global recomp remain available as baselines,
  artifacts clearly record layer_act, layer_gc, activation-offload counters, and
  AsymGEMM/GEMM counts.
```

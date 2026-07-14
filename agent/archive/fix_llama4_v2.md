# Llama4 Activation-Offload Cleanup V2

Goal: keep the working Llama4 ASymGEMM activation-offload behavior, but remove the fragile temporary layer-state metadata path and make `actrecomp/xunpack` easier to validate. This should not change `actrecomp0__xunpack0` behavior.

Existing knobs:

- `ASYM_OFFLOAD_ACT_RECOMPUTE`
- `ASYM_OFFLOAD_X_UNPACKED`

Do not add Llama4-specific env vars.

## Current State

File:

- `asym_gemm/training/llama4_experts.py`

Current Llama4 `xunpack` metadata flow:

1. `AsymLlama4Experts.forward_input_scaled()` stores route metadata temporarily on the layer:
   - `self._llama4_offload_src_hidden`
   - `self._llama4_offload_token_indices`
   - `self._llama4_offload_route_scale`
2. `_ActivationOffloadLlama4ExpertFunction.forward()` reads those fields from `layer`.
3. `forward_input_scaled()` clears the fields in `finally`.

This works, but it is less clean than passing metadata directly. The metadata is per-forward-call state, not durable module state.

Important: not all `try/finally` blocks are bad here. Keep `try/finally` blocks that release real resources:

- `release_lora_weights()` after `gather_lora_weights()`
- `manager.release_stage(...)` for HBM staging buffers
- cleanup around temporary concatenated LoRA-A tensors if retained

Only remove the `try/finally` reason caused by temporary `_llama4_offload_*` layer metadata.

## Stage 1: Pass X-Unpack Metadata Directly

### Scope

Modify:

- `asym_gemm/training/llama4_experts.py`
  - `_ActivationOffloadLlama4ExpertFunction.forward`
  - `_ActivationOffloadLlama4ExpertFunction.backward` return tuple
  - `AsymLlama4Experts._forward_expert_activation_offload`
  - `AsymLlama4Experts.forward_input_scaled`
  - remove `_clear_llama4_offload_metadata`
  - remove `_llama4_offload_*` fields from `__init__`

### Code Changes

Change `_forward_expert_activation_offload`:

```python
def _forward_expert_activation_offload(
    self,
    packed: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    *,
    x_src_hidden: torch.Tensor | None = None,
    x_token_indices: torch.Tensor | None = None,
    x_route_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    self._check_activation_offload_supported(packed)
    return _ActivationOffloadLlama4ExpertFunction.apply(
        packed,
        offsets,
        experts,
        x_src_hidden,
        x_token_indices,
        x_route_scale,
        self.gate_lora_A,
        self.gate_lora_B,
        self.up_lora_A,
        self.up_lora_B,
        self.down_lora_A,
        self.down_lora_B,
        self,
    )
```

Change `_ActivationOffloadLlama4ExpertFunction.forward` signature:

```python
def forward(
    ctx,
    packed: torch.Tensor,
    offsets: torch.Tensor,
    experts: torch.Tensor,
    x_src_hidden: torch.Tensor | None,
    x_token_indices: torch.Tensor | None,
    x_route_scale: torch.Tensor | None,
    gate_lora_A: torch.Tensor,
    gate_lora_B: torch.Tensor,
    up_lora_A: torch.Tensor,
    up_lora_B: torch.Tensor,
    down_lora_A: torch.Tensor,
    down_lora_B: torch.Tensor,
    layer: "AsymLlama4Experts",
) -> torch.Tensor:
```

Replace the current layer-state check:

```python
x_unpacked = (
    _llama4_x_unpacked()
    and lora_a_forward_mode == "hbm"
    and getattr(layer, "_llama4_offload_src_hidden", None) is not None
)
```

with:

```python
x_unpacked = (
    _llama4_x_unpacked()
    and lora_a_forward_mode == "hbm"
    and x_src_hidden is not None
    and x_token_indices is not None
)
```

Replace the current layer-state reads:

```python
if x_unpacked:
    x_cpu = manager.offload(layer._llama4_offload_src_hidden.detach(), "X")
    x_token_indices_cpu = layer._llama4_offload_token_indices.detach().to(device="cpu")
    route_scale = getattr(layer, "_llama4_offload_route_scale", None)
    if route_scale is not None:
        x_route_scale_cpu = route_scale.detach().to(device="cpu", dtype=x_cpu.tensor.dtype).contiguous()
else:
    x_cpu = manager.offload(packed, "X")
layer._clear_llama4_offload_metadata()
```

with direct args:

```python
if x_unpacked:
    x_cpu = manager.offload(x_src_hidden.detach(), "X")
    x_token_indices_cpu = x_token_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
    if x_route_scale is not None:
        x_route_scale_cpu = x_route_scale.detach().to(device="cpu", dtype=x_cpu.tensor.dtype).contiguous()
else:
    x_cpu = manager.offload(packed, "X")
```

Because three inputs were added to `apply`, add three `None` entries to backward return after `grad_packed`:

```python
return (
    grad_packed,
    None,  # offsets
    None,  # experts
    None,  # x_src_hidden
    None,  # x_token_indices
    None,  # x_route_scale
    grad_gate_lora_A,
    grad_gate_lora_B,
    grad_up_lora_A,
    grad_up_lora_B,
    grad_down_lora_A,
    grad_down_lora_B,
    None,  # layer
)
```

In `forward_input_scaled`, remove temporary layer-state assignment:

```python
self._llama4_offload_src_hidden = hidden_states.reshape(metadata.num_tokens, -1)
self._llama4_offload_token_indices = metadata.token_indices
self._llama4_offload_route_scale = metadata.routing_weights
down = self._forward_expert_activation_offload(packed, offsets, experts)
```

and replace with:

```python
down = self._forward_expert_activation_offload(
    packed,
    offsets,
    experts,
    x_src_hidden=hidden_states.reshape(metadata.num_tokens, -1),
    x_token_indices=metadata.token_indices,
    x_route_scale=metadata.routing_weights,
)
```

Remove:

```python
self._clear_llama4_offload_metadata()
```

from `__init__`, remove the `_clear_llama4_offload_metadata` method, and remove the `finally` cleanup call for that metadata. Keep `release_lora_weights()` cleanup:

```python
self.gather_lora_weights()
try:
    ...
finally:
    self.release_lora_weights()
```

### Validation

Run:

```bash
.venv/bin/python -m py_compile asym_gemm/training/llama4_experts.py
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "llama4_forward_input_scaled_exposes_route_scale_for_x_unpacked or llama4"
```

Acceptance:

- no stale `_llama4_offload_*` fields remain
- existing Llama4 tests pass
- `actrecomp0__xunpack0` path still saves packed `X`
- `xunpack=1` still reconstructs packed route input with route scale

## Stage 2: Make Stats Explicit

### Scope

Modify:

- `asym_gemm/training/llama4_experts.py`
  - `_ActivationOffloadLlama4ExpertFunction.forward`
  - `_ActivationOffloadLlama4ExpertFunction.backward`

### Code Changes

Forward stats should include:

```python
activation_offload_stats["llama4_act_recompute"] = bool(act_recompute)
activation_offload_stats["llama4_x_unpacked"] = bool(x_unpacked)
activation_offload_stats["llama4_gate_up_recompute"] = bool(gate_up_recompute)
```

Backward pre-final stats:

```python
activation_offload_stats_pre_release["llama4_act_recompute"] = bool(getattr(ctx, "act_recompute", False))
activation_offload_stats_pre_release["llama4_x_unpacked"] = bool(getattr(ctx, "x_unpacked", False))
activation_offload_stats_pre_release["llama4_gate_up_recompute"] = bool(getattr(ctx, "gate_up_recompute", False))
```

Backward final stats:

```python
activation_offload_stats["llama4_act_recompute"] = bool(getattr(ctx, "act_recompute", False))
activation_offload_stats["llama4_x_unpacked"] = bool(getattr(ctx, "x_unpacked", False))
activation_offload_stats["llama4_gate_up_recompute"] = bool(getattr(ctx, "gate_up_recompute", False))
```

### Validation

Run a Llama4 smoke profile with the knobs off and on:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=0 \
ASYM_OFFLOAD_X_UNPACKED=0 \
bash scripts/lf/profile_lora_lf_test.sh

ASYM_OFFLOAD_ACT_RECOMPUTE=1 \
ASYM_OFFLOAD_X_UNPACKED=1 \
bash scripts/lf/profile_lora_lf_test.sh
```

Inspect `source_profile.json` or training logs for the stats fields.

Acceptance:

- stats clearly show effective `xunpack=False` for `0/0`
- stats clearly show effective `act_recompute=True` for `1/1`
- stats clearly show effective `xunpack=True` only when LoRA-A forward mode is `hbm`

## Stage 3: Keep or Remove Dead `gate_up_recompute`

### Scope

Modify:

- `asym_gemm/training/llama4_experts.py`
  - `_ActivationOffloadLlama4ExpertFunction.forward`
  - `_ActivationOffloadLlama4ExpertFunction.backward`
  - `_recompute_gate_up_cpu`

### Recommendation

Current code hard-codes:

```python
gate_up_recompute = False
```

This means `_recompute_gate_up_cpu` is effectively dead unless future work enables gate/up recompute. It is not a performance issue because the branch is false. It is a readability/maintenance issue.

Recommended action for now: keep the code but label it clearly:

```python
# Reserved for a future gate/up recompute lever. Keep disabled until e2e profiling
# shows gate/up CPU storage is the actual bottleneck and recompute does not blow up backward time.
gate_up_recompute = False
```

Do not enable it without a separate validation stage. Recomputing gate/up would add grouped base GEMM and LoRA work in backward; it can easily hurt latency.

### Validation

No e2e validation required if only adding a comment. If removing the dead code entirely, run all Llama4 tests and one Llama4 e2e smoke:

```bash
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "llama4"
bash scripts/lf/profile_lora_lf_test.sh
```

## Token-Index CPU Copy

Current Llama4 already does this when `xunpack=1`:

```python
x_token_indices_cpu = layer._llama4_offload_token_indices.detach().to(device="cpu")
```

The proposed direct-arg version is:

```python
x_token_indices_cpu = x_token_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
```

This is not a new class of overhead. It is the same GPU-to-CPU copy, just explicit about dtype and contiguity.

Why it is acceptable:

- Size is tiny relative to activation tensors: `num_routes * 8` bytes.
- Example: `4096` routes is about `32 KiB`; `8192` routes is about `64 KiB`.
- `torch.index_select` requires integer indices; `torch.long` is the safe expected dtype.
- Rebuilding packed `X` on CPU needs CPU indices. Keeping indices on GPU would force either a GPU rebuild, which defeats the CPU activation storage goal, or a later copy anyway.

Do not optimize this unless profiling shows it matters. If it ever matters, the next optimization is a pinned reusable CPU index buffer copied with `non_blocking=True`, but that is unnecessary complexity until there is evidence.

## E2E Acceptance

For `0/0`, accept only if Llama4 profiling remains within noise of the current `actrecomp0__xunpack0` run:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=0 \
ASYM_OFFLOAD_X_UNPACKED=0 \
bash scripts/lf/profile_lora_lf_test.sh
```

For `1/1`, accept only if memory improves meaningfully and forward/backward timing does not blow up:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=1 \
ASYM_OFFLOAD_X_UNPACKED=1 \
bash scripts/lf/profile_lora_lf_test.sh
```

Reject if:

- `0/0` latency gets meaningfully worse
- `0/0` memory changes unexpectedly
- `1/1` memory reduction is trivial
- `1/1` backward time increases materially
- loss becomes unstable

## Implementation Status: 2026-06-17

Implemented in `asym_gemm/training/llama4_experts.py`.

Current behavior:

- Llama4 no longer stores temporary `_llama4_offload_*` metadata on the layer.
- `forward_input_scaled()` passes source hidden states, token indices, and route scale directly into `_ActivationOffloadLlama4ExpertFunction`.
- `ASYM_OFFLOAD_X_UNPACKED=0` keeps the original packed-`X` path.
- `ASYM_OFFLOAD_X_UNPACKED=1` offloads source hidden states and CPU route metadata, then rebuilds packed/scaled `X` during backward for the grouped gate/up LoRA-A grad.
- `ASYM_OFFLOAD_ACT_RECOMPUTE=1` releases saved CPU `act` after forward and recomputes it in backward for down LoRA-A grad.
- `gate_up_recompute` remains disabled. It is not needed for the current two flags and should not be enabled without a separate e2e profile because it would add substantial backward recompute work.

Unit validation run:

```bash
.venv/bin/python -m py_compile asym_gemm/training/llama4_experts.py
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "llama4 or qwen3"
```

The tests completed successfully in the local environment.

E2E profiling source:

```bash
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' bash scripts/lf/profile_lora_lf_test.sh
```

Artifact root:

```text
profiling_both/asym_long_sft_smoke__lora__lf__bf16/llama-4-scout-17b-16e__gpus1__b4_s4096_w5_s10_r64_a16_drop000
```

Current e2e results for the available ASymGEMM rows:

| Flags | Train loss | Alloc | Resv | CPU RSS | e2e measured | Fwd | Bwd | Opt | RSS delta vs 0/0 | e2e delta vs 0/0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `act0/x0` | 2.5906 | 27.82G | 29.03G | 900.37G | 69.28s | 7.29s | 56.26s | 1.13s | 0.00G | 0.00s |
| `act0/x1` | 2.5823 | 27.82G | 29.03G | 904.69G | 71.48s | 7.04s | 58.65s | 1.14s | +4.32G | +2.19s |
| `act1/x0` | 2.5865 | 27.82G | 29.03G | 908.24G | 88.10s | 7.00s | 75.36s | 1.12s | +7.87G | +18.82s |

Reference row from the same artifact set:

| Backend | Train loss | Alloc | Resv | CPU RSS | e2e measured | Fwd | Bwd |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zero3_offload` | 2.6417 | 49.53G | 55.93G | 525.72G | 25.53s | 7.12s | 15.89s |

Acceptance decision:

- Do not default-enable either lever for Llama4 Scout in the current implementation.
- `ASYM_OFFLOAD_X_UNPACKED=1` does not reduce CPU RSS or HBM for Scout top-1 routing. It slightly increases CPU RSS and e2e time in the current smoke artifact. This is expected: with `num_experts_per_tok=1`, source hidden and packed routed `X` have nearly the same shape, so the Qwen3-style packed-X compression benefit is not present.
- `ASYM_OFFLOAD_ACT_RECOMPUTE=1` is a clear reject for this workload. It increases backward by about 19s with no memory win in the artifact.
- Loss values remain close across ASymGEMM rows, so the issue is performance/memory usefulness, not an obvious correctness failure.

Current recommended Llama4 Scout setting:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=0
ASYM_OFFLOAD_X_UNPACKED=0
```

Validation caveat:

- The current saved Llama4 artifacts show `AsymLlama4Experts` activation-offload rows, but the expert-specific `llama4_act_recompute` / `llama4_x_unpacked` stats were not present in those older source profiles. The source code now writes those fields. The next non-skipped e2e run should be checked to confirm the fields appear in `source_profile.json`.

Main `profile_lora_lf.sh` rerun status:

```bash
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Default config for that run:

- model: `meta-llama/Llama-4-Scout-17B-16E|1`
- backend sweep begins with `asym_cpuadamwds|norecomp`
- seq/batch: `b8_s8192`
- steps: `1` warmup + `1` measured
- flags: `ASYM_OFFLOAD_ACT_RECOMPUTE=0`, `ASYM_OFFLOAD_X_UNPACKED=0`

Result: failed before any completed training step.

Evidence:

- script exit: `Training command failed with status 137`
- heartbeat stage: `model_forward_enter`
- global step: `0`
- canonical `profile.json`: not created
- `train_results.json`: not created
- partial artifact: `profiling_both/asym_long_sft_smoke__lora__lf__bf16/llama-4-scout-17b-16e__gpus1__b8_s8192_w1_s1_r64_a16_drop000/asym_cpuadamwds__nsys__norecomp__polnone__routerwhole__expact1__attnact1__layeract1__loraafwdhbm__actrecomp0__xunpack0__gradofftrue__weightofftrue/b8_s8192/partial_profile.json`
- partial memory before death: `9.68 GiB` allocated HBM, `9.70 GiB` reserved HBM, `573.55 GiB` CPU RSS

This default rerun produced no valid e2e timing/loss row. It does confirm that the Llama4 wrappers were installed before failure (`llama4_moes_wrapped=48` in `train.log`), but it cannot be used to accept or reject the two activation flags. The acceptance decision above remains based on the completed b4/s4096 e2e artifact set.

## CUDA Graph Follow-Up: 2026-06-17

No Llama4 CUDA-graph benefit is claimed.

Reason:

- The Qwen3 real e2e off/compile A/B already rejected `ASYM_CUDA_GRAPH=compile`
  for the current implementation: it lowered HBM but regressed step median and
  p90 latency.
- Existing Llama4 b4/s4096 activation-offload source profiles are much more
  expensive than Qwen3: the accepted baseline row in this doc has about
  `69.28s` measured e2e per step and roughly `900 GiB` CPU RSS.
- The default Llama4 main profile at b8/s8192 exited with status `137` before
  a completed training step. That makes a full Llama4 CUDA-graph A/B expensive
  and resource-risky.

Current CUDA graph status for Llama4:

- `ASYM_CUDA_GRAPH=compile` is launch-wired for AsymGEMM backends generally,
  but it has not been accepted for Llama4.
- Do not enable CUDA graph by default for Llama4.
- Before claiming Llama4 CUDA graph benefit, run a separate same-shape A/B with
  complete artifacts:

```bash
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
GPU_POOL='0' \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=false \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=20 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false \
ASYM_CUDA_GRAPH=off bash scripts/lf/profile_lora_lf.sh

MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false' \
GPU_POOL='0' \
PROFILERS=source \
PROFILE_MEMORY_BREAKDOWN=false \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=20 \
SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
ASYM_CPU_ADAMW_GRAD_OFFLOAD=true \
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false \
ASYM_CUDA_GRAPH=compile \
ASYM_CUDA_GRAPH_TORCH_LOGS=recompiles \
bash scripts/lf/profile_lora_lf.sh
```

Acceptance is the same as Qwen3:

- same workload/config proof
- same Asym execution counters
- finite loss
- no post-warmup recompiles
- median step improves by at least 5% or forward/backward improves by at least
  8%
- p90 does not regress
- HBM/RSS do not grow materially

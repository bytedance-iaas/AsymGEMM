# Qwen3 ASYM_OFFLOAD_ACT_RECOMPUTE / ASYM_OFFLOAD_X_UNPACKED Implementation

Goal: make the existing generic knobs real for Qwen3 routed expert activation offload, without changing the `0/0` behavior and without adding Qwen3-specific envs.

Knobs:

- `ASYM_OFFLOAD_ACT_RECOMPUTE=1`: do not keep saved CPU `act = silu(gate) * up` across the forward/backward gap; recompute it on CPU in backward only when needed for `down_lora_A` grad.
- `ASYM_OFFLOAD_X_UNPACKED=1`: do not keep packed routed-token `X` across the forward/backward gap; keep source hidden states plus route indices and rebuild packed `X` in backward only when needed for gate/up LoRA-A grads.

Default `ASYM_OFFLOAD_ACT_RECOMPUTE=0 ASYM_OFFLOAD_X_UNPACKED=0` must remain semantically identical to current Qwen3 code: save packed `X`, save CPU `act`, no extra tensor copies, no CPU `index_select`, no recompute.

## Scope

Modify:

- `asym_gemm/training/qwen3_moe.py`
  - `_expert_act_offload_lora_a_fwd_mode`
  - `_ActivationOffloadQwen3ExpertFunction.forward`
  - `_ActivationOffloadQwen3ExpertFunction.backward`
  - `AsymQwen3Experts._forward_expert_activation_offload`
  - `AsymQwen3Experts.forward`
  - `AsymQwen3Experts.forward_input_scaled`

Add tests:

- `tests/training/test_lf_qwen3_asym_backend.py`
  - extend the SM100 activation-offload correctness test to cover `actrecomp/xunpack`
  - add a CPU-safe metadata threading test so the metadata path is covered even when SM100 kernels are unavailable

Do not modify:

- scripts/env names
- Qwen3 non-activation-offload paths
- Llama4 source
- grouped GEMM kernels
- expert routing kernels

## Stage 1: Thread Qwen3 Route Metadata Into Activation Offload

### Code Changes

In `asym_gemm/training/qwen3_moe.py`, add generic env readers near `_expert_act_offload_lora_a_fwd_mode()`:

```python
def _expert_act_offload_act_recompute() -> bool:
    return _env_flag("ASYM_OFFLOAD_ACT_RECOMPUTE", False)


def _expert_act_offload_x_unpacked() -> bool:
    return _env_flag("ASYM_OFFLOAD_X_UNPACKED", False)
```

Change `AsymQwen3Experts._forward_expert_activation_offload` to accept direct metadata args. Do not stash anything on `self`; this avoids `try/finally` cleanup and prevents stale layer state.

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
    return _ActivationOffloadQwen3ExpertFunction.apply(
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

In `AsymQwen3Experts.forward`, pass the original unscaled hidden and token indices:

```python
if self._uses_activation_offload():
    down = self._forward_expert_activation_offload(
        packed,
        offsets,
        experts,
        x_src_hidden=hidden_states.reshape(metadata.num_tokens, -1),
        x_token_indices=metadata.token_indices,
        x_route_scale=None,
    )
```

Reason: normal Qwen3 routes scale expert outputs in `scatter_contiguous`, not expert inputs. Rebuilding packed `X` should only gather hidden rows.

In `AsymQwen3Experts.forward_input_scaled`, pass route scale too:

```python
if self._uses_activation_offload():
    down = self._forward_expert_activation_offload(
        packed,
        offsets,
        experts,
        x_src_hidden=hidden_states.reshape(metadata.num_tokens, -1),
        x_token_indices=metadata.token_indices,
        x_route_scale=metadata.routing_weights,
    )
```

Reason: this path multiplies packed inputs before the expert body, so rebuilding packed `X` in backward must also multiply by `metadata.routing_weights`.

Update `_ActivationOffloadQwen3ExpertFunction.forward` signature:

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
    layer: "AsymQwen3Experts",
) -> torch.Tensor:
```

Because this adds three autograd inputs, the backward return tuple must add three `None` entries after `grad_packed`.

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

### Validation

Run syntax and focused non-SM100 tests:

```bash
.venv/bin/python -m py_compile asym_gemm/training/qwen3_moe.py
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "qwen3 and not sm100"
```

Acceptance:

- no syntax errors
- existing Qwen3 tests still pass
- no Llama4 files changed

## Stage 2: Implement `ASYM_OFFLOAD_X_UNPACKED`

### Code Changes

In `_ActivationOffloadQwen3ExpertFunction.forward`, after `lora_a_forward_mode` is resolved:

```python
act_recompute = _expert_act_offload_act_recompute()
x_unpacked = (
    _expert_act_offload_x_unpacked()
    and lora_a_forward_mode == "hbm"
    and x_src_hidden is not None
    and x_token_indices is not None
)
```

Important: `x_unpacked` must stay disabled for `lora_a_forward_mode == "cpu"`, because CPU LoRA-A forward immediately needs packed `X` in forward. Using source hidden there would be wrong.

Replace current unconditional `x_cpu = manager.offload(packed, "X")` with:

```python
x_token_indices_cpu = None
x_route_scale_cpu = None
with prof_range(layer._forward_range("activation_offload", "x_to_cpu")):
    if x_unpacked:
        x_cpu = manager.offload(x_src_hidden.detach(), "X")
        x_token_indices_cpu = x_token_indices.detach().to(device="cpu", dtype=torch.long).contiguous()
        if x_route_scale is not None:
            x_route_scale_cpu = x_route_scale.detach().to(device="cpu", dtype=x_cpu.tensor.dtype).contiguous()
    else:
        x_cpu = manager.offload(packed, "X")
```

Save on `ctx`:

```python
ctx.x_unpacked = x_unpacked
ctx.x_token_indices_cpu = x_token_indices_cpu
ctx.x_route_scale_cpu = x_route_scale_cpu
```

Add helper near `_activation_offload_cpu_silu_backward`:

```python
def _rebuild_qwen3_packed_x_cpu(
    ctx,
    manager: ActivationOffloadManager,
) -> tuple[CPUActivationHandle, bool]:
    if not getattr(ctx, "x_unpacked", False):
        return ctx.x_cpu, False

    hidden_cpu = ctx.x_cpu.tensor
    token_indices = getattr(ctx, "x_token_indices_cpu", None)
    if token_indices is None:
        raise RuntimeError("Qwen3 unpacked-X reconstruction requires token indices")

    manager.wait_cpu_ready(ctx.x_cpu)
    rebuilt = manager.empty_cpu(
        (int(token_indices.numel()), int(hidden_cpu.shape[1])),
        hidden_cpu.dtype,
        ctx.x_cpu.original_device,
        "X",
    )
    torch.index_select(hidden_cpu, 0, token_indices, out=rebuilt.tensor)

    route_scale = getattr(ctx, "x_route_scale_cpu", None)
    if route_scale is not None:
        rebuilt.tensor.mul_(route_scale.reshape(-1, 1).to(dtype=rebuilt.tensor.dtype))

    return rebuilt, True
```

Use helper in backward gate/up LoRA-A grad:

```python
with prof_range("backward.mlp.activation_offload.gate_up_lora_a_grad"):
    x_handle, release_x_handle = _rebuild_qwen3_packed_x_cpu(ctx, manager)
    grad_gate_lora_A, grad_up_lora_A = grouped_lora_a_pair_grad_cpu_right(
        dS_gate,
        dS_up,
        x_handle.tensor,
        offsets,
        experts,
        num_experts=int(gate_lora_A.shape[0]),
        stats=layer.stats,
    )
    if release_x_handle:
        manager.release_cpu(x_handle)
    manager.release_cpu(ctx.x_cpu)
```

Do not add expert loops. The only extra work for `xunpack=1` is one CPU `index_select` over all routed tokens.

### Validation

Add a CPU-safe metadata-threading test in `tests/training/test_lf_qwen3_asym_backend.py`:

```python
def test_qwen3_activation_offload_threads_unpacked_x_metadata(monkeypatch):
    source = FakeQwen3Experts()
    wrapped = AsymQwen3Experts(
        source,
        backend="torch",
        precision="bf16",
        offload=False,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
        init_lora_weights="peft",
        strict=False,
    )
    wrapped.train()
    captured = {}
    monkeypatch.setattr(wrapped, "_uses_activation_offload", lambda: True)

    def fake_activation_offload(packed, offsets, experts, *, x_src_hidden=None, x_token_indices=None, x_route_scale=None):
        captured["src_hidden"] = x_src_hidden.detach().clone()
        captured["token_indices"] = x_token_indices.detach().clone()
        captured["route_scale"] = None if x_route_scale is None else x_route_scale.detach().clone()
        captured["packed"] = packed.detach().clone()
        return wrapped._forward_expert_body(packed, offsets, experts, dense_experts=True)

    monkeypatch.setattr(wrapped, "_forward_expert_activation_offload", fake_activation_offload)
    x = torch.randn(5, source.hidden_dim, dtype=torch.bfloat16)
    top_k_index, top_k_weights = _routing()

    _ = wrapped(x, top_k_index, top_k_weights)
    metadata = build_contiguous_route_metadata(top_k_index, top_k_weights, num_experts=source.num_experts)
    expected_packed = pack_tokens_contiguous(x, metadata)

    torch.testing.assert_close(captured["src_hidden"], x)
    torch.testing.assert_close(captured["token_indices"], metadata.token_indices)
    assert captured["route_scale"] is None
    torch.testing.assert_close(captured["packed"], expected_packed)
```

Also test `forward_input_scaled` route scale:

```python
_ = wrapped.forward_input_scaled(x, top_k_index, top_k_weights)
assert captured["route_scale"] is not None
torch.testing.assert_close(captured["route_scale"], metadata.routing_weights)
```

Run:

```bash
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "unpacked_x_metadata or activation_offload_lora_a_fwd_selector"
```

Acceptance:

- metadata matches `pack_tokens_contiguous`
- normal Qwen3 has no route scale
- `forward_input_scaled` does have route scale

## Stage 3: Implement `ASYM_OFFLOAD_ACT_RECOMPUTE`

### Code Changes

In `_ActivationOffloadQwen3ExpertFunction.forward`, after `act_cpu` is computed and after forward has used it for down base/down LoRA:

```python
ctx.act_recompute = act_recompute
if act_recompute:
    manager.release_cpu(act_cpu)
    ctx.act_cpu = None
else:
    ctx.act_cpu = act_cpu
```

Do not release `gate_cpu` or `up_cpu`; Qwen3 needs them later for SiLU backward. This stage only drops saved `act`.

In backward `down_lora` block, replace direct use of `ctx.act_cpu.tensor`:

```python
if getattr(ctx, "act_recompute", False):
    act_handle = _activation_offload_cpu_silu_mul(ctx.gate_cpu, ctx.up_cpu, manager, tag="act")
else:
    act_handle = ctx.act_cpu

grad_down_lora_A = grouped_lora_a_grad_cpu_right(
    dS_down,
    act_handle.tensor,
    offsets,
    experts,
    num_experts=int(down_lora_A.shape[0]),
    stats=layer.stats,
    tag="down",
)
manager.release_cpu(act_handle)
```

This preserves the existing grouped LoRA-A gradient kernel. There is no new GEMM fragmentation.

Add stats before returning from forward and backward:

```python
activation_offload_stats["qwen3_act_recompute"] = bool(act_recompute)
activation_offload_stats["qwen3_x_unpacked"] = bool(x_unpacked)
```

For backward final stats, use values from `ctx`:

```python
activation_offload_stats["qwen3_act_recompute"] = bool(getattr(ctx, "act_recompute", False))
activation_offload_stats["qwen3_x_unpacked"] = bool(getattr(ctx, "x_unpacked", False))
```

### Validation

Extend SM100 correctness test to cover toggles:

```python
@pytest.mark.parametrize("lora_a_forward_mode", ["cpu", "hbm"])
@pytest.mark.parametrize("act_recompute", ["0", "1"])
@pytest.mark.parametrize("x_unpacked", ["0", "1"])
def test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend(...):
    monkeypatch.setenv("ASYM_OFFLOAD_ACT_RECOMPUTE", act_recompute)
    monkeypatch.setenv("ASYM_OFFLOAD_X_UNPACKED", x_unpacked)
```

Expected stats:

```python
effective_x_unpacked = x_unpacked == "1" and lora_a_forward_mode == "hbm"
assert stats["qwen3_act_recompute"] is (act_recompute == "1")
assert stats["qwen3_x_unpacked"] is effective_x_unpacked
```

For `lora_a_forward_mode == "cpu"` and `ASYM_OFFLOAD_X_UNPACKED=1`, assert fallback to packed X:

```python
assert stats["qwen3_x_unpacked"] is False
```

Run:

```bash
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "sm100_activation_offload_matches_torch_backend"
```

Acceptance:

- outputs match torch backend
- input grads match torch backend
- all LoRA grads match torch backend
- `xunpack=1` is effective only with HBM LoRA-A forward
- `actrecomp=1` does not change math

## Stage 4: Verify `0/0` Is the Old Path

### Required Behavior

With:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=0
ASYM_OFFLOAD_X_UNPACKED=0
```

the executed tensor path must remain:

```python
x_cpu = manager.offload(packed, "X")
ctx.act_cpu = act_cpu
grad_down_lora_A uses ctx.act_cpu.tensor
grad_gate_lora_A / grad_up_lora_A use ctx.x_cpu.tensor
```

No `index_select` rebuild. No activation recompute. No extra H2D/D2H tensor movement. Only cheap Python boolean checks and three optional autograd inputs.

### Validation

Run focused test:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=0 \
ASYM_OFFLOAD_X_UNPACKED=0 \
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "sm100_activation_offload_matches_torch_backend"
```

Then run Qwen3 e2e smoke using existing script defaults:

```bash
bash scripts/lf/profile_lora_lf.sh
```

For quick smoke if the test script is configured to Qwen3:

```bash
bash scripts/lf/profile_lora_lf_test.sh
```

Acceptance:

- `actrecomp0__xunpack0` L1 loss remains sane and comparable to previous runs
- forward/backward timing within noise of previous `0/0` run
- allocated/reserved HBM within noise of previous `0/0` run
- no new failures in `source_profile.json`, `profile.json`, or `train.log`

Reject if:

- `0/0` latency increases meaningfully
- `0/0` memory changes for no reason
- loss diverges from previous Qwen3 runs

## Stage 5: E2E Memory/Latency Acceptance for `1/1`

### Run

Use the real LF LoRA profiling path. Do not accept based only on toy tests.

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=1 \
ASYM_OFFLOAD_X_UNPACKED=1 \
bash scripts/lf/profile_lora_lf.sh
```

If the full script is too expensive during development, first run the configured smoke:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=1 \
ASYM_OFFLOAD_X_UNPACKED=1 \
bash scripts/lf/profile_lora_lf_test.sh
```

### Acceptance

Keep the change only if Qwen3 ASym activation-offload profiling shows:

- meaningful GPU memory reduction versus `actrecomp0__xunpack0`
- no forward/backward timing blow-up
- no loss instability
- no significant kernel launch count explosion
- CPU RSS stays acceptable

Reject if:

- memory reduction is trivial
- latency increases materially with no meaningful memory win
- CPU `index_select` rebuild becomes a backward bottleneck
- `xunpack=1` adds overhead but does not reduce peak memory for the actual Qwen3 workload

## Risks To Watch

- `ASYM_OFFLOAD_X_UNPACKED=1` may reduce CPU activation storage but not GPU peak if Qwen3 peak is dominated by loss/lm_head or other non-expert activations. This must be decided from e2e profiling, not unit tests.
- `forward_input_scaled` must multiply rebuilt packed `X` by route scale. Normal `forward` must not, because Qwen3 applies route weights at scatter.
- `xunpack=1` is intentionally ineffective with CPU LoRA-A forward. Enabling it there would make forward LoRA-A read the wrong input.

## Implementation Status: 2026-06-17

Implemented in `asym_gemm/training/qwen3_moe.py`.

Current behavior:

- `ASYM_OFFLOAD_ACT_RECOMPUTE=0 ASYM_OFFLOAD_X_UNPACKED=0` keeps the original path: packed routed `X` is offloaded, `act = silu(gate) * up` is saved, and no CPU `index_select` rebuild runs.
- `ASYM_OFFLOAD_X_UNPACKED=1` is effective only for activation offload with `ASYM_LORA_A_FWD_MODE=hbm`; it offloads source hidden states plus CPU route metadata, then rebuilds packed `X` once in backward for the grouped gate/up LoRA-A grad.
- `ASYM_OFFLOAD_ACT_RECOMPUTE=1` releases saved CPU `act` after forward and recomputes it in backward for down LoRA-A grad.
- The implementation does not loop over experts or split grouped GEMMs into small GEMMs.

Unit validation run:

```bash
.venv/bin/python -m py_compile asym_gemm/training/qwen3_moe.py tests/training/test_lf_qwen3_asym_backend.py
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "qwen3 and not sm100"
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "sm100_activation_offload_matches_torch_backend"
.venv/bin/python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py -k "llama4 or qwen3"
```

All completed successfully in the local environment.

E2E profiling source:

```bash
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' bash scripts/lf/profile_lora_lf_test.sh
```

Artifact root:

```text
profiling_both/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000
```

E2E results:

| Flags | Train loss | Alloc | Resv | CPU RSS | e2e measured | Fwd | Bwd | Opt | RSS delta vs 0/0 | e2e delta vs 0/0 | Expert flags | X offload | X peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| `act0/x0` | 1.9089 | 28.53G | 33.56G | 345.87G | 43.42s | 3.37s | 35.62s | 1.04s | 0.00G | 0.00s | `False/False` | 0.500G | 0.500G |
| `act0/x1` | 1.9068 | 28.53G | 33.56G | 323.79G | 44.51s | 3.19s | 36.91s | 1.06s | -22.08G | +1.10s | `False/True` | 0.062G | 0.562G |
| `act1/x0` | 1.9077 | 28.53G | 33.56G | 337.76G | 55.91s | 3.36s | 48.20s | 1.01s | -8.10G | +12.50s | `True/False` | 0.500G | 0.500G |
| `act1/x1` | 1.9078 | 28.53G | 33.56G | 315.92G | 57.83s | 3.20s | 50.17s | 1.04s | -29.94G | +14.41s | `True/True` | 0.062G | 0.562G |

Acceptance decision:

- Keep `ASYM_OFFLOAD_X_UNPACKED=1` as a useful Qwen3 lever. It gives a meaningful CPU RSS reduction, about 22.1 GiB in the smoke run, with a small e2e cost of about 1.1s per measured step. HBM is unchanged, which is expected because this lever targets CPU-side expert activation storage.
- Do not default-enable `ASYM_OFFLOAD_ACT_RECOMPUTE=1`. It reduces CPU RSS, but it adds about 12.5-14.4s per step in this workload, mostly in backward. That fails the memory-vs-latency acceptance gate.
- `X peak` can be larger than `X offload` with `xunpack=1` because the compact source-hidden buffer is persistent across the forward/backward gap, then the packed CPU `X` is rebuilt temporarily during backward. Judge the lever by e2e CPU RSS and timing, not by that per-tag temporary peak alone.

Current recommended Qwen3 setting:

```bash
ASYM_OFFLOAD_ACT_RECOMPUTE=0
ASYM_OFFLOAD_X_UNPACKED=1
```

## CUDA Graph Follow-Up: 2026-06-17

The two activation-offload flags above are separate from CUDA graph validation.
For CUDA graph, activation offload was disabled:

```bash
ASYMM_EXP_ACT_POLICIES='none|false|false|false'
ASYM_OFFLOAD_ACT_RECOMPUTE=0
ASYM_OFFLOAD_X_UNPACKED=0
```

Qwen3 no-activation-offload CUDA graph A/B used the real LF profile script:

```bash
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
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
ASYM_CUDA_GRAPH=off bash scripts/lf/profile_lora_lf.sh \
  --output-root profiling_both/cuda_graph_validation_fixed

ASYM_CUDA_GRAPH=compile \
ASYM_CUDA_GRAPH_TORCH_LOGS=recompiles \
MODEL_SPECS='Qwen/Qwen3-30B-A3B|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp' \
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
bash scripts/lf/profile_lora_lf.sh \
  --output-root profiling_both/cuda_graph_validation_patched
```

Comparator result:

```bash
.venv/bin/python scripts/lf/compare_cuda_graph_profiles.py \
  --baseline-profile profiling_both/cuda_graph_validation_fixed/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s20_r64_a16_drop000/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact0__attnact0__layeract0__loraafwdhbm__actrecomp0__xunpack0__gradofftrue__weightofffalse/b4_s4096/profile.json \
  --candidate-profile profiling_both/cuda_graph_validation_patched/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s20_r64_a16_drop000_cgcompile/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact0__attnact0__layeract0__loraafwdhbm__actrecomp0__xunpack0__gradofftrue__weightofffalse/b4_s4096/profile.json \
  --json-output profiling_both/cuda_graph_validation_patched/qwen3_cuda_graph_compare.json
```

Result table:

| Mode | Step median | Step p90 | Fwd+Bwd median | Fwd+Bwd p90 | Peak alloc | Peak resv | CPU RSS peak | Asym calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `off` | 7479.64 ms | 7646.61 ms | 4768.10 ms | 4960.60 ms | 170.32 GiB | 179.84 GiB | 196.78 GiB | 15575 |
| `compile` | 7779.92 ms | 11691.10 ms | 5023.22 ms | 5541.72 ms | 134.84 GiB | 138.20 GiB | 194.34 GiB | 15575 |

Decision:

- CUDA graph compile mode is **rejected** for this Qwen3 workload. It saves HBM,
  but latency regresses: step median `+4.01%`, forward/backward median
  `+5.35%`, and step p90 `+52.89%`.
- The Asym execution counters match exactly between off and compile, so the
  result is a real same-workload comparison, not a fallback mismatch.
- Compile health did not verify zero post-warmup recompiles, and the train log
  emitted empty CUDA graph capture warnings. Do not claim effective capture of
  the heavy AsymGEMM regions from this run.
- `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=true` is currently unsafe for this Qwen3
  no-activation-offload path: it failed with a `0` vs `64` shape mismatch.
  The valid A/B used `ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=false`.
- `actrecomp=1` saves CPU activation memory, not HBM directly. Its value depends on whether CPU activation pressure or staging indirectly affects the run.
- Do not add per-expert loops or split grouped GEMMs. All existing grouped kernels must stay intact.

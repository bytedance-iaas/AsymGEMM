# Llama4 Activation-Offload Peak Fix

This plan is only for Llama4 activation-offload tensor lifetime and staging. Do not add recompute
math, do not change expert math, and do not change workload/model/optimizer knobs to make the
number look better.

## Hard Goal

Target:

```text
model:    meta-llama/Llama-4-Scout-17B-16E
workload: 4096,4,1
tag:      ligerloss0
```

`ligerloss0` means Liger loss is disabled. Use explicit `|ligerloss0` in `BACKEND_SPECS` and do not
compare against `ligerloss1` rows.

Final offload/norecomp rows that must improve:

```text
asym_cpuadamwds,norecomp | none,true,true,false,true   # exp_act + attn_act + layer_gc
asym_cpuadamwds,norecomp | none,true,true,true,false    # exp_act + attn_act + layer_act
```

Baselines:

```text
asym_cpuadamwds,recomp | none,false,false,false,false  = 28,094.45 MiB
zero3_offload,recomp   | none,false,false,false,false  = 50,716.93 MiB
```

The binding target is `peak_alloc_hbm < 28,094 MiB` for both offload rows. The current explicit
`ligerloss0` artifacts show the failure starts before backward:

```text
config                                                 peak MiB    forward-end MiB    forward-peak MiB
asym_cpuadamwds,recomp | none,false,false,false,false  28,094.45        15,231.13         28,094.45
asym_cpuadamwds,norecomp | none,true,true,false,true   47,626.36        34,597.10         47,626.36
asym_cpuadamwds,norecomp | none,true,true,true,false   47,634.38        34,605.13         47,634.38
zero3_offload,recomp | none,false,false,false,false    50,716.93        18,027.10         39,953.79
```

So the implementation must fix both:

- forward-exit retained HBM: about `34.6 GiB` must move close to the asym recompute forward end
  (`15.2 GiB`);
- backward transient peak: remove packed gate/up and LoRA-dx overlaps so final peak beats
  `28,094 MiB`.

Policy format:

```text
EXPERT_SELECTION_POLICY|ASYMM_EXPERT_ACT_OFFLOAD|ASYMM_ATTN_ACT_OFFLOAD|ASYMM_LAYER_ACT_OFFLOAD|ASYMM_LAYER_GC
```

## Validation Contract

Every implementation stage is accepted only from e2e `profile_lora_lf.sh` metrics. Unit tests and
toy kernels are allowed only to prove correctness before the e2e run.

Use this baseline command at the start of every stage:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
ENABLE_LIGER_KERNEL=false \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
WORKLOADS='4096|4|1' \
BACKEND_SPECS='asym_cpuadamwds|recomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false|false' \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_SYNC=true OVERWRITE=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Use this target command after each implementation stage:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
ENABLE_LIGER_KERNEL=false \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
WORKLOADS='4096|4|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|false|true,none|true|true|true|false' \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_SYNC=true OVERWRITE=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

For final acceptance and any stage that changes native kernels, also run `4096|8|1`. Keep this as
two commands so `asym_cpuadamwds|recomp` is never crossed with activation-offload policies.

Final baseline command:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
ENABLE_LIGER_KERNEL=false \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
WORKLOADS='4096|4|1,4096|8|1' \
BACKEND_SPECS='asym_cpuadamwds|recomp|ligerloss0,zero3_offload|recomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|false|false|false|false' \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_SYNC=true OVERWRITE=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Final target command:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM

NUMACTL_MEMBIND=0,1 NUMACTL_CPUNODEBIND=0,1 \
ENABLE_LIGER_KERNEL=false \
MODEL_SPECS='meta-llama/Llama-4-Scout-17B-16E|1' \
WORKLOADS='4096|4|1,4096|8|1' \
BACKEND_SPECS='asym_cpuadamwds|norecomp|ligerloss0' \
ASYMM_EXP_ACT_POLICIES='none|true|true|false|true,none|true|true|true|false' \
PROFILE_MEMORY_BREAKDOWN=true PROFILE_SYNC=true OVERWRITE=true \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Metrics to compare for every run:

- `peak_allocated_hbm_mib`;
- `forward_alloc_end_mib`;
- `forward_peak_allocated_mib`;
- `backward_peak_allocated_mib`;
- `forward_ms`, `backward_ms`, and `step_ms`;
- `reference_fallback_count`;
- trainable LoRA parameter count and offloaded LoRA-bank availability.

Reject an implementation stage if:

- peak HBM drops by less than `1024 MiB` and the stage is not only instrumentation;
- `forward_ms`, `backward_ms`, or `step_ms` regresses by more than `10%` versus the previous
  accepted version for the same row;
- `reference_fallback_count != 0`;
- LoRA banks, trainable params, or optimizer updates disappear;
- the change introduces per-expert Python loops, many tiny GEMMs, or CPU-source LoRA helpers in the
  hot path.

Final acceptance:

- both target rows have `peak_allocated_hbm_mib < 28,094.45`;
- both target rows have `forward_alloc_end_mib <= asym_recomp_forward_end_mib + 512`;
- final timing is not more than `10%` slower than the previous accepted offload implementation.

## Stage 0: Profiling and Guardrails

Purpose:

- make the measurements trustworthy before changing lifetime code;
- expose forward-end, forward-peak, backward-peak, and hard-goal pass/fail in postprocessing;
- add opt-in live-set logging that does not affect normal acceptance runs.

Scope:

- `scripts/lf/profile_lora_lf.sh`
- `scripts/lf/postprocess_lf_profile_artifacts.py`
- `asym_gemm/profiling/lf_trace.py`
- `asym_gemm/training/llama4_experts.py`
  - `_ActivationOffloadLlama4ExpertFunction.forward`
  - `_ActivationOffloadLlama4ExpertFunction.backward`
- `asym_gemm/training/activation_offload.py`
  - `ActivationOffloadManager.stage`
  - `ActivationOffloadManager.stage_concat_columns`
  - `ActivationOffloadManager.release_stage`
- tests:
  - `tests/lf/test_llama4_activation_offload_acceptance.py`

Implementation:

- Keep the five-field policy string intact in all CSV/JSON artifacts.
- Preserve the explicit backend liger tag, especially `ligerloss0`.
- Add postprocessed fields:
  - `hard_goal_baseline_min_peak_alloc_mib`;
  - `hard_goal_delta_mib`;
  - `hard_goal_pass`;
  - `forward_end_baseline_asym_recomp_mib`;
  - `forward_end_delta_vs_asym_recomp_mib`;
  - `forward_end_pass`.
- Add opt-in `ASYM_LLAMA4_EXPACT_DEBUG_LIVE=1` logging for expert live CUDA tensors. Record only
  tensor name, shape, byte size, data pointer, and current `torch.cuda.memory_allocated()`.
- Do not enable live logging for acceptance timing runs.

Pseudocode:

```python
ACT_POLICIES = {
    "none,true,true,false,true",
    "none,true,true,true,false",
}
BASELINES = {
    ("asym_cpuadamwds", "recomp", "none,false,false,false,false", "ligerloss0"),
    ("zero3_offload", "recomp", "none,false,false,false,false", "ligerloss0"),
}

def annotate_llama4_goal(rows):
    groups = group_by(rows, ["model", "workload", "lora_rank", "lora_alpha", "lora_dropout", "liger_loss"])
    for group in groups.values():
        asym_recomp = find_row(group, "asym_cpuadamwds", "recomp", "none,false,false,false,false")
        baselines = [r for r in group if (r.backend, r.recompute, r.policy, r.liger_loss) in BASELINES]
        if asym_recomp is None or len(baselines) < 2:
            continue
        baseline_min = min(r.peak_allocated_hbm_mib for r in baselines)
        for row in group:
            if row.backend == "asym_cpuadamwds" and row.recompute == "norecomp" and row.policy in ACT_POLICIES:
                row.hard_goal_baseline_min_peak_alloc_mib = baseline_min
                row.hard_goal_delta_mib = row.peak_allocated_hbm_mib - baseline_min
                row.hard_goal_pass = row.peak_allocated_hbm_mib < baseline_min
                row.forward_end_baseline_asym_recomp_mib = asym_recomp.forward_alloc_end_mib
                row.forward_end_delta_vs_asym_recomp_mib = (
                    row.forward_alloc_end_mib - asym_recomp.forward_alloc_end_mib
                )
                row.forward_end_pass = row.forward_end_delta_vs_asym_recomp_mib <= 512.0

def _record_llama4_expert_live(layer, label: str, **tensors):
    if os.environ.get("ASYM_LLAMA4_EXPACT_DEBUG_LIVE", "0").lower() not in {"1", "true", "yes"}:
        return
    rows = []
    seen_ptrs = set()
    device = None
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            continue
        device = tensor.device
        ptr = int(tensor.data_ptr())
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        rows.append({
            "name": name,
            "shape": tuple(int(x) for x in tensor.shape),
            "bytes": tensor.numel() * tensor.element_size(),
            "ptr": ptr,
        })
    if rows:
        layer._activation_live_debug.append({
            "label": label,
            "allocated": int(torch.cuda.memory_allocated(device)),
            "tensors": rows,
        })
```

Validation:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
python -m compileall scripts/lf/postprocess_lf_profile_artifacts.py asym_gemm/profiling/lf_trace.py asym_gemm/training/llama4_experts.py asym_gemm/training/activation_offload.py
pytest -q tests/lf/test_llama4_activation_offload_acceptance.py
```

Then run the baseline command and target command from the validation contract. For one diagnostic
target run only, add:

```bash
ASYM_LLAMA4_EXPACT_DEBUG_LIVE=1
```

Accept:

- artifacts contain full policy, backend, recompute, and `ligerloss0`;
- all required memory/timing fields are populated;
- `hard_goal_pass` and `forward_end_pass` are computed for both target policies;
- normal target runs without debug have no meaningful latency regression.

Risk to watch:

- debug logging can perturb timing. Use it only to identify live tensors, then disable it.

## Stage 1: Forward-Exit Lifetime Cleanup

Purpose:

- reduce `forward_alloc_end_mib` for both offload policies from about `34.6 GiB` to near the asym
  recompute forward end;
- keep existing kernels and math unchanged.

Scope:

- `asym_gemm/training/llama4_experts.py`
  - `_ActivationOffloadLlama4ExpertFunction.forward`
- `asym_gemm/training/activation_offload.py`
  - `ActivationOffloadManager.offload`
  - `ActivationOffloadManager.stage`
  - `ActivationOffloadManager.release_stage`
- tests:
  - `tests/training/test_llama4_activation_offload_lifetime.py`

Implementation:

- Save only CPU offload handles, metadata, offsets, expert ids, and scalar config on `ctx`.
- Never store CUDA forward intermediates on `ctx`.
- Release staged CUDA tensors immediately after their last consumer.
- Delete ordinary CUDA locals immediately after offload or accumulation.
- Do not use CPU-source helpers and do not add recompute.

Forward pseudocode:

```python
def forward(ctx, layer, x, router_metadata, ...):
    manager = layer.activation_offload_manager

    x_cpu = manager.offload(x, "x")
    del x

    gate_up = layer.gate_up_base(...)
    gate, up = gate_up.split(layer.intermediate_size, dim=-1)
    gate_cpu = manager.offload(gate, "gate")
    up_cpu = manager.offload(up, "up")
    del gate, up, gate_up

    gate_lr = grouped_expert_lora(manager.stage(x_cpu, tag="x_for_gate_lora"), gate_lora_A, ...)
    gate_lr_cpu = manager.offload(gate_lr, "gate_lora_A_out")
    del gate_lr

    up_lr = grouped_expert_lora(manager.stage(x_cpu, tag="x_for_up_lora"), up_lora_A, ...)
    up_lr_cpu = manager.offload(up_lr, "up_lora_A_out")
    del up_lr

    gate_stage = manager.stage(gate_cpu, tag="gate_for_act")
    up_stage = manager.stage(up_cpu, tag="up_for_act")
    act = F.silu(gate_stage).mul_(up_stage)
    manager.release_stage(gate_stage, drop_cache=True)
    manager.release_stage(up_stage, drop_cache=True)
    del gate_stage, up_stage
    act_cpu = manager.offload(act, "act")
    del act

    down_lr = grouped_expert_lora(manager.stage(act_cpu, tag="act_for_down_lora"), down_lora_A, ...)
    down_lr_cpu = manager.offload(down_lr, "down_lora_A_out")
    del down_lr

    out = layer.down_base(...)
    down_delta = grouped_expert_lora(..., down_lora_B, ...)
    out.add_(down_delta)
    del down_delta

    ctx.x_cpu = x_cpu
    ctx.gate_cpu = gate_cpu
    ctx.up_cpu = up_cpu
    ctx.act_cpu = act_cpu
    ctx.gate_lr_cpu = gate_lr_cpu
    ctx.up_lr_cpu = up_lr_cpu
    ctx.down_lr_cpu = down_lr_cpu
    ctx.router_metadata = router_metadata
    return out
```

Validation:

1. Run unit checks:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
python -m compileall asym_gemm/training/llama4_experts.py asym_gemm/training/activation_offload.py
pytest -q tests/training/test_llama4_activation_offload_lifetime.py
```

2. Run the baseline command and target command from the validation contract.

Accept:

- for both target policies, `forward_alloc_end_mib <= asym_recomp_forward_end_mib + 512`;
- for both target policies, `peak_allocated_hbm_mib` drops by at least `8192 MiB` versus current
  explicit `ligerloss0` rows, or the stage is rejected as ineffective;
- `forward_ms`, `backward_ms`, and `step_ms` do not regress by more than `10%`;
- no reference fallback and no lost LoRA trainable params.

Risk to watch:

- if `forward_alloc_end_mib` remains high after deleting CUDA locals, the remaining live tensors are
  outside the expert wrapper. Use Stage 0 live logging and memory breakdown to identify whether they
  come from attention, layer glue, loss, or module-level hooks before changing backward code.

## Stage 2: Backward Lifetime Cleanup With Existing Kernels

Purpose:

- reduce backward peak using only lifetime ordering and releases;
- keep the current packed gate/up base-dx API for now.

Scope:

- `asym_gemm/training/llama4_experts.py`
  - `_ActivationOffloadLlama4ExpertFunction.backward`
  - `_llama4_grouped_base_dx`
- `asym_gemm/training/activation_offload.py`
  - `ActivationOffloadManager.stage_concat_columns`
  - `ActivationOffloadManager.release_stage`
- tests:
  - `tests/training/test_llama4_activation_offload_lifetime.py`

Implementation:

- Release down-path low-rank stages and CPU handles immediately after weight-gradient computation.
- Offload `grad_act` immediately after down base dx and LoRA dx accumulation.
- Delay gate/up LoRA input-gradient tensors until after packed gate/up base dx so they do not
  overlap with the `[M, 2I]` stage longer than necessary.
- Delete split views before releasing the base packed stage.
- Keep all grouped operations as grouped operations. No per-expert loops.

Backward pseudocode:

```python
grad_lora = grad_output

dS_down = grouped_expert_lora(grad_lora, down_lora_B.transpose(-1, -2), ...).mul_(scale)
down_lr = manager.stage(ctx.down_lr_cpu, tag="down_lr_for_dB")
grad_down_lora_B = grouped_lora_b_grad(grad_lora, down_lr, ...).mul_(scale)
manager.release_stage(down_lr, drop_cache=True)
manager.release_cpu(ctx.down_lr_cpu)
del down_lr

act_stage = manager.stage(ctx.act_cpu, tag="act_for_dA")
grad_down_lora_A = grouped_lora_a_grad_cpu_right(dS_down, act_stage, ...).mul_(scale)
manager.release_stage(act_stage, drop_cache=True)
manager.release_cpu(ctx.act_cpu)
del act_stage

grad_down_lora_x = grouped_expert_lora(dS_down, down_lora_A.transpose(-1, -2), ...)
del dS_down

grad_act = _llama4_grouped_base_dx(layer.down_base, grad_output, ...)
grad_act.add_(grad_down_lora_x.to(dtype=grad_act.dtype))
del grad_down_lora_x
grad_act_cpu = manager.offload(grad_act, "dact")
del grad_act

grad_gate_cpu, grad_up_cpu = silu_backward_to_cpu(grad_act_cpu, ctx.gate_cpu, ctx.up_cpu, manager)
manager.release_cpu(grad_act_cpu)

grad_gate_up = manager.stage_concat_columns(grad_gate_cpu, grad_up_cpu, tag="dgate_up_for_gate_up_base")
grad_gate_stage, grad_up_stage = grad_gate_up.split(int(grad_gate_cpu.tensor.shape[1]), dim=-1)
manager.release_cpu(grad_gate_cpu)
manager.release_cpu(grad_up_cpu)
del grad_gate_cpu, grad_up_cpu

dS_gate = grouped_expert_lora(grad_gate_stage, gate_lora_B.transpose(-1, -2), ...).mul_(scale)
dS_up = grouped_expert_lora(grad_up_stage, up_lora_B.transpose(-1, -2), ...).mul_(scale)
grad_gate_lora_B = grouped_lora_b_grad(grad_gate_stage, manager.stage(ctx.gate_lr_cpu, ...), ...)
grad_up_lora_B = grouped_lora_b_grad(grad_up_stage, manager.stage(ctx.up_lr_cpu, ...), ...)
grad_gate_lora_A, grad_up_lora_A = grouped_lora_a_pair_grad_cpu_right(dS_gate, dS_up, ctx.x_cpu, ...)

grad_x = _llama4_grouped_base_dx(layer.gate_up_base, grad_gate_up, ...)
manager.release_stage(grad_gate_up, drop_cache=True)
del grad_gate_stage, grad_up_stage, grad_gate_up

grad_gate_lora_x = grouped_expert_lora(dS_gate, gate_lora_A.transpose(-1, -2), ...)
grad_x.add_(grad_gate_lora_x.to(dtype=grad_x.dtype))
del grad_gate_lora_x, dS_gate

grad_up_lora_x = grouped_expert_lora(dS_up, up_lora_A.transpose(-1, -2), ...)
grad_x.add_(grad_up_lora_x.to(dtype=grad_x.dtype))
del grad_up_lora_x, dS_up

manager.release_cpu(ctx.gate_cpu)
manager.release_cpu(ctx.up_cpu)
manager.release_cpu(ctx.gate_lr_cpu)
manager.release_cpu(ctx.up_lr_cpu)
manager.release_cpu(ctx.x_cpu)
return grad_x, ...
```

Validation:

1. Run unit checks:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
python -m compileall asym_gemm/training/llama4_experts.py asym_gemm/training/activation_offload.py
pytest -q tests/training/test_llama4_activation_offload_lifetime.py
```

2. Run the baseline command and target command from the validation contract.

Accept:

- both target policies keep the Stage 1 forward-end improvement;
- `backward_peak_allocated_mib` and `peak_allocated_hbm_mib` drop by at least `1024 MiB` versus
  Stage 1;
- `backward_ms` and `step_ms` do not regress by more than `10%`;
- no reference fallback and no small-GEMM/per-expert-loop pattern.

Risk to watch:

- this stage still materializes `[M, 2I]`, so it may not reach the hard goal. It is accepted only as
  a meaningful intermediate memory win.

## Stage 3: Remove Full Gate/Up `[M, 2I]` Staging

Purpose:

- remove the main backward packed temporary;
- compute `dX = dGate @ W_gate + dUp @ W_up` without staging `[M, 2I]`;
- keep CPU weights in the existing packed CPU-resident format.
- do this only after Stage 1 and Stage 2 e2e profiling proves the remaining peak is dominated by
  packed gate/up staging. Do not start with native kernel work.

Scope:

- `asym_gemm/training/llama4_experts.py`
  - `_ActivationOffloadLlama4ExpertFunction.backward`
  - new `_llama4_grouped_gate_up_base_dx_side_accum`
- `asym_gemm/training/frozen_linear.py`
  - grouped bf16 CPU-right dispatch helpers if wrapper support is needed
- native extension:
  - `csrc/apis/llama4_moe.hpp`
  - `csrc/llama4/llama4_gate_up_side_dx.cu`
  - `csrc/python_api.cpp`
  - `CMakeLists.txt`
  - `setup.py`
  - `asym_gemm/__init__.py`
- tests:
  - `tests/m_grouped/test_llama4_gate_up_side_dx.py`
  - `tests/training/test_llama4_activation_offload_lifetime.py`

Implementation:

- Delete the accepted backward use of `manager.stage_concat_columns()` for Llama4 gate/up dx.
- Stage one side at a time:
  - stage `dGate [M, I]`;
  - run grouped LoRA-B and LoRA-A weight-gradient work for gate;
  - accumulate gate base dx into `grad_x`;
  - release `dGate`;
  - repeat for `dUp`.
- Add a native side-select base-dx API that reads the original packed CPU weight with side offset.
- Do not create CPU split-weight copies.
- Do not use `grouped_lora_b_backward_cpu_source()`.
- Do not split into expert loops.
- If Stage 2 already reaches the hard goal with stable timing, skip this stage.

Python pseudocode:

```python
grad_x = torch.empty(
    (int(ctx.x_shape[0]), int(layer.hidden_dim)),
    device=grad_output.device,
    dtype=ctx.input_dtype,
)

grad_gate_stage = manager.stage(grad_gate_cpu, tag="dgate_for_gate")
gate_lr = manager.stage(ctx.gate_lr_cpu, tag="gate_lr_for_dB")
dS_gate = grouped_expert_lora(grad_gate_stage, gate_lora_B.transpose(-1, -2), ...).mul_(scale)
grad_gate_lora_B = grouped_lora_b_grad(grad_gate_stage, gate_lr, ...).mul_(scale)
manager.release_stage(gate_lr, drop_cache=True)
manager.release_cpu(ctx.gate_lr_cpu)
del gate_lr

_llama4_grouped_gate_up_base_dx_side_accum(
    base=layer.gate_up_base,
    out=grad_x,
    grad_side=grad_gate_stage,
    offsets=offsets,
    experts=experts,
    side=0,
    accumulate=False,
)
manager.release_stage(grad_gate_stage, drop_cache=True)
manager.release_cpu(grad_gate_cpu)
del grad_gate_stage, grad_gate_cpu

grad_up_stage = manager.stage(grad_up_cpu, tag="dup_for_up")
up_lr = manager.stage(ctx.up_lr_cpu, tag="up_lr_for_dB")
dS_up = grouped_expert_lora(grad_up_stage, up_lora_B.transpose(-1, -2), ...).mul_(scale)
grad_up_lora_B = grouped_lora_b_grad(grad_up_stage, up_lr, ...).mul_(scale)
manager.release_stage(up_lr, drop_cache=True)
manager.release_cpu(ctx.up_lr_cpu)
del up_lr

_llama4_grouped_gate_up_base_dx_side_accum(
    base=layer.gate_up_base,
    out=grad_x,
    grad_side=grad_up_stage,
    offsets=offsets,
    experts=experts,
    side=1,
    accumulate=True,
)
manager.release_stage(grad_up_stage, drop_cache=True)
manager.release_cpu(grad_up_cpu)
del grad_up_stage, grad_up_cpu
```

C++ API pseudocode:

```cpp
void llama4_gate_up_side_dx_accum_bf16(
    const torch::Tensor& out,              // CUDA [M, H], overwritten or accumulated
    const torch::Tensor& grad_side,        // CUDA [M, I]
    const torch::Tensor& gate_up_weight,   // pinned CPU packed weight
    const torch::Tensor& offsets,          // CUDA int32
    const torch::Tensor& experts,          // CUDA int32
    bool weight_layout_in_out,
    int64_t split_i,
    int64_t side,                          // 0 gate, 1 up
    bool accumulate,
    std::string compiled_dims,
    torch::Dtype output_dtype);
```

Wrapper pseudocode:

```python
def _llama4_grouped_gate_up_base_dx_side_accum(base, out, grad_side, offsets, experts, *, side, accumulate):
    if not isinstance(base, AsymGroupedFrozenLinear):
        raise RuntimeError("side gate/up dx requires AsymGroupedFrozenLinear")
    if base.precision != "bf16" or base.backend == "torch":
        raise RuntimeError("side gate/up dx requires bf16 AsymGEMM CPU-right backend")
    if not grad_side.is_contiguous():
        grad_side = grad_side.contiguous()
    asym_gemm.llama4_gate_up_side_dx_accum_bf16(
        out,
        grad_side,
        base.host_weight.weight,
        offsets,
        experts,
        base.weight_layout == "in_out",
        int(grad_side.shape[-1]),
        int(side),
        bool(accumulate),
        base.compiled_dims,
        out.dtype,
    )
    base.stats.asym_dx_calls += 1
```

Validation:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
python -m pip install -e /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
python - <<'PY'
import asym_gemm
assert hasattr(asym_gemm, "llama4_gate_up_side_dx_accum_bf16")
print(asym_gemm.__file__)
PY
pytest -q tests/m_grouped/test_llama4_gate_up_side_dx.py
pytest -q tests/training/test_llama4_activation_offload_lifetime.py
```

Then run the baseline command and target command from the validation contract. Also run the final
baseline and final target commands because this stage changes the native hot path.

Accept:

- both target policies keep the Stage 1 forward-end improvement;
- both target policies reduce `peak_allocated_hbm_mib` by at least `8192 MiB` versus Stage 2;
- both target policies are at or very near the hard goal; if still above `28,094 MiB`, the remaining
  live-set must be identified before Stage 4;
- `backward_ms` and `step_ms` do not regress by more than `10%`;
- no host CPU split-weight copy, no CPU-source LoRA helper, no reference fallback, no per-expert
  loops.

Risks to watch:

- packed weight layout can be `in_out` or `out_in`; the native side selector must handle the actual
  layout from `AsymGroupedFrozenLinear`.
- if two side launches are too slow, fuse side selection inside one grouped kernel while preserving
  the same large-operation design. Do not fall back to tiny GEMMs.

## Stage 4: Direct LoRA-Dx Accumulation

Purpose:

- remove full `[M, H]` LoRA-dx temporaries if Stage 3 still misses the hard goal;
- keep the grouped LoRA math and avoid extra launches when possible.
- this is optional. It means accumulating LoRA dx directly into the existing `grad_x` buffer instead
  of first materializing `grad_gate_lora_x` / `grad_up_lora_x` as separate `[M, H]` tensors.
- do not write a native accumulation kernel unless e2e profiles after Stage 3 prove these LoRA-dx
  temporaries are the remaining memory blocker.

Scope:

- `asym_gemm/training/llama4_experts.py`
  - `_ActivationOffloadLlama4ExpertFunction.backward`
- `asym_gemm/training/lora.py`
  - new `grouped_expert_lora_add_`
- optional native extension:
  - `csrc/exp_act_offload/exp_act_offload_kernels.cu`
  - `csrc/apis/exp_act_offload.hpp`
  - `csrc/python_api.cpp`
- tests:
  - `tests/training/test_llama4_activation_offload_lifetime.py`

Implementation:

- First try lifetime-only accumulation: create one LoRA dx tensor, add it into `grad_x`, delete it,
  then create the other side.
- If peak still misses the hard goal, add `grouped_expert_lora_add_` so the grouped LoRA dx writes
  directly into `grad_x`.
- Keep dtypes aligned so `.to(dtype=grad_x.dtype)` does not allocate a full temporary.

Pseudocode A:

```python
grad_gate_lora_x = grouped_expert_lora(dS_gate, gate_lora_A.transpose(-1, -2), offsets, experts, metadata=metadata)
grad_x.add_(grad_gate_lora_x.to(dtype=grad_x.dtype))
del grad_gate_lora_x, dS_gate

grad_up_lora_x = grouped_expert_lora(dS_up, up_lora_A.transpose(-1, -2), offsets, experts, metadata=metadata)
grad_x.add_(grad_up_lora_x.to(dtype=grad_x.dtype))
del grad_up_lora_x, dS_up
```

Pseudocode B:

```python
def grouped_expert_lora_add_(out, x, weight_t, offsets, experts, *, metadata, scale=1.0):
    # One grouped operation over active expert groups. No separate [M, H] output allocation.
    return asym_grouped_lora_dx_accum_(out, x, weight_t, offsets, experts, metadata, scale)

grouped_expert_lora_add_(grad_x, dS_gate, gate_lora_A.transpose(-1, -2), offsets, experts, metadata=metadata)
del dS_gate
grouped_expert_lora_add_(grad_x, dS_up, up_lora_A.transpose(-1, -2), offsets, experts, metadata=metadata)
del dS_up
```

Validation:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
python -m compileall asym_gemm/training/llama4_experts.py asym_gemm/training/lora.py
pytest -q tests/training/test_llama4_activation_offload_lifetime.py
```

Then run the baseline command, target command, final baseline command, and final target command from
the validation contract.

Accept:

- both target policies pass the final hard goal: `peak_allocated_hbm_mib < 28,094.45`;
- both target policies keep `forward_end_pass=true`;
- the new accumulation path reduces peak by at least `1024 MiB` versus Stage 3 if Stage 3 did not
  already pass;
- `backward_ms` and `step_ms` do not regress by more than `10%`;
- trace shows grouped kernels, not small per-expert GEMMs.

Risk to watch:

- if Stage 3 already passes the hard goal with stable timing, skip Stage 4. Do not add a native
  accumulation helper unless the e2e numbers prove it is needed.

## Stage 5: Postprocessing Acceptance Gate

Purpose:

- make it impossible to accidentally accept the wrong row, stale artifact, partial profile, or
  `ligerloss1` result.

Scope:

- `scripts/lf/postprocess_lf_profile_artifacts.py`
- `scripts/lf/profile_lora_lf.sh`
- tests:
  - `tests/lf/test_llama4_activation_offload_acceptance.py`
  - `tests/lf/test_asym_cpu_adamw_args.py`

Implementation:

- Require exact grouping by model, workload, backend, recompute mode, full five-field policy,
  `ligerloss0`, rank/alpha/dropout, and profiler.
- Mark rows from failed or partial profiles as ineligible.
- Add explicit failure reasons:
  - missing asym recompute baseline;
  - missing zero3 recompute baseline;
  - target row peak above binding baseline;
  - target row forward-end above asym recompute forward end by more than `512 MiB`;
  - stale/no-liger-tag artifact mixed into an explicit `ligerloss0` group.

Pseudocode:

```python
def eligible(row):
    return (
        row.profile_complete
        and row.liger_loss == "ligerloss0"
        and row.policy in {
            "none,false,false,false,false",
            "none,true,true,false,true",
            "none,true,true,true,false",
        }
    )

def annotate_acceptance(rows):
    for group in group_by([r for r in rows if eligible(r)], GROUP_KEYS):
        asym_recomp = find_required(group, "asym_cpuadamwds", "recomp", "none,false,false,false,false")
        zero3_recomp = find_required(group, "zero3_offload", "recomp", "none,false,false,false,false")
        binding_peak = min(asym_recomp.peak_allocated_hbm_mib, zero3_recomp.peak_allocated_hbm_mib)
        for policy in ("none,true,true,false,true", "none,true,true,true,false"):
            row = find_required(group, "asym_cpuadamwds", "norecomp", policy)
            row.hard_goal_pass = row.peak_allocated_hbm_mib < binding_peak
            row.forward_end_pass = row.forward_alloc_end_mib <= asym_recomp.forward_alloc_end_mib + 512.0
            row.hard_goal_failed_against = []
            if row.peak_allocated_hbm_mib >= asym_recomp.peak_allocated_hbm_mib:
                row.hard_goal_failed_against.append("asym_cpuadamwds,recomp")
            if row.peak_allocated_hbm_mib >= zero3_recomp.peak_allocated_hbm_mib:
                row.hard_goal_failed_against.append("zero3_offload,recomp")
```

Validation:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
pytest -q tests/lf/test_llama4_activation_offload_acceptance.py tests/lf/test_asym_cpu_adamw_args.py
```

Then run the final baseline command and final target command from the validation contract.

Accept:

- both `4096|4|1` target rows show `hard_goal_pass=true` and `forward_end_pass=true`;
- `4096|8|1` has the same direction: target offload rows peak below the asym recompute and zero3
  recompute rows for that workload;
- no final acceptance row comes from a failed/partial profile or a `ligerloss1` profile;
- no timing regression over `10%` for either target row.

Risk to watch:

- comparing only against zero3 is insufficient. The binding target is the minimum of asym recompute
  and zero3 recompute for the same model/workload/tag.

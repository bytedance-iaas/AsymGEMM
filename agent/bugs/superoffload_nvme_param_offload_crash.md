> ⛔ **DO NOT FIX — FOR RECORD ONLY.**
> This document records a confirmed bug for tracking/reference. Do **not** modify
> `superoffload_stage3.py` or any other source in response to this file. The
> "Proposed fix" section below is reference-only and must **not** be applied
> unless a human explicitly asks for it in a separate request.

# SuperOffload + NVMe param offload crashes at optimizer step (`NoneType.data`)

| Field | Value |
|---|---|
| **Status** | Confirmed, open (unfixed by design — record only) |
| **Severity** | High — makes `superoffload_*_panvme` unusable end-to-end |
| **Date observed** | 2026-06-29 |
| **Component** | `third_party/deepspeed` — SuperOffload ZeRO-3 (`runtime/superoffload/superoffload_stage3.py`) |
| **DeepSpeed** | 0.19.2+unknown (official fork with upstreamed `superoffload/` module) |
| **transformers** | 5.6.0 |
| **Repro model** | Qwen/Qwen3-30B-A3B (also affects Qwen3-235B-A22B), LoRA SFT, 1 GPU |

## Summary
Running any **SuperOffload backend with parameters offloaded to NVMe**
(`superoffload_mem_panvme`, and by extension `superoffload_pnvme` /
`superoffload_mem_opnvme`+param-nvme combos) loads the model, runs the
forward + backward, runs the async CPU-Adam, and then **crashes at the very end
of the first optimizer step** while writing the updated parameter back:

```
AttributeError: 'NoneType' object has no attribute 'data'
```

This only surfaces **after** the `buffer_size` floor is satisfied (see Related),
i.e. once the run actually reaches `optimizer.step()`. Plain ZeRO-3 + NVMe params
(`zero3_offload_mem_panvme`, which uses `stage3.py`, not the SuperOffload
override) is **not** affected.

## Reproduction
```
RUNS='q3-30b-a3b|1 ; superoffload_mem_panvme|unsloth|ligerloss1 ; 1000|8|1 ; none|false|false|false|false|false' \
  bash scripts/lf/profile_lora_lf_test_both.sh
```
- DeepSpeed config: `LlamaFactory/examples/deepspeed/ds_z3_superoffload_mem_panvme_config.json`
  (`offload_param.device = nvme`, `offload_optimizer.device = cpu` + `super_offload: true`).
- Reaches `***** Running training *****`, then fails inside step 0's `optimizer.step()`.

## Observed traceback (tail)
```
File ".../deepspeed/runtime/engine.py", line 2836, in _take_model_step
    self.optimizer.step()
File ".../deepspeed/runtime/superoffload/superoffload_stage3.py", line 216, in step
    self._step_with_clipping(scaled_global_grad_norm, timer_names)
File ".../deepspeed/runtime/superoffload/superoffload_stage3.py", line 251, in _step_with_clipping
    self._reassign_or_swap_out_partitioned_parameters(sub_group_id)
File ".../deepspeed/runtime/superoffload/superoffload_stage3.py", line 116, in _reassign_or_swap_out_partitioned_parameters
    self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
AttributeError: 'NoneType' object has no attribute 'data'
```

## Root cause
`superoffload_stage3.py::_reassign_or_swap_out_partitioned_parameters` (lines 114–126):

```python
def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
    if self.subgroup_to_device[sub_group_id] == 'cpu':
        self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(   # <-- line 116, CRASH
            self.fp32_partitioned_groups_flat[sub_group_id].data)
        self._unflatten_partitioned_parameters(sub_group_id)
        return

    if self.fp16_partitioned_groups_flat[sub_group_id] is not None:   # <-- GPU branch DOES guard None
        self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
            self.fp32_partitioned_groups_flat[sub_group_id].data)
        self._unflatten_partitioned_parameters(sub_group_id)
    else:
        self._partitioned_params_swap_out(sub_group_id)              # <-- NVMe write-back path
```

When `offload_param.device == nvme`, the per-subgroup bf16 flat
`fp16_partitioned_groups_flat[sub_group_id]` is **`None`** (the partition lives on
NVMe and is accessed via the param swapper). The **GPU-subgroup branch** handles
this correctly (`else: _partitioned_params_swap_out(...)`), but the
**CPU-subgroup branch** (`subgroup_to_device == 'cpu'`) does **not** — it
unconditionally calls `None.data.copy_(...)`.

SuperOffload's async CPU-Adam assigns sub-groups to the **CPU** device, so it
always hits the unguarded branch. Net: **SuperOffload's param write-back was
written for CPU param offload only and never handled NVMe (`None`) param flats.**
There may be additional NVMe-unaware spots downstream in the SuperOffload
pipeline (the async path `_reassign_or_swap_out_partitioned_parameters_async`,
swap-in on the next step, etc.) — not yet exercised because this is the first
crash.

## Impact / scope
- Breaks: `superoffload_mem_panvme`, `superoffload_pnvme`, and any SuperOffload
  config with `offload_param.device = nvme`.
- Not affected: `zero3_offload_mem_panvme` / `zero3_offload_panvme` (plain ZeRO-3
  Infinity via `stage3.py`); all CPU-param SuperOffload configs
  (`superoffload`, `superoffload_mem`); all `*_opnvme` (optimizer-only NVMe,
  param stays CPU).
- Practical consequence: models that only fit via **param→NVMe** (e.g.
  Qwen3-235B-A22B, which exceeds the ~915 GiB membind cap at the ~2× CPU load
  peak) cannot currently use the **SuperOffload** baseline; they must fall back
  to `zero3_offload_mem_panvme`.

## Workaround (no code change)
Use **`zero3_offload_mem_panvme`** instead of `superoffload_mem_panvme`. It is the
standard ZeRO-Infinity NVMe path and handles `None` (NVMe) param flats correctly.
Trade-off: it is the plain ZeRO-3 baseline, not SuperOffload (no async
CPU-Adam-during-backward).

## Proposed fix (REFERENCE ONLY — DO NOT APPLY per the record-only directive)
Mirror the GPU branch's `None`-guard into the CPU branch:

```python
def _reassign_or_swap_out_partitioned_parameters(self, sub_group_id):
    if self.subgroup_to_device[sub_group_id] == 'cpu':
        if self.fp16_partitioned_groups_flat[sub_group_id] is not None:
            self.fp16_partitioned_groups_flat[sub_group_id].data.copy_(
                self.fp32_partitioned_groups_flat[sub_group_id].data)
            self._unflatten_partitioned_parameters(sub_group_id)
        else:
            self._partitioned_params_swap_out(sub_group_id)
        return
    ...
```
Caveat: this clears only the *first* crash. SuperOffload + NVMe params is untested
upstream; expect further NVMe-unaware code paths (async reassign, next-step
swap-in) to need the same treatment before the combo works end-to-end.

## Related
- **Precondition / earlier wall:** the run only reaches this point after
  `buffer_size` is raised to fit the largest unsharded tensor. With the default
  `buffer_size` too small, param→NVMe fails earlier at load with
  `AssertionError: More elements <N> than buffer size <M>` in
  `partitioned_param_swapper.py:357` (the fused MoE expert `gate_up` tensor:
  Qwen3-30B-A3B = 402,653,184; Qwen3-235B-A22B = 1,610,612,736). Current configs
  use `buffer_size: 2e9`, `buffer_count: 8`.
- Affected DeepSpeed config files live in
  `third_party/LlamaFactory/examples/deepspeed/ds_z3_superoffload*_panvme_config.json`.

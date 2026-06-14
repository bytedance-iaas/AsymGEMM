# Expert-only PyTorch Checkpoint Baseline (`gc-exp`)

## Goal

Add and validate a clean expert-only gradient-checkpoint baseline named
`gc-exp`.

`gc-exp` checkpoints the whole packed routed expert body with
`torch.utils.checkpoint.checkpoint`, while global LlamaFactory/HF decoder-layer
checkpointing stays disabled. This gives the comparison we need:

```bash
BACKEND_SPECS=asym_cpuadamwds|norecomp
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false
```

Interpretation:

- `none|true`: expert activation offload candidate.
- `gc-exp|false`: expert-only framework checkpoint baseline.
- `none|false`: no expert activation strategy sanity check.

`gc-exp` has no threshold suffix. Do not add `gc-ge1`, `gc-le128`, or any
token-count variant. The packed expert body already contains only active routed
rows, so there is no inactive expert compute to skip.

## Hard Invariants

`gc-exp` must be true selective PyTorch GC on routed experts:

- Do not enable global `GRADIENT_CHECKPOINTING`.
- Do not call `_ThresholdedQwen3ExpertFunction.apply`.
- Do not call `_ActivationOffloadQwen3ExpertFunction.apply`.
- Do not add a new project-owned `torch.autograd.Function`.
- Do not save dropout masks, LoRA low-rank tensors, `gate`, `up`, or
  `activated`.
- Do not silently map `gc-exp` to `tok-*`, `none`, or activation offload.
- Do not split the expert body into per-expert or per-token GEMMs.
- Do not add a fallback path that changes semantics or kernel granularity.

The checkpoint boundary must wrap one call to `_forward_expert_body()` on the
packed route tensor. That preserves the existing grouped AsymGEMM/grouped LoRA
execution. During training with reentrant checkpointing, the expected body call
pattern is one no-grad forward call and one backward recompute call under grad.

## Execution Design

Qwen3/Qwen3.5 routed experts use `AsymQwen3Experts.forward()`.
Llama4 uses `AsymPackedExperts`, which is an alias for `AsymQwen3Experts`; the
Llama4 path enters `forward_input_scaled()` after route-score scaling.

Required control flow for Qwen3/Qwen3.5:

```python
metadata = build_contiguous_route_metadata(top_k_index, top_k_weights, num_experts)
packed = pack_tokens_contiguous(hidden_states, metadata)
offsets, experts = make_dense_group_metadata(metadata.expert_offsets, num_groups=num_experts, device=packed.device)

if self._uses_activation_offload():
    down = self._forward_expert_activation_offload(packed, offsets, experts)
elif self._uses_expert_gc():
    down = self._forward_expert_gc(packed, offsets, experts)
elif self._uses_expert_recompute():
    down = self._forward_expert_policy(packed, offsets, experts, metadata)
else:
    down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)

return scatter_contiguous(down, metadata).to(dtype=input_dtype)
```

Required control flow for Llama4:

```python
metadata = build_contiguous_route_metadata(top_k_index, input_weights, num_experts)
packed = pack_tokens_contiguous(hidden_states, metadata)
route_scale = metadata.routing_weights.reshape(metadata.num_routes, *([1] * (packed.dim() - 1)))
packed = packed * route_scale.to(device=packed.device, dtype=packed.dtype)
offsets, experts = make_dense_group_metadata(metadata.expert_offsets, num_groups=num_experts, device=packed.device)

if self._uses_activation_offload():
    down = self._forward_expert_activation_offload(packed, offsets, experts)
elif self._uses_expert_gc():
    down = self._forward_expert_gc(packed, offsets, experts)
elif self._uses_expert_recompute():
    down = self._forward_expert_policy(packed, offsets, experts, metadata)
else:
    down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)

return _scatter_contiguous_sum(down, metadata).to(dtype=input_dtype)
```

Checkpoint helper:

```python
def _forward_expert_gc(self, packed, offsets, experts):
    if self.lora_dropout_p >= 1.0:
        raise NotImplementedError("gc-exp requires 0.0 <= lora_dropout < 1.0")
    if not packed.is_floating_point():
        raise NotImplementedError("gc-exp requires floating packed expert input")

    if not packed.requires_grad and any(param.requires_grad for param in self.parameters()):
        packed = packed.requires_grad_(True)

    def expert_body(packed_arg):
        return self._forward_expert_body(
            packed_arg,
            offsets,
            experts,
            dense_experts=True,
        )

    return checkpoint(
        expert_body,
        packed,
        use_reentrant=self._expert_gc_use_reentrant(),
        preserve_rng_state=True,
    )
```

`_expert_gc_use_reentrant()` defaults to true. `ASYM_EXPERT_GC_USE_REENTRANT`
is only a profiling/debug override.

```python
def _expert_gc_use_reentrant(self):
    default = _env_flag("ASYM_EXPERT_GC_REENTRANT", True)
    return _env_flag("ASYM_EXPERT_GC_USE_REENTRANT", default)
```

This path does not create new GEMM kernels. It reuses whatever
`_forward_expert_body()` already uses for grouped routed experts. Therefore the
implementation must not introduce loops over experts in Python.

## Feasibility Audit

The design is feasible for the current Qwen3/Qwen3.5/Llama4 packed expert path.

Local probes already checked:

- `.venv` loads `asym_gemm._C` and uses `torch 2.12.0+cu130`.
- Reentrant checkpoint requires at least one floating input/output with
  `requires_grad=True`; forcing `packed.requires_grad_(True)` matches the
  LlamaFactory pattern and propagates LoRA parameter gradients.
- Dropout replay works when the same seed is used and
  `preserve_rng_state=True`.
- Actual `AsymQwen3Experts._forward_expert_body` matched normal autograd on
  CPU/backend=`torch`, CUDA/backend=`torch`, and CUDA/backend=`asym` with
  `offload=True`, for dropout `0.0` and nonzero dropout.
- Successful checkpoint probes showed expert body calls `[False, True]`, which
  means original forward ran under no-grad and backward reran under grad.

One failed tiny CUDA probe used non-production dimensions that violated grouped
LoRA alignment. The aligned `.venv` probe passed, so this is not a design
blocker.

If a future model family cannot use this checkpoint boundary, reject `gc-exp`
for that family. Do not silently downgrade to `tok-*`, activation offload, or
plain `none`.

## Stage 1: Policy Parser And Metadata

Scope:

- `asym_gemm/training/moe.py`
  - `ExpertRecomputeConfig`
  - `parse_expert_recompute_policy_spec()`
- `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args()`
- `scripts/plotting/plot_activation_recompute_sweep.py`
  - expert policy parser helper

Implementation:

```python
@dataclass(frozen=True)
class ExpertRecomputeConfig:
    policy: Literal["none", "tok", "gc"]
    token_threshold: int
    activation_save_policy: Literal["save_all", "tok_act"]
    activation_save_threshold: int
    label: str
    token_min: int = 1
    token_max: int | None = None
    activation_save_min: int = 1
    activation_save_max: int | None = None
    force_custom_autograd: bool = False
    torch_checkpoint: bool = False

    @property
    def recompute_enabled(self) -> bool:
        return expert_recompute_policy_enabled(...)

    @property
    def activation_drop_enabled(self) -> bool:
        return expert_activation_save_policy_enabled(...)

    @property
    def torch_checkpoint_enabled(self) -> bool:
        return bool(self.torch_checkpoint)

    @property
    def custom_autograd_enabled(self) -> bool:
        return bool(self.policy == "tok" or self.activation_drop_enabled or self.force_custom_autograd)

    @property
    def enabled(self) -> bool:
        return self.custom_autograd_enabled or self.torch_checkpoint_enabled
```

Parser branch:

```python
if raw == "gc-exp":
    return ExpertRecomputeConfig(
        policy="gc",
        token_threshold=0,
        activation_save_policy="save_all",
        activation_save_threshold=0,
        label="gc-exp",
        token_min=1,
        token_max=None,
        activation_save_min=1,
        activation_save_max=None,
        force_custom_autograd=False,
        torch_checkpoint=True,
    )
```

Required assertions:

```python
cfg = parse_expert_recompute_policy_spec("gc-exp")
assert cfg.policy == "gc"
assert cfg.label == "gc-exp"
assert cfg.torch_checkpoint_enabled
assert not cfg.recompute_enabled
assert not cfg.custom_autograd_enabled
assert cfg.enabled
```

Profile metadata must report:

```python
"expert_recompute_policy_spec": "gc-exp"
"expert_recompute_policy": "gc"
"expert_recompute_impl": "torch_checkpoint"
"expert_gc_use_reentrant": True
"activation_recompute": False
```

Validation:

```bash
PYTHONPATH="$PWD" .venv/bin/python - <<'PY'
from asym_gemm.training.moe import parse_expert_recompute_policy_spec
cfg = parse_expert_recompute_policy_spec("gc-exp")
assert cfg.label == "gc-exp"
assert cfg.policy == "gc"
assert cfg.torch_checkpoint_enabled
assert not cfg.recompute_enabled
assert not cfg.custom_autograd_enabled
assert cfg.enabled
print(cfg)
PY
```

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  tests/training/test_lf_qwen3_asym_backend.py::test_parse_expert_recompute_policy_spec \
  -q
```

## Stage 2: LF Sweep Wiring

Scope:

- `scripts/lf/profile_lora_lf.sh`
  - `ASYMM_EXP_ACT_POLICIES` default
  - `normalize_expert_policy()`
  - `parse_exp_act_policy_pair()`
  - the sweep loop over policy/offload pairs

Implementation:

```bash
ASYMM_EXP_ACT_POLICIES=${ASYMM_EXP_ACT_POLICIES:-"none|true,gc-exp|false,none|false"}
```

```bash
normalize_expert_policy() {
  case "${raw}" in
    none|gc-exp|tok-le0|tok-le0-act)
      printf '%s\n' "${raw}"
      return
      ;;
  esac
  ...
}
```

```bash
parse_exp_act_policy_pair() {
  [[ "${raw}" == *"|"* ]] || die "ASYMM_EXP_ACT_POLICIES item must be policy|true_or_false"
  policy="$(normalize_expert_policy "${policy_part}")"
  expact="$(bool_value "${expact_part}")"
  if [[ "${expact}" == "true" && "${policy}" != "none" ]]; then
    die "activation offload currently requires expert policy none"
  fi
  printf '%s|%s\n' "${policy}" "${expact}"
}
```

The loop must iterate paired values, not a cross product:

```bash
for exp_act_policy_pair in "${exp_act_policy_pairs[@]}"; do
  expert_policy="${exp_act_policy_pair%%|*}"
  ASYMM_EXPERT_ACT_OFFLOAD="${exp_act_policy_pair#*|}"
  ...
  run_job ... "${expert_policy}" ...
done
```

This prevents invalid combinations such as `gc-exp|true`.

Validation:

```bash
bash -n scripts/lf/profile_lora_lf.sh
```

```bash
OUTPUT_ROOT="$PWD/outputs/gc_exp_dryrun_$(date -u +%Y%m%dT%H%M%SZ)" \
RUN_POST=false \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
SEQ_LENS=128 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
LORA_DROPOUT=0.00 \
PREPARE_DATASETS=false \
DRY_RUN=true \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0
```

Then verify:

```bash
find "$OUTPUT_ROOT" -name command.txt -print -exec rg -n \
  "ASYM_EXPERT_RECOMPUTE_POLICY|GRADIENT_CHECKPOINTING|ASYMM_EXPERT_ACT_OFFLOAD" {} \;
```

Expected dry-run properties:

- one path contains `polgc-exp__routerwhole__expact0`;
- that command contains `ASYM_EXPERT_RECOMPUTE_POLICY=gc-exp`;
- that command contains `GRADIENT_CHECKPOINTING=false`;
- no command contains `gc-exp` with `ASYMM_EXPERT_ACT_OFFLOAD=true`.

## Stage 3: Qwen3 Packed Expert Checkpoint Path

Scope:

- `asym_gemm/training/qwen3_moe.py`
  - import `checkpoint`
  - class `AsymQwen3Experts`
  - `_uses_expert_gc()`
  - `_uses_expert_recompute()`
  - `_expert_gc_use_reentrant()`
  - `_forward_expert_gc()`
  - branch order in `forward()`
  - branch order in `forward_input_scaled()`

Implementation:

```python
from torch.utils.checkpoint import checkpoint
```

```python
def _uses_expert_gc(self):
    config = self.expert_recompute_config
    return bool(config.torch_checkpoint_enabled and self.training and torch.is_grad_enabled())

def _uses_expert_recompute(self):
    config = self.expert_recompute_config
    return bool(config.custom_autograd_enabled and self.training and torch.is_grad_enabled())
```

`_uses_expert_recompute()` must not return true for `gc-exp`; otherwise the
old custom autograd function would run.

```python
def _forward_expert_gc(self, packed, offsets, experts):
    if self.lora_dropout_p >= 1.0:
        raise NotImplementedError("gc-exp requires 0.0 <= lora_dropout < 1.0")
    if not packed.is_floating_point():
        raise NotImplementedError("gc-exp requires floating packed expert input")
    if not packed.requires_grad and any(param.requires_grad for param in self.parameters()):
        packed = packed.requires_grad_(True)

    def expert_body(packed_arg):
        return self._forward_expert_body(packed_arg, offsets, experts, dense_experts=True)

    with prof_range(self._forward_range("expert_gc")):
        return checkpoint(
            expert_body,
            packed,
            use_reentrant=self._expert_gc_use_reentrant(),
            preserve_rng_state=True,
        )
```

Branch order in both forward paths:

```python
if self._uses_activation_offload():
    down = self._forward_expert_activation_offload(packed, offsets, experts)
elif self._uses_expert_gc():
    down = self._forward_expert_gc(packed, offsets, experts)
elif self._uses_expert_recompute():
    down = self._forward_expert_policy(packed, offsets, experts, metadata)
else:
    down = self._forward_expert_body(packed, offsets, experts, dense_experts=True)
```

Validation:

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_torch_recompute_policies_match_none \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_torch_recompute_lora_dropout_matches_none \
  tests/training/test_lf_qwen35_asym_backend.py \
  -q
```

Required focused tests:

- `gc-exp` matches `none` for forward/backward with dropout `0.0`.
- `gc-exp` matches `none` for forward/backward with nonzero dropout when the
  same RNG seed is used.
- monkeypatching `_ThresholdedQwen3ExpertFunction.apply` to raise does not
  break `gc-exp`.
- monkeypatching `_ActivationOffloadQwen3ExpertFunction.apply` to raise does
  not break `gc-exp`.
- wrapping `_forward_expert_body()` with a counter shows one call in forward
  and one recompute call in backward for a single loss backward.
- Qwen3.5 and Llama4 wrappers accept `gc-exp` because they use the same packed
  expert implementation.

Correctness test skeleton:

```python
torch.manual_seed(0)
base = wrapped_with_policy("none")
gc = wrapped_with_policy("gc-exp")
copy_lora_params(base, gc)
x = torch.randn(..., dtype=torch.bfloat16, device=device)

torch.manual_seed(123)
loss0 = base(x, top_k_index, top_k_weights).float().sum()
loss0.backward()

torch.manual_seed(123)
loss1 = gc(x, top_k_index, top_k_weights).float().sum()
loss1.backward()

assert_close(loss1, loss0)
assert_close(gc.gate_lora_A.grad, base.gate_lora_A.grad)
assert_close(gc.up_lora_A.grad, base.up_lora_A.grad)
assert_close(gc.down_lora_A.grad, base.down_lora_A.grad)
```

## Stage 4: Real Smoke Runs

Run small real training shapes before the final workload.
`profile_lora_lf.sh` requires at least five warmup steps, so keep
`WARMUP_STEPS=5` even for smoke runs.

Dropout 0:

```bash
OUTPUT_ROOT="$PWD/outputs/gc_exp_smoke_drop000_$(date -u +%Y%m%dT%H%M%SZ)" \
RUN_POST=false \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="gc-exp|false,none|false" \
SEQ_LENS=2048 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_EXTERNAL_MEMORY=false \
bash scripts/lf/profile_lora_lf.sh --gpus 3
```

Dropout nonzero:

```bash
OUTPUT_ROOT="$PWD/outputs/gc_exp_smoke_drop009_$(date -u +%Y%m%dT%H%M%SZ)" \
RUN_POST=false \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="gc-exp|false,none|false" \
SEQ_LENS=2048 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.09 \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_EXTERNAL_MEMORY=false \
bash scripts/lf/profile_lora_lf.sh --gpus 3
```

Validation goals:

- runs complete;
- loss and LoRA grad norms are finite;
- `source_profile.json` reports `expert_recompute_impl=torch_checkpoint`;
- `source_profile.json` reports `activation_recompute=false`;
- no activation-offload path runs for `gc-exp|false`;
- no custom expert recompute path runs for `gc-exp|false`.

## Stage 5: Reentrant Mode Check

Default behavior must be reentrant checkpointing. Compare the override only to
confirm that it does not change peak HBM for the target workload.

```bash
OUTPUT_ROOT="$PWD/outputs/gc_exp_reentrant_true_source_$(date -u +%Y%m%dT%H%M%SZ)" \
ASYM_EXPERT_GC_USE_REENTRANT=true \
RUN_POST=false \
PROFILERS=source \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="gc-exp|false" \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 3
```

```bash
OUTPUT_ROOT="$PWD/outputs/gc_exp_reentrant_false_source_$(date -u +%Y%m%dT%H%M%SZ)" \
ASYM_EXPERT_GC_USE_REENTRANT=false \
RUN_POST=false \
PROFILERS=source \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
ASYMM_EXP_ACT_POLICIES="gc-exp|false" \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 3
```

Expected result from the current workload: peak allocated/reserved HBM is the
same for true and false. Keep the default true because it matches LF/HF default
semantics.

## Stage 6: Final `b4_s6144` Baseline Comparison

Run the full comparison:

```bash
OUTPUT_ROOT="$PWD/outputs/expact_vs_gc_exp_b4s6144_drop000_$(date -u +%Y%m%dT%H%M%SZ)" \
RUN_POST=false \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="none|true,gc-exp|false,none|false" \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.00 \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_EXTERNAL_MEMORY=false \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Run dropout nonzero for the checkpoint baseline:

```bash
OUTPUT_ROOT="$PWD/outputs/gc_exp_b4s6144_drop009_$(date -u +%Y%m%dT%H%M%SZ)" \
RUN_POST=false \
GPU_POOL=3 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES="gc-exp|false,none|false" \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=10 \
WARMUP_STEPS=5 \
LORA_DROPOUT=0.09 \
PLOT=false \
PLOT_MEMORY_BREAKDOWN=false \
PROFILE_MEMORY_BREAKDOWN=false \
PROFILE_EXTERNAL_MEMORY=false \
bash /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus 3
```

Parse results:

```bash
PYTHONPATH="$PWD" .venv/bin/python - <<'PY'
import json
from pathlib import Path

for p in sorted(Path("outputs").rglob("source_profile.json")):
    data = json.load(open(p))
    cfg = data.get("config", {})
    if cfg.get("expert_recompute_policy_spec") not in {"gc-exp", "none"}:
        continue
    mem = data.get("memory", {})
    rows = data.get("step_samples", {}).get("rows", [])
    measured = [r for r in rows if r.get("measured_step")]
    times = []
    for r in measured:
        ms = r.get("training_step_milliseconds")
        if ms is not None:
            times.append(float(ms) / 1000.0)
    avg = sum(times) / len(times) if times else 0.0
    print(
        p,
        "partial" if data.get("partial") else "complete",
        cfg.get("activation_recompute"),
        cfg.get("expert_recompute_policy_spec"),
        cfg.get("expert_recompute_impl"),
        cfg.get("asymm_expert_act_offload"),
        round(mem.get("peak_allocated_hbm_bytes", 0) / 2**30, 3),
        round(mem.get("peak_reserved_hbm_bytes", 0) / 2**30, 3),
        round(avg, 3),
    )
PY
```

Validation goals:

- all non-OOM dropout-0 rows complete on the common `b4_s6144` workload;
- if `none|false` OOMs, capture the partial profile and report the peak before
  failure;
- `gc-exp|false` is clearly marked `torch_checkpoint`;
- `none|true` is clearly marked `expact1`;
- `none|false` is clearly marked `expact0`;
- the table reports peak allocated HBM, peak reserved HBM, and measured step
  timing for all complete rows.

Observed `b4_s6144` dropout-0 source-profile results:

| Expert policy | Expact | Status | Peak alloc GiB | Peak reserved GiB | Avg step s | Avg fwd s | Avg bwd s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none` | `true` | complete | 143.368 | 158.055 | 42.842 | 9.323 | 33.430 |
| `gc-exp` | `false` | complete | 167.462 | 181.883 | 3.649 | 1.464 | 2.135 |
| `none` | `false` | OOM before measured step | 182.748 | 182.848 | n/a | n/a | n/a |

Runtime call checks from the same run:

- `none|true`: `asym_forward_calls=5055`, `asym_dx_calls=4290`,
  `reference_fallback_count=0`.
- `gc-exp|false`: `asym_forward_calls=6495`, `asym_dx_calls=4290`,
  `reference_fallback_count=0`.
- `none|false`: OOM at first forward after `asym_forward_calls=334`,
  `reference_fallback_count=0`.

The `gc-exp` call increase is the checkpoint forward replay. It does not route
through reference fallback or per-expert Python loops.

## Risks And Required Checks

- Reentrant checkpoint requires one floating checkpoint input requiring grad.
  Set `packed.requires_grad_(True)` when LoRA params require grad and packed
  does not.
- Closing over large tensors defeats the point. Close over only `offsets` and
  `experts`; do not close over `gate`, `up`, low-rank tensors, or masks.
- `gc-exp` must not reach `_forward_expert_policy()` or either custom autograd
  function. Add tests that monkeypatch those entries to raise.
- `gc-exp|true` is invalid. Keep the shell parser rejection.
- Dropout correctness must come from checkpoint RNG replay. Do not add manual
  dropout-mask saving.
- Do not compare `gc-exp` under global `recomp`; the target baseline is expert
  checkpointing only with `BACKEND_SPECS=asym_cpuadamwds|norecomp`.
- Source/profile metadata must distinguish `torch_checkpoint` from `custom`, or
  plots will mix incompatible baselines.

## Completion Criteria

The work is complete when:

- `gc-exp` parses through AsymGEMM, LF profiling scripts, profile metadata, and
  plotting filters;
- Qwen3, Qwen3.5, and Llama4 packed expert wrappers can run `gc-exp`;
- dropout 0 and dropout nonzero correctness tests pass;
- tests prove the old custom expert backward paths are not called;
- tests prove `_forward_expert_body()` is rerun during backward;
- dry-run output paths include `polgc-exp__routerwhole__expact0`;
- real source profiles show `activation_recompute=false` and
  `expert_recompute_impl=torch_checkpoint`;
- final `b4_s6144` profiles report HBM and timing for complete rows and capture
  the partial OOM peak for `none|false` if it does not fit.

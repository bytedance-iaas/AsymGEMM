# KT Expert LoRA Dropout Implementation Plan

## Goal

Enable `lora_dropout > 0` for the active local KTransformers SFT expert
backends on the ARM target without changing the `lora_dropout == 0` fast path.

The implementation scope is:

- `kt_torchbf16`
- `kt_armbf16`

`kt_amxbf16` is not part of this plan. AMX is an x86 Intel backend and is not
compatible with the ARM machine used for this work.

The implementation must make expert LoRA dropout correct for:

- gate LoRA: dropout is applied to the hidden input before `gate_lora_A`
- up LoRA: dropout is applied to the hidden input before `up_lora_A`
- down LoRA: dropout is applied to the post-SwiGLU activation before
  `down_lora_A`
- backward: the same mask used by forward is used for LoRA A gradients, LoRA B
  gradients, and LoRA input-gradient contributions
- gradient checkpoint recompute: first forward and recompute forward use the
  same dropout masks

The implementation is not required to match PEFT's exact PyTorch RNG stream.
It must match PEFT dropout semantics: inverted dropout scaling, independent
dropout modules for gate/up/down projections, training-only behavior, and
deterministic forward/backward consistency.

## Execution Status

Status: implemented and validated in this workspace for `kt_torchbf16` and
`kt_armbf16`. AMX remains intentionally unsupported for nonzero LoRA dropout.

Completed implementation points:

- `lora_dropout` now flows from LLaMAFactory arguments through KT config/env,
  wrapper creation, Python SFT wrappers, and ARM native SFT config.
- The KT torch and ARM BF16 SFT paths use deterministic counter dropout for
  gate, up, and down expert LoRA inputs.
- Forward caches store only `dropout_enabled` and `dropout_seed`; dense masks
  are not materialized.
- Backward reuses the cached seed/state, including checkpoint-style recompute.
- `lora_dropout == 0` and eval mode keep the no-dropout behavior.
- LLaMAFactory now allows nonzero dropout only for validated local KT backends:
  `TORCHBF16`, `TORCHBF16_SFT`, `ARMBF16`, `ARMBF16_SFT`, and `KT_ARM`.
- The LF profiling scripts no longer force KT dropout to zero. The default LF
  sweep includes `0.00,0.10`, and new standalone validation lives under
  `scripts/testing/validate_kt_lora_dropout.py`.

Executed correctness validation:

- `test/per_commit/test_armbf16_sft_reference.py`: pass.
- `test/per_commit/test_torchbf16_sft_wrapper_lifecycle.py`: pass.
- `test/per_commit/test_sft_lora_dropout.py`: pass under both the system Python
  and the LLaMAFactory virtualenv.
- `scripts/testing/validate_kt_lora_dropout.py` passes for:
  `kt_torchbf16` p=0, `kt_torchbf16` p=0.10 checkpoint off/on/eval,
  `kt_armbf16` p=0, and `kt_armbf16` p=0.10 checkpoint off/on/profile.
- The LLaMAFactory virtualenv also passes
  `kt_armbf16 --dropout 0.10 --checkpoint on`.

Executed LF profile validation:

- p=0 source/nsys artifacts are present for normal `torch`, `kt_torchbf16`, and
  `kt_armbf16`. The maximum observed loss delta versus the normal torch source
  run was `0.01365`.
- p=0.10 source/nsys artifacts are present for `kt_torchbf16` and
  `kt_armbf16`. `comparisons/loss_compare.tsv` rows are `ok`; maximum relative
  loss delta was `0.007324` for source and `0.007331` for nsys.
- Per-run source memory plots, config-level combined memory plots, nsys timing
  plots, and nsys C2C plots were generated and checked as nonblank.
- A `COLLECT_EXISTING=true` replay with `ROUTER_MODES=hf` and
  `EXPERT_POLICIES=none` exited cleanly and regenerated the combined artifacts
  without retraining.

Historical reduced-smoke latency evidence:

| Backend | Profiler | p=0 step s | p=0.10 step s | overhead | p=0.10 fwd/bwd s | measured samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `kt_armbf16` | source | 284.524 | 292.959 | +2.96% | 60.632 / 232.326 | 2 |
| `kt_armbf16` | nsys | 274.251 | 283.372 | +3.33% | 58.014 / 225.358 | 1 usable |
| `kt_torchbf16` | source | 28.998 | 70.287 | +142.39% | 18.713 / 51.574 | 2 |
| `kt_torchbf16` | nsys | 35.498 | 99.273 | +179.66% | 27.564 / 71.709 | 1 usable |

The latency target is `kt_armbf16`; `kt_torchbf16` is mainly the correctness
oracle and has high Python/tensor counter-dropout overhead. The ARM BF16
standalone validation-loop microprofile also passed at 50 iterations:

| Backend | p=0 avg s | p=0.10 avg s | overhead |
| --- | ---: | ---: | ---: |
| `kt_armbf16` | 0.003767 | 0.003955 | +4.99% |

The executed LF smoke profiles used `MAX_SAMPLES=16`, `MAX_STEPS=2`, and
`WARMUP_STEPS=1` because the shared machine had unrelated GPU processes
occupying most of GPUs 0, 1, and 3. Treat those rows as historical smoke only,
not final latency evidence. Current LF profiling gates require at least
`WARMUP_STEPS=5`; nsys/source timing comparisons must start measuring only
after those five warmup steps.

## Code Inspection Baseline

The active code path is:

- `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel`
- `/workspace/AsymGEMM-SFT/third_party/LlamaFactory`
- `/workspace/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`

Before this implementation, the optimized KT expert path did not support
nonzero dropout:

- `python/sft/config.py`: `KTConfig` carries `kt_lora_rank` and
  `kt_lora_alpha`, but no `kt_lora_dropout`.
- `python/sft/wrapper.py`: `_build_kt_plugin_from_args` passes rank/alpha only.
- `python/sft/base.py`: `BaseSFTMoEWrapper.forward` and `submit_forward` call
  `_make_forward_task(buffer, save_for_backward)` with no training/dropout seed.
- `python/sft/torch_backend.py`: `TorchBF16SFTMoEWrapper._compute` uses raw
  `x` and `act` for LoRA A inputs.
- `python/sft/arm.py`: the ARM native wrapper passes only `save_for_backward`
  to C++ forward tasks.
- `python/sft/amx.py` exists for x86 AMX, but AMX is out of scope for this ARM
  plan.
- `operators/common.hpp`: `MOESFTConfig` has rank/alpha only.
- `operators/arm/bf16_sft_moe.hpp`: ARM LoRA forward/backward uses raw `x` and
  `act`.
- `ext_bindings.cpp`: SFT bindings expose rank/alpha and
  `forward_sft_task(..., save_for_backward)` only.
- `LlamaFactory/src/llamafactory/hparams/parser.py`: local KT explicitly rejects
  `finetuning_args.lora_dropout != 0`.
- `scripts/lf/profile_lora_lf.sh` and `scripts/lf/run_lf_lora_sft.sh`: both
  reject nonzero KT dropout.
- `scripts/lf/profile_lora_lf.sh`: `MODEL_SPECS` entries are already
  `model|num_gpus`; recompute belongs only in `BACKEND_SPECS` entries as
  `backend|recompute`.
- `scripts/lf/profile_lora_lf.sh`: `LORA_DROPOUT_ENV_SET` and
  `lora_dropout_user_set` only exist to force KT runs back to `0.00` unless the
  user explicitly set dropout. After KT dropout support, remove this special
  case instead of preserving it.
- `scripts/lf/profile_lora_lf.sh`: the public KT backend labels are currently
  `kt_torchbf16` and `kt_armbf16`.
- `scripts/lf/run_lf_lora_sft.sh`: the public KT backend labels are currently
  `kt_torchbf16` and `kt_armbf16`; they map to `TORCHBF16` and `ARMBF16`.
- `LlamaFactory/src/llamafactory/hparams/parser.py` currently treats
  `TORCHBF16`, `TORCHBF16_SFT`, `ARMBF16`, `ARMBF16_SFT`, and `KT_ARM` as local
  KT backends. This plan keeps AMX out of the ARM validation path.

The profiling script currently separates profiler roles correctly:

- `source` turns source memory attribution/breakdown on by default.
- `nsys` turns source memory attribution/breakdown off by default.
- When `nsys` is selected, generic timing/interconnect plots use nsys rows.
- Source memory plots are generated by the memory breakdown plotter and stay
  separate from generic timing plots.

## Implementation Uncertainty Register

There is no unresolved design ambiguity before implementation. The remaining
platform constraints are validation gates, not design questions:

| Topic | Decision |
| --- | --- |
| `MODEL_SPECS` format | Use `model|num_gpus` only. Put recompute only in `BACKEND_SPECS=backend|recompute`. |
| PEFT RNG stream | Do not try to match PEFT's RNG stream. Match PEFT dropout semantics with deterministic KT counter masks. |
| Checkpoint recompute | Draw the KT dropout seed from PyTorch CPU RNG before every KT layer forward; non-reentrant checkpoint restores CPU RNG state, so recompute draws the same seed. |
| Dropout masks | Never allocate or cache dense masks in production. Cache only `(dropout_enabled, dropout_seed)`. |
| Shell guard ownership | Remove the profile/run-script KT p=0-only guard once LLaMAFactory has backend-specific support checks. LLaMAFactory remains the single source of unsupported-backend rejection. |
| `kt_torchbf16` exposure | Enable nonzero dropout after Stage 2 passes. |
| `kt_armbf16` exposure | Enable nonzero dropout after Stage 3 passes on an aarch64 host. |
| AMX exposure | Do not add `kt_amxbf16` script labels or AMX dropout support in this ARM plan. Leave AMX unsupported for nonzero dropout. |
| Source vs nsys plots | Keep source memory attribution plots separate from nsys timing/interconnect plots. |

## Non-Negotiable Design

### Dropout Semantics

Use inverted dropout:

```text
drop(x, p, mask) = x * keep(mask) / (1 - p)
```

where `keep(mask)` is 1 for kept elements and 0 for dropped elements.

Validation and runtime must reject `p < 0` or `p >= 1`.

Dropout is enabled only when all are true:

```text
module.training == true
lora_dropout > 0
LoRA is active for the expert backend
```

Dropout is disabled in eval mode and for `lora_dropout == 0`.

### Mask Identity

Masks are keyed by:

```text
seed
layer_idx
projection_id        # 0 gate, 1 up, 2 down
expert_id
global_token_id      # row in qlen before packing
route_slot           # top-k slot for that token
feature_id           # hidden feature for gate/up, intermediate feature for down
```

This guarantees gate/up/down masks are independent, expert modules do not share
masks, and thread scheduling cannot change masks.

AMX packed-row metadata is intentionally excluded from this ARM implementation.

### RNG

Add a deterministic counter-based helper, not a stateful RNG.

Use the same splitmix64-style function in Python and C++:

```text
uint64 mask_key = hash(seed, layer_idx, projection_id, expert_id,
                       global_token_id, route_slot, feature_id)
u = top_24_bits(mask_key) / 2^24
keep = u >= p
scale = keep ? 1 / (1 - p) : 0
```

The exact constants and bit operations must live in one C++ helper header and
one Python helper module, with a unit test proving identical masks for a fixed
list of counters.

### Checkpoint Recompute

Python must draw one `uint64` seed per KT MoE layer forward from the PyTorch CPU
RNG before submitting the C++ task:

```python
torch.empty((), dtype=torch.int64, device="cpu").random_().item()
```

PyTorch non-reentrant checkpoint preserves CPU RNG state, so the recompute
forward draws the same seed as the first forward. C++ stores the seed in its
forward cache when `save_for_backward=True`, so backward uses the exact forward
mask.

Do not use C++ stateful RNG. Do not allocate dense mask tensors in production.

### Fast Path

When `lora_dropout == 0` or `training == false`, native code must call the
existing LoRA matmul paths with no mask branches inside the hot loops.

The acceptance target is:

```text
p=0 median measured step time regression <= 2%
p=0 peak memory change <= 1%
p=0 native outputs/grads match pre-change behavior within existing tolerances
```

## Algorithm Details

### Config And Wrapper Plumbing

1. LLaMAFactory parses the user `lora_dropout` normally.
2. `ModelArguments.get_kt_config_dict(finetuning_args)` writes
   `"kt_lora_dropout": finetuning_args.lora_dropout`.
3. `apply_kt_config.env_mapping` writes
   `ACCELERATE_KT_LORA_DROPOUT`.
4. `KTConfig.__post_init__` reads `ACCELERATE_KT_LORA_DROPOUT` when the dataclass
   field is unset, normalizes `None` to `0.0`, and rejects values outside
   `[0.0, 1.0)`.
5. `_build_kt_plugin_from_args` passes `kt_lora_dropout` from
   `finetuning_args.lora_dropout`.
6. The KT wrapping function reads `cfg.kt_lora_dropout` and passes
   `lora_dropout` into `KTMoEWrapper`.
7. `KTMoEWrapper` passes `lora_dropout` into the selected SFT wrapper factory.
8. Python SFT wrappers store `self.lora_dropout`.
9. Native wrappers copy `lora_dropout` into `MOESFTConfig.lora_dropout`.

### Forward Seed Algorithm

Each KT SFT wrapper computes the dropout state before submitting work:

```python
dropout_enabled = bool(training and self.lora_dropout > 0.0)
dropout_seed = next_lora_dropout_seed(dropout_enabled)
```

`next_lora_dropout_seed(False)` returns `0` and does not draw RNG.  
`next_lora_dropout_seed(True)` draws exactly one signed 64-bit value from
PyTorch CPU RNG:

```python
torch.empty((), dtype=torch.int64, device="cpu").random_().item()
```

The seed is passed to `_make_forward_task` even when `save_for_backward=False`.
When `save_for_backward=True`, cache `(dropout_enabled, dropout_seed)` next to
the existing forward cache metadata. Backward always uses the cached pair.

### Counter Mask Algorithm

Use identical Python and C++ helpers with the same argument order:

```text
scale = lora_dropout_scale(
    seed,
    layer_idx,
    projection_id,
    expert_id,
    global_token_id,
    route_slot,
    feature_id,
    p,
)
```

The helper returns:

```text
1.0                      if p == 0 or dropout is disabled
1.0 / (1.0 - p)          if hash-derived u >= p
0.0                      otherwise
```

Call sites skip the helper entirely when dropout is disabled. When dropout is
enabled, multiply the LoRA A input element by `scale`; do not multiply base
projection inputs.

### Torch Oracle Algorithm

For each token and routed expert slot:

1. Compute `global_token_id` as the row index in the pre-packed hidden tensor.
2. Compute `route_slot` as the top-k position selected for that token.
3. Gate LoRA path:
   - apply projection id `0`
   - scale hidden features before `gate_lora_A`
4. Up LoRA path:
   - apply projection id `1`
   - scale hidden features before `up_lora_A`
5. Down LoRA path:
   - apply projection id `2`
   - scale post-SwiGLU intermediate features before `down_lora_A`
6. Backward recomputes with the cached `(dropout_enabled, dropout_seed)`.

### ARM Native Algorithm

`operators/arm/bf16_sft_moe.hpp` keeps the current route loop for p=0. For
p>0:

1. Thread `dropout_enabled` and `dropout_seed` through `forward_sft_binding`,
   `forward_sft_task`, `ForwardTask`, `forward_inner`, and `forward_impl`.
2. Store both values in `CacheEntry`.
3. Pass `global_token_id` and `route_slot` to `compute_route`.
4. In `compute_route`, apply counter dropout to `x[h]` before gate/up LoRA A and
   to `act[i]` before down LoRA A.
5. In `backward_impl`, use the cached seed/state and multiply only LoRA
   input-gradient terms by the same dropout scale.

### Backend Guard Algorithm

LLaMAFactory owns unsupported-backend rejection:

1. Canonicalize `model_args.kt_backend` to uppercase.
2. If `finetuning_args.lora_dropout == 0`, keep existing local KT checks and
   allow the backend.
3. If `finetuning_args.lora_dropout > 0`, allow only backends in a
   `SUPPORTED_LOCAL_KT_LORA_DROPOUT_BACKENDS` constant.
4. Populate that constant as stages pass:
   - after Stage 2: `TORCHBF16`, `TORCHBF16_SFT`
   - after Stage 3: add `ARMBF16`, `ARMBF16_SFT`, `KT_ARM`
5. Always reject AMX and other skip/no-LoRA backends for nonzero dropout in this
   ARM plan because they are outside the validated target set.
6. Shell scripts pass requested dropout values through to LLaMAFactory and do not
   implement a second KT-specific dropout policy.

## Files To Change Or Create

### KTransformers Python

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/config.py`:

- Add `kt_lora_dropout: float | None = None` to `KTConfig`.
- In `KTConfig.__post_init__`, read
  `ACCELERATE_KT_LORA_DROPOUT` with `_env_float`.
- Validate default behavior: if unset, set `kt_lora_dropout = 0.0`.

Create `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/dropout.py`:

- `normalize_lora_dropout(value: float | None) -> float`
- `next_lora_dropout_seed(enabled: bool) -> int`
- `counter_dropout_scale(...) -> torch.Tensor`
- `counter_dropout_apply(...) -> torch.Tensor`
- Python splitmix64 helper used by tests and by `TorchBF16SFTMoEWrapper`.

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/base.py`:

- Extend `BaseSFTMoEWrapper.__init__` with
  `lora_dropout: float = 0.0`.
- Extend `_validate_sft_config(lora_rank, lora_alpha, max_cache_depth)` to also
  validate `lora_dropout`.
- Store:
  - `self.lora_dropout`
  - `self._forward_cache_dropout: list[tuple[bool, int]]`
- Add method:
  - `_next_forward_dropout_state(training: bool) -> tuple[bool, int]`
- Change abstract method:
  - `_make_forward_task(self, buffer, save_for_backward, dropout_enabled, dropout_seed)`
- Change public methods:
  - `forward(..., save_for_backward=True, output_device=None, training=True)`
  - `submit_forward(..., save_for_backward=True, training=True)`
- Pass `dropout_enabled` and `dropout_seed` into `_make_forward_task`.

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/torch_backend.py`:

- Extend `TorchBF16SFTMoEWrapper.__init__` with `lora_dropout: float = 0.0`.
- Add `_next_forward_dropout_state(training: bool)`.
- Change `_compute(...)` to accept:
  - `dropout_enabled: bool`
  - `dropout_seed: int`
  - token ids and route ids for each expert selection
- Apply `counter_dropout_apply` before gate/up/down LoRA A matmuls.
- Store `(hidden, expert_ids, weights, dropout_enabled, dropout_seed)` in
  `_forward_cache`.
- In `backward`, recompute using the cached dropout state.

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/arm.py`:

- Extend `ArmBF16SFTMoEWrapper.__init__` with `lora_dropout`.
- Pass it to `BaseSFTMoEWrapper.__init__`.
- In `load_weights`, set `config.lora_dropout`.
- Change `_make_forward_task` to pass `dropout_enabled` and `dropout_seed` to
  `self.moe.forward_sft_task`.

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/wrapper.py`:

- In the wrapping function, read
  `lora_dropout = getattr(cfg, "kt_lora_dropout", 0.0) or 0.0`.
- Pass `lora_dropout=lora_dropout` into `KTMoEWrapper`.
- In `_build_kt_plugin_from_args`, pass
  `kt_lora_dropout=getattr(finetuning_args, "lora_dropout", None)`.

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/python/sft/layer.py`:

- In `KTMoELayerWrapper._submit_and_compute_gpu`, pass
  `training=self.training` into `wrapper.submit_forward` on the single-GPU and
  rank-0 distributed paths.
- Do not key dropout off `save_for_backward`; checkpoint first forward still
  needs dropout.

### KTransformers C++

Create `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/operators/sft_dropout.hpp`:

- `validate_lora_dropout(float p)`
- `uint64_t splitmix64(uint64_t x)`
- `uint64_t lora_dropout_key(...)`
- `float lora_dropout_scale(...)`
- Constants for projection ids:
  - `KT_LORA_DROPOUT_GATE = 0`
  - `KT_LORA_DROPOUT_UP = 1`
  - `KT_LORA_DROPOUT_DOWN = 2`

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/operators/common.hpp`:

- Add `float lora_dropout = 0.0f` to `MOESFTConfig`.

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/ext_bindings.cpp`:

- Bind `MOESFTConfig::lora_dropout`.
- Extend `ForwardSFTBindings::Args` with:
  - `bool dropout_enabled`
  - `uint64_t dropout_seed`
- Extend `ForwardSFTBindings::inner` and `cpuinfer_interface` to call
  `TP_MOE_SFT<T>::forward_sft_binding(..., save_for_backward,
  dropout_enabled, dropout_seed)`.
- Extend the ARM pybind `forward_sft_task`/`forward_sft` signatures the same way.

Change `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`:

- Include `operators/sft_dropout.hpp`.
- Add `lora_dropout_` validation from `config.lora_dropout`.
- Extend:
  - `forward_sft_binding`
  - `forward_sft_task`
  - `ForwardTask`
  - `forward_inner`
  - `forward_impl`
- Store `dropout_enabled` and `dropout_seed` in `CacheEntry`.
- Change `compute_route` to accept token id, route slot, dropout enabled, and
  seed. Apply counter dropout to:
  - `x[h]` for gate/up LoRA A
  - `act[i]` for down LoRA A
- Change `backward_impl` so the recomputed route uses cached dropout state and
  the LoRA input-gradient terms multiply by the same dropout scale.

### LLaMAFactory

Change `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/hparams/model_args.py`:

- In `get_kt_config_dict`, add:
  - `"kt_lora_dropout": getattr(finetuning_args, "lora_dropout", None)`
- In `apply_kt_config.env_mapping`, add:
  - `"kt_lora_dropout": "ACCELERATE_KT_LORA_DROPOUT"`

Change `/workspace/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/hparams/parser.py`:

- Add `SUPPORTED_LOCAL_KT_LORA_DROPOUT_BACKENDS` near `_LOCAL_KT_BACKENDS`.
- Replace the unconditional `lora_dropout != 0` KT error with:
  - `lora_dropout == 0`: preserve existing local KT behavior.
  - `lora_dropout > 0` and backend in the supported constant: allow.
  - `lora_dropout > 0` and backend not in the supported constant: raise
    `ValueError` naming the backend and listing validated backend names.
- Do not add AMX backend names to the supported dropout constant in this ARM
  implementation.

### AsymGEMM Profiling Scripts

Change `/workspace/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/run_lf_lora_sft.sh`:

- Remove the KT-only guard that rejects nonzero `LORA_DROPOUT` after the backend
  support check is in LLaMAFactory.
- Keep the public KT backend labels as `kt_torchbf16` and `kt_armbf16`.
- Do not add `kt_amxbf16` to the ARM profiling scripts.

Change `/workspace/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`:

- Keep `MODEL_SPECS` as `model|num_gpus`. Do not add recompute to model specs.
- Keep recompute only in `BACKEND_SPECS` entries as `backend|recompute`.
- Remove the KT-only block that forces default dropout to `0.00`.
- Remove the KT-only block that rejects `--lora-dropout` values other than
  `0.00`.
- Remove `LORA_DROPOUT_ENV_SET`, `lora_dropout_user_set`, and the argument-parser
  assignments that only feed that KT override.
- Set the sweep default to:
  - `LORA_DROPOUT=${LORA_DROPOUT:-0.00,0.10}`
- Keep `backend_label`, `backend_gpu_count`, `expand_backend_spec`, and help text
  limited to `kt_torchbf16` and `kt_armbf16` for KT backends.
- Keep source/nsys plot separation unchanged.
- Do not add custom run names. Validation must use `OVERWRITE=true` and the
  existing output directory layout.

### Tests And Validation Scripts

Create `/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py`:

- Stage 1 creates this file with config/env/plumbing tests:
  - `KTConfig(kt_lora_dropout=0.1)` works.
  - `ACCELERATE_KT_LORA_DROPOUT=0.1` works.
  - invalid dropout values reject.
  - wrapper constructors accept and store `lora_dropout`.
- Stage 2 extends this file with:
  - Python/C++ mask helper parity for fixed counters.
  - `TorchBF16SFTMoEWrapper` forward/backward against a manual reference.
  - checkpoint-style two-forward behavior: first forward with
    `save_for_backward=False`, second with `save_for_backward=True`, backward
    uses identical masks.
- Stage 3 extends this file with ARM backend reference checks.

Create `/workspace/AsymGEMM-SFT/third_party/AsymGEMM/scripts/testing/validate_kt_lora_dropout.py`:

- Generate deterministic tiny MoE cases.
- Run manual reference plus the requested backend.
- If the requested native backend is unavailable on the current host, fail before
  comparisons with a clear message naming the missing backend and required host
  capability.
- Support cases:
  - `--dropout 0.00`
  - `--dropout 0.10`
  - `--checkpoint off`
  - `--checkpoint on`
  - `--training true`
  - `--training false`
  - `--profile`
- Report max abs and max relative errors for output, grad input, grad weights,
  and all six LoRA gradient tensors.

Acceptance tolerances:

```text
Torch oracle vs manual float32 reference:
  output max_abs <= 2e-2, max_rel <= 2e-2
  gradients max_abs <= 3e-2, max_rel <= 3e-2

Native BF16 vs torch oracle:
  output max_abs <= 6e-2, max_rel <= 5e-2
  gradients max_abs <= 8e-2, max_rel <= 8e-2

All cases:
  no NaN or Inf
  p=0 native fast-path output unchanged from pre-dropout baseline
```

## Stage Plan And Gates

### Stage 0: Baseline Capture

Purpose: prove the current no-dropout path before editing native kernels.

Do:

1. Save pre-change p=0 validation output for `kt_torchbf16`.
2. Save pre-change p=0 profile artifacts with `profile_lora_lf.sh`.
3. Record current rejection behavior for KT p=0.10.

Commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
KT_KERNEL_DIR=/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="kt_torchbf16|recompute" \
PROFILERS="source,nsys" \
SEQ_LENS="4096" \
LORA_DROPOUT="0.00" \
MAX_SAMPLES=128 MAX_STEPS=10 WARMUP_STEPS=5 \
OVERWRITE=true CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh
```

Gate:

- p=0 profile completes.
- `profile.json`, `summary.md`, `source_profile.json`, source memory plots, and
  nsys plots are present.
- p=0.10 still fails before implementation, proving the guard exists.

### Stage 1: Config And Script Plumbing

Purpose: carry `lora_dropout` from LF args into KT wrappers and native config
without changing math.

Do:

1. Add `kt_lora_dropout` to KT config and env mapping.
2. Pass dropout through `wrapper.py` into all wrappers.
3. Add `MOESFTConfig.lora_dropout` and pybind exposure.
4. Extend forward task signatures but keep native code behavior unchanged while
   `dropout_enabled` is ignored.
5. Keep LLaMAFactory and shell guards in place for each backend until that
   backend's validation gate passes and Stage 4 updates the supported-backend
   constant.
6. Create the initial `test_sft_lora_dropout.py` with config/env/plumbing tests
   only.

Validation:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
python -m pytest test/per_commit/test_sft_lora_dropout.py -q -k "config or plumbing"
python -m pytest test/per_commit/test_torchbf16_sft_wrapper_lifecycle.py -q
```

Gate:

- `KTConfig(kt_lora_dropout=0.1)` works.
- `ACCELERATE_KT_LORA_DROPOUT=0.1` works.
- Invalid `-0.1` and `1.0` reject.
- p=0 profiling from Stage 0 remains within 2% median step time.

### Stage 2: Deterministic Mask Helper And Torch Oracle

Purpose: create the correctness oracle before native kernel edits.

Do:

1. Implement Python counter dropout in `python/sft/dropout.py`.
2. Implement matching C++ helper in `operators/sft_dropout.hpp`.
3. Update `TorchBF16SFTMoEWrapper` to apply gate/up/down dropout.
4. Cache dropout seed/state for backward.
5. Extend `test_sft_lora_dropout.py` with mask parity, torch oracle, and
   checkpoint-style tests.

Validation:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
python -m pytest test/per_commit/test_sft_lora_dropout.py -q -k "mask or torch"

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
python scripts/testing/validate_kt_lora_dropout.py --backend kt_torchbf16 --dropout 0.00 --checkpoint off
python scripts/testing/validate_kt_lora_dropout.py --backend kt_torchbf16 --dropout 0.10 --checkpoint off
python scripts/testing/validate_kt_lora_dropout.py --backend kt_torchbf16 --dropout 0.10 --checkpoint on
python scripts/testing/validate_kt_lora_dropout.py --backend kt_torchbf16 --dropout 0.10 --training false
```

Gate:

- Mask parity test passes between Python and C++ helper.
- `kt_torchbf16` matches manual reference within tolerance.
- Checkpoint on/off produces correct gradients.
- Eval mode with p=0.10 equals p=0 behavior.

### Stage 3: ARM Native Backend

Purpose: make the user-facing `kt_armbf16` path correct for dropout.

Do:

1. Modify `operators/arm/bf16_sft_moe.hpp` with dropout-aware
   `compute_route` and `backward_impl`.
2. Store `dropout_enabled` and `dropout_seed` in `CacheEntry`.
3. Update `python/sft/arm.py` to pass the forward seed/state.
4. Keep `lora_dropout == 0` branch branch-free inside route math.

Validation on an aarch64 ARM host:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
python -m pytest test/per_commit/test_armbf16_sft_reference.py -q
python -m pytest test/per_commit/test_sft_lora_dropout.py -q -k "arm"

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
python scripts/testing/validate_kt_lora_dropout.py --backend kt_armbf16 --dropout 0.00 --checkpoint off
python scripts/testing/validate_kt_lora_dropout.py --backend kt_armbf16 --dropout 0.10 --checkpoint off
python scripts/testing/validate_kt_lora_dropout.py --backend kt_armbf16 --dropout 0.10 --checkpoint on
python scripts/testing/validate_kt_lora_dropout.py --backend kt_armbf16 --dropout 0.10 --profile
```

Gate:

- ARM p=0 matches pre-change ARM behavior within existing tolerance.
- ARM p=0.10 matches `kt_torchbf16` oracle within tolerance.
- ARM checkpoint test passes.
- ARM p=0.10 microprofile overhead is recorded.
- Do not remove the `kt_armbf16` nonzero-dropout guard until this gate passes on
  an aarch64 host.

### Stage 4: Remove Guards Backend-By-Backend

Purpose: expose support only for backends that passed validation.

Do:

1. Replace the LLaMAFactory unconditional KT dropout error with a backend support
   check.
2. Allow nonzero dropout for validated backends only by editing
   `SUPPORTED_LOCAL_KT_LORA_DROPOUT_BACKENDS`.
3. Remove the shell-script KT p=0-only guards after LLaMAFactory owns the backend
   support check.
4. Set `profile_lora_lf.sh` default dropout sweep to `0.00,0.10`.
5. Keep rejection messages for unsupported KT backends specific:

```text
Local KT backend <BACKEND> does not yet support lora_dropout > 0.
Validated backends: <comma-separated values from SUPPORTED_LOCAL_KT_LORA_DROPOUT_BACKENDS>.
```

Stage 4 may be executed after Stage 2 with only `TORCHBF16` enabled, or after
Stage 3 with `TORCHBF16` and `ARMBF16` enabled. Do not add a backend to the
supported constant before its validation gate passes.

Validation:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
KT_KERNEL_DIR=/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="kt_torchbf16|recompute" \
PROFILERS="source" \
SEQ_LENS="4096" \
LORA_DROPOUT="0.10" \
MAX_SAMPLES=16 MAX_STEPS=2 WARMUP_STEPS=5 \
OVERWRITE=true CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh
```

Gate:

- KT p=0.10 reaches LF training.
- Unsupported KT backend, if selected, fails before training with a clear
  backend-specific message.
- p=0 guard removal does not change p=0 profiling output layout.

### Stage 5: Full Smoke Profiles

Purpose: prove correctness and latency with the same profiling scripts used for
the actual experiments.

Do not use custom run names. Use `OVERWRITE=true`.

Each smoke command must include only backends that are validated on the ARM host
running the command.

Run this p=0 cross-backend correctness profile on a host where `torch`,
`kt_torchbf16`, and `kt_armbf16` are all validated:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
KT_KERNEL_DIR=/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="torch|recompute,kt_torchbf16|recompute,kt_armbf16|recompute" \
PROFILERS="source,nsys" \
SEQ_LENS="4096" \
LORA_DROPOUT="0.00" \
COMPARE_LOSSES=true \
COMPARE_BASELINE_BACKEND=torch \
COMPARE_CANDIDATE_BACKEND=kt_torchbf16,kt_armbf16 \
MAX_SAMPLES=128 MAX_STEPS=10 WARMUP_STEPS=5 \
OVERWRITE=true CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh
```

Run this p=0.10 KT deterministic-mask profile on a host where `kt_torchbf16`
and `kt_armbf16` are both validated:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
LF_DIR=/workspace/AsymGEMM-SFT/third_party/LlamaFactory \
KT_KERNEL_DIR=/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="kt_torchbf16|recompute,kt_armbf16|recompute" \
PROFILERS="source,nsys" \
SEQ_LENS="4096" \
LORA_DROPOUT="0.10" \
COMPARE_LOSSES=true \
COMPARE_BASELINE_BACKEND=kt_torchbf16 \
COMPARE_CANDIDATE_BACKEND=kt_armbf16 \
MAX_SAMPLES=128 MAX_STEPS=10 WARMUP_STEPS=5 \
OVERWRITE=true CONTINUE_ON_ERROR=false \
scripts/lf/profile_lora_lf.sh
```

Do not compare p=0.10 KT losses against the normal `torch` backend unless the
normal torch backend is changed to use the exact same KT counter masks. PEFT
dropout and KT counter dropout are both valid dropout, but they are different
random streams.

Gate:

- `comparisons/loss_compare.tsv` is all `ok` for p=0.
- `comparisons/loss_compare.tsv` is all `ok` for p=0.10 among KT backends that
  share the counter-mask implementation.
- Source memory plots exist under per-run `memory_plots/`, config
  `memory_combined/`, and precision-root `memory_combined/`.
- Nsys timing/interconnect plots exist under per-run `plots/`, config
  `combined/` and `c2c_combined/`, and precision-root `combined/` and
  `c2c_combined/`.
- p=0 measured step time regression is <= 2%.
- p=0.10 measured step overhead is recorded and <= 5% target.
- No run directories are renamed.

### Stage 6: Documentation And Release Check

Purpose: make the final behavior understandable and prevent future regression.

Do:

1. Add a short note in the KT SFT docs describing supported dropout backends.
2. Add a note in the profiling script help that KT supports `--lora-dropout`
   for validated backends.
3. Keep `source` memory plots separate from `nsys` timing plots.
4. Add a final test matrix to the PR or commit notes:

```text
backend       p=0 correctness  p=0.10 correctness  checkpoint  profile latency
kt_torchbf16  pass             pass                pass        recorded
kt_armbf16    pass             pass                pass        recorded
```

Gate:

- `git diff` shows only intended files.
- p=0 and p=0.10 validation commands above pass.
- Full smoke profile artifacts are present.
- Unsupported backend behavior is explicit.

## Efficiency Notes

- Do not materialize dense dropout masks for native ARM.
- Do not store dropout masks in the forward cache.
- Store only `dropout_enabled` and `dropout_seed` in caches.
- Generate dropout scale on the fly from counters.
- Preserve the old path for `p=0`.

## Final Acceptance Criteria

The work is complete only when:

- `lora_dropout=0.00` remains behaviorally and performance equivalent to the
  current KT path.
- `lora_dropout=0.10` trains through LF with KT backends that have passed their
  validation stage.
- Forward, backward, checkpoint recompute, eval mode, and invalid-input tests
  pass.
- `profile_lora_lf.sh` runs with existing directory names and `OVERWRITE=true`.
- Source memory plots and nsys timing/interconnect plots are both generated and
  not mixed into each other's plot inputs.
- Any backend not validated for nonzero dropout still fails fast with a clear
  message.

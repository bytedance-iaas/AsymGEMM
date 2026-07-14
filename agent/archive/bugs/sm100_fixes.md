# SM100 LoRA-SFT Fix Plan: Items 3-6

Scope: only the LoRA-SFT implementation issues discussed as items 3-6. Kernel accumulation and `num_groups` are intentionally excluded here.

## 3. Remove Grouped Path Host Syncs

### 3.1 `frozen_linear.py` grouped padding/unpadding

Problem:
- `_pad_grouped_input_for_asym` converts CUDA `offsets` to CPU with `offsets.detach().to("cpu").tolist()`.
- It then computes padding metadata in Python and copies each group with Python loops.
- This blocks the CUDA stream every grouped expert call and adds per-layer Python overhead.

Needed change:
- Keep grouped metadata on CUDA for the asym path.
- Build padded start/end/count tensors on device.
- Gather padded rows with device operations, not Python loops.
- Preserve an inverse row mapping so output can be scattered/unpadded on device.
- Prefer a kernel-side fix if feasible: teach the grouped GEMM to consume cumulative offsets directly and handle ragged group sizes, avoiding physical padding.

Acceptance checks:
- No `.to("cpu")`, `.tolist()`, or Python per-group copy loop on the production asym grouped path.
- Grouped forward and dx numerics match the current torch backend.
- Profiling shows the grouped path no longer has the padding-induced host sync before each launch.

### 3.2 `moe.py` route validation `.item()` syncs

Problem:
- `_validate_route_inputs` calls `topk_indices.min().item()` and `topk_indices.max().item()`.
- On CUDA, `.item()` forces a GPU-to-CPU sync.
- Qwen3 calls `build_contiguous_route_metadata` every forward, so this is on the production routing path.

Needed change:
- Move min/max expert-id validation behind a debug/test flag.
- Skip the validation in production learned-router execution, because router `top_k` output is already constrained by the model.
- Keep shape and dtype normalization checks that do not synchronize.

Acceptance checks:
- Production Qwen3 forward does not call `.item()` for route min/max validation.
- Debug mode can still catch invalid expert ids.
- Existing route metadata tests still validate bad inputs.

## 4. Fix Qwen Asym+Offload Recompute Granularity

Problem:
- In `qwen3_moe.py`, the asym+offload recompute branch is all-or-nothing.
- If any expert group matches `tokN-ckpt`, the code checkpoints the whole packed expert body.
- If any group matches activation-drop policy, the code applies it to the whole packed expert body.
- `splitN` effectively provides no useful split behavior in this branch.

Needed change:
- Remove the asym+offload special-case dense shortcut.
- Reuse the same group-subset structure as the non-offload path:
  - compute `active_groups`
  - compute `split_groups`
  - compute `recompute_groups`
  - compute `activation_drop_groups`
  - compute `kept_groups`
- For each non-empty group set, call `_select_subset`.
- Run normal body for kept/split groups.
- Run checkpointed body only for `recompute_groups`.
- Run checkpointed activation/down only for `activation_drop_groups`.
- Scatter subset outputs back into the original packed-row order with `index_copy_`.

Acceptance checks:
- `none`, `splitN`, `tokN-ckpt`, and `tokN-act-ckpt` all preserve output and gradient parity against the torch backend.
- A policy selecting one expert does not checkpoint every expert.
- Runtime/profile metadata makes it clear which groups were recomputed.

## 5. Separate Qwen Gate/Up LoRA Dropout Masks

Problem:
- Qwen packed expert LoRA currently applies one dropout to `x`, then feeds the same dropped tensor into both gate and up LoRA branches.
- With `lora_dropout > 0`, strict PEFT behavior expects independent dropout draws per LoRA branch.
- With `lora_dropout == 0`, this has no effect.

Needed change:
- For strict PEFT parity, apply dropout separately for gate LoRA and up LoRA.
- Keep the current fused/concatenated fast path when `lora_dropout == 0`.
- When `lora_dropout > 0`, split the gate and up LoRA A projections or otherwise generate independent masks before the branch-specific A projections.
- Preserve checkpoint RNG behavior so recompute uses the same dropout masks during backward.

Acceptance checks:
- `lora_dropout == 0` output remains unchanged and still uses the fast fused path.
- `lora_dropout > 0` uses independent masks for gate and up.
- Checkpointed and non-checkpointed runs remain deterministic under a fixed seed.
- PEFT parity tests cover dropout > 0.

## 6. Fix LF Launcher/Parser Asym DDP Handling

Problem:
- `launcher.py` decides whether to auto-launch DDP before YAML is parsed.
- It only checks `USE_ASYM_GEMM=1`, not YAML `use_asym_gemm: true`.
- Direct YAML usage on a multi-GPU machine can therefore auto-spawn DDP before asym guards run.
- The provided script is safe because it exports `USE_ASYM_GEMM=1`, but direct launcher usage remains unsafe.

Needed change:
- Add launcher-side detection for `use_asym_gemm: true` before the auto-DDP decision.
- Detection should check:
  - `USE_ASYM_GEMM` env first
  - train config YAML/JSON file if present in CLI args
  - explicit CLI key/value args if supported by the launcher path
- If asym is detected, suppress auto-DDP launch.
- Add parser-side guard that explicitly rejects unsupported distributed/asym combinations if the user manually launches torchrun/DDP.
- After full argument parsing, export `USE_ASYM_GEMM=1` for child-process consistency when `model_args.use_asym_gemm` is true.

Acceptance checks:
- Multi-GPU direct YAML run with `use_asym_gemm: true` does not auto-DDP.
- Manual torchrun/DDP with asym fails early with a clear error.
- Existing script path with `USE_ASYM_GEMM=1` still works.
- KT and non-asym launcher behavior is unchanged.

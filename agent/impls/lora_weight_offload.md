# LoRA Weight Offload Plan

Goal: add an opt-in LoRA **weight** offload path for single-GPU AsymGEMM LoRA SFT. Keep the trainable expert-LoRA adapter weights resident in **pinned CPU memory** and stage each layer's weights to CUDA just-in-time (JIT) for that layer's forward and backward, then release the CUDA copy. This is the DeepSpeed ZeRO-3 `offload_param` "gather before use / release after use" pattern specialized to single GPU (no collectives) and to LoRA adapters only. The target is to remove the ~6.19 GiB of always-resident expert-LoRA weights from the loss/lm-head peak, which `lora_grad_offload.md` could not move.

This plan composes with, and assumes, the already-implemented LoRA grad offload (`asym_cpu_adamw_grad_offload=true`). Weight offload reuses the grad-offload post-accumulate hook as the single per-parameter "release" point in backward.

### Primary goal, the toggle, and how success is measured

- **PRIMARY GOAL (hard target): with LoRA weight offload enabled (on top of grad offload), peak allocated HBM for the real Qwen3-30B-A3B `b4_s4096` acceptance workload must be `< 30 GiB`**, down from the `34.593 GiB` baseline where only grad offload is on. This single number decides success. Expected landing ≈ `28.5 GiB`, which leaves margin under the `30 GiB` line.
- **A toggle is mandatory, and the profiler axis is a list.** All offload behavior sits behind a default-off per-run flag `asym_cpu_adamw_weight_offload`. The `profile_lora_lf.sh` sweep axis `ASYM_CPU_ADAMW_WEIGHT_OFFLOADS` accepts a comma-separated list of bools — set **`ASYM_CPU_ADAMW_WEIGHT_OFFLOADS=false,true`** to run both modes **sequentially in one command, under one output root**, each in its own `__weightoff{false,true}` job dir. That single invocation is the apples-to-apples A/B that measures the design's efficacy (and lets us turn it off if it ever regresses). This mirrors the already-shipped `ASYM_CPU_ADAMW_GRAD_OFFLOADS` axis.
- **Validation policy — `scripts/lf/profile_lora_lf.sh` is the only decision authority.** The metric that accepts or rejects the design (peak allocated HBM and step latency) must come from the real end-to-end `profile_lora_lf.sh` run on Qwen3-30B-A3B. Unit tests and tiny-model checks are *necessary but never sufficient*: they only gate correctness (numerics, residency, stream safety). No stage is "done", and the `< 30 GiB` goal is never considered met, on toy/unit evidence alone — the deciding number always comes from `profile_lora_lf.sh`.

Implementation status:

- Planned, not yet implemented. This document is the design + staging plan. Each stage below has its own acceptance gate; the binding acceptance is the Stage 5 e2e A/B (meaningful peak-HBM reduction AND no latency blow-up).
- The optimizer already anticipates this work and currently blocks it: `AsymCPUAdamW.__init__` and `_check_post_prepare_devices` raise if a trainable LoRA compute param is not CUDA-resident, with the message "CPU-resident trainable LoRA is Stage 7 and is not supported by CPUAdamW v1." See `asym_gemm/training/cpu_adam.py:133-141` and `:277-285`. **Stage 2 of this plan relaxes these guards** behind the new toggle. (That quoted "Stage 7" is the original code authors' placeholder label in the error string; it is unrelated to this plan's stage numbering.)

## Why grad offload missed the peak and weight offload can hit it

- The measured peak (`34.593 GiB`) occurs during cross-entropy/lm-head **backward**, the very start of `loss.backward()`. The CUDA memory snapshot for the grad-offload debug run (`profiling/lora_grad_offload_peakdebug_20260615T083310Z`) shows the peak live set is three fp32 `[4*4096, 151936]` vocab tensors (one framed `cross_entropy`/`loss` block `9.273 GiB` + two allocator-unframed `9.273 GiB` backward temporaries = `27.82 GiB`) plus `6.221 GiB` of long-lived model CUDA params, plus small norm/attn/expert blocks.
- That `6.221 GiB` of long-lived params is dominated by the trainable **expert-LoRA weights** (`3,321,888,768` params, bf16, `6.187 GiB` / `6336 MiB` on CUDA; attention LoRA is only `~102 MiB`). These are persistent `nn.Parameter`s, so they are resident at the loss peak even though no expert layer's forward or backward is executing at that instant.
- Grad offload removed end-of-backward grad residency (`after_backward` live `13.012 → 6.725 GiB`, `cuda_grad_bytes 6.287 → 0`) but did not touch the peak, because grads are not live at the loss peak. Weight offload targets exactly the bytes that *are* live at the loss peak: the resident expert-LoRA weights.
- Timeline argument: with JIT gather/release, by the time forward reaches lm_head/loss, every decoder layer's expert-LoRA weights have been released (used and freed during that layer's forward). At the loss peak, expert-LoRA resident bytes drop from `6.187 GiB` to ~0 (plus at most a small in-flight prefetch buffer). The new global peak is the cross-entropy floor plus a small staging high-water.

## Facts resolved from local code and docs

- Expert-LoRA weights are six trainable 3-D `nn.Parameter` banks on `AsymQwen3Experts`: `gate_lora_A/B`, `up_lora_A/B`, `down_lora_A/B`, created on the model device in `lora_dtype` (bf16). Shapes are `[num_experts, rank, in]` (A) and `[num_experts, out, rank]` (B). See `asym_gemm/training/qwen3_moe.py:1867` (class) and `:1968-1977` (param creation). For Qwen3-30B-A3B: 48 layers × 128 experts × rank 64 ⇒ `3,321,888,768` params ⇒ `6.187 GiB` bf16 ⇒ ≈ `132 MiB` per decoder layer.
- The expert forward used by the acceptance config (`ASYMM_EXPERT_ACT_OFFLOAD=1`) is the custom autograd Function `_ActivationOffloadQwen3ExpertFunction`. The six LoRA weights are passed as positional **differentiable** inputs to `.apply(...)` (`asym_gemm/training/qwen3_moe.py:2346-2354`), are `save_for_backward`-ed (`:1082-1091`), are read back from `ctx.saved_tensors` in backward (`:1099-1108`), and their grads are returned positionally (`:1320-1331`). Autograd then accumulates those grads into `self.<bank>.grad`, which is what the grad-offload post-accumulate hook consumes.
- Consequence: naively setting `param.data = empty(0)` after forward would either keep the storage alive (because `save_for_backward` holds the GPU tensor) or silently corrupt backward (because `.data` reassignment bypasses autograd version checks). Weight offload therefore requires (a) releasing the `nn.Parameter.data` storage to actually reclaim HBM, and (b) modifying the Function so backward reads the *re-staged* weights from `ctx.layer` rather than from `ctx.saved_tensors`. This is exactly ZeRO-3's "do not keep the gathered param across the fwd→bwd gap; re-gather before backward."
- AsymGEMM already streams a CPU-resident **weight** into a GEMM for frozen base weights: `m_grouped_bf16_asym_gemm_nt_contiguous` consumes a pinned-CPU `b_cpu` weight (`asym_gemm/training/frozen_linear.py`, `_asym_bf16_nt`), and separately streams CPU **activations** via `sm100_m_grouped_bf16_cpu_left_asym_gemm_nt_contiguous` (`asym_gemm/training/cpu_left.py:16,141`). So a "stream the LoRA weight from CPU, never materialize it in HBM" variant is *plausible* (Stage 7a) but is **NOT proven applicable to trainable LoRA** and is not assumed by this plan: the frozen path computes grad_x only, whereas trainable LoRA-B also needs grad_weight, and the kernel's NT-contiguous / grouped / transpose conventions may force a change to the LoRA compute math. Stage 2 therefore uses the simpler, certain ZeRO-3 stage-to-HBM approach to hit the `< 30 GiB` goal; CPU-weight streaming is deferred to Stage 7a behind its own math-design + isolated-kernel validation gate.
- The optimizer keeps, per unique LoRA param, a `_ParamMapping(cuda_param, cpu_param, grad_buffer, ...)` where `cuda_param` is the bf16 CUDA compute weight and `cpu_param` is the fp32 CPU master (`asym_gemm/training/cpu_adam.py:69-81`, `:188-206`). After step it refreshes `cuda_param.data.copy_(cpu_param.data, ...)` (`:395-398`, called at `:478-483`). Weight offload replaces this copy-back target with a pinned bf16 CPU "home" and lets the coordinator own `cuda_param.data` residency.
- Existing offload idioms to reuse: pinned host tensors and a shape-keyed CPU buffer pool (`asym_gemm/training/activation_offload.py`), `torch.autograd.graph.saved_tensors_hooks` pack/unpack with `torch.cuda.Event` ready-events (`asym_gemm/training/attention_activation_offload.py:232-328`, install at `:370`), and `HostWeight` as the "weight lives on pinned CPU, staged with `.to(device=..., non_blocking=True)`" template (`asym_gemm/training/offload.py`/`host_weight.py`). There is currently **no** CUDA side-stream / prefetch primitive in the training package — Stage 3 introduces one.
- Existing toggles to mirror: `ASYMM_EXPERT_ACT_OFFLOAD` (`asym_gemm/training/qwen3_moe.py:2267`), `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD` (`:230-239`). Decoder layers are a standard `model.model.layers` `ModuleList`; the offloaded experts module is built by `wrap_qwen3_experts` → `AsymQwen3MoeBlock` (`:2686`, `:2733`).
- Naive saved-tensor offload (wrapping the Function in `saved_tensors_hooks` to push the saved weights to CPU) does **not** help: the live `nn.Parameter.data` is always resident regardless of what backward saved, so the `6.187 GiB` persists. Only releasing `param.data` reclaims it. (Considered and rejected.)
- DeepSpeed ZeRO-3 reference design (local `third_party/deepspeed/deepspeed/runtime/zero/`):
  - Free a param by replacing `.data` with an empty tensor and flipping availability: `partition_parameters.py:300-316` (`free_param`); persistence threshold keeps tiny params resident: `:1196-1203`.
  - Coordinator records a forward access **trace** on step 0, freezes it, validates each step with a release-everything fallback on mismatch: `partitioned_param_coordinator.py:187-204`, `:236-272`. One frozen trace serves both forward and reverse-order backward.
  - Gather-before-use with a cross-stream wait (not global sync): `partitioned_param_coordinator.py:365-391`; pre-backward re-gather via the same fetch: `parameter_offload.py:530-536`. Release-after-use is reference-counted by active submodules: `partitioned_param_coordinator.py:562-570`; hooks at `parameter_offload.py:321-366,408-411`.
  - Prefetch the next bucket on a side stream bounded by a byte budget; depth-2 event throttle as the pragmatic "double buffer": `partitioned_param_coordinator.py:396-461`, `:122-133,376-387`.
  - Pin once at offload time (pinning blocks the host): `partition_parameters.py:1721-1722`. `record_stream` so the caching allocator does not recycle a cross-stream buffer: `:769-771`.
- Authoritative docs: ZeRO-Infinity (overlap-centric prefetch, fwd/bwd tracing) https://arxiv.org/pdf/2104.07857 ; DeepSpeed config knobs (`stage3_*`) https://www.deepspeed.ai/docs/config-json/ ; ZeRO-3 API https://deepspeed.readthedocs.io/en/latest/zero3.html ; PyTorch pinned-memory / `non_blocking` correctness https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html ; `register_post_accumulate_grad_hook` https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html .

## What "LoRA weight offload (gather/release)" means here

- "Home": each offloaded expert-LoRA bank's canonical bf16 value lives in a pinned CPU tensor. The optimizer's fp32 master stays the training source of truth; after each `step()` the home is refreshed from the master on the CPU (a cheap cast, no PCIe/C2C traffic).
- "Gather": before a layer's experts run, copy that layer's six homes into reusable CUDA staging buffers on a side stream, point `param.data` at the staged buffers, make the compute stream wait on the copy, and `record_stream` the buffers.
- "Release": after the consuming forward (and, separately, after backward grad accumulation), set `param.data` back to a 0-size placeholder and return the staging buffers to the pool. This is the step that actually reclaims HBM.
- "Re-gather for backward": because the custom Function does not keep the gathered GPU weight across the fwd→bwd gap, the coordinator re-gathers each layer's weights just before that layer's backward, and the Function reads them from `ctx.layer`.
- "Prefetch": using the step-0 access trace, start the next layer's gather on the side stream while the current layer computes, so transfers overlap compute (Stage 3).
- Scope: **expert-LoRA only.** Attention LoRA (`~102 MiB` total) is below any sane persistence threshold; offloading it is pure latency with negligible memory benefit, so it stays resident (mirrors ZeRO-3's `stage3_param_persistence_threshold`).

## DeepSpeed ZeRO-3 offload: which lessons apply here (and which do not)

The single-GPU LoRA case is strictly simpler than ZeRO-3. Replicate only the mechanics that matter; deliberately drop the distributed/disk machinery so the design stays minimal and correct. All citations are under `third_party/deepspeed/deepspeed/runtime/zero/`.

Applicable (replicate these):

- **Release-after-use frees the param by swapping `.data` for an empty tensor** (`partition_parameters.py:300-316`, `free_param`). This is the only mechanism that actually reclaims HBM — the heart of Stage 2 and the reason this can hit `< 30 GiB` where grad offload could not.
- **Re-gather before backward; never hold the gathered weight across the fwd→bwd gap** (`parameter_offload.py:530-536`). Drives the Function modification in Stage 2 (read weights from the re-gathered param, not `ctx.saved_tensors`).
- **One frozen forward access trace serves both forward and reverse-order backward** (`partitioned_param_coordinator.py:236-272`; docstring `:45`). Enables prefetch (Stage 3).
- **Prefetch the next bucket on a dedicated side stream; compute waits via a recorded event, not a global `cuda.synchronize()`** (`partitioned_param_coordinator.py:365-391, 396-461`). The overlap mechanism in Stage 3.
- **Depth-2 in-flight event throttle as a pragmatic double buffer** (`:122-133, 376-387`). Simpler and sufficient versus a fixed ping-pong pool for CPU↔GPU.
- **Persistence threshold keeps tiny params resident** (`partition_parameters.py:1196-1203`). Exactly why attention LoRA (`~102 MiB`) and any sub-threshold bank stays on GPU; offloading them is all latency, no benefit.
- **Pin the CPU home once at setup (pinning blocks the host); `record_stream` the staged buffer** (`:1721-1722`, `:769-771`). Mandatory for correct async H2D and to stop the caching allocator recycling a cross-stream buffer.
- **Validate the trace each step with a release-everything fallback** (`partitioned_param_coordinator.py:187-204`). Required because expert recompute / dynamic control flow can change the access order.

Not applicable (deliberately omitted, with why — do not copy these):

- **`all_gather`/collectives and cross-rank parameter partitioning** — single GPU. ZeRO-3's "gather" collapses to a plain `.to(cuda, non_blocking=True)`; there is no sharding and no `dist.*` here.
- **NVMe swapper + slotted pinned arena** (`swap_tensor/partitioned_param_swapper.py`) — CPU DRAM is ample on this host; disk tiering adds complexity with zero benefit.
- **Contiguous flat all-gather buffers for sharded params, and grad reduce-scatter** — no sharding; grad movement is already handled by the implemented grad offload.
- **`offload_optimizer` / `DeepSpeedCPUAdam` as the offload engine** — already provided by `AsymCPUAdamW` (CPU fp32 masters + CPU AdamW). We only reuse its masters as the home source; we do not import ZeRO's optimizer offload.
- **`max_reuse_distance` weight-tying keep-resident logic** — expert-LoRA banks are used once per layer per step; reuse-distance optimization is irrelevant (leave it a no-op knob).

Net: copy the coordinator pipeline (trace → side-stream prefetch → event-gated gather → refcounted release), the persistence threshold, and the pinning / `record_stream` correctness rules. Skip everything distributed and everything on disk.

## Peak HBM target for the real acceptance workload

- Baseline = the existing `weight_offload=false, grad_offload=true` arm: `profiling/lora_grad_offload_accept_20260615T073937Z/.../gradofftrue`, peak allocated HBM `34.593 GiB`, of which expert-LoRA weights are `6.187 GiB`. Config: Qwen3-30B-A3B, single GPU, `b4_s4096`, `warmup=5`, `measure=10`, rank 64, `asym_cpuadamwds|norecomp`, activation offload `none|true|true|true`.
- Cross-entropy floor: the loss region holds ~`27.82 GiB` of fp32 vocab buffers regardless of weights. Weight offload cannot go below this floor; addressing it (chunked CE, etc.) is explicitly out of scope here.
- Expected post-weight-offload peak allocated HBM ≈ `34.593 − 6.187 + staging_high_water_at_peak`. With prefetch depth 1 (≤2 layers in flight, `≤ ~0.26 GiB`) and ~0 expert layers staged at the loss instant, expected ≈ `28.5 GiB`.
- **Hard acceptance gate (the primary goal): with weight offload enabled, peak allocated HBM `< 30 GiB`** (target ≈ `28.5`, leaving margin), as reported by the `scripts/lf/profile_lora_lf.sh` A/B — never by a toy/unit measurement. Plus a reduction of `>= 3.5 GiB` versus the same-run `weight_offload=false` baseline (reject "trivial" reductions). Latency gate: average measured `step_milliseconds` `<= 1.5×` the same-run baseline. Peak reserved HBM is tracked but is not the hard target (allocator caching keeps reserved high).
- Trivial-reject rule (explicit): if `weight_offload=true` peak `> 32.5 GiB` (i.e. < ~2 GiB saved) the stage is rejected even if it "works."

---

## Stage 1: Add the Toggle and A/B Routing With No Behavior Change

Scope:

- `third_party/LlamaFactory/src/llamafactory/hparams/finetuning_args.py`
  - `FinetuningArguments` (new `asym_cpu_adamw_weight_offload: bool = False`)
- `third_party/LlamaFactory/src/llamafactory/hparams/parser.py`
  - dependency check + `_verify_asym_cpu_adamw_args`
- `third_party/LlamaFactory/src/llamafactory/train/trainer_utils.py`
  - `_create_asym_cpu_adamw_optimizer` (pass flag through; no behavior yet)
- `scripts/lf/run_lf_lora_sft.sh` (env + CLI forwarding)
- `scripts/lf/profile_lora_lf.sh` and `scripts/lf/test_profiling.sh`
  - env defaults, usage, CLI parse, bool validation, `job_root_path`, `existing_profile_complete`, `run_job`, `ensure_jobs_tsv`/`append_job_record`
- `scripts/lf/run_lf_profiled_train.py`
  - `_config_from_args`
- Tests: `tests/lf/test_asym_cpu_adamw_args.py`, `tests/lf/test_asym_cpu_adamw_lf_integration.py`, `tests/lf/test_lf_profile_postprocess.py`

Implementation:

1. New default-off LF argument (mirror `asym_cpu_adamw_grad_offload`):

```python
# finetuning_args.py
asym_cpu_adamw_weight_offload: bool = field(
    default=False,
    metadata={"help": "JIT-stage expert-LoRA weights from pinned CPU to CUDA per layer; release CUDA copy after use."},
)
```

2. Validate dependencies (weight offload requires CPUAdamW, single process, and grad offload):

```python
# parser.py
if finetuning_args.asym_cpu_adamw_weight_offload and not finetuning_args.use_asym_cpu_adamw:
    raise ValueError("`asym_cpu_adamw_weight_offload=true` requires `use_asym_cpu_adamw=true`.")
if finetuning_args.asym_cpu_adamw_weight_offload and not finetuning_args.asym_cpu_adamw_grad_offload:
    raise ValueError("`asym_cpu_adamw_weight_offload=true` requires `asym_cpu_adamw_grad_offload=true` "
                     "(the post-accumulate hook is the weight release point).")
# _verify_asym_cpu_adamw_args
if finetuning_args.asym_cpu_adamw_weight_offload and training_args.parallel_mode != ParallelMode.NOT_PARALLEL:
    raise ValueError("AsymGEMM LoRA weight offload is single-process single-device only.")
```

3. Thread the flag to the optimizer constructor (no behavior yet): add `weight_offload=finetuning_args.asym_cpu_adamw_weight_offload` to the `AsymCPUAdamW(...)` call in `_create_asym_cpu_adamw_optimizer`.

4. Shell env + CLI forwarding (mirror the grad-offload axis exactly):

```bash
# run_lf_lora_sft.sh
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD=${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD:-false}
ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="$(bool_string ASYM_CPU_ADAMW_WEIGHT_OFFLOAD "${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}")"
CMD_ARGS+=(--asym_cpu_adamw_weight_offload "${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}")
ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_WEIGHT_OFFLOAD="${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD}"
```

5. Profile sweep axis + routing (mirror `ASYM_CPU_ADAMW_GRAD_OFFLOADS`):

```bash
# profile_lora_lf.sh / test_profiling.sh
ASYM_CPU_ADAMW_WEIGHT_OFFLOADS=${ASYM_CPU_ADAMW_WEIGHT_OFFLOADS:-${ASYM_CPU_ADAMW_WEIGHT_OFFLOAD:-false}}
# CLI: --asym-cpu-adamw-weight-offloads LIST
mapfile -t asym_cpu_adamw_weight_offload_modes < <(tokens "${ASYM_CPU_ADAMW_WEIGHT_OFFLOADS}" | while read -r v; do bool_value "$v"; done | dedupe)
```

   The axis is a comma-separated **list of bools**; `ASYM_CPU_ADAMW_WEIGHT_OFFLOADS=false,true` expands to two jobs run **sequentially** within a single `profile_lora_lf.sh` invocation, producing the full A/B under one output root (a single value like `true` runs just that mode). In the backend loop, non-CPUAdamW backends force `false`; CPUAdamW backends iterate the axis. Nest the weight-offload axis *inside* the existing grad-offload loop and thread `weight_offload` into `run_job`, `job_root_path` (suffix `__weightoff${weight_offload}`, appended after `__gradoff${grad_offload}`), `ensure_jobs_tsv`/`append_job_record` (new `weight_offload` column), and `existing_profile_complete` (compare against `source_profile.config.asym_cpu_adamw_weight_offload`, so a stale `weightofffalse` profile is never accepted for a `weightofftrue` run). Keep a `legacy_job_root_path` fallback that omits the suffix for `weight_offload=false` so older profiles still resolve.

6. Persist in source profiles:

```python
# run_lf_profiled_train.py::_config_from_args
"asym_cpu_adamw_weight_offload": (
    os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_CPU_ADAMW_WEIGHT_OFFLOAD")
    or _option_value(args, "--asym_cpu_adamw_weight_offload") or "false"
).lower() in {"1", "true", "yes", "on"},
```

Ambiguity/uncertainty: none material — this is a mechanical clone of the grad-offload plumbing (`lora_grad_offload.md` Stage 1), already proven in `profiling/lora_grad_offload_accept_*`.

Risks to watch:

- `profile_lora_lf.sh` and `test_profiling.sh` drift; patch both. Two stacked suffixes (`__gradoff*__weightoff*`) lengthen job dirs; keep `safe_label` within filesystem limits.
- Do not let the new dependency check break existing `grad_offload`-only runs (it must only fire when `weight_offload=true`).

Validation before Stage 2:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

"${ENV_PYTHON}" -m pytest -q \
  tests/lf/test_asym_cpu_adamw_args.py \
  tests/lf/test_asym_cpu_adamw_lf_integration.py

OUT=/tmp/asym_lora_weight_offload_stage1_dryrun
rm -rf "${OUT}"
BACKEND_SPECS="asym_cpuadamwds|recomp" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="true" \
ASYM_CPU_ADAMW_WEIGHT_OFFLOADS="false,true" \
PROFILERS=source SEQ_LENS=128 MAX_STEPS=1 WARMUP_STEPS=0 \
PREPARE_DATASETS=false DRY_RUN=true PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

rg -n "weightofffalse|--asym_cpu_adamw_weight_offload false" "${OUT}"
rg -n "weightofftrue|--asym_cpu_adamw_weight_offload true"  "${OUT}"
```

Gate: both modes route to distinct job dirs; the dependency error fires if `weight_offload=true` is set with `grad_offload=false`. No runtime/memory change expected (pure plumbing).

---

## Stage 2: CPU Weight Home + Synchronous JIT Gather/Release (the core memory win)

Scope:

- New file `asym_gemm/training/weight_offload.py`
  - `LoRAWeightOffloadCoordinator` (home registry, GPU staging-buffer pool, gather/release, hook install, trace placeholder)
  - module-level `install_lora_weight_offload(model, coordinator)` helper
- `asym_gemm/training/qwen3_moe.py`
  - `_ActivationOffloadQwen3ExpertFunction.forward` — stop `save_for_backward` of the six LoRA weights; record only what backward needs to re-gather (the layer handle is already on `ctx`)
  - `_ActivationOffloadQwen3ExpertFunction.backward` — read the six weights from `ctx.layer.<bank>` (re-gathered by the coordinator) instead of `ctx.saved_tensors`
  - `AsymQwen3Experts` — add `gather_lora_weights()` / `release_lora_weights()` that delegate to the coordinator; a `_weight_offload` handle
  - `wrap_qwen3_experts` — accept/propagate the coordinator
- `asym_gemm/training/cpu_adam.py` — `AsymCPUAdamW.__init__` (new `weight_offload`/`coordinator` args; relax CUDA-residency guard `:133-141`), `_check_post_prepare_devices` (relax `:277-285`), `_offload_grad_from_hook` (`:347-376`, add weight release), `_copy_master_to_compute_param` (`:395-398`, redirect copy-back to the home), new `attach_weight_offload_coordinator`
- `third_party/LlamaFactory/src/llamafactory/train/trainer_utils.py` — `_create_asym_cpu_adamw_optimizer` (`:530-582`; build coordinator, install, attach — exact order in step 6)
- Tests: `tests/training/test_lora_weight_offload.py` (new), `tests/training/test_qwen3_asym_backend.py` (extend), `tests/training/test_asym_cpu_adamw.py` (extend)

This stage is synchronous (no side stream / prefetch). It proves correctness and the memory reduction. Latency may regress here; Stage 3 restores it. Do not accept the *feature* on Stage 2 alone — Stage 2's gate is correctness + memory drop in an e2e A/B; the latency gate is Stage 5.

Memory and efficiency design:

- The six expert-LoRA banks per layer (`~132 MiB`) are gathered together right before that layer's experts run and released right after. With synchronous gather, at most one layer (forward) or one layer (backward) is staged at a time, so expert-LoRA HBM high-water is `~132 MiB` instead of `6.187 GiB`. At the loss peak, zero expert layers are staged.
- Home dtype is bf16 (the compute dtype), so gather is a pure same-dtype H2D `copy_` (no transient fp32 GPU buffer). The fp32 master remains in the optimizer; the home is refreshed from it after `step()` (step 5 below).
- Gather buffers come from a small reuse pool keyed by `(numel, dtype)` to avoid per-step allocation churn and allocator fragmentation.

1. Coordinator (synchronous core; the side-stream/prefetch fields are added but unused until Stage 3):

```python
# asym_gemm/training/weight_offload.py
class LoRAWeightOffloadCoordinator:
    def __init__(self, *, pin_memory: bool = True, persistence_threshold_numel: int = 1_048_576):
        self.pin_memory = pin_memory
        self.persistence_threshold_numel = persistence_threshold_numel
        self._home: dict[int, torch.Tensor] = {}          # id(param) -> pinned bf16 CPU home
        self._placeholder: dict[int, torch.Tensor] = {}   # id(param) -> 0-size CUDA tensor
        self._meta: dict[int, tuple[torch.Size, torch.dtype, torch.device]] = {}
        self._pool: dict[tuple[int, str], list[torch.Tensor]] = {}  # (numel, dtype) -> free GPU buffers
        self._staged: dict[int, torch.Tensor] = {}        # id(param) -> live GPU buffer (to return to pool)
        self.registered_params: list[torch.nn.Parameter] = []
        self._stream = None                               # Stage 3
        self._inflight: dict[int, torch.cuda.Event] = {}  # Stage 3
        self.staged_high_water_bytes = 0

    def register(self, param: torch.nn.Parameter, name: str) -> bool:
        if param.numel() < self.persistence_threshold_numel:
            return False  # keep tiny params resident (attention LoRA, biases)
        key = id(param)
        if key in self._home:
            return True   # shared-storage param already registered
        self._meta[key] = (param.shape, param.dtype, param.device)
        home = param.detach().to(device="cpu", dtype=param.dtype).contiguous()
        if self.pin_memory and torch.cuda.is_available() and not home.is_pinned():
            try: home = home.pin_memory()
            except RuntimeError: pass
        self._home[key] = home
        self._placeholder[key] = torch.empty(0, dtype=param.dtype, device=param.device)
        self.registered_params.append(param)
        self.release(param)   # start released; nothing resident until first gather
        return True

    @torch.no_grad()
    def gather(self, param: torch.nn.Parameter) -> None:        # synchronous (Stage 2)
        key = id(param)
        if key not in self._home or key in self._staged:
            return
        shape, dtype, device = self._meta[key]
        buf = self._take_buffer(self._home[key].numel(), dtype, device).view(shape)
        buf.copy_(self._home[key], non_blocking=self._home[key].is_pinned())
        self._staged[key] = buf
        self.staged_high_water_bytes = max(self.staged_high_water_bytes,
                                           sum(b.numel()*b.element_size() for b in self._staged.values()))
        param.data = buf  # full-shaped GPU weight for compute AND grad accumulation

    @torch.no_grad()
    def release(self, param: torch.nn.Parameter) -> None:
        key = id(param)
        buf = self._staged.pop(key, None)
        if buf is not None:
            self._return_buffer(buf)
        param.data = self._placeholder[key]  # reclaim HBM

    @torch.no_grad()
    def refresh_home_from_master(self, param, master_fp32) -> None:
        self._home[id(param)].copy_(master_fp32)  # CPU cast, no transfer

    def _take_buffer(self, numel, dtype, device):
        pool = self._pool.get((numel, str(dtype)))
        if pool: return pool.pop()
        return torch.empty(numel, dtype=dtype, device=device)
    def _return_buffer(self, buf):
        self._pool.setdefault((buf.numel(), str(buf.dtype)), []).append(buf.view(-1))
```

2. Per-layer gather/release entry points on the experts module:

```python
# qwen3_moe.py :: AsymQwen3Experts
def _lora_banks(self):
    return (self.gate_lora_A, self.gate_lora_B, self.up_lora_A,
            self.up_lora_B, self.down_lora_A, self.down_lora_B)

def gather_lora_weights(self):
    if getattr(self, "_weight_offload", None) is not None:
        for p in self._lora_banks(): self._weight_offload.gather(p)

def release_lora_weights(self):
    if getattr(self, "_weight_offload", None) is not None:
        for p in self._lora_banks(): self._weight_offload.release(p)
```

3. Modify the custom Function so backward does **not** depend on the saved GPU weights. The weights are still positional differentiable inputs (so grads flow to `param.grad`), but they are read fresh in backward from the re-gathered params on `ctx.layer`:

```python
# qwen3_moe.py :: _ActivationOffloadQwen3ExpertFunction.forward  (tail)
ctx.layer = layer
ctx.weight_offload = getattr(layer, "_weight_offload", None) is not None
if ctx.weight_offload:
    ctx.save_for_backward(offsets, experts)            # weights NOT saved
else:
    ctx.save_for_backward(offsets, experts,
                          gate_lora_A, gate_lora_B, up_lora_A, up_lora_B,
                          down_lora_A, down_lora_B)     # legacy path unchanged
return output

# backward (head)
if ctx.weight_offload:
    offsets, experts = ctx.saved_tensors
    layer = ctx.layer
    layer.gather_lora_weights()                        # re-gather (Stage 3: already prefetched)
    gate_lora_A, gate_lora_B = layer.gate_lora_A, layer.gate_lora_B
    up_lora_A,   up_lora_B   = layer.up_lora_A,   layer.up_lora_B
    down_lora_A, down_lora_B = layer.down_lora_A, layer.down_lora_B
else:
    (offsets, experts, gate_lora_A, gate_lora_B, up_lora_A, up_lora_B,
     down_lora_A, down_lora_B) = ctx.saved_tensors
# ... unchanged grad math; returns (grad_packed, None, None, grad_gate_lora_A, ...) ...
```

Critical ordering note (exact autograd sequence per layer, weight-offload mode):

1. experts `forward_pre_hook` → `gather` (`param.data` ← full GPU weight).
2. `_ActivationOffloadQwen3ExpertFunction.forward` consumes the params; saves only `offsets, experts` (not the weights).
3. experts `forward_hook` → `release` (`param.data` ← 0-size placeholder). HBM reclaimed; the loss/lm-head peak happens with these released.
4. backward reaches the experts module → the Function's `backward` calls `gather_lora_weights()` (`param.data` ← full again) and reads the banks from `ctx.layer`.
5. the Function returns the six weight grads.
6. autograd `AccumulateGrad` writes each `param.grad` — this needs `param.data` full-shaped, which step 4 guaranteed.
7. the per-param grad-offload post-accumulate hook fires and does grad copy-out **and** weight `release`.

So in backward the weight is released by exactly one owner — the post-accumulate hook (step 5 of Implementation). There is **no** `full_backward_hook` stand-in (it would double-release and race the grad hook).

4. Forward gather/release via module hooks (legacy compute untouched). Discover the experts modules **by type** so the install does not depend on the `mlp` attribute name (the confirmed path is `model.model.layers[i].mlp.experts`, an `AsymQwen3Experts`, set at `qwen3_moe.py:2616`; type-discovery is robust to any future rename):

```python
# weight_offload.py :: install_lora_weight_offload(model, coordinator) -> int
from .qwen3_moe import AsymQwen3Experts
BANKS = ("gate_lora_A", "gate_lora_B", "up_lora_A", "up_lora_B", "down_lora_A", "down_lora_B")

def _gather_hook(module, _inputs):              # named fns, NOT lambdas (avoid late-binding capture bugs)
    module.gather_lora_weights()
def _release_hook(module, _inputs, _output):
    module.release_lora_weights()

installed = 0
for experts in model.modules():
    if not isinstance(experts, AsymQwen3Experts):
        continue
    experts._weight_offload = coordinator
    n = sum(coordinator.register(getattr(experts, b), f"{id(experts)}.{b}") for b in BANKS)
    if n:
        experts.register_forward_pre_hook(_gather_hook)
        experts.register_forward_hook(_release_hook)
        installed += n
return installed     # 0 ⇒ weight offload matched nothing; the caller asserts installed > 0
```

5. Optimizer minimal integration (required for the e2e to run — `cpu_adam.py`). `_ParamMapping.cuda_param` is the **same `nn.Parameter` object** as `experts.<bank>` (gathered by `named_lora_parameters(model)` at `lora.py:395`, stored without clone at `cpu_adam.py:200`), so the coordinator (keyed on `id(param)`) and the optimizer act on identical params.

```python
# AsymCPUAdamW.__init__: accept coordinator + flag; relax the CUDA-residency guard (:133-141)
self.weight_offload = bool(weight_offload)
self._coordinator = coordinator                       # may be attached later via attach_weight_offload_coordinator
if torch.cuda.is_available() and not self.weight_offload:
    ... existing not_cuda guard ...                   # when weight_offload, params may be released placeholders
# _check_post_prepare_devices (:277-285): skip the CUDA-residency assert for registered banks when weight_offload

# _offload_grad_from_hook (:347-376) tail — the single backward release point, strictly after AccumulateGrad:
    mapping.cpu_param.grad = mapping.grad_buffer
    mapping.cuda_param.grad = None
    if self.weight_offload and self._coordinator is not None:
        self._coordinator.release(mapping.cuda_param)        # free the backward-staged weight

# _copy_master_to_compute_param (:395-398) — redirect copy-back to the CPU home (no H2D in step()):
def _copy_master_to_compute_param(self, mapping):
    if self.weight_offload and self._coordinator is not None:
        self._coordinator.refresh_home_from_master(mapping.cuda_param, mapping.cpu_param.data)   # cpu cast only
        return
    mapping.cuda_param.data.copy_(mapping.cpu_param.data, non_blocking=mapping.cpu_param.data.is_pinned())

def attach_weight_offload_coordinator(self, coordinator):
    self._coordinator = coordinator
    self.weight_offload = True
```

   `step()` (`:410-489`) and `zero_grad()` (`:491+`) need no further change: they key off `cuda_param.grad`/`grad_buffer`, never the weight storage. The copy-back loop at `:478-484` now refreshes homes instead of doing H2D.

6. Exact install order — load-bearing, because the optimizer builds its fp32 masters from the **full** GPU weight at construction (`cpu_adam.py:190`). Do this in `trainer_utils._create_asym_cpu_adamw_optimizer` (`:530-582`), where both `model` and `optimizer` are in scope:

```python
named_lora = named_lora_parameters(model)                         # lora.py:395
trainable_lora = [(n, p) for n, p in named_lora if p.requires_grad]
optimizer = AsymCPUAdamW(trainable_lora, ...,
                         grad_offload=finetuning_args.asym_cpu_adamw_grad_offload,
                         weight_offload=finetuning_args.asym_cpu_adamw_weight_offload)
# (1) masters are now correct: built while weights are still FULL on CUDA.
if finetuning_args.asym_cpu_adamw_weight_offload:
    from asym_gemm.training.weight_offload import LoRAWeightOffloadCoordinator, install_lora_weight_offload
    coordinator = LoRAWeightOffloadCoordinator(pin_memory=finetuning_args.asym_cpu_adamw_pin_memory)
    installed = install_lora_weight_offload(model, coordinator)    # (2) capture bf16 homes from full weights, THEN release
    assert installed > 0, "asym_cpu_adamw_weight_offload=true but matched 0 expert-LoRA params"
    optimizer.attach_weight_offload_coordinator(coordinator)       # (3) optimizer.release/refresh now target the homes
return optimizer
```

   Inside `install_lora_weight_offload`, `coordinator.register(param, ...)` must capture the bf16 home from the still-full `param.detach()` **before** it calls `release(param)`. That home value equals the optimizer's fresh fp32 master (both derived from the same full weight), so step 0 is consistent without any extra copy-back.

Ambiguity/uncertainty resolved:

- "Does `save_for_backward` keep the GPU storage alive even if I reassign `param.data`?" — Yes; that is why this stage stops saving the weights and reads them from the re-gathered param (Function source `qwen3_moe.py:1082-1108`).
- "Will grad accumulation fail if `param.data` is a 0-size placeholder?" — It would; resolved by re-gathering at the *start* of the Function's backward and releasing only in the post-accumulate hook (step 5), strictly after `AccumulateGrad`.
- "Are the optimizer param and the model param the same object?" — Yes (`lora.py:395` → `cpu_adam.py:200`, no clone), so `coordinator.release(mapping.cuda_param)` frees exactly the param the experts module uses.
- "Is the install attribute path (`mlp`) guaranteed?" — Sidestepped: discover `AsymQwen3Experts` via `model.modules()` instead of hard-coding `.mlp.experts`.
- "What if masters get built from already-released weights?" — Prevented by the explicit order in step 6 (construct optimizer first → install/release second → attach third).

Risks to watch:

- The legacy (non-offload) and recompute (`ASYM_EXPERT_GC_*`, `expert_recompute_config`) expert paths also read `self.<bank>` and rely on standard autograd save. Weight offload is only wired for the activation-offload Function path (the acceptance config). Assert weight offload is rejected when expert recompute is enabled, and leave the plain/recompute paths resident (documented limitation).
- Shared-storage params: the optimizer de-dupes by storage view (`cpu_adam.py:143-160`). The coordinator must register by the same identity to avoid double homes; key on `id(param)` and skip already-registered (done above).
- `param.data` reassignment must preserve `requires_grad` (it does) and contiguity (homes are `.contiguous()`).

Validation before Stage 3:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}

# (a) Numerical correctness + residency unit test (new file). Required assertions:
#  - forward output equal (allclose) with weight_offload on vs off on a tiny AsymQwen3Experts.
#  - every expert-LoRA bank .grad equals the non-offload reference (allclose) after one backward.
#  - after forward (pre-backward), each registered bank's param.data.numel() == 0 (released).
#  - gathering reproduces home values exactly (copy_ round-trip).
"${ENV_PYTHON}" -m pytest -q tests/training/test_lora_weight_offload.py

# (b) e2e MEMORY proof on the real model (real shapes; this is the Stage 2 gate).
OUT=/tmp/asym_lora_weight_offload_stage2_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="true" \
ASYM_CPU_ADAMW_WEIGHT_OFFLOADS="false,true" \
PROFILERS=source SEQ_LENS=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=1 MAX_STEPS=3 WARMUP_STEPS=2 MAX_SAMPLES=64 \
DATASET=asym_long_sft_smoke PREPARE_DATASETS=true MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" \
PROFILE_MEMORY_BREAKDOWN=true PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

"${ENV_PYTHON}" - "${OUT}" <<'PY'
import json, pathlib, sys
rows={}
for p in sorted(pathlib.Path(sys.argv[1]).rglob("profile.json")):
    s=json.loads(p.read_text()); s=s.get("source_profile",s); c=s.get("config",{})
    mem=s.get("memory",{}); peak=(mem.get("gpu",{}).get("peak_allocated_hbm_bytes")
        or mem.get("peak_allocated_hbm_bytes") or 0)/2**30
    rows[bool(c.get("asym_cpu_adamw_weight_offload"))]={"peak":peak,"path":str(p)}
    print(json.dumps({"weight_offload":bool(c.get("asym_cpu_adamw_weight_offload")),"peak_gib":round(peak,3)}))
assert set(rows)=={False,True}, rows
off,on=rows[False]["peak"],rows[True]["peak"]
assert on < off - 3.5, f"weight offload did not meaningfully reduce peak: off={off:.3f} on={on:.3f} GiB"
print(f"OK Stage2 memory: off={off:.3f} on={on:.3f} GiB (saved {off-on:.3f})")
PY
```

Gate: unit correctness passes; e2e shows `weight_offload=true` peak at least `3.5 GiB` below the `false` arm (meaningful, not trivial). Latency recorded but not gated here.

---

## Stage 3: Prefetch + Side-Stream Overlap (latency hardening)

Scope:

- `asym_gemm/training/weight_offload.py`
  - side stream, in-flight event registry, frozen access trace, `prefetch_next(...)`, depth-2 throttle, `ASYM_LORA_WEIGHT_OFFLOAD_SYNC` fallback
- `asym_gemm/training/qwen3_moe.py`
  - forward/backward hooks call `coordinator.note_access(experts)` so the trace can be recorded and replayed
- Tests: `tests/training/test_lora_weight_offload.py` (trace + overlap correctness)

Implementation:

- Side stream + correct cross-stream handoff (replace the synchronous gather body):

```python
@torch.no_grad()
def gather(self, param):
    key = id(param)
    if key in self._staged: return
    ev = self._inflight.pop(key, None)
    if ev is None:                                   # not prefetched: issue now on side stream
        self._issue_copy(param); ev = self._inflight.pop(key)
    torch.cuda.current_stream().wait_event(ev)       # compute waits for the copy, no global sync
    self._staged[key].record_stream(torch.cuda.current_stream())
    param.data = self._staged[key]

def _issue_copy(self, param):
    key=id(param); shape,dtype,device=self._meta[key]
    buf=self._take_buffer(self._home[key].numel(),dtype,device).view(shape)
    with torch.cuda.stream(self._stream):
        buf.copy_(self._home[key], non_blocking=True) # pinned home => true async H2D
    ev=torch.cuda.Event(); ev.record(self._stream)
    self._staged[key]=buf; self._inflight[key]=ev
```

- Trace: on step 0 record the order in which `AsymQwen3Experts` modules are accessed (forward); freeze it as a tuple. Each forward-pre hook, after issuing its own gather, prefetches the next K layers' banks on the side stream, bounded by a byte budget (depth 1–2). Backward replays the same frozen trace in reverse (the Function's `gather_lora_weights()` already issues the re-gather; the coordinator prefetches the *previous* layer). Validate the trace each step; on mismatch (dynamic control flow / recompute), fall back to synchronous gather-on-demand and log it.
- Depth-2 throttle: keep at most two layers' copies in flight (event-bounded), mirroring DeepSpeed `__max_ongoing_fetch_events = 2` (`partitioned_param_coordinator.py:122-133`). This is the pragmatic double buffer; a fixed ping-pong pool is unnecessary for CPU↔GPU.

Hardware note: this platform has C2C (the run emits `c2c_combined`). At C2C bandwidth, staging `6.187 GiB` twice (fwd+bwd) is ≈ tens of ms vs a ~4.7 s step (<1%), so even Stage 2's synchronous gather may already pass the 50% latency gate. Stage 3 is still implemented for portability and margin; if the Stage 2 e2e already meets the latency gate, Stage 3 may be reduced to trace + prefetch without aggressive tuning.

Ambiguity/uncertainty resolved: pinned home is mandatory for real async (`non_blocking=True` is ignored for pageable memory); `record_stream` is mandatory so the caching allocator does not recycle a buffer still read by compute. Both confirmed by the PyTorch pinned-memory doc and DeepSpeed `partition_parameters.py:769-771`.

Risks to watch:

- Stream/event correctness bugs cause intermittent NaNs, not crashes. Keep `ASYM_LORA_WEIGHT_OFFLOAD_SYNC=1` forcing the Stage 2 synchronous path as a fallback and as the unit-test oracle.
- Trace invalidation under expert recompute or variable layer execution → prefetch wrong banks. Validate-and-fallback is required, not optional.
- Prefetching too deep re-inflates the high-water (defeats the memory win). Keep depth ≤ 2 layers and assert the staged-bytes high-water stays `< ~0.5 GiB`.

Validation before Stage 4:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}
"${ENV_PYTHON}" -m pytest -q tests/training/test_lora_weight_offload.py
# Equivalence under overlap: ASYM_LORA_WEIGHT_OFFLOAD_SYNC=1 vs unset must give identical grads (allclose) on the tiny model.
```

Gate: overlap path produces identical numerics to the synchronous oracle; staged-bytes high-water bounded.

---

## Stage 4: Memory/Timing Counters and Postprocess Columns (reporting only, no behavior change)

All offload behavior lands in Stage 2 (so the e2e can run) and Stage 3 (overlap). Stage 4 only makes the offload *measurable* in the `profile_lora_lf.sh` artifacts — it changes no runtime behavior.

Scope:

- `asym_gemm/training/cpu_adam.py` — `asym_cpu_adamw_summary` (add read-only weight-offload counters sourced from the coordinator)
- `asym_gemm/training/weight_offload.py` — expose `summary()` (param_count, home bytes, pinned home bytes, staged high-water, gather ms, persistent-resident bytes)
- `scripts/lf/postprocess_lf_profile_artifacts.py` — `_asym_cpu_adamw_rows` (surface new scalar columns in `asym_cpu_adamw.csv`)
- `asym_gemm/profiling/lf_trace.py` — label the CPU home and live GPU staged buffers in the memory breakdown
- Tests: `tests/test_lf_memory_breakdown.py`, `tests/lf/test_lf_profile_postprocess.py`

Implementation:

1. Counters in `asym_cpu_adamw_summary` (read-only): `weight_offload_enabled`, `weight_offload_param_count`, `weight_offload_home_bytes`, `weight_offload_pinned_home_bytes`, `weight_offload_staged_high_water_bytes`, `weight_gather_ms` (forward+backward H2D time), `weight_offload_persistent_resident_bytes` (banks kept resident under the persistence threshold).
2. Memory-breakdown labels (`lf_trace.py`): count the pinned bf16 homes as `lora_weight_home_cpu` / `lora_weight_home_cpu_pinned`, distinct from `optimizer_state_cpu`, `cpu_master_weight_cpu`, and `offloaded_grad_cpu` (no double counting). Count live GPU staged buffers once by storage pointer as `lora_weight_staged_gpu`.
3. Postprocess: extend `_asym_cpu_adamw_rows` so the new scalars become columns in `asym_cpu_adamw.csv`; add a test asserting their presence.

Risks to watch:

- Do not double-count the bf16 home against the fp32 master or the offloaded grad buffer (separate storages, separate labels).
- `weight_offload_staged_high_water_bytes` must be the peak of live staged buffers sampled across the step, not the value at report time (which is ~0 after release).

Validation before Stage 5:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}
"${ENV_PYTHON}" -m pytest -q tests/test_lf_memory_breakdown.py tests/lf/test_lf_profile_postprocess.py
# After a weight_offload=true smoke run: asym_cpu_adamw.csv must contain weight_offload_enabled, weight_offload_param_count,
# weight_offload_home_bytes, weight_offload_staged_high_water_bytes; and the memory breakdown must show lora_weight_home_cpu.
```

Gate: counters/labels present and correctly de-duplicated; the numbers match the Stage 2 behavior already validated (no behavior change introduced here).

---

## Stage 5: E2E A/B Acceptance (memory + latency)

Scope: no new source changes unless a regression is found. Accept/reject with real LF LoRA profiling.

1. Smoke A/B (routing, hooks, clipping, postprocess), short:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}
OUT=/tmp/asym_lora_weight_offload_smoke_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|norecomp" MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="true" ASYM_CPU_ADAMW_WEIGHT_OFFLOADS="false,true" \
PROFILERS=source SEQ_LENS=512 PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=2 WARMUP_STEPS=0 MAX_SAMPLES=8 PREPARE_DATASETS=true MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"
rg -n '"asym_cpu_adamw_weight_offload": (false|true)|"weight_offload_enabled": (false|true)' "${OUT}"
```

2. Acceptance A/B (the binding run; identical workload shape both modes, `grad_offload=true` fixed, only weight offload varies):

```bash
OUT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling/lora_weight_offload_ab_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|norecomp" MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="true" ASYM_CPU_ADAMW_WEIGHT_OFFLOADS="false,true" \
PROFILERS=source SEQ_LENS=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=10 WARMUP_STEPS=5 MAX_SAMPLES=128 DATASET=asym_long_sft_smoke PREPARE_DATASETS=true \
LORA_RANK=64 LORA_ALPHA=16 LORA_DROPOUT=0.00 MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" PROFILE_MEMORY_BREAKDOWN=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"
```

3. Compare and gate:

```bash
"${ENV_PYTHON}" - "${OUT}" <<'PY'
import json, pathlib, sys
rows=[]
for p in sorted(pathlib.Path(sys.argv[1]).rglob("profile.json")):
    s=json.loads(p.read_text()); s=s.get("source_profile",s); c=s.get("config",{})
    cpu=s.get("asym_cpu_adamw",{}); mem=s.get("memory",{})
    peak=(mem.get("gpu",{}).get("peak_allocated_hbm_bytes") or mem.get("peak_allocated_hbm_bytes") or 0)/2**30
    st=[r for r in s.get("step_samples",{}).get("rows",[]) if isinstance(r.get("step_milliseconds"),(int,float))]
    meas=[r for r in st if not r.get("is_warmup")] or st
    avg=sum(float(r["step_milliseconds"]) for r in meas)/len(meas) if meas else None
    rows.append({"weight_offload":bool(c.get("asym_cpu_adamw_weight_offload")),
                 "grad_offload":bool(c.get("asym_cpu_adamw_grad_offload")),
                 "peak_gib":peak,"avg_step_ms":avg,
                 "staged_hw_gib":(cpu.get("weight_offload_staged_high_water_bytes") or 0)/2**30,
                 "weight_offload_enabled":cpu.get("weight_offload_enabled")})
    print(json.dumps(rows[-1],sort_keys=True))
assert len(rows)==2 and {r["weight_offload"] for r in rows}=={False,True}, rows
on =next(r for r in rows if r["weight_offload"])
off=next(r for r in rows if not r["weight_offload"])
assert all(r["grad_offload"] for r in rows), "grad_offload must be true in both arms"
assert on["weight_offload_enabled"] is True, "weight offload not active in on-arm"
assert 33.5 <= off["peak_gib"] <= 35.5, f"baseline off-arm not near 34.593 GiB: {off['peak_gib']}"
assert on["peak_gib"] < 30.0, f"PRIMARY GOAL MISSED: weight-offload peak must be < 30 GiB, got {on['peak_gib']}"
assert on["peak_gib"] <= off["peak_gib"] - 3.5, f"reduction not meaningful: off={off['peak_gib']} on={on['peak_gib']}"
assert on["peak_gib"] <= 32.5, "reduction is trivial (<~2 GiB) -> reject"
if on["avg_step_ms"] and off["avg_step_ms"]:
    assert on["avg_step_ms"] <= off["avg_step_ms"]*1.5, \
        f"latency regression >50%: off={off['avg_step_ms']} on={on['avg_step_ms']}"
print("ACCEPT", json.dumps({"off":off,"on":on}))
PY
```

Acceptance criteria:

- Both arms complete under one output root with distinct `weightofffalse`/`weightofftrue` paths, both with `grad_offload=true`.
- `weight_offload=false` baseline peak allocated HBM in `[33.5, 35.5] GiB` (else the run shape changed and the target is not comparable — fix that first).
- **`weight_offload=true` peak allocated HBM `< 30 GiB` (the primary goal; expected ≈ `28.5`)**, a reduction `>= 3.5 GiB` (meaningful), and `<= 32.5 GiB` (over the trivial-reject bound). This number is taken from `profile_lora_lf.sh`, not from unit/toy runs.
- Average measured `step_milliseconds` `<= 1.5×` the same-run baseline. If it regresses, inspect `weight_gather_ms`, backward time, and prefetch depth before changing the gate.
- `weight_offload_staged_high_water_bytes < ~0.5 GiB` (proves the memory came from release, not a smaller-but-still-large resident set).
- Loss curve over the 10 measured steps matches the baseline within noise (numerical safety of the re-gather path).

Risks to watch:

- If the on-arm peak stalls above `30.0 GiB` but `staged_high_water` is small, the residual is the cross-entropy floor (`~27.82 GiB`) plus other live tensors — confirm via `PROFILE_MEMORY_SNAPSHOT=true` that expert-LoRA blocks are gone from the peak live set; if so, the remaining gap is loss-side and out of scope for this plan.
- If latency regresses on non-C2C hardware, Stage 3 prefetch depth / bucket sizing is the lever; do not relax the memory gate to compensate.

---

## Stage 6: Confirm the new peak root-cause with a CUDA memory snapshot

Purpose: prove the `< 30 GiB` win is *real and understood* — the expert-LoRA weights are genuinely absent from the peak live set and the residual is the cross-entropy floor — not an allocator-caching accident. This reuses the exact tooling from the grad-offload debug (`PROFILE_MEMORY_SNAPSHOT=true` + `scripts/testing/analyze_cuda_memory_snapshot.py`). It is a required confirmation stage between acceptance (Stage 5) and any optional refinement (Stage 7).

Scope: no source changes — a focused single-step `profile_lora_lf.sh` run with the snapshot enabled on the accepted `weight_offload=true` config, plus the snapshot replay.

Validation:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
export ENV_PYTHON=${ENV_PYTHON:-/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python}
OUT=/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling/lora_weight_offload_peakdebug_$(date -u +%Y%m%dT%H%M%SZ)
BACKEND_SPECS="asym_cpuadamwds|norecomp" MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
ASYM_CPU_ADAMW_GRAD_OFFLOADS="true" ASYM_CPU_ADAMW_WEIGHT_OFFLOADS="true" \
PROFILERS=source SEQ_LENS=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=1 WARMUP_STEPS=5 MAX_SAMPLES=128 DATASET=asym_long_sft_smoke PREPARE_DATASETS=true \
LORA_RANK=64 LORA_ALPHA=16 LORA_DROPOUT=0.00 MAX_GRAD_NORM=1.0 \
ASYMM_EXP_ACT_POLICIES="none|true|true|true" PROFILE_MEMORY_SNAPSHOT=true \
PLOT=false PLOT_MEMORY_BREAKDOWN=false \
bash scripts/lf/profile_lora_lf.sh --gpus 0 --output-root "${OUT}"

SNAP="$(find "${OUT}" -name memory_snapshot.pickle | head -n1)"
"${ENV_PYTHON}" scripts/testing/analyze_cuda_memory_snapshot.py --snapshot "${SNAP}" --min-bytes 1048576 \
  --output-md "${OUT}/weight_offload_peak.md" --output-json "${OUT}/weight_offload_peak.json"
"${ENV_PYTHON}" - "${OUT}/weight_offload_peak.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); GiB=2**30
peak=d["peak_live_bytes"]/GiB
comp={r["component"]: r["bytes"]/GiB for r in d["bucket_rows"]}
print("peak_live_gib", round(peak,3), "components", {k:round(v,3) for k,v in comp.items()})
assert peak < 30.0, f"snapshot peak not < 30 GiB: {peak}"
# expert-LoRA must be essentially gone from the peak live set (was ~6.2 GiB with weight_offload=false):
assert comp.get("routed_experts", 0) < 0.5, f"expert-LoRA still resident at peak: {comp}"
print("OK: peak is cross-entropy-floor dominated; expert-LoRA weights absent from the peak live set")
PY
```

Pass criteria:

- Snapshot `peak_live_bytes < 30 GiB`, and it agrees with the Stage 5 `profile_lora_lf.sh` peak to ~0.1 GiB (cross-check).
- The `routed_experts` / long-lived-param contribution at the peak live set is `< 0.5 GiB` (the ~6.19 GiB is gone; only small staging remains), versus `~6.2 GiB` in the `weight_offload=false` snapshot.
- The peak is dominated by the cross-entropy fp32 vocab blocks (`loss` + `allocator_unframed` backward temporaries). If `routed_experts` is still large, the release path is not firing at the loss instant — debug the forward/grad hooks before claiming the goal met.

Interpretation note (carried from the grad-offload debug): `allocator_unframed` blocks are real caching-allocator blocks with no Python frame, because recording uses `stacks="python"` and cross-entropy backward runs in C++. They are CE temporaries, **not** missing/unaccounted memory — do not misread them as an attribution gap.

---

## Stage 7 (optional): refinements — only after Stages 5-6 have already met `< 30 GiB`

Stage 2's stage-to-HBM design is the shipping design and is expected to reach `< 30 GiB` on its own. Stage 7 is purely optional upside and must never be a prerequisite for the goal.

### Stage 7a — AsymGEMM CPU-weight streaming (Design B): UNCERTAIN — design and validate before building

Status: **not assumed to work; do not implement until the math is designed and validated.** It is true that AsymGEMM can stream a CPU-resident weight into a GEMM without materializing it in HBM (frozen base path, `m_grouped_bf16_asym_gemm_nt_contiguous`, `frozen_linear.py`). It is **not** established that this transfers to the *trainable* expert-LoRA GEMMs, because:

- The frozen kernel emits only grad_x. Trainable LoRA-B also needs grad_weight (`dL/dB = lowrank^T @ grad_out`, grouped per expert). There may be no CPU-streaming kernel that emits grad_weight, so backward might still require staging the weight (or the activation) to HBM — which would erode the memory win.
- The kernel operand layout (NT-contiguous, grouped offsets, transpose convention) may not match the LoRA-B `grouped_mm` math. Making it fit could require **changing the LoRA compute math**, which must first be proven numerically equivalent (and loss-equivalent) to the current path.
- The acceptance config already runs LoRA-A forward in a CPU-left mode (`loraafwdcpu`); the interaction of a CPU-left activation stream with a CPU-right weight stream is unverified.

Design gate (all required before any implementation):

1. Write out the exact forward and backward math for the streamed LoRA-B GEMM — what lives on CPU, what on GPU, and what each kernel call computes — and confirm a grad_weight path exists or design one explicitly.
2. Isolated kernel/autograd test (the one place a kernel-only test is acceptable per the validation policy): forward output and *both* grads must match the current staged `grouped_mm` LoRA-B path within tolerance on representative grouped shapes.
3. Only then wire it behind a sub-flag (e.g. `asym_cpu_adamw_weight_offload_stream`) and re-run the Stage 5 `profile_lora_lf.sh` A/B. Accept only if it lowers peak further or improves latency with **no** change to the loss curve.

If any of (1)-(3) fails, Design B is abandoned and Stage 2's stage-to-HBM remains the shipping design.

### Stage 7b — knob tuning

- `persistence_threshold_numel` (keep more/fewer banks resident), prefetch depth/bucket. Each change must re-pass the Stage 5 `profile_lora_lf.sh` gate (peak still `< 30 GiB`, latency within 50%). Tabulate peak/latency per setting from the e2e profiler, not from toy runs.

Unresolved risks carried forward:

- Expert recompute (`ASYM_EXPERT_GC_*`) + weight offload interaction (double forward changes the access trace): unsupported in this plan; assert-and-reject, revisit if recompute + offload is needed.
- Gradient accumulation (`GRADIENT_ACCUMULATION_STEPS>1`): each micro-batch re-gathers per layer; verify the home is not refreshed mid-accumulation (refresh only after `optimizer.step()`), and add a dedicated A/B at `GAS=2` before relying on it.
- Multi-adapter / weight-tied LoRA: the coordinator keys on `id(param)`; verify tied banks register one home and gather once.
- Plain (non-activation-offload) and attention-LoRA paths remain resident by design; if a future config disables expert activation offload, weight offload is a no-op there and must report `weight_offload_param_count=0` rather than silently doing nothing.

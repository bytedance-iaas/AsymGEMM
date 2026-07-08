# AsymGEMM NVMe Offload — Implementation Plan (v5, code-verified 2026-07-04)

Three composable, opt-in backend tokens on ONE local NVMe store reusing DeepSpeed's AIO engine:

- `asym_cpuadamwds_actnvme` — **activation spill**: forward produces CPU activations in creation (FIFO) order; under a CPU-RAM watermark the oldest spill to NVMe; backward consumes in ≈reverse (LIFO) order and fetches back on demand; every buffer/blob is released the moment its handle is released. This holds **across layers** (unsloth-GC boundary tensors, Substrate A) **and intra-layer** (fine-grained engine handles, Substrate B). *Novel*; raises the CPU ceiling that binds max seq. **Core target.**
- `asym_cpuadamwds_panvme` — base/frozen weights (`HostWeight` CPU homes) → NVMe, bounded pinned cache + trace prefetch. Frees 61 GiB (q3-32b) / 105 GiB (q3.5-35b-a3b) of CPU.
- `asym_cpuadamwds_bothnvme` — both roles; compound for the max-seq hero result.

Flagship anchor (verified in `scripts/lf/runs.log`, 2026-07-04): `asym_cpuadamwds|recomp-off-full-fg|ligerloss1` with policy `none|false|true|false|false|false` (**attnact=true**). Recent operating points: q3-32b s70000–80000 b8 (ceiling probes), llama3.3-70b s32000–36000 b8, q3.5-35b-a3b s80000 b8 (moefg1).

**Delivery ladder (locked): v1 fully SYNC (bit-exact, debuggable) → v2 async spill (writer thread) → v3 reverse-order read prefetch → v4 optional H2D stream.** Every rung env-gated with sync fallback. Correctness never depends on prefetch.

## v5 changes vs v4 (read this if you knew v4)

1. **The capacity target moved.** Measured from real flagship artifacts (see §0.2): at q3-32b s70000 b8, process RSS peaks at **648 GiB already during forward**, of which the `ActivationOffloadManager` engines own only a per-layer window (52 GiB measured on q3.5-35b s80k) and host weights 61 GiB. The dominant, seq-scaling CPU consumer is the **unsloth-GC boundary stream**: one `[M,H]` bf16 CPU tensor per decoder layer saved FIFO in forward and consumed LIFO in backward (`LlamaFactory .../model_utils/checkpointing.py:93,:106`) — ~342 GiB analytic at q3-32b s70k, plus pageable-alloc overhead. v4 targeted only the manager handles; **v5 makes the boundary stream Substrate A (primary) and the manager handles Substrate B (secondary), under one governor.** This is also why the host-mem watchdog (uncommitted `run_lf_lora_sft.sh` edit, 50 GB floor) exists — actnvme is the fix.
2. Under the flagship, `recomp-off-full-fg` runs decoder layers under unsloth GC ⇒ the fine-grained Functions execute **inside each layer's backward recompute window** (outer forward takes the no-grad path, `dense_mlp_finegrained.py:885`). Substrate-B handles therefore have short reuse distance; they are the *newest* entries in the FIFO and spill only under extreme pressure — exactly what oldest-first gives for free.
3. attnact=true is part of the flagship ⇒ attention U/S coverage is promoted into the core rollout (was optional Stage 5). Corrected seal rule for shared q/k/v `U`: seal at the **end of the last acquiring Function's forward** (v_proj), NOT at the context cache-clear point (`attention_activation_offload.py:478-479`) — the cache clears *before* v_proj's own CPU-left read at `:620`.
4. Single un-spill hook: every manager-side consumer passes through `wait_cpu_ready()` (`stage*()` call it internally, `activation_offload.py:238,:253,:279-280`) — EXCEPT attention backward's direct `u_handle.tensor` read at `:715`, which gets one added ensure call. The governor release hook must run **before** `release_cpu`'s internal `wait_cpu_ready` (`:318`) or a spilled-never-consumed handle would be fetched just to be freed.
5. `asym_gemm/lf.py` → **`asym_gemm/integrations/lf.py`**; all script anchors refreshed (they moved; semantics mostly unchanged; one real drift: `existing_profile_complete` takes backend as `$2`, `profile_lora_lf_test_source.sh:1303-1306`).
6. Environment corrections: DeepSpeed is the **editable checkout `../../third_party/deepspeed` (v0.19.2) inside `third_party/LlamaFactory/.venv` (py3.11)** — the repo `.venv` has neither torch nor deepspeed. `.aioenv` is a conda-created libaio (aarch64). Hardware is 4× GB200 (Grace), 1325 GiB RAM, swap 0, `/scratch_local` = 4-member NVMe RAID0 (`md0`, ext4, ~11 TB free).
7. Dropped/kept from v4: still ONE watermark governor (no per-tag policies, no eager-spill lists — budget knob covers eager mode: budget 0 ⇒ spill everything FIFO); optimizer/LoRA-state NVMe still pointless (`cpu_adam.py:124-134` — state is LoRA-only, hard-guarded).

---

## 0. Verified ground truth (all anchors re-checked 2026-07-04, working tree @ d2feadf)

### 0.1 Environment & DeepSpeed AIO

Paths (this tree is also mounted at `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM` — same NFS files):

```bash
REPO=/home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
SFT_ROOT=/home/shutianluo/kevin/AsymGEMM-SFT
LF_PY=$SFT_ROOT/third_party/LlamaFactory/.venv/bin/python     # py3.11.15; torch + deepspeed 0.19.2 (editable
                                                              # $SFT_ROOT/third_party/deepspeed) + asym_gemm (editable)
# .aioenv sidecar REQUIRED for JIT build + runtime; run_lf_lora_sft.sh:34-48 already exports it.
export AIO_HOME=$REPO/.aioenv
export CPATH="$AIO_HOME/include:${CPATH:-}" LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
```

`.aioenv` = conda-forge `libaio-0.3.113` (linux-aarch64), created 2026-06-22; `scripts/lf/setup_aioenv.sh` is the (deb-based) rebuild path, idempotence-guarded on `lib/libaio.so`. Empirically confirmed 2026-07-01: `AsyncIOBuilder().is_compatible()==True` only with `.aioenv`; builds ~23 s; `aio_handle(1048576, 16, False, True, 4)` → `get_alignment()==2048`; pinned pwrite/pread roundtrip and offset-based roundtrip into one file both pass. Same knobs as the repo's `zero3_offload_panvme` DS baseline (`ds_z3_offload_panvme_config.json`: `nvme_path=/scratch_local/user_data/shutian/kevin/cache/ds_nvme_offload`).

Hard API facts (verified in `$SFT_ROOT/third_party/deepspeed`, v0.19.2):
- ctor `aio_handle(block_size, queue_depth, single_submit, overlap_events, intra_op_parallelism)` (`csrc/aio/py_lib/py_ds_aio.cpp:22-29`); `async_pread/async_pwrite(buffer, filename, file_offset=0)` (`:88-112`); `wait()` releases the GIL (`:125-128`).
- `wait()` drains **ALL** pending ops on that handle and `assert(_num_pending_ops > 0)` — **calling wait() idle aborts the process** (`csrc/aio/py_lib/deepspeed_py_io_handle.cpp:201-220`). ⇒ Python-side pending ledgers; **separate read and write handles**.
- `num_bytes % intra_op_parallelism == 0` required (`:222-233`); files opened O_DIRECT (`csrc/aio/common/deepspeed_aio_common.cpp:269`) ⇒ **pad every offset + length to `get_alignment()`**, and IO buffers must be *allocated* padded.
- Unpinned/CUDA buffers silently bounce through a managed pinned buffer (correct, one extra copy — `csrc/aio/py_lib/deepspeed_cpu_op.cpp:25-31,85-104`); keep the common path pinned.
- Each single op is internally split across `intra_op` threads × qd-deep libaio rings (`deepspeed_cpu_op.cpp:67-83`) → up to threads×qd×block in flight per op. Per-op `wait()` is fine.
- Reuse boundary (panvme, Stage 7): **reuse** the AIO engine — `aio_handle` + `swap_in_tensors`/`swap_out_tensors` (`swap_tensor/utils.py:20-27`, the rc==0 primitives) — and the `SwapBufferManager`/`SwapBufferPool` pinned-pool pattern (`:97-227`; sole coupling = a rank-0 `print_object` at `:195`, guard it — `deepspeed.comm` may not be initialized under the asym backend); **mirror** `PartitionedParamStatus` (`partitioned_param_swapper.py:26-34`) and the `PartitionedParameterCoordinator` trace/prefetch/2×-in-flight/Belady logic (`partitioned_param_coordinator.py:187-256,:428-461,:606-630`) as `BaseWeightPager`. **NOT drop-in reusable** (reuse parts, re-implement logic): the top-level `AsyncPartitionedParameterSwapper` — welded to `ds_id`/`ds_tensor`, built only in `zero.Init` (`partitioned_param_swapper.py:39`, `partition_parameters.py:1095`); GDS (no hardware path).

### 0.2 Where the CPU bytes actually are (measured, existing artifacts)

| run (completed flagship artifacts) | RSS peak | host_weight/cpu | manager `max_cpu_peak_bytes_live` | boundary stream (analytic) |
|---|---|---|---|---|
| q3-32b s70000 b8 full-fg (`profiling_ceiling_q3-32b_70000_*/.../b8_s70000_ga1`, partial) | **648 GiB (during forward)** | 61.0 GiB | (not harvested in partial) | 64·(560k·5120·2B) = **342 GiB** |
| q3.5-35b-a3b s80000 b8 full-fg moefg1 (`profiling_q35_final2_asym80k_20260703T101943Z/...`) | fwd 355 / bwd 570 / opt **591 GiB** | 104.6 GiB | **52.1 GiB** | (smaller H; fwd RSS consistent) |

Reading: the **cross-layer** CPU stream that scales with seq is the unsloth-GC boundary tensors (Substrate A). The manager engines (Substrate B) peak at a per-layer window because the fg Functions run inside the recompute. Host weights are the panvme target. The backward RSS growth on q35 (355→570) is per-layer-window transients + `save_on_cpu` recompute packs + pool growth — window-lifetime, pressure-relevant, not spill targets.

### 0.3 Substrate A — unsloth GC boundary stream (PRIMARY actnvme target)

`$SFT_ROOT/third_party/LlamaFactory/src/llamafactory/model/model_utils/checkpointing.py` (file already carries project-owned env hooks `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU`, `UNSLOTH_GC_OUTER_HBM_EVERY_N` — precedent for a minimal env-gated hook):

- `UnslothGradientCheckpointing.forward` (`:84-100`): `saved_hidden_states = hidden_states.to("cpu", non_blocking=True)` (`:93`) — **pageable** (not pinned, not pooled); `ctx.save_for_backward(saved_hidden_states)` (`:97`); layer forward under `no_grad` (`:94-95`).
- `backward` (`:104-117`): `hidden_states.to("cuda", non_blocking=True)` (`:106`), recompute under `torch.autograd.graph.save_on_cpu(pin_memory=True)` when `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU` (`:109-111`), inner `torch.autograd.backward` (`:116`). ctx (and the CPU tensor) is freed right after — LIFO release.
- `get_unsloth_gradient_checkpointing_func()` (`:78-119`) constructs the Function; `_gradient_checkpointing_enable` (`:148-176`) binds it per-module via `_set_gradient_checkpointing`; `get_custom_gradient_checkpointing_func` (`:122-145`) wraps it (trainable-only — with LoRA on all layers, every boundary is saved). `UNSLOTH_GC_OUTER_HBM_EVERY_N` diagnostic keeps every-Nth boundary on HBM (`:62-75,:90`).
- Lifecycle: created strictly in layer order during forward (FIFO), one per layer, consumed strictly newest-first during backward (LIFO), freed immediately after that layer's inner backward. Reuse distance = rest of forward + backward down to that layer ⇒ **ideal NVMe target with seconds of lead time**. Sizes: `M×H×2` bytes (q3-32b s70k b8: 5.47 GiB/layer × 64).
- `asym_gemm` is importable from the LF venv (editable) ⇒ a lazy `from asym_gemm...` inside an env-gated branch is legal LF-side.

### 0.4 Substrate B — the `ActivationOffloadManager` engines

Manager internals (`asym_gemm/training/activation_offload.py`, read in full):
- ONE global pool `_CPU_BUFFER_POOL` keyed `(dtype, shape, pinned)` (`:10`), cap `ASYM_EXPACT_CPU_POOL_MAX_BYTES` (`:21`, default 32 GiB `:13`; full-fg sets 192 GiB for routed MoE). `_alloc_cpu` (`:74-86`, **silent unpinned fallback** `:85-86`), `_return_cpu` (`:89-103`, requires contiguous CPU).
- `CPUActivationHandle` frozen dataclass (`:106-116`): `tag, tensor, original_device, original_dtype, original_shape`; `nbytes` is a **live property over `.tensor`** — the governor must cache nbytes at track time.
- Per-manager accounting keyed by `tensor.data_ptr()`: `_active_cpu_bytes` (`:164`), `_mark_cpu_live` (`:331`); per-manager D2H ready events `_pending_cpu_ready_events` (`:166`, recorded in `offload()` `:194-197`, popped in `wait_cpu_ready` `:230-235`).
- **Choke points**: `stage`/`stage_rows`/`stage_concat_columns` all call `wait_cpu_ready` first (`:238,:253,:279-280`); `release_cpu` also calls it (`:318`) then pops accounting and pool-returns (`:315-326`). There is **no `_pop_active`** — Stage 3 adds it.
- `adopt_cpu` pass-through (`:220-228`) can put **foreign (non-pool, unpadded) tensors** into handles and later into the pool ⇒ spill eligibility must check per-buffer paddedness+pinnedness, not assume it.

Engines live under `recomp-off-full-fg` (derivation: `profile_lora_lf_test_source.sh:3147-3167` common + `:3195-3210` full-fg ⇒ `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD=1`, `ASYMM_ATTN_ACT_OFFLOAD=true`, `ASYMM_EXPERT_ACT_OFFLOAD=false`, `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu`; + for routed MoE models only: `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD=1`, `ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1`, `ASYMM_QWEN3_MOE_FG_DA_GPU=1`, `ASYMM_QWEN3_MOE_FG_KEEP_DGRADS_HBM=1`, pool 192 GiB):

1. **`dense_mlp_finegrained.py` — `_FinegrainedDenseMLPFunction` (`:183`)**, fresh manager per call (`:212`). Cross-phase handles → ctx (`:291-297`): `x_cpu` `mlp.X` (`:216`), `gate_cpu`/`gate_low_rank_cpu` `mlp.gate`/`mlp.S_gate` (`:230-231`), `up_cpu`/`up_low_rank_cpu` (`:246-247`), `act_cpu` (`:252` cpu-act | `:259` gpu-act), `down_low_rank_cpu` `mlp.S_down` (`:268`). Forward consumers: `x_cpu` waits `:226,:242`; `act_cpu` `:265`; gate/up staged `:255-261`. **Seal point: immediately before `layer._last_activation_offload_stats = manager.snapshot()` at `:300`.** Backward consumption order (≈LIFO): S_down stage `:329`; act wait+read `:334-335`; gate/up (cpu-silu `:355-358` | gpu-silu stages `:363,:370,:376`, transients `mlp.dup` `:366` / `mlp.dgate` `:379`); S_gate stage `:386` (rel `:395`); **x wait+read `:396-397` and `:420-421`** (oldest handle, consumed last — perfect FIFO/LIFO endpoints); S_up stage `:415` (rel `:419`); `finally` releases all + snapshot (`:440-453`). Under GC the outer no-grad pass takes `_finegrained_dense_mlp_no_grad_forward` (`:885-886`, `:668-671`) whose handles are created+released in-call (never sealed ⇒ pressure-only).
2. **`attention_activation_offload.py` — `_AsymActivationOffloadLoRALinearFunction` (`:560`)**, fresh per-call manager (`:612`) **plus** one persistent `AttentionActivationOffloadContext.manager` per attention parent (`:447`; installed `integrations/lf.py:1403`; linears installed `:1213`, gate `_attention_act_offload_enabled` `integrations/lf.py:1324-1327`). Handles: `u_handle` `{role}.U` — non-shared `manager.offload` `:615` OR shared via `acquire_source` `:617-618` (allocated in the context manager `:459/:470`; `_SharedActivationSource` refcount `:417-440`; cache clears when v_proj seen / all of `{q,k,v}_proj` seen `:478-479`; final free via `context._release_source` → `release_cpu` `:483`); `s_handle` `{role}.S` `:629`. Forward consumer: `u_handle.tensor` CPU-left read at `:620` (GPU kernel streams CPU memory; stream-ordered, no host wait). Backward: **direct `u_handle.tensor` read `:715` (no `wait_cpu_ready` — needs the one added ensure call)**, `s_stage = manager.stage(s_handle,...)` `:733`; `finally` `:736-743`. Snapshot merge `_update_snapshot` `:501-511` (per-call manager + cumulative `source_context`).
3. **`qwen3_moe_finegrained.py` — `_Qwen3MoeFinegrainedFunction` (`:440`)** (routed MoE models only; gate `qwen3_moe.py:2495-2500`, set from `integrations/lf.py:1334-1337,1899,1930,1989`), fresh manager `:484`. Handles → ctx `:698-704`: `x_cpu` `moe.X` (`:498/:500`), `gate_cpu`/`S_gate` (`:527/:528`), `up_cpu`/`S_up` (`:556/:557`), `act_cpu` (`:568`), `S_down` (`:588`). Forward consumers: x `:508-511,:537-540`; act `:578-581`; stages `:563,:565,:637,:650`. **Seal point: before `layer._last_activation_offload_stats = _record_manager_peaks(layer, manager)` at `:706`.** Backward: S_down `:790/:882`; act reads `:825/:906` (waits `:777/:902`); gate `:937/:955`; up `:949`; S_gate `:987`; **x row-slice reads `:1021`** (wait `:968`); transients `moe.dup` `:945`/`moe.dgate` `:962` (skipped when `KEEP_DGRADS_HBM=1`, which full-fg sets); `finally` `:1393-1408`.

Inactive under the flagship (cover later, same mechanism): `qwen3_moe.py::_ActivationOffloadQwen3ExpertFunction` (`:1010`, manager `:1028`, end-of-body release loop **without finally** `:1470-1482`), `llama4_experts.py::_ActivationOffloadLlama4ExpertFunction` (`:229`, manager `:247`, release loop `:724-736`), `llama4_shared_mlp.py::_Llama4SharedMLPActivationOffloadFunction` (`:236`, `shared.X` `:266`, finally `:425-428`) + `qwen35_shared_mlp.py` subclass. **Non-targets** (no manager): `kt_moe.py` (C++-held saved state), `decoder_activation_offload.py`/`linear_attention_activation_offload.py`/attention saved-tensor wrapper (cumulative-counter `saved_tensors_hooks` wrappers), `decoder_layer_glue_gc.py`, `decoder_checkpoint.py`/`attention_checkpoint.py`, `sdpa_recompute.py`, `chunked_mlp.py` (unwired), `qwen3_moe_routed_gemm.py` (weight kernels), `exp_act_offload_lora.py` (kernel library; `stage_low_rank_from_cpu` `:330-342` calls `manager.stage` → covered via the wait hook).

### 0.5 panvme target — base weights

- Single chokepoint `HostWeight` (`host_weight.py`, byte-stable): `__init__` `:185-242` (pin `:222`; **sole `_tensor` write `:227`**); `.tensor` `:244-246` and `.weight` `:302-304` are plain properties → clean lazy-materialize interception. **BUT** `shape :272`, `dtype :276`, `device :280`, `out_features :292`, `in_features :296`, `grad :300`, `grouped_nt_tensor :315`, `pin_memory :340` also read `self._tensor` directly — the surgery must make the metadata-derivable ones read `self._metadata` (`metadata :266-268`, `nbytes :283-284` already do). `HostWeightMetadata` (`:14-27`) carries everything needed: `shape, dtype, nbytes, device, is_pinned, data_ptr, numa_node, cpu_socket, …` — but `device` is stored as **str** (`_make_metadata` `:158`), so the paged `device` property returns `torch.device(self._metadata.device)`; `data_ptr` goes stale after paging (reporting-only — acceptable). External per-instance attr `_asym_quantized_cache` (`frozen_linear.py:369`; `_get_quantized_host_weight` `:356-380`, bf16 short-circuit `:363-364`) ⇒ panvme registration asserts `precision=="bf16"`.
- Adoption entries: `adopt_host_weight()` (`offload.py:176-213`) called from `integrations/lf.py:1194` (base linears), `dense_mlp_finegrained.py:728/741/754`, `qwen3_moe.py:2953`, `llama4_shared_mlp.py:525/537/549`, `llama4_moe.py:92` (router), plus embeds/norms (`offload.py:352/:394/:442` — never pinned; embed consumed by CPU-side `F.embedding` per microbatch `offload.py:381`) — **embeds/norms excluded by policy**.
- Consumption: dense fwd `frozen_linear.py:1309-1321` (`.weight` `:1312`), dense dX `:1356-1368`, grouped autograd fwd `:1447-1498` (`.weight` `:1486`), grouped dX `qwen3_moe.py:476-501`, native windowed bwd `qwen3_moe.py:1878/:1886/:1893`, llama4 grouped dX `llama4_experts.py:89-99`. **Kernel launches are async** (`_asym_bf16_nt` `frozen_linear.py:707,:726-728`; pinned-stream comment `:807-809`) ⇒ recycle a weight buffer only after a post-enqueue CUDA event completes.
- Runtime `.weight` readers besides GEMMs: `is_pinned()` predicates (`qwen3_moe.py:1775,:2665,:2670`, `llama4_moe.py:242`) — each immediately precedes a consuming kernel (lazy fetch self-consistent); `_ensure_qwen3_moe_finegrained_bases` (`qwen3_moe.py:2509-2542`) **lazily slices fused gate_up into two NEW `AsymGroupedFrozenLinear`** — must run eagerly before registration under panvme+moefg.
- HF expert source params are 0-numel'ed after adoption (`qwen3_moe.py:2144-2153`, `llama4_moe.py:63-69`) — HostWeight owns the only copy; spilling genuinely frees RSS. Insertion point for the registration walk: end of `apply_lf_asym_lora`, before `return model, report` at `integrations/lf.py:2426`. Optimizer state is LoRA-only (`cpu_adam.py:124-134,:194,:213-219`); `weight_offload.py` is test-only/dormant (only its idle stream at `:95`).

### 0.6 Backend tokens & script dispatch (working tree; uncommitted `run_lf_lora_sft.sh` edit = host-mem watchdog, orthogonal)

Grammar: `model|gpus ; backend|recompute|liger[|kernelcode] ; seq|batch|grad_accum ; policy-tuple ; [lora] ; [flash_attn]`. Policy tuple = `policy|expact|attnact|layeract|layergc|sdparecomp`, parser `parse_exp_act_policy_tuple` (`profile_lora_lf_test_source.sh:704-735`) — **accepts 4–6 fields** (trailing two default false). Policy-list override `ASYMM_EXP_ACT_POLICIES` (`:2297`; CLI `--asymm-exp-act-policies` `:2332-2333`). `recompute_label()` `:1008-1030` (+ `-kerXYZ` `:975-986`, `-ohbm<N>` `:989-1006`); env derivation lives in `run_job` (`:3140-3151`, `:3168-3210`).

Sites to extend in Stage 2 — `profile_lora_lf_test_source.sh` (4370 lines): `append_backend_spec` `:1069-1153` (case arms `:1103-1126`, die `:1126`), `backend_gpu_count` `:899-908` (1-GPU asym line `:903`), `cpuadam_backend_for_label` `:1054-1060`, per-job derivation `:3291-3307`, `run_env` block `:3436-3597` (`ASYM_GEMM_LF_CONFIG_*` mirror `:3546-3593`); backend is the first `path_label` component (`job_root_path` `:1890-1915`, label `:1906`) → run dirs auto-disambiguate; completion checks: `job_profile_complete` backend=`$3` (`:1259,:1262`), `existing_profile_complete` backend=**`$2`** (`:1303,:1305`). **`profile_lora_lf_test_both.sh` is byte-identical except the `PROFILERS` default (`:180-181`) — apply identical edits.** `run_lf_lora_sft.sh` (2973 lines): BACKEND case `:295-415` (`asym_cpuadamwds` arm `:397-403`, die `:414` — new arm goes after `:403`); `is_zero_backend_run` requires `BACKEND==torch` (`:708-710`) → our arms set `BACKEND=asym`, no `--deepspeed` leakage; `.aioenv` exports `:34-48`; fine-grained env defaults `:103-106`; `ASYM_GEMM_LF_CONFIG_*` mirror `:2282-2295`; `RECORD_IO=1` default `:199` + `/sys/block/md0/stat` sampler `start_io_sampler` `:2604-2626` (→ `io_samples.csv`, reduced by `scripts/lf/summarize_nvme_offload.py` → `offload_io.json` `nvme.read_gb/write_gb` — the free device-level cross-check). `run_lf_profiled_train.py` (3047 lines): classification `:577-599` (`is_asym_deepspeed_cpuadamw` `:579-582`), `_config_from_args` `:546` (env-mirror prefix `ASYM_GEMM_LF_CONFIG_` `:39`), `report()` `:2829-2919` (`activation_offload` key `:2908`). DS NVMe baselines exist: `zero3_offload_{opnvme,panvme,mem_opnvme,mem_panvme}`, `superoffload_mem_{opnvme,panvme}` (configs under `$LF_DIR/examples/deepspeed/`).

### 0.7 Profiling / gating infra

- Canonical driver `scripts/lf/profile_lora_lf_test_source.sh` (defaults: `PROFILERS=source` `:181`, `GPU_POOL` `:34`, `OVERWRITE` `:253`, `PREPARE_DATASETS` `:263`, `OUTPUT_ROOT` `:274`, `PLOT` `:320`, `MAX_STEPS`/`WARMUP_STEPS` `:192-193`; model shorthands `M[q3-32b]` etc `:48-61`; built-in default RUNS `:82` = q3-32b s60000 flagship).
- `source_profile.json` from `report()` (`run_lf_profiled_train.py:2829-2919`). `activation_offload` block from `_activation_offload_counters_from_model()` (`:2198-2280`; walks `root.named_modules()` `:2208`; per-module `_last_activation_offload_stats` dicts flow into rows **automatically** — new `snapshot()` keys propagate for free); the **aggregates tail** `:2265-2280` is explicit and must be extended. Per-step RSS = `step_samples.rows[].{forward,backward,optimizer,training_step}_process_rss_peak_end_bytes` (generic emit `:2708`, bind `:2749`). `memory_attribution` via `trace_handle.memory_summary` (`:2882-2886`), rows have `category=host_weight, device=cpu`.
- Row-semantics trap (3 lifetimes): fg dense/MoE rows = per-(module, last call) snapshots (fwd `:300`/`:706`, bwd overwrite `:453`/`:1408`) ⇒ per-microbatch directly; attention rows = per-call local + **cumulative** `source_context` (`:501-511`); saved-tensor wrapper rows = cumulative across the run (`decoder_activation_offload.py:201-211`).
- Compare-gate template `scripts/lf/compare_liger_loss_profiles.py` (args `:33-42`, `_fail` → `{"ok":false}` + `SystemExit(2)` `:46-51,:377-378`, memory `:156`, medians from `step_samples.csv` `:208`). Postprocess: `_asym_cpu_adamw_rows` `postprocess_lf_profile_artifacts.py:378` (csv emit pattern), `memory.md` emitter `:1803` (written `:2200`). Eyeball: `scripts/lf/show_metrics.py`.

### 0.8 Prior-art cross-check (verified in local checkouts) & hardware

- **Megatron** `$SFT_ROOT/third_party/megatron-lm/megatron/core/pipeline_parallel/fine_grained_activation_offload.py`: FIFO offload / LIFO reload deques (`:408,:182,:990`); **terminal margin** — never offload the last groups, their reload would stall backward immediately (`:430-431,:536-554`); `should_bulk_offload` (`:962-984`). Our watermark generalizes the margin: everything within the CPU budget stays resident; the newest (first-consumed) tensors are by construction the last to spill.
- **TE** `cpu_offload_v1.py`: dedicated `d2h_stream`/`h2d_stream` (`:366-367`) + reload double-buffering (`:578`, ctor `:328,:351`, D2H `:464-466`) — the v4-rung H2D-stream pattern. (`cpu_offload.py` is a newer refactor with different anchors — cite v1.)
- **DeepSpeed**: no activation prefetch (its `cpu_checkpointing` restore is sync); param coordinator = the trace record→freeze→invalidate + byte-budgeted lookahead pattern reused by the panvme pager.
- Hardware: 4× GB200 (Grace, aarch64, driver 580.105.08), 144 cores, **1325 GiB RAM, swap 0** (`nvidia-smi` currently hangs in sandboxes — use `lspci`/device nodes). `/scratch_local` = `md0` RAID0 (4× 3.5 TB NVMe partitions, 64k chunk, ext4, **~11 TB free**); measured ~26 GB/s read / ~14 GB/s write. Host-mem watchdog (uncommitted): `HOST_MEM_WATCHDOG=true`, floor 50 GB, SIGSTOP→SIGINT→kill escalation — heavy baselines die here; actnvme must keep MemAvailable above the floor.
- Feasibility: actnvme traffic = **overflow only** ≈ `max(0, footprint − budget)` written once + read once per step. q3-32b s70k: footprint ≈ 342 GiB boundary + ~60-120 GiB window; budget 250 GiB ⇒ ~150 GiB/step each way; at observed long-seq step times (minutes) ⇒ ~1-2 GB/s average — far under 14 GB/s. Endurance: overflow-only at research duty cycles is a non-issue; arena reuses offsets every microbatch (holds ~one microbatch's overflow).

---

## Design contract (all stages)

1. **One store, role-tagged.** `NVMeStore` serves `{base_weight, activation}`; `actnvme→{activation}`, `panvme→{base_weight}`, `bothnvme→both`. Consumers see only the store API. Placement policy is NOT abstracted — it stays in the governor/pager (it *is* the algorithm).
2. **Zero compute-semantics changes.** What is saved vs recomputed, every kernel, every launch: untouched. actnvme changes only the *backing residency* of already-CPU tensors, and only under pressure. **No new small GEMMs, no per-expert loops, no added hot-path kernel launches** — the only new CUDA ops are one event record per Function forward (seal) and one per boundary offload.
3. **Spill on pressure, FIFO; consume LIFO; release on last use.** Activations stay in CPU RAM up to a budget; only overflow spills, oldest-first (= farthest future use under LIFO consumption = exact Belady). Budget `0` ⇒ eager spill-everything (max-capacity mode). No producer lists, no tag policies.
4. **No kernel computes from NVMe.** Always `NVMe → pinned CPU → (H2D) → compute`; `pinned CPU → NVMe`.
5. **Single-owner AIO handles.** Write handle owned by ONE thread (main thread in sync mode; writer thread in async); read handle by the main thread. Python pending ledgers; never `wait()` idle. Debug-assert thread identity.
6. **Event-gated buffer reuse.** Any pinned buffer a GPU kernel may still stream is recycled only after a post-enqueue CUDA event completes (the seal/boundary event).
7. **Off = byte-identical.** No `*nvme` token ⇒ no deepspeed import, no thread, no file, identical allocations; every hook is a `None` check on a module-level singleton.
8. Every file offset and IO length padded to `store.align`; IO buffers allocated padded.
9. **Sync before async.** Each stage lands the sync path first and gates on correctness (bit-exact unit tests + e2e loss match) before any overlap flag is flipped.

---

## Stage 0 — NVMe traffic census (postprocess-only; review artifacts before coding Stage 3+ budgets)

**Purpose:** turn existing + two fresh profile runs into the decision artifact: per-substrate/per-layer bytes per step, overflow-vs-budget table, feasibility verdict. §0.2 already establishes the headline (boundary stream dominates); the census makes it per-layer and per-tag and pins budgets for the gates.

**Scope:** NEW `scripts/lf/project_nvme_traffic.py` (callable standalone and from `postprocess_lf_profile_artifacts.py` beside the other emitters). ZERO runtime edits.

```python
# Input: run dir (source_profile.json [+ step_samples.csv]). Output: nvme_traffic_projection.{csv,md}
# 1. steps = trainer measured steps + warmup; micro = steps * grad_accum.
# 2. Substrate A (boundary): num_layers × M × hidden × 2  (M = seq×batch from config; layers/hidden
#    from the model shorthand table kept in this script). Cross-check: forward RSS peak − host_weight
#    − pool caches − baseline slack ≈ A (report both, flag >15% disagreement).
# 3. Substrate B: for each activation_offload.rows[] row classify semantics by module class:
#    per-call fg row (mlp.*/moe.* — bytes are already per-microbatch) | attention mixed row
#    (local per-call + source_context cumulative ÷ microbatch-forwards) | wrapper row (cumulative ÷ …).
#    Emit per (module, tag): fwd_saved_bytes_step (write candidate; read ≈ write), transient_bytes_step
#    (mlp.dact/dgate/dup, moe.dgate/dup, *_for_*, {role}.S_stage — never-spill, shown for context),
#    stage_bytes_step (H2D volume, v4-rung input).
# 4. panvme sheet: memory_attribution host_weight/cpu rows → per-component bytes; per-step read = 2×(fwd+bwd
#    reuse); model total = Σ eligible components (exclude embed/norms rows).
# 5. Summary: total A+B write ceiling at budget 0; write_seconds_step = total/14e9, read_seconds_step = /26e9
#    vs measured step_seconds; overflow table for budget ∈ {150,250,400,600} GiB; per-layer top-N; per-tag totals.
# Keep the tag lists in one dict at the top. When the run is a DS *nvme baseline, print offload_io.json
# device totals next to the projection (pipeline-vs-device sanity).
```

### Validation (Stage 0 gate)

```bash
cd $REPO
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage0_census PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=1 MAX_STEPS=4 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 45000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
$LF_PY scripts/lf/project_nvme_traffic.py --run-dir profiling_nvme/stage0_census/<run_dir>   # each seq point
```

Accept: artifacts exist; Substrate-A analytic vs RSS-residual agree ≤15%; B rows sum to the `activation_offload` aggregates; the two seq points scale ≈ linearly in tokens; feasibility verdict unambiguous (`write_seconds_step ≪ step_seconds` at the intended budgets); budgets for Stage 3/5 gates chosen and recorded in the .md. Also run it once on the existing `profiling_q35_final2_asym80k_*` dir (free MoE data point).

Risks/watch: row-semantics misclassification (per-call vs cumulative — assert cumulative rows grow with MAX_STEPS across the two runs, per-call rows don't); model-shape table drift (keep in one dict).

---

## Stage 1 — `NVMeStore` substrate (sync + async write paths)

**Scope:** NEW `asym_gemm/training/nvme_store.py` + NEW `tests/training/test_nvme_store.py`. Zero edits elsewhere ⇒ isolated unit gate suffices (the one exception to the e2e rule — this is pure IO plumbing with no training-visible behavior).

```python
# asym_gemm/training/nvme_store.py
from __future__ import annotations
import os, queue, threading, time
from dataclasses import dataclass, field
from typing import Any, Callable
import torch

@dataclass(frozen=True)
class NVMeStoreConfig:
    path: str                                    # ASYM_NVME_PATH (required when roles set)
    roles: frozenset[str]                        # {"base_weight","activation"} from ASYM_NVME_ROLES
    sync: bool = True                            # ASYM_NVME_SYNC (v1 default 1; Stage 5 flips to 0)
    aio_block_size: int = 1 << 20                # ASYM_NVME_AIO_* overrides
    aio_queue_depth: int = 16
    aio_intra_op_parallelism: int = 4
    aio_single_submit: bool = False
    aio_overlap_events: bool = True
    min_swappable_bytes: int = 1 << 20           # ASYM_NVME_MIN_SWAP_BYTES
    activation_arena_bytes: int = 1 << 41        # sparse file, 2 TiB logical, allocated on demand
    max_inflight_spill_bytes: int = 8 << 30      # writer backpressure (async only)

def _config_from_env() -> NVMeStoreConfig | None:
    roles = frozenset(r.strip() for r in os.environ.get("ASYM_NVME_ROLES", "").split(",") if r.strip())
    if not roles: return None
    bad = roles - {"base_weight", "activation"}
    if bad: raise ValueError(f"unknown ASYM_NVME_ROLES: {sorted(bad)}")
    path = os.environ.get("ASYM_NVME_PATH") or _die("ASYM_NVME_PATH required with ASYM_NVME_ROLES")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1: _die("asym NVMe store is single-process only (deferred)")
    return NVMeStoreConfig(path=path, roles=roles, sync=_env_bool("ASYM_NVME_SYNC", True), **_overrides())

def _pad(n: int, a: int) -> int: return (n + a - 1) // a * a

def _flat_u8(t: torch.Tensor) -> torch.Tensor:
    """uint8 alias of t's WHOLE storage (zero-copy). All IO uses this — one tensor == one IO op,
    never fragmented, never per-row/per-expert loops."""
    out = torch.empty(0, dtype=torch.uint8)
    out.set_(t.untyped_storage(), 0, (t.untyped_storage().nbytes(),))
    return out

def io_ready(t: torch.Tensor, align: int) -> bool:
    """Spill-eligibility predicate: pinned + whole-storage length aligned. Foreign adopt_cpu
    tensors that fail this simply stay resident (counted for pressure, never spilled)."""
    return t.is_pinned() and t.untyped_storage().nbytes() % align == 0

def alloc_padded_pinned(shape, dtype, *, align) -> torch.Tensor:
    """Pinned CPU tensor over PADDED backing storage; exact contiguous view returned.
    Caching host allocator makes this cheap after warmup; on pin failure fall back to
    torch.empty(...) unpinned (io_ready() then excludes it — never crash the run)."""
    nbytes = _pad(_numel(shape) * torch.empty(0, dtype=dtype).element_size(), align)
    storage = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    t = torch.empty(0, dtype=dtype)
    t.set_(storage.untyped_storage(), 0, shape)          # contiguous strides
    return t

@dataclass
class BlobRef:
    role: str
    file: str            # per-blob file (base_weight) | role arena file (activation)
    offset: int          # aligned; 0 for per-blob files
    length: int          # padded bytes on disk == source storage nbytes
    logical_nbytes: int
    durable: threading.Event = field(default_factory=threading.Event)

@dataclass
class NVMeStoreStats:    # surfaces in the profile `asym_nvme` block
    bytes_written: dict = ...; bytes_read: dict = ...; write_ops: dict = ...; read_ops: dict = ...
    spill_wait_ms: float = 0.0       # sync-mode inline write time (the v1 stall we measure)
    fetch_wait_ms: float = 0.0; spill_backpressure_ms: float = 0.0
    inflight_peak_bytes: int = 0; arena_peak_bytes: dict = ...; wasted_writes: int = 0
    def as_dict(self) -> dict[str, Any]: ...

class NVMeStore:
    def __init__(self, cfg):
        from deepspeed.ops.op_builder import AsyncIOBuilder     # imported ONLY when enabled (rule 7)
        m = AsyncIOBuilder().load(verbose=False)
        mk = lambda: m.aio_handle(cfg.aio_block_size, cfg.aio_queue_depth, cfg.aio_single_submit,
                                  cfg.aio_overlap_events, cfg.aio_intra_op_parallelism)
        self.cfg = cfg
        self._read_h = mk()                                     # MAIN THREAD ONLY
        self.align = int(self._read_h.get_alignment())          # 2048 measured @ intra=4
        self.stats = NVMeStoreStats()
        self._write_h = mk()
        self._writer = None if cfg.sync else _WriterThread(self._write_h, cfg, self.stats)  # sole write-h owner in async
        if self._writer: self._writer.start()
        os.makedirs(os.path.join(cfg.path, "base_weight"), exist_ok=True)
        self._arena_path = os.path.join(cfg.path, f"activation.{os.getpid()}.arena")
        self._arena_cursor = 0; self._arena_live = 0
        self._pending_reads: dict[int, BlobRef] = {}            # MAIN THREAD ONLY

    def has_role(self, role): return role in self.cfg.roles

    # -- activation arena: bump allocator, reset-when-empty. Blob lifetime = one microbatch fwd→bwd,
    #    so live==0 recurs every microbatch; exact under grad accumulation, no trainer hooks. --
    def _arena_alloc(self, nbytes):
        length = _pad(nbytes, self.align)
        off = self._arena_cursor; self._arena_cursor += length; self._arena_live += 1
        self.stats.arena_peak_bytes["activation"] = max(self.stats.arena_peak_bytes.get("activation", 0), self._arena_cursor)
        if self._arena_cursor > self.cfg.activation_arena_bytes:
            raise RuntimeError("activation arena full — raise ASYM_NVME_ACTIVATION_ARENA_BYTES")
        return self._arena_path, off

    def blob_done(self, ref):                    # after final fetch OR dropped blob
        if ref.role == "activation":
            self._arena_live -= 1
            if self._arena_live == 0: self._arena_cursor = 0

    # -- write paths --
    def spill_sync(self, role, tensor) -> BlobRef:
        """MAIN THREAD. Caller has already synchronized the seal/ready event. Blocking write."""
        buf = _flat_u8(tensor); assert buf.nbytes % self.align == 0
        file, off = (self._blob_file(), 0) if role == "base_weight" else self._arena_alloc(buf.nbytes)
        ref = BlobRef(role, file, off, buf.nbytes, tensor.numel() * tensor.element_size())
        t0 = time.perf_counter()
        self._write_h.async_pwrite(buf, ref.file, ref.offset)
        n = self._write_h.wait(); assert n == 1
        self.stats.spill_wait_ms += (time.perf_counter() - t0) * 1e3
        self._count_write(ref, buf.nbytes); ref.durable.set()
        return ref

    def spill_async(self, role, tensor, *, ready_event, on_done) -> BlobRef:
        """ANY THREAD → writer thread (Stage 5). ready_event synchronized ON THE WRITER before pwrite;
        on_done(buf, ref) runs in writer-thread context — NEVER touch CUDA there."""
        ...  # ref allocation identical; _writer.submit(ready_event, _flat_u8(tensor), ref, on_done)

    # -- read path (MAIN THREAD ONLY) --
    def submit_pread(self, ref, dst_padded_pinned) -> None:
        if not ref.durable.is_set(): ref.durable.wait()   # write in flight (async mode) → bounded rare block
        dst = _flat_u8(dst_padded_pinned); assert dst.nbytes == ref.length
        self._read_h.async_pread(dst, ref.file, ref.offset)
        self._pending_reads[id(ref)] = ref

    def drain_reads(self) -> set[int]:
        """Blocks until ALL pending reads complete (wait() drains the whole handle). Returns arrived ref ids."""
        if not self._pending_reads: return set()
        n = self._read_h.wait(); assert n == len(self._pending_reads)
        for r in self._pending_reads.values(): self._count_read(r)
        done = set(self._pending_reads); self._pending_reads.clear()
        return done

    def fetch_into(self, ref, dst_padded_pinned) -> set[int]:
        t0 = time.perf_counter()
        self.submit_pread(ref, dst_padded_pinned); done = self.drain_reads()
        self.stats.fetch_wait_ms += (time.perf_counter() - t0) * 1e3
        return done                                        # includes any prefetches that arrived

class _WriterThread(threading.Thread):   # Stage 5; sole owner of the write handle in async mode
    # submit(): condition-variable backpressure on max_inflight_spill_bytes (count spill_backpressure_ms);
    # run(): pop → ready_event.synchronize() → async_pwrite → wait()==1 → count → ref.durable.set()
    #        → on_done(buf, ref).  _STOP sentinel for shutdown.
    ...

_STORE = None; _STORE_INIT = False
def get_nvme_store() -> NVMeStore | None:
    """Lazy env singleton. Without ASYM_NVME_ROLES: returns None, imports nothing, allocates nothing."""
```

Locked decisions: base_weight = one file per HostWeight (static, GB-scale, written once); activation = one pid-suffixed arena file + bump/reset-when-empty (thousands of transient blobs/step); no cancel path in the store (fetch-before-durable waits; consumed-before-submit handled ABOVE by the governor's CLAIMED state); per-op `wait()` in the writer (each op is internally 4×16×1MB parallel; batch submit is a flagged follow-up); `io_ready()` is the paddedness/pinnedness gate that makes foreign `adopt_cpu` tensors safely non-spillable.

**Efficiency:** IO unit = one whole tensor storage (GiB-scale for Substrate A: a single 5.5 GiB pwrite saturates the array via internal threading) — never fragmented; zero extra memcpys (the padded pinned buffer IS the D2H destination and the IO buffer); single-owner handles → no hot-path locks; sequential arena writes at max bandwidth.

### Validation (Stage 1 gate)

```bash
cd $REPO && export AIO_HOME=$PWD/.aioenv CPATH="$AIO_HOME/include:${CPATH:-}" \
  LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
ASYM_NVME_PATH=/scratch_local/user_data/shutian/kevin/cache/asym_nvme_test \
$LF_PY -m pytest tests/training/test_nvme_store.py -q
```

Required tests: bf16/fp32 roundtrips below/at/above 1 MiB (sync + async); two arena blobs at different offsets, no cross-corruption; arena reset-when-empty across two simulated microbatches; async spill gated on a CUDA event (write CUDA tensor D2H, record, spill_async, fetch, compare) — skip-if-no-CUDA variant with a pre-set event; fetch-before-durable blocks then succeeds; 3-deep prefetch ledger reconciles via `drain_reads`; backpressure blocks and resumes; `io_ready` rejects unpinned/unpadded; disabled env → `get_nvme_store() is None` and `"deepspeed" not in sys.modules`; clean writer shutdown; pid-suffixed arena avoids cross-run collision.

Risks/watch: handle thread-ownership is a rule, not API-enforced — assert `threading.get_ident()` in debug (`ASYM_NVME_DEBUG=1`); arena overflow at extreme seq×accumulation → loud error + env knob; O_DIRECT on ext4-over-md0 verified by the existing DS baseline (same mount).

---

## Stage 2 — Backend tokens, env plumbing, profile counters, compare gate

**Scope (no tensors move):** `scripts/lf/profile_lora_lf_test_source.sh` (+ identical edits in `_both.sh`), `scripts/lf/run_lf_lora_sft.sh`, `scripts/lf/run_lf_profiled_train.py`, `scripts/lf/postprocess_lf_profile_artifacts.py`, NEW `scripts/lf/compare_nvme_profiles.py`.

**(a) profile script** — one grouped arm in `append_backend_spec`'s case (`:1103-1126`), tokens added to `backend_gpu_count`'s 1-GPU asym line (`:903`) and `cpuadam_backend_for_label` (`:1054-1060`):

```bash
    asym_cpuadamwds|asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme) printf 'deepspeed\n' ;;
```

plus a helper + `run_env` additions inside the `:3436-3597` block (and the `ASYM_GEMM_LF_CONFIG_*` mirror `:3546-3593`):

```bash
nvme_roles_for_backend() {
  case "${1}" in
    asym_cpuadamwds_panvme)   printf 'base_weight\n' ;;
    asym_cpuadamwds_actnvme)  printf 'activation\n' ;;
    asym_cpuadamwds_bothnvme) printf 'base_weight,activation\n' ;;
    *) printf '\n' ;;
  esac
}
job_nvme_roles="$(nvme_roles_for_backend "${backend}")"
if [[ -n "${job_nvme_roles}" ]]; then
  run_env+=( "ASYM_NVME_ROLES=${job_nvme_roles}"
             "ASYM_NVME_PATH=${ASYM_NVME_PATH:-/scratch_local/user_data/shutian/kevin/cache/asym_nvme}"
             "ASYM_NVME_SYNC=${ASYM_NVME_SYNC:-1}"
             "ASYM_NVME_ACT_CPU_BUDGET_BYTES=${ASYM_NVME_ACT_CPU_BUDGET_BYTES:-auto}"
             "ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES=${job_nvme_roles}"
             "ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH=${ASYM_NVME_PATH:-…}" )
fi
```

**(b) run_lf_lora_sft.sh** — one grouped arm cloned from `asym_cpuadamwds` (`:397-403`), inserted before `asym_torch` (`:404`); die at `:414` untouched:

```bash
  asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-${BACKEND,,}}
    USE_ASYM_CPU_ADAMW=true; ASYM_CPU_ADAMW_BACKEND=deepspeed; CPUADAM_ALIAS_SELECTED=1
    case "${BACKEND,,}" in
      *_panvme)   ASYM_NVME_ROLES="base_weight" ;;
      *_actnvme)  ASYM_NVME_ROLES="activation" ;;
      *_bothnvme) ASYM_NVME_ROLES="base_weight,activation" ;;
    esac
    export ASYM_NVME_ROLES
    export ASYM_NVME_PATH="${ASYM_NVME_PATH:-/scratch_local/user_data/shutian/kevin/cache/asym_nvme}"
    # Stage 3 adds (NOT in Stage 2 — the LF hook's import target doesn't exist yet):
    # if [[ ",${ASYM_NVME_ROLES}," == *",activation,"* ]]; then export ASYM_UNSLOTH_GC_NVME=1; fi
    BACKEND=asym
    ;;
```

Mirror `ASYM_GEMM_LF_CONFIG_ASYM_NVME_{ROLES,PATH,SYNC,ACT_CPU_BUDGET_BYTES}` in the `:2282-2295` block. (`is_zero_backend_run` `:708-710` stays false — `BACKEND=asym`.)

**(c) run_lf_profiled_train.py:**

```python
_ASYM_CPUADAMW_DS_BACKENDS = {"asym_cpuadamwds", "asym_cpuadamwds_panvme",
                              "asym_cpuadamwds_actnvme", "asym_cpuadamwds_bothnvme"}   # near :579
# _config_from_args (:546): "asym_nvme_roles", "asym_nvme_path", "asym_nvme_sync",
#   "asym_nvme_act_cpu_budget_bytes" — os.environ.get(mirror) or os.environ.get(raw, "")
# report() (:2829): sibling of "activation_offload" (:2908):
"asym_nvme": _asym_nvme_summary_from_model(),
```

```python
def _asym_nvme_summary_from_model() -> dict[str, Any]:
    try: from asym_gemm.training.nvme_store import get_nvme_store
    except Exception as exc: return {"enabled": False, "reason": repr(exc)}
    store = get_nvme_store()
    if store is None: return {"enabled": False}
    out = {"enabled": True, "roles": sorted(store.cfg.roles), "path": store.cfg.path,
           "sync": store.cfg.sync, "alignment": store.align, **store.stats.as_dict()}
    from asym_gemm.training.act_spill_governor import get_act_spill_governor      # Stage 3
    gov = get_act_spill_governor()
    if gov is not None: out["act_governor"] = gov.summary()                       # incl. per-substrate splits
    from asym_gemm.training.gc_boundary_offload import get_boundary_offload_stats # Stage 3
    out["gc_boundary"] = get_boundary_offload_stats()                             # boundary manager snapshot
    model, _ = _model_and_base_model()
    pager = getattr(model, "_asym_base_weight_pager", None)                       # Stage 7
    if pager is not None: out["base_weight_pager"] = pager.summary()
    return out
```

Aggregates tail of `_activation_offload_counters_from_model` (`:2265-2280`): add `total_nvme_spilled_bytes`, `total_nvme_bytes_read`, `total_nvme_fetch_wait_ms`, `total_nvme_spill_wait_ms` summed from row stats (they appear in rows automatically once the governor extends `snapshot()`).

**(d) postprocess:** `_asym_nvme_rows()` flattener → `asym_nvme.csv` (clone the `_asym_cpu_adamw_rows` pattern at `:378`); one NVMe line in `memory.md` (emitter `:1803`).

**(e) compare_nvme_profiles.py** — clone of the liger tool (`:33-42` args, `:46-51` fail, `:208` medians):

```text
--baseline DIR --candidate DIR --target {no_change, activation_cpu, base_weight_cpu, maxseq}
--memory-metric DOTTED   ("step_samples.<col>" = max over measured csv rows; default
                          step_samples.training_step_process_rss_peak_end_bytes)
--min-memory-drop-gib | --min-memory-drop-pct   (no_change: --max-memory-drift-gib)
--max-step-ratio/--max-forward-ratio/--max-backward-ratio   (sync-capacity 1.5 informational; async ≤1.05-1.10)
--expect-nvme-role ROLE   (asserts enabled + ROLE∈roles + bytes_written>0 + bytes_read>0)
--max-loss-delta FLOAT    (median |loss_i − baseline loss_i| over measured steps)
Checks: artifacts exist (+asym_nvme.csv on candidates); finite losses; measured_steps>=3; roles match config.
{"ok":…}; SystemExit(2) on failure.
```

### Validation (Stage 2 gate — paired e2e no-change at a REAL operating point)

```bash
bash -n scripts/lf/profile_lora_lf_test_source.sh scripts/lf/profile_lora_lf_test_both.sh scripts/lf/run_lf_lora_sft.sh
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage2_nochange PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=1 MAX_STEPS=4 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds_actnvme|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
$LF_PY scripts/lf/compare_nvme_profiles.py \
  --baseline profiling_nvme/stage2_nochange/<asym_cpuadamwds dir> --candidate <…_actnvme dir> \
  --target no_change --max-memory-drift-gib 2 --max-step-ratio 1.02 --max-forward-ratio 1.02 \
  --max-backward-ratio 1.02 --max-loss-delta 0
```

Accept: drift/latency inside bounds; candidate `config.asym_nvme_roles=="activation"`, `asym_nvme.enabled==true` with `bytes_written==0` (governor not yet wired — the store idles; keep `ASYM_UNSLOTH_GC_NVME` export commented until Stage 3 lands); `asym_nvme.csv` present; losses identical (nothing moved); the tool demonstrably fails on a mismatched pair (run once against a stale dir to prove exit 2). Heavy runs sequential; both fit RAM at s30000.

Risks/watch: `_both.sh` must receive identical arms (byte-diff the two scripts after editing, expect only `:180-181`); cached run-dir reuse keys backend as first path component → distinct dirs; still pass `--overwrite true`.

---

## Stage 3 — actnvme v1 SYNC: spill governor + Substrate A (unsloth boundary) ← FIRST tensor-moving stage; THE capacity lever

**What:** run the flagship exactly as today, but boundary hidden-states become governor-tracked pinned handles; when live activation bytes cross the CPU budget, the governor spills oldest-first to NVMe **inline (sync)**; each layer's backward fetches its boundary back (sync pread) right before the H2D reload; buffers/blobs are freed at exactly today's release points. Zero compute changes; budget `≥ footprint` ⇒ behavior identical (plus pinned instead of pageable boundary copies).

**Scope:** NEW `asym_gemm/training/act_spill_governor.py`; NEW `asym_gemm/training/gc_boundary_offload.py`; `asym_gemm/training/activation_offload.py` (~30 lines of hooks); `$SFT_ROOT/third_party/LlamaFactory/src/llamafactory/model/model_utils/checkpointing.py` (**one 4-line env-gated branch** — house-consistent: the file already hosts `UNSLOTH_GC_*` project hooks); `scripts/lf/run_lf_lora_sft.sh`/profile script (enable the `ASYM_UNSLOTH_GC_NVME` export from Stage 2). Substrate-B seals are Stage 4 — but ALL manager hooks land now (inert without seals).

### 3.1 The governor

Handle state machine (all transitions under one small lock; in sync mode everything is main-thread so the lock is uncontended):

```text
                    on_offload            on_seal(ev)          pressure(sync write)
   (untracked) ───────────────▶ TRACKED ───────────▶ QUEUED ───────────────────────▶ DURABLE(buffer freed)
                                    │                   │                                   │ ensure_local
                                    │                   │ ensure_local                      ▼ (sync pread)
                                    └── on_release      └────────▶ CLAIMED             FETCHED ── on_release
   Async mode (Stage 5) adds QUEUED→SUBMITTED (write in flight; buffer still valid — writer only READS it;
   SUBMITTED→CLAIMED on consume: on_durable sees CLAIMED → drops the blob, counts wasted_write).
```

```python
# asym_gemm/training/act_spill_governor.py
TRACKED, QUEUED, SUBMITTED, DURABLE, CLAIMED, FETCHED = range(6)
_SENTINEL = torch.empty(0, dtype=torch.uint8)        # data_ptr()==0 ⇒ stock release paths no-op gracefully

@dataclass
class _Rec:
    handle: Any; manager: Any; nbytes: int           # nbytes CACHED (handle.nbytes is live over .tensor)
    substrate: str                                    # "boundary" | "fg"  (stats only; NO policy on it)
    state: int = TRACKED
    seal_event: Any = None
    ref: Any = None                                   # BlobRef once written
    order: int = 0                                    # creation index; FIFO queue position

class ActSpillGovernor:
    """Global singleton (role 'activation'). FIFO spill under CPU pressure; LIFO consumption
    with (Stage 5) byte-budgeted reverse prefetch. All policy is oldest-first + watermark."""

    def __init__(self, store):
        self._store = store
        self._lock = threading.Lock()
        self._by_id: dict[int, _Rec] = {}             # id(handle) → rec; handle alive on ctx until release
        self._queue: deque[_Rec] = deque()            # QUEUED only, creation order (sealed ⇒ spillable)
        self._spilled: list[_Rec] = []                # SUBMITTED/DURABLE in spill order (prefix of queue)
        self._order = itertools.count()
        self.live_cpu_bytes = 0                       # ALL tracked bytes (incl. never-sealed transients)
        # ASYM_NVME_ACT_CPU_BUDGET_BYTES: int | "auto" (0.85×MemAvailable at init, logged) | 0 (eager spill-all)
        self.hi = _budget_from_env()
        self.lo = max(0, self.hi - _env_int("ASYM_NVME_ACT_LOW_SLACK_BYTES", 16 << 30))   # hysteresis
        self.prefetch_bytes = _env_int("ASYM_NVME_ACT_PREFETCH_BYTES", 0)                  # 0 until Stage 5
        self.stats = GovStats()   # per-substrate spilled/fetched bytes+counts, live_peak, wasted_writes, …

    # ---- producer side ----
    def on_offload(self, manager, handle, substrate="fg") -> None:
        """From ActivationOffloadManager.offload/adopt_cpu/empty_cpu (+ boundary module). Pressure
        accounting for EVERY handle; nothing is spillable until sealed."""
        with self._lock:
            rec = _Rec(handle, manager, handle.nbytes, substrate, order=next(self._order))
            self._by_id[id(handle)] = rec
            self.live_cpu_bytes += rec.nbytes
            self.stats.note_live_peak(self.live_cpu_bytes)

    def on_seal(self, manager, handles, event=None) -> None:
        """ONE call at the END of Function.forward (Substrate B) or per boundary offload (Substrate A).
        The event orders after (a) the D2H fill and (b) every already-enqueued forward consumer —
        all on the same stream. Handles failing io_ready(tensor, store.align) or < min_swappable_bytes
        are left TRACKED (pressure-only)."""
        ev = event or _record_event()
        with self._lock:
            for h in handles:
                rec = self._by_id.get(id(h))
                if rec and rec.state is TRACKED and rec.nbytes >= self._store.cfg.min_swappable_bytes \
                       and io_ready(h.tensor, self._store.align):
                    rec.seal_event = ev; rec.state = QUEUED; self._queue.append(rec)
        self._maybe_spill()

    def _maybe_spill(self) -> None:
        """Oldest-first until live ≤ lo. SYNC mode: inline blocking write on the caller (main) thread.
        The oldest sealed handles' events are typically long complete ⇒ synchronize() is ~free."""
        while True:
            with self._lock:
                if self.live_cpu_bytes <= self.hi or not self._queue: return
                rec = self._queue.popleft()
                if rec.state is not QUEUED: continue          # CLAIMED/released while queued
                rec.state = SUBMITTED
            rec.seal_event.synchronize()                       # D2H + forward consumers done ⇒ bytes stable
            if self._store.cfg.sync:
                rec.ref = self._store.spill_sync("activation", rec.handle.tensor)
                self._finish_spill(rec)                        # main thread; see below
            else:                                              # Stage 5
                rec.ref = self._store.spill_async("activation", rec.handle.tensor,
                                                  ready_event=None,  # already synced above
                                                  on_done=self._make_on_durable(rec))
            with self._lock:
                self._spilled.append(rec)
                if self.live_cpu_bytes <= self.lo: return

    def _finish_spill(self, rec) -> None:                      # sync mode / writer callback body
        with self._lock:
            if rec.state is CLAIMED:                           # async-only race: consumed while in flight
                self._store.blob_done(rec.ref); self.stats.wasted_writes += 1; return
            rec.state = DURABLE
            self.live_cpu_bytes -= rec.nbytes
        rec.manager._pop_active(rec.handle)                    # accounting + pending-event entry off old ptr
        _return_cpu(rec.handle.tensor, pin_memory=True)        # pool reuse safe: seal event synced
        object.__setattr__(rec.handle, "tensor", _SENTINEL)    # stray compute read ⇒ loud shape error

    # ---- consumer side (MAIN THREAD) ----
    def ensure_local(self, handle) -> None:
        """First line of ActivationOffloadManager.wait_cpu_ready() (which stage*() already call),
        plus the one direct-read site (attention bwd :715). O(1) dict miss for untracked handles."""
        rec = self._by_id.get(id(handle))
        if rec is None: return
        with self._lock:
            if rec.state in (TRACKED, QUEUED, SUBMITTED):      # CPU-valid (writer only READS the buffer)
                rec.state = CLAIMED; return                    # dequeue is lazy (state check in _maybe_spill)
            if rec.state in (CLAIMED, FETCHED): return
            assert rec.state is DURABLE
        bounce = _alloc_cpu(handle.original_shape, handle.original_dtype, pin_memory=True)  # padded pool buffer
        arrived = self._store.fetch_into(rec.ref, bounce)      # drains in-flight prefetches too (Stage 5)
        self._store.blob_done(rec.ref)
        object.__setattr__(handle, "tensor", bounce)
        rec.manager._mark_cpu_live(handle)                     # re-enter accounting under the new data_ptr
        with self._lock:
            rec.state = FETCHED; self.live_cpu_bytes += rec.nbytes
        self._settle_prefetches(arrived)                       # Stage 5: swap tensors of arrived preads
        self._prefetch_reverse(rec)                            # Stage 5: no-op while prefetch_bytes == 0
        self._maybe_spill()                                    # fetch may push live over hi (spills oldest
                                                               # = deepest-remaining layer = still Belady)

    def on_release(self, handle) -> None:
        """FIRST line of release_cpu() — MUST run before its internal wait_cpu_ready (:318), else a
        spilled-never-consumed handle would be fetched just to be freed. Covers every terminal state."""
        rec = self._by_id.pop(id(handle), None)
        if rec is None: return
        with self._lock:
            if rec.state in (TRACKED, QUEUED, CLAIMED, FETCHED):
                self.live_cpu_bytes -= rec.nbytes              # DURABLE bytes already decremented at spill
            if rec.state is SUBMITTED: rec.state = CLAIMED     # async: on_durable will drop the blob
            elif rec.state is DURABLE: self._store.blob_done(rec.ref)   # spilled, never fetched
            elif rec.state is QUEUED:  rec.state = CLAIMED     # lazy-dequeue marker
        # stock release_cpu then runs: sentinel tensors no-op (ptr 0 not in accounting), fetched/resident
        # tensors pool-return normally.
```

Why the event algebra is safe: the seal event is recorded after every D2H fill and every forward kernel launch that streams these buffers — all earlier on the same stream. The spiller synchronizes it before pwrite (stable bytes) and pool-return happens only after that same sync (no kernel can still be streaming a recycled buffer). Backward consumers always pass through `ensure_local` first. In async mode `SUBMITTED→CLAIMED` is safe because the writer only *reads* the buffer; the wasted blob is dropped unread. The `data_ptr`-keyed accounting hazard is avoided because `_pop_active` runs before pool-return and the handle's tensor is swapped to the sentinel.

### 3.2 Manager integration (`activation_offload.py`, ~30 lines — lands ALL hooks now)

```python
# module top: _GOV cached via get_act_spill_governor() (None unless role "activation" — rule 7)
# offload()/adopt_cpu()/empty_cpu(): after _mark_cpu_live → `if _GOV: _GOV.on_offload(self, handle)`
# wait_cpu_ready(handle): FIRST line → `if _GOV: _GOV.ensure_local(handle)`        (covers stage*/direct reads)
# release_cpu(handle):    FIRST line → `if _GOV: _GOV.on_release(handle)`          (BEFORE :318's wait)
# NEW _pop_active(handle): pops _active_cpu_bytes[ptr] (stats decrement exactly like release_cpu :319-324,
#   WITHOUT pool return) AND pops _pending_cpu_ready_events[ptr]; called only by the governor.
# NEW seal(*handles): `if _GOV: _GOV.on_seal(self, [h for h in handles if h is not None])`  (engine sugar)
# _alloc_cpu: when _GOV — allocate via nvme_store.alloc_padded_pinned (pool key unchanged; pooled buffers
#   just carry padded storages); pin-failure fallback stays (io_ready() excludes those buffers).
# _return_cpu/_alloc_cpu: guard _CPU_BUFFER_POOL with a module lock ONLY when _GOV (Stage 5 writer thread
#   returns buffers off-main-thread; sync mode is main-thread-only but the lock is uncontended and simple).
```

The eligibility rule (locked): **`on_offload` tracks every handle for pressure; only handles passed to `seal()` ever spill.** Backward-created transients (`mlp.dact/dgate/dup`, `moe.*`, stage scratch) are never sealed ⇒ structurally excluded, no phase detection.

### 3.3 Substrate A wiring (`gc_boundary_offload.py` + the 4-line LF hook)

```python
# asym_gemm/training/gc_boundary_offload.py
_BOUNDARY_MANAGER: ActivationOffloadManager | None = None      # ONE process-global manager owns boundaries
def get_asym_unsloth_gc_func():
    """Drop-in replacement for LF's UnslothGradientCheckpointing.apply, used only when the
    activation store is enabled. Same math, same recompute, same save_on_cpu recompute region."""
    mgr = _get_boundary_manager()
    class AsymUnslothGCOffload(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, forward_function, hidden_states, *args):
            if hidden_states.is_cuda and not _keep_on_hbm_diagnostic():        # preserves OUTER_HBM_EVERY_N
                handle = mgr.offload(hidden_states, "gc.boundary")             # pinned pool + async D2H
                mgr.seal(handle)                                               # sole consumer is backward ⇒
                ctx.asym_handle = handle                                       # sealed immediately (the seal
            else:                                                              # event covers the D2H copy)
                ctx.asym_handle = None; ctx.hbm_hidden = hidden_states
            with torch.no_grad():
                outputs = forward_function(hidden_states, *args)
            ctx.forward_function = forward_function; ctx.args = args
            ctx.save_for_backward()                                            # tensor rides the handle, NOT
            return outputs                                                     # autograd — autograd would pin
        @staticmethod                                                          # the buffer against spilling
        @torch.cuda.amp.custom_bwd
        def backward(ctx, grad_output):
            h = ctx.asym_handle
            if h is not None:
                mgr.wait_cpu_ready(h)                                          # ensure_local: un-spill if DURABLE
                hidden = h.tensor.to("cuda", non_blocking=True)                # pinned H2D (faster than today's
                ev = torch.cuda.Event(); ev.record()                           # pageable path)
            else:
                hidden = ctx.hbm_hidden
            hidden = hidden.detach().requires_grad_(True)
            with torch.enable_grad():
                if _env_flag("UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU"):
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        outputs = ctx.forward_function(hidden, *ctx.args)
                else:
                    outputs = ctx.forward_function(hidden, *ctx.args)
                output = outputs[0] if isinstance(outputs, tuple) else outputs
            torch.autograd.backward(output, grad_output)
            if h is not None:
                ev.synchronize()                                               # H2D done ⇒ buffer reusable
                mgr.release_cpu(h)                                             # LIFO release → pool + governor
            return (None, hidden.grad) + (None,) * len(ctx.args)
    return AsymUnslothGCOffload.apply
# Boundary stats surface: no nn.Module owns this manager, so expose get_boundary_offload_stats()
# (= manager.snapshot()) and merge it into the asym_nvme block (§Stage-2c); governor summary() carries
# the substrate="boundary" split.
```

LF hook (`checkpointing.py`, first lines of `get_unsloth_gradient_checkpointing_func` `:78` — mirrors the existing `UNSLOTH_GC_*` env-hook style):

```python
    if _env_flag("ASYM_UNSLOTH_GC_NVME"):        # exported by run_lf when ASYM_NVME_ROLES contains "activation"
        from asym_gemm.training.gc_boundary_offload import get_asym_unsloth_gc_func
        return get_asym_unsloth_gc_func()
```

Correctness notes: `ctx.save_for_backward()` empty is legal (no double-backward here — the inner region uses reentrant `torch.autograd.backward`, same as today); `hidden.grad` flows exactly as today (`:117`); the boundary tensor is bf16 `[M,H]` contiguous → ONE pool shape class per model ⇒ perfect pool reuse layer-over-layer (steady-state pinned usage ≈ budget, not footprint); `_keep_on_hbm_diagnostic` preserves the `UNSLOTH_GC_OUTER_HBM_EVERY_N` counter semantics.

### 3.4 Efficiency (this stage)

Memory: boundary copies move from pageable to pooled pinned (same bytes, recycled); spilled bytes leave RAM entirely. Latency: sync spill blocks the main thread `bytes/14GBps` per spill (~0.4 s per 5.5 GiB boundary) during forward, sync fetch `bytes/26GBps` (~0.2 s/layer) during backward — the accepted v1 price, eliminated in Stage 5; because oldest-first spills layer-0-side handles whose seal events completed long ago, `seal_event.synchronize()` is ~free. Kernel launches: +1 event record per boundary (64/step) + 1 per fg Function (Stage 4) — noise. NO GEMM shape/count changes anywhere.

### Validation (Stage 3 gate — capacity mode, REAL e2e)

Unit (`tests/training/test_act_spill_governor.py`): state transitions (sync: TRACKED→QUEUED→DURABLE→FETCHED; QUEUED→CLAIMED; DURABLE released-unfetched drops blob); oldest-first order; hysteresis stops at `lo`; unsealed handles never spill under pressure; `io_ready`-failing handles never spill; sentinel compute-read raises; budget=0 eager mode spills everything and stays bit-exact; a toy 2-layer module under `get_asym_unsloth_gc_func()`: fwd+bwd **bit-identical** with `ASYM_NVME_ROLES=""` vs `"activation"`+tiny budget; existing `tests/training/test_dense_mlp_finegrained.py` passes with NVMe off AND on (hooks inert without seals).

E2E (budget forces spilling deterministically; sequential; `kill -TERM` only):

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage3_actnvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=1 MAX_STEPS=4 \
ASYM_NVME_ACT_CPU_BUDGET_BYTES=$((120*1024*1024*1024)) \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds_actnvme|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
$LF_PY scripts/lf/compare_nvme_profiles.py --baseline <base dir> --candidate <actnvme dir> \
  --target activation_cpu --min-memory-drop-gib 30 --max-step-ratio 1.5 --max-forward-ratio 1.5 \
  --max-backward-ratio 1.5 --expect-nvme-role activation --max-loss-delta 0
```

Accept: candidate per-step RSS drops ≈ (boundary footprint − budget) (s30000 boundary ≈ 147 GiB analytic; budget 120 ⇒ expect ≥30 GiB drop); `asym_nvme.bytes_written ≈ bytes_read ≈` overflow volume; `act_governor.live_peak ≈` budget; **losses identical to baseline step-for-step** (same math, same RNG — `--max-loss-delta 0`); HBM unchanged; step ratio recorded (informational at this rung). **Then the headline probe:** raise seq until the baseline dies (host-OOM watchdog) and the candidate still trains — e.g. s90000 with budget 400 GiB; verify real `input_ids` length in `train.log`; record both max-seq numbers.

### Risks / watch
- **Reentrant nesting:** the inner `torch.autograd.backward` runs Substrate-B Functions whose `wait_cpu_ready`→`ensure_local` may nest inside the outer boundary fetch path — all main-thread (no deadlock; the governor lock is NOT held around IO), but debug-assert `ensure_local` non-reentrancy per handle.
- Pool-cap churn: 5.5 GiB boundary buffers vs the 32 GiB default pool cap ⇒ set `ASYM_EXPACT_CPU_POOL_MAX_BYTES` ≥ budget-scale for dense models too (full-fg already sets 192 GiB for MoE); watch `cpu_pool_evictions`.
- If step-1 loss differs at all: rerun with budget=∞ (tracks, never spills — isolates the pinned-copy change from the spill path) before suspecting the governor.
- Governor budget vs watchdog: `auto` = 0.85×MemAvailable at init is wrong on a shared box — ALWAYS set the env explicitly for gate runs.
- Decorators: LF venv torch is **2.12.0+cu130** and the LF file's `@torch.cuda.amp.custom_fwd/_bwd` (`checkpointing.py:83,:103`) runs in production today — mirror those decorators verbatim in `AsymUnslothGCOffload` (deprecation shim is alive; do not "modernize" and diverge from the reference Function).

---

## Stage 4 — actnvme Substrate B seals: dense-fg + attention U/S (flagship-complete)

**What:** add `seal()` calls so intra-layer fine-grained handles participate; they are the newest FIFO entries ⇒ spill only under extreme pressure (terminal-margin property for free). No governor changes.

**Scope:** `dense_mlp_finegrained.py` (1 line), `attention_activation_offload.py` (~6 lines), env `ASYM_NVME_ACT_ENGINES=boundary,dense_fg,attention` (default all-on when role enabled; per-engine kill-switch for bisection).

```python
# dense_mlp_finegrained.py — immediately before `layer._last_activation_offload_stats = …` (:300):
manager.seal(x_cpu, gate_cpu, up_cpu, act_cpu, gate_low_rank_cpu, up_low_rank_cpu, down_low_rank_cpu)

# attention_activation_offload.py — in _AsymActivationOffloadLoRALinearFunction.forward, after the ctx
# stashes (:632-650), before return (:653):
manager.seal(s_handle)
if ctx.shared_source is None:
    manager.seal(u_handle)                        # non-shared U: this Function is the only consumer
elif ctx.shared_source.closed:                    # NEW bool on _SharedActivationSource: set True by
    attention_context.manager.seal(u_handle)      # acquire_source in BOTH cache-clear branches (:478-479).
                                                  # Seal at the END of the LAST acquirer's forward — the
                                                  # cache-clear itself happens BEFORE v_proj's own CPU-left
                                                  # read (:620), so sealing there would be premature.
# backward (:715), BEFORE `u_source = _pad_cpu_rows_to(u_handle.tensor, …)`:
_gov_ensure_local(u_handle)                       # the one manager-API-bypassing direct read in the tree.
# Confirmed 2026-07-04: _pad_cpu_rows_to (:64) is a HOST-side CPU pad/copy feeding
# asym_bf16_cpu_right_matmul as the CPU operand, and NO wait_cpu_ready precedes it in this backward —
# today's safety is backward-after-forward stream ordering only. The finally (:736-745) releases
# s_stage/s_handle, then u_handle (non-shared) or shared_source.release(), then _update_snapshot.
```

Audit note (verified §0.4): every other backward read in both engines passes through `stage*()`/`wait_cpu_ready` ⇒ covered by the Stage-3 hook. The dense-fg backward's mid-body releases (`:333,:357-358,:373,:381,:395,:410,:419,:436`) plus the `finally` (`:440-453`) give exact release-on-last-use; `on_release` is idempotent against the double calls.

**Efficiency:** at sane budgets these handles never spill (newest); when they do (budget ≪ window), the fg backward consumes them within the same layer window — sync fetch cost bounded by window size; per-layer `x_cpu` (created first, consumed last `:396,:420`) is the only fg handle with real lead time — and exactly the one oldest-first picks first. No new launches beyond one seal event per Function.

### Validation (Stage 4 gate)

Unit: extend governor tests with a real `_FinegrainedDenseMLPFunction` fwd+bwd (toy shapes, CUDA) bit-identical off vs on+tiny-budget; attention tests (`test_attention_activation_offload_lora.py`, `_helpers.py`) pass off AND on; NEW test: q/k/v share + spill shared U after the v_proj seal, verify all three projections' backwards fetch once and `source_share_released_bytes` is unchanged.

E2E: rerun the exact Stage-3 command pair with `ASYM_NVME_ACT_CPU_BUDGET_BYTES=$((60*1024*1024*1024))` (deep pressure — forces fg spills at s30000): accept losses identical, `act_governor` shows `substrate=fg` spilled bytes > 0, `cpu_peak_by_tag` for `mlp.*`/`*.U`/`*.S` collapses vs baseline, `wasted_writes==0` (sync mode), step ratio recorded. Then one MoE smoke `q3.5-35b-a3b|1 ; … ; 45000|8|1` with the MoE engine NOT yet sealed — proves partial coverage degrades to status quo (moe.* stays resident), never corrupts.

### Risks / watch
- Shared-U `closed` flag: `acquire_source` has two clear branches (`role=="v_proj"` OR all roles seen, `:478-479`) — set `closed` in both; debug-assert a spilled shared U was sealed only after close.
- `_update_snapshot` (`:501-511`): governor/nvme counters ride the per-call manager snapshot; extend `AttentionActivationOffloadContext.snapshot()` (`:485-498`) so shared-U spill counters surface under `source_context`.
- If profiles show U/S never spill even at tiny budgets that is CORRECT (newest-first-resident) — check `substrate=fg` counters before hunting ghosts.

---

## Stage 5 — actnvme v2/v3 ASYNC: writer thread + reverse-order prefetch (+ optional H2D stream)

**What:** flip `ASYM_NVME_SYNC=0`: spills go to the writer thread (forward never blocks on NVMe; backpressure at `max_inflight_spill_bytes`); backward hides fetch latency with byte-budgeted **reverse-creation-order** prefetch (`ASYM_NVME_ACT_PREFETCH_BYTES`, e.g. 2–3 boundary blobs ≈ 12–16 GiB) — the spilled list walked newest→oldest matches consumption (layer k+1's set before layer k's; intra-layer approximately — a mis-ordered prefetch is an efficiency blip, never a bug, because `drain_reads` + durable events serialize correctness).

**Scope:** `nvme_store.py` `_WriterThread` (Stage-1 code path, now exercised), governor: `_make_on_durable` (= `_finish_spill` body running on the writer thread — CUDA-free except `ready_event.synchronize()`, which moves there to unblock the main thread fully: pass `ready_event=rec.seal_event`, don't pre-sync), `_prefetch_reverse` + `_settle_prefetches`:

```python
def _prefetch_reverse(self, just_fetched) -> None:
    if self.prefetch_bytes <= 0: return
    inflight = self._inflight_prefetch_bytes()
    for r in self._iter_spilled_newer_first(from_rec=just_fetched):   # reverse creation order
        if inflight >= self.prefetch_bytes: break
        if r.state is DURABLE and r.prefetch_buf is None:
            r.prefetch_buf = _alloc_cpu(r.handle.original_shape, r.handle.original_dtype, pin_memory=True)
            self._store.submit_pread(r.ref, r.prefetch_buf); inflight += r.nbytes

def _settle_prefetches(self, arrived_ref_ids) -> None:               # main thread, after any drain
    for rec in self._take_arrived(arrived_ref_ids):
        self._store.blob_done(rec.ref)
        object.__setattr__(rec.handle, "tensor", rec.prefetch_buf); rec.prefetch_buf = None
        rec.manager._mark_cpu_live(rec.handle)
        with self._lock: rec.state = FETCHED; self.live_cpu_bytes += rec.nbytes
```

Async-only correctness pieces (already in the state machine): `SUBMITTED→CLAIMED` consume-while-in-flight (writer's callback sees CLAIMED → drops blob, buffer stays handle-owned, `wasted_writes+=1`); pool lock active (writer returns buffers); `ref.durable.wait()` in `submit_pread` covers prefetch-of-in-flight-write. Optional v4 rung (`ASYM_NVME_ACT_H2D_STREAM=1`, only if profiles show H2D on the critical path): boundary H2D on a side stream + event — TE-v1 pattern (`cpu_offload_v1.py:366-367,:578`); default OFF.

**Efficiency:** forward's only NVMe cost becomes the CV-wait when >8 GiB of writes are in flight (RAID write bw 14 GB/s drains 5.5 GiB boundaries faster than long-seq layers produce them); backward fetch wait → ≈0 when prefetch ≥ per-layer consumption × fetch latency; prefetch allocates INTO the live budget (assert `prefetch_bytes < hi − lo` at init so prefetch cannot re-trigger spill thrash).

### Validation (Stage 5 gate — throughput at the same operating point)

```bash
# same paired RUNS as Stage 3 (s30000, budget 120 GiB) but with:
ASYM_NVME_SYNC=0 ASYM_NVME_ACT_PREFETCH_BYTES=$((12*1024*1024*1024))
$LF_PY scripts/lf/compare_nvme_profiles.py --baseline <asym_cpuadamwds dir> --candidate <actnvme async dir> \
  --target activation_cpu --min-memory-drop-gib 30 --max-step-ratio 1.10 --max-forward-ratio 1.10 \
  --max-backward-ratio 1.10 --expect-nvme-role activation --max-loss-delta 0
# plus candidate-vs-candidate: async step ≤ sync step − 0.8×(spill_wait_ms+fetch_wait_ms)/step
```

Accept: memory drop as Stage 3; step ≤1.10× baseline; `spill_backpressure_ms ≈ 0` at this point; `fetch_wait_ms` ↓ ≥5× vs the sync run; `wasted_writes ≈ 0`; losses identical to baseline AND to the sync candidate (same math, bit-for-bit). Cross-check device IO: `offload_io.json` (from `io_samples.csv`, O_DIRECT ⇒ page-cache-free) totals ≈ `asym_nvme` byte counters ±10%. Then re-run the max-seq probe with async on — step-ratio is now the meaningful capacity price.

### Risks / watch
- If `spill_backpressure_ms` dominates at the capacity point: overflow rate exceeds ~14 GB/s — raise budget or accept/report the stall (capacity price). The Stage-0 census predicts this — check it before blaming code.
- Prefetch-order vs consumption-order mismatch intra-layer (x before S_up, `:396` vs `:415`) — bounded by one layer's bytes; visible as small residual `fetch_wait_ms`; do NOT special-case tags for it.
- Writer-thread discipline: `ready_event.synchronize()` is host-side (allowed); everything else CUDA-free — assert no `torch.cuda` allocs in the writer path under `ASYM_NVME_DEBUG=1`.

---

## Stage 6 — actnvme Substrate B coverage: qwen3/qwen3.5 MoE fine-grained engine

**What:** one seal line + wait-audit for `_Qwen3MoeFinegrainedFunction` — required for the MoE hero models (q3.5-35b-a3b, q3.5-122b-a10b, q3-30b-a3b).

**Scope:** `qwen3_moe_finegrained.py` (1 seal line + up to 2 ensure calls if the audit finds uncovered direct reads).

```python
# immediately before `layer._last_activation_offload_stats = _record_manager_peaks(layer, manager)` (:706):
manager.seal(x_cpu, gate_cpu, up_cpu, act_cpu, gate_low_rank_cpu, up_low_rank_cpu, down_low_rank_cpu)
# Audit RESOLVED (2026-07-04): backward direct reads are wait-covered — act via :777/:902; for x, the
# single `manager.wait_cpu_ready(ctx.x_cpu)` at the HEAD of the `down_scatter_block_experts > 0` branch
# (:968) dominates every `ctx.x_cpu.tensor[row_slice]` read inside the block loop (~:1021+) — one hook
# invocation un-spills x for the whole loop; no per-block ensure calls, no extra work.
# Transients moe.dup/dgate (:945/:962) are never sealed; full-fg sets KEEP_DGRADS_HBM=1 anyway.
```

Legacy engines (`qwen3_moe.py` FnA `:1010`, `llama4_experts.py` `:229`, shared MLPs `:236`) get seals the same one-line way ONLY if a target config re-activates them (`ASYMM_EXPERT_ACT_OFFLOAD`-gated, off in full-fg) — deferred, not dead. Their release loops lack `finally` (`qwen3_moe.py:1470-1482`, `llama4_experts.py:724-736`): an exception mid-backward leaks governor recs (pre-existing leak shape; training aborts anyway — noted, not fixed here).

### Validation (Stage 6 gate)

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage6_moe PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=1 MAX_STEPS=4 ASYM_NVME_ACT_CPU_BUDGET_BYTES=$((150*1024*1024*1024)) \
RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 45000|8|1 ; none|false|true|false|false|false || q3.5-35b-a3b|1 ; asym_cpuadamwds_actnvme|recomp-off-full-fg-ker101|ligerloss1 ; 45000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
# compare: --target activation_cpu --min-memory-drop-gib 30 --max-step-ratio 1.10 --max-loss-delta 0
```

Accept: identical losses; `moe.*` in `cpu_peak_by_tag` collapses under deep pressure (repeat with budget 60 GiB); step ≤1.10×; `expert_token_distribution` unchanged (routing untouched). Verify the tuple spelling (ker101 etc.) against the existing `profiling_q35_final2_asym80k_*` run dir before launching.

Risks/watch: `_record_manager_peaks` (`:96-112`) decorates the snapshot — confirm nvme keys survive its copy; MoE pool cap (192 GiB) and governor budget are SEPARATE knobs (pool caches FREE buffers; governor counts LIVE handles) — both enter the Stage-8 RAM ledger.

---

## Stage 7 — `panvme`: base weights → NVMe

**Scope:** `host_weight.py` (property surgery), NEW `asym_gemm/training/base_weight_pager.py`, `integrations/lf.py` (registration walk immediately before `return model, report` at `:2426`), `qwen3_moe.py` (eager fine-grained split call).

```python
# host_weight.py — surgery (everything else byte-identical; ~15 lines)
class HostWeight:
    # NEW instance fields, default None, set only by pager.register(): _pager, _pager_key
    @property
    def weight(self):
        if self._pager is None: return self._tensor          # today's exact path (one None check)
        return self._pager.touch(self._pager_key)
    tensor = weight                                           # keep both properties in lockstep
    # shape (:272) / dtype (:276) / device (:280) / out_features (:292) / in_features (:296):
    #   `if self._pager is not None: return <from self._metadata>` — reporting/predicates never fetch.
    # nbytes (:283) / metadata (:266): already metadata-backed — unchanged.
    # grad (:300): return None when paged (frozen weights — assert no grad ever set).
    # is_pinned: True when paged (pager buffers are pinned).
    # grouped_nt_tensor (:315): route through touch() then today's logic (compute path — fetch is correct).
    # pin_memory (:340) / to() / cuda(): raise when paged (registration is post-placement).
```

```python
# base_weight_pager.py — trace-prefetching pinned cache (DeepSpeed-coordinator pattern; MAIN THREAD ONLY)
ABSENT, INFLIGHT, RESIDENT = range(3)

class BaseWeightPager:
    def __init__(self, store, *, cache_bytes=_env_int("ASYM_NVME_BASE_WEIGHT_CACHE_BYTES", 16 << 30),
                 prefetch_bytes=_env_int("ASYM_NVME_BASE_WEIGHT_PREFETCH_BYTES", 0)):   # 0 → auto: 2×largest blob
        self._entries: dict[str, _Entry] = {}; self._by_ref_id = {}
        self._free: dict[tuple, list] = {}            # (dtype, shape) → free padded pinned bufs
        self._quarantine: list[tuple] = []            # (buf, cuda_event, class) — event-gated reuse (rule 6)
        self._trace: list[str] = []; self._trace_build = []; self._frozen = False; self._disabled = False
        self._cursor = -1; self._last_key = None
        self.misses = self.misses_after_freeze = 0

    def register(self, key, hw):
        t = hw._tensor
        if t is None or t.numel() * t.element_size() < store.cfg.min_swappable_bytes: return
        padded = alloc_padded_pinned(tuple(t.shape), t.dtype, align=store.align); padded.copy_(t)
        ref = store.spill_sync("base_weight", padded)              # written once at startup; sync is right
        self._entries[key] = _Entry(hw=hw, ref=ref, shape=tuple(t.shape), dtype=t.dtype,
                                    padded_nbytes=padded.untyped_storage().nbytes(),
                                    buf=None, view=None, state=ABSENT, positions=[])
        self._by_ref_id[id(ref)] = self._entries[key]
        hw._pager, hw._pager_key = self, key
        hw._tensor = None                                          # the GB-scale home is freed NOW

    def touch(self, key):
        e = self._entries[key]
        if key != self._last_key:                    # dedupe: .weight is read several times per Function
            self._last_key = key
            self._record_or_advance(e); self._issue_prefetches()   # trace-driven lookahead
        if e.state is RESIDENT: return e.view
        if e.state is INFLIGHT:
            for rid in store.drain_reads(): self._by_ref_id[rid].state = RESIDENT
            return e.view
        e.buf = self._take_buffer(e); e.view = e.buf               # ABSENT miss (step 1; ~never after freeze)
        store.fetch_into(e.ref, e.buf); e.state = RESIDENT
        self.misses += 1; self.misses_after_freeze += int(self._frozen)
        return e.view

    # _record_or_advance: build the touch trace during step 1 (fwd + bwd); freeze when the FIRST key recurs
    #   the 3rd time (fwd@0, its bwd, fwd again = start of period 2); then cursor-advance with a jitter
    #   window of 8; any mismatch → _disabled = True (miss-driven sync fallback, counted, loud in summary).
    # _issue_prefetches: byte-budgeted lookahead over the frozen trace (uniform lead TIME under mixed blob
    #   sizes — MoE 3D groups vs dense 2D); guarded by _would_evict_nearer_than (tight-cache, DS max_live analog).
    # _take_buffer: free-list by (dtype, shape) → else grow while resident+padded ≤ cache_bytes → else evict
    #   farthest-next-use on the frozen trace (exact Belady); evictee's buf → _quarantine with a post-launch
    #   CUDA event; _sweep_quarantine returns event-complete buffers to the free list.
```

Registration walk (end of `apply_lf_asym_lora`, `integrations/lf.py`, before `:2426`) + the eager split:

```python
store = get_nvme_store()
if store is not None and store.has_role("base_weight"):
    if _qwen3_moe_finegrained_offload_enabled():
        for m in model.modules():                                  # BEFORE spilling: force the lazy
            if is_qwen3_experts(m): m._ensure_qwen3_moe_finegrained_bases()   # fused→gate/up split (:2509)
    pager = BaseWeightPager(store)
    for name, mod in model.named_modules():
        hw = getattr(mod, "host_weight", None)
        # eligible: AsymFrozenLinear (attention + mlp_dense) + AsymGroupedFrozenLinear (experts, shared);
        # EXCLUDED: embed_tokens (CPU-side F.embedding per microbatch, offload.py:381), norms (tiny,
        # unpinned by policy), anything with precision != "bf16" (quantized cache builds from .weight).
        if isinstance(hw, HostWeight) and _panvme_component_eligible(name, mod):
            assert getattr(mod, "precision", "bf16") == "bf16"
            pager.register(name, hw)
    model._asym_base_weight_pager = pager
```

Correctness anchors: fwd+bwd interleaved trace ⇒ farthest-next-use is exact Belady (late layers, reused first in backward, survive after forward); event-gated quarantine per rule 6 (`_asym_bf16_nt` launches return immediately while streaming the pinned weight, `frozen_linear.py:726-728,:807-809`); `touch` dedupe absorbs the multi-read pattern (`is_pinned` predicate + kernel read within one Function, `qwen3_moe.py:1775` etc.); reporting reads metadata (never fetches); step-1 is miss-driven by design (⇒ `WARMUP_STEPS≥1` mandatory); grouped 3D expert blobs are GB-scale — assert `cache_bytes ≥ 2×largest_padded_nbytes` at registration.

**Efficiency:** whole-blob IO (a 2 GiB expert bank is ONE pread, internally 4×16-parallel); grouped weights stay grouped — no per-expert loops introduced anywhere; prefetch keeps the compute stream fed with zero HBM change; `touch` fast path = one dict get + one identity compare.

### Validation (Stage 7 gate)

Unit (`tests/training/test_base_weight_pager.py`): register frees `_tensor` (RSS probe); roundtrip bit-exact (2D + grouped 3D bf16); freeze at 3rd first-key occurrence; Belady eviction picks farthest; quarantine blocks reuse until event completes; jitter tolerance; disable-fallback correctness; metadata properties never fetch (counter assert); NVMe-off ⇒ byte-identical (existing `test_cpu_resident_frozen_base.py` + `test_lf_qwen3_asym_backend.py` unmodified must pass).

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage7_panvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds_panvme|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
$LF_PY scripts/lf/compare_nvme_profiles.py --baseline <base> --candidate <panvme> --target base_weight_cpu \
  --min-memory-drop-gib 40 --max-step-ratio 1.05 --max-forward-ratio 1.05 --max-backward-ratio 1.05 \
  --expect-nvme-role base_weight --max-loss-delta 0
```

Accept: per-step RSS −40 GiB+ (q3-32b host_weight = 61 GiB; `memory_attribution` host_weight/cpu rows shrink to match); HBM unchanged; ≤5% step; `misses_after_freeze == 0`; `trace_disabled == false`; `bytes_read ≈ 2×base×steps`; losses identical. Then one MoE pass (q3.5-35b-a3b, 105 GiB base, moefg on) — exercises the eager-split path.

Risks/watch: panvme+moefg hard-errors at registration until the eager-split path is validated; if ≤5% fails at short seq (weight reads / step_seconds too big), rerun at s45000+ and reclassify panvme as capacity-mode-only for short seq; any future `.weight` reader must pick compute (fetch) vs reporting (metadata) — grep on rebase. (`HostWeightMetadata` field coverage RESOLVED — see §0.5: all surgery fields present; `device` needs the str→`torch.device` rewrap.)

---

## Stage 8 — `bothnvme`: compose + hero max-seq table

No new mechanism: both roles on one store (governor + pager share it; separate files/arena). Startup RAM ledger assert (loud, with numbers): `pager cache + governor budget + pool caps + prefetch budgets + inflight caps + watchdog floor (50 GB) + slack < MemTotal (1325 GiB)`.

Gate: on q3-32b, then llama3.3-70b, then q3.5-35b-a3b — demonstrate max trainable seq `bothnvme > actnvme ≥ baseline` at fixed b8 (baseline bound = host-OOM watchdog kill; verify real `input_ids` lengths in `train.log`); report HBM peak, per-step RSS, per-role NVMe bytes + wait-ms, step time, overlap fraction (`1 − nvme_wait/step`). Compare against `zero3_offload_panvme` / `superoffload_mem_panvme` ceilings for the capability table. Use the ceiling-probe protocol (`profiling_ceiling_*` naming, sequential runs, `kill -TERM` only).

---

## Deferred (explicitly out of scope now)
- Multi-GPU/per-rank stores (rank-suffixed arenas; all ranks pread shared read-only base-weight files) and a DeepSpeed-owned backend behind the same store API.
- Legacy expert engines' seals (`qwen3_moe.py` FnA, `llama4_experts.py`, shared MLPs) — same 1-line recipe when a config needs them.
- `save_on_cpu` recompute-pack spilling (per-layer-window lifetime — no capacity win; revisit only if window transients bind after Substrate A ships).
- GDS/GPU-direct (no hardware path); gradient/optimizer NVMe (LoRA-tiny, `cpu_adam.py` guard); saved-tensor wrapper configs (`layeract`/`layergc`) — superseded by Substrate A for the flagship.

## Global run rules (every e2e gate)
Heavy runs **sequentially** (600–800 GiB RSS observed; 1325 GiB box, no swap); stop with `kill -TERM`, never `-9` (corrupts the DeepSpeed JIT cache); `PREPARE_DATASETS=true` on first use of a workload and **verify real `input_ids` length in `train.log`**; measure from `step_samples.csv` measured rows; set `ASYM_NVME_ACT_CPU_BUDGET_BYTES` explicitly for gates (never `auto` on a shared box); `.aioenv` env is exported by `run_lf_lora_sft.sh` — export manually (§0.1) for direct pytest use; the host-mem watchdog (floor 50 GB) is the baseline's OOM referee — leave it ON for max-seq probes.

## Implementation order = stage order

```text
Stage 0: traffic census (postprocess-only)        headline already verified §0.2; pins budgets — REVIEW FIRST
Stage 1: nvme_store.py substrate                  isolated; unit + AIO smoke gate
Stage 2: tokens + env + counters + compare        paired e2e no-change gate (q3-32b s30000 flagship policy)
Stage 3: governor + Substrate A (boundary), SYNC  capacity gate (−GiB @ budget, loss-identical) + max-seq probe   ← core
Stage 4: Substrate B seals (dense-fg + attention) flagship-complete correctness under deep pressure
Stage 5: async writer + reverse prefetch          throughput gate ≤1.10 (then re-probe max seq)
Stage 6: MoE fg engine seal                       q3.5-35b-a3b gate
Stage 7: panvme pager                             ≤1.05 gate, RSS −40 GiB+ (q3-32b) / −100 GiB (q3.5)
Stage 8: bothnvme hero                            max-seq capability table vs DS NVMe baselines
```

Why this order: Substrate A is the measured capacity binder (§0.2) and the smallest tensor-moving diff (one GC function + governor); sync-first makes every later rung bisectable against a bit-exact reference; Substrate B lands before async so the async gate covers the full flagship engine set; panvme after actnvme because its 61–105 GiB is additive but only *required* for the Stage-8 compound hero.

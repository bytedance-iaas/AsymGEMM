# AsymGEMM NVMe Offload — Implementation Plan (v4, code-verified 2026-07-02)

Three composable, opt-in backend tokens on ONE local NVMe store reusing DeepSpeed's AIO engine:

- `asym_cpuadamwds_actnvme` — **the recompute-path (`recomp-off-*-fg`) intra-layer saved tensors → NVMe, spilled only under CPU pressure.** *Novel*; raises the CPU ceiling that binds max seq. Capacity/hero mode. **Implemented first (core target).**
- `asym_cpuadamwds_panvme` — base/frozen weights (`HostWeight` CPU homes) → NVMe, bounded pinned cache + trace prefetch. DeepSpeed-parity idea; frees ~60 GiB CPU on q3-32b.
- `asym_cpuadamwds_bothnvme` — both roles; compound for the max-seq hero result.

**v4 changes (2026-07-02) — the old activation design is REMOVED, do not resurrect it:**
1. ~~Substrate-A stage (per-layer module-wise `saved_tensors_hooks` wrapper spill)~~ — **dropped.** actnvme targets ONLY the fine-grained recompute path (`ActivationOffloadManager`-based Functions: dense-MLP fg first, then attention/MoE). The layer-GC wrapper configs are out of scope.
2. ~~Eager spill-everything + per-tag spill policy + hot-tail-window flag~~ — **dropped, all three.** Replaced by ONE mechanism: a **watermark spill governor** — activations stay in CPU RAM; spilling starts only when live activation bytes cross a CPU budget (OOM-avoidance), oldest-first. The hot window is subsumed (the entire CPU budget *is* the hot window); no tag/list selection of any kind — **zero forward-semantics changes**.
3. NVMe traffic per step = **only the overflow beyond the CPU budget**, not the full activation footprint (e.g. footprint 800 GiB, budget 600 GiB → ~200 GiB written+read per step, not 800).

Stages in execution order: **0 traffic census (measurement-only, FIRST — no implementation until its artifacts are reviewed)** → 1 substrate → 2 wiring/gates → 3 actnvme (fg recompute path + governor) → 4 panvme → 5 actnvme coverage (attention/MoE engines) → 6 bothnvme.

Dropped long ago (re-verified): optimizer-state / LoRA-weight-home NVMe — `AsymCPUAdamW` state is LoRA-only (`cpu_adam.py:126-129,194,212-218`); LoRA homes are small pinned slabs. Nothing to save.

---

## 0. Verified ground truth (read before implementing)

### 0.1 DeepSpeed AIO — proven working on this box

```bash
# .aioenv sidecar REQUIRED for JIT build + runtime (run_lf_lora_sft.sh:29-40 already exports it).
export AIO_HOME=$PWD/.aioenv
export CPATH="$AIO_HOME/include:${CPATH:-}" LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
```

Empirically confirmed (2026-07-01): `AsyncIOBuilder().is_compatible()==True` **only** with `.aioenv`; builds ~23 s; `aio_handle(1048576, 16, False, True, 4)` → `get_alignment()==2048`; pinned pwrite/pread roundtrip **and offset-based roundtrip into one file both pass**. Same knobs as the repo's `zero3_offload_panvme` baseline (block 1MB / qd 16 / threads 4).

Hard API facts (`csrc/aio/py_lib/`):
- ctor `aio_handle(block_size, queue_depth, single_submit, overlap_events, intra_op_parallelism)` (`py_ds_aio.cpp:23`); `async_pread/async_pwrite(buffer, filename|fd, file_offset=0)` (`:87-112`); `wait()` releases the GIL (`:125-128`).
- `wait()` drains **ALL** pending ops on that handle and `assert(_num_pending_ops > 0)` — **calling wait() idle aborts the process** (`deepspeed_py_io_handle.cpp:201-220`). Python pending ledgers; **separate read and write handles**.
- `num_bytes % intra_op_parallelism == 0` required (`:222-233`); files are O_DIRECT (`deepspeed_aio_common.cpp:269`) ⇒ **pad every offset + length to `get_alignment()`**, and IO buffers must be *allocated* padded.
- Unpinned/CUDA buffers silently bounce through a managed pinned buffer (correct, one extra copy — `deepspeed_cpu_op.cpp:25-31,86-105`); keep the common path pinned.
- Each single op is internally split across `intra_op` threads × qd-deep libaio rings (`deepspeed_cpu_op.cpp:68-84`) → up to threads×qd×block in flight per op.
- NOT reusable: `AsyncPartitionedParameterSwapper` / `offload_param.device=nvme` (need `ds_id`/`ds_tensor`/comm; built only in `zero.Init` — `partitioned_param_swapper.py:76-161`, `partition_parameters.py:1081-1095`); `SwapBufferManager` (`dist.get_rank()`, `utils.py:195`); `GDSBuilder`.

### 0.2 actnvme target — the fine-grained recompute path (`ActivationOffloadManager`)

The target config class is `asym_cpuadamwds|recomp-off-*-fg|ligerloss1` (recompute+offload hybrid; env flags `ASYMM_DENSE_MLP_FINEGRAINED_OFFLOAD` / `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD`, `run_lf_lora_sft.sh:101-104`). Its saved tensors flow through **`ActivationOffloadManager`** (`asym_gemm/training/activation_offload.py`) — one manager per autograd-Function call, ONE global pinned pool `_CPU_BUFFER_POOL` keyed `(dtype, shape, pinned)` (`:10,74-103`; cap `ASYM_EXPACT_CPU_POOL_MAX_BYTES`).

Handle lifecycle in the primary engine, `_FinegrainedDenseMLPFunction` (`asym_gemm/training/dense_mlp_finegrained.py:183-465`) — verified line-by-line:
- **Cross-phase handles (created in forward, consumed in backward) — the NVMe targets:** `x_cpu` (`:216`), `gate_cpu`/`S_gate` (`:230-231`), `up_cpu`/`S_up` (`:246-247`), `act_cpu` (`:259`, config-dependent), `S_down` (`:268`); all stashed on ctx at `:291-297`; all released in the backward `finally` (`:441-453`).
- **Forward-consumed:** `x_cpu` is read *during forward* by the gate/up CPU-operand GEMMs (`wait_cpu_ready(x_cpu)` `:226,:242`); gate/up may be re-staged in forward for the act path (`:255-261`). ⇒ spill eligibility begins only at the **seal point** = end of `Function.forward` (anchor: just before `layer._last_activation_offload_stats = manager.snapshot()` at `:300`), where one CUDA event orders after every forward consumer.
- **Backward consumers:** via `manager.stage*(...)` (H2D staging) **and via `wait_cpu_ready(handle)` + direct `handle.tensor` reads by CPU-operand kernels** (`:334,:396,:420`) ⇒ the un-spill hook must cover `wait_cpu_ready` too, not just `stage*()`.
- **Backward-transients (never spill):** backward itself offloads `dact/dgate/dup` staging tensors (`:353,:366,:379`) and fills CPU-silu outputs via `empty_cpu` (`:151-176`) — produced and consumed within the same layer's backward. They are never sealed (seals happen only in forward) ⇒ structurally excluded from spilling; they still count toward CPU pressure.
- **Recompute is real** in this mode: e.g. the CPU silu-bwd path recomputes activation derivatives from saved gate/up instead of saving `act` (`:151-176,363-381`). Recomputed quantities never touch CPU homes or NVMe.
- `CPUActivationHandle` is `@dataclass(frozen=True)` (`activation_offload.py:106-116`); manager accounting keyed by `handle.tensor.data_ptr()` (`:164-166`); handles never used as dict keys → governor side-table by `id(handle)` is safe; the one sanctioned mutation point swaps `tensor` via `object.__setattr__`.
- Handles are consumed by **async GPU kernels reading CPU memory** (CPU-left/right AsymGEMM) ⇒ a buffer may be pool-returned only after an event recorded post-enqueue completes. Today's safety is single-stream ordering; the IO thread breaks it → the seal event restores it.
- Attention under this config uses its own Substrate-B path (`AsymActivationOffloadLoRALinear`, U/S handles, shared q/k/v source refcounted via `_SharedActivationSource`, released at the v_proj cache-clear point `attention_activation_offload.py:477-479`) — covered in Stage 5. The per-layer `saved_tensors_hooks` wrappers (decoder/attention/linear-attention) are **NOT targeted** in v4.
- **No copy stream / no prefetch exists anywhere in the activation path today** (only `weight_offload.py:95` holds an unused stream). All overlap infra is new.

### 0.3 panvme target — base weights

- Single tensor-level chokepoint: `HostWeight.__init__` (`host_weight.py:185-242`, pin `:222`). `.weight`/`.tensor` are **properties over `self._tensor`** (`:244-246,302-304`) → clean lazy-materialize interception; `metadata`/`nbytes` (`:283-284`) are copy-free (reporting must never fetch).
- Adoption entries: `adopt_host_weight()` (`offload.py:176-213`, called from `lf.py:1192`) and `AsymGroupedFrozenLinear.__init__` (`frozen_linear.py:1624`). Embeds/norms excluded by policy (embed consumed by CPU-side `F.embedding` per microbatch, `offload.py:381`; norms tiny).
- Consumption: dense fwd `frozen_linear.py:1307-1318`, dense dX `:1353-1365`, grouped dX `qwen3_moe.py:462-486`, grouped fwd `frozen_linear.py:729-754`, attention base fwd `attention_activation_offload.py:599-608`, native windowed bwd `qwen3_moe.py:1878`.
- **Kernel launches are async** (`frozen_linear.py:723-726` returns immediately while the GPU streams the pinned weight) ⇒ recycle a weight buffer only after a post-enqueue CUDA event completes.
- Runtime `.weight` readers besides GEMMs: `is_pinned()` predicates (`qwen3_moe.py:1760,2648-2654`, `llama4_moe.py:242`) — all immediately precede a consuming kernel (lazy fetch self-consistent); and `_ensure_qwen3_moe_finegrained_bases` (`qwen3_moe.py:2492-2515`) **lazily slices fused gate_up into two NEW AsymGroupedFrozenLinear** — must run eagerly at conversion under panvme (Stage 4 risks).
- HF expert source params are 0-numel'ed after adoption (`qwen3_moe.py:2127-2136`) — HostWeight owns the only copy; spilling genuinely frees RSS.

### 0.4 Backend tokens — dispatch sites (exact-token case arms; no glob collision with `zero3_offload_panvme`)

Grammar: `model|gpus ; backend|recompute|liger[|kernelcode] ; seq|batch|grad_accum ; policy|expact|attnact|layeract|layergc|sdparecomp`. Policy-list env override = **`ASYMM_EXP_ACT_POLICIES`** (`profile_lora_lf_test_source.sh:2122`; CLI `--asymm-exp-act-policies` `:2157`). The recompute field accepts `recomp-off-full-fg` etc. (`recompute_label()`), which drives the `ASYMM_*_FINEGRAINED_*` env.

Sites (extended in Stage 2): `profile_lora_lf_test_source.sh` — `append_backend_spec` `:956-978`, `backend_gpu_count` `:789-795`, `cpuadam_backend_for_label` `:918-923`, per-job derivation `~:3094-3110`, `run_env` block `:3343-3387`; backend token is the first `path_label` component (`job_root_path` `:1733-1741`) → run dirs auto-disambiguate; completion checks receive `backend=$3` (`:1104,:1148`). `run_lf_lora_sft.sh` — main case `:352-358` + die `:369`; `is_zero_backend_run` requires `BACKEND==torch` (`:659-661`) → our arms set `BACKEND=asym`, no `--deepspeed` leakage; `.aioenv` exports `:29-40`; `RECORD_IO` `/sys/block` sampler `:187,2417-2430` = free cross-check. `run_lf_profiled_train.py` — classification `:577-599` (`:579-582`), `_config_from_args` `:546` (env-mirror pattern `:732-734`). **No LlamaFactory changes** — config rides env (ASYMM_* house pattern).

### 0.5 Profiling / gating infra

- Canonical driver `scripts/lf/profile_lora_lf_test_source.sh` (`PROFILERS` default `source` `:131`; shorthands `M[q3-32b]` `:34-46`; `PREPARE_DATASETS` `:204`; CLI `--gpus/--overwrite/--prepare-datasets`). `_both.sh` = same driver (nsys default); apply identical edits.
- `source_profile.json` from `report()` (`run_lf_profiled_train.py:2849-2919`). `activation_offload` block from `_activation_offload_counters_from_model()` (`:2198-2282`): per-module `_last_activation_offload_stats` dicts flow into rows **automatically** (new `snapshot()` keys propagate for free); the **aggregates** at `:2265-2281` are explicit and must be extended. Per-step RSS = `step_samples.rows[].training_step_process_rss_peak_end_bytes` (use this, not VmHWM). `memory_attribution.rows[]` has `category=host_weight, device=cpu, bytes, pinned_bytes`.
- Compare-gate template: `scripts/lf/compare_liger_loss_profiles.py` (args `:33-42`, memory `:156`, median step/fwd/bwd from `step_samples.csv` `:208-235`, `{"ok":...}` + `SystemExit(2)`). Cloned in Stage 2. Eyeball: `scripts/lf/show_metrics.py`.

### 0.6 Prefetch/spill design cross-check (DeepSpeed / TE / Megatron, audited 2026-07-01)

- **DeepSpeed has NO activation prefetch** (its `cpu_checkpointing` restore is sync; dead `transport_stream`, `checkpointing.py:613-649`). Its param coordinator is the pattern source: correctness never depends on prefetch; byte-budgeted lookahead (`partitioned_param_coordinator.py:428-433`); trace record→freeze→invalidate; release-by-reuse-distance; split R/W aio handles.
- **Megatron fine-grained offload** (`fine_grained_activation_offload.py`): FIFO offload in creation order, LIFO reload, issue-ahead decoupled from consumer-wait, per-name in-flight caps, **terminal margin** (never offload the last group — its reload would stall backward immediately, `:539-554,962-984`). Our watermark governor generalizes the margin: everything within the CPU budget stays resident — the newest (first-consumed) tensors are by construction the last to spill.
- **Adopted:** byte-budgeted prefetch (`ASYM_NVME_ACT_PREFETCH_BYTES`); oldest-first spill = exact Belady under LIFO consumption (farthest future use); watermark-with-hysteresis pressure control.
- **Deferred rung (v3 of the ladder, not shelved):** TE v1/Megatron overlap the H2D reload leg on a dedicated stream + double buffering (`cpu_offload_v1.py:366-367,578`); our staging H2D is synchronous today — add `ASYM_NVME_ACT_H2D_STREAM` only if Stage-3 measurements show the H2D copy matters. Ladder: v1 fully sync (correct, bit-exact, debuggable) → v2 prefetch flag → v3 H2D stream; every rung flag-gated with sync fallback.

### 0.7 Hardware & feasibility

4× Samsung PM9A3 3.84 TB RAID0 → `/scratch_local` (~26 GB/s read, ~14 GB/s write). `ASYM_NVME_PATH` defaults there.
- **actnvme traffic = overflow only**: per step ≈ `max(0, activation_footprint − CPU_budget)` written once + read once. Example: footprint 800 GiB, budget 600 GiB → ~200 GiB/step each way → at a 300 s long-seq step, ~0.7 GB/s — trivial. Feasibility precheck (before Stage 3, from existing profiles): `(activation_offload.max_cpu_peak_bytes_live − intended_budget) ÷ step_seconds ≲ 14 GB/s`.
- panvme: ~60 GiB base read fwd+bwd = ~120 GiB/step → 4-6 GB/s at 20-30 s steps — overlappable; ≤5% gate.
- Endurance: ~1 DWPD class (~7 PBW/drive, 28 PBW RAID). Writes-only wear; overflow-only volumes at research duty cycles = non-issue (only months of 24/7 saturated spill would matter). Disk capacity: the arena reuses offsets every microbatch — holds ~one microbatch's overflow, not accumulating.

---

## Design contract (all stages)

1. **One store, role-tagged.** `NVMeStore` serves `{base_weight, activation}`; tokens map `actnvme→{activation}`, `panvme→{base_weight}`, `bothnvme→both`. Consumers see only the store API — the swappable seam (local now; per-rank or DeepSpeed-owned later). **Placement is NOT abstracted** — it stays in the consumers (it *is* AsymGEMM's algorithm).
2. **Zero forward-semantics changes.** What is saved vs recomputed, every kernel, every launch: untouched. actnvme changes only the *backing residency* of already-offloaded CPU tensors, and only under pressure.
3. **Spill on pressure, not eagerly.** Activations live in CPU RAM up to a budget; only the overflow spills, oldest-first (= farthest future use under LIFO consumption = exact Belady). No producer lists, no tag policies, no per-layer selection.
4. **No kernel computes from NVMe.** Always `NVMe → pinned CPU → (H2D) → compute`; `pinned CPU → NVMe`.
5. **Single-owner AIO handles.** Write handle owned by ONE writer thread; read handle by the main thread. Python pending ledgers; never `wait()` idle.
6. **Event-gated buffer reuse.** Any pinned buffer a GPU kernel may still stream is recycled only after a post-enqueue CUDA event completes (the seal event).
7. **Off = byte-identical.** No `*nvme` token ⇒ no deepspeed import, no thread, no file, identical allocations; every hook is a `None` check.
8. Every file offset and IO length padded to `store.align`; IO buffers allocated padded.

---

## Stage 0 — NVMe traffic census (measurement-only; MUST complete before any implementation)

**Purpose:** produce, from ordinary CPU-offload profile runs, the **projected NVMe traffic**: per-layer read+write per step and whole-model read+write per step (+ feasibility vs the 14/26 GB/s ceilings, + overflow-vs-budget table). This artifact is what decides budgets, which engines matter, and whether Stage 3 is even bandwidth-feasible. **No implementation starts until these artifacts are reviewed.**

### What already exists (verified) — and what does not

- **Device-level ACTUAL NVMe IO — exists:** `RECORD_IO=1` (default when `PROFILE=1`) samples `/sys/block/<dev>/stat` → `io_samples.csv` (`run_lf_lora_sft.sh:187,2417-2430`); `scripts/lf/summarize_nvme_offload.py` reduces it (+nsys nvtx) → `offload_io.json` (`nvme.read_gb/write_gb`, peak GB/s). Whole-run totals only, **no per-layer**, and it measures *real* disk IO — today that's only the DeepSpeed `zero3_offload_*nvme` baselines; later it is the independent cross-check for our own implementation. It cannot *predict*.
- **The projection source data — exists but is not surfaced:** every offload module publishes `_last_activation_offload_stats` → `source_profile.json.activation_offload.rows[]` (per module: `offloaded_bytes`, `offload_bytes_by_tag`, `stage_bytes_by_tag`, `num_offloads`, peaks; harvested at `run_lf_profiled_train.py:2198-2282`). Base-weight bytes per component are in `memory_attribution.rows[]` (`category=host_weight, device=cpu`). **No artifact today turns these into per-layer/per-step traffic.**
- **Semantics trap (must be handled or the numbers are garbage):** row counters have THREE different lifetimes —
  1. **fg dense-MLP rows**: the manager is created per Function call and snapshotted at the end of forward and backward (`dense_mlp_finegrained.py:300,:453`) ⇒ the row is **per-(layer, microbatch) directly** — exactly the per-layer-per-step number wanted (backward snapshot includes the backward-transient offloads).
  2. **attention rows**: `_update_snapshot` merges a per-call local manager with the **persistent, cumulative** q/k/v source context (`attention_activation_offload.py:501-511`) ⇒ local part per-call, `source_context` part cumulative (÷ total forward passes).
  3. **saved-tensor wrapper rows** (decoder/attention/linear-attention wrappers): counters accumulate on the wrapper across the whole run (`decoder_activation_offload.py:201-207`) ⇒ **cumulative** (÷ total forward passes = (warmup+measured steps) × grad_accum).
- **Tag classification** separates would-be NVMe traffic from noise: fwd-saved cross-phase tags (`mlp.X/gate/up/act/S_*`, `*.U`, `*.S`) = NVMe write candidates (and each is read back once ⇒ read ≈ write); backward-transient tags (`mlp.dact/dgate/dup`) and staging tags (`*_for_*`) = **never-spill**, excluded.

### Needed changes (postprocess-only; ZERO runtime edits)

NEW `scripts/lf/project_nvme_traffic.py` (also callable from `postprocess_lf_profile_artifacts.py` next to the other emitters):

```python
# Input: a run dir (source_profile.json). Output: nvme_traffic_projection.csv + .md
# 1. steps = trainer.timing.measured_steps + config warmup; micro = steps * grad_accum
# 2. for each activation_offload.rows[] row: classify semantics (per-call fg row | mixed attention row |
#    cumulative wrapper row) by module class; normalize everything to bytes PER MICROBATCH-STEP.
# 3. per (layer=module name, tag): fwd_saved_bytes_step (NVMe write candidate), read_bytes_step (=write),
#    transient_bytes_step (excluded, shown for context), stage_bytes_step (H2D volume, v3 H2D-stream input).
# 4. panvme sheet: per memory_attribution host_weight/cpu row → weight bytes; per-step read = 2× (fwd+bwd);
#    model total = Σ eligible components.
# 5. summary block:
#    total_fwd_saved_bytes_step  (= NVMe write/read ceiling at budget 0)
#    write_seconds_step = total/14e9 ; read_seconds_step = total/26e9 ; vs measured step_seconds
#    overflow table for budget ∈ {100,200,400,600 GiB}: overflow/step, write-s/step, %step
#    per-layer table (top-N by bytes) and per-tag totals.
```

Optional cross-check row: when the run is a DeepSpeed `*nvme` baseline, print `offload_io.json` device totals next to the projection (sanity that the accounting pipeline and the device sampler agree on a system where real IO exists).

### Validation (Stage 0 gate — this IS the decision artifact)

```bash
# one fg profile at two operating points (small + the intended capacity point):
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage0_census PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false || q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 16384|8|1 ; none|false|false|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
.venv/bin/python scripts/lf/project_nvme_traffic.py --run-dir profiling_nvme/stage0_census/<run_dir>
```

Accept: `nvme_traffic_projection.{csv,md}` exist; per-layer rows sum to the model total; totals are consistent with `activation_offload` aggregates (`total_cpu_owned/offloaded` relations) and with per-step RSS deltas within ~10%; the two seq points scale ≈ linearly in tokens; the md's feasibility block gives an unambiguous verdict (`write_seconds_step` vs `step_seconds`, overflow table). **Review the artifact and fix the budget/engine decisions before Stage 1 begins.**

### Risks / watch
- Row-semantics misclassification (the ×steps vs ×1 trap) — assert per-call rows change across two runs with different MAX_STEPS only in cumulative rows.
- Tag lists drift as engines evolve — keep the fwd-saved/transient tag tables in one dict at the top of the script.

---

## Stage 1 — `NVMeStore` substrate

**Scope:** NEW `asym_gemm/training/nvme_store.py` + NEW `tests/training/test_nvme_store.py`. Zero edits elsewhere ⇒ isolated unit gate suffices (the one exception to the e2e rule).

### Implementation

```python
# asym_gemm/training/nvme_store.py
from __future__ import annotations
import os, queue, threading, time
from dataclasses import dataclass, field
from typing import Any, Callable
import torch

@dataclass(frozen=True)
class NVMeStoreConfig:
    path: str                                    # ASYM_NVME_PATH (required)
    roles: frozenset[str]                        # {"base_weight","activation"} from ASYM_NVME_ROLES
    aio_block_size: int = 1 << 20
    aio_queue_depth: int = 16
    aio_intra_op_parallelism: int = 4
    aio_single_submit: bool = False
    aio_overlap_events: bool = True
    min_swappable_bytes: int = 1 << 20
    activation_arena_bytes: int = 1 << 40        # sparse file, allocated on demand
    max_inflight_spill_bytes: int = 8 << 30      # writer backpressure

def _config_from_env() -> NVMeStoreConfig | None:
    roles = frozenset(r.strip() for r in os.environ.get("ASYM_NVME_ROLES", "").split(",") if r.strip())
    if not roles: return None
    bad = roles - {"base_weight", "activation"}
    if bad: raise ValueError(f"unknown ASYM_NVME_ROLES: {sorted(bad)}")
    path = os.environ.get("ASYM_NVME_PATH") or _fail("ASYM_NVME_PATH required")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1: _fail("asym NVMe store is single-process only")
    return NVMeStoreConfig(path=path, roles=roles, **_ints_and_bools_from_env())

def _pad(n, a): return (n + a - 1) // a * a

def _flat_u8(t: torch.Tensor) -> torch.Tensor:
    """uint8 alias of t's WHOLE storage (strided/padded safe; zero-copy). All IO uses this."""
    out = torch.empty(0, dtype=torch.uint8)
    out.set_(t.untyped_storage(), 0, (t.untyped_storage().nbytes(),))
    return out

def alloc_padded_pinned(shape, dtype, stride=None, *, align) -> torch.Tensor:
    """Pinned CPU tensor with PADDED backing storage; exact shape/stride view returned.
    Torch caching host allocator → cheap after warmup; pin-fail falls back unpinned (aio bounces)."""
    stride = stride or _contiguous_strides(shape)
    storage = torch.empty(_pad(required_storage_nbytes(shape, stride, dtype), align),
                          dtype=torch.uint8, pin_memory=True)
    t = torch.empty(0, dtype=dtype)
    t.set_(storage.untyped_storage(), 0, shape, stride)
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
    bytes_written: dict = field(default_factory=dict); bytes_read: dict = field(default_factory=dict)
    write_ops: dict = field(default_factory=dict); read_ops: dict = field(default_factory=dict)
    fetch_wait_ms: float = 0.0; spill_backpressure_ms: float = 0.0
    inflight_peak_bytes: int = 0; arena_peak_bytes: dict = field(default_factory=dict)
    def as_dict(self) -> dict[str, Any]: ...

_STOP = object()

class _WriterThread(threading.Thread):
    """Sole owner of the write aio_handle."""
    def __init__(self, handle, cfg, stats):
        super().__init__(name="asym-nvme-writer", daemon=True)
        self._h, self._cfg, self._stats = handle, cfg, stats
        self._q = queue.Queue(); self._inflight = 0; self._cv = threading.Condition()

    def submit(self, ready_event, buf_u8, ref, on_done) -> None:
        t0 = time.perf_counter()
        with self._cv:
            while self._inflight > self._cfg.max_inflight_spill_bytes:  # backpressure
                self._cv.wait()
            self._inflight += buf_u8.nbytes
            self._stats.inflight_peak_bytes = max(self._stats.inflight_peak_bytes, self._inflight)
        self._stats.spill_backpressure_ms += (time.perf_counter() - t0) * 1e3
        self._q.put((ready_event, buf_u8, ref, on_done))

    def run(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP: return
            ready_event, buf, ref, on_done = item
            if ready_event is not None:
                ready_event.synchronize()               # seal event: D2H done + forward consumers done
            self._h.async_pwrite(buf, ref.file, ref.offset)
            n = self._h.wait(); assert n == 1           # sole owner → exactly this op
            self._stats.bytes_written[ref.role] = self._stats.bytes_written.get(ref.role, 0) + buf.nbytes
            self._stats.write_ops[ref.role] = self._stats.write_ops.get(ref.role, 0) + 1
            ref.durable.set()
            with self._cv:
                self._inflight -= buf.nbytes; self._cv.notify_all()
            on_done(buf, ref)                           # writer-thread context; NEVER touch CUDA here

class NVMeStore:
    def __init__(self, cfg: NVMeStoreConfig):
        from deepspeed.ops.op_builder import AsyncIOBuilder   # imported ONLY when enabled
        m = AsyncIOBuilder().load(verbose=False)
        mk = lambda: m.aio_handle(cfg.aio_block_size, cfg.aio_queue_depth, cfg.aio_single_submit,
                                  cfg.aio_overlap_events, cfg.aio_intra_op_parallelism)
        self.cfg = cfg
        self._read_h, write_h = mk(), mk()               # separate: wait() drains per-handle
        self.align = int(self._read_h.get_alignment())   # 2048 @ intra=4 (measured)
        self.stats = NVMeStoreStats()
        self._writer = _WriterThread(write_h, cfg, self.stats); self._writer.start()
        os.makedirs(os.path.join(cfg.path, "base_weight"), exist_ok=True)
        self._arena_path = os.path.join(cfg.path, "activation.arena")
        self._arena_cursor = 0; self._arena_live = 0
        self._pending_reads: dict[int, BlobRef] = {}     # MAIN THREAD ONLY

    def has_role(self, role): return role in self.cfg.roles

    # -- activation arena: bump allocator, reset-when-empty (blob lifetime = one microbatch fwd→bwd,
    #    so live==0 recurs each microbatch; exact under grad accumulation, no trainer hooks) --
    def _arena_alloc(self, nbytes):
        length = _pad(nbytes, self.align)
        off = self._arena_cursor; self._arena_cursor += length; self._arena_live += 1
        self.stats.arena_peak_bytes["activation"] = max(self.stats.arena_peak_bytes.get("activation", 0), self._arena_cursor)
        if self._arena_cursor > self.cfg.activation_arena_bytes:
            raise RuntimeError("activation arena full — raise ASYM_NVME_ACTIVATION_ARENA_BYTES")
        return self._arena_path, off

    def blob_done(self, ref):                            # after final fetch OR dropped blob
        if ref.role == "activation":
            self._arena_live -= 1
            if self._arena_live == 0: self._arena_cursor = 0

    # -- write path (any thread → writer) --
    def spill(self, role, tensor, *, ready_event, on_done) -> BlobRef:
        buf = _flat_u8(tensor)
        assert buf.nbytes % self.align == 0, "allocate via alloc_padded_pinned"
        file, off = (self._per_blob_file(tensor), 0) if role == "base_weight" else self._arena_alloc(buf.nbytes)
        ref = BlobRef(role, file, off, buf.nbytes, int(tensor.numel() * tensor.element_size()))
        self._writer.submit(ready_event, buf, ref, on_done)
        return ref

    # -- read path (MAIN THREAD ONLY) --
    def submit_pread(self, ref, dst_padded_pinned) -> None:
        if not ref.durable.is_set(): ref.durable.wait()  # write in flight → bounded rare block
        dst = _flat_u8(dst_padded_pinned); assert dst.nbytes == ref.length
        self._read_h.async_pread(dst, ref.file, ref.offset)
        self._pending_reads[id(ref)] = ref

    def drain_reads(self) -> set[int]:
        """Blocks until ALL pending reads complete (wait() drains the whole handle). Returns arrived ids."""
        if not self._pending_reads: return set()
        n = self._read_h.wait(); assert n == len(self._pending_reads)
        for r in self._pending_reads.values():
            self.stats.bytes_read[r.role] = self.stats.bytes_read.get(r.role, 0) + r.length
            self.stats.read_ops[r.role] = self.stats.read_ops.get(r.role, 0) + 1
        done = set(self._pending_reads); self._pending_reads.clear()
        return done

    def fetch_into(self, ref, dst_padded_pinned) -> None:
        t0 = time.perf_counter()
        self.submit_pread(ref, dst_padded_pinned); self.drain_reads()
        self.stats.fetch_wait_ms += (time.perf_counter() - t0) * 1e3

_STORE = None; _STORE_INIT = False
def get_nvme_store() -> NVMeStore | None:
    """Lazy env singleton. Without ASYM_NVME_ROLES: returns None, imports nothing, allocates nothing."""
```

Locked decisions: base_weight = one file per HostWeight (static, GB-scale, written once); activation = one arena file + bump/reset-when-empty (thousands of transient blobs/step); no cancel path in the store (fetch-before-durable waits; consumed-before-submit is handled ABOVE the store by the governor's CLAIMED state, Stage 3); writer waits per-op (each op is already 4-thread×qd16 internally; batch-submit is a flagged follow-up).

### Efficiency
Per-op internal parallelism 4×16×1MB; writer-thread overlap (GIL released in waits); multi-pread prefetch ledger; zero extra memcpys (padded pinned buffer IS the D2H destination and the IO buffer); single-owner handles → no hot-path locks.

### Validation (Stage 1 gate)

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export AIO_HOME=$PWD/.aioenv CPATH="$AIO_HOME/include:${CPATH:-}" \
       LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
ASYM_NVME_PATH=/scratch_local/user_data/shutian/kevin/cache/asym_nvme_test \
.venv/bin/python -m pytest tests/training/test_nvme_store.py -q
```

Required tests: bf16/fp32 roundtrips below/at/above 1 MiB; strided roundtrip restores strides; two arena blobs at different offsets, no cross-corruption; spill gated on a CUDA event (write CUDA tensor D2H, record, spill, fetch, compare); fetch-before-durable blocks then succeeds; 3-deep prefetch ledger reconciles; arena reset-when-empty across two simulated microbatches; backpressure blocks and resumes; disabled env → `get_nvme_store() is None` and `deepspeed` absent from `sys.modules`; clean writer shutdown.

### Risks / watch
- Handle thread-ownership is a rule, not API-enforced — assert thread identity in debug mode.
- Arena overflow at extreme seq×accumulation → loud error + env knob.

---

## Stage 2 — Backend tokens, env plumbing, profile counters, compare gate

**Scope (no tensors move):** `scripts/lf/profile_lora_lf_test_source.sh` (+ identical in `_both.sh`), `scripts/lf/run_lf_lora_sft.sh`, `scripts/lf/run_lf_profiled_train.py`, `scripts/lf/postprocess_lf_profile_artifacts.py`, NEW `scripts/lf/compare_nvme_profiles.py`.

**(a) profile script:** three exact-token arms in `append_backend_spec` (`:956-978`) + die; add tokens to `backend_gpu_count`'s 1-GPU asym line (`:789-795`) + die; `cpuadam_backend_for_label` (`:918-923`):

```bash
    asym_cpuadamwds|asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme) printf 'deepspeed\n' ;;
```

plus a `nvme_roles_for_backend()` helper and `run_env` additions near `:3343`:

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
             "ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES=${job_nvme_roles}"
             "ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH=${ASYM_NVME_PATH:-...}" )
fi
```

**(b) run_lf_lora_sft.sh:** one grouped arm cloned from `asym_cpuadamwds` (`:352-358`) + die (`:369`):

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
    BACKEND=asym
    ;;
```

Mirror `ASYM_GEMM_LF_CONFIG_ASYM_NVME_{ROLES,PATH}` near `:2297`.

**(c) run_lf_profiled_train.py:**

```python
_ASYM_CPUADAMW_DS_BACKENDS = {"asym_cpuadamwds", "asym_cpuadamwds_panvme",
                              "asym_cpuadamwds_actnvme", "asym_cpuadamwds_bothnvme"}
is_asym_deepspeed_cpuadamw = backend in _ASYM_CPUADAMW_DS_BACKENDS or (...)
# _config_from_args:
"asym_nvme_roles": os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES") or os.environ.get("ASYM_NVME_ROLES", ""),
"asym_nvme_path":  os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH")  or os.environ.get("ASYM_NVME_PATH", ""),
# report(), sibling of "activation_offload" (:2846):
"asym_nvme": _asym_nvme_summary_from_model(),
```

```python
def _asym_nvme_summary_from_model() -> dict[str, Any]:
    try: from asym_gemm.training.nvme_store import get_nvme_store
    except Exception as exc: return {"enabled": False, "reason": repr(exc)}
    store = get_nvme_store()
    if store is None: return {"enabled": False}
    out = {"enabled": True, "roles": sorted(store.cfg.roles), "path": store.cfg.path,
           "alignment": store.align, "aio": {...}, **store.stats.as_dict()}
    from asym_gemm.training.act_spill_governor import get_act_spill_governor   # Stage 3
    gov = get_act_spill_governor()
    if gov is not None: out["act_governor"] = gov.summary()
    model, _ = _model_and_base_model()
    pager = getattr(model, "_asym_base_weight_pager", None)                    # Stage 4
    if pager is not None: out["base_weight_pager"] = pager.summary()
    return out
```

Aggregates tail of `_activation_offload_counters_from_model` (`:2265-2281`): add `total_nvme_spilled_bytes`, `total_nvme_bytes_read`, `total_nvme_fetch_wait_ms`, `total_nvme_spill_backpressure_ms` summed from row stats.

**(d) postprocess:** `_asym_nvme_rows()` flattener → `asym_nvme.csv` next to `_asym_cpu_adamw_rows` (`:378`); one NVMe line in `memory.md` (`:1803`).

**(e) compare_nvme_profiles.py** — clone of the liger tool:

```text
--baseline DIR --candidate DIR --target {no_change, base_weight_cpu, activation_cpu, maxseq}
--memory-metric DOTTED  ("step_samples.<col>" = median of measured csv rows)
--min-memory-drop-gib | --min-memory-drop-pct  (no_change: --max-memory-drift-gib)
--max-step-ratio/--max-forward-ratio/--max-backward-ratio (default 1.05; capacity 1.10)
--expect-nvme-role ROLE  (asserts enabled + ROLE∈roles + bytes_written>0 + bytes_read>0)
Checks: artifacts exist (+asym_nvme.csv on candidates); finite losses; measured_steps>=5; config roles match.
{"ok":...}; SystemExit(2) on failure.
```

### Validation (Stage 2 gate — paired e2e no-change)

```bash
bash -n scripts/lf/profile_lora_lf_test_source.sh scripts/lf/run_lf_lora_sft.sh
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage2_nochange PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage2_token PROFILERS=source PLOT=false \
PREPARE_DATASETS=false WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds_actnvme|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline profiling_nvme/stage2_nochange/<run_dir> --candidate profiling_nvme/stage2_token/<run_dir> \
  --target no_change --memory-metric memory.gpu.peak_allocated_hbm_bytes --max-memory-drift-gib 0.5 \
  --max-step-ratio 1.02 --max-forward-ratio 1.02 --max-backward-ratio 1.02
```

Accept: drift/latency inside bounds; candidate `config.asym_nvme_roles=="activation"`, `asym_nvme.enabled==true` with `bytes_written==0` (Stage 3 not yet implemented); `asym_nvme.csv` present; the tool demonstrably fails on a mismatched pair. (Verify the fg policy tuple against an existing `recomp-off-full-fg` run dir before the run.)

### Risks / watch
- Apply identical arms to `_both.sh`. Cached run-dir reuse keys on backend `$3` → distinct; still pass `--overwrite true`.

---

## Stage 3 — `actnvme`: watermark spill on the fine-grained recompute path ← FIRST tensor-moving stage

**What it is:** run `asym_cpuadamwds_actnvme|recomp-off-*-fg|ligerloss1` exactly as today — same recompute decisions, same offloads, **zero forward changes** — and let a global **spill governor** move already-offloaded CPU tensors to NVMe **only when live activation bytes cross a CPU budget**, oldest-first (queue-based spill), fetched back in backward's reverse order (stack-based consumption).

**Why oldest-first is optimal:** backward consumes newest-first (LIFO). The oldest tensors have the farthest future use ⇒ spilling them first is exact Belady. The newest (first-consumed) tensors never spill unless pressure is extreme — Megatron's terminal-margin property falls out for free.

**Precheck (5 min, existing profiles):** `(activation_offload.max_cpu_peak_bytes_live − intended_budget) ÷ step_seconds ≲ 14 GB/s`.

**Scope:** NEW `asym_gemm/training/act_spill_governor.py`; `asym_gemm/training/activation_offload.py` (manager hooks); `asym_gemm/training/dense_mlp_finegrained.py` (ONE seal line). Nothing else.

### The governor (the "carefully designed scheduler")

Per-handle state machine, all transitions under one small lock:

```text
RESIDENT(queued) ──pressure──▶ SUBMITTED ──write done──▶ DURABLE(buffer freed) ──backward──▶ FETCHED
      │                            │
      └──backward consumes──▶ CLAIMED (dequeue; never spilled)   [SUBMITTED→CLAIMED: buffer still valid —
                                                                  writer only READS it; on_durable sees
                                                                  CLAIMED → skips free, drops the blob]
```

```python
# asym_gemm/training/act_spill_governor.py
RESIDENT, SUBMITTED, DURABLE, CLAIMED, FETCHED = range(5)

@dataclass
class _Rec:
    handle: Any; manager: Any; nbytes: int
    state: int = RESIDENT
    seal_event: Any = None       # set at seal; spill eligibility
    ref: Any = None              # BlobRef once SUBMITTED
    order: int = 0               # creation index → prefetch walks it reversed

class ActSpillGovernor:
    """Global singleton (role 'activation'). Queue-based spill under CPU pressure;
    stack-based (LIFO) consumption with byte-budgeted reverse prefetch."""

    def __init__(self, store):
        self._store = store
        self._lock = threading.Lock()
        self._by_id: dict[int, _Rec] = {}          # id(handle) → rec (handle alive on ctx until backward)
        self._pending: deque[_Rec] = deque()        # RESIDENT, sealed or not, creation order
        self._spilled: list[_Rec] = []              # SUBMITTED/DURABLE in spill order (prefix of queue)
        self.live_cpu_bytes = 0                     # ALL manager-offloaded bytes (incl. backward transients)
        # budget: ASYM_NVME_ACT_CPU_BUDGET_BYTES; 0/auto → 0.85 × MemAvailable at init (logged)
        self.hi = _budget_from_env(); self.lo = max(0, self.hi - _env_int("ASYM_NVME_ACT_LOW_SLACK", 8 << 30))
        self.prefetch_bytes = _env_int("ASYM_NVME_ACT_PREFETCH_BYTES", 0)   # 0 = sync v1
        self.stats = GovStats()   # spilled/claimed/fetched bytes+counts, live_peak, wasted_writes

    # ---- forward side ----
    def on_offload(self, manager, handle, *, sealable: bool) -> None:
        """Called from ActivationOffloadManager.offload()/adopt_cpu(). sealable=False for
        handles created while a Function.backward is running (backward transients) — they are
        counted for pressure but can NEVER spill (never sealed)."""
        with self._lock:
            self.live_cpu_bytes += handle.nbytes
            self.stats.live_peak = max(self.stats.live_peak, self.live_cpu_bytes)
            if sealable and handle.nbytes >= self._store.cfg.min_swappable_bytes:
                rec = _Rec(handle, manager, handle.nbytes, order=self._next_order())
                self._by_id[id(handle)] = rec
                self._pending.append(rec)
        # NOTE: no spill here — eligibility requires the seal event.

    def on_seal(self, manager, handles) -> None:
        """ONE call at the END of Function.forward: every forward consumer of these handles is
        enqueued on the current stream. One CUDA event covers (a) the D2H fills and (b) all
        forward CPU-operand kernels streaming the buffers."""
        ev = torch.cuda.Event(); ev.record()
        with self._lock:
            for h in handles:
                rec = self._by_id.get(id(h))
                if rec is not None: rec.seal_event = ev
        self._maybe_spill()

    def _maybe_spill(self) -> None:               # pressure check; also invoked from on_offload
        with self._lock:
            projected = self.live_cpu_bytes
            while projected > self.hi and self._pending:
                rec = self._pending[0]
                if rec.state is not RESIDENT: self._pending.popleft(); continue
                if rec.seal_event is None: break   # oldest not sealed yet (its Function.forward still
                                                   # running) → nothing older can spill; retry at next seal
                self._pending.popleft()
                rec.state = SUBMITTED
                rec.ref = self._store.spill("activation", rec.handle.tensor,
                                            ready_event=rec.seal_event,
                                            on_done=self._make_on_durable(rec))
                self._spilled.append(rec)
                projected -= rec.nbytes             # hysteresis target
                if projected <= self.lo: break

    def _make_on_durable(self, rec):
        def cb(buf, ref):                          # WRITER THREAD
            with self._lock:
                if rec.state is CLAIMED:           # backward consumed it while write in flight
                    self._store.blob_done(ref)     # drop the (wasted) blob; buffer stays handle-owned
                    self.stats.wasted_writes += 1
                    return
                rec.state = DURABLE
                self.live_cpu_bytes -= rec.nbytes
            rec.manager._pop_active(rec.handle)                     # accounting off this data_ptr
            _return_cpu(rec.handle.tensor, pin_memory=True)         # pool reuse safe: seal event synced
            object.__setattr__(rec.handle, "tensor", _SENTINEL)     # stray reads fail loudly
        return cb

    # ---- backward side (MAIN THREAD) ----
    def ensure_local(self, handle) -> None:
        """First line of wait_cpu_ready() AND stage*(). O(1) dict miss for non-tracked handles."""
        rec = self._by_id.get(id(handle))
        if rec is None: return
        with self._lock:
            if rec.state in (RESIDENT, SUBMITTED):     # still CPU-valid (writer only READS the buffer)
                rec.state = CLAIMED                    # dequeue lazily; on_durable handles the race
                return
            if rec.state in (CLAIMED, FETCHED): return
            assert rec.state is DURABLE
        bounce = _alloc_cpu(handle.original_shape, handle.original_dtype, pin_memory=True)  # padded pool buffer
        self._store.fetch_into(rec.ref, bounce)        # drains any in-flight prefetches too
        self._store.blob_done(rec.ref)
        object.__setattr__(handle, "tensor", bounce)
        rec.manager._mark_cpu_live(handle)             # re-enter accounting under new data_ptr
        with self._lock:
            rec.state = FETCHED; self.live_cpu_bytes += rec.nbytes
        self._prefetch_reverse(rec)

    def _prefetch_reverse(self, rec) -> None:
        """Byte-budgeted lookahead over the spilled prefix, newest-spilled → oldest (= consumption order).
        drain-all + durable events make a mis-ordered prefetch an efficiency blip, never a bug."""
        if self.prefetch_bytes <= 0: return
        inflight = 0
        for r in self._iter_spilled_older_than(rec):   # walk self._spilled reversed from rec's position
            if inflight >= self.prefetch_bytes: break
            if r.state is DURABLE and not r.prefetch_issued:
                r.bounce = _alloc_cpu(...); self._store.submit_pread(r.ref, r.bounce)
                r.prefetch_issued = True; inflight += r.nbytes

    def on_release(self, handle) -> None:
        """From manager.release_cpu(): backward finally-block. Covers every terminal state."""
        rec = self._by_id.pop(id(handle), None)
        with self._lock:
            if rec is None:
                self.live_cpu_bytes -= handle.nbytes if _tracked_untracked(handle) else 0
                return
            if rec.state in (RESIDENT, CLAIMED, FETCHED, SUBMITTED):
                self.live_cpu_bytes -= rec.nbytes
            if rec.state is SUBMITTED:  # released without consumption while write in flight
                rec.state = CLAIMED     # on_durable drops the blob; normal pool-return proceeds
            elif rec.state is DURABLE:  # spilled, never fetched (e.g. grad not needed)
                self._store.blob_done(rec.ref)
```

### Manager integration (`activation_offload.py`, ~25 lines)

```python
# offload()/adopt_cpu()/empty_cpu(): when a store with role "activation" exists,
#   (a) _alloc_cpu allocates PADDED storage (pool key unchanged; pooled buffers just carry padded storages;
#       off-path allocations untouched — rule 7),
#   (b) call governor.on_offload(self, handle, sealable=not _in_function_backward())
#       where _in_function_backward() is a thread-local flag set by a tiny context manager that
#       Function.backward bodies already enter via prof_range — or simpler: sealable=True only for
#       offload() calls made while torch.is_grad_enabled() is False inside forward... DECIDE AT IMPL:
#       the robust minimal rule is "sealable ⇔ the handle is later passed to on_seal", so on_offload
#       marks nothing and on_seal is the sole eligibility source; unsealed handles simply never spill.
# wait_cpu_ready(handle): FIRST line → governor.ensure_local(handle)
# stage()/stage_rows()/stage_concat_columns(): FIRST line → governor.ensure_local(handle)
# release_cpu(handle): notify governor.on_release(handle) before pool-return; skip pool-return when
#   the governor already recycled/sentineled the buffer (DURABLE path).
# _return_cpu: guard the global pool with a small lock (writer thread returns buffers off-main-thread).
```

The simplest correct eligibility rule (adopted): **`on_offload` tracks every handle for pressure accounting; only handles passed to `on_seal` ever become spillable.** Backward-created transients are never sealed → never spilled, no phase-detection needed.

### Engine call site (`dense_mlp_finegrained.py`, ONE line)

```python
# end of _FinegrainedDenseMLPFunction.forward, right before
# `layer._last_activation_offload_stats = manager.snapshot()` (:300):
get_act_spill_governor_or_noop().on_seal(manager,
    (x_cpu, gate_cpu, up_cpu, act_cpu, gate_low_rank_cpu, up_low_rank_cpu, down_low_rank_cpu))
```

The backward `finally` (`:441-453`) already calls `release_cpu` on every handle → `on_release` covers spilled-never-fetched and claimed states with zero engine changes.

### Why the event algebra is safe
The seal event is recorded after (a) every D2H fill and (b) every forward kernel launch that streams these buffers — all earlier on the same stream. The writer syncs it before pwrite (stable bytes) and pool-return happens only after that same sync in `on_durable` (no kernel can still be streaming a recycled buffer). Backward consumers always pass through `ensure_local` first. `SUBMITTED→CLAIMED` is safe because the writer only *reads* the buffer; the wasted write's blob is dropped and never read. The `data_ptr`-keyed accounting hazard is avoided because `_pop_active` runs before pool-return and the handle's tensor is swapped to a sentinel.

### Efficiency
Zero forward-semantics changes; IO units are the wide fg tensors ([M,I]/[M,H], tens of MB–GiB); traffic = overflow only; spill FIFO = sequential arena writes at max bandwidth; consumption LIFO = the resident suffix is hit first (no NVMe wait while the read pipeline warms); overlap ladder v1 sync → v2 `ASYM_NVME_ACT_PREFETCH_BYTES` → v3 `ASYM_NVME_ACT_H2D_STREAM` (0.6), each flag-gated with sync fallback.

New counters (flow via `snapshot()` + governor summary): `nvme_spilled_bytes/count`, `nvme_claimed_before_spill`, `wasted_writes`, `nvme_bytes_read`, `nvme_fetch_wait_ms`, `governor_live_peak_bytes`, `spill_backpressure_ms`.

### Validation (Stage 3 gate — capacity mode)

Unit (`tests/training/test_act_spill_governor.py`): state machine transitions incl. SUBMITTED→CLAIMED race (mock writer delay) and DURABLE→FETCHED; oldest-first order; hysteresis stops at `lo`; unsealed handles never spill under pressure; release-without-fetch drops blobs; sentinel read raises; fg Function fwd+bwd **bit-identical** with `ASYM_NVME_ROLES=""` vs `"activation"` + tiny budget (forces spill) on a toy layer; existing `tests/training/test_dense_mlp_finegrained.py` passes with NVMe off AND on.

E2E (q3-32b; budget set LOW so spilling engages deterministically; sequential; `kill -TERM` only):

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage3_actnvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=3 MAX_STEPS=8 \
ASYM_NVME_ACT_CPU_BUDGET_BYTES=$((200*1024*1024*1024)) \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 16384|8|1 ; none|false|false|false|false|false || q3-32b|1 ; asym_cpuadamwds_actnvme|recomp-off-full-fg|ligerloss1 ; 16384|8|1 ; none|false|false|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true

.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline <asym_cpuadamwds dir> --candidate <asym_cpuadamwds_actnvme dir> \
  --target activation_cpu \
  --memory-metric step_samples.training_step_process_rss_peak_end_bytes \
  --min-memory-drop-gib 50 --max-step-ratio 1.10 --max-forward-ratio 1.10 --max-backward-ratio 1.10 \
  --expect-nvme-role activation
```

Accept: candidate's `governor_live_peak_bytes` **plateaus ≈ the budget** while baseline's `max_cpu_peak_bytes_live` exceeds it; per-step RSS drops by ≈ (baseline footprint − budget); HBM unchanged; step ≤1.10×; step-1 loss bit-identical; `nvme_bytes_written ≈ nvme_bytes_read ≈` overflow volume; `wasted_writes ≈ 0`. **Then the headline:** raise seq (and/or lower budget) until the baseline CPU-OOMs and the candidate still trains — record both max-seq numbers (verify real `input_ids` lengths in `train.log`).

### Risks / watch
- Verify the fg policy-tuple spelling against an existing `recomp-off-full-fg` run dir before the gate run.
- If `spill_backpressure_ms` dominates: overflow rate exceeds ~14 GB/s — raise the budget or accept/report the stall (capacity price).
- The `x_cpu` layer input is consumed near the END of each layer's backward (`:396,:420`) — good for fetch lead time; `act/S_down` are needed FIRST per layer — the reverse-order prefetch covers them; watch `nvme_fetch_wait_ms` per tag.
- Pinned-cache size-class churn from padded pool allocations — watch allocator RSS; mitigate by rounding padded sizes to 2 MiB classes.
- Auto budget (0.85×MemAvailable) races other jobs on a shared box — set the env explicitly for gate runs.

---

## Stage 4 — `panvme`: base weights → NVMe

**Scope:** `host_weight.py` (property surgery), NEW `base_weight_pager.py`, `lf.py` (one registration walk at end of `apply_lf_asym_lora`, ~`:2410`), `qwen3_moe.py` (eager fine-grained split).

```python
# host_weight.py — surgery (everything else unchanged)
class HostWeight:
    # NEW fields (default None; set only by pager.register): _pager, _pager_key
    @property
    def weight(self) -> torch.Tensor:
        pager = getattr(self, "_pager", None)
        if pager is None:
            return self._tensor                      # today's exact path (one attribute check added)
        return pager.touch(self._pager_key)
    tensor = weight
    # nbytes/shape/dtype/is_pinned/metadata read self._metadata → never fetch.
```

```python
# base_weight_pager.py
ABSENT, INFLIGHT, RESIDENT = range(3)

class BaseWeightPager:
    """Residency owner for registered HostWeights. MAIN THREAD ONLY."""
    def __init__(self, store, *, cache_bytes=_env_int("ASYM_NVME_BASE_WEIGHT_CACHE_BYTES", 8 << 30),
                 prefetch_bytes=_env_int("ASYM_NVME_BASE_WEIGHT_PREFETCH_BYTES", 0)):  # 0 → auto: 2×largest blob
        self._entries: dict[str, _Entry] = {}; self._by_ref_id = {}
        self._free: dict[tuple, list] = {}          # (dtype, shape) → free padded pinned bufs
        self._quarantine: list[tuple] = []          # (buf, cuda_event, shape_class)
        self._trace: list[str] = []; self._trace_build = []; self._frozen = False; self._disabled = False
        self._cursor = -1; self._last_key = None
        self.misses = self.misses_after_freeze = 0

    def register(self, key, hw):
        t = hw._tensor
        if t is None or t.numel() * t.element_size() < store.cfg.min_swappable_bytes: return
        padded = alloc_padded_pinned(tuple(t.shape), t.dtype, align=store.align); padded.copy_(t)
        ref = store.spill("base_weight", padded, ready_event=None, on_done=lambda b, r: None)
        self._entries[key] = _Entry(hw=hw, ref=ref, shape=tuple(t.shape), dtype=t.dtype,
                                    padded_nbytes=_pad(...), buf=None, state=ABSENT, positions=[])
        self._by_ref_id[id(ref)] = self._entries[key]
        hw._pager, hw._pager_key = self, key
        hw._tensor = None                            # ~GB home freed NOW (transient = one weight)

    def touch(self, key):
        e = self._entries[key]
        if key != self._last_key:                    # dedupe: .weight read several times per Function
            self._last_key = key
            self._record_or_advance(e); self._issue_prefetches(); self._evict_to_budget()
        if e.state is RESIDENT: return e.view
        if e.state is INFLIGHT:
            for rid in store.drain_reads(): self._by_ref_id[rid].state = RESIDENT
            return e.view
        e.buf = self._take_buffer(e); e.view = e.buf # ABSENT miss (step 1; ~never after freeze)
        store.fetch_into(e.ref, e.buf); e.state = RESIDENT
        self.misses += 1; self.misses_after_freeze += int(self._frozen)
        return e.view

    def _record_or_advance(self, e):
        if self._disabled: return
        if not self._frozen:
            self._trace_build.append(e.key)
            first = self._trace_build[0]
            if e.key == first and self._trace_build.count(first) == 3:
                # first key: fwd@0, its bwd, then fwd again = start of period 2 → freeze [0, here)
                self._trace = self._trace_build[:-1]
                for i, k in enumerate(self._trace): self._entries[k].positions.append(i)
                self._frozen = True; self._cursor = 0
            return
        n = len(self._trace)
        for d in range(0, 8):                        # jitter window
            if self._trace[(self._cursor + d) % n] == e.key:
                self._cursor = (self._cursor + d) % n; return
        self._disabled = True; self._trace = []      # dynamic order → miss-driven sync fallback, counted

    def _issue_prefetches(self):
        # Byte-budgeted lookahead (0.6): uniform lead TIME under mixed blob sizes (MoE);
        # degenerates to a fixed count on dense (uniform shapes).
        if not self._frozen or self._disabled: return
        n = len(self._trace); d = 1; inflight = self._inflight_prefetch_bytes()
        while inflight < self.prefetch_bytes and d < n:
            e = self._entries[self._trace[(self._cursor + d) % n]]; d += 1
            if e.state is not ABSENT: continue
            if self._would_evict_nearer_than(e): break   # tight-cache guard (DS max_live analog)
            e.buf = self._take_buffer(e); e.view = e.buf
            store.submit_pread(e.ref, e.buf); e.state = INFLIGHT; inflight += e.padded_nbytes

    def _take_buffer(self, e):
        self._sweep_quarantine()
        free = self._free.get((e.dtype, e.shape))
        if free: return free.pop()
        if self._resident_bytes() + e.padded_nbytes <= self.cache_bytes:
            return alloc_padded_pinned(e.shape, e.dtype, align=store.align)
        self._evict_one(exclude=e)                   # farthest-next-use (exact Belady on the frozen trace)
        ...
    def _evict_one(self, exclude=None):
        v = max((x for x in self._entries.values() if x.state is RESIDENT and x is not exclude),
                key=self._next_use_distance)
        ev = torch.cuda.Event(); ev.record()          # after all launches that consumed v (single stream)
        self._quarantine.append((v.buf, ev, (v.dtype, v.shape))); v.buf = None; v.state = ABSENT
    def _sweep_quarantine(self):                      # ring-return buffers whose event.query() is True
        ...
```

Registration walk (end of `apply_lf_asym_lora`, env-gated) + the eager split:

```python
store = get_nvme_store()
if store is not None and store.has_role("base_weight"):
    if _truthy(os.environ.get("ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD", "")):
        for m in model.modules():                    # BEFORE spilling: force the lazy fused→gate/up split
            if isinstance(m, AsymQwen3Experts): m._ensure_qwen3_moe_finegrained_bases()
    pager = BaseWeightPager(store)
    for name, mod in model.named_modules():
        hw = getattr(mod, "host_weight", None)
        # eligible: AsymFrozenLinear (attn + mlp_dense) + AsymGroupedFrozenLinear (experts);
        # EXCLUDED: embeddings (CPU-side F.embedding per microbatch) + norms (tiny).
        if isinstance(hw, HostWeight) and _panvme_component_eligible(name, mod):
            pager.register(name, hw)
    model._asym_base_weight_pager = pager
```

Correctness anchors: fwd+bwd interleaved trace ⇒ farthest-next-use is exact Belady (late layers, reused first in backward, survive after forward); event-gated quarantine per 0.3's async-launch rule; `touch` dedupe (`.weight` read multiple times per Function); reporting never fetches; step-1 is miss-driven by design (WARMUP_STEPS≥3); `precision=="bf16"` asserted at registration (quantized cache builds from `.weight`, `frozen_linear.py:372` — out of scope).

### Validation (Stage 4 gate)

Unit (`tests/training/test_base_weight_pager.py`): register frees `_tensor`; roundtrip bit-exact (2D + grouped 3D bf16); freeze at 3rd first-key occurrence with each key twice; farthest-reuse eviction; quarantine blocks reuse until `ev.query()`; jitter tolerance; disable fallback correct; NVMe-off → HostWeight byte-identical (existing `test_cpu_resident_frozen_base.py` + `test_lf_qwen3_asym_backend.py` unmodified).

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage4_panvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=3 MAX_STEPS=8 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false || q3-32b|1 ; asym_cpuadamwds_panvme|recomp-off-full-fg|ligerloss1 ; 8192|8|1 ; none|false|false|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true

.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline <asym_cpuadamwds dir> --candidate <asym_cpuadamwds_panvme dir> \
  --target base_weight_cpu \
  --memory-metric step_samples.training_step_process_rss_peak_end_bytes \
  --min-memory-drop-gib 40 --max-step-ratio 1.05 --max-forward-ratio 1.05 --max-backward-ratio 1.05 \
  --expect-nvme-role base_weight
```

Accept: per-step RSS −40 GiB+ (`memory_attribution` host_weight/cpu rows shrink to match); HBM unchanged; ≤5%; `misses_after_freeze ≈ 0`; `trace_disabled == false`; `bytes_read ≈ 2×base×steps`; losses match baseline.

### Risks / watch
- panvme + `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD` must hard-error at registration until the eager-split path is tested.
- If ≤5% fails at short seq (120 GiB/step vs fast steps), rerun at a longer-seq point and reclassify panvme capacity-mode for short seq.
- Grouped-expert blobs are GB-scale — assert `cache_bytes ≥ 2×largest_padded_nbytes` at registration.
- Any future new `.weight` reader: compute path (fine) vs reporting path (must use metadata) — audit.

---

## Stage 5 — actnvme coverage: attention U/S + MoE expert engines

The governor + manager hooks from Stage 3 are engine-agnostic; coverage = adding **one `on_seal(...)` call per engine Function** and auditing its forward consumers, behind `ASYM_NVME_ACT_ENGINES=dense_fg[,attention][,qwen3_moe][,llama4]`:

- **Attention** (`_AsymActivationOffloadLoRALinearFunction.forward`): seal `s_handle` at the end of each Function; the **shared q/k/v U** seals at the context's cache-clear point (`attention_activation_offload.py:477-479` — the structural "all q/k/v consumers enqueued" marker); o_proj's own U seals in its Function.
- **MoE expert engine / llama4 / shared-MLP**: `manager.seal(x_cpu, gate_cpu, up_cpu, ...)` as the last forward statement, one engine at a time, each audited for "is this after the last forward consumer".
- An unsealed handle never spills ⇒ partial coverage degrades to status quo, never corrupts.

Gate: rerun the Stage-3 command pair with each engine enabled — additional `cpu_peak_by_tag` collapse for that engine's tags at the same thresholds; one `q3-30b-a3b` MoE run before claiming MoE support.

---

## Stage 6 — `bothnvme`: compose + hero max-seq

No new mechanism: both roles on the one store. Startup assert: `base_weight cache + rings + governor budget + inflight caps < host RAM headroom`.

Gate: on q3-32b (then q3-30b-a3b) demonstrate max trainable seq `bothnvme > actnvme ≥ baseline` at fixed batch; verify real `input_ids` lengths; report HBM peak, per-step RSS, per-role NVMe bytes + wait-ms, step time, overlap fraction (`1 − nvme_wait/step`). Compare against the SuperOffload/zero3 ceilings for the paper's capability table.

---

## Deferred
- DeepSpeed-owned backend behind the same store API (ownership exclusive with `HostWeight`/`AsymCPUAdamW`); per-rank multi-GPU = rank-suffixed arenas + all ranks pread the same read-only base-weight files.
- GDS/GPU-direct (hardware absent); Unsloth-GC boundary offload (LlamaFactory-side); gradient/optimizer NVMe (LoRA-tiny); per-layer `saved_tensors_hooks` wrapper spill (dropped in v4 — revisit only if a layer-GC config becomes a target again).

## Global run rules (every e2e gate)
Heavy offload runs **sequentially** (665–802 GiB RSS observed); stop with `kill -TERM`, never `-9` (corrupts the DeepSpeed cpu_adam JIT cache); `PREPARE_DATASETS=true` on first use of a workload and **verify real `input_ids` length in `train.log`**; measure from `step_samples.csv` measured rows; `.aioenv` env is exported by `run_lf_lora_sft.sh` — export it manually for direct pytest/python store use.

## Implementation order = stage order

```text
Stage 0: NVMe traffic census (postprocess-only)      (per-layer + model read/write per step, feasibility,
                                                      overflow-vs-budget table — REVIEW BEFORE ANY CODE)
Stage 1: nvme_store.py substrate                     (isolated; unit + AIO smoke gate)
Stage 2: tokens + env + counters + compare           (paired e2e no-change gate)
Stage 3: actnvme — watermark governor on the fg path (capacity gate ≤1.10 + max-seq demo)  ← core target
Stage 4: panvme                                      (throughput gate ≤1.05)
Stage 5: actnvme coverage (attention / MoE engines)  (per-engine seal rollout, Stage-3 gate)
Stage 6: bothnvme hero                               (max-seq capability table)
```

Why actnvme first: it is the core research target; Stages 1+2 already de-risk the substrate; the governor + one seal line is a smaller, safer diff than the panvme pager; and it fails-fast on the capacity thesis. panvme's CPU-freeing is only *required* by the Stage-6 compound hero.

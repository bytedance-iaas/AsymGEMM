# AsymGEMM NVMe Offload — Implementation Plan (v3.1, code-verified 2026-07-01)

Three composable, opt-in backend tokens on ONE local NVMe store that reuses DeepSpeed's AIO engine:

- `asym_cpuadamwds_actnvme` — offloaded forward activations (CPU activation homes) → NVMe. *Novel*; raises the CPU-capacity ceiling that binds max seq. Capacity/hero mode. **Implemented first (core target).**
- `asym_cpuadamwds_panvme` — base/frozen weights (`HostWeight` CPU homes) → NVMe, bounded pinned cache + trace prefetch. DeepSpeed-parity idea, *not novel*; frees ~60 GiB CPU on q3-32b.
- `asym_cpuadamwds_bothnvme` — both roles; compound for the max-seq hero result.

Stages are numbered **in execution order**: 1 substrate → 2 wiring/gates → 3 actnvme-A → 4 panvme → 5 actnvme-B → 6 bothnvme.

Dropped (re-verified): optimizer-state and LoRA-weight-home NVMe — `AsymCPUAdamW` holds fp32 master+moments **only for LoRA params** (`cpu_adam.py:126-129,194,212-218`); LoRA homes are small pinned bf16 slabs (`weight_offload.py:174-182`). Nothing to save.

All facts below re-verified line-by-line on 2026-07-01 (incl. an on-box AIO smoke test). **Where any earlier doc version disagrees, this file wins.**

---

## 0. Verified ground truth (read before implementing)

### 0.1 DeepSpeed AIO — proven working on this box

```bash
# .aioenv sidecar REQUIRED for JIT build + runtime (run_lf_lora_sft.sh:29-40 already exports it).
export AIO_HOME=$PWD/.aioenv
export CPATH="$AIO_HOME/include:${CPATH:-}" LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
```

Empirically confirmed (2026-07-01): `AsyncIOBuilder().is_compatible()==True` **only** with `.aioenv` (JIT fails without); builds ~23 s; `aio_handle(1048576, 16, False, True, 4)` → `get_alignment()==2048` (=`intra_op_parallelism*512`); pinned pwrite/pread roundtrip **and offset-based roundtrip into one file both pass**. These are the same knobs the repo's `zero3_offload_panvme` baseline runs (`ds_z3_offload_panvme_config.json`: block 1MB / qd 16 / thread_count 4).

Hard API facts (`csrc/aio/py_lib/`):
- ctor `aio_handle(block_size, queue_depth, single_submit, overlap_events, intra_op_parallelism)` (`py_ds_aio.cpp:23`); `async_pread(buffer, filename, file_offset=0)` / `async_pwrite(buffer, filename|fd, file_offset=0)` (`py_ds_aio.cpp:87-112`); `wait()` releases the GIL (`:125-128`); `get_alignment()`/`get_intra_op_parallelism()` bound (`:35-36`).
- `wait()` drains **ALL** pending ops on that handle and `assert(_num_pending_ops > 0)` — **calling wait() idle aborts the process** (`deepspeed_py_io_handle.cpp:201-220`). Keep Python-side pending ledgers; **separate read and write handles** so a read drain never waits on writes.
- `num_bytes % intra_op_parallelism` must be 0 or the op is rejected (`:222-233`). Files are O_DIRECT (`deepspeed_aio_common.cpp:269`). ⇒ **pad every file offset and IO length to `get_alignment()`**, and IO buffers must be *allocated* at padded size (a padded-length pwrite from an exact-size buffer reads out of bounds).
- Unpinned/CUDA buffers silently **bounce** through an internal managed pinned buffer, full extra copy (`deepspeed_cpu_op.cpp:25-31,86-105`) — correct but slow; keep the common path pinned (zero-copy).
- Intra-op parallelism = each single pread/pwrite is split across N C++ threads at `file_offset + tid*chunk` (`deepspeed_cpu_op.cpp:68-84`), each thread a qd-deep libaio ring → up to `threads×qd×block` in flight *per op*.
- Reusable as-is: constants `MIN_AIO_BYTES=1MiB`, `AIO_ALIGNED_BYTES=1024` (`runtime/swap_tensor/utils.py:15-16`). NOT reusable: `AsyncPartitionedParameterSwapper` + `offload_param.device=nvme` (need `ds_id`/`ds_tensor`/`deepspeed.comm`, built only inside `zero.Init` — `partitioned_param_swapper.py:76-161`, `partition_parameters.py:1081-1095`); `SwapBufferManager` (calls `dist.get_rank()`, `utils.py:195`); `GDSBuilder` (needs cufile). We use torch's caching pinned allocator + our own rings instead of `SwapBufferPool`.

### 0.2 actnvme target — two substrates, different difficulty

**Substrate A (`saved_tensors_hooks` wrappers) — simple, and the layer-GC bulk.** Three near-identical wrappers: `DecoderSavedTensorOffloadWrapper` (`decoder_activation_offload.py:99`; `_pack` `:188-222`, `_unpack` `:224-248`), `AttentionSavedTensorOffloadWrapper` (`attention_activation_offload.py:187`), `LinearAttentionSavedTensorOffloadWrapper` (`linear_attention_activation_offload.py:99`). `DecoderLayerGlueGCWrapper` "custom" mode reuses the decoder `_pack`/`_unpack` (`decoder_layer_glue_gc.py:238-246`).
- `_pack`: fresh **stride-preserving** pinned CPU tensor per saved tensor (`_empty_strided_cpu_like` `:71-83`; torch caching host allocator recycles — no pool), D2H `copy_(non_blocking)`, `ready_event` on current stream. Handle = **mutable** `@dataclass _SavedTensorOffloadHandle` (`:86-97`).
- `_unpack`: `ready_event.synchronize()` → fresh HBM `empty_strided` → sync copy. **The CPU buffer is touched only by the D2H copy (write) and the unpack copy (read). No CPU-side kernel ever consumes it** ⇒ spill is safe once `ready_event` fired; after the pwrite completes the buffer can be freed outright (unpack fetches into a NEW buffer).
- Filter: CUDA, dtype∈{bf16,fp16,fp32}, ≥1 MiB, not `nn.Parameter` (`:159-181`); decoder/linear `require_grad=False` default, attention `True`. Stats published per-module as `_last_activation_offload_stats` (`:280-281`).

**Substrate B (`ActivationOffloadManager`, `activation_offload.py`) — needs a seal-point protocol.** One manager per autograd-Function call; ONE global pinned pool `_CPU_BUFFER_POOL` keyed `(dtype, shape, pinned)` (`:10,74-103`; cap `ASYM_EXPACT_CPU_POOL_MAX_BYTES`, default 32 GiB `:13`).
- `CPUActivationHandle` is `@dataclass(frozen=True)` (`:106-116`); accounting keyed by `int(handle.tensor.data_ptr())` (`:164-166`). Handles never used as dict keys → side-table by `id(handle)` is safe.
- **Handles are consumed on CPU by async GPU kernels in forward**: `_dense_lora_a_cpu_left(u_handle.tensor, ...)` reads offloaded U right after `offload()` (`attention_activation_offload.py:619-625`); backward CPU-right kernels read U/S again. q/k/v share one U via refcounted `_SharedActivationSource` (`:417-441`), released at the v_proj cache-clear point (`:477-479`).
- ⇒ pool-return of a Substrate-B buffer needs: D2H `ready_event` fired AND all enqueued GPU kernels streaming it complete (event-gated). Today's safety is pure single-stream ordering; an IO thread breaks it → we add events.
- Un-spill points: `stage()/stage_rows()/stage_concat_columns()` (`:237-299`); pool return: `release_cpu()` (`:315-326`).

**Neither substrate has a copy stream or prefetch today** (only `weight_offload.py:95` holds an unused `_stream`). All overlap infra is new.

### 0.3 panvme target — base weights

- Single tensor-level chokepoint: `HostWeight.__init__` (`asym_gemm/training/host_weight.py:185-242`, pin `:222`). `.weight`/`.tensor` are **properties over `self._tensor`** (`:244-246,302-304`) → clean lazy-materialize interception. `metadata`/`nbytes` (`:283-284`) are copy-free — reporting paths must use them, never the tensor.
- Adoption entries: `adopt_host_weight()` (`offload.py:176-213`, called from `lf.py:1192` in `_wrap_lf_linear_leaf`) and `AsymGroupedFrozenLinear.__init__` (`frozen_linear.py:1624`, grouped experts). Embeds/norms via `offload.py:352,394,441` (excluded by default policy — embed is consumed by CPU-side `F.embedding` `offload.py:381` every microbatch; norms are tiny).
- Consumption: dense fwd `frozen_linear.py:1307-1318`; dense dX `:1353-1365`; grouped dX `qwen3_moe.py:462-486`; grouped fwd `frozen_linear.py:729-754`; attention act-offload base fwd `attention_activation_offload.py:599-608`; native windowed bwd `qwen3_moe.py:1878`.
- **Kernel launches are async**: `asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(...)` returns immediately (`frozen_linear.py:723-726`) while the GPU streams the CPU-pinned weight ⇒ a weight buffer may be recycled **only after a CUDA event recorded post-enqueue completes**.
- Runtime `.weight` readers besides the GEMMs: `is_pinned()` predicates (`qwen3_moe.py:1760,2648-2654`, `llama4_moe.py:242`) — each sits immediately before a consuming kernel, so lazy fetch is self-consistent; and `_ensure_qwen3_moe_finegrained_bases` (`qwen3_moe.py:2492-2515`) **lazily slices the fused gate_up weight into two NEW AsymGroupedFrozenLinear** — under panvme this must run eagerly at conversion (see Stage 4 risks).
- After expert adoption the HF source params are shrunk to 0-numel (`qwen3_moe.py:2127-2136`) — HostWeight owns the only copy; spilling genuinely frees RSS.

### 0.4 Backend tokens — dispatch sites (exact-token case arms; no glob collision with `zero3_offload_panvme`)

Grammar: `model|gpus ; backend|recompute|liger[|kernelcode] ; seq|batch|grad_accum ; policy|expact|attnact|layeract|layergc|sdparecomp`. Policy-list env override is **`ASYMM_EXP_ACT_POLICIES`** (`profile_lora_lf_test_source.sh:2122`; CLI `--asymm-exp-act-policies` `:2157`).

Sites (all extended in Stage 2): `profile_lora_lf_test_source.sh` — `append_backend_spec` `:956-978`, `backend_gpu_count` `:789-795`, `cpuadam_backend_for_label` `:918-923`, per-job derivation `~:3094-3110`, `run_env`/`ASYM_GEMM_LF_CONFIG_*` block `:3343-3387`; the backend token is the first `path_label` component (`job_root_path` `:1733-1741`) → run dirs auto-disambiguate, **no label tag needed**; completion checks receive `backend=$3` (`:1104,:1148`). `run_lf_lora_sft.sh` — main case `:352-358` + die `:369`; `is_zero_backend_run` requires `BACKEND==torch` (`:659-661`) → our arms set `BACKEND=asym`, no `--deepspeed` leakage; `.aioenv` exports `:29-40`; `RECORD_IO` `/sys/block` sampler `:187,2417-2430` is backend-agnostic (free cross-check). `run_lf_profiled_train.py` — classification `:577-599` (`is_asym_deepspeed_cpuadamw` `:579-582`), `_config_from_args` `:546` (env-mirror pattern `:732-734`). **No LlamaFactory changes** — config rides env like every ASYMM_* feature (e.g. `ASYMM_ATTN_ACT_OFFLOAD` read at `lf.py:1322-1325`).

### 0.5 Profiling / gating infra

- Canonical driver `scripts/lf/profile_lora_lf_test_source.sh` (`PROFILERS` default `source` `:131`; shorthands `M[q3-32b]="Qwen/Qwen3-32B"` `:34-46`; `PREPARE_DATASETS` `:204`; CLI `--gpus/--overwrite/--prepare-datasets`). `_both.sh` = same driver, nsys default; apply identical edits.
- `source_profile.json` from `report()` (`run_lf_profiled_train.py:2849-2919`). `activation_offload` block from `_activation_offload_counters_from_model()` (`:2198-2282`): per-module `_last_activation_offload_stats` dicts flow into `rows` **automatically** (new snapshot() keys propagate for free — Stages 3/5 exploit this), but the **aggregates** at `:2265-2281` are explicit and must be extended. Host RSS: `memory.process.rss_peak_bytes` = VmHWM (lifetime peak, load-transient-polluted); **per-step RSS** = `step_samples.rows[].training_step_process_rss_peak_end_bytes` — use for steady-state gates. `memory_attribution.rows[]` has `category=host_weight, device=cpu, bytes, pinned_bytes`.
- Compare-gate template: `scripts/lf/compare_liger_loss_profiles.py` (args `:33-42`, `load_memory_metrics` `:156`, median step/fwd/bwd from `step_samples.csv` `:208-235`, `{"ok":...}` + `SystemExit(2)`). Cloned in Stage 2. Eyeball: `scripts/lf/show_metrics.py`.

### 0.6 Prefetch design cross-check (DeepSpeed / TE / Megatron, audited 2026-07-01)

Explored: DeepSpeed ZeRO-3 `PartitionedParameterCoordinator` + NVMe swapper + pipelined optimizer swapper; TE `cpu_offload.py`/`cpu_offload_v1.py`; Megatron-core `fine_grained_activation_offload.py`. Verdicts:

- **DeepSpeed has NO activation prefetch.** Its `cpu_checkpointing` restore is a synchronous `.to(device)` at recompute time; the `transport_stream` is dead commented-out code (`checkpointing.py:613-649`). The famous DS prefetcher is the *param* coordinator — our activation prefetcher has no DS incumbent.
- **Our design already matches DS's key param-prefetch patterns:** correctness never depends on prefetch (their demand-fetch + inflight registry ≡ our `drain_reads` ledger + `durable` events); FIFO spill in creation order with in-flight byte backpressure (their `AsyncTensorSwapper` rotating buffers ≡ our writer cap); trace lifecycle RECORD→COMPLETE→INVALID (their `partitioned_param_coordinator.py:44-50,180-204` ≡ our panvme freeze/jitter/disable); release-by-reuse-distance + persistence threshold (`:572-604`, `parameter_offload.py:287-305` ≡ our Belady eviction + `min_swappable_bytes`); split read/write aio handles (their swapper does the same, `partitioned_param_swapper.py:111-121`).
- **Adopted upgrade 1 — byte-budgeted lookahead (from DS).** DS bounds prefetch by `min(prefetch_bucket_sz, max_live − live)` *elements*, not a fixed count (`coordinator:428-433`). Activation sizes vary wildly → Stage 3's prefetch v2 uses a byte budget (`ASYM_NVME_ACT_PREFETCH_BYTES`), not a count.
- **Adopted upgrade 2 — hot tail window (from Megatron/DS reuse-distance).** Megatron never offloads the *last* group per name — its reload would stall backward immediately (`fine_grained_activation_offload.py:539-554,962-984`). Our analog: a deferred-spill window of the most recently packed bytes (`ASYM_NVME_ACT_HOT_WINDOW_BYTES`) — the forward tail stays CPU-resident, so backward's first unpacks are CPU hits while the read pipeline warms, and short-lived tensors never round-trip disk at all.
- **Noted, deferred:** TE v1/Megatron run D2H and H2D on two dedicated CUDA streams with double-buffered reload (`cpu_offload_v1.py:366-367,578`); our substrates keep GPU copies on the compute stream (existing behavior) — a side-stream H2D staging upgrade is an *existing-substrate* optimization independent of NVMe, listed under Deferred. One stronger property we have that DS lacks: our activation order is **recorded in the same step it is consumed** (exact by construction), vs DS's cross-step trace assumption.

### 0.7 Hardware & feasibility

4× Samsung PM9A3 3.84 TB RAID0 → `/scratch_local` (~26 GB/s read, ~14 GB/s write aggregate). `ASYM_NVME_PATH` defaults there.
- actnvme: spill written once + read once per step; **write side (~14 GB/s) binds**. Feasibility precheck (before Stage 3, from existing profiles): `activation_offload` offloaded bytes/step ÷ step seconds ≲ 14 GB/s at the intended operating point, else the writer backpressures by that ratio → pick longer seq. Capacity mode: ≤10% gate + max-seq demo.
- panvme q3-32b: ~60 GiB base read fwd+bwd = ~120 GiB/step → 4-6 GB/s sustained at 20-30 s steps — overlappable; ≤5% gate.
- Endurance: ~1 DWPD class (~7 PBW/drive, ~28 PBW RAID). Research volumes = non-issue.

---

## Design contract (all stages)

1. **One store, role-tagged.** `NVMeStore` serves `{base_weight, activation}`; tokens map `actnvme→{activation}`, `panvme→{base_weight}`, `bothnvme→both`. Consumers see only the store API (`register/spill/fetch/prefetch`) — the swappable seam (local NVMe now; per-rank or DeepSpeed-owned later). **Placement is deliberately NOT abstracted** — it stays in the consumers (it *is* AsymGEMM's algorithm).
2. **No kernel computes from NVMe.** Always `NVMe → pinned CPU → (H2D) → compute`, `pinned CPU → NVMe`. Kernels receive exactly what they receive today.
3. **Compute shape untouched.** No change to any GEMM, grouping, slab, or launch pattern. IO units are whole tensors (≥1 MiB) — never per-expert or per-chunk loops.
4. **Single-owner AIO handles.** Write handle owned by ONE writer thread; read handle by the main thread. Python pending ledgers; never `wait()` idle.
5. **Event-gated buffer reuse.** Any pinned buffer a GPU kernel may still stream is recycled only after a post-enqueue CUDA event completes. (Single-stream ordering protects the status quo; the IO thread breaks it; events restore it.)
6. **Off = byte-identical.** No `*nvme` token ⇒ no deepspeed import, no AIO build, no thread, no file, identical allocations; every hook is a `None` check.
7. Every file offset and IO length padded to `store.align`; IO buffers allocated at padded size.

---

## Stage 1 — `NVMeStore` substrate

**Scope:** NEW `asym_gemm/training/nvme_store.py` + NEW `tests/training/test_nvme_store.py`. Zero edits to existing files ⇒ isolated unit gate is sufficient (the one exception to the e2e rule).

### Implementation

```python
# asym_gemm/training/nvme_store.py
from __future__ import annotations
import os, queue, threading, time
from dataclasses import dataclass, field
from typing import Any, Callable
import torch

# ---------- config ----------
@dataclass(frozen=True)
class NVMeStoreConfig:
    path: str
    roles: frozenset[str]                        # subset of {"base_weight", "activation"}
    aio_block_size: int = 1 << 20                # ASYM_NVME_AIO_BLOCK_SIZE
    aio_queue_depth: int = 16                    # ASYM_NVME_AIO_QUEUE_DEPTH
    aio_intra_op_parallelism: int = 4            # ASYM_NVME_AIO_INTRA_OP_PARALLELISM
    aio_single_submit: bool = False
    aio_overlap_events: bool = True
    min_swappable_bytes: int = 1 << 20           # ASYM_NVME_MIN_SWAPPABLE_BYTES
    activation_arena_bytes: int = 1 << 40        # ASYM_NVME_ACTIVATION_ARENA_BYTES (sparse file)
    max_inflight_spill_bytes: int = 8 << 30      # ASYM_NVME_MAX_INFLIGHT_SPILL_BYTES (writer backpressure)

def _config_from_env() -> NVMeStoreConfig | None:
    roles = frozenset(r.strip() for r in os.environ.get("ASYM_NVME_ROLES", "").split(",") if r.strip())
    if not roles:
        return None
    bad = roles - {"base_weight", "activation"}
    if bad: raise ValueError(f"unknown ASYM_NVME_ROLES entries: {sorted(bad)}")
    path = os.environ.get("ASYM_NVME_PATH") or _fail("ASYM_NVME_PATH required when ASYM_NVME_ROLES is set")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1: _fail("asym NVMe store is single-process only (for now)")
    return NVMeStoreConfig(path=path, roles=roles, **_ints_and_bools_from_env())

# ---------- helpers ----------
def _pad(n: int, a: int) -> int: return (n + a - 1) // a * a

def _flat_u8(t: torch.Tensor) -> torch.Tensor:
    """uint8 alias of t's WHOLE storage (strided/padded safe; zero-copy). IO always uses this."""
    out = torch.empty(0, dtype=torch.uint8)
    out.set_(t.untyped_storage(), 0, (t.untyped_storage().nbytes(),))
    return out

def required_storage_nbytes(shape, stride, dtype) -> int:
    if 0 in shape: return 0
    last = sum((s - 1) * st for s, st in zip(shape, stride)) + 1
    return last * torch.empty(0, dtype=dtype).element_size()

def alloc_padded_pinned(shape, dtype, stride=None, *, align) -> torch.Tensor:
    """Pinned CPU tensor with PADDED backing storage; returned view has exact shape/stride.
    Goes through torch's caching host allocator (cheap after warmup)."""
    stride = stride or _contiguous_strides(shape)
    storage = torch.empty(_pad(required_storage_nbytes(shape, stride, dtype), align),
                          dtype=torch.uint8, pin_memory=True)          # falls back unpinned on failure (aio bounces)
    t = torch.empty(0, dtype=dtype)
    t.set_(storage.untyped_storage(), 0, shape, stride)
    return t

# ---------- blob ref ----------
@dataclass
class BlobRef:
    role: str
    file: str            # per-blob file (base_weight) or the role arena file (activation)
    offset: int          # aligned; 0 for per-blob files
    length: int          # padded bytes on disk == source buffer storage nbytes
    logical_nbytes: int
    durable: threading.Event = field(default_factory=threading.Event)

@dataclass
class NVMeStoreStats:      # all keys surface in the profile's `asym_nvme` block
    bytes_written: dict = field(default_factory=dict)   # per role
    bytes_read: dict = field(default_factory=dict)
    write_ops: dict = field(default_factory=dict)
    read_ops: dict = field(default_factory=dict)
    fetch_wait_ms: float = 0.0
    spill_backpressure_ms: float = 0.0
    inflight_peak_bytes: int = 0
    arena_peak_bytes: dict = field(default_factory=dict)
    def as_dict(self) -> dict[str, Any]: ...

# ---------- writer thread (sole owner of the write handle) ----------
_STOP = object()

class _WriterThread(threading.Thread):
    def __init__(self, handle, cfg: NVMeStoreConfig, stats: NVMeStoreStats):
        super().__init__(name="asym-nvme-writer", daemon=True)
        self._h, self._cfg, self._stats = handle, cfg, stats
        self._q: queue.Queue = queue.Queue()
        self._inflight = 0
        self._cv = threading.Condition()

    def submit(self, ready_event, buf_u8, ref: BlobRef, on_done: Callable) -> None:
        t0 = time.perf_counter()
        with self._cv:
            while self._inflight > self._cfg.max_inflight_spill_bytes:   # backpressure: caller blocks
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
                ready_event.synchronize()                # D2H copy done → source bytes stable
            self._h.async_pwrite(buf, ref.file, ref.offset)   # buf.nbytes == ref.length (padded)
            n = self._h.wait()                           # sole owner → exactly this op
            assert n == 1
            self._stats.bytes_written[ref.role] = self._stats.bytes_written.get(ref.role, 0) + buf.nbytes
            self._stats.write_ops[ref.role] = self._stats.write_ops.get(ref.role, 0) + 1
            ref.durable.set()
            with self._cv:
                self._inflight -= buf.nbytes
                self._cv.notify_all()
            on_done(buf, ref)                            # writer-thread context: drop/recycle buf; NEVER touch CUDA

# ---------- store ----------
class NVMeStore:
    def __init__(self, cfg: NVMeStoreConfig):
        from deepspeed.ops.op_builder import AsyncIOBuilder     # imported ONLY when enabled
        m = AsyncIOBuilder().load(verbose=False)
        mk = lambda: m.aio_handle(cfg.aio_block_size, cfg.aio_queue_depth, cfg.aio_single_submit,
                                  cfg.aio_overlap_events, cfg.aio_intra_op_parallelism)
        self.cfg = cfg
        self._read_h, write_h = mk(), mk()               # separate handles: wait() drains per-handle
        self.align = int(self._read_h.get_alignment())   # 2048 @ intra=4 (measured); satisfies %intra too
        self.stats = NVMeStoreStats()
        self._writer = _WriterThread(write_h, cfg, self.stats); self._writer.start()
        os.makedirs(os.path.join(cfg.path, "base_weight"), exist_ok=True)
        self._arena_path = os.path.join(cfg.path, "activation.arena")
        self._arena_cursor = 0
        self._arena_live = 0
        self._pending_reads: dict[int, BlobRef] = {}     # id(ref) -> ref; MAIN THREAD ONLY

    def has_role(self, role: str) -> bool: return role in self.cfg.roles

    # ---- activation arena: bump allocator, reset-when-empty ----
    # Activation blob lifetime = within one microbatch fwd→bwd, so live==0 recurs each microbatch;
    # resetting the cursor there is exact and needs no trainer hooks (works under grad accumulation).
    def _arena_alloc(self, nbytes: int) -> tuple[str, int]:
        length = _pad(nbytes, self.align)
        off = self._arena_cursor
        self._arena_cursor += length
        self._arena_live += 1
        self.stats.arena_peak_bytes["activation"] = max(self.stats.arena_peak_bytes.get("activation", 0), self._arena_cursor)
        if self._arena_cursor > self.cfg.activation_arena_bytes:
            raise RuntimeError("activation arena full — raise ASYM_NVME_ACTIVATION_ARENA_BYTES")
        return self._arena_path, off

    def blob_done(self, ref: BlobRef) -> None:           # after the final fetch of an activation blob
        if ref.role == "activation":
            self._arena_live -= 1
            if self._arena_live == 0:
                self._arena_cursor = 0

    # ---- write path (any thread; enqueues to writer) ----
    def spill(self, role: str, tensor: torch.Tensor, *, ready_event, on_done) -> BlobRef:
        """tensor: pinned CPU with PADDED storage (from alloc_padded_pinned). Returns immediately.
        on_done(buf, ref) runs on the writer thread after durability — release the buffer there."""
        buf = _flat_u8(tensor)
        assert buf.nbytes % self.align == 0, "IO buffers must be padded (alloc_padded_pinned)"
        if role == "base_weight":
            file, off = self._per_blob_file(tensor), 0    # {path}/base_weight/{stable_id}.bin
        else:
            file, off = self._arena_alloc(buf.nbytes)
        ref = BlobRef(role, file, off, buf.nbytes, logical_nbytes=int(tensor.numel() * tensor.element_size()))
        self._writer.submit(ready_event, buf, ref, on_done)
        return ref

    # ---- read path (MAIN THREAD ONLY) ----
    def submit_pread(self, ref: BlobRef, dst_padded_pinned: torch.Tensor) -> None:
        if not ref.durable.is_set():
            ref.durable.wait()                           # write in flight → bounded, rare block (no cancel path: simpler, always correct)
        dst = _flat_u8(dst_padded_pinned)
        assert dst.nbytes == ref.length
        self._read_h.async_pread(dst, ref.file, ref.offset)
        self._pending_reads[id(ref)] = ref

    def drain_reads(self) -> set[int]:
        """Blocks until ALL pending reads complete (wait() drains the whole handle — A1 rule).
        Returns the id(ref) set that arrived; callers flip their own state on it."""
        if not self._pending_reads: return set()
        n = self._read_h.wait()
        assert n == len(self._pending_reads)
        done = set(self._pending_reads.keys())
        for r in self._pending_reads.values():
            self.stats.bytes_read[r.role] = self.stats.bytes_read.get(r.role, 0) + r.length
            self.stats.read_ops[r.role] = self.stats.read_ops.get(r.role, 0) + 1
        self._pending_reads.clear()
        return done

    def fetch_into(self, ref: BlobRef, dst_padded_pinned: torch.Tensor) -> None:   # sync convenience
        t0 = time.perf_counter()
        self.submit_pread(ref, dst_padded_pinned)
        self.drain_reads()
        self.stats.fetch_wait_ms += (time.perf_counter() - t0) * 1e3

# ---------- singleton ----------
_STORE: NVMeStore | None = None
_STORE_INIT = False
def get_nvme_store() -> NVMeStore | None:
    """Lazy env-driven singleton. Without ASYM_NVME_ROLES: returns None, imports nothing, allocates nothing."""
    global _STORE, _STORE_INIT
    if not _STORE_INIT:
        cfg = _config_from_env()
        _STORE = NVMeStore(cfg) if cfg is not None else None
        _STORE_INIT = True
    return _STORE
```

Locked decisions: **base_weight = one file per HostWeight** (50 MB–GB blobs, written once — zero offset bookkeeping, no metadata overhead at that size); **activation = one arena file** + bump/reset-when-empty (thousands of transient blobs/step — per-blob files would hammer fs metadata); **no cancel path** (fetch-before-durable just waits — simpler, always correct); writer waits per-op (each op is already 4-thread×qd16 internally — batch-submit is a flagged follow-up if the writer starves).

### Efficiency
Per-op internal parallelism 4×16×1MB; inter-op overlap via writer thread (spills overlap compute; GIL released in waits) + multi-pread prefetch ledger; zero extra memcpys (pinned direct O_DIRECT; the padded pinned buffer IS the D2H destination and the IO buffer); no locks on hot paths (handles single-owner).

### Validation (Stage 1 gate)

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export AIO_HOME=$PWD/.aioenv CPATH="$AIO_HOME/include:${CPATH:-}" \
       LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
ASYM_NVME_PATH=/scratch_local/user_data/shutian/kevin/cache/asym_nvme_test \
.venv/bin/python -m pytest tests/training/test_nvme_store.py -q
```

Required tests: bf16/fp32 roundtrip below/at/above 1 MiB (logical bytes exact); **strided** tensor roundtrip restores exact strides; two arena blobs at different offsets, no cross-blob corruption; spill of a D2H-copied tensor gated on `ready_event` (write a CUDA tensor, record event, spill, fetch, compare); fetch-before-durable blocks then succeeds; 3-deep prefetch ledger reconciles (`drain_reads` returns all ids); arena reset-when-empty across two simulated microbatches; backpressure blocks at the cap and resumes; `get_nvme_store()` is `None` and `deepspeed` absent from `sys.modules` when env unset; writer clean shutdown.

### Risks / watch
- Handle thread-ownership is a rule, not enforced by the API — assert thread identity in debug mode.
- Arena overflow at extreme seq×accumulation → loud error + env knob; revisit if hit.

---

## Stage 2 — Backend tokens, env plumbing, profile counters, compare gate

**Scope:** `scripts/lf/profile_lora_lf_test_source.sh` (+ identical edits in `_both.sh`), `scripts/lf/run_lf_lora_sft.sh`, `scripts/lf/run_lf_profiled_train.py`, `scripts/lf/postprocess_lf_profile_artifacts.py`, NEW `scripts/lf/compare_nvme_profiles.py`. **No tensors move in this stage.**

### Implementation

**(a) `profile_lora_lf_test_source.sh`** — three exact-token arms in `append_backend_spec` (`:956-978`) + die list:

```bash
    asym_cpuadamwds_panvme) backend=asym_cpuadamwds_panvme ;;
    asym_cpuadamwds_actnvme) backend=asym_cpuadamwds_actnvme ;;
    asym_cpuadamwds_bothnvme) backend=asym_cpuadamwds_bothnvme ;;
```

`backend_gpu_count` (`:789-795`): append the three tokens to the 1-GPU asym line + die list. `cpuadam_backend_for_label` (`:918-923`):

```bash
    asym_cpuadamwds|asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme) printf 'deepspeed\n' ;;
```

(this alone makes the per-job derivation at `~:3094-3110` set `job_use_asym_cpu_adamw=true` correctly). New helper + `run_env` additions next to `:3343`:

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
  run_env+=(
    "ASYM_NVME_ROLES=${job_nvme_roles}"
    "ASYM_NVME_PATH=${ASYM_NVME_PATH:-/scratch_local/user_data/shutian/kevin/cache/asym_nvme}"
    "ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES=${job_nvme_roles}"
    "ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH=${ASYM_NVME_PATH:-/scratch_local/user_data/shutian/kevin/cache/asym_nvme}"
  )
fi
```

**(b) `run_lf_lora_sft.sh`** — one grouped arm cloned from `asym_cpuadamwds` (`:352-358`) + die list (`:369`):

```bash
  asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme)
    PROFILE_BACKEND_LABEL=${PROFILE_BACKEND_LABEL:-${BACKEND,,}}
    USE_ASYM_CPU_ADAMW=true
    ASYM_CPU_ADAMW_BACKEND=deepspeed
    CPUADAM_ALIAS_SELECTED=1
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

Mirror `ASYM_GEMM_LF_CONFIG_ASYM_NVME_{ROLES,PATH}` next to `:2297`.

**(c) `run_lf_profiled_train.py`** — classification (`:579-582`):

```python
_ASYM_CPUADAMW_DS_BACKENDS = {"asym_cpuadamwds", "asym_cpuadamwds_panvme",
                              "asym_cpuadamwds_actnvme", "asym_cpuadamwds_bothnvme"}
is_asym_deepspeed_cpuadamw = backend in _ASYM_CPUADAMW_DS_BACKENDS or (...)
```

`_config_from_args` (pattern of `:732-734`):

```python
"asym_nvme_roles": os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES") or os.environ.get("ASYM_NVME_ROLES", ""),
"asym_nvme_path":  os.environ.get("ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH")  or os.environ.get("ASYM_NVME_PATH", ""),
```

Report block (sibling of `activation_offload` at `:2846`):

```python
def _asym_nvme_summary_from_model() -> dict[str, Any]:
    try:
        from asym_gemm.training.nvme_store import get_nvme_store
    except Exception as exc:
        return {"enabled": False, "reason": f"import failed: {exc!r}"}
    store = get_nvme_store()
    if store is None:
        return {"enabled": False}
    out = {"enabled": True, "roles": sorted(store.cfg.roles), "path": store.cfg.path,
           "alignment": store.align,
           "aio": {"block_size": store.cfg.aio_block_size, "queue_depth": store.cfg.aio_queue_depth,
                   "intra_op_parallelism": store.cfg.aio_intra_op_parallelism},
           **store.stats.as_dict()}
    model, _ = _model_and_base_model()
    pager = getattr(model, "_asym_base_weight_pager", None)
    if pager is not None:
        out["base_weight_pager"] = pager.summary()
    return out
# report(): "asym_nvme": _asym_nvme_summary_from_model(),
```

Aggregates tail of `_activation_offload_counters_from_model` (`:2265-2281`) — add (rows carry the keys once Stages 3/5 land):

```python
"total_nvme_spilled_bytes":  sum(_i(r["activation_offload_stats"].get("nvme_spilled_bytes")) for r in rows),
"total_nvme_bytes_read":     sum(_i(r["activation_offload_stats"].get("nvme_bytes_read")) for r in rows),
"total_nvme_fetch_wait_ms":  sum(_f(r["activation_offload_stats"].get("nvme_fetch_wait_ms")) for r in rows),
"total_nvme_spill_backpressure_ms": sum(...),
```

**(d) postprocess** — `_asym_nvme_rows(profile)` (flatten `asym_nvme`: one row per role from `bytes_written/bytes_read/ops` dicts + scalars) written as `asym_nvme.csv` next to `_asym_cpu_adamw_rows` (`:378`); one NVMe summary line in `memory.md` (`_source_memory_markdown` `:1803`).

**(e) `compare_nvme_profiles.py`** — clone of `compare_liger_loss_profiles.py` with:

```text
--baseline DIR --candidate DIR
--target {no_change, base_weight_cpu, activation_cpu, maxseq}
--memory-metric DOTTED   # source_profile.json path; "step_samples.<col>" = median of measured csv rows
--min-memory-drop-gib X | --min-memory-drop-pct X   (no_change: --max-memory-drift-gib X)
--max-step-ratio 1.05 --max-forward-ratio 1.05 --max-backward-ratio 1.05
--expect-nvme-role ROLE  # asserts candidate asym_nvme.enabled, ROLE∈roles, bytes_written>0 and bytes_read>0 for ROLE
Checks: artifacts exist (source_profile.json, step_samples.csv, memory.md, asym_cpu_adamw.csv; asym_nvme.csv on candidates);
finite losses; measured_steps>=5; config.asym_nvme_roles matches. Output {"ok":...}; SystemExit(2) on failure.
```

### Validation (Stage 2 gate — paired e2e no-change)

```bash
bash -n scripts/lf/profile_lora_lf_test_source.sh scripts/lf/run_lf_lora_sft.sh
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage2_nochange PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds|norecomp|ligerloss1 ; 8192|8|1 ; none|false|true|false|true|true' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage2_token PROFILERS=source PLOT=false \
PREPARE_DATASETS=false WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds_actnvme|norecomp|ligerloss1 ; 8192|8|1 ; none|false|true|false|true|true' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline profiling_nvme/stage2_nochange/<run_dir> --candidate profiling_nvme/stage2_token/<run_dir> \
  --target no_change --memory-metric memory.gpu.peak_allocated_hbm_bytes --max-memory-drift-gib 0.5 \
  --max-step-ratio 1.02 --max-forward-ratio 1.02 --max-backward-ratio 1.02
```

Accept: drift/latency inside bounds; candidate `config.asym_nvme_roles=="activation"`, `asym_nvme.enabled==true` with `bytes_written==0` (Stages 3–5 not yet implemented); `asym_nvme.csv` present; the tool demonstrably fails on a mismatched pair.

### Risks / watch
- Apply identical arms to `_both.sh` (same driver).
- Cached run-dir reuse keys on backend `$3` → distinct; still pass `--overwrite true` on gate runs.

---

## Stage 3 — `actnvme` part 1: Substrate A (saved-tensor wrappers — the layer-GC bulk) ← FIRST tensor-moving stage

**Why first:** core research target; Stages 1+2 already de-risked the substrate; this is the *smallest* tensor-moving diff (no CPU-kernel consumers — 0.2); fails-fast on the capacity thesis. **Precheck before coding** (5 min, from any existing profile): `activation_offload` offloaded bytes/step ÷ step seconds ≲ 14 GB/s at the intended operating point; else pick longer seq.

**Scope:** identical ~30-line diff in `decoder_activation_offload.py`, `attention_activation_offload.py` (wrapper part only), `linear_attention_activation_offload.py` (deliberately NOT deduping the triplets — smallest behavioral diff). New shared helpers imported from `nvme_store`.

### Implementation (shown for `decoder_activation_offload.py`; twins identical)

```python
# handle: two new fields (dataclass already mutable)
@dataclass
class _SavedTensorOffloadHandle:
    ...
    spill_ref: Any = None          # BlobRef while the bytes live on NVMe

# NEW: allocation switch — padded+pinned ONLY when this tensor will spill (off-path allocations untouched)
def _alloc_cpu_for_pack(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    store = get_nvme_store()
    if store is not None and store.has_role("activation"):
        nbytes = _tensor_storage_nbytes(tensor)
        if nbytes >= store.cfg.min_swappable_bytes:
            return alloc_padded_pinned(tuple(tensor.shape), tensor.dtype,
                                       tuple(tensor.stride()), align=store.align), True
    return _empty_strided_cpu_like(tensor, pin_memory=self.pin_memory), False

# _pack (:188-222) — replace the alloc line and extend the tail:
def _pack(self, tensor):
    if not self._should_offload(tensor):
        return tensor
    cpu, spillable = self._alloc_cpu_for_pack(tensor)
    non_blocking = bool(cpu.is_pinned())
    with torch.no_grad():
        cpu.copy_(tensor.detach(), non_blocking=non_blocking)
    ready_event = None
    if non_blocking:
        ready_event = torch.cuda.Event(); ready_event.record(torch.cuda.current_stream(tensor.device))
    ... existing nbytes/tag/dtype/shape counter lines unchanged ...
    handle = _SavedTensorOffloadHandle(tensor=cpu, ..., ready_event=ready_event)
    if spillable:
        store = get_nvme_store()
        handle.spill_ref = store.spill("activation", cpu, ready_event=ready_event,
                                       on_done=lambda buf, ref: None)
        handle.tensor = None
        # ^ writer thread now holds the only strong ref to `cpu`; after the pwrite is durable the
        #   closure drops it → torch's caching host allocator recycles the pinned block. Safe because
        #   Substrate-A buffers have no consumer other than the D2H copy (gated by ready_event) —
        #   verified 0.2. _unpack never reads handle.tensor when spill_ref is set.
        self.nvme_spilled_bytes += nbytes
        self.cpu_owned_bytes -= nbytes        # residency moved to the bounded in-flight window
        self.cpu_bytes_by_tag[tag] -= nbytes  # (keep offloaded_bytes/tag counters as-is)
    self._sync_module_stats()
    return handle

# _unpack (:224-248) — fetch head; existing body unchanged after it:
def _unpack(self, packed):
    if not isinstance(packed, _SavedTensorOffloadHandle):
        return packed
    if packed.spill_ref is not None:
        store = get_nvme_store()
        bounce = alloc_padded_pinned(packed.original_shape, packed.original_dtype,
                                     packed.original_stride, align=store.align)
        t0 = time.perf_counter()
        store.fetch_into(packed.spill_ref, bounce)          # waits durability if needed; drains prefetches
        self.nvme_fetch_wait_ms += (time.perf_counter() - t0) * 1e3
        self.nvme_bytes_read += packed.nbytes
        store.blob_done(packed.spill_ref)                   # arena live-count → reset-when-empty
        packed.tensor, packed.spill_ref = bounce, None      # bounce freed when handle dies post-unpack
    if packed.ready_event is not None:                      # long fired for spilled handles; None-safe
        packed.ready_event.synchronize()
    ... existing empty_strided HBM + copy_ + counters unchanged ...
```

New `snapshot()` keys (auto-flow into profiler rows; aggregates added in Stage 2): `nvme_spilled_bytes`, `nvme_bytes_read`, `nvme_fetch_wait_ms`, plus store-level `spill_backpressure_ms`/`inflight_peak_bytes` in the `asym_nvme` block.

**Prefetch v2 (flag `ASYM_NVME_ACT_PREFETCH_BYTES`, default 0 = sync)** — byte-budgeted lookahead (DS pattern, 0.6): the wrapper appends each spilled handle to `self._pack_order`; backward unpacks in ~exact reverse. At each unpack of index *i*, walk backward from *i−1* allocating bounces and `submit_pread`-ing until the in-flight read bytes reach the budget; the next unpack's `fetch_into` drains them together. Order is recorded same-step (exact), and drain-all + `durable` events make a mis-ordered prefetch an efficiency blip, never a correctness issue. Ship sync first; enable after measuring `nvme_fetch_wait_ms`.

**Hot tail window v2 (flag `ASYM_NVME_ACT_HOT_WINDOW_BYTES`, default 0 = off)** — Megatron-margin analog (0.6): `_pack` appends spillable handles to a deferred deque instead of submitting immediately; when the deque exceeds the window, pop-oldest and submit its spill. Effect: the most recent W bytes of forward stay CPU-resident, so (a) backward's first unpacks (the forward tail, consumed before the read pipeline warms) are CPU hits with zero NVMe wait, and (b) short-lived saved tensors never round-trip disk. `_unpack` checks the deque first: still deferred → use `handle.tensor` directly (spill was never submitted — no cancel machinery needed). Costs W bytes of pinned CPU, on top of the writer's in-flight cap.

### Efficiency
What is offloaded, every kernel, and every launch are untouched; IO units are the wide [M,I]/[M,H] tensors (tens of MB–GiB — ideal O_DIRECT sizes); the padded pinned buffer is simultaneously the D2H destination and the pwrite source (zero extra memcpys); writer-thread overlap; backpressure bounds pinned growth at `max_inflight_spill_bytes` (a blocked `_pack` is the honest capacity price, reported not hidden).

### Validation (Stage 3 gate — capacity mode)

Unit (`tests/training/test_actnvme_saved_tensor.py`): pack→unpack roundtrip bit-exact (contiguous + strided, bf16/fp32); toy model with `DecoderSavedTensorOffloadWrapper`: loss+grads **bit-identical** `ASYM_NVME_ROLES=""` vs `"activation"` (same seed); existing `test_decoder_activation_offload.py`, `test_decoder_layer_glue_gc.py`, `test_linear_attention_activation_offload.py` pass with NVMe off AND on.

E2E (q3-32b, CPU-heavy point; sequential; `kill -TERM` only):

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage3_actnvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=3 MAX_STEPS=8 \
RUNS='q3-32b|1 ; asym_cpuadamwds|norecomp|ligerloss1 ; 16384|8|1 ; none|false|true|false|true|true || q3-32b|1 ; asym_cpuadamwds_actnvme|norecomp|ligerloss1 ; 16384|8|1 ; none|false|true|false|true|true' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true

.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline <asym_cpuadamwds dir> --candidate <asym_cpuadamwds_actnvme dir> \
  --target activation_cpu \
  --memory-metric activation_offload.max_cpu_peak_bytes_live \
  --min-memory-drop-pct 50 --max-step-ratio 1.10 --max-forward-ratio 1.10 --max-backward-ratio 1.10 \
  --expect-nvme-role activation
```

Accept: `max_cpu_peak_bytes_live` −50%+ (check `cpu_peak_by_tag` — the `decoder.saved.*` tags must collapse); per-step RSS (`step_samples.training_step_process_rss_peak_end_bytes`) drops accordingly; HBM unchanged; step ≤1.10×; `nvme_bytes_written ≈ nvme_bytes_read ≈` baseline Substrate-A offload volume; step-1 loss identical to baseline. **Then the headline: raise seq on the candidate until the baseline CPU-OOMs and the candidate still trains** — record both max-seq numbers (verify real `input_ids` length in `train.log`).

### Risks / watch
- If `nvme_spill_backpressure_ms` dominates the step: write bandwidth is the wall — raise the inflight cap (more pinned) or accept/report the stall; don't hide it.
- Attention wrapper's `require_grad=True` default changes which tensors spill vs decoder — verify per-tag counters against baseline (no tag silently stops offloading).
- Torch pinned-cache size-class churn from padded allocs — watch allocator RSS; mitigation: round padded sizes to 2 MiB classes.

---

## Stage 4 — `panvme`: base weights → NVMe

**Scope:** `asym_gemm/training/host_weight.py` (property surgery), NEW `asym_gemm/training/base_weight_pager.py`, `asym_gemm/integrations/lf.py` (one registration walk at the end of `apply_lf_asym_lora`, ~`:2410`), `asym_gemm/training/qwen3_moe.py` (eager fine-grained split — risks).

### Implementation

```python
# host_weight.py — surgery (everything else unchanged)
class HostWeight:
    # NEW instance fields (default None; set only by pager.register):
    #   _pager, _pager_key
    @property
    def weight(self) -> torch.Tensor:
        pager = getattr(self, "_pager", None)
        if pager is None:
            return self._tensor                      # today's exact path: one attribute check added
        return pager.touch(self._pager_key)          # resident view; fetches on miss; advances prefetch
    tensor = weight
    # nbytes/shape/dtype/is_pinned/metadata properties already read self._metadata → never fetch (0.3).
```

```python
# base_weight_pager.py
ABSENT, INFLIGHT, RESIDENT = 0, 1, 2

class _Entry:
    __slots__ = ("key", "hw", "ref", "shape", "stride", "dtype", "padded_nbytes",
                 "buf", "view", "state", "positions")   # positions: this key's slots in the trace

class BaseWeightPager:
    """Residency owner for registered HostWeights. MAIN THREAD ONLY (training loop thread)."""

    def __init__(self, store, *, cache_bytes=_env_int("ASYM_NVME_BASE_WEIGHT_CACHE_BYTES", 8 << 30),
                 prefetch_depth=_env_int("ASYM_NVME_PREFETCH_DEPTH", 2)):
        self._store, self.cache_bytes, self.prefetch_depth = store, cache_bytes, prefetch_depth
        self._entries: dict[str, _Entry] = {}
        self._by_ref_id: dict[int, _Entry] = {}
        self._free: dict[tuple, list[torch.Tensor]] = {}      # (dtype, shape) -> free padded pinned bufs
        self._quarantine: list[tuple[torch.Tensor, torch.cuda.Event, tuple]] = []
        self._resident_bytes = 0
        # trace: sequence of keys touched in one full fwd+bwd period (each key appears twice: fwd + bwd dX)
        self._trace: list[str] = []; self._trace_build: list[str] = []
        self._frozen = False; self._disabled = False
        self._cursor = -1; self._last_key = None
        self.misses = self.misses_after_freeze = self.forced_syncs = 0

    # ---- registration (conversion time, weights still resident) ----
    def register(self, key: str, hw) -> None:
        t = hw._tensor
        if t is None or t.numel() * t.element_size() < self._store.cfg.min_swappable_bytes:
            return                                            # small weights stay CPU-resident
        padded = alloc_padded_pinned(tuple(t.shape), t.dtype, align=self._store.align)
        padded.copy_(t)                                       # one-time copy into padded storage
        ref = self._store.spill("base_weight", padded, ready_event=None,
                                on_done=lambda buf, r: None)  # buf freed after durability (writer drops ref)
        e = _Entry(key=key, hw=hw, ref=ref, shape=tuple(t.shape), stride=None, dtype=t.dtype,
                   padded_nbytes=_pad(t.numel() * t.element_size(), self._store.align),
                   buf=None, view=None, state=ABSENT, positions=[])
        self._entries[key] = e; self._by_ref_id[id(ref)] = e
        hw._pager, hw._pager_key = self, key
        hw._tensor = None                                     # original pinned home freed NOW
        # transient = one weight extra during registration (same as adoption's pin_memory transient)

    # ---- the single hot entry point ----
    def touch(self, key: str) -> torch.Tensor:
        e = self._entries[key]
        if key != self._last_key:                             # dedupe: .weight is read several times per Function
            self._last_key = key
            self._record_or_advance(e)
            self._issue_prefetches()
            self._evict_to_budget()
        if e.state is RESIDENT:
            return e.view
        if e.state is INFLIGHT:
            for rid in self._store.drain_reads():             # drains ALL pending; flip each arrival
                self._by_ref_id[rid].state = RESIDENT
            assert e.state is RESIDENT
            return e.view
        # ABSENT miss (always in step 1; ~never after freeze):
        e.buf = self._take_buffer(e)
        e.view = e.buf                                        # exact-shape view (alloc_padded_pinned returns it)
        self._store.fetch_into(e.ref, e.buf)
        e.state = RESIDENT; self._resident_bytes += e.padded_nbytes
        self.misses += 1
        if self._frozen: self.misses_after_freeze += 1
        return e.view

    # ---- trace: build during step 1, freeze at period end, then follow ----
    def _record_or_advance(self, e) -> None:
        if self._disabled: return
        if not self._frozen:
            self._trace_build.append(e.key)
            first = self._trace_build[0]
            if e.key == first and self._trace_build.count(first) == 3:
                # occurrences of the first key: fwd@0, its bwd, then fwd again = start of period 2
                self._trace = self._trace_build[:-1]
                for i, k in enumerate(self._trace): self._entries[k].positions.append(i)
                self._frozen = True; self._cursor = 0
            return
        # frozen: advance cursor to this key within a small jitter window
        n = len(self._trace)
        for d in range(0, 8):
            if self._trace[(self._cursor + d) % n] == e.key:
                self._cursor = (self._cursor + d) % n
                return
        self._disabled = True                                 # dynamic order (e.g. variable expert paths):
        self._trace = []                                      # fall back to miss-driven sync fetches; counted
        log_once("base-weight trace disabled — prefetch off, miss-driven fetches")

    def _issue_prefetches(self) -> None:
        if not self._frozen or self._disabled: return
        n = len(self._trace)
        for d in range(1, self.prefetch_depth + 1):
            k = self._trace[(self._cursor + d) % n]
            e = self._entries[k]
            if e.state is ABSENT:
                e.buf = self._take_buffer(e); e.view = e.buf
                self._store.submit_pread(e.ref, e.buf)
                e.state = INFLIGHT; self._resident_bytes += e.padded_nbytes

    # ---- buffers: per-(dtype,shape) rings + Belady eviction, event-gated ----
    def _take_buffer(self, e) -> torch.Tensor:
        self._sweep_quarantine()
        free = self._free.get((e.dtype, e.shape))
        if free: return free.pop()
        if self._resident_bytes + self._quarantine_bytes() + e.padded_nbytes <= self.cache_bytes:
            return alloc_padded_pinned(e.shape, e.dtype, align=self._store.align)   # grow within budget
        self._evict_one(exclude=e)                            # frees a same-or-other shape victim
        free = self._free.get((e.dtype, e.shape))
        if free: return free.pop()
        return alloc_padded_pinned(e.shape, e.dtype, align=self._store.align)       # shapes differ: alloc; budget keeps global bound

    def _evict_one(self, exclude=None) -> None:
        victims = [x for x in self._entries.values() if x.state is RESIDENT and x is not exclude]
        v = max(victims, key=self._next_use_distance)          # farthest next use = Belady on the frozen trace;
        ev = torch.cuda.Event(); ev.record()                   # after all launches that consumed v (single stream)
        self._quarantine.append((v.buf, ev, (v.dtype, v.shape)))
        v.buf = v.view = None; v.state = ABSENT
        self._resident_bytes -= v.padded_nbytes

    def _sweep_quarantine(self) -> None:
        keep = []
        for buf, ev, kls in self._quarantine:
            if ev.query(): self._free.setdefault(kls, []).append(buf)
            else: keep.append((buf, ev, kls))
        self._quarantine = keep

    def _next_use_distance(self, e) -> int:
        if not self._frozen or self._disabled: return 1 << 30
        n = len(self._trace)
        return min((p - self._cursor) % n or n for p in e.positions)

    def _evict_to_budget(self) -> None:
        while self._resident_bytes > self.cache_bytes and any(x.state is RESIDENT for x in self._entries.values()):
            self._evict_one()
        self._sweep_quarantine()

    def summary(self) -> dict:   # surfaced under asym_nvme.base_weight_pager
        return {"registered": len(self._entries), "resident_bytes": self._resident_bytes,
                "cache_bytes": self.cache_bytes, "trace_frozen": self._frozen, "trace_disabled": self._disabled,
                "trace_len": len(self._trace), "misses": self.misses,
                "misses_after_freeze": self.misses_after_freeze}
```

Registration walk (end of `apply_lf_asym_lora`, env-gated; and the eager split):

```python
store = get_nvme_store()
if store is not None and store.has_role("base_weight"):
    if os.environ.get("ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD", "") in _TRUTHY:
        for m in model.modules():                       # BEFORE spilling: force the lazy fused→gate/up split
            if isinstance(m, AsymQwen3Experts): m._ensure_qwen3_moe_finegrained_bases()
    pager = BaseWeightPager(store)
    for name, mod in model.named_modules():
        hw = getattr(mod, "host_weight", None)
        if isinstance(hw, HostWeight) and _panvme_component_eligible(name, mod):
            # eligible: AsymFrozenLinear (attn + mlp_dense) and AsymGroupedFrozenLinear (experts).
            # EXCLUDED: AsymFrozenEmbedding / norms (CPU-side F.embedding per microbatch; tiny norms).
            pager.register(name, hw)
    model._asym_base_weight_pager = pager
```

Correctness anchors: fwd+bwd interleaved trace makes farthest-reuse exact Belady (after forward, late layers — reused first in backward — rank nearest and survive); eviction event-gating per 0.3's async-launch rule; `touch` dedupe handles repeated `.weight` reads within one Function (predicates + kernel + shape reads); reporting never fetches (metadata properties); step-1 is miss-driven by design (excluded via WARMUP_STEPS≥3); `precision=="bf16"` asserted at registration (non-bf16 builds `QuantizedHostWeight` caches from `.weight`, `frozen_linear.py:372` — out of scope v1).

### Validation (Stage 4 gate)

Unit (`tests/training/test_base_weight_pager.py`): register→`_tensor is None`; `touch` roundtrip bit-exact vs pre-spill clone (2D + grouped 3D bf16); trace freezes at 3rd first-key occurrence with each key twice; farthest-reuse eviction; quarantine blocks reuse until `ev.query()` (mock); jitter window tolerates a one-slot swap; trace-disable fallback stays correct; NVMe-off → `HostWeight` byte-identical (run existing `test_cpu_resident_frozen_base.py`, `test_lf_qwen3_asym_backend.py` unmodified).

E2E:

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage4_panvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=3 MAX_STEPS=8 \
RUNS='q3-32b|1 ; asym_cpuadamwds|norecomp|ligerloss1 ; 8192|8|1 ; none|false|true|false|true|true || q3-32b|1 ; asym_cpuadamwds_panvme|norecomp|ligerloss1 ; 8192|8|1 ; none|false|true|false|true|true' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true

.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline <asym_cpuadamwds dir> --candidate <asym_cpuadamwds_panvme dir> \
  --target base_weight_cpu \
  --memory-metric step_samples.training_step_process_rss_peak_end_bytes \
  --min-memory-drop-gib 40 --max-step-ratio 1.05 --max-forward-ratio 1.05 --max-backward-ratio 1.05 \
  --expect-nvme-role base_weight
```

Accept: per-step RSS −40 GiB+ (base ≈60 GiB minus 8 GiB cache+rings; `memory_attribution` host_weight/cpu rows shrink to match); HBM unchanged; ≤5% latency; `base_weight_pager.misses_after_freeze ≈ 0` (else cache too small — raise and rerun); `trace_disabled == false`; `bytes_read ≈ 2×base×steps`; losses equal baseline within noise.

### Risks / watch
- **MoE+fine-grained double residency**: handled by the eager split above; until the MoE path is tested, panvme + `ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD` must hard-error at registration rather than silently double-fetch.
- If ≤5% fails at s8192 (120 GiB/step vs short steps), rerun at a longer-seq operating point and reclassify panvme as capacity-mode for short seq — decide from the measured run.
- Any future new `.weight` reader must be audited: compute path (fine) vs reporting path (must use metadata).
- Grouped-expert blobs are GB-scale — ring holds ≥2 per shape or prefetch stalls; assert `cache_bytes ≥ 2×largest_padded_nbytes` at registration.

---

## Stage 5 — `actnvme` part 2: Substrate B (`ActivationOffloadManager` — attn U/S, experts, fine-grained)

**Scope:** `asym_gemm/training/activation_offload.py` + one `seal(...)` line per Function forward (`attention_activation_offload.py`; then `qwen3_moe.py` / `dense_mlp_finegrained.py` / llama4 engines one at a time). Dense layer-GC configs are already covered by Stage 3; this stage moves the `*.U`/`*.S` + expert tensors. Rollout behind `ASYM_NVME_ACT_SUBSTRATE_B=1`.

### Implementation

```python
# activation_offload.py
@dataclass
class _SpillState:
    ref: Any = None            # BlobRef once sealed+submitted
    sealed: bool = False

class ActivationOffloadManager:
    def __init__(self, *, pin_memory=True):
        ...
        self._spill: dict[int, _SpillState] = {}       # id(handle) -> state (handle alive on ctx; popped on fetch/release)

    def _spill_eligible(self, handle) -> bool:
        store = get_nvme_store()
        return (store is not None and store.has_role("activation")
                and _env_bool("ASYM_NVME_ACT_SUBSTRATE_B", False)
                and handle.nbytes >= store.cfg.min_swappable_bytes)

    # offload()/empty_cpu(): switch _alloc_cpu to padded storage when a store is active
    # (pool key unchanged — pooled buffers just carry padded storages); register intent:
    def offload(self, tensor, tag):
        handle = ...existing...
        if self._spill_eligible(handle):
            self._spill[id(handle)] = _SpillState()
        return handle

    def seal(self, *handles) -> None:
        """Call as the LAST statement of a Function.forward: every forward consumer of these
        handles is now enqueued on the current stream. One event covers (a) the D2H fill and
        (b) all forward kernels streaming the buffer."""
        store = get_nvme_store()
        pending = [h for h in handles if h is not None
                   and (st := self._spill.get(id(h))) is not None and not st.sealed]
        if not pending: return
        ev = torch.cuda.Event(); ev.record()
        for h in pending:
            st = self._spill[id(h)]
            st.sealed = True
            st.ref = store.spill("activation", h.tensor, ready_event=ev,
                                 on_done=self._make_on_durable(h))

    def _make_on_durable(self, handle):
        def cb(buf, ref):        # WRITER THREAD: ev synced (D2H done + fwd consumers done) and bytes durable
            st = self._spill.get(id(handle))
            if st is None or st.ref is not ref: return       # already fetched/released on main thread
            self._pop_active(handle)                          # cpu_owned_bytes -= ; accounting off this ptr
            _return_cpu(handle.tensor, pin_memory=self.pin_memory)   # pool reuse now safe (event proven)
            object.__setattr__(handle, "tensor", _SPILLED_SENTINEL)  # 0-elem tensor: any stray read fails loudly
        return cb

    def _ensure_local(self, handle) -> None:
        st = self._spill.pop(id(handle), None)
        if st is None or st.ref is None:                      # never sealed (or spill off) → tensor valid
            return
        store = get_nvme_store()
        bounce = _alloc_cpu(handle.original_shape, handle.original_dtype, pin_memory=True)  # padded pool buffer
        store.fetch_into(st.ref, bounce)                      # waits durability; drains prefetches
        store.blob_done(st.ref)
        object.__setattr__(handle, "tensor", bounce)          # single sanctioned frozen-dataclass mutation point
        self._mark_cpu_live(handle)                           # re-enter accounting under the new data_ptr

    # stage()/stage_rows()/stage_concat_columns(): insert `self._ensure_local(handle)` as the FIRST line
    # (before wait_cpu_ready). release_cpu(): if id(handle) still in self._spill with a ref and no fetch
    # happened → pop, store.blob_done(ref) (drop the blob), skip pool-return (buffer already recycled).
```

Call sites (one line each): `_AsymActivationOffloadLoRALinearFunction.forward` ends with `manager.seal(s_handle)`; the **shared q/k/v U** seals at the context's existing cache-clear point (`attention_activation_offload.py:477-479` — the structural "all q/k/v consumers enqueued" marker): `self.manager.seal(shared.handle)` (o_proj's own U seals in its Function like S). Expert / fine-grained engines: `manager.seal(x_cpu, gate_cpu, up_cpu, ...)` as the last forward statement — **one engine at a time**, each audited for "is this really after the last forward consumer". An unsealed handle simply never spills (degrades to status quo, never corrupts).

Event algebra (why this is safe): `ev` is recorded after (a) the D2H copy and (b) every forward kernel launch streaming the buffer — both earlier on the same stream. Writer syncs `ev` before pwrite (stable bytes) — and pool-return happens after that same sync, so no kernel can still be streaming a recycled buffer. Backward consumers always pass through `_ensure_local` first. The `data_ptr` accounting hazard (0.2) is avoided because `_pop_active` runs before pool-return and the handle's tensor is swapped to a sentinel.

### Validation (Stage 5 gate)

Unit: `test_attention_activation_offload_lora.py` parametrized over `ASYM_NVME_ACT_SUBSTRATE_B` — fwd+bwd bit-exact on vs off; shared U spilled once, fetched once, with q/k/v all consuming; `stage_rows`/`stage_concat_columns` on spilled handles; release-without-fetch drops the blob; sentinel read raises.

E2E: rerun the Stage 3 command pair with `ASYM_NVME_ACT_SUBSTRATE_B=1` — additional `cpu_peak_by_tag` collapse for `*.U`/`*.S` tags at the same gate thresholds; before claiming MoE, one `q3-30b-a3b` run with the expert engine's offload policy config.

### Risks / watch
- Seal-point audit per engine is the whole risk — keep the env flag per-engine if needed; unsealed = safe.
- `empty_cpu`-created handles filled by CPU GEMM outputs never pass `offload()` → not spilled in v1; visible in `cpu_peak_by_tag`, extend only if dominant.
- Writer-thread `_return_cpu` touches the global pool from a non-main thread → guard the pool with a small lock (uncontended; taken per-blob, not per-byte).

---

## Stage 6 — `bothnvme`: compose + hero max-seq

No new mechanism: token enables both roles on the one store (per-blob weight files + activation arena coexist; shared handles/threads). Startup assert: `base_weight cache + rings + activation inflight cap + pool limit < free host RAM headroom`.

Gate: on q3-32b (then q3-30b-a3b) demonstrate max trainable seq `bothnvme > actnvme ≥ baseline` at fixed batch; verify real `input_ids` lengths; report HBM peak, per-step RSS, per-role NVMe bytes + wait-ms, step time, and overlap fraction (`1 − nvme_wait/step`). Compare against the SuperOffload/zero3 ceilings for the paper's capability table.

---

## Deferred
- DeepSpeed-owned backend behind the same store API (ownership exclusive with `HostWeight`/`AsymCPUAdamW`); per-rank multi-GPU = rank-suffixed arenas + all ranks pread the **same** read-only base-weight files.
- GDS/GPU-direct (hardware absent); Unsloth-GC boundary offload (LlamaFactory-side, third role later); gradient/optimizer NVMe (LoRA-tiny).
- Side-stream H2D staging + double-buffered reload for the CPU→HBM leg (TE v1 / Megatron pattern, 0.6): the substrates' unpack H2D copy is synchronous today; moving it to a dedicated stream one group ahead is an existing-substrate optimization independent of NVMe — measure `_unpack` copy time after Stage 3 before investing.

## Global run rules (every e2e gate)
Heavy offload runs **sequentially** (665–802 GiB RSS observed); stop with `kill -TERM`, never `-9` (corrupts the DeepSpeed cpu_adam JIT cache); `PREPARE_DATASETS=true` on first use of a workload and **verify real `input_ids` length in `train.log`**; measure from `step_samples.csv` measured rows; `.aioenv` env is exported by `run_lf_lora_sft.sh` — export it manually for direct pytest/python store use.

## Implementation order = stage order

```text
Stage 1: nvme_store.py substrate            (isolated; unit + AIO smoke gate)
Stage 2: tokens + env + counters + compare  (paired e2e no-change gate)
Stage 3: actnvme / Substrate A              (capacity gate ≤10% + max-seq demo)   ← core target first
Stage 4: panvme                             (throughput-preserving gate ≤5%)
Stage 5: actnvme / Substrate B              (per-engine seal rollout, Stage-3 gate)
Stage 6: bothnvme hero                      (max-seq capability table)
```

Why actnvme before panvme (decided 2026-07-01): Stages 1+2 already de-risk the substrate (AIO smoke-tested on-box; wiring proven no-op); actnvme-A is the smaller diff (no CPU-kernel consumers, bit-exact testable) while panvme carries the pager machine and the stricter gate; and it fails-fast on the core research target. panvme's CPU-freeing is only *required* by the Stage-6 compound hero.

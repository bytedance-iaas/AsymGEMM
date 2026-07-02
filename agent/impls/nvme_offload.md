# AsymGEMM NVMe Offload — Implementation Plan (v3, code-verified 2026-07-01)

Three composable, opt-in backend tokens, all built on ONE local NVMe store that reuses DeepSpeed's AIO substrate:

- `asym_cpuadamwds_panvme` — base/frozen weights (`HostWeight` CPU homes) → NVMe, bounded pinned cache + order-prefetch. DeepSpeed-parity (ZeRO-Infinity param-NVMe), *not novel*; frees ~60 GiB CPU on q3-32b.
- `asym_cpuadamwds_actnvme` — offloaded forward activations (the CPU activation homes) → NVMe. *Novel*; raises the CPU-capacity ceiling that binds max seq. Capacity/hero mode.
- `asym_cpuadamwds_bothnvme` — both roles; compound for the max-seq hero result.

Dropped (re-verified): optimizer-state and LoRA-weight-home NVMe — `AsymCPUAdamW` holds fp32 master+moments **only for LoRA params** (`cpu_adam.py:126-129,194,212-218`); LoRA homes are small pinned bf16 slabs (`weight_offload.py:174-182`). Nothing to save.

Everything below was re-verified line-by-line against the live repo on 2026-07-01 (second pass, including an on-box AIO smoke test). Line numbers are current. **Where any earlier doc/version disagrees, this file wins.**

---

## 0. Verified ground truth (read before implementing)

### 0.1 DeepSpeed AIO — proven working on this box

```bash
# .aioenv sidecar is REQUIRED for JIT build + runtime (run_lf_lora_sft.sh:29-40 already exports it).
export AIO_HOME=$PWD/.aioenv
export CPATH="$AIO_HOME/include:${CPATH:-}" LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
.venv/bin/python -c "from deepspeed.ops.op_builder import AsyncIOBuilder; print(AsyncIOBuilder().is_compatible())"
```

Empirically confirmed (2026-07-01): `is_compatible()==True` with `.aioenv`, **False without it** (JIT fails); op builds in ~23 s; `aio_handle(1048576, 16, False, True, 4)` gives `get_alignment()==2048` (= `intra_op_parallelism*512`); pinned pwrite/pread roundtrip **and offset-based roundtrip into one file both pass**.

Hard API facts (from `csrc/aio/py_lib/`):
- ctor `aio_handle(block_size, queue_depth, single_submit, overlap_events, intra_op_parallelism)` (`py_ds_aio.cpp:23`); `async_pread(buffer, filename, file_offset=0)` / `async_pwrite(buffer, filename|fd, file_offset=0)` (`py_ds_aio.cpp:87-112`); `wait()` releases the GIL (`py_ds_aio.cpp:125-128`).
- `wait()` drains **ALL** pending ops on that handle and `assert(_num_pending_ops > 0)` — **calling wait() with nothing pending aborts the process** (`deepspeed_py_io_handle.cpp:201-220`). Track pending counts in Python; never call wait() idle. Use **separate read and write handles** so a read drain never waits on writes.
- `num_bytes % intra_op_parallelism` must be 0 or the op is rejected (`deepspeed_py_io_handle.cpp:222-233`). Files are O_DIRECT (`deepspeed_aio_common.cpp:269`). ⇒ **pad every IO offset and length to `handle.get_alignment()`**, and IO buffers must be allocated at padded size (never pwrite a padded length from an exact-size buffer — OOB read).
- Unpinned/CUDA buffers silently **bounce** through an internal managed pinned buffer with a full extra copy (`deepspeed_cpu_op.cpp:25-31,86-105`) — correct but slow; keep the common path pinned so IO is zero-copy.
- Reusable as-is: `SwapBufferPool`/`get_sized_buffer(s)`/`MIN_AIO_BYTES=1MiB`/`AIO_ALIGNED_BYTES=1024` (`runtime/swap_tensor/utils.py:15-16,97,230`). NOT reusable: `AsyncPartitionedParameterSwapper` + `offload_param.device=nvme` (needs `ds_id`/`ds_tensor`/`deepspeed.comm`, built only inside `zero.Init` — `partitioned_param_swapper.py:76-161`, `partition_parameters.py:1081-1095`). `SwapBufferManager` calls `dist.get_rank()` (`utils.py:195`) — skip. `GDSBuilder` — skip (opt-in, needs cufile). We don't strictly need SwapBufferPool either — torch's caching pinned allocator + our own ring is simpler; decided below.

### 0.2 panvme target — base weights

- Single tensor-level chokepoint: `HostWeight.__init__` (`asym_gemm/training/host_weight.py:185-242`; pin at `:222`). `.weight`/`.tensor` are **properties over `self._tensor`** (`:244-246,302-304`) → clean lazy-materialize interception. `metadata.nbytes` (`:283-284`) is copy-free — reporting must use it, never the tensor.
- Adoption entries: `adopt_host_weight()` (`offload.py:176-213`, called from `lf.py:1192` in `_wrap_lf_linear_leaf`) and `AsymGroupedFrozenLinear.__init__` (`frozen_linear.py:1624`) for grouped experts. Embeddings/norms via `offload.py:352,394,441`.
- Consumption: dense fwd `frozen_linear.py:1307-1318`, dense dX `frozen_linear.py:1353-1365` (both via `_dispatch_nt(..., host_weight.weight, ...)`); grouped-expert dX `qwen3_moe.py:462-486`; grouped fwd via `_asym_grouped_bf16_nt` (`frozen_linear.py:729-754`); attention act-offload base fwd `attention_activation_offload.py:599-608`; native windowed bwd consumes it directly (`qwen3_moe.py:1878`).
- **Kernel launches are async**: `asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous(...)` returns immediately (`frozen_linear.py:723-726`); the GPU kernel streams the CPU-pinned weight during execution. ⇒ a weight buffer may be recycled **only after a CUDA event recorded on the current stream (after the last consuming launch) has completed**.
- Runtime `.weight` readers that are NOT the GEMM itself: `is_pinned()` predicates (`qwen3_moe.py:1760`, `:2648-2654`; `llama4_moe.py:242`) — all sit immediately before a consuming kernel, so lazy fetch is self-consistent; and `_ensure_qwen3_moe_finegrained_bases` (`qwen3_moe.py:2492-2515`) **lazily slices the fused gate_up weight into two NEW AsymGroupedFrozenLinear** — under panvme this must run eagerly at conversion (see Stage 3 risks).
- After expert adoption the HF source params are shrunk to 0-numel (`qwen3_moe.py:2127-2136`) — HostWeight owns the only copy; spilling it genuinely frees RSS.

### 0.3 actnvme target — two substrates, different difficulty

**Substrate A (`saved_tensors_hooks` wrappers) — simple, and the layer-GC bulk.** Three near-identical wrappers: `DecoderSavedTensorOffloadWrapper` (`decoder_activation_offload.py:99`; `_pack` `:188-222`, `_unpack` `:224-248`), `AttentionSavedTensorOffloadWrapper` (`attention_activation_offload.py:187`), `LinearAttentionSavedTensorOffloadWrapper` (`linear_attention_activation_offload.py:99`). `DecoderLayerGlueGCWrapper` "custom" mode reuses the decoder `_pack`/`_unpack` (`decoder_layer_glue_gc.py:238-246`). Facts:
- `_pack`: fresh **stride-preserving** pinned CPU tensor per saved tensor (`_empty_strided_cpu_like` `:71-83`; torch's caching host allocator recycles, no pool), D2H `copy_(non_blocking)`, `ready_event` recorded on the current stream. Handle = **mutable** `@dataclass _SavedTensorOffloadHandle` (`:86-97`, has `released` flag mutated in `_unpack`).
- `_unpack`: `ready_event.synchronize()` → fresh HBM `empty_strided` → sync copy from CPU. **The CPU buffer is touched only by the D2H copy (write) and the unpack H2D copy (read). No CPU-side kernel ever consumes it.** ⇒ spill is safe as soon as `ready_event` fired; after pwrite completes the buffer can be freed outright (unpack under NVMe fetches into a NEW buffer).
- Filter: CUDA, dtype∈{bf16,fp16,fp32}, ≥1 MiB, not `nn.Parameter` (`decoder_activation_offload.py:159-181`); decoder/linear `require_grad=False` default, attention `True`.
- Per-module stats dict published as `module._last_activation_offload_stats` (`:280-281`) and harvested by the profiler (see 0.5).

**Substrate B (`ActivationOffloadManager`, `activation_offload.py`) — needs a seal-point design.** One manager per autograd-Function call, but ONE global pinned pool `_CPU_BUFFER_POOL` keyed `(dtype, shape, pinned)` (`:10,74-103`; cap env `ASYM_EXPACT_CPU_POOL_MAX_BYTES`, default 32 GiB `:13`). Facts that shape the design:
- `CPUActivationHandle` is `@dataclass(frozen=True)` (`:106-116`); manager accounting keyed by `int(handle.tensor.data_ptr())` (`_active_cpu_bytes` `:164`, `_pending_cpu_ready_events` `:166`). Handles are never dict keys → spill state can live in a side table or a new construction-time field.
- **Handles are consumed on CPU by async GPU kernels**: `_dense_lora_a_cpu_left(u_handle.tensor, ...)` reads the offloaded U **in forward** (`attention_activation_offload.py:619-625`), and backward CPU-right kernels read U/S again. q/k/v share one U handle refcounted via `_SharedActivationSource` (`:417-441`), released at the v_proj cache-clear point (`:477-479`).
- ⇒ freeing a Substrate-B buffer to the pool needs: D2H `ready_event` fired (pwrite may read only stable bytes), **and** all enqueued GPU kernels that stream it have completed (event-gated quarantine). Pool reuse today is safe purely because everything is on one stream — the IO thread breaks that invariant, so we add events.
- `stage()/stage_rows()/stage_concat_columns()` (`:237-299`) are the un-spill points; `release_cpu()` (`:315-326`) returns buffers to the pool; `wait_cpu_ready()` (`:230-235`) is the event gate.

**Today there is no copy stream and no prefetch anywhere in the activation path** (only `weight_offload.py:95` holds an unused `_stream`). NVMe overlap infra is new.

### 0.4 Backend tokens — every dispatch site (exact-token case arms, no globs → no collision with `zero3_offload_panvme`)

Grammar: `model|gpus ; backend|recompute|liger[|kernelcode] ; seq|batch|grad_accum ; policy|expact|attnact|layeract|layergc|sdparecomp`. Env override for the policy list is **`ASYMM_EXP_ACT_POLICIES`** (`profile_lora_lf_test_source.sh:2122`; CLI `--asymm-exp-act-policies` `:2157`).

Sites to extend for the three new tokens:
- `scripts/lf/profile_lora_lf_test_source.sh`: `append_backend_spec` case + die (`:956-978`); `backend_gpu_count` case + die (`:789-795`); `cpuadam_backend_for_label` (`:918-923`); per-job derivation (`~:3094-3110`); `run_env` block already forwards `ASYM_GEMM_LF_CONFIG_*` (`:3343-3344,3387`). The backend token is the first component of `path_label` (`job_root_path` `:1733-1741`) → run dirs auto-disambiguate; **no extra label tag needed**. `job_profile_complete`/`existing_profile_complete` (`:1104,:1148`) receive `backend=$3` → completion checks disambiguate too.
- `scripts/lf/run_lf_lora_sft.sh`: main `case "${BACKEND,,}"` — clone the `asym_cpuadamwds` arm (`:352-358`) + extend the die list (`:369`). `is_zero_backend_run` requires `BACKEND==torch` (`:659-661`) → our arms set `BACKEND=asym`, so no `--deepspeed` leakage (`assert_deepspeed_scope` `:678-690` stays happy). `.aioenv` exports at `:29-40` and the `RECORD_IO` `/sys/block/<dev>/stat` sampler (`:187,2417-2430`) are backend-agnostic — free NVMe-IO cross-check.
- `scripts/lf/run_lf_profiled_train.py`: backend classification (`:577-599` — `is_asym_deepspeed_cpuadamw` at `:579-582` must accept the new tokens); `_config_from_args` (`:546`, env-mirror pattern e.g. `:732-734`).
- **No LlamaFactory changes.** Config rides env like every other ASYMM_* feature (e.g. `ASYMM_ATTN_ACT_OFFLOAD` read in `lf.py:1322-1325`); the profile records it via `ASYM_GEMM_LF_CONFIG_ASYM_NVME_*` mirrors. This replaces the earlier plan's LF-dataclass section.

### 0.5 Profiling / gating infra

- Canonical driver `scripts/lf/profile_lora_lf_test_source.sh` (only fork besides `_both`; `PROFILERS` default `source` at `:131`; model shorthands `M[q3-32b]="Qwen/Qwen3-32B"` etc. `:34-46`; `PREPARE_DATASETS` `:204`; CLI `--gpus/--overwrite/--prepare-datasets` `:2143,2291,2161`).
- `source_profile.json` produced by `report()` (`run_lf_profiled_train.py:2849-2919`); `activation_offload` block from `_activation_offload_counters_from_model()` (`:2198-2282`) which harvests every module's `_last_activation_offload_stats` dict into `rows` **automatically** (new keys in `snapshot()` propagate for free) but computes **explicit aggregates** at `:2265-2281` (`max_cpu_peak_bytes_live`, `total_cpu_owned_bytes`, …) — NVMe aggregates must be added there. Host RSS: `memory.process.rss_peak_bytes` = VmHWM (lifetime peak — polluted by load transients); **per-step RSS** lives in `step_samples.rows[].training_step_process_rss_peak_end_bytes` — use that for steady-state gates. `memory_attribution.rows[]` carries `category=host_weight, device=cpu, bytes, pinned_bytes` per component.
- Compare-gate template: `scripts/lf/compare_liger_loss_profiles.py` (args `:33-42`; `load_memory_metrics` `:156`; `_timing_from_step_samples` `:208` reads median `step/forward/backward_milliseconds` from `step_samples.csv`; verdict `{"ok":bool,...}` + `SystemExit(2)`). Clone → `compare_nvme_profiles.py` (Stage 2).
- Eyeball loop: `scripts/lf/show_metrics.py` (fwd/bwd/opt/step s, HBM peaks, RAM = host RSS GiB).

### 0.6 Hardware & feasibility math

4× Samsung PM9A3 3.84 TB (~6.5 GB/s read, ~3.5 GB/s write each) in RAID0 `md0` → `/scratch_local` (~14 TB): ~26 GB/s read, ~14 GB/s write aggregate. Put `ASYM_NVME_PATH` on `/scratch_local/...` (DeepSpeed NVMe configs already use it).
- panvme q3-32b: base ≈ 60 GiB bf16, read fwd+bwd = ~120 GiB/step → needs ≥ `120GiB/step_time` sustained read; at 20-30 s steps that is 4-6 GB/s — comfortably overlappable. Gate stays throughput-preserving (≤5%).
- actnvme: spill volume = offloaded-activation bytes per microbatch (tens-to-hundreds of GiB at long seq), written once + read once per step → write bandwidth is the binding side (~14 GB/s). At long-seq step times (minutes) this overlaps; at short seq it will NOT hide → actnvme is a **capacity mode** (≤10% gate at the capacity operating point, plus a max-seq demo).
- Endurance: ~1 DWPD class (~7 PBW/drive, ~28 PBW across RAID0). Research volumes (bounded runs, few TB/day) ⇒ non-issue; only months of 24/7 saturated activation spill would matter.

---

## Design contract (all stages)

1. **One store, role-tagged.** `NVMeStore` serves roles `{base_weight, activation}`; tokens map `panvme→{base_weight}`, `actnvme→{activation}`, `bothnvme→both`. Consumers (`HostWeight`, the activation substrates) know only the store API — that is the modular decoupling; a future DeepSpeed-owned backend can replace the store behind the same API.
2. **No kernel computes from NVMe.** Always `NVMe → pinned CPU → (H2D) → compute` and `pinned CPU → NVMe`. AsymGEMM kernels keep receiving exactly what they receive today (pinned CPU operand / HBM tensor).
3. **Compute shape untouched.** No change to any GEMM, grouping, slab, or launch pattern. IO units are whole tensors (≥1 MiB) — never per-expert, never per-chunk loops.
4. **Single-owner AIO handles.** Write handle owned by ONE writer thread; read handle owned by the main thread. Python-side pending counters; `wait()` only when pending>0.
5. **Event-gated buffer reuse.** Any pinned buffer that a GPU kernel may still be streaming is recycled only after a CUDA event recorded post-enqueue has completed. (Stream ordering protects the status quo; the IO thread breaks it, events restore it.)
6. **Off = byte-identical.** No `*nvme` token ⇒ no AIO build, no thread, no file, no behavioral change; every hook is a `None` check.
7. Alignment: every file offset and IO length padded to `handle.get_alignment()`; IO buffers allocated at padded size.

---

## Stage 1 — `NVMeStore` substrate (new file, fully isolated)

**Scope:** add `asym_gemm/training/nvme_store.py` + `tests/training/test_nvme_store.py`. Zero edits to existing files ⇒ this is the one stage where isolated unit tests are a sufficient gate.

### Code (near-real pseudocode)

```python
# asym_gemm/training/nvme_store.py
from __future__ import annotations
import os, threading, time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import torch

TensorRole = Literal["base_weight", "activation"]

def _env_int(name, default): ...
def _env_bool(name, default): ...

@dataclass(frozen=True)
class NVMeStoreConfig:
    path: str                                   # ASYM_NVME_PATH (required)
    roles: frozenset[str]                       # from ASYM_NVME_ROLES ("base_weight,activation")
    aio_block_size: int = 1 << 20               # ASYM_NVME_AIO_BLOCK_SIZE
    aio_queue_depth: int = 16
    aio_intra_op_parallelism: int = 4
    aio_single_submit: bool = False
    aio_overlap_events: bool = True
    min_swappable_bytes: int = 1 << 20          # never spill blobs below this
    activation_arena_bytes: int = 1 << 40       # 1 TiB sparse file (allocated on demand)
    max_inflight_spill_bytes: int = 8 << 30     # writer backpressure cap
    def from_env() -> "NVMeStoreConfig | None":  # returns None when disabled
        roles = {r for r in os.environ.get("ASYM_NVME_ROLES", "").split(",") if r}
        if not roles: return None
        path = os.environ["ASYM_NVME_PATH"]      # KeyError = loud config error
        ...

def _flat_u8(t: torch.Tensor) -> torch.Tensor:
    """uint8 alias of t's whole storage (works for strided/padded storages; no copy)."""
    out = torch.empty(0, dtype=torch.uint8, device=t.device)
    out.set_(t.untyped_storage(), 0, (t.untyped_storage().nbytes(),))
    return out

@dataclass
class BlobRef:
    role: str
    file: str                # per-blob file (base_weight) or role arena file (activation)
    offset: int              # aligned; 0 for per-blob files
    length: int              # padded length actually on disk
    logical_nbytes: int
    durable: threading.Event = field(default_factory=threading.Event)  # set after pwrite completes
    canceled: bool = False   # fetch-before-write cancels the spill (buffer still valid)

class _WriterThread(threading.Thread):
    """Sole owner of the write aio_handle. Items: (ready_event|None, buffer_u8, ref, on_done)."""
    def run(self):
        while True:
            item = self._q.get()
            if item is _STOP: break
            ready_event, buf, ref, on_done = item
            if ref.canceled:
                on_done(buf, ref); continue                    # consumer raced us; skip IO
            if ready_event is not None:
                ready_event.synchronize()                      # D2H copy must be complete
            self._h.async_pwrite(buf, ref.file, ref.offset)    # padded length == buf.nbytes
            assert self._h.wait() == 1                         # one op in flight per iteration (v1: simple+correct)
            ref.durable.set()
            self._inflight_bytes -= buf.nbytes                 # releases backpressure
            on_done(buf, ref)                                  # e.g. free/return the pinned buffer
    # submit() blocks while _inflight_bytes > cfg.max_inflight_spill_bytes  (backpressure)

class NVMeStore:
    def __init__(self, cfg: NVMeStoreConfig):
        from deepspeed.ops.op_builder import AsyncIOBuilder      # import only when enabled
        m = AsyncIOBuilder().load(verbose=False)
        mk = lambda: m.aio_handle(cfg.aio_block_size, cfg.aio_queue_depth,
                                  cfg.aio_single_submit, cfg.aio_overlap_events,
                                  cfg.aio_intra_op_parallelism)
        self._read_h, self._write_h = mk(), mk()                 # separate: wait() drains per-handle
        self.align = int(self._read_h.get_alignment())           # 2048 for intra=4 (measured)
        self._writer = _WriterThread(self._write_h, cfg); self._writer.start()
        self._arena_cursor = {role: 0}; self._arena_live = {role: 0}   # bump allocator per arena
        self._pending_reads: dict[int, BlobRef] = {}             # main-thread-only prefetch ledger
        self.stats = NVMeStoreStats()                            # bytes/ops/wait-ms per role+direction

    # ---- allocation ----
    def _pad(self, n): a = self.align; return (n + a - 1) // a * a
    def alloc_arena(self, role, nbytes) -> tuple[str, int]:
        """Bump-pointer in the role arena. Reset cursor when live-blob count returns to 0
        (activation lifetime = within one microbatch fwd→bwd, so 'reset when empty' is exact
        and needs no step hooks; works under grad accumulation)."""
        length = self._pad(nbytes)
        off = self._arena_cursor[role]; self._arena_cursor[role] += length
        self._arena_live[role] += 1
        if self._arena_cursor[role] > cfg.activation_arena_bytes: raise RuntimeError("arena full: raise ASYM_NVME_ACTIVATION_ARENA_BYTES")
        return self._arena_file(role), off
    def blob_done(self, ref):     # called after final fetch/free of an activation blob
        self._arena_live[ref.role] -= 1
        if self._arena_live[ref.role] == 0: self._arena_cursor[ref.role] = 0

    # ---- write path (any thread) ----
    def spill(self, role, tensor, *, ready_event=None, on_done) -> BlobRef:
        """tensor: pinned CPU with PADDED storage (caller guarantees; see per-stage alloc helpers).
        Enqueues to the writer thread; returns immediately. on_done(buf, ref) runs on the writer
        thread after the write is durable — free/return the buffer there."""
        buf = _flat_u8(tensor)                                   # padded storage alias
        file, off = (self._per_blob_file(role, tensor), 0) if role == "base_weight" \
                    else self.alloc_arena(role, buf.nbytes)
        ref = BlobRef(role, file, off, buf.nbytes, logical_nbytes=tensor_logical_nbytes)
        self._writer.submit(ready_event, buf, ref, on_done)      # blocks on backpressure cap
        return ref

    # ---- read path (MAIN THREAD ONLY) ----
    def fetch_into(self, ref: BlobRef, dst_padded_pinned: torch.Tensor) -> None:
        """Blocking read. If prefetches are in flight, one wait() drains them all —
        reconcile the ledger (A1 fix: never assume wait() returns 1)."""
        if not ref.durable.is_set():
            if self._writer.try_cancel(ref): pass                # raced a queued spill: buffer still live, caller handles
            else: ref.durable.wait()                             # write in flight: brief block
        if id(ref) in self._pending_reads: self.drain_reads(); return   # prefetch already brought it
        self._read_h.async_pread(_flat_u8(dst_padded_pinned), ref.file, ref.offset)
        self._pending_reads[id(ref)] = ref
        self.drain_reads()
    def prefetch_into(self, ref, dst) -> None:                   # async; arrival via drain_reads()
        self._read_h.async_pread(_flat_u8(dst), ref.file, ref.offset)
        self._pending_reads[id(ref)] = ref
    def drain_reads(self):
        if self._pending_reads:
            n = self._read_h.wait()                              # drains ALL pending reads
            assert n == len(self._pending_reads)
            for r in self._pending_reads.values(): r.arrived = True
            self._pending_reads.clear()

_STORE: NVMeStore | None = None
def get_nvme_store() -> NVMeStore | None:
    """Lazy singleton from env. Returns None unless ASYM_NVME_ROLES is set — the
    disabled path allocates nothing and never imports deepspeed."""
```

Notes locked in:
- **base_weight = one file per HostWeight** (`{path}/base_weight/{sanitized_module_path}.bin`): weights are 50 MB–GBs, static, written once — per-blob files avoid offset bookkeeping with zero metadata overhead at this size. **activation = one preallocated arena file per role** with a bump allocator + reset-when-empty: thousands of transient blobs/step; per-blob files would hammer filesystem metadata.
- Writer does one op per `wait()` in v1 — simple, provably matching the pending-counter, and write latency is hidden by the thread. Batching submissions (submit k, wait once) is a flagged follow-up, not v1.
- Padded-storage allocation helpers (used by the consumer stages):

```python
def alloc_padded_pinned(shape, dtype, stride=None, *, align) -> torch.Tensor:
    nbytes = required_storage_nbytes(shape, stride, dtype)
    storage = torch.empty(pad(nbytes, align), dtype=torch.uint8, pin_memory=True)  # torch caching host allocator
    t = torch.empty(0, dtype=dtype)
    t.set_(storage.untyped_storage(), 0, shape, stride or contiguous_strides(shape))
    return t     # exact logical view; storage padded so pwrite/pread of full length is in-bounds
```

### Efficiency
Zero extra memcpys on the common path (pinned direct O_DIRECT); GIL released during waits; one writer thread + main-thread reads (no lock contention on handles); no GEMM/launch changes (nothing wired yet).

### Validation (Stage 1 gate — isolated, unit tests sufficient)

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
export AIO_HOME=$PWD/.aioenv CPATH="$AIO_HOME/include:${CPATH:-}" \
       LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
ASYM_NVME_PATH=/scratch_local/user_data/shutian/kevin/cache/asym_nvme_test \
.venv/bin/python -m pytest tests/training/test_nvme_store.py -q
```

Required tests: bf16/fp32 roundtrip below/at/above 1 MiB (logical bytes exact, padding invisible); strided-tensor roundtrip restores exact strides; two blobs at different offsets in one arena file, no cross-blob corruption; spill→fetch with `ready_event` ordering (write a CUDA tensor D2H then spill; verify bytes); fetch-before-write-durable blocks correctly and cancel path returns the live buffer; prefetch→drain ledger reconciliation with 3 in-flight reads; arena reset-when-empty across two simulated microbatches; backpressure blocks at the cap; `get_nvme_store()` returns None and imports nothing without env; writer thread clean shutdown. Plus: with `ASYM_NVME_ROLES` unset, `python -c "import asym_gemm.training.nvme_store"` must not import deepspeed (assert via `sys.modules`).

### Risks / watch
- aio handle thread-ownership: submission from writer thread + wait on same thread is the tested pattern; never share a handle across threads (rule 4).
- Arena sizing at extreme seq (s60k+ spill can exceed 1 TiB across accumulation steps if blobs outlive a microbatch) — `ASYM_NVME_ACTIVATION_ARENA_BYTES` knob + loud error; revisit if hit.
- O_DIRECT on `/scratch_local` (ext4 on md0) is what DeepSpeed's zero3 NVMe backends already use there — no new filesystem risk.

---

## Stage 2 — Backend tokens, env plumbing, profile counters, compare gate

**Scope (exact sites):**
- `scripts/lf/profile_lora_lf_test_source.sh`: `append_backend_spec` `:956-978` — add three arms `asym_cpuadamwds_panvme) backend=asym_cpuadamwds_panvme ;;` etc. + extend the die message `:978`; `backend_gpu_count` `:789-795` — add tokens to the asym line (1 GPU) + die list; `cpuadam_backend_for_label` `:918-923` — map all three → `deepspeed`; per-job derivation `~:3094-3110` follows automatically from `cpuadam_backend_for_label`; in the `run_env` build (`:3387` block), export `ASYM_NVME_ROLES`/`ASYM_NVME_PATH` derived from the token:

```bash
case "${backend}" in
  asym_cpuadamwds_panvme)  job_nvme_roles="base_weight" ;;
  asym_cpuadamwds_actnvme) job_nvme_roles="activation" ;;
  asym_cpuadamwds_bothnvme) job_nvme_roles="base_weight,activation" ;;
  *) job_nvme_roles="" ;;
esac
run_env+=( "ASYM_NVME_ROLES=${job_nvme_roles}" "ASYM_NVME_PATH=${ASYM_NVME_PATH:-/scratch_local/user_data/shutian/kevin/cache/asym_nvme}"
           "ASYM_GEMM_LF_CONFIG_ASYM_NVME_ROLES=${job_nvme_roles}" "ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH=..." )
```

- `scripts/lf/run_lf_lora_sft.sh`: clone the `asym_cpuadamwds` arm (`:352-358`) three times, each additionally `export ASYM_NVME_ROLES=...` (and default `ASYM_NVME_PATH` if unset), still setting `BACKEND=asym`; extend die `:369`. Mirror into `ASYM_GEMM_LF_CONFIG_ASYM_NVME_*` next to `:2297`.
- `scripts/lf/run_lf_profiled_train.py`: `:579-582` — `is_asym_deepspeed_cpuadamw = backend in {"asym_cpuadamwds","asym_cpuadamwds_panvme","asym_cpuadamwds_actnvme","asym_cpuadamwds_bothnvme"} or (...)`; `_config_from_args` — add `asym_nvme_roles`/`asym_nvme_path` keys via the `ASYM_GEMM_LF_CONFIG_*`-or-default pattern (`:732-734` as template); `report()` — add sibling block next to `activation_offload` (`:2846`):

```python
"asym_nvme": _asym_nvme_summary_from_model(),   # {"enabled":bool,"roles":[...],"path":str, "store": store.stats.as_dict(),
                                                #  per-role: bytes_written/read, ops, wait_ms, inflight_peak_bytes, arena_peak_bytes}
```

`_asym_nvme_summary_from_model()` = `from asym_gemm.training.nvme_store import get_nvme_store` → `{"enabled": False}` when None. Extend the aggregates tail of `_activation_offload_counters_from_model` (`:2265-2281`) with `total_nvme_spilled_bytes`, `total_nvme_bytes_written/read`, `max_nvme_inflight_bytes`, `total_nvme_fetch_wait_ms` summed from row stats (keys arrive automatically once Stage 4/5 add them to `snapshot()`).
- `scripts/lf/postprocess_lf_profile_artifacts.py`: emit `asym_nvme.csv` (flatten the `asym_nvme` block; one row per role) next to `_asym_cpu_adamw_rows` (`:378`); add an NVMe line to `memory.md` (`_source_memory_markdown` `:1803`).
- **New `scripts/lf/compare_nvme_profiles.py`** — clone of `compare_liger_loss_profiles.py` with metric extractors swapped:

```text
--baseline DIR --candidate DIR --target {no_change, base_weight_cpu, activation_cpu, maxseq}
--memory-metric  (dotted path into source_profile.json; step_samples.<col> reads median of measured rows)
--min-memory-drop-gib / --min-memory-drop-pct   (no_change: --max-memory-drift-gib instead)
--max-step-ratio 1.05 --max-forward-ratio 1.05 --max-backward-ratio 1.05   (capacity mode: 1.10)
--expect-nvme-role ROLE          (asserts candidate asym_nvme.roles contains ROLE and bytes_written>0 read>0)
Checks: required artifacts exist (source_profile.json, step_samples.csv, memory.md, asym_cpu_adamw.csv,
        asym_nvme.csv for candidates); finite losses; measured_steps>=5; config.asym_nvme_roles matches.
Output: {"ok":bool,"failures":[...],"metrics":{...}} ; SystemExit(2) on failure.
```

### Runtime validation hook (config only — Stage 2 moves no tensors)
`get_nvme_store()` validates at construction: path exists & writable; roles ⊆ {base_weight, activation}; refuse when `WORLD_SIZE>1` or `ACCELERATE_USE_DEEPSPEED`/`--deepspeed` env present (single-process single-GPU only); `activation` role warns (not fails) when no `ASYMM_*_ACT_OFFLOAD`/`ASYMM_LAYER_GC` flag is on.

### Validation (Stage 2 gate — e2e no-change, both directions)

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
bash -n scripts/lf/profile_lora_lf_test_source.sh scripts/lf/run_lf_lora_sft.sh
# (a) baseline token untouched:
GPU_POOL=<free-gpu> OUTPUT_ROOT=profiling_nvme/stage2_nochange PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds|norecomp|ligerloss1 ; 8192|8|1 ; none|false|true|false|true|true' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
# (b) new token, roles wired but Stage 3/4 not yet implemented → must behave identically
#     EXCEPT config.asym_nvme_roles + asym_nvme.enabled=true with zero bytes:
GPU_POOL=<free-gpu> OUTPUT_ROOT=profiling_nvme/stage2_token PROFILERS=source PLOT=false \
PREPARE_DATASETS=false WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds_actnvme|norecomp|ligerloss1 ; 8192|8|1 ; none|false|true|false|true|true' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline profiling_nvme/stage2_nochange/<run_dir> --candidate profiling_nvme/stage2_token/<run_dir> \
  --target no_change --memory-metric memory.gpu.peak_allocated_hbm_bytes --max-memory-drift-gib 0.5 \
  --max-step-ratio 1.02 --max-forward-ratio 1.02 --max-backward-ratio 1.02
```

Accept: identical HBM/RSS/latency within drift bounds; `source_profile.json.config.asym_nvme_roles=="activation"` in (b); `asym_nvme.enabled==true, bytes_written==0`; `asym_nvme.csv` written; the compare tool demonstrably fails when pointed at a mismatched pair.

### Risks / watch
- The `_both.sh` fork: apply the same case-arm edits or accept that NVMe tokens only run under `_source.sh` for now (recommended: edit both, they're the same driver; keep the diffs identical).
- `existing_profile_complete` reuses cached run dirs keyed by backend `$3` → token-distinct dirs; still pass `--overwrite true` on gate runs.

---

## Stage 3 — `panvme`: base weights → NVMe

**Scope:** `asym_gemm/training/host_weight.py` (HostWeight surgery), new `asym_gemm/training/base_weight_pager.py`, `asym_gemm/integrations/lf.py` (one registration call at the end of `apply_lf_asym_lora`, ~`:2410`, after all wrapping), `asym_gemm/training/qwen3_moe.py` (eager fine-grained split — see risks).

### Design

Weights are hot (every layer, fwd+bwd, every step), read-only, and their per-step consumption order is **fixed after step 1**. So: spill all eligible weights at registration; keep a bounded pinned **shape-ring cache** (all layers share a handful of shapes → per-shape rings, zero fragmentation); prefetch by the observed consumption trace; evict = return ring buffer, gated on a CUDA event.

```python
# host_weight.py — surgery (minimal; everything else unchanged)
class HostWeight:
    # NEW fields set by pager at registration: _pager=None, _pager_key=None
    @property
    def weight(self) -> torch.Tensor:
        if self._tensor is None:                      # spilled
            self._tensor = self._pager.materialize(self._pager_key)   # blocking on miss; hit = O(1)
        if self._pager is not None:
            self._pager.on_touch(self._pager_key)     # trace record + prefetch advance + evict-check
        return self._tensor
    tensor = weight                                    # same property
    # nbytes/shape/dtype/metadata properties already read self._metadata → NO fetch on reporting paths.
    # NEW: def _detach_resident(self): t, self._tensor = self._tensor, None; return t   (pager-only)
```

```python
# base_weight_pager.py
class BaseWeightPager:
    """Owns residency of registered HostWeights. Main-thread only (matches training loop)."""
    def __init__(self, store: NVMeStore, *, cache_bytes=_env_int("ASYM_NVME_BASE_WEIGHT_CACHE_BYTES", 8<<30),
                 prefetch_depth=_env_int("ASYM_NVME_PREFETCH_DEPTH", 2)):
        self._entries: dict[key, _Entry] = {}   # key=stable module path; _Entry: hw, ref(BlobRef), shape/dtype/stride,
                                                # resident buffer|None, last_use_event|None, trace positions
        self._rings: dict[(dtype, shape), list[padded_pinned_buf]] = {}   # per-shape free buffers
        self._trace: list[key] = []; self._trace_frozen = False; self._cursor = 0

    def register(self, key, hw: HostWeight):
        if hw.nbytes < store.cfg.min_swappable_bytes: return          # norms etc. stay resident
        t = hw.weight  # still resident (registration happens right after adoption, pre-training)
        padded = alloc_padded_pinned(t.shape, t.dtype, align=store.align)
        padded.copy_(t)                                               # one-time CPU copy into padded storage
        ref = store.spill("base_weight", padded, ready_event=None, on_done=self._recycle_into_ring)
        hw._pager, hw._pager_key = self, key
        hw._tensor = None                                             # free the original pinned home NOW
        self._entries[key] = _Entry(hw=hw, ref=ref, ...)
        # NOTE: original home freed immediately; the padded copy is freed by on_done after the write
        # → transient = one weight extra, same as adoption-time pin_memory() transient.

    def materialize(self, key) -> torch.Tensor:
        e = self._entries[key]
        if e.buf is None:
            e.buf = self._take_ring_buffer(e.shape, e.dtype)          # may event-gated-evict a victim
            store.fetch_into(e.ref, e.buf)                            # blocking miss (or prefetch-arrived)
            self.stats.misses += 1
        return e.buf_view                                             # exact-shape view of padded buf

    def on_touch(self, key):
        self._record_trace(key)                                       # freeze after first full step repeats
        if self._trace_frozen:
            for nxt in self._next_k_in_trace(key, self.prefetch_depth):
                if self._entries[nxt].buf is None and not in_flight(nxt):
                    buf = self._take_ring_buffer(...); store.prefetch_into(self._entries[nxt].ref, buf)
        self._evict_to_budget()

    def _evict_to_budget(self):
        while resident_bytes > cache_bytes:
            victim = max(evictable_entries, key=self._next_use_distance)   # farthest reuse (Belady on the trace);
            ev = torch.cuda.Event(); ev.record()                      # all victim-consuming launches already enqueued
            self._quarantine.append((victim.buf, ev)); victim.buf = None
        self._sweep_quarantine()                                      # ring-return buffers whose event.query() is True
```

Registration walk (end of `apply_lf_asym_lora`, env-gated):

```python
if (store := get_nvme_store()) and store.has_role("base_weight"):
    pager = BaseWeightPager(store)
    for name, mod in model.named_modules():
        hw = getattr(mod, "host_weight", None)         # AsymFrozenLinear / AsymGroupedFrozenLinear / (embeds excluded)
        if isinstance(hw, HostWeight) and component_eligible(name):   # default: attn + mlp_dense + routed experts;
            pager.register(name, hw)                                  # embed_tokens/norms excluded (hot CPU-side ops, small)
    model._asym_base_weight_pager = pager              # summary hook for profiler
```

Key correctness points (each maps to a verified fact):
- The fwd+bwd interleaved trace makes Belady eviction exact: after forward, late layers (reused first in backward) rank nearest — the farthest-reuse rule automatically keeps them and drops early layers. First step (trace unfrozen): pure miss-driven, synchronous — slow warmup step, excluded from measurement (WARMUP_STEPS≥2).
- Eviction event-gating (0.2): the evict-time `Event.record()` on the current stream is *after* every launch that consumed the victim (single-stream invariant) — buffer reuse only after `event.query()`.
- `.weight` lazy fetch covers every runtime reader, including the `is_pinned()` predicates (`qwen3_moe.py:1760`) that immediately precede kernel use. Reporting paths (`memory_attribution`, summaries) read `metadata`/`nbytes` properties → no fetch. Grep-verified reader list in 0.2 — audit any NEW `.weight` reader added later against "is this a compute path?".
- Kernel/launch efficiency: untouched — same `_dispatch_nt`/grouped calls, same one-slab semantics; the only new work is one `pread` per weight per phase, whole-tensor sized (50 MB–GB ⇒ far above `MIN_AIO_BYTES`).

### Validation (Stage 3 gate)

Unit (`tests/training/test_base_weight_pager.py`): register→spill frees `_tensor`; materialize roundtrip bit-exact vs pre-spill clone (bf16, 2D + grouped 3D); trace freeze after simulated fwd+bwd order; eviction picks farthest-reuse and respects the event gate (mock event); ring reuse across same-shape layers; `min_swappable_bytes` exclusion; disabled env = HostWeight identical to today (run existing `tests/training/test_cpu_resident_frozen_base.py` + `test_lf_qwen3_asym_backend.py` unmodified).

E2E (dense q3-32b — the validated `none|F|T|F|T|T` config; sequential, `kill -TERM` only, never `-9`):

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage3_panvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=3 MAX_STEPS=8 \
RUNS='q3-32b|1 ; asym_cpuadamwds|norecomp|ligerloss1 ; 8192|8|1 ; none|false|true|false|true|true || q3-32b|1 ; asym_cpuadamwds_panvme|norecomp|ligerloss1 ; 8192|8|1 ; none|false|true|false|true|true' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true

.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline <asym_cpuadamwds run dir> --candidate <asym_cpuadamwds_panvme run dir> \
  --target base_weight_cpu \
  --memory-metric step_samples.training_step_process_rss_peak_end_bytes \
  --min-memory-drop-gib 40 --max-step-ratio 1.05 --max-forward-ratio 1.05 --max-backward-ratio 1.05 \
  --expect-nvme-role base_weight
```

Accept: per-step RSS drops ≥ 40 GiB (q3-32b base ≈ 60 GiB minus the 8 GiB cache + rings; also check `memory_attribution` host_weight/cpu rows shrink accordingly); HBM unchanged; step/fwd/bwd ≤ 5%; `asym_nvme.base_weight.bytes_read ≈ 2×base×steps`; misses ≈ 0 after warmup (report `pager.misses_after_freeze` in the summary — must be ~0, else cache too small); losses finite and equal to baseline within noise; `RECORD_IO` `io_samples.csv` shows matching device-level reads (cross-check).

### Risks / watch
- **MoE + fine-grained** (`ASYMM_QWEN3_MOE_FINEGRAINED_OFFLOAD`): `_ensure_qwen3_moe_finegrained_bases` (`qwen3_moe.py:2492-2515`) lazily slices the fused gate_up HostWeight → force it eagerly during conversion when both flags are on, register the SPLIT bases, and leave the fused parent unregistered-and-freed (else double residency + a surprise mid-step fetch). Dense q3-32b (this stage's gate) is unaffected. Until implemented, `panvme+moefg` must raise loudly at registration.
- Non-bf16 precisions build `QuantizedHostWeight` caches from the weight (`frozen_linear.py:372`) — panvme v1 gates on `precision=="bf16"` (the training path); raise otherwise.
- First-step latency spike (synchronous misses while tracing) — fine under WARMUP_STEPS≥3; do not gate on step-1 timing.
- Dynamic module order (e.g. layers skipped under some config) would break the frozen trace → on a trace mismatch, log + fall back to miss-driven synchronous fetches for the rest of the run (correct, slower), and count `trace_disabled=1` in the summary.
- If step time at s8192/b8 is short enough that 120 GiB/step exceeds overlap capacity, the ≤5% gate fails honestly → rerun gate at the capacity operating point (longer seq) and reclassify panvme as capacity-mode for short-seq. Decide from the measured run, not up front.

---

## Stage 4 — `actnvme` part 1: Substrate A (saved-tensor wrappers — the layer-GC bulk)

**Scope:** `decoder_activation_offload.py`, `attention_activation_offload.py` (wrapper part), `linear_attention_activation_offload.py` — identical ~30-line diff in each (deliberately NOT deduping the three wrappers first; smallest possible behavioral diff, dedupe later if desired). Plus counter keys.

### Design

Substrate A buffers have exactly one writer (the D2H copy, event-tracked) and one reader (the H2D unpack copy) — no CPU-kernel consumers (0.3). So the spill lifecycle is the simple one:

```python
# each wrapper file — handle gains two fields (dataclass already mutable):
@dataclass
class _SavedTensorOffloadHandle:
    ...
    spill_ref: Any = None            # BlobRef when spilled
    # tensor: set to None once the buffer is handed to the writer (frees it on write completion)

# _pack (after the existing copy_ + ready_event lines, decoder_activation_offload.py:191-198):
handle = _SavedTensorOffloadHandle(tensor=cpu, ..., ready_event=ready_event)
store = get_nvme_store()
if store is not None and store.has_role("activation") and nbytes >= store.cfg.min_swappable_bytes:
    # cpu must have PADDED storage: _pack's alloc switches to alloc_padded_pinned(shape, stride, align)
    handle.spill_ref = store.spill("activation", cpu, ready_event=ready_event,
                                   on_done=lambda buf, ref: None)    # buffer freed by dropping our ref:
    handle.tensor = None                                             # writer thread holds the only ref until durable,
    self.nvme_spilled_bytes += nbytes                                # then torch's caching host allocator recycles it
    # cpu_owned_bytes accounting: decrement at spill enqueue (CPU bytes now owned by the bounded
    # in-flight window, reported separately as nvme_inflight_bytes)

# _unpack (before the existing empty_strided+copy, :224-236):
if packed.spill_ref is not None:
    bounce = alloc_padded_pinned(packed.original_shape, packed.original_dtype,
                                 packed.original_stride, align=store.align)
    t0 = perf_counter(); store.fetch_into(packed.spill_ref, bounce); self.nvme_fetch_wait_ms += ...
    store.blob_done(packed.spill_ref)
    packed.tensor, packed.spill_ref = bounce, None
# existing path continues unchanged: ready_event sync (None here), empty_strided HBM, copy_
```

Correctness points:
- pwrite is gated on `ready_event` inside the writer (Stage 1) — never reads bytes still in D2H flight.
- Backpressure (`max_inflight_spill_bytes`, default 8 GiB) bounds pinned growth when forward outruns 14 GB/s writes — `_pack` blocks in `store.spill()` past the cap; that stall is the honest capacity trade and is reported (`nvme_spill_backpressure_ms`).
- Fetch-before-durable (a tensor unpacked immediately after pack — tiny graphs, sanity paths): `fetch_into` waits on `ref.durable` or cancels a still-queued spill (Stage 1); correctness never depends on timing.
- v1 unpack is a synchronous read (size/26 GB/s ⇒ ~40 ms/GiB against ≥100 ms/layer backward at capacity workloads). v2 (flag `ASYM_NVME_ACT_PREFETCH_DEPTH`, default 0): backward consumes handles in exact reverse pack order → at each unpack, `prefetch_into` the next K refs from the recorded pack list into ring buffers; `drain_reads()` at the next unpack. Ship v1, measure, then enable v2.
- New stats keys in `snapshot()` (auto-flow to profiler rows): `nvme_spilled_bytes`, `nvme_bytes_written`, `nvme_bytes_read`, `nvme_fetch_wait_ms`, `nvme_spill_backpressure_ms`, `nvme_inflight_peak_bytes`. Aggregates added in Stage 2 pick them up.

Efficiency: no change to what is offloaded or to any kernel; IO is whole-tensor (the wide [M,I]/[M,H] tensors, tens of MB–GiB each — ideal O_DIRECT sizes); one writer thread coalesces all spills; zero extra CPU memcpys (the padded pinned buffer IS the D2H destination and the pwrite source).

### Validation (Stage 4 gate — capacity mode)

Unit (`tests/training/test_actnvme_saved_tensor.py`): pack→unpack roundtrip bit-exact for contiguous + strided bf16/fp32 tensors with NVMe on; graph-level test — small model, `DecoderSavedTensorOffloadWrapper` installed, loss/grad **bit-identical** with `ASYM_NVME_ROLES=""` vs `"activation"` (same seed); existing `tests/training/test_decoder_activation_offload.py`, `test_decoder_layer_glue_gc.py`, `test_linear_attention_activation_offload.py` pass unmodified with NVMe off AND on.

E2E — same config pair as Stage 3, token `asym_cpuadamwds_actnvme`, at a **CPU-heavy operating point** (raise seq until baseline activation CPU is large; start s16384|b8):

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage4_actnvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=3 MAX_STEPS=8 \
RUNS='q3-32b|1 ; asym_cpuadamwds|norecomp|ligerloss1 ; 16384|8|1 ; none|false|true|false|true|true || q3-32b|1 ; asym_cpuadamwds_actnvme|norecomp|ligerloss1 ; 16384|8|1 ; none|false|true|false|true|true' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true

.venv/bin/python scripts/lf/compare_nvme_profiles.py \
  --baseline <actnvme-less dir> --candidate <actnvme dir> --target activation_cpu \
  --memory-metric activation_offload.max_cpu_peak_bytes_live \
  --min-memory-drop-pct 50 --max-step-ratio 1.10 --max-forward-ratio 1.10 --max-backward-ratio 1.10 \
  --expect-nvme-role activation
```

Accept: `activation_offload.max_cpu_peak_bytes_live` (Substrate-A portion; check `cpu_peak_by_tag` for `decoder.saved.*`) drops ≥ 50%; per-step RSS drops accordingly; HBM unchanged; losses match baseline bit-for-bit at step 1 (same seed — spill/fetch is byte-exact); step ≤ 1.10× (capacity gate); `nvme_bytes_written ≈ nvme_bytes_read ≈` baseline Substrate-A offloaded bytes/step. Then the **capacity demo**: raise seq on the candidate until baseline CPU-OOMs (or hits the [[gb200 ceiling|~0.8 TB]]) and candidate still trains — record max-seq both ways; that number is the stage's headline.

### Risks / watch
- Verify the per-step *write* volume fits the step: if `nvme_spill_backpressure_ms` dominates, either raise `max_inflight_spill_bytes` or accept the stall as the capacity price — report it, don't hide it.
- Attention wrapper `require_grad=True` default means fewer/different tensors spill vs decoder — per-tag counters (`offload_bytes_by_tag`) tell you exactly what moved; verify no tag unexpectedly stopped offloading.
- Torch caching-host-allocator growth: padded allocations change the size-class mix; watch `rss` for allocator bloat (mitigation: round padded sizes to 2 MiB classes).

---

## Stage 5 — `actnvme` part 2: Substrate B (ActivationOffloadManager — attn U/S, experts, fine-grained)

**Scope:** `asym_gemm/training/activation_offload.py` (+ ~5-line seal calls in `attention_activation_offload.py` and, when enabling MoE, `qwen3_moe.py` / `dense_mlp_finegrained.py` Function forwards). Needed for MoE-expert and attn-U/S bytes; dense layer-GC configs are already covered by Stage 4.

### Design (the seal-point protocol)

A Substrate-B buffer may be consumed on CPU by async GPU kernels **in forward, after `offload()` returns** (0.3). So spill cannot free at pack time; it needs a *seal*: the moment after the last forward consumer of that handle has been enqueued.

```python
# activation_offload.py
class ActivationOffloadManager:
    def __init__(self, *, pin_memory=True):
        ...
        self._spill: dict[int, _SpillState] = {}     # id(handle) -> state; handle alive on ctx until backward

    def offload(self, tensor, tag):                  # unchanged except:
        handle = ...                                 # buffer from _alloc_cpu → switch to padded storage alloc
        if _spill_eligible(handle):
            self._spill[id(handle)] = _SpillState(pending_seal=True, ref=None)
        return handle

    def seal(self, *handles):
        """Call at the END of a Function.forward (all forward consumers of these handles are
        now enqueued on the current stream). Records ONE event and hands the buffers to the writer."""
        ev = torch.cuda.Event(); ev.record()
        for h in handles:
            st = self._spill.get(id(h))
            if st is None or not st.pending_seal: continue
            st.ref = get_nvme_store().spill("activation", h.tensor, ready_event=ev,
                                            on_done=self._on_durable(h))
            st.pending_seal = False
    def _on_durable(self, h):
        def cb(buf, ref):
            # writer thread: bytes durable AND ev synced (kernel reads done: read-read overlap with
            # pwrite is fine; ev covers the D2H write ordering AND, because it was recorded after the
            # last consumer launch, buffer reuse is now safe) → return to pool for reuse:
            self._release_spilled_buffer(h)          # pops _active_cpu_bytes[ptr], _return_cpu(buf-as-shape)
        return cb

    def _ensure_local(self, handle):
        st = self._spill.get(id(handle))
        if st is None or st.ref is None: return       # not spilled → handle.tensor valid
        bounce = _alloc_cpu(handle.original_shape, handle.original_dtype, pin_memory=True)  # padded alloc
        store.fetch_into(st.ref, bounce); store.blob_done(st.ref)
        object.__setattr__(handle, "tensor", bounce)  # frozen dataclass: sanctioned single mutation point
        self._mark_cpu_live(handle)                   # re-enters accounting under the new data_ptr
        del self._spill[id(handle)]

    # stage()/stage_rows()/stage_concat_columns(): first line becomes self._ensure_local(handle)
    # release_cpu(): if still spilled (never re-fetched), drop the ref (store.blob_done) instead of pool-return
```

Call-site diffs (one line each): `_AsymActivationOffloadLoRALinearFunction.forward` ends with `manager.seal(s_handle)` and, for the U source, the **shared-source seal** happens at the context's existing v_proj cache-clear point (`attention_activation_offload.py:477-479` — the structural "all q/k/v consumers done" marker): `self.manager.seal(shared.handle)`. Expert/fine-grained Functions: `manager.seal(x_cpu, gate_cpu, up_cpu, ...)` as the last forward statement, per Function. Enable per-engine behind `ASYM_NVME_ACT_SUBSTRATE_B=1` until each engine's Function has its seal audited.

Why the event algebra is right: `ev` (recorded at seal) orders after (a) the D2H `copy_` that filled the buffer and (b) every forward kernel launch that streams it — both enqueued earlier on the same stream. The writer thread syncs `ev` before pwrite (covers (a)); pool-return happens after pwrite completes, and any kernel still running past pwrite could only *read* concurrently with pread-free reuse — prevented because return-to-pool is also after `ev` synced (covers (b)). Backward consumers always go through `_ensure_local` first.

`_return_cpu`'s existing "handle→pool while handle still referenced" aliasing is avoided: we pop the accounting entry and null the spill state before return, and the handle's `tensor` is only replaced under `_ensure_local` — a handle is never simultaneously pool-free and accounting-live (the data_ptr-keying hazard from 0.3).

### Validation (Stage 5 gate)

Unit: attention LoRA linear fwd+bwd bit-exact NVMe-on vs off (existing `test_attention_activation_offload_lora.py` parametrized over the env); shared U spilled once and fetched once with q/k/v all consuming; `stage_rows`/`stage_concat_columns` on spilled handles; release-without-fetch path. E2E: rerun the Stage 4 command pair with `ASYM_NVME_ACT_SUBSTRATE_B=1` — additional `cpu_peak_by_tag` drops for `*.U`/`*.S` tags, same gate thresholds; for MoE, one `q3-30b-a3b` run with the expert engine's policy config before claiming MoE support.

### Risks / watch
- Every Substrate-B engine (qwen3 experts, llama4, shared-MLP, dense-MLP-finegrained) needs its own seal audit — do them one engine at a time behind the env flag; an unsealed handle is safe (never spills), so partial coverage degrades to status quo, not corruption.
- `empty_cpu`-created handles filled by CPU GEMM output (not via `offload()`) are NOT spilled in v1 — visible in `cpu_peak_by_tag`; extend only if a profile shows them dominant.
- The `object.__setattr__` on the frozen handle is contained to `_ensure_local`; alternatively flip the dataclass to `frozen=False` (verified: nothing hashes handles) — decide at implementation, keep to one mechanism.

---

## Stage 6 — `bothnvme`: compose + hero max-seq

No new mechanism: token enables both roles on the one store (shared handles/threads; base_weight per-blob files + activation arena coexist). Sizing rule asserted at startup: `base_weight_cache + prefetch rings + activation inflight cap + pool limit < free host RAM headroom`.

Gate: on q3-32b (then q3-30b-a3b), demonstrate max trainable seq: `bothnvme > actnvme ≥ baseline`, at fixed batch; verify actual `input_ids` length in `train.log` (stale-dataset trap); report HBM peak, per-step RSS, per-role NVMe bytes + wait-ms, step time, and the overlap fraction (`1 − nvme_wait/step`). Compare vs the SuperOffload/zero3 ceilings from the existing baselines for the paper's capability table.

---

## Deferred (unchanged)
- DeepSpeed-ZeRO-owned backend behind the same store API (only if multi-GPU/ZeRO becomes a target; ownership must not stack with `HostWeight`/`AsymCPUAdamW`).
- GDS/GPU-direct: hardware not available; CPU-staged path is the design.
- Unsloth-GC boundary offload ([M,H]×L, the 293 GiB CPU item at s60k under `unsloth` recompute configs) → a third role later; lives in LlamaFactory's checkpointing, not asym_gemm.
- Gradient/optimizer-state NVMe: dropped (LoRA-tiny).

## Global run rules (every e2e gate)
Heavy offload runs **sequentially** (host RAM 665–802 GiB observed); stop with `kill -TERM` (never `-9` — corrupts the DeepSpeed cpu_adam JIT cache); `PREPARE_DATASETS=true` on the first run of a workload and **verify the real `input_ids` length in `train.log`**; measure from `step_samples.csv` measured rows (warmup excluded); `.aioenv` env is exported by `run_lf_lora_sft.sh` automatically — export it manually for any direct pytest/python use of the store.

## Implementation order

```text
Stage 1: nvme_store.py substrate            (isolated; unit + AIO smoke gate)
Stage 2: tokens + env + counters + compare  (e2e no-change gate, both tokens)
Stage 3: panvme                             (throughput-preserving gate, ≤5%)
Stage 4: actnvme / Substrate A              (capacity gate ≤10% + max-seq demo)
Stage 5: actnvme / Substrate B              (per-engine seal rollout, same gate)
Stage 6: bothnvme hero                      (max-seq capability table)
```

Ship panvme before actnvme: it proves the substrate + pager + gating loop on the easy, cacheable, read-only target and frees the CPU that Stages 4–6 spend on activations.

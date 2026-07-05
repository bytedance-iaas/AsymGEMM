# AsymGEMM NVMe Offload — Execution-Ready Staged Plan (v6, re-verified 2026-07-05 @ HEAD 39dfffc)

This refines `nvme_offload.md` (v5) into a per-stage, third-person-executable plan. Every code anchor
below was re-verified against the working tree at HEAD `39dfffc` (v5 was verified @ `d2feadf`). Each
stage gives: **What → Scope (exact file:line) → Complete pseudocode → Efficiency → Validation (exact
commands) → Risks**. Correctness never depends on prefetch; every rung is env-gated with a sync fallback;
`off` (no `*nvme` token) is byte-identical.

Goal recap: three opt-in backend tokens on ONE local NVMe store reusing DeepSpeed's AIO engine.
- `asym_cpuadamwds_actnvme` — **activation spill** (the capacity lever). Primary target = the unsloth-GC
  cross-layer boundary stream (**Substrate A**, ~342 GiB @ q3-32b s70k); secondary = the fine-grained
  engine handles (**Substrate B**). One watermark governor, FIFO spill / LIFO consume.
- `asym_cpuadamwds_panvme` — frozen base weights (`HostWeight`) → NVMe with a trace-prefetch pinned cache.
  Frees 61 GiB (q3-32b) / 105 GiB (q3.5-35b-a3b) of CPU.
- `asym_cpuadamwds_bothnvme` — both roles, compound, for the max-seq hero result.

Delivery ladder (locked): **v1 SYNC (bit-exact) → v2 async writer → v3 reverse-order read prefetch →
v4 optional H2D stream.** Stages map: 0 census · 1 store · 2 tokens · 3 governor+Substrate A (SYNC, core) ·
4 Substrate B seals · 5 async+prefetch · 6 MoE seal · 7 panvme · 8 bothnvme hero.

---

## ⛔ HARD RUN CONSTRAINTS (non-negotiable — read before launching ANY process)

These govern **every** training/profiling process this project runs, at every stage. The NVMe design multiplies
host-CPU-RAM pressure (pinned governor buffers up to the budget + `HostWeight` homes + CPU pools), so a process
launched without these guards can hard-OOM the box, corrupt the DeepSpeed JIT cache (`kill -9`-class events), or
let the kernel OOM-killer take down an unrelated process/session. **All four guards are built into
`scripts/lf/run_lf_lora_sft.sh` (defaults ON); the profile drivers wrap through it and inherit them.** Verified
@ HEAD `39dfffc`.

- **HC1 — Launch ONLY through the guarded scripts.** Every run goes through
  `scripts/lf/profile_lora_lf_test_source.sh` (or `_both.sh`) **or** `scripts/lf/run_lf_lora_sft.sh`. **NEVER**
  launch training directly (`python scripts/lf/run_lf_profiled_train.py …`, `llamafactory-cli`, `python -m
  llamafactory …`, bare `deepspeed …`) — that bypasses HC2-HC4 and *will* cause serious issues. New unit tests
  (`pytest tests/training/...`) are the only exception (they allocate little and touch no training tree); their
  own e2e gates still go through the scripts.
- **HC2 — Host memory = the 2 Grace CPU NUMA nodes; GPUs are compute-only, NOT a memory pool.** The scripts wrap
  the process in `numactl --membind=0,1 --cpunodebind=0,1` (`NUMACTL_ENABLE=1`, `NUMACTL_MEMBIND=0,1`,
  `NUMACTL_CPUNODEBIND=0,1`, `NUMACTL_MODE=membind`; `run_lf_lora_sft.sh:57-62,:2513-2530`). All host
  allocations — pinned NVMe IO/governor buffers, `HostWeight`, the `_CPU_BUFFER_POOL` — land on the **two Grace
  LPDDR nodes (0,1)**, never GPU HBM. **Do not disable numactl and do not add a GPU-HBM NUMA node to
  `--membind`.** (The watchdog's free-memory measurement is only correct because the process is membound to nodes
  0,1 — HC4 depends on HC2.)
- **HC3 — `oom_score_adj = 1000` on the training tree.** `TRAIN_OOM_SCORE_ADJ=1000` (default) is written to
  `/proc/self/oom_score_adj` for the launched tree (`run_lf_lora_sft.sh:219,:1695/:1711`), so if host RAM is ever
  exhausted the kernel OOM-killer targets **this run first** — not sshd, the watchdog, or another job. Never
  lower it.
- **HC4 — Host-mem watchdog ON, floor 35 GB.** `HOST_MEM_WATCHDOG=true`, `HOST_MEM_WATCHDOG_FLOOR_GB=35`
  (`:220-221`) — polls **per-CPU-NUMA-node** free memory (`:1629`) and gracefully interrupts (SIGSTOP→SIGINT→
  SIGCONT, then escalates after `KILL_GRACE_SECONDS=60`) **before** the kernel OOM-killer fires. Leave it ON for
  every run — it is also the baseline's OOM referee for the max-seq gates.

**Corollaries for this design (enforced in the stages):** (a) the actnvme `auto` budget = 0.85×(CPU-node
MemAvailable) is only meaningful under HC2 — **always set `ASYM_NVME_ACT_CPU_BUDGET_BYTES` explicitly** for gate
runs; (b) size ALL pinned pools to fit inside the 2-node LPDDR minus the floor — the Stage-8 startup ledger
asserts `governor.hi + pager.cache_bytes + ASYM_EXPACT_CPU_POOL_MAX_BYTES + prefetch(act+base) +
max_inflight_spill_bytes + 35 GiB floor + slack < MemTotal (1325 GiB)`; (c) the NVMe store path stays on
`/scratch_local` (`md0`, not RAM/tmpfs) so spilled bytes actually leave host RAM.

---

## 0′. Corrections & confirmations from re-verification (read this first)

**Confirmed exactly** (line-for-line vs v5): the `ActivationOffloadManager` internals in
`asym_gemm/training/activation_offload.py` (pool `:10`, `_alloc_cpu :74-86`/silent-unpinned-fallback `:85-86`,
`_return_cpu :89-103`, `CPUActivationHandle :106-116` with live `nbytes` property, `_active_cpu_bytes :164`,
`_pending_cpu_ready_events :166`, `offload` D2H-event record `:194-197`, `wait_cpu_ready :230-235`,
`release_cpu` internal `wait` `:318`, `_mark_cpu_live :331`, `adopt_cpu` foreign fast-path `:220-228`, **no
`_pop_active` exists**); the LF `checkpointing.py` boundary Function (`:78-119`, D2H `:93`, `save_for_backward
:97`, H2D `:106`, `save_on_cpu :109-111`, HBM-diagnostic `:62-75/:90`); `host_weight.py` (`_tensor` sole write
`:227`, all paged-property sites `:245/:272/:276/:280/:292/:296/:300/:315`, `nbytes/is_pinned/metadata` already
metadata-backed); all three fine-grained seal points (**dense `:300`**, **moe `:706`**, attention forward
`:632-653`); all script anchors (§0.6 unchanged despite the watchdog commit — the `BACKEND` case sits above the
watchdog function bodies).

**Corrected / newly pinned:**

| # | v5 said | Reality @ HEAD | Impact |
|---|---|---|---|
| C1 | DeepSpeed v0.19.2 | Editable `../deepspeed` self-reports `0.19.2` (`version.txt`), `git describe`=`v0.19.1-15-g…`; LF venv `dist-info`=`0.19.1`. **AIO API byte-identical across both** (empty `git diff` over `csrc/aio/`, `op_builder/async_io.py`). | None — API stable. Note the version ambiguity, don't chase it. |
| C2 | `wait()` idle aborts; separate handles | **All AIO asserts are LIVE** (`-O0`, no `-DNDEBUG`). `async_pwrite/pread` return `-1` on failure **without scheduling**; a following `wait()` then aborts the process. `pread` also asserts `offset+len ≤ filesize` (aborts past EOF). | **Store MUST check `rc == 0` before every `wait()`**, and read back the exact padded write length. Baked into Stage-1 pseudocode. |
| C3 | `get_alignment()==2048` | `get_alignment() = intra_op_parallelism × 512` = **2048 @ intra=4**. Padding to it **auto-satisfies** the `len % intra_op == 0` divisibility check too. | One padding rule (`_pad(n, store.align)`) covers both O_DIRECT and divisibility. Read `align` from the handle, never hardcode. |
| C4 | watchdog uncommitted, floor 50 GB | **Committed** in `39dfffc`. `HOST_MEM_WATCHDOG=true`, `HOST_MEM_WATCHDOG_FLOOR_GB` default **35** (`run_lf_lora_sft.sh:220-221`); primary metric = **per-NUMA-node free memory** (`:1629-1631`), global `MemAvailable` only a fallback (`:1639`); SIGSTOP→SIGINT→SIGCONT escalation. | Stage-8 RAM ledger uses floor **35** not 50. Baselines still die here first → the actnvme capacity win. |
| C5 | hook `on_offload` in `offload()/adopt_cpu()/empty_cpu()` | `offload()` **delegates to** `empty_cpu()`; hooking all three double-counts. `_mark_cpu_live` is called at exactly two sites: `empty_cpu:184`, `adopt_cpu:227`. | Hook `on_offload` **only** at those two sites (co-located with `_mark_cpu_live`), and NOT inside `_mark_cpu_live` itself (the governor re-calls it on fetch). |
| C6 | attention `closed` flag; direct read `:715` | `_SharedActivationSource` (`:417-440`) has `released`+`refcount`, **no `closed`** — must add it. `acquire_source` is a method at **`:456-480`** (v5 cited `:417-440`). Backward `u_handle.tensor` read at `:715` is the **only** uncovered direct read in the whole tree (no `wait_cpu_ready` anywhere in attention backward). | Add `closed`; add one `ensure_local(u_handle)` before `:715`. Everything else is `stage*/wait_cpu_ready`-covered. |
| C7 | MoE x reads need care | **Exactly one** `wait_cpu_ready(ctx.x_cpu)` at `:968` dominates both block loops (reads `:1021`,`:1106`); the else-path uses per-read waits (`:1198→:1202`,`:1299→:1303`). No per-block waits to add. | The single wait-hook covers MoE for free; do not add per-block ensures. |
| C8 | `alloc_padded_pinned` via `set_(storage, …)` | Multi-dim `set_` onto an untyped storage is error-prone. Use `torch.empty(padded_numel, pin_memory=True)[:numel].view(shape)`; keep whole-storage `set_` only for the flat byte view. `is_pinned()` is storage-backed ⇒ views preserve it. | Cleaner, correct `alloc_padded_pinned` in Stage 1. |
| C9 | boundary tensor via `ctx` stash | Storing tensors on `ctx` **can** leak (reference cycle) — but only when the tensor carries a `grad_fn`. The boundary copy is a **detached CPU blob made inside `Function.forward` (no-grad) ⇒ no `grad_fn` ⇒ no cycle**. `save_for_backward` is unusable (its `SavedVariable` pins the buffer, defeating spill). | Keep the `ctx.asym_handle` design; add mitigations: `try/finally` release, `del ctx.asym_handle`, a debug governor-leak counter. |
| C10 | dense no-grad path `:885` | `_finegrained_dense_mlp_no_grad_forward` **def** is `:668`; `:885-886` is its call site. | Cosmetic; no seal there (in-call create+release). |

Environment (unchanged, re-confirmed): torch **2.12.0+cu130**; LF venv py3.11 at
`third_party/LlamaFactory/.venv/bin/python` (`$LF_PY`); `.aioenv` = conda libaio (aarch64), exported by
`run_lf_lora_sft.sh:34-48`; `/scratch_local` = `md0` RAID0 ext4, **12 TB free**, writable, and the existing DS
NVMe baseline (`ds_z3_offload_panvme_config.json → nvme_path=/scratch_local/user_data/shutian/kevin/cache/ds_nvme_offload`)
**proves O_DIRECT works on this exact mount**; measured ~26 GB/s read / ~14 GB/s write.

---

## Design contract (all stages)

1. **One store, role-tagged.** `NVMeStore` serves `{base_weight, activation}`; placement policy lives in the
   governor/pager (it *is* the algorithm), not the store.
2. **Zero compute-semantics changes.** No new small GEMMs, no per-expert loops, no added hot-path launches.
   The only new CUDA ops are one event record per Function forward (seal) and one per boundary offload.
3. **Spill on pressure, FIFO; consume LIFO; release on last use.** Only overflow spills, oldest-first
   (= farthest future use under LIFO = exact Belady). Budget `0` ⇒ eager spill-everything.
4. **No kernel computes from NVMe.** Always `NVMe → pinned CPU → (H2D) → compute`; `pinned CPU → NVMe`.
5. **Single-owner AIO handles.** Write handle owned by one thread (main in sync, writer in async); read handle
   by main. Python pending ledgers; **never `wait()` idle; always check `rc==0` first (C2)**.
6. **Event-gated buffer reuse.** A pinned buffer a GPU kernel may still stream is recycled only after a
   post-enqueue CUDA event completes.
7. **Off = byte-identical.** No `*nvme` token ⇒ no deepspeed import, no thread, no file; every hook is a `None`
   check on a module-level singleton.
8. Every file offset and IO length padded to `store.align`; IO buffers allocated padded.
9. **Sync before async.** Each stage lands sync first and gates on correctness (bit-exact unit + e2e loss
   match) before any overlap flag flips.

---

## Stage 0 — NVMe traffic census (postprocess-only; review before coding Stage 3+ budgets)

**What:** turn existing + two fresh short profiles into the budget-decision artifact: per-substrate/per-layer
bytes/step, overflow-vs-budget table, feasibility verdict. §0.2 already establishes the headline (boundary
stream dominates); the census makes it per-layer/per-tag and pins the gate budgets.

**Scope:** NEW `scripts/lf/project_nvme_traffic.py` (standalone + callable from
`postprocess_lf_profile_artifacts.py` beside the other emitters). **ZERO runtime edits.**

```python
# scripts/lf/project_nvme_traffic.py
# Input:  a run dir (source_profile.json [+ step_samples.csv]).  Output: nvme_traffic_projection.{csv,md}
# One dict at the top holds model shapes AND tag classification lists (single source of truth):
MODEL_SHAPES = {"q3-32b": dict(layers=64, hidden=5120), "q3.5-35b-a3b": dict(layers=..., hidden=...), ...}
TRANSIENT_TAGS = {"mlp.dact","mlp.dgate","mlp.dup","moe.dgate","moe.dup"}      # never spill; report for context
STAGE_TAGS_SUFFIX = ("_stage",)                                               # H2D volume, v4-rung input

def project(run_dir):
    prof = json.load(source_profile.json); cfg = prof["config"]
    steps = prof["trainer"]["measured_steps"] + warmup;  micro = steps * cfg["grad_accum"]
    M = cfg["seq_len"] * cfg["batch_size"];  shp = MODEL_SHAPES[cfg["model"]]
    # (2) Substrate A (boundary): analytic write candidate.
    A_bytes_step = shp["layers"] * M * shp["hidden"] * 2                       # bf16
    # cross-check: fwd RSS peak - host_weight - pool_cached - baseline_slack ≈ A ; flag > 15% disagreement.
    # (3) Substrate B: classify each activation_offload.rows[] row by module class:
    #     fg row (module class in dense_mlp_finegrained/qwen3_moe_finegrained) → bytes already per-microbatch
    #     attention mixed row → local per-call + source_context cumulative ÷ microbatch-forwards
    #     wrapper row (decoder_activation_offload) → cumulative ÷ microbatch-forwards
    #   emit per (module, tag): fwd_saved_bytes_step (write candidate; read≈write),
    #     transient_bytes_step (TRANSIENT_TAGS, never-spill), stage_bytes_step (H2D, v4 input).
    # (4) panvme sheet: memory_attribution host_weight/cpu rows → per-component bytes;
    #     per-step read = 2×(fwd+bwd reuse); model total = Σ eligible components (exclude embed/norm rows).
    # (5) Summary: A+B write ceiling @ budget 0; write_s_step = total/14e9, read_s_step = total/26e9 vs
    #     measured step_seconds; overflow table for budget ∈ {150,250,400,600} GiB; per-layer top-N; per-tag totals.
    # If the run is a DS *nvme baseline, print offload_io.json {read_gb,write_gb} beside the projection.
```

**Validation (Stage 0 gate)** — postprocess only, but run two *real* short profiles to feed it:

```bash
cd $REPO   # /home/kevinni/AsymGEMM-SFT-39/third_party/AsymGEMM ; LF_PY=../LlamaFactory/.venv/bin/python
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage0_census PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=1 MAX_STEPS=4 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 45000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
$LF_PY scripts/lf/project_nvme_traffic.py --run-dir profiling_nvme/stage0_census/<run_dir>   # each seq point
$LF_PY scripts/lf/project_nvme_traffic.py --run-dir profiling_q35_final2_asym80k_*/...        # free MoE point
```

**Accept:** artifacts exist; Substrate-A analytic vs RSS-residual agree ≤15%; B rows sum to the
`activation_offload` aggregates; the two seq points scale ≈ linearly in tokens; `write_seconds_step ≪
step_seconds` at intended budgets (feasibility unambiguous); **budgets for Stage 3/4/6 gates chosen and
recorded in the .md**.

**Risks/watch:** row-semantics misclassification (assert cumulative rows grow with `MAX_STEPS` across the two
runs, per-call rows don't); model-shape drift (single dict).

---

## Stage 1 — `NVMeStore` substrate (sync + async write paths)

**Scope:** NEW `asym_gemm/training/nvme_store.py` + NEW `tests/training/test_nvme_store.py`. Zero edits
elsewhere ⇒ **isolated unit gate suffices** (the one exception to the e2e rule — pure IO plumbing, no
training-visible behavior). AIO facts baked in from re-verification (C1-C3, C8).

```python
# asym_gemm/training/nvme_store.py
from __future__ import annotations
import math, os, queue, threading, time
from dataclasses import dataclass, field
from typing import Any, Callable
import torch

def _env_bool(n, d):  v=os.environ.get(n); return d if v in (None,"") else v.strip().lower() in {"1","true","yes","y","on"}
def _env_int(n, d):
    v=os.environ.get(n)
    try: return int(v)
    except (TypeError,ValueError): return d
def _die(m): raise RuntimeError(m)
def _pad(n, a): return (n + a - 1) // a * a
def _numel(shape): 
    p=1
    for d in shape: p*=int(d)
    return p

@dataclass(frozen=True)
class NVMeStoreConfig:
    path: str                                    # ASYM_NVME_PATH (required when roles set)
    roles: frozenset                             # subset of {"base_weight","activation"} from ASYM_NVME_ROLES
    sync: bool = True                            # ASYM_NVME_SYNC (v1 default 1; Stage 5 flips to 0)
    aio_block_size: int = 1 << 20                # ctor arg 1
    aio_queue_depth: int = 16                    # ctor arg 2 (AIO default is 128; we pin 16)
    aio_single_submit: bool = False              # ctor arg 3
    aio_overlap_events: bool = True              # ctor arg 4
    aio_intra_op_parallelism: int = 4            # ctor arg 5 → get_alignment() = 4*512 = 2048
    min_swappable_bytes: int = 1 << 20           # ASYM_NVME_MIN_SWAP_BYTES
    activation_arena_bytes: int = 1 << 41        # 2 TiB logical cap; sparse, allocated on demand
    max_inflight_spill_bytes: int = 8 << 30      # writer backpressure (async only)

def _config_from_env():
    roles = frozenset(r.strip() for r in os.environ.get("ASYM_NVME_ROLES","").split(",") if r.strip())
    if not roles: return None
    bad = roles - {"base_weight","activation"}
    if bad: raise ValueError(f"unknown ASYM_NVME_ROLES: {sorted(bad)}")
    if int(os.environ.get("WORLD_SIZE","1")) > 1: _die("asym NVMe store is single-process only (deferred)")
    path = os.environ.get("ASYM_NVME_PATH") or _die("ASYM_NVME_PATH required with ASYM_NVME_ROLES")
    return NVMeStoreConfig(
        path=path, roles=roles, sync=_env_bool("ASYM_NVME_SYNC", True),
        aio_block_size=_env_int("ASYM_NVME_AIO_BLOCK_SIZE", 1<<20),
        aio_queue_depth=_env_int("ASYM_NVME_AIO_QUEUE_DEPTH", 16),
        aio_intra_op_parallelism=_env_int("ASYM_NVME_AIO_INTRA_OP", 4),
        min_swappable_bytes=_env_int("ASYM_NVME_MIN_SWAP_BYTES", 1<<20),
        activation_arena_bytes=_env_int("ASYM_NVME_ACTIVATION_ARENA_BYTES", 1<<41),
        max_inflight_spill_bytes=_env_int("ASYM_NVME_MAX_INFLIGHT_SPILL_BYTES", 8<<30))

def _flat_u8(t: torch.Tensor) -> torch.Tensor:
    """uint8 alias of t's WHOLE storage (padded length) — one tensor == one IO op, never fragmented.
    Uses set_ over the untyped storage (the standard flat-byte-view idiom)."""
    storage = t.untyped_storage()
    out = torch.empty(0, dtype=torch.uint8)
    out.set_(storage, 0, (storage.nbytes(),))          # 1-D: contiguous stride (1,) implied
    return out

def alloc_padded_pinned(shape, dtype, *, align) -> torch.Tensor:
    """Pinned CPU tensor whose WHOLE storage is align-padded; exact contiguous [shape] view returned.
    (C8) empty(padded_numel, pin)[:numel].view(shape) — avoids multi-dim set_; is_pinned is storage-backed
    so the view stays pinned. On pin failure fall back to unpinned (io_ready() then excludes it)."""
    elem = torch.empty(0, dtype=dtype).element_size()
    numel = _numel(shape)
    padded_numel = _pad(numel * elem, align) // elem   # align % elem == 0 for bf16/fp32 (align=2048)
    try:
        buf = torch.empty(padded_numel, dtype=dtype, pin_memory=True)
    except RuntimeError:
        buf = torch.empty(padded_numel, dtype=dtype)
    return buf[:numel].view(shape)                     # contiguous; storage == padded buf

def io_ready(t: torch.Tensor, align: int) -> bool:
    """Spill-eligibility: pinned + whole-storage length aligned. Foreign adopt_cpu tensors that fail this
    simply stay resident (counted for pressure, never spilled)."""
    return t.is_pinned() and (t.untyped_storage().nbytes() % align == 0)

@dataclass
class BlobRef:
    role: str; file: str; offset: int; length: int; logical_nbytes: int
    durable: threading.Event = field(default_factory=threading.Event)

@dataclass
class NVMeStoreStats:
    bytes_written: dict = field(default_factory=dict); bytes_read: dict = field(default_factory=dict)
    write_ops: dict = field(default_factory=dict);     read_ops: dict = field(default_factory=dict)
    spill_wait_ms: float = 0.0; fetch_wait_ms: float = 0.0; spill_backpressure_ms: float = 0.0
    inflight_peak_bytes: int = 0; arena_peak_bytes: dict = field(default_factory=dict); wasted_writes: int = 0
    def as_dict(self):  # flat, prefixed so it survives the profile row merge unchanged
        return {"asym_nvme_bytes_written": dict(self.bytes_written), "asym_nvme_bytes_read": dict(self.bytes_read),
                "asym_nvme_write_ops": dict(self.write_ops), "asym_nvme_read_ops": dict(self.read_ops),
                "asym_nvme_spill_wait_ms": self.spill_wait_ms, "asym_nvme_fetch_wait_ms": self.fetch_wait_ms,
                "asym_nvme_spill_backpressure_ms": self.spill_backpressure_ms,
                "asym_nvme_inflight_peak_bytes": self.inflight_peak_bytes,
                "asym_nvme_arena_peak_bytes": dict(self.arena_peak_bytes), "asym_nvme_wasted_writes": self.wasted_writes}

class NVMeStore:
    def __init__(self, cfg: NVMeStoreConfig):
        from deepspeed.ops.op_builder import AsyncIOBuilder            # imported ONLY when enabled (rule 7)
        m = AsyncIOBuilder().load(verbose=False)
        mk = lambda: m.aio_handle(cfg.aio_block_size, cfg.aio_queue_depth, cfg.aio_single_submit,
                                  cfg.aio_overlap_events, cfg.aio_intra_op_parallelism)
        self.cfg = cfg
        self._read_h  = mk()                                           # MAIN THREAD ONLY
        self._write_h = mk()                                           # main (sync) or writer thread (async)
        self.align = int(self._read_h.get_alignment())                # = intra_op*512 = 2048 (C3)
        self.stats = NVMeStoreStats()
        self._debug = _env_bool("ASYM_NVME_DEBUG", False)
        self._main_ident = threading.get_ident()
        os.makedirs(os.path.join(cfg.path, "base_weight"), exist_ok=True)
        self._blob_seq = 0
        self._arena_path = os.path.join(cfg.path, f"activation.{os.getpid()}.arena")
        self._arena_cursor = 0; self._arena_live = 0
        self._pending_reads: dict = {}                                 # id(ref) → BlobRef (MAIN THREAD ONLY)
        self._writer = None if cfg.sync else _WriterThread(self._write_h, cfg, self.stats)
        if self._writer: self._writer.start()

    def has_role(self, role): return role in self.cfg.roles
    def _assert_main(self):
        if self._debug: assert threading.get_ident() == self._main_ident, "read/main-only API off main thread"

    # --- filenames ---
    def _blob_file(self):
        self._blob_seq += 1
        return os.path.join(self.cfg.path, "base_weight", f"w{self._blob_seq:06d}.bin")

    # --- activation arena: bump allocator, reset-when-empty (blob lifetime = one microbatch fwd→bwd) ---
    def _arena_alloc(self, nbytes):
        length = _pad(nbytes, self.align)
        off = self._arena_cursor; self._arena_cursor += length; self._arena_live += 1
        self.stats.arena_peak_bytes["activation"] = max(self.stats.arena_peak_bytes.get("activation",0), self._arena_cursor)
        if self._arena_cursor > self.cfg.activation_arena_bytes:
            raise RuntimeError("activation arena full — raise ASYM_NVME_ACTIVATION_ARENA_BYTES")
        return self._arena_path, off
    def blob_done(self, ref):                                          # after final fetch OR dropped blob
        if ref.role == "activation":
            self._arena_live -= 1
            if self._arena_live == 0: self._arena_cursor = 0

    def _count_write(self, ref, nbytes):
        self.stats.bytes_written[ref.role] = self.stats.bytes_written.get(ref.role,0)+nbytes
        self.stats.write_ops[ref.role]     = self.stats.write_ops.get(ref.role,0)+1
    def _count_read(self, ref):
        self.stats.bytes_read[ref.role] = self.stats.bytes_read.get(ref.role,0)+ref.length
        self.stats.read_ops[ref.role]   = self.stats.read_ops.get(ref.role,0)+1

    # --- write paths ---
    def spill_sync(self, role, tensor) -> BlobRef:
        """MAIN THREAD. Caller has already synchronized the seal/ready event. Blocking write."""
        buf = _flat_u8(tensor); assert buf.numel() % self.align == 0
        file, off = (self._blob_file(), 0) if role == "base_weight" else self._arena_alloc(buf.numel())
        ref = BlobRef(role, file, off, buf.numel(), tensor.numel()*tensor.element_size())
        t0 = time.perf_counter()
        rc = self._write_h.async_pwrite(buf, ref.file, ref.offset)     # C2: check rc BEFORE wait()
        if rc != 0: raise RuntimeError(f"async_pwrite rc={rc} (alignment/open) file={ref.file} off={ref.offset}")
        n = self._write_h.wait(); assert n == 1
        self.stats.spill_wait_ms += (time.perf_counter()-t0)*1e3
        self._count_write(ref, buf.numel()); ref.durable.set()
        return ref

    def spill_async(self, role, tensor, *, ready_event, on_done) -> BlobRef:
        """ANY THREAD → writer (Stage 5). ready_event synchronized ON THE WRITER before pwrite; on_done(ref)
        runs in writer context — NEVER touch CUDA there. ref allocation identical to spill_sync."""
        buf = _flat_u8(tensor); assert buf.numel() % self.align == 0
        file, off = (self._blob_file(), 0) if role == "base_weight" else self._arena_alloc(buf.numel())
        ref = BlobRef(role, file, off, buf.numel(), tensor.numel()*tensor.element_size())
        self._writer.submit(ready_event, buf, ref, on_done)            # writer sets ref.durable + counts
        return ref

    # --- read path (MAIN THREAD ONLY) ---
    def submit_pread(self, ref, dst_padded_pinned) -> None:
        self._assert_main()
        if not ref.durable.is_set(): ref.durable.wait()                # write in flight (async) → bounded rare block
        dst = _flat_u8(dst_padded_pinned); assert dst.numel() == ref.length   # C2: exact padded length
        rc = self._read_h.async_pread(dst, ref.file, ref.offset)
        if rc != 0: raise RuntimeError(f"async_pread rc={rc} file={ref.file} off={ref.offset} len={ref.length}")
        self._pending_reads[id(ref)] = ref
    def drain_reads(self) -> set:
        """Blocks until ALL pending reads complete (wait() drains the whole handle)."""
        if not self._pending_reads: return set()                       # C2: never wait() idle
        n = self._read_h.wait(); assert n == len(self._pending_reads)
        for r in self._pending_reads.values(): self._count_read(r)
        done = set(self._pending_reads); self._pending_reads.clear()
        return done
    def fetch_into(self, ref, dst_padded_pinned) -> set:
        t0 = time.perf_counter()
        self.submit_pread(ref, dst_padded_pinned); done = self.drain_reads()   # drains any in-flight prefetches too
        self.stats.fetch_wait_ms += (time.perf_counter()-t0)*1e3
        return done

    def shutdown(self):
        if self._writer: self._writer.stop()

class _WriterThread(threading.Thread):                                 # Stage 5; SOLE owner of write handle in async
    _STOP = object()
    def __init__(self, write_h, cfg, stats):
        super().__init__(daemon=True); self._h=write_h; self._cfg=cfg; self._stats=stats
        self._q = queue.Queue(); self._inflight = 0; self._cv = threading.Condition()
    def submit(self, ready_event, buf, ref, on_done):
        with self._cv:
            t0 = time.perf_counter()
            while self._inflight + ref.length > self._cfg.max_inflight_spill_bytes and self._inflight > 0:
                self._cv.wait()                                        # backpressure
            self._stats.spill_backpressure_ms += (time.perf_counter()-t0)*1e3
            self._inflight += ref.length
            self._stats.inflight_peak_bytes = max(self._stats.inflight_peak_bytes, self._inflight)
        self._q.put((ready_event, buf, ref, on_done))
    def run(self):
        while True:
            item = self._q.get()
            if item is self._STOP: return
            ready_event, buf, ref, on_done = item
            if ready_event is not None: ready_event.synchronize()      # host-side wait — allowed off main thread
            rc = self._h.async_pwrite(buf, ref.file, ref.offset)
            if rc != 0: raise RuntimeError(f"writer async_pwrite rc={rc}")
            n = self._h.wait(); assert n == 1
            self._stats.bytes_written[ref.role] = self._stats.bytes_written.get(ref.role,0)+ref.length
            self._stats.write_ops[ref.role] = self._stats.write_ops.get(ref.role,0)+1
            ref.durable.set()
            with self._cv: self._inflight -= ref.length; self._cv.notify_all()
            on_done(ref)                                               # governor callback (CUDA-free)
    def stop(self): self._q.put(self._STOP); self.join()

_STORE = None; _STORE_INIT = False
def get_nvme_store():
    """Lazy env singleton. Without ASYM_NVME_ROLES: returns None, imports nothing, allocates nothing (rule 7)."""
    global _STORE, _STORE_INIT
    if _STORE_INIT: return _STORE
    cfg = _config_from_env()
    _STORE = NVMeStore(cfg) if cfg is not None else None
    _STORE_INIT = True
    return _STORE
```

**Locked decisions:** base_weight = one file per HostWeight (static, written once); activation = one
pid-suffixed arena file + bump/reset-when-empty; per-op `wait()` in the writer (each op is internally
4×16×1MB-parallel — a single 5.5 GiB pwrite already saturates the array); `io_ready()` is the
paddedness/pinnedness gate that makes foreign `adopt_cpu` tensors safely non-spillable.

**Efficiency:** IO unit = one whole tensor storage (GiB-scale for Substrate A: one 5.5 GiB pwrite) — never
fragmented; zero extra memcpys (the padded pinned buffer IS both the D2H destination and the IO buffer);
single-owner handles → no hot-path locks; sequential arena writes at max bandwidth.

**Validation (Stage 1 gate) — isolated unit + AIO smoke:**

```bash
cd $REPO && export AIO_HOME=$PWD/.aioenv CPATH="$AIO_HOME/include:${CPATH:-}" \
  LIBRARY_PATH="$AIO_HOME/lib:${LIBRARY_PATH:-}" LD_LIBRARY_PATH="$AIO_HOME/lib:${LD_LIBRARY_PATH:-}"
ASYM_NVME_PATH=/scratch_local/user_data/shutian/kevin/cache/asym_nvme_test \
$LF_PY -m pytest tests/training/test_nvme_store.py -q
```

**Required tests:** bf16/fp32 roundtrips below/at/above 1 MiB (sync + async); two arena blobs at different
offsets, no cross-corruption; arena reset-when-empty across two simulated microbatches; async spill gated on a
CUDA event (write CUDA tensor D2H, record, `spill_async`, fetch, compare) — skip-if-no-CUDA variant with a
pre-set event; fetch-before-durable blocks then succeeds; 3-deep prefetch ledger reconciles via `drain_reads`;
backpressure blocks then resumes; **`io_ready` rejects unpinned/unpadded, and `alloc_padded_pinned(...).is_pinned()`
is True with storage nbytes % align == 0 (C8)**; **a deliberately misaligned `async_pwrite` returns rc≠0 and the
store raises rather than aborting (C2)**; disabled env → `get_nvme_store() is None` and `"deepspeed" not in
sys.modules`; clean writer shutdown; pid-suffixed arena avoids cross-run collision.

**Risks/watch:** handle thread-ownership is a rule not API-enforced — `_assert_main` under `ASYM_NVME_DEBUG=1`;
arena overflow at extreme seq×accumulation → loud error + env knob; **O_DIRECT needs real block-backed FS with
≥2048-byte alignment — empirically OK on `/scratch_local` (DS baseline uses it); if a device ever needs 4096,
bump `ASYM_NVME_AIO_INTRA_OP=8` (align→4096)** (C2/C3).

---

## Stage 2 — Backend tokens, env plumbing, profile counters, compare gate (no tensors move)

**Scope (all anchors re-confirmed @ HEAD):** `scripts/lf/profile_lora_lf_test_source.sh` (+ identical edits in
`_both.sh`), `scripts/lf/run_lf_lora_sft.sh`, `scripts/lf/run_lf_profiled_train.py`,
`scripts/lf/postprocess_lf_profile_artifacts.py`, NEW `scripts/lf/compare_nvme_profiles.py`.

**(a) `profile_lora_lf_test_source.sh`** — extend the three backend dispatch points (clone the
`asym_cpuadamwds` arm each time):

```bash
# append_backend_spec case (arms :1103-1126, asym_cpuadamwds arm :1108, die :1126) — grouped arm:
    asym_cpuadamwds|asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme) backend="${token}" ;;
# backend_gpu_count 1-GPU asym line :903 — add the three tokens:
    asym|asym_torch|asym_cpuadamwtorch|asym_cpuadamwds|asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme) printf '1\n' ;;
# cpuadam_backend_for_label asym arm :1057 — add the three tokens (all → deepspeed):
    asym_cpuadamwds|asym_cpuadamwds_panvme|asym_cpuadamwds_actnvme|asym_cpuadamwds_bothnvme) printf 'deepspeed\n' ;;
```

Inside `run_job`'s `run_env` block (`:3436-3597`; backend var set `:3073`, `BACKEND=` emit `:3464`), add a
helper + role plumbing, and mirror the nvme vars in the `ASYM_GEMM_LF_CONFIG_*` block (`:3546-3593`):

```bash
nvme_roles_for_backend() {                       # single source of truth for token→roles
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
             "ASYM_GEMM_LF_CONFIG_ASYM_NVME_PATH=${ASYM_NVME_PATH:-/scratch_local/user_data/shutian/kevin/cache/asym_nvme}"
             "ASYM_GEMM_LF_CONFIG_ASYM_NVME_SYNC=${ASYM_NVME_SYNC:-1}"
             "ASYM_GEMM_LF_CONFIG_ASYM_NVME_ACT_CPU_BUDGET_BYTES=${ASYM_NVME_ACT_CPU_BUDGET_BYTES:-auto}" )
fi
```

`job_root_path`/`path_label` (`:1906`) already puts backend first ⇒ run dirs auto-disambiguate. Completion
checks key backend at `$3` (`job_profile_complete :1262`) / `$2` (`existing_profile_complete :1305`) — the new
tokens flow through unchanged.

**(b) `profile_lora_lf_test_both.sh`** — byte-identical except the `PROFILERS` default (`:180-181`). Apply the
SAME edits; then `diff` the two and assert only `:180-181` differs.

**(c) `run_lf_lora_sft.sh`** — one grouped arm cloned from `asym_cpuadamwds` (`:397-403`), inserted before
`asym_torch` (`:404`); `die` (`:414`) untouched; `is_zero_backend_run` (`:709`, needs `BACKEND==torch`) stays
false because we set `BACKEND=asym`:

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
    if [[ ",${ASYM_NVME_ROLES}," == *",activation,"* ]]; then export ASYM_UNSLOTH_GC_NVME=1; fi   # Stage 3 hook
    BACKEND=asym
    ;;
```

Mirror `ASYM_GEMM_LF_CONFIG_ASYM_NVME_{ROLES,PATH,SYNC,ACT_CPU_BUDGET_BYTES}` in the `:2282-2295` block. **NB:
the `ASYM_UNSLOTH_GC_NVME` export is COMMENTED in Stage 2** (its import target `gc_boundary_offload.py` doesn't
exist yet) and un-commented in Stage 3.

**(d) `run_lf_profiled_train.py`:**

```python
_ASYM_CPUADAMW_DS_BACKENDS = {"asym_cpuadamwds","asym_cpuadamwds_panvme",
                              "asym_cpuadamwds_actnvme","asym_cpuadamwds_bothnvme"}   # near :579
# is_asym_deepspeed_cpuadamw (:579-582): `backend in _ASYM_CPUADAMW_DS_BACKENDS or (...)`
# _config_from_args (:546): read "asym_nvme_roles"/"asym_nvme_path"/"asym_nvme_sync"/
#   "asym_nvme_act_cpu_budget_bytes" via os.environ.get(ASYM_GEMM_LF_CONFIG_<UPPER>) or os.environ.get(raw,"")
# report() (:2829): sibling of "activation_offload" (:2908):
"asym_nvme": _asym_nvme_summary_from_model(),
```

```python
def _asym_nvme_summary_from_model():
    try: from asym_gemm.training.nvme_store import get_nvme_store
    except Exception as exc: return {"enabled": False, "reason": repr(exc)}
    store = get_nvme_store()
    if store is None: return {"enabled": False}
    out = {"enabled": True, "roles": sorted(store.cfg.roles), "path": store.cfg.path,
           "sync": store.cfg.sync, "alignment": store.align, **store.stats.as_dict()}
    try:
        from asym_gemm.training.act_spill_governor import get_act_spill_governor          # Stage 3
        gov = get_act_spill_governor()
        if gov is not None: out["act_governor"] = gov.summary()
        from asym_gemm.training.gc_boundary_offload import get_boundary_offload_stats     # Stage 3
        out["gc_boundary"] = get_boundary_offload_stats()
    except Exception: pass
    model, _ = _model_and_base_model()
    pager = getattr(model, "_asym_base_weight_pager", None)                               # Stage 7
    if pager is not None: out["base_weight_pager"] = pager.summary()
    return out
```

Extend the `_activation_offload_counters_from_model` aggregates tail (`:2265-2280`) with
`total_nvme_spilled_bytes / total_nvme_bytes_read / total_nvme_fetch_wait_ms / total_nvme_spill_wait_ms` summed
from row stats (they appear in rows automatically once the governor extends `snapshot()`).

**(e) `postprocess_lf_profile_artifacts.py`:** `_asym_nvme_rows()` flattener → `asym_nvme.csv` (clone the
`_asym_cpu_adamw_rows` pattern at `:378`); one NVMe line in `memory.md` (`_source_memory_markdown` `:1803`,
written `:2200`).

**(f) NEW `scripts/lf/compare_nvme_profiles.py`** — clone of `compare_liger_loss_profiles.py` (`_fail`→
`{"ok":false}`+`SystemExit(2)` `:46-51`; measured = non-`is_warmup` rows; medians from `step_samples.csv`).
Columns confirmed present: `loss`, `is_warmup`, `{forward,backward,optimizer,training_step}_process_rss_peak_end_bytes`,
`step/forward/backward_milliseconds`.

```text
--baseline DIR --candidate DIR --target {no_change, activation_cpu, base_weight_cpu, maxseq}
--memory-metric DOTTED   (step_samples.<col> = MAX over measured csv rows;
                          default step_samples.training_step_process_rss_peak_end_bytes)
--min-memory-drop-gib | --min-memory-drop-pct     (no_change: --max-memory-drift-gib)
--max-step-ratio / --max-forward-ratio / --max-backward-ratio   (sync-capacity 1.5 informational; async ≤1.10)
--expect-nvme-role ROLE   (asserts asym_nvme.enabled AND ROLE∈roles AND bytes_written>0 AND bytes_read>0)
--max-loss-delta FLOAT    (median |loss_i − baseline_loss_i| over measured steps, paired by step)
Preflight: artifacts exist (+asym_nvme.csv on candidates); losses finite; measured_steps≥3; config.asym_nvme_roles matches.
Output: {"ok":…, "failures":[…], "checks":{…}}; SystemExit(2) on any failure.
```

**Validation (Stage 2 gate — paired e2e NO-CHANGE at a real operating point):**

```bash
bash -n scripts/lf/profile_lora_lf_test_source.sh scripts/lf/profile_lora_lf_test_both.sh scripts/lf/run_lf_lora_sft.sh
diff scripts/lf/profile_lora_lf_test_source.sh scripts/lf/profile_lora_lf_test_both.sh   # expect only :180-181
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage2_nochange PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=1 MAX_STEPS=4 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds_actnvme|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
$LF_PY scripts/lf/compare_nvme_profiles.py \
  --baseline profiling_nvme/stage2_nochange/<asym_cpuadamwds dir> --candidate <…_actnvme dir> \
  --target no_change --max-memory-drift-gib 2 --max-step-ratio 1.02 --max-forward-ratio 1.02 \
  --max-backward-ratio 1.02 --max-loss-delta 0
```

**Accept:** drift/latency inside bounds; candidate `config.asym_nvme_roles=="activation"`,
`asym_nvme.enabled==true` with `bytes_written==0` (governor not wired yet — store idles); `asym_nvme.csv`
present; losses identical (nothing moved); **the tool demonstrably exits 2 on a mismatched pair** (run once vs a
stale dir). Heavy runs sequential; both fit RAM at s30000.

**Risks/watch:** `_both.sh` must get identical arms (`diff` after editing); the `ASYM_UNSLOTH_GC_NVME` export
stays commented until Stage 3; still pass `--overwrite true` (backend-first dir keys avoid stale reuse).

---

## Stage 3 — actnvme v1 SYNC: governor + Substrate A (unsloth boundary) ← FIRST tensor-moving stage; THE lever

**What:** run the flagship exactly as today, but the unsloth-GC boundary hidden-states become governor-tracked
**pinned** handles. When live activation bytes cross the CPU budget, the governor spills oldest-first to NVMe
**inline (sync)**; each layer's backward fetches its boundary back (sync pread) right before H2D reload; buffers
and blobs free at exactly today's release points. **Zero compute changes; budget ≥ footprint ⇒ behavior
identical** (plus pinned instead of pageable boundary copies).

**Scope:** NEW `asym_gemm/training/act_spill_governor.py`; NEW `asym_gemm/training/gc_boundary_offload.py`;
`activation_offload.py` (~30 lines of hooks + `_pop_active`); LF `checkpointing.py` (**one 4-line env-gated
branch** at the head of `get_unsloth_gradient_checkpointing_func` `:78`); un-comment the `ASYM_UNSLOTH_GC_NVME`
export (Stage 2c). Substrate-B seals are Stage 4 — but **all manager hooks land now** (inert without seals).

### 3.1 Governor state machine

```text
                 on_offload           on_seal(ev)          pressure (_maybe_spill, sync write)
(untracked) ─────────────▶ TRACKED ───────────▶ QUEUED ───────────────────────────▶ DURABLE (buffer freed)
                              │                    │                                     │ ensure_local
                              │ on_release         │ ensure_local                        ▼ (sync pread)
                              └───────────         └──────────▶ CLAIMED             FETCHED ── on_release
Async (Stage 5) adds QUEUED→SUBMITTED (write in flight; buffer still valid — writer only READS it);
SUBMITTED→CLAIMED on consume; on_durable sees CLAIMED → drops the blob, counts wasted_write.
```

```python
# asym_gemm/training/act_spill_governor.py
import itertools, threading
from collections import deque
from dataclasses import dataclass
from typing import Any
import torch
from .nvme_store import get_nvme_store, io_ready, alloc_padded_pinned
from .activation_offload import _alloc_cpu, _return_cpu           # reuse the pool

TRACKED, QUEUED, SUBMITTED, DURABLE, CLAIMED, FETCHED = range(6)
_SENTINEL = torch.empty(0, dtype=torch.uint8)                     # data_ptr()==0 ⇒ stock release paths no-op

@dataclass
class _Rec:
    handle: Any; manager: Any; nbytes: int                        # nbytes CACHED (handle.nbytes is live over .tensor)
    substrate: str                                                # "boundary" | "fg"  (stats only; NO policy)
    order: int; state: int = TRACKED
    seal_event: Any = None; ref: Any = None; prefetch_buf: Any = None

class ActSpillGovernor:
    def __init__(self, store):
        self._store = store; self._lock = threading.RLock()
        self._by_id = {}                                          # id(handle) → _Rec (handle alive until release)
        self._queue = deque()                                     # QUEUED recs in creation order
        self._spilled = []                                        # SUBMITTED/DURABLE recs in spill order
        self._order = itertools.count()
        self.live_cpu_bytes = 0
        self.hi = _budget_from_env()                              # ASYM_NVME_ACT_CPU_BUDGET_BYTES (int|auto|0)
        self.lo = max(0, self.hi - _env_int("ASYM_NVME_ACT_LOW_SLACK_BYTES", 16<<30))   # hysteresis
        self.prefetch_bytes = _env_int("ASYM_NVME_ACT_PREFETCH_BYTES", 0)               # 0 until Stage 5
        assert self.prefetch_bytes < max(1, self.hi - self.lo), "prefetch must fit in the hi-lo band"
        self.stats = _GovStats()                                  # per-substrate spilled/fetched, live_peak, leaks

    # ---------- producer side ----------
    def on_offload(self, manager, handle, substrate="fg"):
        """Called ONLY at the two _mark_cpu_live sites (empty_cpu:184, adopt_cpu:227) — NOT offload() (C5).
        Pressure accounting for EVERY handle; nothing spillable until sealed."""
        with self._lock:
            rec = _Rec(handle, manager, handle.nbytes, substrate, order=next(self._order))
            self._by_id[id(handle)] = rec
            self.live_cpu_bytes += rec.nbytes
            self.stats.note_live_peak(self.live_cpu_bytes, substrate)

    def on_seal(self, manager, handles, event=None):
        """ONE call at END of Function.forward (Substrate B) or per boundary offload (A). The event orders
        after (a) the D2H fill and (b) every already-enqueued forward consumer (same stream). Handles failing
        io_ready or < min_swappable_bytes stay TRACKED (pressure-only)."""
        ev = event or _record_event()
        with self._lock:
            for h in handles:
                rec = self._by_id.get(id(h))
                if (rec and rec.state == TRACKED and rec.nbytes >= self._store.cfg.min_swappable_bytes
                        and io_ready(h.tensor, self._store.align)):
                    rec.seal_event = ev; rec.state = QUEUED; self._queue.append(rec)
        self._maybe_spill()

    def _maybe_spill(self):
        """Oldest-first until live ≤ lo. SYNC: inline blocking write on the caller (main) thread. Oldest sealed
        handles' events are long complete ⇒ synchronize() is ~free."""
        while True:
            with self._lock:
                if self.live_cpu_bytes <= self.hi or not self._queue: return
                rec = self._queue.popleft()
                if rec.state != QUEUED: continue                  # CLAIMED/released while queued (lazy dequeue)
                rec.state = SUBMITTED
            rec.seal_event.synchronize()                          # D2H + fwd consumers done ⇒ bytes stable
            if self._store.cfg.sync:
                rec.ref = self._store.spill_sync("activation", rec.handle.tensor)
                self._finish_spill(rec)                           # main thread
            else:                                                 # Stage 5
                rec.ref = self._store.spill_async("activation", rec.handle.tensor,
                                                  ready_event=None,   # already synced above
                                                  on_done=self._make_on_durable(rec))
            with self._lock:
                self._spilled.append(rec)
                if self.live_cpu_bytes <= self.lo: return

    def _finish_spill(self, rec):                                 # sync body / writer callback (Stage 5)
        with self._lock:
            if rec.state == CLAIMED:                              # async-only: consumed while in flight
                self._store.blob_done(rec.ref); self.stats.wasted_writes += 1; return
            rec.state = DURABLE
            self.live_cpu_bytes -= rec.nbytes
            self.stats.note_spill(rec.substrate, rec.nbytes)
        rec.manager._pop_active(rec.handle)                       # drop accounting + stale D2H event off old ptr
        _return_cpu(rec.handle.tensor, pin_memory=True)          # pool reuse safe: seal event already synced
        object.__setattr__(rec.handle, "tensor", _SENTINEL)      # stray compute read ⇒ loud 0-numel error

    # ---------- consumer side (MAIN THREAD) ----------
    def ensure_local(self, handle):
        """First line of wait_cpu_ready() (stage*() already call it) + the one direct-read site (attn bwd :715).
        O(1) dict miss for untracked handles."""
        rec = self._by_id.get(id(handle))
        if rec is None: return
        with self._lock:
            if rec.state in (TRACKED, QUEUED, SUBMITTED):        # CPU-valid (writer only READS the buffer)
                rec.state = CLAIMED; return                      # dequeue is lazy (state check in _maybe_spill)
            if rec.state in (CLAIMED, FETCHED): return
            assert rec.state == DURABLE
        bounce = _alloc_cpu(handle.original_shape, handle.original_dtype, pin_memory=True)   # padded pool buffer
        arrived = self._store.fetch_into(rec.ref, bounce)        # drains in-flight prefetches too (Stage 5)
        self._store.blob_done(rec.ref)
        object.__setattr__(handle, "tensor", bounce)
        rec.manager._mark_cpu_live(handle)                       # re-enter accounting under the new data_ptr
        with self._lock:
            rec.state = FETCHED; self.live_cpu_bytes += rec.nbytes
            self.stats.note_fetch(rec.substrate, rec.nbytes)
        self._settle_prefetches(arrived)                         # Stage 5 (no-op while prefetch_bytes==0)
        self._prefetch_reverse(rec)                              # Stage 5 (no-op)
        self._maybe_spill()                                      # fetch may push live over hi → spill oldest (still Belady)

    def on_release(self, handle):
        """FIRST line of release_cpu() — MUST precede its internal wait_cpu_ready (:318), else a
        spilled-never-consumed handle is fetched just to be freed. Covers every terminal state; idempotent."""
        rec = self._by_id.pop(id(handle), None)
        if rec is None: return
        with self._lock:
            if rec.state in (TRACKED, QUEUED, CLAIMED, FETCHED):
                self.live_cpu_bytes -= rec.nbytes                # DURABLE bytes already decremented at spill
            if rec.state == SUBMITTED: rec.state = CLAIMED       # async: on_durable will drop the blob
            elif rec.state == DURABLE: self._store.blob_done(rec.ref)   # spilled, never fetched
            elif rec.state == QUEUED:  rec.state = CLAIMED       # lazy-dequeue marker
        # stock release_cpu then runs: sentinel no-ops (ptr 0 not in accounting); resident/fetched pool-return.

    def summary(self): return self.stats.as_dict() | {"live_peak_bytes": self.stats.live_peak,
                                                       "hi": self.hi, "lo": self.lo, "open_recs": len(self._by_id)}

_GOV = None; _GOV_INIT = False
def get_act_spill_governor():
    global _GOV, _GOV_INIT
    if _GOV_INIT: return _GOV
    store = get_nvme_store()
    _GOV = ActSpillGovernor(store) if (store is not None and store.has_role("activation")) else None
    _GOV_INIT = True
    return _GOV
```

**Why the event algebra is safe:** the seal event is recorded after every D2H fill and every forward kernel
launch that streams these buffers (all earlier on the same stream). The spiller `synchronize()`s it before
`pwrite` (stable bytes); pool-return happens only after that same sync (no kernel can still stream a recycled
buffer). Backward consumers always pass through `ensure_local` first. In async mode `SUBMITTED→CLAIMED` is safe
because the writer only *reads* the buffer; the wasted blob is dropped unread. The `data_ptr`-keyed accounting
hazard is avoided because `_pop_active` runs before pool-return and the handle's tensor is swapped to the sentinel.

### 3.2 Manager integration (`activation_offload.py`, ~30 lines — lands ALL hooks now, inert without seals)

```python
# module top:  from .act_spill_governor import get_act_spill_governor ; _GOV = get_act_spill_governor()
#   (None unless role "activation" — rule 7; import is cheap, get_nvme_store() returns None when disabled)
# empty_cpu:  after self._mark_cpu_live(handle)  (line :184) →  if _GOV: _GOV.on_offload(self, handle)
# adopt_cpu:  after self._mark_cpu_live(handle)  (line :227) →  if _GOV: _GOV.on_offload(self, handle)
#   (C5: exactly the two _mark_cpu_live sites; offload() delegates to empty_cpu so it is covered transitively.)
# wait_cpu_ready(handle): FIRST line →  if _GOV: _GOV.ensure_local(handle)     (covers stage*/direct reads)
# release_cpu(handle):    FIRST line →  if _GOV: _GOV.on_release(handle)        (BEFORE the :318 wait)
# NEW seal(*handles):  if _GOV: _GOV.on_seal(self, [h for h in handles if h is not None])   (engine sugar)
# NEW _pop_active(handle):  # governor-only; mirror release_cpu :319-324 stat decrements WITHOUT pool return,
#                           # AND drop the stale D2H event for the recycled ptr.
def _pop_active(self, handle):
    ptr = int(handle.tensor.data_ptr())
    self._pending_cpu_ready_events.pop(ptr, None)
    entry = self._active_cpu_bytes.pop(ptr, None)
    if entry is None: return
    nbytes, tag = entry
    self.stats.cpu_owned_bytes = max(0, self.stats.cpu_owned_bytes - nbytes)
    self.stats.cpu_bytes_by_tag[tag] = max(0, self.stats.cpu_bytes_by_tag.get(tag, 0) - nbytes)
# _alloc_cpu: when _GOV — allocate via nvme_store.alloc_padded_pinned(shape, dtype, align=_GOV._store.align)
#   so pooled buffers carry ALIGNED storages (io_ready True). Pool key (dtype,shape,pinned) unchanged; pin-failure
#   fallback stays (io_ready excludes those). Guard _CPU_BUFFER_POOL with a module lock ONLY when _GOV (Stage 5
#   writer returns buffers off-main-thread; sync is main-only but the lock is uncontended and simple).
```

**Eligibility rule (locked):** `on_offload` tracks every handle for pressure; **only handles passed to `seal()`
ever spill.** Backward-created transients (`mlp.dact/dgate/dup`, `moe.*`, stage scratch) are never sealed ⇒
structurally excluded, no phase detection.

### 3.3 Substrate A wiring (`gc_boundary_offload.py` + the 4-line LF hook)

```python
# asym_gemm/training/gc_boundary_offload.py
import os, torch
from .activation_offload import ActivationOffloadManager
from .act_spill_governor import get_act_spill_governor

_BOUNDARY_MANAGER = None                                          # ONE process-global manager owns boundaries
def _get_boundary_manager():
    global _BOUNDARY_MANAGER
    if _BOUNDARY_MANAGER is None: _BOUNDARY_MANAGER = ActivationOffloadManager(pin_memory=True)
    return _BOUNDARY_MANAGER
def get_boundary_offload_stats():
    return _BOUNDARY_MANAGER.snapshot() if _BOUNDARY_MANAGER is not None else {"enabled": False}

_OUTER_SAVE_COUNTER = 0
def _keep_on_hbm_diagnostic():                                   # replicate checkpointing.py:65-75 verbatim
    every_n = int(os.environ.get("UNSLOTH_GC_OUTER_HBM_EVERY_N", "0") or 0)
    if every_n <= 0: return False
    global _OUTER_SAVE_COUNTER
    keep = (_OUTER_SAVE_COUNTER % every_n) == 0; _OUTER_SAVE_COUNTER += 1
    return keep

def get_asym_unsloth_gc_func():
    """Drop-in for LF's UnslothGradientCheckpointing.apply, used only when role 'activation' is enabled.
    Same math, same recompute, same save_on_cpu region. Decorators MIRROR the reference verbatim (C9-torch:
    torch.cuda.amp.* still works in 2.12; do not 'modernize' and diverge)."""
    mgr = _get_boundary_manager(); gov = get_act_spill_governor()

    class AsymUnslothGCOffload(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, forward_function, hidden_states, *args):
            if hidden_states.is_cuda and not _keep_on_hbm_diagnostic():
                handle = mgr.offload(hidden_states, "gc.boundary")   # pinned pool + async D2H; on_offload tracks
                mgr.seal(handle)                                     # sole consumer is backward ⇒ seal now
                ctx.asym_handle = handle; ctx.hbm_hidden = None      # tensor rides the handle, NOT autograd (C9)
            else:
                ctx.asym_handle = None; ctx.hbm_hidden = hidden_states
            with torch.no_grad():
                outputs = forward_function(hidden_states, *args)
            ctx.forward_function = forward_function; ctx.args = args
            ctx.save_for_backward()                                 # empty: detached CPU blob has no grad_fn (C9)
            return outputs

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, grad_output):
            h = ctx.asym_handle
            try:
                if h is not None:
                    mgr.wait_cpu_ready(h)                           # ensure_local: un-spill if DURABLE
                    hidden = h.tensor.to("cuda", non_blocking=True) # pinned H2D (faster than today's pageable)
                    ev = torch.cuda.Event(); ev.record()
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
                if h is not None: ev.synchronize()                 # H2D done ⇒ buffer reusable
                return (None, hidden.grad) + (None,) * len(ctx.args)
            finally:
                if h is not None: mgr.release_cpu(h)                # LIFO release → pool + governor (C9 mitigation)
                ctx.asym_handle = None                             # drop the stash ref (C9 mitigation)
    return AsymUnslothGCOffload.apply
```

LF hook — first lines of `get_unsloth_gradient_checkpointing_func` (`checkpointing.py:78`), mirroring the
existing `UNSLOTH_GC_*` env-hook style:

```python
    if _env_flag("ASYM_UNSLOTH_GC_NVME"):        # exported by run_lf when ASYM_NVME_ROLES contains "activation"
        from asym_gemm.training.gc_boundary_offload import get_asym_unsloth_gc_func
        return get_asym_unsloth_gc_func()
```

**Correctness notes:** `ctx.save_for_backward()` empty is legal (no double-backward — the inner region uses
reentrant `torch.autograd.backward`, same as today); `hidden.grad` flows exactly as today (`:117`); the boundary
tensor is bf16 `[M,H]` contiguous → ONE pool shape class per model ⇒ perfect pool reuse layer-over-layer
(steady-state pinned usage ≈ budget, not footprint); the `try/finally` guarantees `release_cpu` even if the
inner backward raises (C9); a debug leak counter = governor `open_recs` at step end (should be 0).

### 3.4 Efficiency

Memory: boundary copies move pageable→pooled-pinned (same bytes, recycled); spilled bytes leave RAM entirely.
Latency (v1, removed in Stage 5): sync spill blocks main `bytes/14GBps` (~0.4 s per 5.5 GiB boundary) in
forward; sync fetch `bytes/26GBps` (~0.2 s/layer) in backward; oldest-first spills layer-0-side handles whose
seal events completed long ago ⇒ `synchronize()` ~free. Launches: +1 event/boundary (64/step) — noise. **NO
GEMM shape/count changes anywhere.**

**Validation (Stage 3 gate — capacity mode, REAL e2e):**

Unit (`tests/training/test_act_spill_governor.py`): state transitions (sync
TRACKED→QUEUED→DURABLE→FETCHED; QUEUED→CLAIMED; DURABLE-released-unfetched drops blob); oldest-first order;
hysteresis stops at `lo`; unsealed handles never spill under pressure; `io_ready`-failing handles never spill;
sentinel compute-read raises; budget=0 eager spills everything and stays bit-exact; a toy 2-layer module under
`get_asym_unsloth_gc_func()`: fwd+bwd **bit-identical** with `ASYM_NVME_ROLES=""` vs `"activation"`+tiny budget;
existing `test_dense_mlp_finegrained.py` passes NVMe off AND on (hooks inert without seals).

E2E (budget forces spilling; sequential; `kill -TERM` only):

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

**Accept:** candidate per-step RSS drops ≈ (boundary footprint − budget) (s30000 boundary ≈ 147 GiB analytic;
budget 120 ⇒ expect ≥30 GiB drop); `asym_nvme.bytes_written ≈ bytes_read ≈` overflow volume; `act_governor.
live_peak ≈` budget; **losses identical step-for-step** (`--max-loss-delta 0`); HBM unchanged; step ratio
recorded (informational at this rung); `open_recs==0` at end. **Then the headline probe:** raise seq until the
baseline dies (watchdog, floor 35 GB) and the candidate still trains — e.g. s90000 budget 400 GiB; verify real
`input_ids` length in `train.log`; record both max-seq numbers.

### Risks / watch
- **Reentrant nesting:** the inner `torch.autograd.backward` runs Substrate-B Functions whose
  `wait_cpu_ready→ensure_local` may nest inside the outer boundary fetch path — all main-thread (no deadlock;
  the governor lock is NOT held around IO — note the `RLock`), but debug-assert `ensure_local` non-reentrancy
  per handle.
- **Pool-cap churn:** 5.5 GiB boundary buffers vs the 32 GiB default pool cap ⇒ set
  `ASYM_EXPACT_CPU_POOL_MAX_BYTES` ≥ budget-scale even for dense models; watch `cpu_pool_evictions`.
- **C9 leak:** confirm governor `open_recs==0` after each step; if it grows, the `ctx.asym_handle` cleanup or a
  backward exception path leaked — the `try/finally` is the guard.
- If step-1 loss differs at all: rerun with budget=∞ (tracks, never spills — isolates the pinned-copy change
  from the spill path) before suspecting the governor.
- **`auto` budget** = 0.85×MemAvailable at init is wrong on a shared box — ALWAYS set the env explicitly for gates.
- **Decorators (C9-torch):** mirror `@torch.cuda.amp.custom_fwd/_bwd` verbatim (the reference uses them in
  production; they still delegate correctly in 2.12); do not switch to `torch.amp.custom_fwd(device_type=...)`
  and risk a behavior diff during bring-up.

---

## Stage 4 — actnvme Substrate B seals: dense-fg + attention U/S (flagship-complete)

**What:** add `seal()` calls so intra-layer fine-grained handles participate. They are the newest FIFO entries
⇒ spill only under extreme pressure (terminal-margin property for free). **No governor changes.**

**Scope:** `dense_mlp_finegrained.py` (1 seal line), `attention_activation_offload.py` (~6 lines + one bool on
`_SharedActivationSource`), env `ASYM_NVME_ACT_ENGINES=boundary,dense_fg,attention` (default all-on when role
enabled; per-engine kill-switch for bisection — the manager `seal()` checks the engine flag before calling
`_GOV.on_seal`).

```python
# dense_mlp_finegrained.py — immediately before `layer._last_activation_offload_stats = manager.snapshot()` (:300):
manager.seal(x_cpu, gate_cpu, up_cpu, act_cpu, gate_low_rank_cpu, up_low_rank_cpu, down_low_rank_cpu)
#   handles created at :216/:230/:246/(:252|:259)/:231/:247/:268 ; x_cpu (oldest, consumed last :421) is the
#   only fg handle with real lead time — and exactly the one oldest-first picks first.
```

```python
# attention_activation_offload.py
# (i) _SharedActivationSource (:417-440) — ADD a bool (it has `released`+`refcount`, NO `closed` — C6):
class _SharedActivationSource:
    def __init__(self, ...):
        ...; self.closed = False                          # set True by acquire_source at both cache-clear branches
# (ii) AttentionActivationOffloadContext.acquire_source (:456-480) — set closed in BOTH clear branches (:477-479):
#      when role=="v_proj" OR all of {q,k,v}_proj seen → source.closed = True  (BEFORE _cache.clear() at :478)
# (iii) _AsymActivationOffloadLoRALinearFunction.forward — after the ctx stashes (:632-650), before return (:653):
manager.seal(s_handle)                                    # per-call S: this Function is its sole consumer
if ctx.shared_source is None:
    manager.seal(u_handle)                                # non-shared U: this Function is the only consumer
elif ctx.shared_source.closed:
    attention_context.manager.seal(u_handle)              # shared U: seal via the PERSISTENT manager that owns
                                                          # the U buffer, at the END of the LAST acquirer's fwd.
                                                          # cache-clear (:478) happens BEFORE v_proj's own CPU-left
                                                          # read (:620), so sealing at clear-time would be premature.
# (iv) backward (:715), BEFORE `u_source = _pad_cpu_rows_to(u_handle.tensor, …)` — the ONE uncovered direct read
#      (no wait_cpu_ready anywhere in this backward — C6):
if _GOV: _GOV.ensure_local(u_handle)
#   _pad_cpu_rows_to (:64) is a HOST-side CPU pad feeding asym_bf16_cpu_right_matmul (:718-728) as the CPU
#   operand. The finally (:736-743) releases s_stage/s_handle, then u_handle (non-shared) or shared_source
#   release, then _update_snapshot (:743). Extend AttentionActivationOffloadContext.snapshot() (:485-498) so
#   shared-U spill counters surface under source_context (via _update_snapshot :501-511).
```

**Audit note (verified §0.4 + Agent-2 re-check):** every OTHER backward read in both engines passes through
`stage*()`/`wait_cpu_ready` ⇒ covered by the Stage-3 hook. Dense-fg backward mid-body releases
(`:333/:357-358/:373/:381/:395/:410/:419/:436`) + the `finally` (`:440-453`) give exact release-on-last-use;
`on_release` is idempotent against the double calls.

**Efficiency:** at sane budgets these never spill (newest); when they do (budget ≪ window) the fg backward
consumes them within the same layer window — sync fetch bounded by window size. No new launches beyond one seal
event per Function.

**Validation (Stage 4 gate):**

Unit: extend the governor test with a real `_FinegrainedDenseMLPFunction` fwd+bwd (toy shapes, CUDA)
bit-identical off vs on+tiny-budget; attention tests (`test_attention_activation_offload_lora.py`,
`_helpers.py`) pass off AND on; NEW test: q/k/v share + spill shared U after the v_proj seal, verify all three
projections' backwards fetch once and `source_share_released_bytes` is unchanged.

```bash
# rerun the exact Stage-3 command pair with a DEEPER budget to force fg spills at s30000:
ASYM_NVME_ACT_CPU_BUDGET_BYTES=$((60*1024*1024*1024)) \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds_actnvme|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false' \
  ... (same driver + compare as Stage 3, --target activation_cpu --max-loss-delta 0)
```

**Accept:** losses identical; `act_governor` shows `substrate=fg` spilled bytes > 0; `cpu_peak_by_tag` for
`mlp.*`/`*.U`/`*.S` collapses vs baseline; `wasted_writes==0` (sync); step ratio recorded. Then one MoE smoke
(`q3.5-35b-a3b|1 ; … ; 45000|8|1`) with the MoE engine NOT yet sealed — proves partial coverage degrades to
status quo (`moe.*` stays resident), never corrupts.

**Risks / watch:**
- Shared-U `closed`: `acquire_source` has two clear branches (`role=="v_proj"` OR all roles seen, `:477`) — set
  `closed` in both; debug-assert a spilled shared U was sealed only after close. **Verify the u_handle passed to
  `attention_context.manager.seal` is the handle that manager actually owns** (the shared buffer), not a per-call
  alias — small unit test with q/k/v.
- `_update_snapshot` (`:501-511`): governor/nvme counters ride the per-call manager snapshot; the shared-U
  manager is the persistent one, so its counters surface via `source_context`.
- If U/S never spill even at tiny budgets, that is CORRECT (newest-first-resident) — check `substrate=fg`
  counters before hunting ghosts.

---

## Stage 5 — actnvme v2/v3 ASYNC: writer thread + reverse-order prefetch (+ optional H2D stream)

**What:** flip `ASYM_NVME_SYNC=0`. Spills go to the writer thread (forward never blocks on NVMe; backpressure at
`max_inflight_spill_bytes`); backward hides fetch latency with byte-budgeted **reverse-creation-order** prefetch
(`ASYM_NVME_ACT_PREFETCH_BYTES`, e.g. 2-3 boundary blobs ≈ 12-16 GiB). The spilled list walked newest→oldest
matches LIFO consumption; a mis-ordered prefetch is an efficiency blip, never a bug (`drain_reads` + durable
events serialize correctness).

**Scope:** `nvme_store.py` `_WriterThread` (Stage-1 code, now exercised); governor `_make_on_durable`,
`_prefetch_reverse`, `_settle_prefetches`.

```python
# governor: async spill callback = _finish_spill running on the writer thread (CUDA-free). ready_event moves to
# the writer so the MAIN thread never blocks on the D2H sync:
def _make_on_durable(self, rec):
    def _cb(ref): self._finish_spill(rec)          # runs in writer context; rec.ref already set
    return _cb
# and in _maybe_spill's async branch pass ready_event=rec.seal_event (NOT pre-synced):
#     rec.ref = self._store.spill_async("activation", rec.handle.tensor,
#                                       ready_event=rec.seal_event, on_done=self._make_on_durable(rec))

def _prefetch_reverse(self, just_fetched):
    if self.prefetch_bytes <= 0: return
    inflight = self._inflight_prefetch_bytes()
    for r in self._iter_spilled_newer_first(from_rec=just_fetched):     # reverse creation order = LIFO order
        if inflight >= self.prefetch_bytes: break
        if r.state == DURABLE and r.prefetch_buf is None:
            r.prefetch_buf = _alloc_cpu(r.handle.original_shape, r.handle.original_dtype, pin_memory=True)
            self._store.submit_pread(r.ref, r.prefetch_buf); inflight += r.nbytes

def _settle_prefetches(self, arrived_ref_ids):                          # main thread, after any drain
    for rec in self._take_arrived(arrived_ref_ids):
        self._store.blob_done(rec.ref)
        object.__setattr__(rec.handle, "tensor", rec.prefetch_buf); rec.prefetch_buf = None
        rec.manager._mark_cpu_live(rec.handle)
        with self._lock: rec.state = FETCHED; self.live_cpu_bytes += rec.nbytes
```

Async-only correctness pieces (already in the Stage-3 state machine): `SUBMITTED→CLAIMED`
consume-while-in-flight (writer's `_finish_spill` sees CLAIMED → drops blob, buffer stays handle-owned,
`wasted_writes+=1`); pool lock active (writer returns buffers); `ref.durable.wait()` in `submit_pread` covers
prefetch-of-in-flight-write. Optional v4 rung (`ASYM_NVME_ACT_H2D_STREAM=1`, only if profiles show H2D on the
critical path): boundary H2D on a side stream + event (TE-v1 `cpu_offload_v1.py:366-367,:578`); default OFF.

**Efficiency:** forward's only NVMe cost becomes the CV-wait when >8 GiB of writes are in flight (14 GB/s drains
5.5 GiB boundaries faster than long-seq layers produce them); backward fetch wait → ≈0 when prefetch ≥ per-layer
consumption × fetch latency; the `assert prefetch_bytes < hi-lo` at init keeps prefetch from re-triggering spill
thrash.

**Validation (Stage 5 gate — throughput at the same operating point):**

```bash
# same paired RUNS as Stage 3 (s30000, budget 120 GiB) but with:
ASYM_NVME_SYNC=0 ASYM_NVME_ACT_PREFETCH_BYTES=$((12*1024*1024*1024)) ... (same driver)
$LF_PY scripts/lf/compare_nvme_profiles.py --baseline <asym_cpuadamwds dir> --candidate <actnvme async dir> \
  --target activation_cpu --min-memory-drop-gib 30 --max-step-ratio 1.10 --max-forward-ratio 1.10 \
  --max-backward-ratio 1.10 --expect-nvme-role activation --max-loss-delta 0
# plus candidate-vs-candidate: async step ≤ sync step − 0.8×(spill_wait_ms+fetch_wait_ms)/step
```

**Accept:** memory drop as Stage 3; step ≤1.10× baseline; `spill_backpressure_ms ≈ 0`; `fetch_wait_ms` ↓ ≥5× vs
the sync run; `wasted_writes ≈ 0`; losses identical to baseline AND to the sync candidate (bit-for-bit).
Cross-check device IO: `offload_io.json` (from `io_samples.csv`, O_DIRECT ⇒ page-cache-free) totals ≈
`asym_nvme` byte counters ±10%. Then re-run the max-seq probe with async on — step-ratio is now the meaningful
capacity price.

**Risks / watch:**
- If `spill_backpressure_ms` dominates at the capacity point: overflow rate exceeds ~14 GB/s — raise budget or
  accept/report the stall. The Stage-0 census predicts this — check it before blaming code.
- Prefetch-order vs consumption-order mismatch intra-layer (x before S_up, `:396` vs `:415`) — bounded by one
  layer's bytes; small residual `fetch_wait_ms`; do NOT special-case tags.
- Writer-thread discipline: `ready_event.synchronize()` is host-side (allowed off main); everything else
  CUDA-free — assert no `torch.cuda` allocs in the writer path under `ASYM_NVME_DEBUG=1`.

---

## Stage 6 — actnvme Substrate B coverage: qwen3/qwen3.5 MoE fine-grained engine

**What:** one seal line for `_Qwen3MoeFinegrainedFunction` — required for the MoE hero models (q3.5-35b-a3b,
q3.5-122b-a10b, q3-30b-a3b). **The direct-read audit is RESOLVED (C7): no extra ensure calls needed.**

**Scope:** `qwen3_moe_finegrained.py` (1 seal line).

```python
# immediately before `layer._last_activation_offload_stats = _record_manager_peaks(layer, manager)` (:706):
manager.seal(x_cpu, gate_cpu, up_cpu, act_cpu, gate_low_rank_cpu, up_low_rank_cpu, down_low_rank_cpu)
#   handles created at (:498|:500)/:527/:556/:568/:528/:557/:588.
# Audit RESOLVED (C7): backward direct reads are wait-covered — act via :777/:902; for x, the SINGLE
# `manager.wait_cpu_ready(ctx.x_cpu)` at :968 (head of the down_scatter_block_experts>0 branch) DOMINATES every
# `ctx.x_cpu.tensor[row_slice]` read in both block loops (:1021 gate, :1106 up) — one hook un-spills x for the
# whole loop; the else-path uses per-read waits (:1198→:1202, :1299→:1303). NO per-block ensure calls.
# Transients moe.dup/dgate (:945/:962) are never sealed; full-fg sets KEEP_DGRADS_HBM=1 anyway.
# _record_manager_peaks (:96-112) calls manager.snapshot() then decorates in place ⇒ nvme keys survive.
```

Legacy engines (`qwen3_moe.py` FnA `:1010`, `llama4_experts.py` `:229`, shared MLPs) get seals the same
one-line way ONLY if a target config re-activates them (`ASYMM_EXPERT_ACT_OFFLOAD`-gated, off in full-fg) —
deferred, not dead. Their release loops lack `finally` — an exception mid-backward leaks governor recs
(pre-existing leak shape; training aborts anyway — noted, not fixed here).

**Validation (Stage 6 gate):**

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage6_moe PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=1 MAX_STEPS=4 ASYM_NVME_ACT_CPU_BUDGET_BYTES=$((150*1024*1024*1024)) \
RUNS='q3.5-35b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101|ligerloss1 ; 45000|8|1 ; none|false|true|false|false|false || q3.5-35b-a3b|1 ; asym_cpuadamwds_actnvme|recomp-off-full-fg-ker101|ligerloss1 ; 45000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
# compare: --target activation_cpu --min-memory-drop-gib 30 --max-step-ratio 1.10 --max-loss-delta 0
```

**Accept:** identical losses; `moe.*` in `cpu_peak_by_tag` collapses under deep pressure (repeat budget 60 GiB);
step ≤1.10×; `expert_token_distribution` unchanged (routing untouched). **Verify the tuple spelling (`ker101`
etc.) against the existing `profiling_q35_final2_asym80k_*` run dir before launching.**

**Risks/watch:** MoE pool cap (192 GiB) and governor budget are SEPARATE knobs (pool caches FREE buffers;
governor counts LIVE handles) — both enter the Stage-8 RAM ledger.

---

## Stage 7 — `panvme`: base weights → NVMe (trace-prefetch pinned cache)

**Scope:** `host_weight.py` (property surgery, ~15 lines), NEW `asym_gemm/training/base_weight_pager.py`,
`integrations/lf.py` (registration walk before `return model, report` at **`:2426`**, after
`_release_replaced_module_memory()` `:2425`), `qwen3_moe.py` (force the eager fine-grained split `:2509`).

**Consumer map (Agent-3 verified — this is what makes the surgery correct):** every runtime `.weight`/`.tensor`
COMPUTE read goes **through the property** (frozen_linear `:1312/:1358/:1468/:1486/:1528/:1532/:2049`, qwen3_moe
`:488/:1893`, the fused-slice `:2516`, state-dict saves `:1681/:2078`) ⇒ a property getter hook covers them. But
`HostWeight.shape/dtype/device/in_features/out_features` read `self._tensor` **directly** (`:270-296`) and fire
**around** the GEMM (`in_features:1303/1306`, `out_features:1338` *before* the `.weight` fetch at `:1312`) ⇒ they
must be **rerouted to `self._metadata` when paged**, else `_tensor=None` crashes them. `is_pinned()` predicates
(qwen3_moe `:1775/:2665/:2670`) do `.weight.is_pinned()` — they fetch then get True; self-consistent (each
precedes a consuming kernel, so `touch()` dedupe reuses the buffer).

```python
# host_weight.py — surgery (everything else byte-identical). NEW instance fields default None, set by register():
#   self._pager = None ; self._pager_key = None   (add to __init__)
    @property
    def weight(self):
        if self._pager is None: return self._tensor              # today's exact path (one None check)
        return self._pager.touch(self._pager_key)
    @property
    def tensor(self):
        if self._pager is None: return self._tensor
        return self._pager.touch(self._pager_key)                # keep both in lockstep
    # shape (:272)/dtype (:276)/out_features (:292)/in_features (:296): when paged, read self._metadata:
    @property
    def shape(self):        return self._tensor.shape        if self._pager is None else torch.Size(self._metadata.shape)
    @property
    def dtype(self):        return self._tensor.dtype        if self._pager is None else self._metadata.dtype
    @property
    def out_features(self): return int((self._tensor.shape if self._pager is None else self._metadata.shape)[0])
    @property
    def in_features(self):  return int((self._tensor.shape if self._pager is None else self._metadata.shape)[1])
    @property
    def device(self):       return self._tensor.device      if self._pager is None else torch.device(self._metadata.device)  # str→device
    @property
    def grad(self):         return None if self._pager is not None else self._tensor.grad   # frozen; assert never set
    def grouped_nt_tensor(self):                                 # route through touch() (unused repo-wide, defensive)
        return (self._tensor if self._pager is None else self._pager.touch(self._pager_key)).unsqueeze(0)
    # pin_memory (:337) / to() (:317) / cuda() (:334): raise when paged (registration is post-placement).
    # nbytes (:283) / is_pinned (:287) / metadata (:266): already metadata-backed — UNCHANGED.
    #   is_pinned stays True when paged (pager buffers are pinned).
```

```python
# base_weight_pager.py — trace-prefetching pinned cache (DeepSpeed param-coordinator pattern; MAIN THREAD ONLY)
from dataclasses import dataclass, field
from typing import Any
import torch
from .nvme_store import alloc_padded_pinned, _pad
ABSENT, INFLIGHT, RESIDENT = range(3)

@dataclass
class _Entry:
    hw: Any; ref: Any; shape: tuple; dtype: Any; padded_nbytes: int
    buf: Any = None; view: Any = None; state: int = ABSENT; positions: list = field(default_factory=list)

class BaseWeightPager:
    def __init__(self, store, *, cache_bytes=_env_int("ASYM_NVME_BASE_WEIGHT_CACHE_BYTES", 16<<30),
                 prefetch_bytes=_env_int("ASYM_NVME_BASE_WEIGHT_PREFETCH_BYTES", 0)):   # 0 → auto: 2×largest blob
        self._store = store; self.cache_bytes = cache_bytes; self.prefetch_bytes = prefetch_bytes
        self._entries = {}; self._by_ref_id = {}
        self._free = {}                          # (dtype,shape) → [padded pinned bufs] free list
        self._quarantine = []                    # (buf, cuda_event, (dtype,shape)) — event-gated reuse (rule 6)
        self._trace = []; self._frozen = False; self._disabled = False
        self._cursor = -1; self._last_key = None; self._resident_bytes = 0
        self.misses = 0; self.misses_after_freeze = 0

    def register(self, key, hw):
        t = hw._tensor
        if t is None or t.numel()*t.element_size() < self._store.cfg.min_swappable_bytes: return
        blob = _pad(t.numel()*t.element_size(), self._store.align)
        assert self.cache_bytes >= 2*blob, f"cache_bytes must hold ≥2× largest blob ({2*blob} B)"
        self.prefetch_bytes = self.prefetch_bytes or 2*blob
        padded = alloc_padded_pinned(tuple(t.shape), t.dtype, align=self._store.align); padded.copy_(t)
        ref = self._store.spill_sync("base_weight", padded)         # written once at startup; sync is right
        e = _Entry(hw=hw, ref=ref, shape=tuple(t.shape), dtype=t.dtype, padded_nbytes=blob)
        self._entries[key] = e; self._by_ref_id[id(ref)] = e
        hw._pager, hw._pager_key = self, key
        hw._tensor = None                                            # GB-scale home freed NOW

    def touch(self, key):
        e = self._entries[key]
        if key != self._last_key:                                   # dedupe: .weight read several times per Function
            self._last_key = key
            self._record_or_advance(e); self._issue_prefetches()
        if e.state == RESIDENT: return e.view
        if e.state == INFLIGHT:
            for rid in self._store.drain_reads(): self._by_ref_id[rid].state = RESIDENT
            return e.view
        e.buf = self._take_buffer(e); e.view = e.buf                # ABSENT miss (step 1; ~never after freeze)
        self._store.fetch_into(e.ref, e.buf); e.state = RESIDENT
        self.misses += 1; self.misses_after_freeze += int(self._frozen)
        return e.view

    # _record_or_advance: during step 1 append key to self._trace and set positions; freeze when the FIRST key
    #   recurs its 3rd time (fwd@period0, its bwd, fwd@period1). After freeze, cursor-advance with a jitter
    #   window of 8; a mismatch beyond the window → self._disabled = True (miss-driven sync fallback, counted,
    #   loud in summary()).
    # _issue_prefetches: byte-budgeted lookahead over the frozen trace (uniform lead TIME under mixed blob sizes
    #   — MoE 3D groups vs dense 2D); guarded by _would_evict_nearer_than (tight-cache DS max_live analog):
    #   for the next entries on the trace, if free/growable and not already INFLIGHT/RESIDENT, alloc buf, mark
    #   INFLIGHT, submit_pread; stop at prefetch_bytes.
    # _take_buffer(e): pop free-list by (dtype,shape) → else grow while _resident_bytes+padded ≤ cache_bytes →
    #   else evict the RESIDENT entry with the farthest next-use on the frozen trace (exact Belady); the
    #   evictee's buf → _quarantine with a post-launch CUDA event; _sweep_quarantine returns event-complete
    #   buffers to _free.  Update _resident_bytes.
    def summary(self):
        return {"registered": len(self._entries), "misses": self.misses,
                "misses_after_freeze": self.misses_after_freeze, "trace_frozen": self._frozen,
                "trace_disabled": self._disabled, "resident_bytes": self._resident_bytes,
                "cache_bytes": self.cache_bytes, "prefetch_bytes": self.prefetch_bytes}
```

Registration walk (end of `apply_lf_asym_lora` `:1709`, inserted after `:2425`, before `return model, report`
`:2426`) + the eager split:

```python
store = get_nvme_store()
if store is not None and store.has_role("base_weight"):
    if _qwen3_moe_finegrained_offload_enabled():                    # force lazy fused→gate/up split (:2509) FIRST,
        for m in model.modules():                                   # while the fused parent's _tensor is resident
            if is_qwen3_experts(m): m._ensure_qwen3_moe_finegrained_bases()
    pager = BaseWeightPager(store)
    for name, mod in model.named_modules():
        hw = getattr(mod, "host_weight", None)                      # catches AsymFrozenLinear AND grouped/expert banks
        # eligible: bf16 AsymFrozenLinear (attn + dense mlp) + AsymGroupedFrozenLinear (experts, shared, router);
        # EXCLUDED: embed_tokens (CPU-side F.embedding per microbatch, offload.py:381), norms (unpinned by policy),
        #           anything precision != "bf16" (quantized cache builds from .weight).
        if isinstance(hw, HostWeight) and _panvme_component_eligible(name, mod):
            assert getattr(mod, "precision", "bf16") == "bf16"      # bf16 short-circuit at frozen_linear.py:363-364
            pager.register(name, hw)
    model._asym_base_weight_pager = pager
```

**Correctness anchors:** fwd+bwd interleaved trace ⇒ farthest-next-use is exact Belady (late layers, reused
first in backward, survive after forward); event-gated quarantine per rule 6 (`_asym_bf16_nt` launches return
immediately while streaming the pinned weight, `frozen_linear.py:726-728,:807-809`); `touch` dedupe absorbs the
multi-read pattern (`.weight.is_pinned()` predicate + kernel read within one Function); reporting reads metadata
(never fetches); step-1 is miss-driven by design (⇒ `WARMUP_STEPS≥1` mandatory); grouped 3D expert blobs are
GB-scale — `assert cache_bytes ≥ 2×largest_padded_nbytes` at registration.

**Efficiency:** whole-blob IO (a 2 GiB expert bank is ONE pread, internally 4×16-parallel); grouped weights
stay grouped — **no per-expert loops introduced anywhere**; prefetch keeps the compute stream fed with zero HBM
change; `touch` fast path = one dict get + one identity compare.

**Validation (Stage 7 gate):**

Unit (`tests/training/test_base_weight_pager.py`): register frees `_tensor` (RSS probe); roundtrip bit-exact
(2D + grouped 3D bf16); freeze at 3rd first-key occurrence; Belady eviction picks farthest; quarantine blocks
reuse until event completes; jitter tolerance; disable-fallback correctness; **metadata properties
(shape/dtype/device/in_features/out_features) never fetch when paged (counter assert)**; NVMe-off ⇒
byte-identical (existing `test_cpu_resident_frozen_base.py` + `test_lf_qwen3_asym_backend.py` unmodified must
pass).

```bash
GPU_POOL=<gpu> OUTPUT_ROOT=profiling_nvme/stage7_panvme PROFILERS=source PLOT=false \
PREPARE_DATASETS=true WARMUP_STEPS=2 MAX_STEPS=5 \
RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds_panvme|recomp-off-full-fg|ligerloss1 ; 30000|8|1 ; none|false|true|false|false|false' \
scripts/lf/profile_lora_lf_test_source.sh --overwrite true
$LF_PY scripts/lf/compare_nvme_profiles.py --baseline <base> --candidate <panvme> --target base_weight_cpu \
  --min-memory-drop-gib 40 --max-step-ratio 1.05 --max-forward-ratio 1.05 --max-backward-ratio 1.05 \
  --expect-nvme-role base_weight --max-loss-delta 0
```

**Accept:** per-step RSS −40 GiB+ (q3-32b host_weight = 61 GiB; `memory_attribution` host_weight/cpu rows shrink
to match); HBM unchanged; ≤5% step; `misses_after_freeze == 0`; `trace_disabled == false`; `bytes_read ≈
2×base×steps`; losses identical. Then one MoE pass (q3.5-35b-a3b, 105 GiB base, moefg on) — exercises the
eager-split path.

**Risks/watch:** panvme+moefg hard-errors at registration if the eager-split isn't forced first — the walk
order above prevents it. If ≤5% fails at short seq (weight reads / step_seconds too big), rerun at s45000+ and
reclassify panvme as capacity-mode-only for short seq. Any future `.weight` reader must pick compute (fetch) vs
reporting (metadata) — **grep `\.weight` / `\.tensor` / `\._tensor` on rebase** (Agent-3 map is the baseline).
Checkpoint-time `_save_to_state_dict` (`:1681/:2078`) fetches under paging — acceptable (rare); note it.

---

## Stage 8 — `bothnvme`: compose + hero max-seq table

No new mechanism: both roles on one store (governor + pager share it; separate files/arena). **Startup RAM
ledger assert** (loud, with numbers), using the CURRENT watchdog floor:

```
pager.cache_bytes + governor.hi (budget) + ASYM_EXPACT_CPU_POOL_MAX_BYTES + prefetch budgets (act + base)
  + max_inflight_spill_bytes + watchdog floor (35 GiB, run_lf_lora_sft.sh:221) + slack  <  MemTotal (1325 GiB)
```

**Gate:** on q3-32b, then llama3.3-70b, then q3.5-35b-a3b — demonstrate max trainable seq `bothnvme > actnvme
≥ baseline` at fixed b8 (baseline bound = host-mem watchdog kill; verify real `input_ids` lengths in
`train.log`); report HBM peak, per-step RSS, per-role NVMe bytes + wait-ms, step time, overlap fraction
(`1 − nvme_wait/step`). Compare against `zero3_offload_panvme` / `superoffload_mem_panvme` ceilings for the
capability table. Ceiling-probe protocol: `profiling_ceiling_*` naming, **sequential** runs, `kill -TERM` only.

---

## Deferred (explicitly out of scope now)
- Multi-GPU/per-rank stores (rank-suffixed arenas; all ranks pread shared read-only base-weight files) + a
  DeepSpeed-owned backend behind the same store API.
- Legacy expert engines' seals (`qwen3_moe.py` FnA, `llama4_experts.py`, shared MLPs) — same 1-line recipe when
  a config needs them.
- `save_on_cpu` recompute-pack spilling (per-layer-window lifetime — no capacity win; revisit only if window
  transients bind after Substrate A ships).
- GDS/GPU-direct (no hardware path); gradient/optimizer NVMe (LoRA-tiny, `cpu_adam.py:124-134` guard);
  saved-tensor wrapper configs (`layeract`/`layergc`) — superseded by Substrate A for the flagship.

## Global run rules (every e2e gate)
**First obey the HARD RUN CONSTRAINTS (HC1-HC4) at the top** — launch only via the guarded scripts, numactl
membind to CPU nodes 0,1, `oom_score_adj=1000`, watchdog ON @ floor 35. Then:
Heavy runs **sequentially** (600-800 GiB RSS observed; 1325 GiB box, swap 0); stop with `kill -TERM`, never
`-9` (corrupts the DeepSpeed JIT cache); `PREPARE_DATASETS=true` on first use of a workload and **verify real
`input_ids` length in `train.log`**; measure from `step_samples.csv` measured (non-`is_warmup`) rows; set
`ASYM_NVME_ACT_CPU_BUDGET_BYTES` explicitly for gates (never `auto` on a shared box); `.aioenv` is exported by
`run_lf_lora_sft.sh:34-48` — export manually (§Env) for direct pytest; leave the host-mem watchdog (floor 35
GiB) ON for max-seq probes — it is the baseline's OOM referee.

## Implementation order = stage order

```text
Stage 0: traffic census (postprocess-only)        headline verified §0.2; pins budgets — REVIEW FIRST
Stage 1: nvme_store.py substrate                  isolated; unit + AIO smoke gate (bakes in C2/C3/C8 safety)
Stage 2: tokens + env + counters + compare        paired e2e no-change gate (q3-32b s30000 flagship policy)
Stage 3: governor + Substrate A (boundary), SYNC  capacity gate (−GiB @ budget, loss-identical) + max-seq probe ← core
Stage 4: Substrate B seals (dense-fg + attention) flagship-complete under deep pressure (adds `closed`, :715 ensure)
Stage 5: async writer + reverse prefetch          throughput gate ≤1.10 (then re-probe max seq)
Stage 6: MoE fg engine seal                       q3.5-35b-a3b gate (audit resolved — 1 seal line)
Stage 7: panvme pager                             ≤1.05 gate, RSS −40 GiB+ (q3-32b) / −100 GiB (q3.5)
Stage 8: bothnvme hero                            max-seq capability table vs DS NVMe baselines
```

**Why this order:** Substrate A is the measured capacity binder (§0.2) and the smallest tensor-moving diff (one
GC Function + governor); sync-first makes every later rung bisectable against a bit-exact reference; Substrate B
lands before async so the async gate covers the full flagship engine set; panvme after actnvme because its
61-105 GiB is additive but only *required* for the Stage-8 compound hero.

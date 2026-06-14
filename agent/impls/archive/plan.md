# Expert-Side HBM Diagnosis: Staged Implementation Plan

Goal: instrument the `none|true` activation-offload path so we can state, with
allocator ground truth, exactly what HBM the routed-expert path holds at the
true peak, split into LoRA params (kept), LoRA grads (kept), transient
workspace, live activation, and misattributed/non-expert bytes. This is
diagnosis tooling only; it changes no expert math and lands no kernel.

Final verdict target (filled in S5):

```text
At the true peak of the none|true step the routed-expert path holds
L GiB params (kept) + G GiB grads (kept) + W GiB workspace + A GiB activation
+ M GiB misattributed. Expert floor F = peak - (non-expert). Remaining real
expert reduction R = W + A, enumerated with per-item acceptance gates, or R = 0.
```

## Established Facts (verified this session — do not re-derive)

- Forward saves no CUDA activations: `_ActivationOffloadQwen3ExpertFunction.forward`
  (`asym_gemm/training/qwen3_moe.py:885`) `ctx.save_for_backward` at
  `qwen3_moe.py:997` stores only route metadata + the 6 LoRA weight params;
  the 7 activations are CPU `CPUActivationHandle`s (`qwen3_moe.py:988-994`).
  The Function body runs in no-grad, so wide intermediates are created and
  `del`'d inside forward (`qwen3_moe.py:942-984`).
- Scatter does not save `[M,H]`: `scatter_contiguous` (`asym_gemm/training/moe.py:740`)
  does `weighted = flat * weights`; router is detached (`qwen3_moe.py:2542`),
  so `mul` keeps only `weights [M,1]`. `pack_tokens_contiguous` (`moe.py:727`)
  is `index_select` (no value save).
- The CUDA memory snapshot is already implemented and plumbed.
  `run_lf_profiled_train.py:1349` `_start_memory_snapshot_recording` calls
  `torch.cuda.memory._record_memory_history()`; `:1361` `_dump_memory_snapshot`
  calls `torch.cuda.memory._dump_snapshot()`. Driven by
  `ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT` (`:2365`), dumped to
  `<dir>/memory_snapshot.pickle` (`:2404`) in the run `finally` (`:2405`).
  Shell flag: `PROFILE_MEMORY_SNAPSHOT` (`scripts/lf/profile_lora_lf.sh:93`) →
  `ASYM_GEMM_LF_PROFILE_MEMORY_SNAPSHOT` (`scripts/lf/run_lf_lora_sft.sh:1794`).
  The dump is at end-of-run, so the live `segments` are not the peak; the peak
  must be reconstructed by replaying `device_traces`.
- The breakdown attribution is heuristic and multi-source:
  `LFMemoryBreakdownProfiler` (`asym_gemm/profiling/lf_trace.py:663`) attributes
  saved tensors by a module-hook `_component_stack`, tracks live activations by
  weakref, marks param/buffer storages persistent (`_persistent_storage_keys`),
  and infers a `workspace_residual = peak - known` (`lf_trace.py:598-610`). Any
  of these can land non-expert or param bytes in `routed_experts`.
- Env reality: `torch` is not importable on the current login node; GPU runs go
  through the LF workflow. The S1 analyzer must be torch-free so it runs and is
  tested locally; all GPU steps run via `profile_lora_lf.sh`.

Model/byte facts (`config.json`, `b4_s6144`): H=2048, 48 layers, q-dim 4096,
kv-dim 512, I=768, E=128, top_k=8, rank 64; T=24576, M=196608. bf16:
`X/output [M,H]=768 MiB`, `gate/up/act [M,I]=288 MiB`, `gate_up [M,2I]=576 MiB`,
`S_* [M,r]=24 MiB`. LoRA params (kept) = 6336 MiB.

## Shared Diagnostic LF Run

Stages S2-S4 reuse this single-step run (adds `PROFILE_MEMORY_SNAPSHOT=true`;
the old `ASYM_CUDA_MEMORY_SNAPSHOT` env in the prior draft does not exist):

```bash
cd /home/shutianluo/kevin/AsymGEMM-SFT/third_party/AsymGEMM
OUTPUT_ROOT="$PWD/outputs/plan_expact_diag_b4s6144_$(date -u +%Y%m%dT%H%M%SZ)" \
GPU_POOL=0 \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=6144 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
MAX_STEPS=1 \
WARMUP_STEPS=5 \
PROFILE_MEMORY_ATTRIBUTION=true \
PROFILE_MEMORY_BREAKDOWN=true \
PROFILE_MEMORY_BREAKDOWN_INTERVAL=1 \
PROFILE_MEMORY_SNAPSHOT=true \
PLOT=false \
RUN_POST=false \
scripts/lf/profile_lora_lf.sh --gpus 0
```

Snapshot lands at
`.../asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact1/b4_s6144/memory_snapshot.pickle`.
Accepted-baseline artifacts to diff against:
`outputs/expact_vs_gc_exp_b4s6144_drop000_20260613T074927Z/.../expact1/b4_s6144/{memory.md,memory_by_category.csv,source_profile.json}`.

---

## S1: Peak-Attribution Analyzer (Ground Truth)

### Scope

- New `scripts/testing/analyze_cuda_memory_snapshot.py` (torch-free).
- New `tests/testing/test_analyze_cuda_memory_snapshot.py` (torch-free).
- Edit `scripts/lf/run_lf_profiled_train.py::_start_memory_snapshot_recording`
  (`:1349`) to harden the capture call.

### Changes

1. Harden the capture so the trace is complete and carries Python frames, with a
   version-safe fallback to the current bare call:

```python
def _start_memory_snapshot_recording(enabled):
    if not enabled: return {"enabled": False}
    if not torch.cuda.is_available():
        return {"enabled": True, "record_started": False, "error": "cuda unavailable"}
    try:
        torch.cuda.memory._record_memory_history(
            max_entries=200_000, stacks="python", context="all")
    except TypeError:
        torch.cuda.memory._record_memory_history()   # older torch positional API
    except Exception as exc:
        return {"enabled": True, "record_started": False, "error": str(exc)}
    return {"enabled": True, "record_started": True}
```

2. Analyzer reconstructs the peak by replaying `device_traces` (the end-of-run
   `segments` are not the peak). Pure pickle/dict processing, no torch import:

```python
snap = pickle.load(open(args.snapshot, "rb"))          # {"segments", "device_traces"}
live = {}            # addr -> (size, frames)
cur = peak = 0; peak_live = None
for ev in snap["device_traces"][args.device]:
    a = ev["action"]
    if a == "alloc":
        live[ev["addr"]] = (ev["size"], ev.get("frames", []))
        cur += ev["size"]
        if cur > peak: peak, peak_live = cur, dict(live)
    elif a in ("free_completed", "free_requested"):
        sz_fr = live.pop(ev["addr"], None)
        if sz_fr and a == "free_completed": cur -= sz_fr[0]
# attribute peak_live blocks by deepest non-torch python frame
```

   - Aggregate `peak_live` bytes by owning frame: group by the first frame whose
     `filename` is under `asym_gemm/` or `scripts/`, falling back to the deepest
     frame; bucket `qwen3_moe.py`/`moe.py`/`exp_act_offload_*`/`frozen_linear.py`/
     `activation_offload.py` vs `*norm*`/`*attention*`/SDPA/loss/`lm_head`.
   - Classify each block param/grad/activation/workspace by frame function name.
   - Emit `--output-json` (peak bytes, per-bucket totals, top-N blocks with
     size+frame) and a markdown table; flag any block `>= --min-bytes`.
   - Handle `free_requested` vs `free_completed` (free at requested to avoid
     counting caching-allocator deferral), and blocks with empty `frames`
     (bucket `allocator/unframed`).

3. Test builds a synthetic `device_traces` (a few allocs/frees with fake
   frames), asserts the analyzer finds the known peak and attributes bytes to
   the expected bucket.

### Risks / Watch

- `_record_memory_history` kwargs differ across torch versions — the `TypeError`
  fallback covers it; watch that the fallback still yields `frames` (older API
  may omit Python frames → analyzer then buckets `allocator/unframed`; if so,
  rerun is needed with a torch that supports `stacks="python"`).
- `max_entries=200_000` could truncate a long multi-step trace and drop the
  peak. Mitigated by `MAX_STEPS=1`. Watch: if analyzer peak < `memory.md`
  `peak_allocated_hbm_bytes`, raise `max_entries` or add a per-step dump.
- Caching allocator reuses addresses; the replay keys by `addr` and frees on
  `free_completed`, which matches allocator semantics. Watch for `segment_alloc`/
  `segment_free` events — ignore for the live-block sum (they are pool-level).

### Validation (before S2)

```bash
# torch-free unit test (runs on this login node)
.venv/bin/python -m pytest -q tests/testing/test_analyze_cuda_memory_snapshot.py

# produce a real snapshot via the Shared Diagnostic LF Run above, then:
.venv/bin/python scripts/testing/analyze_cuda_memory_snapshot.py \
  --snapshot "<run_dir>/.../expact1/b4_s6144/memory_snapshot.pickle" \
  --device 0 --top 40 --min-bytes 268435456 \
  --output-json outputs/plan_peak_attrib.json
```

Pass when: analyzer reconstructed peak equals `memory.md`
`peak_allocated_hbm_bytes` (153,940,737,024 B) within allocator rounding; every
block `>= 256 MiB` is printed with its owning source frame; the bytes the
breakdown called `routed_experts saved_activations` are now traced to concrete
frames.

---

## S2: Expert-Block Timeline Probes (Non-Disturbing)

### Scope

- `asym_gemm/training/qwen3_moe.py`: `AsymQwen3Experts.forward` (`:2323`),
  `_forward_expert_activation_offload` (`:2220`),
  `_ActivationOffloadQwen3ExpertFunction.backward` (`:1011`).
- New tiny JSONL logger helper (in `qwen3_moe.py` or a small
  `asym_gemm/training/_diag_log.py`).

### Changes

1. Gate everything behind a new env `ASYM_EXPACT_PEAK_PROBE=1` (off by default).
2. Use **`torch.cuda.memory_allocated()`** deltas only — never
   `reset_peak_memory_stats()` mid-step (it would clobber the allocator peak the
   run reports). Record current-allocated at: experts-forward entry/exit,
   loss/compute-loss entry (via existing heartbeat point if reachable), and
   each Function `backward` entry/exit. Emit one JSONL row per probe:
   `{step, layer_seq, region, phase, allocated_bytes, t}`.

```python
def _probe(region, phase):
    if not _EXPACT_PROBE: return
    _diag_jsonl({"region": region, "phase": phase,
                 "allocated_bytes": int(torch.cuda.memory_allocated())})
```

3. Purpose is timeline, not per-region transient peaks: it tells us the step
   phase at which global allocated is maximal (end-of-forward vs loss vs a
   specific layer's backward). True per-region transient peaks come from the S1
   analyzer (group `peak_live` blocks by `qwen3_moe` frame/function).

### Risks / Watch

- `memory_allocated()` is current, not peak, so it gives steady-state growth,
  not in-region spikes. That is intentional here; do not substitute
  `max_memory_allocated`/`reset_peak` (global-peak corruption). Watch item:
  confirm the run's `peak_allocated_hbm_bytes` is unchanged with the probe on.
- `compute_loss` entry may not be reachable from `qwen3_moe.py`; if so, drop the
  loss probe and infer the loss phase from the S1 trace timestamps instead.

### Validation (before S3)

```bash
# real-dim micro path exercises the same Function; confirm JSONL emits
ASYM_EXPACT_PEAK_PROBE=1 PYTHONPATH="$PWD" .venv/bin/python \
  scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 24576 --top-k 8 --num-experts 128 --hidden-dim 2048 \
  --intermediate-dim 768 --rank 64 --warmup 1 --iters 1 \
  --output-json outputs/plan_probe_smoke.json
# then the Shared Diagnostic LF Run with ASYM_EXPACT_PEAK_PROBE=1 added
```

Pass when: JSONL has ordered per-layer/per-region allocated rows; the run's
reported `peak_allocated_hbm_bytes` matches the no-probe run (probe is
non-disturbing); the global-peak phase is identified.

---

## S3: ctx / Manager Live-HBM Audit

### Scope

- `asym_gemm/training/activation_offload.py`: `ActivationOffloadManager`
  (`:79`), `snapshot` (`:213`), stats `staged_bytes`/`_active_stage_bytes`.
- `asym_gemm/training/qwen3_moe.py`: Function forward exit (after `:1007`).

### Changes

1. Add a process-global weak registry of live managers and an aggregate, so we
   can sum HBM still staged across all 48 simultaneously-live layer `ctx`s:

```python
import weakref
_LIVE_MANAGERS = weakref.WeakSet()
# in __init__: _LIVE_MANAGERS.add(self)
@classmethod
def live_staged_bytes(cls):
    return sum(int(m.stats.staged_bytes) for m in _LIVE_MANAGERS)
```

2. At Function forward exit, gated by `ASYM_EXPACT_PEAK_PROBE`, assert/log the
   invariant: `manager.stats.staged_bytes == 0` (act_for_down_base released at
   `qwen3_moe.py:982`) and every `ctx.*_cpu.tensor.device.type == "cpu"`. Any
   violation is a real, in-scope expert-side leak — log tag, shape, bytes.
3. Emit `live_staged_bytes()` into the S2 JSONL at experts-forward entry so the
   accumulation (if any) across layers is visible at the global-peak phase.

### Risks / Watch

- The weakset must hold weakrefs only; never a strong ref (it would itself keep
  managers/ctx alive and create the leak we are testing for).
- `staged_bytes` is the Python-side accounting; async D2H/H2D in flight is not
  reflected. Acceptable — we are auditing Python-side retention, not transfer
  overlap. Watch item: if S1 shows HBM the manager accounting does not, the gap
  is async/stage-cache and needs a separate look.

### Validation (before S4)

```bash
ASYM_EXPACT_PEAK_PROBE=1 PYTHONPATH="$PWD" .venv/bin/python -m pytest -q \
  tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend
```

Pass when: per-layer `staged_bytes == 0` between layers and all ctx handles are
CPU (or the leaking tensor is named); aggregate `live_staged_bytes()` at the
global-peak phase is quantified and ~0.

---

## S4: Attribution Reconciliation & Fix in lf_trace.py

### Scope

- `asym_gemm/profiling/lf_trace.py`: `LFMemoryBreakdownProfiler` (`:663`), its
  `_saved_activation_bytes_at_peak` / `_activation_bytes_at_peak` /
  `_persistent_storage_keys` (`:688-699`), `_row_attributed_gpu_bytes` workspace
  residual (`:598-610`).

### Changes

1. Add a debug dump (gated by `ASYM_GEMM_LF_BREAKDOWN_DEBUG=1`) listing, for the
   selected peak row, every saved/live storage in the `routed_experts` bucket:
   `(ptr, bytes, shape, dtype, component, is_persistent)`. This makes the 24 GiB
   inspectable per-storage instead of as a bucket total.
2. Reconcile against S1: line up `routed_experts saved_activations +
   live_activation + workspace_residual` from the breakdown against the S1
   analyzer's per-frame total for expert frames. Then fix only what S1 proves
   wrong, most likely:
   - saved LoRA param storages not in `_persistent_storage_keys` → exclude them
     (params already counted in the persistent census `memory_by_category.csv`).
   - `_component_stack` leakage: a module's component left on the stack after its
     forward returns, so a later non-expert save is tagged `experts`. Fix push/
     pop in the module hooks.
   - `workspace_residual` (`peak - known`) over-assigned to the expert row when
     the expert row is merely the selected peak row (`_select_memory_breakdown_row`,
     `:620`). Constrain residual to frames S1 attributes to experts.
3. Re-run; confirm the corrected `routed_experts` total equals the S1 allocator
   truth for expert frames, and the sum across all components still equals peak.

### Risks / Watch

- This is the murkiest stage; treat S1 as ground truth and change the heuristic
  only to match it. Do not "fix" a number without an S1 frame that contradicts
  it.
- A fix that moves bytes out of `routed_experts` must move them into the correct
  component (norms/attention/residual), not into `other`. Watch: total must
  still reconcile to `peak_allocated_hbm_bytes`.

### Validation (before S5)

```bash
ASYM_GEMM_LF_BREAKDOWN_DEBUG=1  # add to the Shared Diagnostic LF Run env, rerun
# compare against S1:
.venv/bin/python scripts/testing/analyze_cuda_memory_snapshot.py \
  --snapshot "<run_dir>/.../expact1/b4_s6144/memory_snapshot.pickle" \
  --device 0 --bucket-by frame --output-json outputs/plan_peak_attrib.json
```

Pass when: the breakdown `routed_experts` peak total equals the S1 expert-frame
total within a stated tolerance; the per-storage debug dump shows no param or
non-expert storage left in the expert activation bucket; all components still
sum to peak.

---

## S5: Real-Dim Micro-Isolation & Verdict

### Scope

- `scripts/testing/profile_qwen3_activation_offload.py` (already reports
  `reset_peak_memory_stats`/`max_memory_allocated`, `:140-158`; here it is the
  isolated single-layer case, no model-level peak corruption concern).
- Append `## Findings` to this file.

### Changes

1. Run the harness at real Qwen3-30B expert dims to get the expert-only HBM
   floor and the per-layer transient with zero norm/attention/loss noise.
2. Synthesize: fill `L, G, W, A, M, F, R` from S1 (allocator truth), S2/S3
   (timeline + leak audit), S4 (corrected attribution), S5 (isolated floor).
3. For each nonzero `W`/`A` contributor, write a one-line removal hypothesis +
   the acceptance gate it must pass (lower the global LF peak or a reported
   expert-block peak, per the v2 rule). If `R == 0`, declare the expert
   activation side at its floor; the only remaining expert HBM is the kept LoRA
   params+grads.

### Risks / Watch

- The harness uses `FakeQwen3Experts` with `offload=True`; confirm it routes
  through `_ActivationOffloadQwen3ExpertFunction` (it does via `_make_model`,
  `:82`). Requires SM100 BF16 (`_require_sm100_bf16`, `:45`) → GPU node only.

### Validation

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/testing/profile_qwen3_activation_offload.py \
  --tokens 24576 --top-k 8 --num-experts 128 --hidden-dim 2048 \
  --intermediate-dim 768 --rank 64 --warmup 1 --iters 2 \
  --output-json outputs/plan_expact_isolated_b4s6144.json
```

Pass when: the isolated expert-only floor is a hard number that reconciles with
the S1 per-frame expert total and the S2 per-layer deltas; the `## Findings`
verdict block is filled with measured `L, G, W, A, M, F, R`; the `R`-enumeration
(possibly empty) exists with per-item acceptance gates.

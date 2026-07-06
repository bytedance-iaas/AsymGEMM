# panvme handoff — continue INSIDE the container (written 2026-07-06, prior session ran OUTSIDE by mistake)

## ✅ RESOLVED 2026-07-06 (inside-container session) — GATE PASSED, all numbers final
**step_ratio 1.0003 (211.35 s/it), fwd 1.0000 / bwd 1.0008, memory_drop 40.49 GiB (352.3→311.8),
median|max loss Δ 0.00085|0.0032, reads 914 GB/run fully hidden (fetch_wait 1.7 s/step),
quarantine_block 131 ms/run, trace period 1280.** Baseline `20260706T053034Z_*_container`, candidate
`20260706T101731Z_*_container_final` (earlier `_container`/`_container_pf12`/`_container_qd64` dirs =
the ×1.085 diagnosis chain). Unit gates 26/11/36/137 green inside (LF venv + repo .venv). Env findings:
all host workarounds unnecessary inside (numactl present, system CUDA 13.0.88, repo .venv healthy ⇒ no
ENV_DIR, no DS_SKIP_CUDA_CHECK); **HF_HOME inside container = /scratch_local/user_data/shutian/kevin/cache/huggingface**
(the handoff's kevinni path does not exist in-container). Two root causes fixed beyond the quarantine fix
(details: nvme_offload_impl.md Stage 7 bring-up notes 3–5): (a) profiler memory walks fetched every paged
blob via `HostWeight.tensor` — 87% of all NVMe reads; now `is_paged`-guarded metadata-only; (b) Belady
eviction rescans cost 6.4 s/step — now O(1) successor-gap clock. Prefetch/aio knobs stay at defaults
(12 GiB prefetch thrashes: evict-before-use; QD64/intra8 neutral; RAID0 does 25 GB/s, we consume ~4.3).
A leftover host-side gate run (PID tree 354174→356411, GPU 1, 322 GiB RSS) was SIGTERM'd before relaunch.
Remaining from step 6 (not run): MoE point q3.5-35b-a3b + Stage 8. `test_nvme_store.py` rule-7 test now
matches the `deepspeed` package exactly (editable-install finder shim false-positive).

## ⚠️ Context: why this handoff exists
The entire prior session (implementation + tests + e2e attempts 1–6) ran **directly on the host — the
container was never started**. The user killed the in-flight run (attempt 6, step 2/7, SIGTERM, GPU 1
drained). **Code and unit-test results are substrate-independent and stand. All e2e timings/RSS numbers
and every env workaround must be re-validated inside the container.**

## Mission (unchanged)
`asym_cpuadamwds_panvme` (agent/impls/nvme_offload_impl.md Stage 7 + §7.0): frozen base weights
(`HostWeight`) → NVMe once at startup, read back through a DeepSpeed-style trace-prefetching pinned
cache. Gate: vs `asym_cpuadamwds` baseline — RSS −40+ GiB, step ≤1.05×, median |Δloss| ≤0.02,
`--expect-nvme-role base_weight` (tool: `scripts/lf/compare_nvme_profiles.py --target base_weight_cpu`).

## Code state (DONE, all in working tree, uncommitted)
- **NEW `asym_gemm/training/base_weight_pager.py`** — the engine. DS-lineage: {ABSENT,INFLIGHT,RESIDENT}
  status machine; **step-boundary trace freeze** via `mark_step()` (DeepSpeed `reset_step` analog —
  occurrence-count freezing FAILS under GC-recompute: each weight is touched ~18×/step); byte-budgeted
  prefetch (default 2× largest blob); Belady eviction on the frozen trace; quarantine with **no hot-path
  event sync** (transient over-cap alloc = DS buffer_count slack, `ASYM_NVME_BASE_WEIGHT_OVERSHOOT_BYTES`
  default 8 GiB, floor 2×largest at finalize; hard sync only at the ceiling — the earlier per-eviction
  `event.synchronize()` cost a measured **33 s/step**); trace mismatch
  ⇒ `_disabled` = miss-driven sync fallback (correct, slow). `get_base_weight_pager()` module accessor.
- **`asym_gemm/training/host_weight.py`** — `_pager/_pager_key` fields; `.weight/.tensor/grouped_nt_tensor`
  → `pager.touch()` when paged; `shape/dtype/device/in_features/out_features/grad/pinned_cpu_bytes` →
  metadata when paged (never fetch); `pin_memory()` self when paged. `_pager is None` ⇒ byte-identical.
- **`asym_gemm/integrations/lf.py`** — `_install_base_weight_pager(model)` called at the very end of
  `apply_lf_asym_lora` (after `_release_replaced_module_memory()`); gated on store role `base_weight`;
  forces eager qwen3-moefg gate_up split first; registers bf16 `AsymFrozenLinear`+`AsymGroupedFrozenLinear`
  only (embeds/norms/non-bf16 excluded); `model._asym_base_weight_pager = pager`.
- **`asym_gemm/training/nvme_store.py`** — added `shutdown()` read-drain + atexit (a crash with in-flight
  preads otherwise SIGABRTs in the C++ handle dtor, masking the real error).
- **`scripts/lf/run_lf_profiled_train.py`** — `mark_step()` wired inside the wrapped optimizer-step
  (success path, right after `original_step`), try/except-guarded.
- **NEW `tests/training/test_base_weight_pager.py`** — 11 tests (roundtrip 2D/3D bit-exact, homes freed,
  metadata-never-fetches, step-boundary freeze period=2k−2 under touch-dedupe, exact-Belady vs brute
  force, jitter tolerance, beyond-window disable fallback correctness, quarantine gating w/o hot-path
  sync, held≤cache, alias/small skip).
- Stage 1/2 (store, token plumbing, compare tool) were already landed before this session.

## Test state (all run OUTSIDE container — rerun inside to confirm; logic should hold)
`test_base_weight_pager.py` 11/11 · `test_nvme_store.py` 26/26 · `test_cpu_resident_frozen_base.py`
36/36 UNMODIFIED · `test_lf_qwen3_asym_backend.py` 137/137 UNMODIFIED.

## E2E results so far (q3-32b s20000 b8 ga1, flagship policy `none|false|true|false|false|false`,
recomp-off-full-fg ligerloss1, WARMUP 2 / MAX_STEPS 5, GPU 1) — **numbers from OUTSIDE the container**
| | baseline | panvme no-prefetch (attempt 4, trace disabled) | panvme prefetch (attempt 5) |
|---|---|---|---|
| ram_g (peak RSS) | 352.2 | 304.9 (−47.3 ✓) | 305.5 (−46.7 ✓) |
| step_s (median measured) | 251.4 | 287.6 (×1.14) | 279.3 (×1.11) ✗ gate |
| loss vs baseline | — | ≤0.002 ✓ | ≤0.002 ✓ |
| pager | — | disabled (old heuristic) | **frozen ✓ marks=7, period=8015, misses_after_freeze=110, quarantine_block_ms=230932 ← the ×1.11 culprit** |
- Artifacts: `profiling_nvme/stage7_panvme/20260706T024318Z_q3-32b_s20000b8/` (baseline + no-prefetch pair)
  and `.../20260706T035839Z_q3-32b_s20000b8_prefetch/` (attempt 5). Attempt 6 (killed) dir `...044109Z..._prefetch2` is garbage — delete.
- **Attempt 6 = attempt 5 + the no-hot-path-sync fix, was never completed.** Expected to remove the
  33 s/step stall → inside ×1.05. THIS IS THE NEXT MEASUREMENT.

## What the outside-container mistake does / does not invalidate
- **Persisted into the shared NFS venv (visible in container, probably fine, verify):** `liger-kernel`
  pip-installed; cu13 toolkit repinned (`nvidia-cuda-nvcc/nvvm/crt==13.0.88`, `nvidia-cuda-cccl==13.0.*`);
  unversioned `lib*.so` symlinks + `lib64→lib` + triton `cuobjdump` symlink in
  `LlamaFactory/.venv/.../nvidia/cu13`. NEVER use `.venv/bin/pip` (shebang → twin tree); always
  `$LF_PY -m pip`.
- **Launch-time workarounds that may be UNNECESSARY (or wrong) inside the container — re-evaluate each:**
  `NUMACTL_ENABLE=0` (host lacked numactl; container may have it → HC2 wants it ON),
  `CUDA_HOME=<venv cu13>` + PATH/LD_LIBRARY_PATH (container may ship a real CUDA 13.0),
  `DS_SKIP_CUDA_CHECK=1`, `ENV_DIR=<LF venv>` (host's `AsymGEMM/.venv` had a broken 3.11 relink over
  3.12 site-packages; the container may have the proper interpreter → try WITHOUT `ENV_DIR` first).
- **Always still required:** `ASYM_NVME_PATH=/scratch_local/user_data/shutian/kevin/cache/asym_nvme`
  (the script default `/scratch_local/asym_nvme` is root-owned), `HF_HOME=/scratch_local/user_data/kevinni/hf_cache`
  (Qwen3-32B downloaded there), heavy runs sequential, `kill -TERM` only.

## Next steps, in order
1. **Inside the container**, probe: `numactl --version`; `nvcc --version` (system?); LF venv python
   imports (`torch, deepspeed, datasets, liger_kernel`); `AsyncIOBuilder().is_compatible()` with the
   `.aioenv` exports (run_lf_lora_sft.sh:34-48 sets them; add CFLAGS/LDFLAGS if probing manually);
   `AsymGEMM/.venv/bin/python -c "import datasets"` (decides whether `ENV_DIR` override is needed).
2. Rerun unit gates: `test_nvme_store.py`, `test_base_weight_pager.py`, then the two UNMODIFIED suites.
3. Rerun the paired e2e FRESH inside the container (both rows, one session dir):
   ```
   STAMP=$(date -u +%Y%m%dT%H%M%SZ)
   HF_HOME=/scratch_local/user_data/kevinni/hf_cache \
   ASYM_NVME_PATH=/scratch_local/user_data/shutian/kevin/cache/asym_nvme \
   GPU_POOL=<free gpu> OUTPUT_ROOT=profiling_nvme/stage7_panvme/${STAMP}_q3-32b_s20000b8_container \
   PROFILERS=source PLOT=false PREPARE_DATASETS=true WARMUP_STEPS=2 MAX_STEPS=5 \
   RUNS='q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 20000|8|1 ; none|false|true|false|false|false || q3-32b|1 ; asym_cpuadamwds_panvme|recomp-off-full-fg|ligerloss1 ; 20000|8|1 ; none|false|true|false|false|false' \
   scripts/lf/profile_lora_lf_test_source.sh --overwrite true
   ```
   (+ the container-appropriate subset of CUDA_HOME/ENV_DIR/NUMACTL_ENABLE per step 1.)
4. Gate:
   ```
   $LF_PY scripts/lf/compare_nvme_profiles.py --baseline <asym_cpuadamwds dir> --candidate <..._panvme dir> \
     --target base_weight_cpu --expect-nvme-role base_weight --min-memory-drop-gib 40 \
     --max-step-ratio 1.05 --max-forward-ratio 1.05 --max-backward-ratio 1.05 --max-loss-delta 0.02
   ```
   Health in candidate `source_profile.json → asym_nvme.base_weight_pager`: want `trace_frozen=true,
   trace_disabled=false, step_marks=7, misses_after_freeze≈O(100), quarantine_block_ms≈0,
   over_budget_allocs small, fetch_wait_ms/step ≪ step`.
5. If step ratio still >1.05, levers in order: raise `ASYM_NVME_BASE_WEIGHT_PREFETCH_BYTES` (e.g. 8–16 GiB;
   hides more of the ~920 GiB/step read stream), raise `ASYM_NVME_BASE_WEIGHT_CACHE_BYTES` toward
   ~19 GiB (stay under the 40 GiB-drop constraint: drop ≈ 59.6 − cache − pool-slack), then consider
   per-entry event gating (record reuse events at last-use instead of eviction time).
6. Then: MoE point (q3.5-35b-a3b, exercises the eager gate_up split; −100 GiB), doc bring-up note in
   nvme_offload_impl.md Stage 7, memory update with final numbers.

## Concurrency note
Another agent works in this same tree on config renaming / ker-ceil profile-script tokens (`-ceil<N>`
suffix; asym dirs carry `-ker<XYZ>-ceil<NNN>`). Do not fight concurrent script edits; re-read the two
profile scripts before launching. GPU 3 is occupied by a non-panvme job — leave it alone.

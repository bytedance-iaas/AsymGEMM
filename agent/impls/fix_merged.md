# fix_merged.md — post-merge fix list + pre-push validation plan (2026-07-15)

Scope: the SFT-39 (qwen3.5 memory) x SFT (memory/latency) merge. `main_kevin` =
`389db00` (merge `fd2eb57` + scoreboard `4c2b4f2` + probe finding `389db00`), **not
pushed**. Backup `main_kevin_qwen35` = `5a81335`, pushed. Merge completeness is proven by
git, not inspection: `git log sft/main_kevin ^main_kevin` is empty and
`git merge-base --is-ancestor sft/main_kevin main_kevin` is true; the only delta over
their tip is the qwen3.5 line (+133 code lines / 7 files).

Full merge record: `agent/handoffs/merge.md` (Merge 2). Post-merge scoreboard and the
fg101 probe verdict: `agent/impls/archive/fix_qwen3.5.md` §9b-post-merge / §9c.

Only nontrivial items are recorded. Anything fixable by reading was fixed, not filed.

---

## 0. Already fixed by code reasoning (no test needed — do not re-open)

| # | Fix | Where | Why it was wrong |
|---|---|---|---|
| F1 | Fold `offload()`'s byte accounting into `record_cpu_ready` | `activation_offload.py` | `offload()` records `num_offloads`/`offloaded_bytes`/`offload_bytes_by_tag`; `empty_cpu()` records only `num_cpu_allocs`. The blocked/chunked paths fill via `empty_cpu`+`record_cpu_ready`, so **every blocked write was invisible**. Demonstrated on the 45k run: `moe.act` = 110 GiB — one of the three largest MoE tags — would have read **0**. Also fixes `dense_mlp_finegrained`'s identical gap. |
| F2 | Declare `qwen3_moe_finegrained_fused_home_released` | `frozen_linear.py` | Set dynamically at `qwen3_moe.py:2568`, but an undeclared dataclass field never reaches `profile.json` — RELEASE_FUSED_HOME shipped with an unreadable engagement counter. |
| F3 | Correct the async-restage comment | `activation_offload.py` | Claimed "the copy overlaps preceding compute". It does not: `side.wait_stream(compute)` + `compute.wait_event(done)` with nothing enqueued between = same device ordering as issuing on the compute stream. Only real gain is host run-ahead. |
| F4 | Forward `ASYM_SAVED_TENSOR_ASYNC_UNPACK` as `ASYM_GEMM_LF_CONFIG_*` | `profile_lora_lf_test_both.sh` | Knob worked (bare-env inheritance) but `_env_config()` only harvests the prefixed form ⇒ **no D4 artifact recorded whether the flag was on**. |
| F5 | Add `pin_fallback_calls` to the lin-attn snapshot | `linear_attention_activation_offload.py` | A failed pin silently returns a **pageable** buffer ⇒ `ready_event=None` ⇒ `_unpack` takes the host-blocking branch — exactly what the async-unpack flag exists to remove, with zero signal. |
| F6 | Correct the D4 receipt | `fix_throughput.md` (D4 CORRECTION) | See B3. |

F3/F4/F5/F6 are documentation/telemetry only — byte-inert on the compute path. F1/F2
change only counters. None alter numerics.

---

## 1. BLOCKERS — do not push until resolved

### B1. Cross-Model Non-Regression Matrix was NOT run (required by our own rule)

`archive/fix_finegrained_qwen3.5_moe.md` §Cross-Model Non-Regression Matrix is **"required
for ANY shared-code edit"**. This merge edits shared code: `record_cpu_ready` lives in
`activation_offload.py` and is called by **both** `qwen3_moe_finegrained.py` **and
`dense_mlp_finegrained.py`**. Only the qwen3.5 rows were run. Outstanding:

- [ ] **qwen3-30b spot runs** (canonical stack): `s20000` ctl band loss 1.775 ± 0.05,
      grad_norm ~0.49; `s80000` ker101 band `step_H` 80,521 MiB ± ~3%, loss 1.689 ± 0.05.
      (grad_norm there is the Known Systemic Issue — record, don't gate.)
- [ ] **dense qwen3.5-27b s30000** — the *only* gate that exercises
      `dense_mlp_finegrained`, i.e. the other consumer of the F1 change. Expect `step_H`
      71,228.31 MiB, loss 1.10–1.15, grad_norm ~0.17,
      `dense_mlp_finegrained_offload_wrapped=64`, moefg stays 0.
- [ ] **llama4-scout** — Stage-1 gate only if a shared file changed semantics for
      non-matched models. F1 changes counters only, so likely waivable — decide explicitly
      rather than by omission.

Done already: Stage-1 probe qwen3.5 default + `--zero-b` (both PASS); pytest
`test_lf_qwen35_asym_backend.py` + `test_lf_qwen3_asym_backend.py` → **171 passed**;
qwen3.5 smoke / 45k / 80k (§9b-post-merge).

### B2. The G1 claim rests on a pre-merge B

§9b-post-merge reports G1 MET (45k **0.473×**, 80k **0.557×** vs the 0.80× bar) using B's
**recorded** 72.4 / 103.0 from 2026-07-14. B was not re-measured post-merge. Defensible —
B is `superoffload_mem|unsloth-off` and never enters the asym fg path this merge changed —
but the headline "80k goes 0.89 FAIL → 0.56 PASS" is a ratio against a number from a
different tree.

- [ ] Either run B at 45k/80k post-merge, or downgrade the claim to "A improved 37% at
      80k; G1 ratio taken against the 2026-07-14 B".

### B3. `gc_async_offload.py` is unreachable but D4 says it was measured

93 lines, **zero external callers** (`git log -S "async_save_on_cpu" --all` returns only
the commit that added it). D4 recorded `ASYM_SAVED_TENSOR_ASYNC_UNPACK=1` as covering
"decoder/lin-attn unpack + pinned GC root + **async save_on_cpu**" with "engagement
verified: root saved pinned" — but no flag reaches the `async_save_on_cpu` half; the
pinned-root observation came from the **pre-existing** `gc_boundary_offload.py:69-70,90`
path. The file's docstring premise ("stock LF saves the root via an UNPINNED
`hidden.to('cpu', non_blocking=True)`") does not describe this repo. The D4 row is now
annotated (F6); the code decision stands:

- [ ] **Wire it or delete it.** Do not ship a recorded result pointing at dead code. If
      wired it needs its own A/B — and note the D4 null is structurally explained (F3), so
      expect no win without restructuring the stream logic.

---

## 2. Real defects found by reading — need isolated tests, cannot be settled by argument

### V1. `wait_cpu_ready` is device-only, but the CPU-left path does a **host** memcpy

`wait_cpu_ready` (`activation_offload.py`) does only `current_stream().wait_event()` and is
explicitly commented "never block the host". `cpu_left.py:188`
(`padded[...].copy_(x_cpu[...])`) is a **host-side** memcpy out of that pinned buffer, so
the device-side wait does not order it. `27dde72` (2026-07-07, **predates 1896825** — not
merge-caused) moved the accidental host sync (`cpu_left.py:129,139`: blocking `.to("cpu")`
+ `.item()`) under `if cached is None:`, so the 2nd/3rd CPU-left calls per forward now skip
it (all three share memo key `(128, R, cuda:0)`, `cpu_left.py:125-127`).

Status: the guard is genuinely absent; **unproven to fire**. NOT the fg101 cause (§3). Not
fixable by reasoning: the fix adds a host sync, which costs exactly the latency `27dde72`
removed ⇒ needs measurement, not argument.

- [ ] Isolated test: force a slow D2H (large R, contended H2D) and assert the CPU-left
      padded buffer matches a synchronously-copied reference, bitwise, over N iterations.
- [ ] If it fires: host-side wait in the CPU-left path (or event-synchronize the handle
      before the host read), then re-measure the D-table rows `27dde72` won.

### V2. The CPU buffer pool can recycle a buffer with a D2H still in flight

`release_cpu` (`activation_offload.py:429,436`) returns buffers to the module-global
`_CPU_BUFFER_POOL` after only a **device-side** wait. A buffer can therefore enter the pool
while its D2H is outstanding, and the next `_alloc_cpu` hands it to an unrelated consumer.
Same class as V1, wider blast radius — the pool is global and shared across the attention /
moe / dense paths.

- [ ] Isolated test: poison-fill recycled buffers in `_alloc_cpu`; assert no consumer reads
      poison.
- [ ] Interacts with V3 — a plausible mechanism for the probe's cross-case contamination.

### V3. The Stage-1 probe contaminates itself (`--qwen3 --tokens 655360`)

`fg101` **alone**: rel_fro **0.006273** (PASS). Canonical 5-case order: **0.170918**
(BROKEN). Same commit, shapes, inputs — 27×, and the only variable is whether
`plain`/`fg000` ran first. The gate is lying and **ker101 is not implicated** (§9c). The
contamination's own root cause is **not identified**.

- [ ] Bisect the poisoner: run `plain→fg101` and `fg000→fg101` separately. Whichever
      reproduces 0.17 names the culprit.
- [ ] Suspects, in order: the module-global `_CPU_BUFFER_POOL` (V2);
      `_asym_kernel_meta_memo` (`qwen3_moe_routed_gemm.py:89,108`) — keyed **only** by
      `str(device)`, ignoring `experts`, though it hangs off the `offsets` tensor so it is
      exploitable only if the same `offsets` object is reused with different `experts`
      (**unverified**); `_asym_pad_memo` (`frozen_linear.py:643+`); `os.environ` mutated
      between cases against module-level env caches.
- [ ] Fix the probe: fresh engine + cleared pool per `run_case`, or one case per process.
      Until then the ≤4096-token shape sets are the trustworthy ones.
- [ ] NB: whatever this is, it regressed **before** `1896825` —
      `archive/fix_finegrained_qwen3.5_moe.md:74-78` records this exact command passing all
      five cases at ≤0.8% on 2026-07-03.

---

## 3. Dead ends — refuted with evidence, do NOT re-investigate

- **int32 overflow in the routed kernel** (R*I = 4.03e9 > int32 max at T=655360). REFUTED:
  the kernel never forms a 32-bit linear index — `act` is addressed by TMA **row
  coordinate** (max 5.24e6, fits `uint32_t`) with 64-bit descriptor strides, and the
  scatter is
  `atomicAdd(&route_scatter_out[static_cast<uint64_t>(token_row) * stride + col], ...)`
  (`include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh:1043-1045`). `token_indices` is int64.
- **Pool bucketing/narrowing at the probe's R.** REFUTED narrowly: R = 5,242,880 = exactly
  80 × 65536, so `_alloc_cpu` allocates at exact shape and `_narrow` returns the base
  untouched; `cpu_left.py:186` also `zero_()`s. Does **not** refute pool reuse hazards in
  general — that is V2.
- **"fg101 is a host/device race" / "ker101 is broken".** REFUTED: a race cannot reproduce
  rel_fro to six decimals across three commits and three processes; and fg000 shares the
  identical code path up to the CPU-left call, diverging only at
  `qwen3_moe_finegrained.py:988-1008`, so a CPU-left defect would break fg000 too — it
  passes. Confirmed by V3's isolation result.

---

## 4. Open / deferred — decide, but not push blockers

- **D1. The blocked forward never engages in the flagship qwen3.5 row.** Gated on
  `da_gpu and lora_a_fwd_gpu`; the flagship row is `ker101` + **`loraafwdcpu`**. What runs
  is the sibling `act_chunk` branch (gated on `not lora_a_fwd_gpu`) + silu-bwd chunking —
  confirmed by `stage_rows_calls` = 2880 (45k) / 4800 (80k). ⇒ the −37% at 80k is the
  **chunking family**, and "`_fg_elementwise_blocks` beats H3b" is **not** established
  head-to-head. To settle: A/B with `ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=1`.
- **D2. nograd keeps a full-R `act`** `[R, I]` bf16 (~1.4 GiB @30B-a3b/120k/b8) that H3b's
  blocked down+scatter avoided. Live under GC (the outer forward is no-grad). Deferred at
  merge as unvalidated code on a validated base. Re-port if the fwd peak needs it.
- **D3. dscatter coverage gap.** Their blocked path needs `da_gpu`, which
  `DOWN_SCATTER_BLOCK_EXPERTS>0` forces off ⇒ no fg fwd blocking under dscatter, a case H3b
  covered. Only matters if a dscatter row returns to the dial.
- **D4. Long-seq (120k+) unvalidated.** Validation stopped at 80k, but `runs.log` shows 120k
  rows in flight pre-merge. V1, V2 and the new `pin_fallback_calls` are all most likely to
  bite under pinned pressure at 120k/131k — that is where this merge is least tested.
- **D5. `_record_attn_hbm_gemm` increments outside the chunk loop**
  (`attention_activation_offload.py:720,798`) ⇒ `ASYMM_ATTN_ACT_LORA_CHUNK=0` and `=1` are
  **indistinguishable** in stats; the flag cannot be engagement-verified. Left alone
  deliberately: the counter's "one per logical GEMM" semantics are shared with existing
  rows, so changing it would silently rebase historical comparisons. Add a *separate* chunk
  counter if that flag is ever dialed.
- **D6. `ASYMM_ATTN_ACT_LORA_CHUNK=1` silently no-ops if `ASYMM_FG_ELEMENTWISE_CHUNK_MB=0`**
  — undocumented cross-knob dependency (default 1024 MB, so it works out of the box).
- **D7. Zero tests** reference `ASYM_SAVED_TENSOR_ASYNC_UNPACK`, `_async_unpack_enabled`,
  `_add_matmul_rows_`, or `ASYMM_ATTN_ACT_LORA_CHUNK`.
- **D8. smoke grad_norm** 0.2136–0.2671 vs the recorded 0.22–0.25 band — marginally wider at
  both ends on 4 steps. Almost certainly noise; confirm it is not systematic if another
  smoke is run.

---

## 5. Verified clean (reviewed, no bug — recorded so they are not re-reviewed)

`_fg_elementwise_blocks` per-block offset rebasing; `_HBMKeepManager` guards (it lacks
`record_cpu_ready`/`stage_rows`/`empty_cpu` by design, and every call site is
`hasattr`-gated or unreachable when `keep_acts_hbm=1`); the `_stage_cache` /
`_release_chunk_stages` aliasing question; `fg_chunk_rows`'s `max(8192, rows)` floor; the
blocked-path bf16 vs `input_dtype` asymmetry (latent only if a non-bf16 config ever runs —
flagship is bf16); the lin-attn async unpack's use-after-free question (the pinned source is
protected by PyTorch's `CachingHostAllocator_recordEvent`, **not** by the
`_asym_restage_keepalive` attribute, which is redundant and not load-bearing — do not rely
on it if the pattern is reused); `_add_matmul_rows_` chunk tiling and dtype chain; and F1
cannot double-count (each `record_cpu_ready` fires once per handle; `offload()` does not
call it).

RELEASE_FUSED_HOME + the `unsupported_reasons` pin-check relaxation are confirmed a
**matched pair**: the release is proven to fire (running plain after an fg forward now
raises `grouped weight must be 3D, got shape (0,)`), and without the relaxation
`is_pinned()` on the released empty tensor is False, so `unsupported_reasons` would raise on
fg forward **#2** — 160 forwards at 80k prove it holds. Note the release makes the plain
path unusable after any fg forward **within the same process**; that is loud
(`NotImplementedError`, never a silent fallback) and real training picks one path.

# Memory mode (asym_cpuadamwds + recomp-off-full-fg): memory floor + latency recovery

Working log for the 2026-07-11/12 effort: make the asym memory mode as memory-lean as
possible (capacity angle: longer max seq), then recover latency while staying within
~1–2 GiB of the memory floor. All runs: LF LoRA-SFT, b8, r64 a16 drop0, ligerloss1,
1 GPU (GB200 185 GiB HBM, ~1.7 TB host RAM, host watchdog floor 35 GiB),
`scripts/lf/profile_lora_lf_test_both.sh`, dataset asym_long_sft_smoke.

## FINAL STATE (2026-07-12): one config wins both axes

`asym_cpuadamwds + recomp-off-full-fg-ker101` + (now-default) staged down-dx +
1024 MB fg chunks. 30B@120k b8: **97.2 alloc / 109.0 reserved / 536.6 s/it**
(pre-effort: 118 / 180.4 / 587). 32B@52k: **88.0 / 103.0 / 557.7** (was
95.5/148.6/1024 @49k). 30B@160k: **127.6 / 138.5 / 779** (was impossible, then
130.4/182.6/1255). Full history + probe evidence below.

## TL;DR state (mid-effort snapshot, 2026-07-12 morning)

| Model/seq | Config | GPU alloc | GPU reserved | CPU RSS | s/it | Notes |
|---|---|---|---|---|---|---|
| 30B-A3B @120k | superoffload_mem | 139.8 | 140.7 | 618 | 494 | comparator |
| 30B-A3B @120k | asym PRE-fix (ker101) | 118.0 | 180.4 | 695 | 587 | reserved ~G-OOM |
| 30B-A3B @120k | asym POST ker101 | 123.5 | 172.5 | 574 | 537 | balanced point |
| 30B-A3B @120k | **asym POST ker111** | **96.3** | **117.4** | 575 | 816 | capacity point |
| 30B-A3B @160k | asym POST ker111 | 130.4 | 182.6 | 822 | 1255 | fits; PRE-fix impossible; loss=0 artifact (smoke data <131k, labels fully masked) — memory numbers valid |
| 32B @52k | superoffload_mem | 115.2 | 126.5 | 644 | 418 | comparator |
| 32B @49k | asym PRE-fix | 95.5 | 148.6 | 699 | 1024 | 52–53k = host-RAM ceiling for BOTH backends (55k watchdog-OOM) |
| 32B @52k | **asym POST** | **88.0** | **130.9** | 638 | **622** | wins memory AND 1.7x faster than own baseline |

Loss parity verified on every comparable run (e.g. 30B@120k: superoffload
1.6492/1.6293/1.6977 vs asym post-fix 1.6554/1.6287/1.7016).

## Diagnosis (what the peak actually was)

Artifacts used: `memory.md`, `memory_actual_peak_breakdown.csv`, `memory_breakdown.jsonl`
(per-step/phase peaks + `peak_growth_bytes_at_peak` + live-activation details), `lat.md`,
`throughput_breakdown.md`, train.log it/s bars, `asym_execution_stats` in profile.json.

- The asym GPU peak was **transient full-width workspace**, not saved activations.
  PRE-fix 30B@120k: peak growth at peak = routed_experts 45.6, norms 22.1(→33.1 @120k),
  attention 9.8, embed 2.4 GiB; live (saved) activations were only ~2.5 GiB.
- Specific full-width `[R×inter]` / `[R×hidden]` offenders (R = tokens×topk = 7.68M rows
  @120k b8; inter=768, hidden=2048 → 11.8 / 31.5 GiB two-byte tensors per instance):
  1. fg silu·mul forward: staged full gate AND up + `.contiguous()` act copy (3 instances).
  2. fg silu backward: same shape × 3 (dup, dgate, grad_act) + full re-stages of gate/up.
  3. LoRA-B delta adds (`S @ B^T` materialized full then added).
  4. LoRA dx (`d_S @ A`) materialized full.
  5. MoE: gathered X `[R,2048]` (31.5 GiB), routed grad_2d `[R,2048]`, scattered deltas.
  6. RMSNorm fp32 upcast chain: `x.float()`, normed fp32, out fp32 saved by autograd →
     ~3 full-width fp32 tensors per call; "norms" growth 33–41 GiB @120k. LF wraps norms
     in `AsymFrozenRMSNorm` (asym_gemm/training/offload.py), so the fix owns the module.
- Reserved-vs-allocated gap (PRE: 62.4 GiB @120k) tracks the giant-buffer churn:
  fresh `torch.empty` stage buffers + `record_stream` deferral + expandable segments.
  Shrinking transients shrank the gap mechanically (62 → 21 GiB on ker111).

## Implementation (all default-ON via `ASYMM_FG_ELEMENTWISE_CHUNK_MB`, 0 = legacy)

Env: `ASYMM_FG_ELEMENTWISE_CHUNK_MB` (default 2048 MB) — byte budget per staged chunk.
Forwarded by the driver as `ASYM_GEMM_LF_CONFIG_ASYMM_FG_ELEMENTWISE_CHUNK_MB`.

- `asym_gemm/training/activation_offload.py`
  - `fg_elementwise_chunk_bytes()` / `fg_chunk_rows(total_rows, row_width, elt)` —
    chunk sizing (min 8192 rows; 0 = don't chunk).
  - `ActivationOffloadManager.record_cpu_ready(handle)` — register D2H-complete event
    for handles filled by chunked row writes (offload() does this internally; direct
    `tensor[rows].copy_` writers must call it once after the last chunk).
- `asym_gemm/training/dense_mlp_finegrained.py`
  - `_add_lora_b_delta_` (chunked delta add), `_release_chunk_stages`.
  - Forward: chunked silu·mul via `stage_rows` writing straight to pinned act rows;
    LoRA-B deltas chunk-added into gate/up/out.
  - Backward: row-chunked silu backward (gate/up staged per chunk; dgate/dup written
    straight to pinned CPU rows); grad_act stays full-width (it is the down-dx GEMM
    output — do NOT chunk asym dx, see Trap below).
  - No-grad forward (`_finegrained_dense_mlp_no_grad_gpu_forward`): chunked delta adds.
- `asym_gemm/training/qwen3_moe_finegrained.py`
  - `_fg_elementwise_blocks(offsets, experts, rows, width)` — expert-aligned row blocks
    sized to the budget; every grouped-GEMM segment stays identical to full-width
    (bit-exact LoRA grads), each expert still touched exactly once.
  - `_add_grouped_lora_b_delta_`, `_apply_lora_dx_` (scatter or packed dest).
  - Grad-enabled forward (flagship env FG_LORA_A_FWD_GPU=1 + FG_DA_GPU=1): blocked
    gate/up/act pipeline — per expert block: gather packed_block from hidden, base
    gate/up GEMM, GPU LoRA-A, chunk-added delta, silu·mul, write act rows to pinned CPU,
    GPU down-LoRA-A. Never materializes gathered X [R,2048] nor full gate/up/act GPU
    tensors. Full-width fallback preserved (fwd_blocks empty / non-flagship env).
  - Backward: blocked down-LoRA (per block: gather grad rows, dS, dB accumulate, LoRA
    dx into grad_act rows; dS accumulated full [R,r] then ONE CPU-right dA call);
    chunked silu backward compatible with KEEP_DGRADS_HBM (writes to HBM dests) and
    with the CPU path (writes to pinned rows); gate/up dx LoRA contributions via
    `_apply_lora_dx_`.
  - No-grad forward: blocked gate/up/act + blocked down-delta scatter.
- `asym_gemm/training/offload.py`
  - `_RMSNormChunkedFunction` + hook in `AsymFrozenRMSNorm.forward` (non-gated path):
    frozen scale ⇒ saves ONLY bf16 input (a reference to storage the caller already
    holds) + `[rows,1]` fp32 rstd; fwd/bwd math in fp32 row chunks. Forward bit-exact
    vs legacy; dx within bf16 rounding (~1e-3 rel). Norms growth 33–41 → 7.7 GiB.

### THE TRAP (measured, do not regress)

Per-call host cost is wildly asymmetric:
- asym *forward* grouped GEMM ≈ 55 ms/call (720 calls ≈ 40 s of the fwd).
- asym *dx* calls and CPU-right `_lora_a_grad_cpu` calls ≈ **2 s/call** host-blocking.

v2 (reverted) blocked the down backward including per-block `_base_dx` + per-block
CPU-right dA on ker101: backward went 474 s → 1845 s (3.7×). Rule: **block only
GPU-only work** (grouped_expert_lora, scatter/index ops, elementwise, stage_rows H2D);
keep exactly ONE asym-dx and ONE CPU-right call per layer on accumulated full tensors.
ker111's `down_dx_gather` route kernel replaces the down asym-dx entirely (no grad_2d),
which is why ker111 gets the low peak; its +39% step time is the route-kernel cost, not
the chunking.

### Verification

- Dense fg end-to-end (backend=torch): output, dx, all 6 LoRA grads **bit-exact**
  chunked vs legacy (scratchpad `test_fg_chunking.py`).
- MoE fg Function + nograd (faked grouped bases, flagship env, 1 MB budget →
  many blocks): all 6 LoRA grads **bit-exact**; out/dx/nograd within bf16
  scatter-atomics noise (`test_moe_fg_blocked.py`).
- `_RMSNormChunkedFunction`: fwd exact; bwd ~1e-3 rel.
- E2E loss parity on every run in the TL;DR table.

## Operating points (30B@120k, POST-fix)

| kerXYZ | route kernels | alloc | reserved | s/it | use |
|---|---|---|---|---|---|
| ker101 | fwd_scatter + gateup_dx_scatter | 123.5 | 172.5 | 537 | balanced (faster than pre-fix 587) |
| ker111 | + down_dx_gather (no grad_2d) | 96.3 | 117.4 | 816 | capacity |

ker111 residual peak @120k (memory_breakdown.jsonl step3):
- after_forward pk 59.7 (live: attention 15.7 [q 7.3 + hiddens], embed 3.7, norms 3.7)
- after_backward pk 96.0; growth_at_peak: attention 38.3, routed_experts 46.0,
  norms 7.7, embed 3.7. Live at peak only ~7.4 → still transient-dominated:
  - routed_experts ≈ dgate+dup HBM-kept (2×11.8, KEEP_DGRADS_HBM=1) + grad_act 11.8
    + stage buffers.
  - attention ≈ FA backward (q/k/v restage + dq/dkv + workspace) — q alone
    [8,120k,4096]=7.3 GiB ×(q,dq)+kv+attn-grad chain.

## Probe log (Phase 1: memory floor)

Method: RUN_NAME=probe_* dirs under profiling_both/asym_long_sft_smoke__lora__lf__bf16/;
PROFILERS=source, MAX_STEPS=2 WARMUP_STEPS=1 (peaks stabilize at step 2; s/it from bar).
Baseline to beat: ker111 @120k = 96.3 alloc / 117.4 reserved / 816 s/it.

NOTE: PROFILERS=source probes land under `profiling/` (not `profiling_both/`);
RUN_NAME becomes `<name>__b8_s120000_ga1_drop000`. Probe s/it averages 3 steps
(incl. warmup) vs 4 for the full runs — treat <±5% latency deltas as noise.

| Probe | Flags vs baseline | alloc | reserved | resv-unalloc | CPU | s/it | verdict |
|---|---|---|---|---|---|---|---|
| (baseline ker111) | — | 96.3 | 117.4 | 21.1 | 575 | 816 | |
| probe_kd0 | KEEP_DGRADS_HBM=0 | 96.5 | 113.9 | 17.4 | 604 | 779 | REJECT: no alloc win (peak is elsewhere), +29 GiB host RAM (pinned dgrads) |
| probe_c1024 | CHUNK_MB=1024 | 99.3 | **104.2** | **4.8** | 572 | 774 | **WIN: fragmentation gap collapses 21→5; reserved −13.2, latency ≤ baseline** |
| probe_kd0_c1024 | both | 99.4 | 106.2 | 6.8 | 604 | 787 | kd0 doesn't stack; reject |

Findings batch 1:
- The reserved-unallocated gap is chunk-size-driven: 2048 MB chunks → 21 GiB gap;
  1024 MB → 4.8 GiB. Reserved (the OOM-relevant number) is minimized by SMALLER
  chunks even though alloc ticks up ~3 GiB (block-boundary co-location).
- keep-dgrads OFF is strictly worse: alloc unchanged → the backward peak window is
  NOT the dgate/dup residency; it's attention-bwd + down-phase co-location.
- All three probes ≈775–790 s/it vs 816 baseline — treat as ≤5% noise band.

Batch 2 (attn chunking + c512; baseline now probe_c1024 = 99.3/104.2/774):

| Probe | alloc | reserved | gap | s/it | verdict |
|---|---|---|---|---|---|
| probe_c1024_attn | 99.3 | 114.8 | 15.5 | 771 | REJECT: no peak effect; small interleaved allocs WORSEN fragmentation +10.6 |
| probe_c512 | 98.8 | 107.8 | 9.0 | 791 | chunk sweet spot is ~1024, not smaller |
| probe_c512_attn | 97.9 | 108.8 | 10.8 | 788 | ditto |

REVISED peak theory (batch-2 evidence): kd0 removed 23.6 GiB from the MoE gate/up
window with ZERO peak change, and attn LoRA chunking (removing the projection-GEMM
transients) also changed nothing → the global backward peak is the **attention
backward window itself**: FA bwd tensor set (q,k,v,out,dout,dq,dk,dv ≈ 33 GiB at
120k b8) + projection dx transients + baseline residents. This is close to inherent
for full-sequence FA backward — alloc floor ≈97–99 reached; only fragmentation was
left (captured by c1024: gap 4.8). ASYMM_ATTN_ACT_LORA_CHUNK stays default OFF
(the unconditional in-place delta/dx adds stay — value-identical, strictly fewer
allocations).

Phase-1 floor: **ker111 + CHUNK_MB=1024 → 99.3 alloc / 104.2 reserved / ~775 s/it**
(vs phase start 96.3/117.4/816; vs pre-fix 118/180.4/587).

Historical peak-instant anatomy (superseded by the revision above —
kept for the reasoning trail; probe_c1024, step2 after_backward pk 98.4):
growth_at_peak = attention 41.6 + routed_experts 45.9 + norms 5.7 + embed 3.7.
Both components alive at ONE instant because unsloth-GC recompute-backward holds the
layer's recomputed attention tensors (q 7.3 / kv / attn_out 7.3 / o_out 3.7 + LoRA
transients) while the MoE backward window peaks (grad_act 11.8 + dgate/dup HBM 23.6 +
stages). kd0's null alloc-result is consistent: removing dgrads residency in the
gate/up phases doesn't touch this instant. Remaining levers on the instant:
(a) attention LoRA chunk-adds (batch 2), (b) sdparecomp policy bit — recompute SDPA
in the inner backward instead of holding it (batch 3), (c) attention-recompute
tensors to HBM-keep vs re-offload semantics (attn_act_hbm_* stats say the recompute
keeps them in HBM by design — D3-era choice at healthy margin).

Condition note: mid-batch, attention_activation_offload gained (a) unconditional
in-place LoRA delta/dx adds (value-identical, one fewer full-width buffer per
projection in fwd AND bwd), (b) `ASYMM_ATTN_ACT_LORA_CHUNK` (default OFF) that
chunks those adds via fg_chunk_rows. probe_kd0 ran PRE-edit; probe_c1024 and
probe_kd0_c1024 include (a) but not (b). Batch 2 will A/B (b) explicitly.
Unit test: chunked vs full `_add_matmul_rows_` exact-equal.

Next-lever candidates (by expected value):
1. Attention backward transients (38 GiB): inspect attention_activation_offload staging;
   q/k/v restage granularity, fp32 dq accumulation, sdparecomp flag semantics.
2. Chunk MB 512 if 1024 is free; watch launch-count latency.
3. reserved-unalloc (21 GiB): stage-buffer reuse across tags/layers (single grow-only
   workspace per shape family instead of per-tag cache with drop_cache=True churn).

## Phase 2 (latency at the floor)

Timing anatomy @120k ker111 (timing_by_stage.csv, 4-step run): fwd 91 s,
bwd 723 s, optimizer ~2 s. ker101 bwd ≈ 484 s → the down_dx_gather route kernel
costs ≈ +250 s/step (≈1.3 s per layer-backward). keep_dgrads=1 stays (kd0 gave
no memory and +29 GiB host RSS).

Batch 3 results:

| Probe | alloc | reserved | gap | s/it | verdict |
|---|---|---|---|---|---|
| probe_ker101_c1024 | 124.3 | 143.8 | 19.5 | **494.8** | **latency point**: fastest asym config ever measured here; reserved ≈ superoffload (140.7); grad_2d window keeps alloc high |
| probe_c1024_sdpar | 99.7 | 108.6 | 8.9 | 770 | REJECT + theory CONFIRMED: peak sits inside attention bwd (out needed there regardless) |

Two blessed operating points after phase 1+2 probing (30B@120k b8):
- **Capacity**: ker111 + CHUNK_MB=1024 → 99.3 alloc / **104.2 reserved** / ~775 s/it.
- **Latency**: ker101 + CHUNK_MB=1024 → 124.3 alloc / 143.8 reserved / ~495 s/it.
(Both dominate the pre-fix 118/180.4/587; pick per job by seq headroom.)

Capacity-point latency tax: bwd 723 s (ker111) vs 484 s (ker101) → down_dx_gather
≈ +1.24 s per layer-backward. (ASYM_EP_DXTIME probe failed silently — the env var
is not in the driver's forwarded run_env list; use a standalone bench instead.)

**Microbench (scratchpad bench_down_dx.py, real shapes R=7.68M, hidden 2048,
inter 768, E=128, random token routing):**
- down_dx_gather route kernel (CPU weight): **5542 ms/call**
- HBM-resident per-expert torch mm floor:   **71.8 ms**
- The packed down weight is only ~400 MB → H2D stage ≈ 1 ms over C2C.
→ The tax is the kernel's token-space gather access pattern, NOT weight streaming.
An HBM-staged resident mm is ~77× faster.

**Fix implemented: `ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1`** (default off;
driver-forwarded). In the blocked down backward (down_dx_gather=0 route codes),
stage the packed down weight to HBM once per layer backward (~400 MB) and compute
the base dx per expert segment with resident cuBLAS mms directly into grad_act
rows — no grad_2d [R,hidden], no route gather kernel, no full asym dx. Memory =
ker111-class (blocked, +0.4 GB transient weight), speed = resident-mm class.
Verified in the e2e harness: all 6 LoRA grads bit-exact vs legacy; dx within bf16
noise.

**probe_dxstaged_c1024 (ker101 codes + DOWN_DX_STAGED=1 + CHUNK_MB=1024, 30B@120k):
alloc 96.7 / reserved 104.0 / 489.6 s/it — SUPERSEDES both operating points**
(best memory: 104.0 ≤ ker111-c1024's 104.2; best latency: 489.6 < ker101-c1024's
494.8 < pre-fix 587 < ker111-c1024's 775). route_gather_calls=0 (kernel bypassed),
losses in family (1.6568/1.6316).

The memory-mode recommendation is now ONE config:
`recomp-off-full-fg-ker101 + ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 +
ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024`.

## FINAL VALIDATION (2026-07-12, full table conditions) — all `ok`, loss parity

| Run | alloc | reserved | CPU | s/it | vs pre-fix / vs phase-start best |
|---|---|---|---|---|---|
| 30B@120k final (both profilers, w1+s3) | 97.2 | **109.0** | 574 | **536.6** (bwd 442 s) | pre-fix 118/180.4/587; ker111-c2048 96.3/117.4/816 |
| 32B@52k dense + c1024 | 88.0 | **103.0** | 644 | **557.7** | was 88.0/130.9/622 → reserved −27.9, 10% faster |
| 30B@160k new config | 127.6 | **138.5** | 792 | **779** | old ker111: 130.4/182.6/1255 → reserved −44, 38% faster |

120k losses 1.6556/1.6335/1.7031 (superoffload comparator 1.6492/1.6293/1.6977);
52k losses 0.9988/1.0308/1.1939 (comparator 0.9960/1.0277/1.1915). 160k losses
nonzero this rebuild (2.44/2.49/2.27) — dataset rebuild produced valid labels;
memory numbers comparable either way. 160k reserved 138.5 ⇒ ~46 GiB headroom:
~200k seq @ b8 now plausible on 185 GiB (untested).

**DEFAULTS FLIPPED (2026-07-12):**
- `ASYMM_FG_ELEMENTWISE_CHUNK_MB` default 2048 → **1024** (activation_offload.py
  + the RMSNorm chunk sizing in offload.py).
- `ASYMM_QWEN3_MOE_DOWN_DX_STAGED` default off → **ON** (qwen3_moe_finegrained.py;
  self-gates to down_dx_gather=0 codes; set 0 to restore grad_2d + asym-dx).
Both test suites re-run green after the flips (LoRA grads bit-exact; dx bf16-noise).

**Memory-mode recommendation:** `asym_cpuadamwds + recomp-off-full-fg-ker101`
with defaults (staged down-dx + 1024 MB chunks) — nothing else to set.
ker111 (down_dx_gather) is now strictly dominated: same memory, +47% step time;
keep only as a fallback if the staged path misbehaves.

2026-07-13 postscripts:
- Reader bug fixed: the driver forwards unset knobs as EMPTY strings; the staged-dx
  reader treated "" as off → default-ON silently disabled in runs that didn't set the
  env explicitly. Readers must treat "" as default. (qwen3_moe_finegrained.py fixed;
  unit-tested.)
- Rare hang observed (once in ~15 runs @120k): step-2 forward frozen 7 h in
  `build_contiguous_route_metadata` bincount with GPU 100%; config had completed 3×
  before → system-level. Ladder runner now wraps rungs in timeout+retry
  (`scripts/lf/run_dial_ladder.sh`).
- Latency↔memory dial work continues in `agent/impls/scheduler_v2.md` (formulation + ladder).

## History / provenance

- Baselines preserved: `qwen3-30b-a3b__...s120000...__prechunk_baseline` (PRE-fix),
  `__blockedv1` (fwd-blocking only), `__v2` (down-bwd blocking incl. the 3.7× latency
  regression — kept as the what-not-to-do receipt). 32B: `...s52000...__chunkv1`.
- 55k 32B attempts (both backends) died to the host-mem watchdog (soft host OOM at
  ~33 GiB free), NOT GPU OOM: 32B capacity is host-RAM-bound on this box; next capacity
  lever there is CPU-side (actnvme / -ceil budget), not HBM.
- 160k 30B needs a real >131k-token dataset for loss-valid runs (smoke source maxes
  ~131k; tokenizer warning at build time; labels fully masked → loss 0.0).

---

## Post-merge dial verification (2026-07-13, main_kevin merge 36af646)

After merging EP work (SFT-38) into main_kevin, re-ran the 120k dial (source
profiler, w1+m4) to check the memory/latency modes vs the banked records. `_C`
rebuilt against venv torch 2.12 (scripts/lf/rebuild_asymgemm.sh).

| config | s/it mine→rec | alloc HBM mine→rec | reserved mine→rec | CPU RSS mine→rec | loss |
|---|---|---|---|---|---|
| memory (L0, ker101)          | 483.8 → 483.6 | 104.6 → 97.3  | 116.2 → 100.3 | 614 → 572 | 1.647 |
| latency (L3, ker000+staged+keepacts) | 352.6 → 352.5 | 126.7 → 118.0 | 193.3 → 180.0 | 555 → 517 | 1.644 |
| REF superoffload_mem         | 474.1 → ~475  | 150.1 → 139.8 | 151.0 → 140.7 | 662 → 617 | 1.642 |

Findings:
- **Speed matches records exactly** for all three (Δs/it < 0.3 s). Relationship
  intact: memory = lower HBM + slower; latency = higher HBM + 27% faster; both
  dominate superoffload. Loss parity across all three.
- Every config sits ~+7–10 GiB HBM / +40–45 GB CPU RSS above its own record by a
  **uniform** amount — INCLUDING superoffload_mem, a fully independent code path
  (unsloth GC, roots-HBM, no asym fg kernels / pad path / chunking). Because the
  non-asym baseline shifted by the same amount, the offset is a system-wide
  measurement/env baseline drift vs the 2026-07-12 bank, NOT a merge regression in
  the asym memory/latency machinery.
- torch_fallback=0 on both asym modes; L3 shows torch_forward via the staged
  native-mm dispatch (expected, ASYM_GEMM_DISPATCH=staged).

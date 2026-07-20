# fix_asym — close the asym-latency backward tax so it beats superoffload-unsloth-ohbm0

Owner-facing, evidence-first change list. Gap source of truth:
`agent/impls/s04-p1-dgx-02-c12/test_throughput_results.md` (PHASE B). Prior fix ledger:
`agent/impls/fix_throughput.md` §3 (C0–C5). This doc: WHAT is measured, WHAT is refuted,
WHAT to instrument/change next (file:function:line), and the A/B that proves each step.
AUDITED 2026-07-18: v1's #1 fix (C1b "backward GEMMs streaming") was REFUTED by run
counters — see §2.0. Do not resurrect it.

================================================================================
## 0. THE MEASURED PROBLEM (Phase B, c12) — receipts, not guesses
================================================================================
Asym's best latency config (KA = ASYM_GEMM_DISPATCH=staged + ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1
[+AU, +ohbm16 — both measured null]) vs superoffload unsloth-ohbm0:

| point        | sup ohbm0 tok/s (MFU) | asym best tok/s (MFU) | margin |
|--------------|-----------------------|-----------------------|--------|
| q3-32b 128k  | 1110 (22.1%) b2       | 957  (19.1%) b3       | -14%   |
| q3-32b 160k  | 942  (21.4%) b2       | 857  (19.5%) b3       | -9%    |
| q3-32b 192k  | 816  (20.8%) b1       | 731  (18.7%) b2       | -10%   |
| llama 128k   | 792  (32.6%) b2       | 677  (27.9%) b3       | -15%   |
| llama 192k   | 601  (31.5%) b1       | 553  (29.0%) b2       | -8%    |

Per-token phase split (q3-32b 128k: asym KA b3 vs sup b2; profile.json step.rows / tokens):
| phase    | asym us/tok | sup us/tok | Δ |
|----------|-------------|------------|---|
| forward  | 168.1 | 166.2 | **+1.9 (PARITY)** |
| backward | 875.8 | 732.9 | **+142.8 (THE ENTIRE GAP)** |
| optimizer| 4.1   | 0.3   | +3.8 (known-small; C5 scope) |

Flag A/Bs (all at q3-32b 128k, measured): KA ESSENTIAL (without it: 753, -21%); AU null
(957 vs 956); ohbm16 null (957); LEAN regression. Flags NOT yet swept (low expectation,
listed so "exhausted" is precise): ASYMM_FG_ELEMENTWISE_CHUNK_MB (fixed 1024),
ASYMM_ATTN_ACT_LORA_CHUNK (unset), dense LoRA-A-on-GPU variant (see S1 — this one is REAL).

================================================================================
## 1. WIN CONDITION (corrected arithmetic — v1 overstated)
================================================================================
Per-token-limited regime (≥192k tok/step): tok/s ≈ 1/c_pertoken; batch is NOT a lever
(only the ~10-15% c_fix amortizes; asym already runs the larger batch and loses).
- Closing the FULL +142.8 us/tok backward tax ⇒ asym ≈ 903 us/tok ⇒ ~1104 tok/s @128k
  ⇒ **PARITY with 1110, not a clear win**. (v1's "1153" assumed asym holds its 24k-plateau
  MFU at 128k — impossible; attention share is 56% at 128k and grows to 66% @192k. The bar
  is the MEASURED sup number; "would-be saturated" extrapolations are retracted.)
- The win margin must come from removing work sup does NOT pay — candidates in §2 (e.g.
  S1's D2H+host-GEMM feed has no sup counterpart; sup's own act-offload is its analog and
  is already priced into 733).
- Sup's deficit FLATTENS with seq (~21% dense / ~32% llama through 192k; b1 re-amortizes):
  waiting for longer seqs buys nothing; the crossover requires asym-side fixes.
- TARGET: asym backward ≤ 733 us/tok @q3-32b 128k (from 875.8). Stretch < 700 = clear win.

================================================================================
## 2. EVIDENCE-RANKED WORK LIST
================================================================================
### 2.0 REFUTED — do not pursue (with the receipts)
- ~~"Backward GEMMs run on the ⅓-eff streamed kernel; route to staged" (v1 C1b)~~
  REFUTED by asym_execution_stats of the KA/LEAN/ohbm16 runs (archived profile.json):
  `asym_forward_calls=0, asym_dx_calls=0, torch_forward_calls=4480, torch_dx_calls=3520`
  — with ASYM_GEMM_DISPATCH=staged, `_dispatch_nt` (frozen_linear.py:1236) reroutes EVERY
  bf16 GEMM (fwd AND dX; transpose_b included) to the native path. The streamed kernel is
  completely unused in latency mode. There is no GEMM-engine tax.
- ~~W-panel reuse as a standalone fix (v1 C1c ~20-30 us/tok)~~ WRONG SIZING: weight H2D
  per step ≈ 1600 calls × ~85 MB ≈ 136 GB ≈ 0.5 s/step ≈ 1.4 us/tok, and largely
  overlapped. Micro-opt at best; fold into whatever GEMM-adjacent work lands, don't
  schedule it alone.
- ~~C2 "stage-skip in keep-acts" as free copies~~ PARTLY MOOT: KA swaps in `_HBMKeepManager`
  (dense_mlp_finegrained.py:246→qwen3_moe_finegrained._HBMKeepManager), so stage()/offload()
  are already HBM-local handle ops, not copies. Residual churn is real but not the naive
  "remove the D2H/H2D pairs" win v1 claimed.
- Batch size as tok/s lever; AU; ohbm pinning — measured null/non-levers (§0).

### PROVENANCE (audited vs CURRENT tree, 2026-07-18)
The run receipts are NOT stale: asym_gemm/ working tree is clean at 66b9be7 (07-16 16:55),
which predates every Phase B run; the install is EDITABLE (`pip -e`), so all .py behavior in
those runs came live from this exact tree; only `_C` is build-stamped (rebuilt 07-17 04:32,
after the last commit). Every §2 claim below was additionally re-read in the current source.

### 2.1 OPEN SUSPECTS for the +142.8 us/tok (ranked by receipt strength)
**S1' — 335 GB/step D2H with only 13 GB/step read back = mostly WASTED offload traffic,
and it is NOT the dense MLP.** Receipts (KA b3 128k profile + current source):
- activation_offload totals: d2h 1674 GB / 5 steps, h2d 13 GB total → ~96% of offloaded
  bytes are never staged back.
- The dense-MLP path is CLEAN under KA in the current tree: outer fwd is the no-grad pure-GPU
  body (`_finegrained_dense_mlp_no_grad_gpu_forward`, dense_mlp_finegrained.py:569 — GPU
  LoRA-A at :595/:607/:618, explains fwd parity); the GC-recompute grad-forward's LoRA-A
  ALSO runs on GPU under KA (`_cpu_left_lora_a`:894-896 "keep-acts-HBM: plain GPU matmul");
  `_HBMKeepManager.offload()` keeps the GPU tensor (qwen3_moe_finegrained.py:200) → no MLP D2H.
  ⇒ v2's "host LoRA-A GEMMs" theory is DEAD for KA mode (the cpu_left_lora_a_calls counter
  counts calls whose is_cuda branch ran on GPU).
- ⇒ The 335 GB/step emitter is the DECODER/ATTENTION saved-tensor + GC-boundary offload
  machinery (decoder_activation_offload.py pack path [AU], attention_activation_offload.py,
  gc_async_offload.py / decoder_checkpoint.py) — NOT covered by the dense-MLP KA flag. The
  13 GB read-back says backward consumes HBM-resident aliases while the host copies rot:
  offload emission that should be gated off when the resident path is active. Costs: D2H
  stream occupancy (≥1.3 s/step), pin-buffer alloc + host memcpy work, event ordering.
LOCALIZED (per-tag receipt, archived KA run): **q_proj.U = 252 GB/step + o_proj.U = 81 GB/step
= 333 of the 335 GB/step; ZERO U-tag H2D** (only the tiny S low-rank stashes, 0.6 GB/step,
round-trip). The attention act-offload wrapper D2H-offloads the projection inputs (U) even
though backward reads the HBM-resident copies (loraafwdhbm path; attn_act_hbm_calls=5120) —
pure-waste emission, 2 tags × 64 layers × 5 steps = 640 calls.
FIX: in the attention wrapper (exp_act_offload_lora.py / attention_activation_offload.py),
skip the U-offload when the HBM-resident path serves backward. Receipt: d2h 335→~1 GB/step.
⚠ HONEST SIZING: 333 GB/step ≈ only ~1-2 s/step of link time (≤5 us/tok) even if unoverlapped
— this fix is hygiene + C2C-link decontention (it shares the link with C0 grad-D2H), NOT the
143. The 55 s/step gap is still UNLOCALIZED in time terms after all counter analysis ⇒ S0
(nsys) is genuinely the required next step; no more zero-GPU inference can size it.
**S2 — fg wrapper churn (v1 C2, resized).** 128 fg-forward invocations/step (64 outer +
64 GC-recompute — receipt: dense_mlp_finegrained_forward_calls=128/step) × per-call
Python/handle/cast/chunk-loop overhead + gpu_silu_bwd chunking (64/step). Sup's analog
measured 3.6 s vs asym 12.5 s bwd wrapper wall (MoE@32k; dense unmeasured — nsys sizes it).
Diet: M=0 guards, fewer per-chunk kernel launches, cast fusion (`.to(bf16).contiguous()`
copies), reuse staged handles across gate/up/down within a layer.
**S3 — GC-boundary offload machinery** (gc_async_offload.py / decoder_checkpoint.py):
boundary-x D2H+H2D per layer per step; ~500 GB/step class if boundary inputs round-trip.
Part of the same nsys attribution.

### S0 — DONE (2026-07-18). ATTRIBUTION RESULT — this supersedes all prior sizing:
nsys pair @128k (asym KA b3 vs sup b2), per-token buckets over 2 measured steps:
| bucket | asym | sup | Δ us/tok |
|---|---|---|---|
| attention kernels | 632.3 | 636.6 | -4 (PARITY — excluded, as predicted) |
| GEMM kernels | 149.9 | 126.9 | +23 |
| elementwise/other | 132.8 | 110.1 | +23 (C2 wrapper, matches estimate) |
| memcpy H2D | 72.6 | 6.8 | **+66** |
| memcpy D2H | 58.8 | 3.5 | **+55** |
| memcpy D2D | 10.4 | 1.5 | +9 |
Wall math: asym busy 1058 vs wall 1046 → copies ~SERIAL with compute (only ~12 us/tok
overlapped). D2H rate ≈ 333 GB/step / 22.6 s/step ≈ 15 GB/s = PAGEABLE-class, not pinned
C2C ⇒ the offload staging is likely pin-fallback-degraded. (v2.1's "≤5 us/tok hygiene"
sizing assumed full-rate async — WRONG; at pageable+serial rates the memcpy bucket is the
MAIN EVENT.)
FINAL FIX RANKING:
1. **S-mem (~120 us/tok): (a) kill never-read U-offloads (q_proj.U/o_proj.U, 333 GB/step,
   zero read-back); (b) verify/fix pinned staging (_PIN_FALLBACK_CALLS telemetry,
   decoder_activation_offload.py:88 — if >0 fix the allocator); (c) side-stream + overlap
   remaining copies.** Recovers more than the 145 needed ALONE.
2. C2 elementwise diet (+23 buffer). 3. GEMM +23 (b3 shapes/route-LoRA) — tertiary.
Recoverable ≈ 166 us/tok vs 145 needed ⇒ win margin exists.

### S0 (protocol, for reference — already executed)
A single nsys timeline of q3-32b 128k b3 KA (3 steps, kernel+memcpy+NVTX; the driver
supports PROFILERS=nsys) attributing backward wall into:
{GEMM kernels | elementwise/cast kernels | memcpy D2H | memcpy H2D | idle/gaps-waiting-host}.
Compare the same for sup 128k b2. The bucket that holds the ~55 s/step (=143 us/tok × 384k)
picks S1 vs S2 vs S3. Everything after S0 is a targeted fix with a sized expectation,
validated per §3. (nsys at 128k is heavy but 3 steps is tractable; fall back to 64k b3 —
▼6% regime — only if nsys chokes.)

### vs RECOMP — can fixed-asym + larger batch beat it? (objective assessment, 2026-07-18)
STRUCTURAL FACT (code-level): recomp = plain LF GC + resident GEMMs — no offload machinery
at all. A FULLY-fixed asym-KA converges to nearly the same computation (GC recompute +
native GEMMs + GPU LoRA + HBM acts) ⇒ the fix's CEILING vs recomp is per-token PARITY,
plus only the ~2-5% c_fix batch-amortization sliver from asym's larger batch. Recomp's
MFU is FLAT to its b1 wall (dense 28-29% through 96k; llama 39-40% except the pocket)
because b1×long-seq already exceeds its knee.
Verdict by model:
- DENSE q3-32b: NO realistic post-fix win while both fit. No under-saturation pocket
  exists (dense knee ~6k tokens; rc b1 = 96k+ tokens/step, flat ~28%). Fixed-asym best
  case ≈ parity + sliver ≈ +0-5%, inside noise. Win remains DNF-only past rc's b1 wall
  (~173k; asym 731@192k BANKED). ✅ HOLE CLOSED (measured 2026-07-18): dense recomp 128k b1
  = 1101 tok/s, 21.9% MFU — ≈ unsloth's 1110/22.1%, NOT ~1380. Recomp (zero offload
  machinery) shows the SAME 96k→128k MFU step-down as unsloth ⇒ the dense long-seq
  collapse is a shared attention/regime cost, not offload overhead (strengthens §1
  calibration fact (a)). One bar at 128k: ~1105±5. Asym's 957 trails BOTH by the same
  ~145 us/tok — one fix, two baselines beaten.
- LLAMA: MARGINAL YES in the 52-70k b1 pocket only (rc ▼2-9%, 1277-1404 tok/s): fixed-asym
  at b3-b4 ≈ high-30s% MFU ≈ 1350-1400 tok/s ⇒ +5-10% — REAL but narrow, and conditional
  on the fix reaching full sup-parity. Elsewhere rc is saturated (96k b1 = 40.3%) until
  its wall; ≥112k is DNF (BANKED: 677/553).
- Uncertainty: if S0 shows part of the 143 us/tok is structural (unremovable wrapper cost),
  even the pocket win evaporates. Do not promise recomp throughput wins beyond DNF until
  S0 + one pocket A/B confirm.

### Scope notes
- MoE (q3-30b) explicitly OUT OF SCOPE here (c14 partition). Same suspects apply via
  ker101/`qwen3_moe_finegrained.py`; plus `ASYMM_QWEN3_MOE_ROUTE_LORA=1` A/B (2.27 s/step
  token-space LoRA scatter, fix_throughput C2).
- You CANNOT dodge the fg wrapper by a non-fg dense config: keep-acts (ESSENTIAL, +280
  us/tok) is implemented INSIDE the fg wrapper. Any "bypass fg" idea forfeits KA — reject.
- Attention fwd/bwd kernels are identical in both systems (LF flash auto) — excluded.
- C3 sync-unpack sites (decoder_activation_offload.py:273, linear_attention:291): unreachable
  under KA unless `_PIN_FALLBACK_CALLS` (…:88) > 0 — check it in each new run's artifacts;
  if nonzero, fix the pinned allocator, not the flag.
- Host-RSS risk: KA at b3 128k uses 484-919 GB RSS; at 160k+ b3 approaches the host
  watchdog. If a validation run is watchdog-killed (status 143 + HOST_OOM_EVIDENCE), drop
  batch, not the fix.

================================================================================
## 3. VALIDATION PROTOCOL (per change — mandatory)
================================================================================
1. Rebuild: `bash scripts/lf/rebuild_asymgemm.sh` (container, venv torch — NEVER system).
2. A/B at identical (model,seq,B): pre-fix vs post-fix build (flag-gated where possible):
   `env ASYMM_FG_ELEMENTWISE_CHUNK_MB=1024 ASYMM_QWEN3_MOE_DOWN_DX_STAGED=1 \
        ASYM_GEMM_DISPATCH=staged ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 [fix flag] \
    bash scripts/lf/tp_probe.sh q3-32b tputasK0 \
      "asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm0|ligerloss1" 128000 3`
3. Parse: `python3 /workspace/AsymGEMM-SFT/.parse_any.py "tputask0_q3-32b__b3_s128000_ga1_drop000"`.
4. Decompose: `python3 /workspace/AsymGEMM-SFT/.phase_cmp.py <asym-dir> "tput_q3-32b__b2_s128000_ga1_drop000" 384000 256000`
   → recovered us/tok must land in BACKWARD. Also re-read asym_execution_stats + act-offload
   d2h/h2d totals to confirm the mechanism (e.g. S1 ⇒ d2h collapses from 335 GB/step).
5. Steady timing: PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 MAX_SAMPLES=1024 (mid w1+m4).
6. Correctness: step-1 loss matches pre-fix within 1e-2 bf16; numerics changes rejected.
7. Ceiling guard after any memory-relevant change: re-run ceiling search; max-seq within 1k.
8. Archive runs to profiling_results/profiling_tp_$(hostname)/ post-campaign.

## 4. SUCCESS CRITERIA
- PRIMARY: asym ≥ sup-ohbm0 tok/s at q3-32b 128k (>1110) and llama 128k (>792), both fitting,
  no thrash, at asym's B_max.
- Mechanism receipt: backward ≤ 733 us/tok @q3-32b 128k AND the S-suspect's counter moved
  (e.g. d2h/step collapsed).
- Regression guards: unsloth-OFF wins hold (+34-39%); recomp-DNF wins hold; capacity
  ceilings within 1k; memory-mode byte-identical when latency flags off.

## 5a. PHASE CLOSE-OUT (2026-07-18, user ruling: fix-then-fallback)
FINAL LADDERS: q3-32b @128k b2: 957→**1067** (−3.9% vs sup 1110; was −14%). q3-30b @480k b1:
948→**975** (−1.5% vs sup 990; was −4.2%). F-B result: +3 (fwd −3.0 us/tok cache hits on
outer-pass panels; resv +4.5). Five flag-gated code changes shipped (attn-KA, async-pack ×2
modules, fused-addmm, F-A packed-X, F-B panel-LRU) — all default-off, parity-gated.
RULING: remaining margin is covered by the SCHEDULER FALLBACK (scheduler_v2.md §8 emits
sup wherever it is argmax — baseline preserved by construction). The fix loop STOPS here;
remaining named items (D2H overlap residual +5.3, dX index ~+3-5, M-prefetch true-prefetch)
stay speculative until a future session re-opens them. Scheduler regime boundaries (q3-30b,
measured): asym ≲130k (wins) · sup-fallback 160-600k (−1.5..−2% asym) · asym ≥640k
(parity→exclusive). Every number above has a run receipt in profiling_tp_s04-p1-dgx-02-c14/.

## 5. STATUS LEDGER
- [DONE] C0 async-grad hooks (2026-07-09, fix_throughput.md).
- [REFUTED 2026-07-18] v1-C1b streamed-backward-GEMM theory; v1-C1c standalone panel reuse;
  naive C2 stage-skip. Receipts in §2.0 — do not re-derive.
- [S0 DONE 2026-07-18] attribution: memcpy +120 us/tok (serial, pageable-class) + elementwise
  +23 + GEMM +23; attention parity. [NEXT = S-mem]: (a) U-offload kill, (b) pin-fallback
  check/fix, (c) copy overlap — then re-run the §3 A/B at 128k b3 (target ≥1110 tok/s).
- [S-mem(a) IMPLEMENTED 2026-07-18, c14] `ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1` (default off) in
  attention_activation_offload.py: keep-HBM branch in _AsymActivationOffloadLoRALinearFunction
  fwd+bwd — skips the U/S offloads AND the backward stage-backs (both memcpy buckets die
  together: mechanism refined vs §2.1 — U is not never-read; dA re-staged it via a RAW
  `.to()` invisible to the U-tag h2d counter, which is why the counter said zero. The
  round-trip, not the emission, is the waste). LoRA-A fwd = native GPU GEMM on the resident
  source. Peak cost ≈ one layer's qkv+o (~8 GB @128k b3, LIFO). .py-only (editable install
  — no rebuild). TOY PARITY: y/grad_x/grad_A/grad_B all max|Δ|=0.0 vs flag-off (torch
  backend, staged semantics). Pending: §3 A/B tputask0 (pre-fix baseline, running) vs
  tputask1 (flag on) @q3-32b 128k b3.
- [S-mem(a) VALIDATED 2026-07-18, c14 A/B @q3-32b 128k b3] tputask0 (baseline, reproduces
  c12 receipts exactly: 957 tok/s, bwd 875.6 us/tok, d2h 1558.6 GiB/5st) vs tputask1
  (ASYMM_ATTN_ACT_KEEP_ACTS_HBM=1): **1016 tok/s (+6.2%), bwd 814.6 us/tok (−61),
  fwd byte-identical 167.7, act-offload d2h/h2d counters → ZERO, loss parity ≤0.002,
  resv +15.6 GiB (151.9, fits at 82%)**. Recovered 61 of the ~143; remaining ≈82 us/tok.
  Pin-fallback counters: 0 in both runs → S-mem(b) allocator theory MOOT for the attention
  path; the S0 "pageable-class" rate was the raw `.to()` stages (now dead). NEXT: nsys on
  the FIXED build (tputask1n, running) to re-bucket the remaining 82 (candidates: C2
  elementwise +23, GEMM +23, residual GC-boundary/decoder memcpy ~35).
- [S0' RE-ATTRIBUTION on fixed build, 2026-07-18 c14, tputask1n nsys, attention-anchored
  scaling] GEMM now AT SUP PARITY (~128 vs 126.9 — the +23 died with S-mem(a): it was the
  raw-stage + padded-GEMM shapes). Elementwise +24 (C2, unchanged). **Remaining memcpy
  ≈ +90 us/tok = D2H 1492.7 GiB/step + H2D 1628.8 GiB/step of NON-attention-wrapper
  traffic (act-offload counters read zero) ⇒ the unsloth-GC save_on_cpu=true machinery
  (recomp-off stages force it; sup-unsloth runs it FALSE). This is S3, and it dwarfs C2.**
  Knob exists: ASYM_GC_SAVE_ON_CPU_OVERRIDE=false (profile script :3329). A/B = tputask2
  (running). Memory risk: recompute-saved acts stay HBM (layer-scoped LIFO — sup's own
  regime); resv was 151.9/185 with headroom; fallback = drop to b2 if OOM.
- [S3 step-1 VALIDATED 2026-07-18, tputask2 = +ASYM_GC_SAVE_ON_CPU_OVERRIDE=false @128k b3]
  **1050 tok/s (+34), bwd 781.6 us/tok (−33), loss parity ✓, RSS −37 GB.** BUT resv 182.5
  (98.6% = edge — b3 fails the health filter with everything resident) AND the traffic
  MOVED not died: act-offload now shows d2h=h2d=4218.8 GiB/5st (≈844 GB/step, exactly
  equal = full round-trip) — the ATTENTION SAVED-TENSOR wrapper (attnact1) re-offloads
  what GC-save-CPU used to take; sup runs attnact0. Ladder: 957 → 1016 (attn-KA) → 1050
  (GC-save-HBM). Next: tputask3 = +ASYMM_ATTN_ACT_OFFLOAD=0 at b2 (the sup-regime
  apples-to-apples; b3 cannot hold the fully-resident set). Projection ~1080; C2
  elementwise (+24) is the remaining reserve to clear 1110.
- [S3 step-2 2026-07-18] tputask3 (+ASYMM_ATTN_ACT_OFFLOAD=0 @b2): env knob DID NOT ENGAGE
  (tag still attnact1 — the recomp-off-full stage forces the attention saved-tensor wrapper;
  its saved set CANNOT be resident: ~832 GB @b2). STRUCTURAL TRADE identified: sup pays
  +131 us/tok attention re-fprop (unsloth GC); asym recomp-off pays the saved-set round-trip
  (~703 GB/step each way ≈ 2-3 us/tok IF overlapped, ~90 serial today) ⇒ overlap is the fix,
  not removal. b2 rebase: 1058 tok/s @125.5 GiB (68%, healthy).
- [S-mem(c) H2D 2026-07-18] tputask4 (+ASYM_SAVED_TENSOR_ASYNC_UNPACK=1): **NULL** (1057,
  +0.5 us/tok) — diagnostic: the traffic rides ActivationOffloadManager (attention
  saved-tensor wrapper's engine), NOT decoder._unpack; and stage()'s own comment proves its
  side-stream is structurally null without true prefetch (issue-ahead at layer boundary —
  future milestone M-prefetch).
- [S-mem(c) D2H IMPLEMENTED 2026-07-18] ASYM_SAVED_TENSOR_ASYNC_PACK=1 now drives side-stream
  D2H in BOTH ActivationOffloadManager.offload (the 2812 GiB path) and decoder._pack.
  Pack D2H is fire-and-forget until the backward ready-event wait ⇒ genuine overlap
  available (unlike stage-H2D). A/B = tputask5 (running).
- [S-mem(c) D2H 2026-07-18] tputask5 (+ASYM_SAVED_TENSOR_ASYNC_PACK=1, manager+decoder):
  **NULL** (1056, +0.8 us/tok). TWO nulls ⇒ RE-PRICED ATTRIBUTION: at the current rung
  (GC-save-HBM, b2) the saved-set copies are already substantially hidden; the honest
  remaining gap is bwd 775.7 vs sup 732.9 = **+43 us/tok, dominated by C2 elementwise
  (+24) + residual GEMM/copy (~19)**. The b3 nsys memcpy buckets were measured on the
  PRE-GC-save build and are stale for this rung. Async-pack/unpack code kept (harmless,
  flag-gated, may matter at other anchors/prefetch).
- LADDER @q3-32b 128k (sup-unsloth = 1110 @b2): 957 baseline-b3 → 1016 attn-KA-b3 →
  1050 GC-save-HBM-b3(edge) → **1058 @b2 healthy (68% HBM)** = −4.7% from −14%.
  Remaining: C2 elementwise diet (+24 target) → ~1090; then GEMM-shape/prefetch for the bar.
- [C2 step-1 VALIDATED 2026-07-18] ASYMM_FUSED_LORA_ADDMM=1 (addmm_ epilogue fusion in
  _add_matmul_rows_ + _add_lora_b_delta_; toy parity bit-exact): tputask6 = **1067 tok/s
  (+9), fwd 164.8 (−3.0), bwd 769.9 (−5.0), loss series identical**. Ladder now
  957→1016→1050→1058→**1067** (−3.9%). Residual ≈37 us/tok: remaining elementwise
  (silu-bwd chunk chains, _lora_ds/_lora_b_grad mul passes, dX chains) + GEMM residual.
  Next C2 sites need a FRESH nsys on the current build (b2, all flags) — the b3 pre-fix
  buckets are stale. MoE flip attempt first (tputaslx2, running): needs only +22 us/tok.
- [MoE 2026-07-18] tputaslx @480k b1 (attn-KA + GC-save-HBM stack): 948 → **969** (−2.1%
  vs sup 990, was −4.2%; bwd −26 us/tok; resv 98.8 = 53%; step-1 loss exact). tputaslx2
  (+FUSED_LORA_ADDMM): **NULL for MoE** (969, −0.3 us/tok) — expert LoRA adds live in
  qwen3_moe_finegrained's own chunk paths, not the fused helpers; attention adds at
  h=2048 too small at b1. Remaining MoE gap 22 us/tok — MoE-specific attribution needed:
  tputaslxn nsys @208k b2 current build (running). MoE later-step loss wobble ±0.02-0.04 =
  router noise (pre-existing across unfused runs); step-1 gate holds exact.
- [MoE TWIN LAUNCHED] tputaslx q3-30b 480k b1 with the validated stack (attn-KA +
  GC-save-HBM + MoE class-1 pins): pre-fix gap was only −4.2% (948 vs 990) ⇒ nearest
  window-flip candidate; then 320k b2 (−7.0%) and 208k b2 (−10.3%).
- [MoE nulls 2026-07-18, clean build @480k b1] FUSED_LORA_ADDMM (expert adds live in
  qwen3_moe chunk paths, not the fused helpers) and ROUTE_LORA=1 (**D1 CLOSED**: null even
  without the old noise floor — indexFuncLargeIndex is COMMON-MODE token routing, sup's
  HF router pays it too). LESSON: MoE gap attribution requires the SUP TWIN TRACE (asym
  buckets alone mislead); tputn-c14 sup nsys @208k b2 running for the honest diff.
- [MoE TWIN-TRACE DIFF 2026-07-18 — the decisive attribution, @208k b2, attention-anchored]
  asym-current vs sup-unsloth, us/tok: **index/scatter +14.9** (asym 21.1 vs sup 6.2) ·
  **H2D +8.3** (weight-panel restage ×3 passes; 563 GB/step) · **D2H +5.3** (attention
  saved-set residual) · gemm +2.7 · **elementwise −3.6 (asym BETTER — chunking pays)**.
  Sum +28.5 ≈ the measured gap ✓ books balance. Named fixes, ranked:
  F-A **packed-X reuse for dA (+~8-12)**: under KA the fwd builds route-space `packed`
      [R,H] once (qwen3_moe_finegrained.py:717) then frees it; backward dA re-gathers
      per chunk (:265 index_select) from ctx.x_cpu (4 call-sites :1560/:1646/:1743/:1838).
      Keep `packed` via _HBMKeepManager (layer-scoped, ~13.6 GB @208k b2 — fits) and slice.
      ⚠ TRAP: packed is PRE-MULTIPLIED by routing_weights when input_weighted (:721-722)
      while dA consumes UNWEIGHTED X + weights internally — save the pre-mul tensor or
      gate the reuse on not-input_weighted. Flag: ASYMM_QWEN3_MOE_FG_REUSE_PACKED_X.
  F-B **W-panel cross-pass reuse (+~6-8)**: expert panels re-staged H2D on each of the
      ~3 passes/step (outer fwd, recompute, dX) — LRU-cache staged panels within a step
      (D5-lite; the standalone-refuted C1c was DENSE-sized at +1.4, MoE is +8).
  F-C residual D2H overlap (M-prefetch, designed earlier).
  Projection: F-A alone ≈ 979; F-A+F-B ≈ 987-993 = **the 990 flip line**.
- [F-A IMPLEMENTED 2026-07-18] ASYMM_QWEN3_MOE_FG_REUSE_PACKED_X=1 (default off):
  fwd keeps route-space packed X via _HBMKeepManager ("moe.Xp", guarded on KA + da_gpu +
  NOT input_weighted); _grouped_da_gpu takes packed_rows and slices instead of per-chunk
  index_select; both gate/up call-sites wired; released beside ctx.x_cpu. 8 edits,
  compile-clean. A/B = tputaslx4 @480k b1 (running). Gates: step-1 loss exact
  (~5.727-5.729 series), resv +~14 GB (≤115, healthy), target ≥979.
- [F-A VALIDATED-SMALL 2026-07-18] tputaslx4 @480k b1: 972 (+2), bwd −1.3 us/tok, resv
  +17.2 (mechanism engaged ✓ packed kept). ATTRIBUTION CORRECTION: the +14.9 index bucket
  spreads over pack ×2 passes (structural, sup pays too), dA (now fixed, the honest ~+2),
  and dX/scatter ops — I over-assigned it to dA. Keep the flag (free win, right direction).
- [F-B IMPLEMENTED 2026-07-18] ASYM_W_PANEL_CACHE_GB (default 0=off): stage-once LRU for
  frozen weight panels in frozen_linear.py (_stage_weight_panel; all 4 staging sites
  wired: dense _staged_nt + 3 grouped variants). RECEIPT for sizing: H2D histogram shows
  384-MiB expert stacks staged 12×/layer/step = 918 GiB/4-steps of redundant copies of
  FROZEN weights; LIFO adjacency (recompute/dX/dA same-layer) means a ~6 GB LRU dedupes
  3 of 4 uses. A/B = tputaslx5 @480k b1 (running; full stack, cache 6 GB). Ladder target:
  972 → ~980+; flip line 990.
- [null] AU, ohbm16, batch-as-lever, decoder async-unpack (wrong module for this config),
  MoE ROUTE_LORA (D1 closed), MoE FUSED_LORA_ADDMM (wrong paths for expert adds).

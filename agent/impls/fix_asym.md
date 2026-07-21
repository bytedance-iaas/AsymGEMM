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
VOLUME ANALYSIS (nsys mem_size, 2026-07-18 — supersedes the tag-level 333 GB story):
| op | asym /step | sup /step | asym rate |
|---|---|---|---|
| H2D | 3.5 TB | 0.36 TB | 126 GB/s |
| D2H | 2.8 TB | 0.17 TB | 124 GB/s |
| D2D | 13.0 TB | 1.3 TB | 3.25 TB/s |
Copy-engine wall = 131 us/tok (sup: 10) ≡ the memcpy gap. Corrections this forces:
- NOT pageable-slow (124-126 GB/s) and NOT the U-tags (0.33 TB of the 2.8) — the traffic is
  the FG ACTIVATION PIPELINE round-tripping gate/up/act/x (~36 GB/layer/pass) D2H+H2D, plus
  **13 TB/step of D2D = `_HBMKeepManager.stage()` CLONE churn** (keep-acts keeps tensors in
  HBM but clones them at every stage() for mutation-safety).
- U-offloads ARE consumed (host LoRA-A fwd `_dense_lora_a_cpu_left(u_handle.tensor,...)`
  attention_activation_offload.py:741 + backward dA host read :842) — "never read back" was
  wrong (reads are host-side, invisible to the h2d counter). U is a WITHIN-LAYER round trip:
  offloaded in the GC-recompute fwd, consumed by that layer's backward moments later.
FINAL FIX RANKING (converged):
1. **S-mem-a: fg D2H/H2D round-trip elimination under KA** — audit which fg managers still
   offload for real with ASYMM_DENSE_MLP_FG_KEEP_ACTS_HBM=1 (the KA swap at
   dense_mlp_finegrained.py:246 covers the dense-MLP grad-fwd manager only; attention ctx
   manager :727 and others still ship). Route them through the HBM-keep path. [~50-70 us/tok]
2. **S-mem-b: D2D clone diet** — `_HBMKeepManager.stage()` returns a fresh clone ALWAYS
   (qwen3_moe_finegrained.py:200); clone only for consumers that mutate in place (silu),
   return the tensor for read-only consumers (GEMM inputs, dB/dS reads). [13→~2 TB/step,
   ~25-30 us/tok]
3. **S-mem-c: attention U within-layer HBM residency + GPU LoRA-A** — mirror the dense
   :894 is_cuda pattern for the attention path (one layer's U ≈ 8 GB at b3 128k, fits in the
   40 GiB headroom); dA becomes a GPU GEMM. Kills the U D2H + host GEMM waits. [~15-25 us/tok]
4. C2 elementwise diet (+23 buffer). 5. GEMM +23 — tertiary.
Recoverable ≈ 120-145 of the 145 needed; validate each sub-fix separately per §3.

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

## 5. STATUS LEDGER
- [DONE] C0 async-grad hooks (2026-07-09, fix_throughput.md).
- [REFUTED 2026-07-18] v1-C1b streamed-backward-GEMM theory; v1-C1c standalone panel reuse;
  naive C2 stage-skip. Receipts in §2.0 — do not re-derive.
- [S0 DONE 2026-07-18] attribution: memcpy +120 us/tok + elementwise +23 + GEMM +23;
  attention-kernel parity. (Rate re-analysis: copies run at 124-126 GB/s — C2C-normal, the
  problem is VOLUME+SERIALIZATION, not pageable slowness.)
- [S-mem-b MEASURED NULL 2026-07-18] clone-diet A/B (tputasN1 128k b3): 400.4 vs 401.6 s/it
  = +0.3% (noise). Flag verified live (23.6 GB/step noclone vs 7.9 clone; loss 1.127 ok);
  D2D churn is high-bandwidth (3.25 TB/s) + overlapped ⇒ not wall time. Implementation kept
  (harmless, default-off) but NOT a lever. Lesson logged: nsys VOLUME ≠ wall unless the
  engine is slow or serialized — only the A/B decides.
- [S-mem-c A/B #1 (tputasN2, dense-fg attn-off, b3): MECHANISM CONFIRMED / BATCH WRONG]
  420.9 s/it = 912 tok/s (-5% vs KA 957) BUT d2h -> 0 (traffic kill works) and resv
  136->183 GiB (98%): the un-wrapped attention saved-tensors (+47 GiB at b3) push into
  allocator-churn (bwd 335->355). The copy volume itself was already overlapped.
- [H2D BULK IDENTIFIED from nsys Count/Max]: asym H2D = 9739 copies avg 719 MB, max 6.3 GB
  — the attention CPU-LEFT LoRA path STREAMS U (~3.9 GB) from host per call, fwd + dA
  re-read ≈ ~1 TB/step. attnact0 kills exactly this (and the host GEMM serialization);
  the cost is the +47 GiB saved-tensor footprint.
- [N4 MEASURED 2026-07-18] dense-fg+KA+NOCLONE @128k b2: 261.8 s/it = **978 tok/s — new
  asym best** (+2.2% vs 957; per-token 1046→1022.6 us; resv 148.3 healthy; fwd 167=parity).
  Still -12% vs bar 1110. Backward 853 vs sup 733 us/tok with copies now ~zero in-config
  ⇒ attn-U streams were ALSO mostly overlapped; the ~+23 gain ≈ the host-GEMM serialization
  share. THREE overlap-theories now refuted by A/B (D2D clones, D2H acts, U streams) —
  remaining +120 us/tok backward is NOT copy volume.
- [S0' DONE — ROOT CAUSE, architectural] N4-config nsys: kernels at PARITY (+16 us/tok:
  attn 631 vs 637, gemm 122 vs 127, elem +26); gap = memcpy wall 144 vs 10 us/tok, ~serial.
  Copy signature: max 4.19 GB = [256k tok, 8192] = ATTENTION q/attn-out saved tensors
  (q3-32b q-dim = 64 heads×128 = 8192 ≠ hidden 5120), ~2.2-2.4 TB/step via the DECODER
  saved-tensor offload — NOT the fg managers (their counters read 0), NOT attn-wrapper
  (off in N4). WHY: `recomp-off` = RECOMPUTE OFF — latency mode SAVES+OFFLOADS everything;
  sup RECOMPUTES (traffic only 336 GB/step = the boundary roots ×2). Arithmetic closes:
  asym bwd 853 = 567 pure + 286 offload-tax; sup bwd 733 = 567 + 166 recompute.
- [ENDGAME MEASURED 2026-07-18 — R1 = PARITY, and parity is the ceiling]:
  `asym_cpuadamwds|unsloth-ohbm0` + staged dispatch, q3-32b:
  | seq|B | asym R1 | sup uns | Δ |
  |---|---|---|---|
  | 128k b2 | 1104 (906.1 us/tok, 116.0 GiB) | 1110 (901.0, 127.9) | -0.6% |
  | 128k b3 | 1097 (911.5, 181.2 edge) | — (b3 OOM for sup) | batch=non-lever |
  | 160k b2 | 938 (1066.6, 144.8) | 942 (1061.6, 159.1) | -0.4%, -14 GiB |
  | 192k b2 | 812 (1232.1, 175.3) | 816 b1 (1226.1, 96.4) | -0.5% |
  llama R1 replication (2026-07-18): 128k b2 = 786 (1271.9, 128.7 GiB) vs sup 792
  (1262.9, 147.7) = -0.7% at -19 GiB — parity holds cross-model. 192k b2 = 577 (1733.9,
  182.7 @97.7% edge) vs sup b1 601 (1663.0, 60%) = -4%: the extra batch FITS where sup
  OOM'd but the 97.7%-edge alloc pressure costs ~70 us/tok — the "fit more batch at the
  edge" play LOSES to the healthy b1 (edge-pressure tax > c_fix amortization; consistent
  with the convergence law + thrash-edge physics). R1's B-choice rule: run the LARGEST
  batch that stays ≤~92%, never the largest that fits.
  CLOSURE (scheduler's own choice, measured): llama 192k b1 R1 = **603 tok/s (1658.9
  us/tok, 96.3 GiB)** vs sup b1 601 (1663.0, 110.2) — **+0.3%, first (hair) strict win,
  at -14 GiB**. The B<=0.92 rule turns the -4% edge loss into +0.3%. llama R1 row complete:
  128k -0.7%, 192k +0.3% — parity band, cross-model, rule-validated.
  R1 recovers the old mode's -15% to a PARITY BAND (-0.5..-0.6%, ~5 us/tok = host-weight
  staging residue) with -12 GiB HBM.
  MECHANISM CORRECTION (2026-07-19, code-verified): recomp-off-* is NOT "recompute off" —
  it ALSO runs unsloth-GC recompute (driver :3322-3326 sets gradient_checkpointing=true,
  use_unsloth_gc=true). The name = recompute-OFFLOAD. Its tax vs R1 is
  `unsloth_recompute_save_on_cpu=true` (recompute-side attention saved-tensors D2H, the
  ~2.2 TB/step — the F1 note at :3327 says exactly this) + the fg wrapper offloads. ALL
  THREE modes recompute; they differ only in where recompute-era tensors live
  (R1: HBM transients · R2/latency: attn->CPU + MLP acts HBM-kept · R3/memory: all->CPU). **A strict tok/s beat on dense does NOT exist in this
  regime**: per-token CONVERGES across batch and config at long seq (rc b1 ≡ uns b2 at
  160k; R1 b2 ≡ sup b1 at 192k) — the long-seq deficit is SHARED attention/regime cost,
  not per-config waste; batch amortization is refuted by three independent pairs. The C2
  elementwise delta (+26) belongs to the recomp-off wrapper, absent in R1.
  ⇒ DELIVERABLE (dense): R1 parity at every fitting (s,B) + asym-only capacity beyond
  sup's walls (426@384k vs sup edge 424; 490k est own wall) + the R2/R3 regimes for
  batch/capacity where sup can't fit. The scheduler (scheduler_v2.md §8) is the win
  mechanism, not a per-point tok/s delta. REMAINING upside: llama + MoE R1 replication
  (same physics predicts parity; sup walls 326k/660k), and C1a kernel work (shared-cost
  reduction) as the only path to strict dense wins.
- [null] AU, ohbm16, batch-as-lever.

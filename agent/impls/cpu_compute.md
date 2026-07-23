# CPU Compute — record & steering doc for CPU-side optimization of AsymGEMM LoRA-SFT

**Rewritten clean 2026-07-21** (the prior 1,800-line layered version is preserved verbatim at
`agent/impls/archive/cpu_compute_full_pre0721.md`; per-gate execution detail lives in
`fix_cpu_compute.md`; placement rules in `placement.md`; environment facts in
`../project_rules.md` — read that first).

Measurement law (summary): `PROFILERS=source`; 1 warmup + 4 measured steps; steady =
mean of middle 2; **same-day adjacent-pair A/Bs only** (drift: 32k ±0.1 s within-day / ~±1 s day-to-day;
128k ±10 s day-to-day and monotonic within heavy days, 616→663 — hence adjacent pairs); flags verified via `/proc/<gpu-pid>/environ` + ENGAGED markers; leaves
overwrite → snapshot numbers immediately; process truth = `nvidia-smi
--query-compute-apps`; CPU microbenchmarks only in clean windows (GPU idle); host pool
= **~957 GB** (NUMA 0+1; `free`'s 1.69 TB is fabric-inflated); lossless only.

Models: 30B = Qwen3-30B-A3B (MoE, 48 layers). 32B = Qwen3-32B (dense, 64 layers).
All runs LoRA r64/α16, batch 8, 1×GB200 GPU unless stated.

---

## 1. MASTER TABLE — every CPU module: what, why, micro, e2e, verdicts

Cell notation: **t** = steady step seconds a→b (same-day pair) · **G** = peak GPU GiB
a→b · **C** = peak host GiB a→b · "forced" = feature forced on to justify auto-off ·
each row's pair has its own same-day reference (the campaign spanned days; never compare
cells across rows by absolute t).

| Module | What / Why | Micro (isolated) | 32k MoE (t·G·C) | 128k MoE (t·G·C) | Dense 32k (t·G·C) | Research | Adopt where |
|---|---|---|---|---|---|---|---|
| SwiGLU fwd on CPU (+chunked) | activation computed on CPU where inputs already live, overlapped under GPU down-GEMMs; frees staging + HBM | kernel 190→44.8 ms (4.2× vs PyTorch-CPU; @128k 703.6→178.5) | t 97.39→95.13 (−2.3%); chunked t 91.06→90.69, G used-peak 51→38.4 (−13) · C ≈ | forced t 629.8→634.4 (+0.7%, within 128k band — auto-off decided by MECHANISM: window exhaustion, 16T ablation) ❌ | excluded by bytes-guard (13.1 GB act; sync arm +9.3% ❌) | ✅ crossover exhibit (lead claim) | 32k-class MoE only — automatic rows/bytes gate |
| SwiGLU bwd on CPU | backward twin; output consumed immediately — no idle window exists at any length | kernel 320.7→58.9 ms (5.4×) | placement t 91.66→98.71 (+7.7%) ❌ | not run (structural) | ❌ | ✅ negative control of crossover | nowhere (kernel retained as enabler) |
| Expert adapter-grad deposit | LoRA wgrad on CPU off the critical path; fp32 born in optimizer buffer; only dS (~0.27 GB/layer/proj @32k) crosses vs 8.6 GB/layer of X re-reads freed | layer-set: CPU 334+334+117 ms vs GPU-remote 919+919+938 vs copy+GPU 42+42+16 (isolated-best; loses e2e — 3-arm ablation) | t 95.13→92.64 (−2.6%) · G 50.4→51.4 · C ≈ | t −0.4% (repeat −0.42% ✓) · G 176.3→178.6 · C 693.0→695.1 | ❌ host-OOM all variants (retention-bound ~10 GiB/s ramp; unblock spec'd) | ✅ selective bwd-op placement (main exhibit) | MoE both lengths; dense blocked |
| Attention adapter-grad deposit | same mechanism for q/k/v/o | CPU 60.8 vs GPU-remote 106.5 vs copy+GPU 5.2 ms | t 92.70→91.66 (−1.1%) · G 54.2→50.0 (−4.2) · C 364.9→367.1 | forced t 616.5→622.5 (+1.0%, within band — auto-off by mechanism: rows ×4 + C/G collateral) · G +3 · C +13 ❌ | ⬜ untested single | ✅ coverage of same claim | 32k-class MoE only — automatic rows gate |
| Norm recompute (skip save+reload) | re-derive fp32 norm outputs bit-identically from one saved bf16 input at backward-use; kills the largest non-dedupable save class | recompute 23.9 ms sum / 14.0 ms overlapped vs replaced roundtrip 82.9 ms (~3.5× sum, ~6× overlapped; @128k class 333 ms) | t 90.62→87.95 (−2.95%) · G −1.5 · C −16 | t 618.30→596.23 (−3.57%) · G ≈ · C 695→631 (−64) | t 385.39→369.37 (−4.16%) · G −2.2 · C −32 | ✅ recompute arm of the 3-way policy | **everywhere** — the only universal win (first dense win) |
| Rope recipe recompute | saved rope/SDPA operands replaced by rebuild-recipes (graph untouched; bitwise rebuild from offloaded input) | 34/34 bitwise units; −0.46 TB/step D2H @128k (~4× smaller class @32k) | t 85.60→86.88 (**+1.5% — real small regression**, 12× the 32k band; why OFF here) · C −4.6 | t 600.32→602.48 (+0.4% noise) · C 632→614 (−18) | t 330.08→332.38 (+0.7% noise) · C −9 | 🟡 extension of recompute claim | memory feature: ON where host RAM binds (128k + dense); OFF 32k |
| Restage prefetch | reloads issued one layer ahead on a dedicated 2nd H2D stream, per-tensor ready events | per-tensor wait 15.2→0.00 ms (@128k 61.2→6.3) | t 88.11→85.57 (−2.9%; step wait 4.20→1.82 s) · G 54.2→57.7 (+3.5 held stages, guarded) · C ≈ | neutral — regime bandwidth-bound, guard no-ops (honest; 49.6 s/step unrecoverable) | t 373.39→328.97 (−11.9%) · C part of dense −48 | 🟡 the wait-ATTRIBUTION method (CUDA-event pairs) is the paper-worthy part | 32k + dense; 128k harmless-neutral |
| Boundary pinned+async copy | checkpoint boundary: pageable/host-blocking → pinned async → prefetched ahead | 22.1→4.9→0.00 ms per boundary | t 93.73→92.70 (−1.1%) · G +3.6 (pinned pool) · C ≈ | ≈neutral (same-day isolation: joint with dup-copy removal ≈ 1.0 s BENEFIT — 613.5 with vs 614.5 without; within the ±10 s band) | ≈neutral (+0.3%) | ❌ engineering | everywhere (never hurts) |
| Duplicate-copy removal | delete literally doubled H2D stages | 2 copies→1 (cached) | t 91.66→91.06 (−0.7%) | ≈neutral | n/a (MoE path) | ❌ engineering | everywhere applicable |
| Fused optimizer drain | bf16→fp32 widen + grad-norm in ONE pass | 73.3→5.5 ms (13.3×) | ≤0.1% substage | same | same | ❌ engineering | everywhere |
| Save-dedup | duplicate small saves → shared handle | pack 0.24→~0.00 ms | ~neutral | ~neutral | ~neutral | ❌ — but its NEGATIVE finding (big fp32 pairs are distinct storages ⇒ not dedupable) is the recorded motivation for norm-recompute | shipped, **default-OFF** (no measured win ⇒ no default-on; env-armable) |
| Wgrad 96-thread two-socket | deposit GEMM spread across both CPU sockets (compute-bound ⇒ scales; streaming does not) | 1.96→2.85 TF/s (+45%; up to 3.24) | t 85.50→85.58 (neutral — work is hidden; headroom win) | rides deposits | inert (P8 blocks dense deposits) | ❌ engineering | on under policy |
| Byte-diet stage-reuse | stage gate/up once per layer backward and reuse (−2.25 TB/step target) | mechanisms unit-green; stream-race found+fixed en route | inert (0 attempts) t 85.52→85.63 | t 600.71→601.91 (neutral) · G 175→179 — **guard-starved 953/960: no free-HBM window mid-backward (free dips to 0.2 GiB)** | ⬜ | ❌ | dormant — auto-engages when G headroom exists (e.g. 2-GPU) |
| Placement-policy module | ONE runtime object makes every CPU↔GPU decision from measured thresholds; per-run decision tracing | 13/13 dry-run exact (P10 sets) | t = manual within noise (90.81 vs 90.73) | **automates the manual 32k/128k config split** | dense kill-switches verified | ✅ **system centerpiece** | everywhere — the one production flag |
| Batch scaling (measurement, not feature) | can freed HBM convert to throughput? | — | b8→b16 tok/s exactly neutral (2824.7); b24 −12.4% ❌ (policy sheds wins as rows grow) | n/a (no HBM headroom) | b16 host-OOMs | ❌ measured finding | nowhere — freed HBM's value = the seq ceiling, not batch; if batch scaling is ever wanted, P1/P3 thresholds must be re-measured at b16/b24 shapes (separate study) |
| **CUMULATIVE (one flag `ASYM_PLACEMENT_POLICY=1`, lossless)** | | | **t 97.39→85.57 (−12.1%)** · C **−13** (endpoint 364.5→351.5; recompute −16, deposit +2.2) | **t 618.3→596.2 (−3.6%, same-day pair; later-day full-stack refs ≈600.3–600.7)** · **C −82** (binding resource) | **t 385.4→332.4 (−13.8%, one-flag endpoint incl. rope)** · **C −57** (329.0/−14.6% was the pre-rope best; rope adds C −9 at +0.7% t noise) | | |

**128k saturation statement (campaign boundary, 2026-07-21/22):** after all shipped
features, ~47.4 s/step (7.9%) of exposed H2D wait remains, spread over ~20 tags (largest
7.7 s/step), running at 2.5× LPDDR contention; no hold/keep scheduling lever can engage
at G≈176/184 (byte-diet guard-starved). Remaining lever classes are out of scope:
fp8/quantized staging (lossy), 2-GPU sharding (frees headroom ⇒ dormant mechanisms
engage), hardware bandwidth. Three consecutive neutral/negative 128k levers ⇒ measured-saturated
under 1-GPU bf16.

## 2. MODULE MICROBENCH — FINAL (isolated proof per module)

Clean-window, production shapes, both regimes; rerunnable:
`ASYM_CPU_OPS_THREADS=48 .venv/bin/python tests/bench_modules.py --final` (3-arm detail:
`--fair3`). Arms: **ours-CPU** (data already CPU-resident) · **PyTorch-CPU** (same math,
stock ops) · **GPU-remote** (GPU kernel reading CPU memory over C2C, no copy) ·
**copy+GPU** (H2D then compute; overlapped = max(copy, compute)). Production choice is
critical-path/contention-driven, NOT isolated speed — the two facts that flip verdicts:
(a) the wait counters measured **60.1 s/step (9.7%) of GPU-exposed reload wait @128k**
(4.2 @32k, 65 @dense) on the shared copy stream, so "cheap" copies queue in reality;
(b) off-critical-path CPU seconds are hidden inside dependency windows (≈free), proven
by the 3-arm e2e ablation where the copy+GPU plumbing variant was the WORST config
(94.12 s vs deposit 91.66 s @32k).

| # | Module | What it does | Numeric benefit (isolated) | Per-step aggregate @32k / @128k | Production | Research-grade? |
|---|---|---|---|---|---|---|
| 1 | SwiGLU fwd kernel + placement | MLP activation computed on CPU where its inputs already live, overlapped under the GPU's down-GEMMs; removes the gate/up H2D staging | kernel **4.2×** vs PyTorch-CPU (190.0→44.8 ms; @128k 703.6→178.5); copy+GPU ovl 30.4/164.2; −5.8 GB/layer link staging | 2.15 s hidden / — (auto-off) | CPU @32k · GPU @128k | ✅ crossover exhibit (win@32k / lose@128k IS the lead claim's evidence) |
| 2 | SwiGLU bwd kernel | Same op's backward; kernel exists, output consumed immediately by the next GPU GEMM | kernel **5.4×** vs PyTorch-CPU (320.7→58.9; @128k 1301.5→233.0); placement forced e2e **+7.7%** | — | GPU both regimes | ✅ negative control of the crossover story |
| 3 | Adapter-grad, attention (deposit) | LoRA weight-grad on CPU off the critical path; fp32 result born in the optimizer's CPU buffer; only dS (32 MB) crosses | frees **106.5→0 ms per projection (×4/layer)** of critical-path GPU kernel; −~5.2 GB/layer link reads (251.7 GB/step class @32k); CPU cost 60.8 ms hidden; copy+GPU isolated 5.2 ms (loses e2e — fact (b)) | 11.7 s hidden → −1.1% e2e / auto-off (rows ×4) | CPU @32k only | ✅ selective backward-op placement claim — 3-arm micro complete |
| 4 | Adapter-grad, expert (deposit) | Same mechanism, MoE expert projections (largest grads) | frees **~2.8 s→0 per layer-set** (919.6+919.6+937.9 ms) of GPU-remote kernels; −8.6 GB/layer link reads; CPU 334+334+117 ms hidden (@128k ×4, window still fits) | 37.7 s / 151 s hidden | CPU both lengths | ✅ same claim, main exhibit |
| 5 | Norm(+rope-class) recompute | Stop saving+reloading fp32 norm outputs during layer recompute; re-derive bit-identically from one saved bf16 input at backward-use | **~3.5× (sum) / ~6× (overlapped)** cheaper than the replaced roundtrip (recompute 23.9 ms sum, 14.0 ms overlapped vs save+reload 82.9 ms; @128k class 333.3 ms) | replaces 4.5 s / **18.0 s** of copies → e2e −2.95% / **−3.57%** / dense −4.16% | GPU-recompute everywhere (bit-identical; the CPU-kernel arm is ≤1 ulp = outside the lossless claim) | ✅ recompute arm of the 3-way checkpoint policy |
| 6 | Boundary copy | Checkpoint boundary: pageable-blocking → pinned-async → prefetched one layer ahead | **22.1 → 4.9 → 0.00 ms** per boundary | 235→~0 / 948→~0 ms | pinned + prefetch | ❌ engineering (fully measured) |
| 7 | Reload prefetch | Offloaded-tensor reloads issued ahead of need on a dedicated 2nd H2D stream, per-tensor ready events | per-tensor wait **15.2→0.00 ms** (@128k 61.2→6.3); step wait **4.20→1.82 s** @32k | −2.4 s @32k (−2.9% e2e) / 49.6 s unrecoverable @128k (bandwidth-bound — honest) | on where G-headroom allows | 🟡 the wait-ATTRIBUTION method (CUDA-event pairs) is paper-worthy; prefetch itself standard |
| 8 | Optimizer drain | bf16→fp32 widen + grad-norm in ONE fused pass | **13.3×** (73.3→5.5 ms per drain) | 73→5.5 ms both | fused | ❌ engineering |
| 9 | Save-dedup pack | Duplicate checkpoint saves packed once (shared handle) | pack 0.24→~0.00 ms; class ~23/92 ms per step | e2e-neutral (honest) | shipped, default-OFF (env-armable) | ❌ engineering — but its NEGATIVE finding (the big fp32 pairs are distinct storages ⇒ not dedupable by any lossless key) is cited as the motivation for #5 |

---

## 3. WHAT THE MEASUREMENTS ESTABLISHED (the paper story, guard-verified)

**Lossless constraint (standing):** everything shipped is bit-exact or ≤bf16-rounding
with SMOKE loss-parity gates judged against the measured 0.67–1.0% rerun envelope (route101 atomic-scatter nondeterminism makes cross-process bitwise parity unachievable; envelope + no-drift is the formal criterion, plus per-feature bitwise unit suites — qknorm 9/9, dedup 9/9, prefetch/recipe 34/34, byte-diet 19/19); no int8/lossy arms.

1. **Characterization (measured):** operator-level CPU↔GPU placement on a coherent
   superchip has a measurable crossover — the same op wins at small size and loses at
   large (SwiGLU: −2.3% @2.1M rows vs +0.7% @8.2M rows; attention-grad: −1.1% vs +1.0%) —
   mechanism = overlap-window exhaustion, NOT DRAM contention (16-thread ablation);
   utilization ≠ schedulability (window-inventory analysis: only ~3% of idle
   core-seconds are schedulable under the production schedule); **schedule fit beats
   kernel speed** (same wgrad kernel: blocking placement loses, deposit placement wins).
2. **System:** the placement-policy module (`placement_policy.py`, rules P1–P15 in
   `placement.md`) makes every decision from measured thresholds with per-run decision
   tracing; it reproduces hand-tuning at 32k and **automates the 32k/128k config
   split**. Selective backward-operator placement (deposits) + the recompute policy arm
   are the two mechanisms that survive at the heavy workloads.
3. **Artifacts:** Grace/ARM SVE kernel family — fused SwiGLU fwd (4.2×) + bwd (5.4×),
   grouped LoRA wgrad, rmsnorm (7×, ≤1 ulp), fused widen+sqsum (13.3×) — all
   parity-tested; the pinned-memory ledger (cudaHostAlloc is invisible to OS Mlocked
   counters — ours is the only accurate accounting); the GPU-exposed-wait attribution
   counters (CUDA-event pairs at every restage site).
4. **Honest negatives (measured, cited as findings):** batch scaling ≠ throughput at
   32k (b8→b16 exactly tok/s-neutral at 2824.7; b24 −12.4% — structural: the policy's
   rows-gates shed the placement wins as batch grows); dense-model
   CPU deposits infeasible-as-built (retention-bound, not pinning-bound — ~10 GiB/s
   ramp; process RSS blind to it); the big fp32 save pairs are distinct storages
   (dedup impossible ⇒ recompute is the only collector); 128k reload wait is
   bandwidth-bound (prefetch cannot recover it; 49.6 s/step stands).

**Novelty guards (verified against literature; full citation lists preserved in
`archive/cpu_compute_full_pre0721.md` §1.3/§6 + the C2 'Mandatory citation groups' block and fix_cpu_compute.md):** SuperOffload owns optimizer-on-Grace /
GraceAdam + speculative-update-with-rollback (cite; our exactness needs no rollback and
our CPU work is fwd/bwd operators, which theirs never touches). KT-SFT owns
whole-expert-module CPU training (capacity-forced, layer-serial, x86-AMX; ours =
per-op, policy-driven, overlap-scheduled, Grace). Zero-Bubble-PP / OOO-Backprop own
same-GPU wgrad deferral (ours crosses processors, exactly, within-step).
NEO/Fiddler/CaraServe are inference-only. Poolside's GB200 blog = two-way manual
activation policy (ours adds the recompute arm + automation). PEARC'24 = bare BLAS
threshold dispatch (ours is training-graph-aware). **Dropped by decision:** microbatch
pipelining (user veto) · SVE Adam (= GraceAdam) · int8/lossy (lossless constraint).

---

## 4. REMAINING WORK (live; per-gate detail in fix_cpu_compute.md)

| Item | State | Why it matters |
|---|---|---|
| Dense prefetch re-gate (guard 4×→2×) + expert-grad 128k repeat pair | ✅ closed | see fix_cpu_compute.md chain12 entry |
| Rope/SDPA-operand recompute | ✅ shipped (tokens-gated) | memory feature — pays where C binds (128k + dense); OFF at 32k (real small regression) |
| Per-socket act-pool placement | ❌ closed negative (socket-1 H2D 125.3 vs 211.4 GB/s = −41%, fabric-bound; no e2e spent) — streaming kernels stay single-socket; two-socket WGRAD kept (P14, +45%) | closed |
| **128k byte-diet (P15 direct-reuse)** | ❌ closed neutral | guard-starved at G≈176/184 (953/960 denied); mechanisms dormant behind guard |
| **128k regime: measured-saturated** | 📌 boundary | 3 consecutive neutral/negative levers; residual 47.4 s/step exposed H2D needs fp8 staging / 2-GPU / hardware — user decision |
| Ceiling search with the adopted stack | ⬜ proposed | converts the −82 GB host saving into a **longer max context** (measured ΔC/Δseq slope ≈ 2.7–3.4 GB per 1k tokens ⇒ −82 GB ≈ est. ~24–30k tokens; proposal-level) — the memory-efficiency headline |
| sdpa policy-map arm · nsys overlap figure | ⬜ optional | paper exhibits |
| rmsnorm-BACKWARD CPU kernel (K-7 follow-up — recorded precondition for ever running a CPU arm of P12 norm recompute) | ⬜ optional enabler | opens the P12 CPU arm |
| Rope-recipe v2 shared staged-x (v1 shares nothing; recorded as v2 if G allows) | ⬜ optional | rope G reduction |
| Kernel-variant queue (widen+sqsum retry @62% STREAM; FEXPA exp; prefetch-distance sweep; wgrad BFDOT dS-packing; kt-kernel tricks) | ⬜ optional paper-optics — measured-value argument says hidden-work kernel speed is e2e-invisible, so class is closed for e2e | kernel completeness only |
| Rope 64k-class gate study (default tokens-gate turns rope ON at 64k×b8 — unmeasured regime: does it pay C with time-neutrality?) | ⬜ proposed/optional | closes the one unmeasured cell of the rope gate |
| Cross-suite test-ordering failure (save_dedup→pinned_ledger in one process trips the unpinned-fallback assert) | ⬜ harness-only (production-safe; each suite green alone; workaround = separate processes; next: bisect) | test hygiene |
| 32B deposit unblock (retention backpressure at source granularity) | ⬜ spec'd only | would open the dense column for deposits |

## 5. PROTOCOL — exact commands (copy-paste)

```bash
# 30B MoE, 32k (short-context reference)
GPU_ID=0 PROFILERS=source MAX_STEPS=4 WARMUP_STEPS=1 OVERWRITE=true \
ASYM_PLACEMENT_POLICY=1 ASYM_CPU_OPS_THREADS=48 \
RUNS="q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker101-ceil0000-ohbm0|ligerloss1 ; 32000|8|1 ; none|false|false|false|false|false" \
bash scripts/lf/profile_lora_lf_test_both.sh

# 30B MoE, 128k (heavy-workload headline): same, RUNS seq 128000|8|1
# 32B dense, 32k: RUNS="q3-32b|1 ; asym_cpuadamwds|recomp-off-full-fg-ker000-ceil0000-ohbm8|ligerloss1 ; 32000|8|1 ; ..."
#   (b16 host-OOMs on 32B — do not use)
# SMOKE parity: seq 8000|4|1, MAX_STEPS=6, same seed, feature ON vs OFF
# Feature flags: the placement policy reads placement.md's rules; per-feature flags are listed there.
```

Gotchas (each cost ≥an hour once): venv relocation breaks shebangs
(`grep -rIl 'AsymGEMM-SFT-<OLD>' .venv` → sed-repoint) · host watchdog SIGSTOPs when
free CPU RAM <35 GB (soft-OOM, not your bug) · `PROFILERS=both` must run alone ·
kt-kernel's working tree holds the newest ARM SFT code — never checkout it away ·
rebuild `MAX_JOBS=8 .venv/bin/python setup.py build_ext --inplace` after csrc edits ·
launch e2e runs detached (`setsid nohup`) with stall-aware waiters (log-mtime silent
>15 min ⇒ py-spy + diagnose) or runs outlive agent task rounds.

## 6. EVIDENCE LOG (compressed; full per-gate detail in fix_cpu_compute.md + artifact leaves)

- 07-13 — census (fwd 18.6 / bwd 74.4 / opt 1.15 s @32k; optimizer = 1.2% ⇒ idle-CPU
  thesis confirmed); fused SwiGLU kernels shipped (parity ≤1–2 ulp); async placement
  −2.3%; the crossover found (128k +0.7% WORSE when forced; 16-thread ablation ⇒ window exhaustion, not
  DRAM); auto rows-threshold shipped; wgrad deposit −2.6% / −0.4% (the both-lengths
  win); blocking-wgrad control rejected by dominance arithmetic.
- 07-14/15 — op-coverage campaign, 19 gated runs: boundary pin −1.1%; attention deposit
  −1.1% @32k / +1.0% @128k; 3-arm ablation (copy-plumbing arm worst: 94.12 s);
  chunked mul −0.4% & −13 GB HBM; silu-bwd placement +7.7% (predicted fail); BFDOT
  m-pair micro-fail; stage dedup −0.7%; 32B deposits all host-OOM (memory-sampler: ~10 GiB/s ramp,
  1,091 GiB in ~8 min); consistency matrix; window-inventory analysis (dated 07-13: ~3% of idle core-seconds schedulable). Bugs
  found+fixed en route: worker RLock deadlock (3.7 h hang), wgrad thread-collapse
  (30×), pool 3-D bucketing (would-be 1 TB alloc), deposit deferred-release sweep cadence (retained up to 64 layers of handles before step-end), bytes-vs-rows guard.
- 07-16 — placement-policy module shipped+gated (13/13 dry-run exact, SMOKE ≤0.87%, e2e
  both lengths; automated the config split); save-dedup shipped-neutral with the
  distinct-storages proof; pinned ledger+caps shipped (cudaHostAlloc invisible to OS
  Mlocked); dense C-bound cells closed; adoption audit (cross-day claims retracted;
  same-day 128k isolation: boundary+dup-copy-removal joint ≈ 1.0 s benefit = neutral-within-band there).
- 07-16/17 — batch-scaling measurement (b8 2824.7 tok/s = b16 exactly; b24 −12.4% —
  freed HBM ≠ throughput; policy rows-gates shed wins as batch grows); item-4 dense
  re-gate quantified mlp-family retention 56.5→290 GB (first dense retention numbers).
- 07-17/18 — norm recompute gated everywhere (−2.95% @32k C−16 · −3.57% @128k C−63.7 ·
  dense −4.16% C−31.8 — first dense win); fair 3-arm microbench; GPU-exposed-wait
  counters (60.1 s/step @128k, 4.2 @32k, 65 dense).
- 07-18 — prefetch shipped (34/34 units; 32k −2.9% → 85.57 s, wait 4.20→1.82 s/step;
  128k honest-neutral (bandwidth-bound); dense partial → tuned re-gate queued).
- 07-20 — MODULE MICROBENCH — FINAL (9 rows, both regimes, per-step aggregates);
  chain12: dense guard-tuned prefetch re-gate 373.39→328.97 (−11.9%); expert-grad 128k
  repeat −0.42% (reproduces −0.4%; stays ON).
- 07-21 — kernel campaign: wgrad two-socket 96T +45% micro (1.96→2.85 TF/s), e2e ride
  85.50→85.58 neutral-as-predicted → default-on (P14); rmsnorm hoist −21% (355 GB/s =
  71% STREAM); silu FDIV-kill ❌ (+19–20% slower AND 917-ulp break — reverted);
  NT-stores closed (Grace auto WA-evasion); per-socket act pools ❌ (socket-1 H2D −41%,
  fabric-bound); BFMMLA closed by measured-value argument (kernel speed e2e-invisible
  for hidden work). Rope recipe recompute (P2) shipped as a MEMORY feature: 128k +0.4%
  t (noise) C −18; 32k +1.5% t (real small regression) C −4.6 → OFF; dense +0.7% t
  (noise) C −9 → ON where C binds (policy-encoded, min-tokens gate).
- 07-21/22 — byte-diet (P15): 128k 600.71→601.91 neutral, G 175→179; root cause =
  guard-starved (953/960 denied; no free-HBM window mid-backward, dips to 0.2 GiB);
  default-OFF, dormant → auto-engages with G headroom (2-GPU). En route: stream-race
  in mech-3 first cut found+fixed ("never write to an offload source" rule); zombie-
  trainer kill discipline (kill nvidia-smi PIDs, not patterns); parity gate formally
  recalibrated to the rerun envelope (route101 atomic scatter ⇒ bitwise cross-process
  parity unachievable; 19-test bitwise unit suite + envelope + no-drift is the gate).
  128k declared measured-saturated (residual 47.4 s/step exposed H2D across ~20 tags,
  largest 7.7; no scheduling lever can engage at G≈176/184). Campaign paused for user
  review.


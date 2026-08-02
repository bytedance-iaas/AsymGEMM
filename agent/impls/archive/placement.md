# PLACEMENT POLICY — authoritative record of every CPU↔GPU placement rule

Companion to `cpu_compute.md` (evidence) and `fix_cpu_compute.md` (implementation plan).
This file is the SPEC the runtime policy module implements. Every rule cites its measured
basis. Update this file whenever a threshold or verdict changes — this is the single
source of truth for placement decisions.

Hardware context: GB200 (Grace 2×72 Neoverse-V2, ~500 GB/s LPDDR/socket; C2C ~375/297
GB/s; GPU ~186 GiB HBM). All thresholds measured on Qwen3-30B-A3B (MoE) and Qwen3-32B
(dense), LoRA r64, 1×GPU, `PROFILERS=source`, same-day A/Bs (see cpu_compute.md).

---

## P1. MoE SwiGLU activation on CPU (async, worker-overlapped)
- **Rule: ON iff** expanded routed rows ≤ 4.2M **AND** activation bytes ≤ ~6.4 GB
  **AND** the CPU worker is enabled. Else GPU path.
- Basis: win −2.3% at 2.1M rows (32k×b8); loss −0.7% at 8.2M rows (128k) — mechanism =
  overlap-window exhaustion (16-thread ablation ruled out DRAM contention); dense
  13.1 GB act = +9.3% catastrophic (sync) and bytes-guard-excluded (async).
- Flags today: `ASYMM_QWEN3_MOE_FG_CPU_ACT(+_ASYNC)`, `ASYMM_CPU_WORKER`, rows guard
  `ASYMM_QWEN3_MOE_FG_CPU_ACT_MAX_ROWS=4194304`, bytes guard (added 2026-07-14).
- Status: gate AUTOMATIC (rows+bytes). The chunked-multiply variant rides this gate
  (inactive above threshold by construction — code-proven, `qwen3_moe_finegrained.py`
  split_silu branch).

## P2. MoE LoRA-A weight-gradients on CPU (deposit)
- **Rule: ON for MoE models, all context lengths** — subject to P7 (retention budget).
  OFF on dense models until P8 (pinned cap) exists.
- Basis: −2.6% @32k and −0.4% @128k (the only feature positive at both lengths); its
  off-critical-path window (rest of layer backward) GROWS with seq. Dense: all attempts
  host-OOM (node-level pinned growth; process RSS exonerated).
- Flags: `ASYMM_LORA_A_GRAD_CPU=1`, bg deposit worker `ASYM_CPU_WORKER_BG=1`.
- Design law (measured, 3-arm ablation): the win is the CPU *placement* (frees the GPU
  kernel + its C2C activation reads), NOT earlier D2H — GPU-wgrad + direct-D2H was the
  WORST arm (+2.7% vs deposit).

## P3. Attention LoRA-A weight-gradients on CPU (deposit)
- **Rule: ON iff** per-projection rows ≤ ~256k (≈ batch×seq at b8×32k). Else GPU.
- Basis: −1.1% @32k (256k rows); +1.0% @128k (1.02M rows — window outgrown).
- Flags: `ASYMM_ATTN_LORA_A_GRAD_CPU=1`. **Gate AUTOMATED (2026-07-16, fix_cpu_compute
  item 1): `placement_policy.attn_wgrad_deposit(proj_rows)` applies the 256k rows rule
  per call — gated same-day at 32k (policy ON ≈ manual, deposit ENGAGED) and 128k
  (auto-OFF, no K-2 marker, 616.5 s ≤ manual 628.0). The env flag remains the manual
  mechanism for policy-OFF runs.**

## P4–P6. Never-hurts class (always ON, all models/contexts)
- **P4** Pinned+async gradient-checkpoint boundary offload (was pageable/blocking):
  −1.1% @32k; ~neutral @128k (same-day isolation 2026-07-16: joint with P5 ≈ +1.0 s);
  neutral on dense 32B. Wins scale with exposed copy time.
- **P5** Duplicate host↔GPU staging-copy removal: −0.7% @32k; ~neutral @128k.
- **P6** Fused bf16→fp32 widen + squared-norm in the optimizer drain: substage win,
  seq-independent, loss-parity clean.
- Kernels backing all placements: fused SVE SwiGLU fwd/bwd + software prefetch —
  bit-accurate (≤1–2 ulp), 3–4× vs PyTorch-CPU.

## P7. Deposit retention backpressure (safety, always armed)
- **Rule:** deferred-source retained bytes ≤ `ASYM_DEPOSIT_RETAIN_BUDGET_GB` (48);
  the producer blocks on the oldest outstanding task when the budget would be exceeded.
  Converts OOM into bounded slowdown; lossless; unit-proven bound.
- Note: necessary but NOT sufficient for dense 32B (growth there is node-level pinned
  memory outside Python-visible allocations — see P8).

## P8. Dense-model (32B-class) CPU compute — OFF (retention-bound; sharper diagnosis 2026-07-16)
- **Rule: all CPU-compute placements OFF on dense 32B-class** (host-memory-bound regime;
  every attempt soft-OOM'd at 0 measured steps; only boundary-pin measured ≈ neutral).
- **Item-4 re-gate outcome (3 attempts, 2026-07-16)**: the pinned ledger + per-family
  caps WORK (enforcement held total pinned at the 350 GB cap; per-family attribution:
  mlp act-pool retention grew 56.5 → 290 GB under deposits — first quantified), and
  the cap-denial unpinned fallback works end-to-end (cpu-left/cpu-right consumers
  route to the staged-GPU GEMM, unit-parity exact). BUT the node still soft-OOM'd
  (watchdog at 35 GB avail on the REAL ~957 GB CPU pool): **the growth is RETENTION
  (deposit-deferred mlp act handles), not page-locking — a pinned cap only changes
  the memory KIND (unpinned drain ~1 GB/s to the floor)**. Also proven:
  cudaHostAlloc pinning is INVISIBLE to Mlocked/Unevictable on this kernel (185 MB
  reported vs 245–364 GB ledger-booked) — the ledger is the only reliable tracker.
- **Unblock condition (revised)**: a RETENTION budget on the dense deposit path
  (deposit_ctx deferred-source bytes, mirroring the attention P7 budget) — not a
  pinned cap; then re-gate. Until then P8 stays the kill-switch.

## P9. Rejected placements (measured — do not re-enable without new evidence)
- CPU SwiGLU-backward + LoRA-B weight-grads on CPU: **+7.7%** (zero-width window by
  dataflow; kernels retained as enablers only).
- Paired-BFDOT GEMM microkernel: micro-fail (1.43 < 1.58 TF/s) → never e2e'd.
- GPU-wgrad + direct-D2H (hook bypass): worst ablation arm.
- Microbatch / gradient-accumulation pipelining: dropped by user decision.
- int8 / any lossy compute: excluded by the losslessness constraint.
- SVE AdamW: SuperOffload's claimed contribution (GraceAdam) — cite, don't build.

## P12. Norm recompute-instead-of-save (q/k-norm; save-traffic transform, NOT CPU compute)
- **Rule:** save the bf16 norm INPUT once (pooled pinned offload); rebuild the exact
  fp32 chain at backward by re-running the module's original forward on the same device
  under `enable_grad`. **Bit-identical by construction** (same bf16 input + same math on
  the same device ⇒ identical intermediates and gradients — unit gate asserts exact
  `torch.equal`). Collects the proven-non-dedupable double-fp32-save class (item 2 /
  smoke6: the two [B,S,H,D] fp32 upcasts per norm are DISTINCT allocations, ~400 GB/step
  @32k, ~1.6 TB/step @128k; recompute is the only collector).
- **NOT gated by P8**: this is a save-traffic transform, not a CPU-compute placement —
  one input is offloaded then released within the same layer backward (retention-safe;
  the R3 dense pair confirms the pinned/retention counters stay flat). So it is eligible
  on dense 32B where every deposit is blocked.
- **CPU-kernel note**: the shipped `cpu_rmsnorm_bf16` (K-7, 7×, ≤1 ulp) is FORWARD-only;
  the norm BACKWARD needs the exact fp32 chain on the GPU, and any nonzero ulp breaks
  bit-parity — so `ASYMM_QKNORM_RECOMPUTE_CPU` is accepted but resolves to the GPU
  recompute with a one-line notice. A CPU path requires an rmsnorm-BACKWARD kernel first.
- **Rope variant GATED 2026-07-21 (P2 recipes — a MEMORY feature)**: graph-untouched
  recipe design (pack stores a recompute recipe for SDPA's saved q_embed/k_embed;
  unpack rebuilds bit-identically from the norm's offloaded x). Same-day pairs:
  time-neutral everywhere (+0.36%/+1.5%/+0.7%, noise), **C −18 GB @128k, −9 dense,
  −4.6 @32k**. Adoption: ON where C binds — dense-class + MoE ≥524,288 tokens/call
  (`ASYM_POLICY_ROPE_MIN_TOKENS`); OFF at 32k. Historical note (superseded design):
  in the qwen3 graph cos/sin are frozen, so the rope muls save no operand-shaped tensors
  (measured: smoke tag census shows only bf16 SDPA operands, which are distinct rope
  outputs, not inputs); the variant can only ADD saved bytes here. Kept for graphs whose
  rope operands ARE saved (trainable rotary scales).
- **Flag:** `ASYMM_QKNORM_RECOMPUTE`; policy rule `P12.qknorm_recompute` = **DEFAULT-ON
  (2026-07-21)** — gates: unit 9/9, SMOKE, e2e 128k −3.57% (C −63.7 GB), 32k −2.95%,
  dense −4.16%. Env flag still force-arms when the policy is off.
- **R5 restage-gap counters** (`restage_gap` block in the profile): CUDA-timing-event
  measure of the GPU-EXPOSED H2D restage wait at every restage site (compute-stream
  arrival mark vs side-stream copy-done — per scheduler.md's "never attribute from
  wall-time alone" law). Kill-switch `ASYM_RESTAGE_GAP_COUNTERS=0`.

## P15 — moe_direct_reuse (byte-diet): permanently False
Direct reuse of GPU-born gate/up/act in place of their offload→restage roundtrips
(flag `ASYMM_MOE_FG_DIRECT_REUSE`, per-mechanism G-guard). Gated 2026-07-21/22:
128k pair 600.71→601.91 NEUTRAL with guard_denied 953/960 — at G≈176/184 the holds
(12.6–25.2 GiB) never fit, and restage exposure is unchanged (237.1 s/5-step both
arms). 32k truly inert (0 attempts). Rule returns False; the guard auto-engages the
dormant mechanisms only on configs with real headroom (e.g. 2-GPU sharding). Do not
loosen min-free below 8 GiB at 128k: free dips to ~0.2 GiB mid-backward (OOM'd a run).

## P13. Restage prefetch/reuse (G-headroom-gated latency recovery)
- **Rule:** where HBM headroom exists, (a) issue H2D restages EARLY on a SECOND
  dedicated prefetch stream with per-tensor events (begin/commit — the consumer waits
  its own event at use time, not at stage time); (b) reuse ONE gate stage in the
  silu-bwd blocks (out-of-place silu; bitwise-identical dgrads); (c) dense: keep
  dgate/dup ON-GPU (skip their offload→restage roundtrip); (d) gc boundary: TRUE
  one-layer-ahead prefetch (boundary tensors exist from forward; LIFO registry).
  Guard: `prefetch_free_ok` (`ASYM_PREFETCH_MIN_FREE_GB`, default 16) — every
  mechanism holds stage memory earlier/longer and must fit the G ceiling.
- **Basis (R5 counters + chain11 same-day pairs):** exposed restage wait 30B@32k
  4.20→**1.82 s/step** ⇒ e2e **88.11→85.57 s (−2.9%, new 32k best)**; 30B@128k
  guard no-ops (G≈180/186) ⇒ honest NEUTRAL (602.5 vs 602.1 — the 49.6 s/step
  residual is bandwidth/regime cost, not schedulable latency); dense 32B −1.1%
  PARTIAL (guard demand was over-sized 4×, flapped at free≈64–78 GiB — tuned to 2×,
  re-gate b32_r5b). The saved-tensor class is prefetchable only from region exit
  (born during backward-recompute ⇒ one-layer-ahead impossible by construction).
- **Flag:** `ASYMM_ATTN_RESTAGE_PREFETCH` (alias `ASYM_RESTAGE_PREFETCH`); policy rule
  `P13.restage_prefetch` = **DEFAULT-ON (2026-07-21)** — gates: units 34/34, SMOKE,
  32k −2.9% (85.57 best), dense −11.9% (guard-tuned), 128k neutral-never-hurts.
- **2026-07-20 update:** dense guard-tuned re-gate (b32_r5b) PASSED big: 373.39 →
  **328.97 s (−11.9%)**, C −15.8 GB — the three-cell case for default-on is complete
  (32k −2.9%, dense −11.9%, 128k neutral-never-hurts by guard).

## P14. Kernel thread placement (2026-07-21 sweep — measured, 5-run medians)
- **Rule:** BANDWIDTH-bound CPU kernels (SwiGLU fwd/bwd, rmsnorm, widen+sqsum) stay
  SINGLE-SOCKET (48T; fwd@≥4M rows: 72T) — cross-socket interleave LOSES (238–243 vs
  287 GB/s socket-local; NUMA-remote caps streaming). COMPUTE-bound kernels (grouped
  LoRA wgrad) SCALE ACROSS SOCKETS: 1.96 TF/s @48T → 2.63 @72T → **2.85 @96T spread
  threads/data-home (the production layout)** → 3.24 @144T with interleaved operands.
- Mechanism: `ASYM_CPU_OPS_THREADS_WGRAD` (per-kernel override, default = global
  `ASYM_CPU_OPS_THREADS`); wired at all three deposit sites. E2e gate = same-day 32k
  pair (deposits are off-critical-path ⇒ expected neutral-or-better; the micro win
  widens the deposit-window headroom for longer contexts).
- P3-implication (two-socket question CLOSED): the 128k placement-flip case holds for
  the WGRAD half only; the CPU-act flip would additionally need per-socket data
  placement of the act pools (not built; recorded).

## P10. Production placement sets (acceptance targets for the policy module)
- **MoE @32k-class:** P1 + P2 + P3 + P4/P5/P6 + guards → **90.69 s** reference.
- **MoE @128k-class:** P2 + P4/P5/P6; P1 auto-off (rows), P3 OFF (rows) → **613.5 s** ref.
- **Dense 32B-class:** P4/P5/P6 only (per P8) → **≈352.3 s** reference.
- **Batch-scaling note (item 7, 2026-07-17, same-chain @30B·32k):** the production
  set is CALIBRATED AT b8. Growing batch trips the policy's own rows-gates and
  MONOTONICALLY sheds the wins: b8 (P1+P3 on) 2824.7 tok/s → b16 (P1 on, P3 auto-off at
  512k>262,144) 2824.7 tok/s (exactly neutral) → b24 (P1+P3 both auto-off at 6.144M>4.2M)
  2475.6 tok/s (−12.4%). Freed HBM (G 54→103→137 of 186) does NOT convert to throughput
  at 32k. The freed-HBM lever is the SEQ CEILING (128k, G≈180/186), not batch. If batch
  scaling is ever wanted, P1/P3 thresholds must be RE-MEASURED at b16/b24 shapes.
- The policy module MUST reproduce these sets exactly from its rules before it may
  become the default (fix_cpu_compute.md item 1 acceptance gate).
- **GATED 2026-07-16 (item 1 PASS)**: `ASYM_PLACEMENT_POLICY=1` alone materializes all
  three sets from the rules — dry-run harness asserts them exactly
  (tests/test_placement_policy.py), and same-day e2e: 30B@32k policy 90.81 s vs manual
  90.73 s (both reproduce the 90.69 reference; decisions P1×240/P2×480/P3×960 exact);
  30B@128k policy 616.5 s vs manual 628.0 s (P1/P3 auto-off by rows — the manual
  config split is automated; no regression); 32B dense smoke: model_class=dense →
  P8 kills every CPU-compute placement, P4/P6 stay on (zero engagement markers).
  The per-feature env flags remain the manual mechanism when the policy is OFF.

## P11. Decision tracing (metrics — required in every run)
- Every placement decision logged ONCE per run: rule id (P1–P8, P12), inputs (rows,
  bytes, context length, model class), decision, thresholds → `runtime_counters.json`
  under `placement_policy` + one human-readable line in train.log.
- Counters that must stay wired: worker-job wall-ms per job kind; deposit counts +
  retained-bytes high-water; per-feature ENGAGED markers; pinned-pool high-water.
- Any threshold change ⇒ update THIS file and re-run the affected P10 acceptance set.
- **IMPLEMENTED 2026-07-16 (item 1)**: `asym_gemm/training/placement_policy.py` —
  per-(rule,decision) dedup'd train.log lines with inputs+thresholds, decision COUNTS
  in the sidecar `placement_policy.json` (next to source_profile.json) and in the
  profile's `placement_policy` block → runtime_counters.json; plus P7
  `deposit_retention` high-water, `save_dedup` hit/miss/bytes, and the item-4
  `pinned_ledger` per-family live/high-water/denials (with
  torch host-allocator stats and /proc/meminfo Mlocked — NOTE: cudaHostAlloc pinning
  is INVISIBLE to Mlocked on this kernel; the ledger is the reliable tracker).

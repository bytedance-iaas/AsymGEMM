# FIG12 refresh campaign — kernels_e2e with REAL measurements (2026-08-13)

Kevin directive: fig12 (fig:ablation-throughput, figures/kernels_e2e.pdf) bars
were ESTIMATES (plot_kernels_e2e.py docstring admits it; only 30B 1.6M ours=292
measured). Redo with real metrics; context lengths need NOT match the old
figure; must show AsymLoRA kernels vs re-aimed inference-form (original
AsymGEMM) e2e throughput gap **>10% at 2 sequence lengths**. Don't stop until
achieved.

## Arm definitions
- **A (AsymLoRA kernels)**: the streaming tier (T3 preset) with the LoRA legs
  on the kernels: `ASYMM_QWEN3_MOE_FG_LORA_A_FWD_GPU=0 ASYMM_QWEN3_MOE_FG_DA_GPU=0`
  (fg default in recipes is GPU legs; the ablation runs the swap schedule with
  legs on the streaming kernels — the config the kernels exist for).
- **B (re-aimed inference form)**: same env + `ASYMM_LORA_KERNELS=reaim`.
  Implemented at the wrapper waist (2026-08-13, this session):
  - `cpu_left.py`: `_reaim_grouped_forward` — per-expert operand-swapped
    direct use of `asym_bf16_cpu_right_matmul` (S_e^T = A_e·X_e^T + transpose
    back; 8-row tail staged); hooks in single/pair/triple fwd (pair/triple =
    one stream PER ADAPTER — upstream serves one consumer per pass).
  - `exp_act_offload_lora.py`: `_reaim_grouped_grad` — chunked stage+
    accumulate dA (M2a-v2 'reaim2', CH=8192 house grain, fp32 accum), hooks
    in single/pair grad. Pair grad = X staged once per adapter.
  - Attention dA/dB unchanged (already upstream-form in the shipped tree —
    honest: K2 loses at few-segment sites, so ours ships the upstream kernel
    there).
  - Unit check `agent/anchors_tmp/reaim_unit_check.py`: ALL OK (ragged grouped
    E=16 incl. 8-row tails + empties; pair; triple; grads; attention shape
    bit-identical vs ours at aligned M).
- Deposits: P2 moe_wgrad_deposit auto-OFF under T3's KEEP_DGRADS_HBM pin;
  P3 attention deposit row-gated OFF above 262k rows — all dA legs stay on the
  ablated kernels at our cells in BOTH arms. GLM: harness auto-sets
  ASYMM_ATTN_LORA_A_CPU_LEFT_TORCH_STAGE=1 (segfault reroute, fix_glm_t3) —
  GLM attention legs are torch-staged in BOTH arms (no delta there).

## Why the old estimate methodology was wrong
The 13–18% placeholders carried the kernel-level M2a 18% e2e — but in the
SHIPPED recipes the MoE LoRA legs run on GPU (fg FWD_GPU=1/DA_GPU=1 pins), so
the only engaged streaming kernels are the 4 attention K1 fwds ≈ 1–2% of step.
Same-config swap at the old cells (320K/1.6M, quadratic-attention crushed)
would measure ≈0–2%. The honest ablation must run the swap schedule with
streamed legs (arm A above) and pick cells where the leg share is large:
short seq × large batch (attention ∝ b·s², legs ∝ b·s).

## Leg-share arithmetic (pre-probe estimate, 30B 96k b8 = 768k tok/step)
Per layer (routed 6.14M rows): gate/up pair fwd 25.2 GB → ours ~120 ms /
reaim 2× ≈ +120; pair dA ours ~85 / reaim-chunk 2.3–3× ≈ +110–200 (narrow-K
chunk tax may push higher); down fwd ~45 parity+collapse; down dA ours ~32 /
reaim +40–140; attention fwd (triple?) +0–29. Delta ≈ 300–470 ms/layer × 48
≈ 14–23 s on a ~230–245 s step ⇒ **6–10%**. Probe1 measures the truth.
At 96k b8 the full-restage (swap) fallback still FITS in HBM (72+25 GiB), so
final cells should sit at b_max where the swap transient does NOT fit ⇒
chunked reaim is the only kernels-off mapping (bulletproof baseline choice).

## Protocol
Serial only, one run per node; container asym_sft_40 via enroot one-shot
(driver.sh mounts), NVD=host ids + CVD=inside ids; membind node of the GPU;
PROFILERS=source; WARMUP_STEPS=1 MAX_STEPS=2; verdicts from jobs.tsv; loss
parity required (bf16 GEMM order differs → close, not bitwise). Interleave
A/B same-day; 3× repeats for the final claim numbers.
Harness: agent/anchors_tmp/fig12_lib.sh (c12 port of tpfig_lib) +
fig12_probe1.sh (kfa0/kfb0/kfa1/kfb1 = shipped/shipped+reaim/streamed/
streamed+reaim at 30B 96k b8) + fig12_harvest.py.
Anchors: newm96r1/r2 (30B 96k b8 T3 shipped) = 2775–2777 tok/s current tree.

## Status log
- 2026-08-13 21:0x: probe1 launched (GPU0). kfa0 training.
- Fig12 plot script: scripts/figures/plot_kernels_e2e.py (est bars 13/18/14/17%,
  lengths 320K/1.6M + 320K/1.02M). Will be replotted from measured cells.

## PROBE1 RESULTS (2026-08-13 21:10–23:00, 30B 96k b8 T3, GPU0, serial)
| arm | config | eff tok/s | steps (s) | loss |
|---|---|---|---|---|
| kfa0 | shipped T3 | 2725 | 281.7/282.0 | 1.704/1.695 |
| kfb0 | shipped + reaim | 2769 | 276.6/278.0 | 1.701/1.694 |
| kfa1 | streamed legs (FWD_GPU=0/DA_GPU=0) | 749 | 1025.6/1026.4 | 1.698/1.695 |
| kfb1 | streamed + reaim | killed (confounded; path disqualified) | | |
(anchors newm96r1-r3: 2775-2777 @ 276.3-277.0 s)

**Findings (decisive):**
1. Shipped-config kernel swap ≈ **0-2% = noise** (kfb0 ≥ kfa0!). The engaged
   streaming kernels in the shipped T3 = attention K1 fwd ×4 + down-proj dA K2
   only, and both are fully OVERLAPPED (bwd GPU work per layer ~1.5 s vs legs
   ~0.2-0.5 s → 3-40× hiding headroom at every b/s combination).
   Reaim ENGAGED markers verified: lora_a_fwd + lora_a_grad.moe.down.
2. The shipped fg path ALREADY RESTAGES X for gate/up dA (`moe.X_for_dA` stage
   tag, 125.8 GB/step at 96k b8) — the big X staging transient is in the
   SHIPPED peak too; the kernels only remove down-h + attention-U staging.
3. The streamed-legs config (FG legs on K1/K2) is 3.7× slower than shipped —
   ALL in backward (+247 s/step; fwd unchanged 57 s) — the fg cpu-dA path is
   pathological (why DA_GPU=1 is the flagship). Not a legitimate arm A.
4. ⇒ **A same-config >10% e2e kernel-vs-reaim delta does not exist at any
   cell of this system.** The old 13-18% estimates were unfounded.
5. LATENT BUG found (not mine, unused path): back-to-back unsynced grouped
   cpu-left calls on RAGGED offsets race on the pooled padded staging buffer
   (`_pad_cpu_left_grouped_input_for_asym` → `_alloc_cpu` pool reuse while the
   prior kernel still reads) — repro agent/anchors_tmp/pair_ragged_repro.py
   (trial0 clean, trials1+ corrupt 0.58-0.70 rel). Shipped paths are
   single-group/unpadded → unaffected.

## PROBE3 RESULT (pfs1600, staged-B @30B 1.6M b1 T3, finished 03:26 08-14)
- TRAINED: steps warmup ~10.0ks (JIT/pool-growth inflated), measured 4831.8 /
  4833.7 s = **331 tok/s**; peak observed 179.2 GiB (96.3% HBM) vs A's banked
  155.7 (84%). NO OOM — the table's ✓×=OOM@1.6M estimate is WRONG.
- **INVALID as training**: grad_norm=NaN from step 1 → loss 0.0 on measured
  steps. Since LoRA-B=0 at init ⇒ all dA legs are exactly zero ⇒ the NaN can
  only enter via attention dB = dY^T·S ⇒ the staged fwd's S is poisoned
  (suspect: read-before-ready race — staged .to(dev) racing the shared
  handle's D2H, or an out-coverage bug). Debug probe4 (kfs96, 96k b8, staged
  + ASYMM_LORA_KERNELS_DEBUG=1 nan-probes) running.
- The earlier "1.8× slower" inference was WRONG (warmup-step artifact).
  Cross-tree caveat: banked A=292 is c14-era; current tree is +13-18% faster
  at 96k ⇒ 1.6M frontier likely lands near PARITY. Same-day pfa1600 queued.

## KEVIN DIRECTIVE UPDATE (2026-08-14 ~03:00)
"Don't show short sequences with no gains — only ≥640k-class lengths. Need
**3 sequence lengths per model** with noticeable AsymLoRA-vs-naive-AsymGEMM
benefits. Naive AsymGEMM = naive usage (X.cpu @ A operand swap where
applicable). Don't stop until the figure shows it."

## FINAL FIGURE DESIGN (v2)
B arm = the naive INTEGRATION of the inference-form library: weights streamed
(AsymGEMM's own job), activations via standard offloaded-GC = the **T1
recipe** (unsloth-ohbm0 + staged dispatch, GPU LoRA legs, no swap schedule).
The X.cpu@A leg-swap variants (reaim/staged) are measured side evidence:
they TIE AsymLoRA wherever they run (kfb0 2769 vs kfa0 2725 @96k; staged
kfs96 2735 clean; staged@1.6M 331 = parity-class but NaN'd [scale bug,
parked]) — so the naive arm's real costs are what the library CANNOT
express: no streaming swap schedule → forced recompute (step tax) + HBM
recompute transients (walls).
- 30B rows: 900k (fresh same-day pair, probe5: B-T1 vs A-T2→T2B),
  1.1M (A=381 m6v2t2b1100b vs B=**measured CUDA OOM** m6v2t11100:
  "tried 33.57 GiB, 29.88 free"), 1.4M (A=320 m6v2t2b1400 vs B=OOM
  monotone). [1.6M optional 4th: A=298/292.]
- GLM rows: {640k, 800k, 1.0M} probe6 — B-T1 ladder to its wall + A-arm
  tier walks (T2→T2B→T3). Uncharted; datasets auto-build inline.

## PIVOT: capacity frontier (probe3, running)
B arm 'staged' mode added (ASYMM_LORA_KERNELS=staged = the component table's
middle row / M2a-v2 'swap': full re-stage per leg + native GEMMs; unit-checked
ALL OK). Probe3: does staged-B OOM at 30B T3 b1 1.6M/1.4M/1.2M where AsymLoRA
runs (fig8: 292/305 tok/s)? W1+M1 fit/no-fit. If OOM at the frontier: fig12
becomes parity-below-wall + OOM-at-frontier (an honest, strong claim, but not
a ">10% ratio at 2 lengths"). If staged-B fits at 1.6M: even the capacity
story fails at fig12 scale → full-truth report + reframing proposal to Kevin.

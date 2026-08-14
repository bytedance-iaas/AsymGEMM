# FIG12 refresh — measured verdict and options (2026-08-14, DRAFT pending pfs1600)

## What was asked
Replace fig12's estimated bars (13-18% e2e, admitted placeholders in
plot_kernels_e2e.py) with REAL measurements showing AsymLoRA kernels vs
re-aimed original-AsymGEMM kernels with >10% e2e throughput gap at 2 lengths.

## What the measurements say (all same-day, serial, house protocol, c12/GPU0)

### 1. The same-config swap is FREE (0-2%, = noise)
30B 96k×b8 T3 (768k tok/step; anchors 2775-2777):
- kernels (kfa0): 2725 tok/s — fwd 19.1 s, bwd 72.6 s /step
- re-aimed (kfb0): 2769 tok/s — fwd 19.1 s, bwd 71.1 s /step
- loss parity; reaim engagement verified ([asym-reaim] ENGAGED markers).
Mechanism: in the SHIPPED recipes the only streaming kernels on the hot path
are attention LoRA-A fwd (×4/layer) and down-proj dA; their link time
(~0.2-0.5 s/layer even re-aimed) sits under ~1.5 s/layer of concurrent GPU
work — 3-40× overlap headroom at EVERY batch/length (both scale ∝ b·s).
The shipped fg path already restages routed X for gate/up dA
(moe.X_for_dA), so those legs are GPU GEMMs in BOTH arms.

### 2. Putting all MoE legs on the kernels is NOT the system
FWD_GPU=0/DA_GPU=0 (kfa1): 749 tok/s = 3.7× slower than shipped, all in
backward (+247 s/step; fwd identical). The fg cpu-dA path is pathological —
this is WHY the flagship pins are GPU legs. Not a legitimate arm A.

### 3. Capacity frontier: staged kernels-off arm at 1.6M
pfs1600 (ASYMM_LORA_KERNELS=staged = the component table's middle-row
'swap-backs', unit-verified): [VERDICT PENDING — fits ~86% HBM through step-1
backward at 2 h elapsed]. If TRAINED: the table's ✓× = OOM @1.6M estimate is
FALSIFIED — the kernels-off transients (down-h ~19.7 GB, attn-U 6.6 GB) fit
inside the 1.6M headroom because the big X restage is already in the shipped
peak.

## Conclusion
**An honest >10% e2e throughput gap between the AsymLoRA kernels and the
re-aimed inference-form kernels does not exist at any cell of this system.**
The kernel swap is throughput-neutral wherever both forms fit (measured), and
the capacity channel at fig12's lengths is [pending pfs1600: likely too small
to move walls]. The old 13-18% figures were estimate artifacts carrying a
leg-level ratio (M2a's 2-3×) to e2e, which overlap erases.

## Options for the figure (Kevin's call)
A. **Parity + memory framing (recommended)**: fig12 shows measured bars
   A-vs-B at 2 lengths ×2 models (parity, Δ≤2%) with the measured peak-HBM
   delta annotated; claim = "the kernels return the offloaded bytes and the
   staging transients WITHOUT costing throughput"; pairs with tab.11; prose
   drops the 13-18% claim; the speed story stays at the leg level (fig. M2a,
   real 2-3×/2.0×/2.28×).
B. **Drop fig12**; fold the kernel story into M2a + the memory table
   (corrected per pfs1600).
C. **Reframe as system-vs-system** (AsymLoRA vs naive AsymGEMM training
   integration = unsloth-style T1): >10% exists only past T1's walls, where
   the comparison collapses into fig8's existing OOM story (redundant).

## Paper-integrity flags (must fix regardless)
- fig12 caption/prose "13-18% e2e" — unsupported, remove.
- tab:ablation-components ✓× OOM @1.6M — [pending pfs1600; likely to correct]
- plot_kernels_e2e.py placeholder data — replace with measured JSON.
- Latent race (unused ragged grouped cpu-left path, pooled pad buffer):
  agent/anchors_tmp/pair_ragged_repro.py.

Can CODA (arXiv 2605.19269, "Rewriting Transformer Blocks as GEMM-Epilogue Programs"; Guo, Dao et al.) help AsymGEMM-inspired LoRA SFT? **Verdict: no for latency / core, yes as the memory follow-on, plus one real linear-fusion touchpoint.**

## What CODA is
- CUTLASS/CuTeDSL kernel abstraction: **fixed GEMM mainloop + programmable epilogue** (`EpilogueVisitorTree`: `consumer_visit`, `producer_tma_load`, …). Mutates the output tile **while it's on-chip, before write-out**.
- Fuses RMSNorm / SwiGLU / residual / bias / row-col reductions, in **fwd and bwd**. Target: **Hopper, dense, HBM weights, no MoE, no quant**.
- Premise: load bandwidth is cheap; the cost is HBM write-back of intermediates.

## Why it does NOT fit AsymGEMM's kernel
- **Tile not resident.** CODA assumes the full output tile is complete on-chip at epilogue time. AsymGEMM's weight-stationary **K-outer/M-inner** schedule streams partials to HBM via `REDUCE_ADD` (`sm100_bf16_asym_gemm.cuh:983–1004`) so each fetched B-tile is reused across all tokens → no complete tile for any **nonlinear** epilogue.
- **Inverted bottleneck.** CODA assumes loads are cheap. AsymGEMM is **load-bound on the C2C weight fetch**; CODA's `producer_tma_load` of epilogue aux-inputs would *contend* with the weight fetch.
- ⇒ SwiGLU / RMSNorm / quant fusion are **side add-ons** here, not core. (Epilogue hook exists but unused: `csrc/jit_kernels/impls/epilogue.hpp:12` → `EpilogueIdentity`; dead `epilogue_type_t` in fp8/fp4 impls.)

## Latency axis — CODA does NOT help the core
- Step is bound by the **C2C weight fetch** (paid in fwd `x@W` *and* bwd `dy@W`) + **CPU-Adam** (~994 ms, overlapping). Backward ≈ 2× forward.
- CODA is post-MMA / output-side → cannot make the fetch cheaper, fewer, or faster, and cannot touch the optimizer. **Out of scope for latency.**

## Memory axis — CODA IS the natural follow-on
- AsymGEMM moves weights to CPU DRAM + keeps optimizer tiny (adapters only) ⇒ **on-GPU memory is now activation-dominated** (~14.5 GB measured).
- That is exactly CODA's turf: epilogue fusion keeps intermediates on-chip and cuts the activation peak → also lets you **dial down activation recompute** (helps bwd indirectly).
- Clean composition: **AsymGEMM kills weight memory → CODA kills activation memory.**

## The one real synergy (core path)
- Fuse the **linear LoRA-B accumulation** `s@B·α + bias` into the asym base-GEMM epilogue (lorafusion already does this for HBM weights: `lorafusion/ops/triton_ops/fused_lora_xw_sb.py`, `ops/lora_v1.py`).
- Linear ⇒ survives the K-outer / `REDUCE_ADD` schedule; runs in the shadow of the C2C stall ⇒ adapter math ≈ free on top of the base fetch.
- Caveat: approximable with concurrent streams ⇒ **real but modest**, not a breakthrough.

## What the actual core levers are (not CODA)
- Exploit that the **frozen weight is read-only & identical every step**: cross-layer/cross-step **weight prefetch + overlap**, **reuse** the fwd-fetched W in bwd before eviction, **hot/cold expert residency** (pin hot experts in HBM).
- Shrink the **3.3 B trainable LoRA surface** + faster CPU-Adam.

## Bottom line
- **Latency / core thesis: skip CODA.** Wrong axis, partly conflicts with the schedule.
- **Memory: adopt the principle** — it's the logical next wall after AsymGEMM. Plus the single linear LoRA-B epilogue fusion. Leave SwiGLU/RMSNorm/quant fusion alone.

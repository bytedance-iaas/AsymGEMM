# KT ARM BF16 SFT Fix Plan v14 — Remaining LOSSLESS + MEANINGFUL optimizations

**KT remains native ARM BF16 CPU code. DeepSpeed is not used.**

Scope: this file lists ONLY changes that are **(a) lossless** (no training-accuracy
impact — no new FP32->BF16 downcasts of currently-FP32 values, no weight
quantization) **and (b) meaningfully** help latency or memory. Trivial,
speculative, or lossy ideas are intentionally excluded (see bottom).

Repo map (these are 3 separate repos): **KT** = `../ktransformers` (the ARM BF16
MoE kernel + `kt-kernel/python/sft/*` glue); **LlamaFactory (LF)** =
`../LlamaFactory` (the training framework: trainer, optimizer, gradient
checkpointing); **AsymGEMM** = this repo's `asym_gemm/*` (a *different* backend,
NOT used by the `kt_armbf16` path). The KT training path = LF + KT kernel.

State after v13: avg_step ~155 s (fwd ~36 s, bwd ~116 s), steady RSS ~94.6 GiB,
~93% CPU-MoE-bound, GPU idle ~94%.

## 1. Eliminate the gradient-checkpoint recompute-FORWARD (pass #2) — biggest, lossless
The MoE forward is computed twice in the backward region: pass #2 = LF gradient
checkpointing re-running the whole decoder layer's forward (~34.6 s, the heavier
full-forward path incl. route packing), and pass #3 = KT's lean in-kernel recompute
(~16 s, eliminated by v13). They are the SAME redundant work from opposite ends.
**v13 removed the smaller one (pass #3) and kept the bigger one (pass #2).** The
larger win is to remove pass #2 instead and keep the lean in-kernel recompute.
- How: stop checkpointing the MoE (GPU has headroom, 37/47 GiB) so the MAIN forward
  saves the KT input cache (input-only, ~67.5 MiB x 48 = ~3.2 GiB) and PyTorch does
  not re-run the MoE forward in backward. Backward regenerates intermediates via the
  lean in-kernel recompute (pass #3).
- Where: KT `kt-kernel/python/sft/layer.py` (the `first_forward` cache-skip) + the
  LF checkpointing setting (`../LlamaFactory` trainer). Both are reachable; the
  layer.py part is inside the KT boundary.
- Lossless: uses saved/recomputed-once activations; identical values. Net vs v13:
  ~ -18 s additional (pass #2 34.6 s out, pass #3 16 s back in) => ~ -20% step.
- Interaction: MUTUALLY EXCLUSIVE with v13's intermediate cache (which needs pass #2
  to populate it per-layer). Adopting this REVERTS v13 to input-only cache + keeps
  the in-kernel recompute. Net it is still the bigger win.
- Effort: medium. Risk: low-medium (must confirm non-MoE activation memory fits
  without MoE checkpointing). Validate: short+full LF profile, loss parity, RSS.

## 2. Use the second CPU socket (multi-NUMA) — potentially largest, lossless, high effort
Host is 2x72-core Grace; the ARM MoE runs on ONE NUMA node (`threadpool_count==1`,
64 threads on subpool 0 — logs show `subpool_count=1`). The whole second socket
(72 cores + memory controllers) is idle. The MoE is memory-bandwidth-bound (why
v11/v13 helped), so adding the 2nd socket's aggregate bandwidth could cut the
CPU-bound compute by tens of %. Same math, more cores => lossless.
- Step 0 (cheap probe): `KT_NUM_THREADS` 64->72 (fills node 0) to gauge core-vs-BW
  scaling — launcher env only.
- Full: enable a multi-NUMA worker pool (the code in `kt-kernel/python/sft/arm.py`
  rejects `threadpool_count>1` today) + NUMA-aware weight placement. INTERLEAVE the
  weights across nodes (lossless, no extra RAM, ~2x aggregate BW); do NOT replicate
  (that would add ~58 GiB).
- Effort: high. Risk: high (NUMA correctness, cross-socket latency). All KT-side.

## 3. BFMMLA on the base GEMMs — ~6-10 s (~4-6%), lossless, medium effort
The base matmuls (`arm_bf16_grad_matmul_reg`: forward gate/up/down + backward
grad_act/grad_x) are MB=4 BFDOT. BFMMLA (`svbfmmla`) is the next tier — measured
**1.5x not 2x** on this core, net 1.33-1.5x after 2x2 packing. Same BF16 inputs +
FP32 accumulate as the current BFDOT => lossless (only negligible fp32 accumulation
-order differences, same class as the accepted v11 MB=4 reblock).
- Enabler: pre-pack the (stable, shared) transposed weights into the 2x2 tile layout
  once per layer in `transpose_base_weights`; pack activations per-tile (fuse into
  the existing bf16 staging). 8x4 micro-tile fits 32 Z-regs; gate on
  `__ARM_FEATURE_SVE_BF16`, keep a BFDOT fallback.
- Effort: medium. Risk: medium (new microkernel + 2x2 deinterleave store).

## 4. mmap the frozen base experts — memory -~54 GiB RSS, lossless accuracy
The 58 GiB base experts are frozen and identical on disk; mmap them read-only
instead of a private bf16 copy -> ~54 GiB moves from RSS into the page cache. Same
bf16 values => accuracy-lossless. Caveat: page-in stalls can add LATENCY (not
accuracy) — NUMA-place the mapping and validate no latency regression.
- Where: `kt-kernel/python/sft/arm.py` loader + C++ load path. Effort: medium.

## 5. CPU<->GPU backward overlap — ~6-9 s, lossless, capped, complex (lowest priority)
GPU is idle ~94%. Wire the existing unused `submit_backward_async`/`sync_backward`
(`kt-kernel/python/sft/base.py`, `arm.py`) so layer-N MoE backward overlaps
layer-(N-1) GPU attention backward. Same compute, overlapped => lossless. Capped at
the GPU's ~9.6 s budget (CPU pool already saturated) + needs autograd scheduling.
Do after #1/#2 (which change the CPU budget).

## Recommended order
2-step-0 (64->72 probe) -> 1 (pass #2, certain ~20%) -> 2-full (NUMA, biggest if it
scales) -> 3 (BFMMLA) -> 4 (mmap, memory) -> 5 (overlap). Each gated by the standard
short+full LF e2e profile with loss-parity and steady-RSS checks.

## Deliberately EXCLUDED (lossy or not meaningful)
- LoRA backward-grad FMLA->BFDOT: needs BF16 downcast of currently-FP32
  grad_y/grad_u/u/input -> LOSSY.
- int8/int4 base-expert quantization: LOSSY.
- Optimizer moments FP32->BF16/8-bit (-12..18 GiB): LOSSY (moment precision); it's
  an LF/torch optimizer-choice change, not KT.
- route_grad_x->BF16 (0.5 GiB), svaddv removal (~1%), MB=8 (register spill ->
  regresses), host-copy pinning (<1 s), forward CPU/GPU overlap (hollow on Qwen3):
  trivial / non-meaningful.
- Speculative FP32 grad-accum re-tiling: unquantified -> not a committed item.

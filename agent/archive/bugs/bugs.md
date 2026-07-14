# AsymGEMM + LlamaFactory LoRA-SFT — Comprehensive Code Review (Bugs / Findings)

> Generated 2026-05-31 by a 14-agent adversarial review (3-skeptic panels for high/critical correctness; refute-by-default). Static review (no GPU run) corroborated by the repo's own measured profiling artifacts and by hand-tracing the transpose_b dx math, the frozen-base autograd, the launcher gating, and the Qwen3 wrapper.

> **No source files were modified** to produce this report — it is read-only analysis. The pre-existing working-tree edits (`lf.py`, `frozen_linear.py`, `moe.py`, `qwen3_moe.py`, scripts/tests) predate this review.

> **Tally:** 53 confirmed findings · 1 uncertain (needs GPU run) · 9 refuted false-alarms (kept below with the reason they are NOT bugs).


---

## Executive verdict

| Question | Verdict |
|---|---|
| **1. AsymGEMM + transpose correct & efficient?** | **Correct** math/design (transpose_b dx is elegant & provably right), but **2 real kernel correctness bugs** (FP8 data race + bf16 accumulation precision loss) and **structurally under-optimized**: kernels are slower than torch, single-buffered, hard-tiled. |
| **2. LoRA-SFT correct & efficient?** | **Correct** (LoRA math, frozen autograd, MoE recompute, Qwen3 parity all verified). It is a **memory optimization (saves 29–41% HBM), NOT a speed one** — ~1.2–1.6× slower e2e, with per-layer host syncs and an fp8/fp4 HBM-staging bug. |
| **3. Integrated like ktransformers?** | **Yes — complete & faithfully mirrors `use_kt`.** Integration logic is sound. Real gap is **test/CI coverage**: CUDA kernels never run in CI, and there is no automated asym-vs-HF loss-parity check. |

## Fix priority (highest first)

1. 🔴 Add missing `__syncthreads()` in the FP8 `k>0` full-tile path — it is a data race (K3).
2. 🔴 FP32 K-tile partial-sum accumulation — bf16 RMW loses precision on every multi-K-tile GEMM (K1).
3. 🟡 Add the dense `k%64` dx alignment guard — silently-wrong-gradient risk (K5).
4. 🔴 Wire `compare_lf_smoke_losses.py` into a run/CI gate + add a GPU CI job — otherwise none of the above is caught automatically (I3).
5. 🔴 Default transpose `block_m=128` + per-shape block-tile autotuning — the biggest measured latency win (K6/L5).
6. Cache the device-resident quantized weight instead of staging per-call (L1); remove route-validation `.item()` syncs on the hot path (L3/L4); gate the FP4 `fprintf`s (K6).
7. 🟡 Launcher: auto-export `USE_ASYM_GEMM` from the flag, or add a parser guard forbidding DDP under asym (I2).

---

## Q1 — AsymGEMM kernels & transpose variants — correctness, efficiency, improvements


### [K1] BF16 asym GEMM device kernels (SM90/SM100)

- **Correctness assessment:** Mostly correct in control flow (K-outer/M-inner loop, asymScheduler M bounds, sentinel/masked skip, ragged-M via host BLOCK_M padding, first-K-tile-write/rest-read-modify-write), but has one materially important numerical-accuracy defect: cross-K-tile partial sums are accumulated in the output buffer's native dtype, which is bf16 by default, via SM90_TMA_REDUCE_ADD. When K spans more than one BLOCK_K tile the running sum is repeatedly rounded to bf16 in HBM, giving a lossier result than FP32 accumulation. This is reachable in the default forward (K-major, block_k=512) and especially the backward dx path (MN-major, block_k=256, k>=768 -> up to 16 tiles).
- **Efficiency assessment:** The central design goal (load each weight K-tile once into a single smem slot and reuse it across the entire inner M loop) is correctly realized, eliminating redundant B fetches and atomic contention. The cost is heavy HBM partial-sum traffic: each (m,n) output tile is written once and then read-modified-written once per extra K-tile, so with K=4096/block_k=512 the output tile touches HBM ~15x. For memory-bound CPU-resident-weight GEMM this staging traffic is significant and is the main efficiency liability.
- **Bottom line:** The K-outer/M-inner accumulation, scheduler M-iteration, TMA producer/consumer pipeline, and ragged-M/sentinel handling are correct. The one defensible correctness issue is bf16-precision accumulation of K-tile partial sums in HBM (REDUCE_ADD into a bf16 D buffer) whenever K > BLOCK_K, which degrades numerical accuracy versus FP32 accumulation. Efficiency is dominated by the HBM read-modify-write partial-sum traffic, which is inherent to the chosen staging scheme.

#### K1.1. 🔴 HIGH · correctness · K-tile partial sums accumulated in bf16 HBM (REDUCE_ADD into bf16 D) lose precision when K > BLOCK_K

- **Location:** `asym_gemm/include/asym_gemm/impls/sm90_bf16_asym_gemm.cuh:429-435`
- **Confidence:** 0.78 · **Verification:** adversarial panel real=3 refuted=0

**Problem (context + error):** For block_k_iter==0 the epilogue does SM90_TMA_STORE_2D (overwrite); for every later K-tile it does SM90_TMA_REDUCE_ADD_2D into the SAME (m_idx,n_idx). The accumulation buffer is D whose dtype is cd_dtype_t = d.scalar_type(). The default output dtype is torch.bfloat16 (frozen_linear.py _asym_bf16_nt output_dtype=torch.bfloat16, d=torch.empty), so each cross-K-tile partial sum is rounded to bf16 and added in bf16 in global memory. This is independent of kWithAccumulation: the K-tile RMW is hardcoded, while the FP32 guard at line 64-65 only fires when kWithAccumulation is true. With block_k=512 and K=4096 the forward does ~7 bf16 RMW adds; the transpose/backward path (sm100 block_k=256, k>=768) does up to 15. The numerical result therefore differs from FP32-accumulated x@W^T by an error that grows with the number of K-tiles.

**Evidence:** sm90 line 429-435: `if (block_k_iter == 0) { cute::SM90_TMA_STORE_2D::copy(...); } else { cute::SM90_TMA_REDUCE_ADD_2D::copy(...); }`. The static guard only covers kWithAccumulation: `if constexpr (kWithAccumulation) DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>...)` (line 64-65), but the K-tile RMW always runs. Host allows bf16 D: gemm.hpp:527 `DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat)`; default caller uses bf16 (frozen_linear.py:516,526).

**Fix:** Accumulate K-tile partial sums in FP32: either force the D/CD TMA buffer to FP32 whenever block_k > 1 (i.e. ceil_div(shape_k,BLOCK_K) > 1) and convert to bf16 only after the final K-tile, or keep a dedicated FP32 partial-sum buffer for the RMW and write bf16 only on the last K-tile. At minimum, add a DG_STATIC_ASSERT/host check that forbids bf16 cd_dtype_t when more than one K-tile is used.

#### K1.2. 🟡 MEDIUM · efficiency · HBM read-modify-write partial-sum traffic scales with number of K-tiles

- **Location:** `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh:982-1004`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** Because the accumulator is staged to HBM between K-tiles, every (m_tile,n_tile) output is written once (k=0 STORE) and then read+written once per additional K-tile (REDUCE_ADD). For K=4096 with block_k=512 (8 tiles) each output element incurs ~15 HBM accesses (1 write + 7 read-modify-writes) before it is final. For a memory-bound, CPU-resident-weight GEMM this staging traffic competes with the (intended) one-shot weight fetch for HBM bandwidth and can dominate when M*N is large. The smaller backward block_k=256 doubles the tile count and the staging traffic.

**Evidence:** sm100 line 984-1004: k=0 uses SM90_TMA_STORE_2D, else SM90_TMA_REDUCE_ADD_2D to the same n_idx/m_idx; same pattern sm90:429-435. block_k = ceil_div(shape_k, BLOCK_K) (sm100:229) with host block_k=512 forward / 256 transpose (sm100_bf16_asym_gemm.hpp:139-142).

**Fix:** Where M fits in on-chip/accumulator capacity, keep the accumulator resident across K-tiles instead of round-tripping HBM each K-tile; alternatively increase BLOCK_K to reduce tile count, or split-K with a single final reduction. Quantify against the profiling traces before changing, but the RMW staging is the dominant avoidable traffic.

#### K1.3. ⚪ LOW · correctness · asymScheduler.current_shape_k is read but never initialized (dead but latent)

- **Location:** `asym_gemm/include/asym_gemm/common/asymScheduler.cuh:75`
- **Confidence:** 0.9 · **Verification:** single verifier

**Problem (context + error):** asymScheduler declares current_shape_k (line 75) but the constructor (lines 79-109) never assigns it for any GemmType. Both asym kernels compute `num_total_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K)` (sm100:277 and sm100:450) from this uninitialized field. Today this is harmless because num_total_k_blocks is dead — the actual K loop bound is `block_k = ceil_div_device(shape_k, BLOCK_K)` (sm100:229). But the uninitialized read is a UB/latent bug: any future use of num_total_k_blocks would consume garbage, and it is easy to mistake the variable for a live bound during maintenance.

**Evidence:** asymScheduler.cuh:75 `uint32_t current_shape_k, ...` is never written in the constructor (compare Scheduler in scheduler.cuh:76 `current_shape_k = shape_k;`). sm100:277 `const auto& num_total_k_blocks = ceil_div_device(scheduler.current_shape_k, BLOCK_K);` and sm100:450 same; grep confirms num_total_k_blocks has no other use.

**Fix:** Either initialize current_shape_k = shape_k in the asymScheduler constructor (matching Scheduler), or delete the unused num_total_k_blocks computations in both kernels to remove the uninitialized read.

#### K1.4. ⚪ LOW · design · SM100 TMA-load warp mutates n_idx inside the inner M loop (only safe because asym is multicast-on-A)

- **Location:** `asym_gemm/include/asym_gemm/impls/sm100_bf16_asym_gemm.cuh:292-295`
- **Confidence:** 0.62 · **Verification:** single verifier

**Problem (context + error):** Inside the per-M-tile TMA-A issue block, when kNumMulticast>1 the code does `n_idx += kIsMulticastOnA ? 0 : (block_rank * LOAD_BLOCK_N)`. n_idx is a function-scope variable (line 231) reused as the B/output N base; adding into it inside the M loop would corrupt subsequent iterations if the increment were ever nonzero. It is only safe because asym GEMM is configured kIsMulticastOnA=true (so the increment is 0) and/or kNumMulticast==1. This is a fragile invariant: a future config with B-side multicast would accumulate n_idx across M iterations and produce wrong B/output offsets.

**Evidence:** sm100:231 `uint32_t n_idx = scheduler.n_idx;` then inside `for (block_m_iter...)` at sm100:292-295 `if constexpr (kNumMulticast > 1) { m_idx += ...; n_idx += kIsMulticastOnA ? 0 : (cute::block_rank_in_cluster() * LOAD_BLOCK_N); }`. The SM90 path correctly comments that n_idx is const for the block (sm90:220).

**Fix:** Compute the multicast N offset into a local (e.g. n_idx_mc = n_idx + ...) rather than mutating the loop-invariant n_idx, and/or static_assert kIsMulticastOnA when kGemmType is the asym grouped type, to make the invariant explicit and crash-safe.


### [K2] FP8 / FP4 asym GEMM kernels + scale application

- **Correctness assessment:** Correct, with one minor robustness caveat. The 1d1d scale application is sound across all four kernels: SM100 FP8/FP4 use hardware block-scaled UMMA where UE8M0/UE4M3 scales are staged into TMEM and consumed by the MMA, with per-K-block partial results scaled before TMA-REDUCE_ADD into HBM (so each K-block's contribution is scaled before summation — no single-scale-over-whole-accumulator bug). The sf_id pack-cycling math (FP8: k_block_idx%4; FP4: pack-per-64-K) correctly indexes the 4 packed scales, the FP4 E2M1 nibble packing (even-K low nibble, odd-K high nibble) matches host quantization, K//2 packed strides are consistent, and SM90 FP8 applies per-token x per-channel float scales per K-block then REDUCE_ADDs. The UE8M0 host cast rounds the exponent UP (ceil of log2), which is the overflow-safe direction. No wrong-result, race, or mis-pack issue found in the scale path.
- **Efficiency assessment:** Reasonable but with one structural latency exposure and no FP8/FP4 measured evidence. B is single-buffered by design (fetched once per K-tile, reused across all M-tiles) which is the intended K-outer reuse, but it serializes the next K-block's B/SFB TMA load behind the full M-sweep consuming the current B (no B prefetch overlap). MMA shapes (FP8 UMMA_K=32, FP4 UMMA_K=64) and SF granularities match the hardware block-scaled instructions. All profiling artifacts under profiling*/ are BF16-only, so no measured FP8/FP4 efficiency data exists to confirm occupancy or fetch-bound behavior.
- **Bottom line:** The FP8 and FP4 asymmetric GEMM kernels are numerically correct in their scale application, UE8M0/UE4M3 handling, and FP4 packing: each K-block is scaled before REDUCE_ADD accumulation, sf_id pack indexing is right, and nibble order matches the host quantizer. The main caveats are efficiency-structural (single-buffered B exposes per-K-block fetch latency) and a precision-coverage limitation (FP4 NVFP4 path uses E4M3 block scales with no per-tensor second-level FP32 scale). No correctness-breaking bug was found in the reviewed scale/pack/accumulation logic.

#### K2.1. ⚪ LOW · improvement · FP4 NVFP4 path omits the per-tensor second-level FP32 scale, capping dynamic range

- **Location:** `asym_gemm/utils/math.py:100-116`
- **Confidence:** 0.7 · **Verification:** single verifier

**Problem (context + error):** per_token_cast_to_nvfp4_e4m3 produces only E4M3 per-16-element block scales (sf = (amax/6.0).to(e4m3)) with no global per-tensor FP32 scale-of-scales, and the FP4 SM100 kernel consumes only those E4M3 scales from TMEM (no FP32 multiply in the epilogue). Standard NVFP4 uses a two-level scheme: per-block E4M3 scales plus a per-tensor FP32 scale, so that the E4M3 scale itself does not have to span the full tensor dynamic range. Without the second level, tensors whose block-amax values span a wide range lose precision because the E4M3 scale (4-bit exponent) must absorb the entire range. This is self-consistent (dequant is correct), so it is a precision-coverage limitation, not a wrong-result bug.

**Evidence:** math.py:109 `sf = (x_amax / 6.0).to(torch.float8_e4m3fn)` then math.py:112 `x_scaled = x_view * (1.0 / sf_decoded.unsqueeze(2))` — the scale is a single E4M3 value per 16-element block with no FP32 global factor returned or applied. The FP4 kernel (sm100_fp4_asym_gemm_1d1d.cuh:374-376) builds the instr desc with cutlass::float_ue4m3_t scales only and the epilogue (lines 637-668) does a plain TMEM->bf16 cast with no scale multiply.

**Fix:** If FP4 accuracy on wide-dynamic-range weights/activations is a concern, add a per-tensor FP32 global scale (fold it into the bf16 epilogue cast, or pre-scale A) to realize the full NVFP4 two-level scheme. Otherwise document that this is intentionally a single-level (per-16 E4M3) NVFP4 variant.

#### K2.2. ⚪ LOW · design · Dead per-lane B-descriptor staging in MMA warp (only stage 0 / index 0 ever used for B)

- **Location:** `asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh:356,429`
- **Confidence:** 0.7 · **Verification:** single verifier

**Problem (context + error):** b_desc_lo is computed per-lane with a stage stride (lane_idx * SMEM_B_SIZE_PER_STAGE/16) as if B were multi-staged, but B is single-buffered and the MMA always shuffles lane 0 (`__shfl_sync(..., b_desc_lo, 0)`). The per-lane staged B descriptor computation is dead code carried over from the A path and can mislead a reader into thinking B is multi-buffered, obscuring the single-slot serialization analyzed above. Same pattern in the FP4 kernel (lines 396, 464).

**Evidence:** Line 356 `uint32_t b_desc_lo = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;` then line 429 `const auto& b_desc_base_lo = __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(0));` — only lane 0's value (stride 0) is ever read, so the per-lane stride is computed and discarded.

**Fix:** Compute b_desc_lo as a plain scalar from smem_b[0] (no per-lane stage stride, no shuffle) to make the single-buffered-B nature explicit and remove the misleading staged-descriptor code.


### [K3] SM80/SM89 MoE GEMM (PCIe cp.async path)

- **Correctness assessment:** One genuine data race: the SM89 FP8 contiguous and masked kernels omit a block-level __syncthreads() between the cp.async load of sX and the warp-cooperative LDSM read of sX, in the full-tile (m_actual==BLOCK_M) k>0 branch only. cp_async_wait<0> guarantees per-thread completion, not cross-thread smem visibility, and LDSM reads bytes written by other threads — so this is undefined behaviour. It is exercised by the existing multi-K-tile tests (K>=512 with token counts that are multiples of 128) and is a classic latent race that can pass on hardware by luck. Every other load path in these kernels (k=0, partial-tile k>0, and the entire BF16 kernel) does insert the sync, which confirms the omission is a bug, not a deliberate optimisation. Aside from that, the index math, expert/offset handling, per-token vs per-tensor scale selection and application, partial-tile predication, FP32-accumulate/BF16-partial-sum read-modify-write, and contiguous-vs-masked addressing all check out.
- **Efficiency assessment:** The BF16 sm80_moe_gemm_impl uses an M-outer/K-inner loop and re-fetches the entire W tile for EVERY M-tile (M_tiles x K_tiles fetches instead of K_tiles), directly contradicting the project's stated 'each weight K-tile fetched ONCE, reused across M' core idea — though BF16 requires b in HBM (b.is_cuda() asserted) so L2 absorbs much of it. The FP8 SM89 kernels correctly implement K-outer/M-inner W-reuse. Both BF16 and FP8 paths are single-buffered: each cp.async is immediately followed by cp_async_wait<0> + __syncthreads(), so PCIe/HBM fetch latency is fully exposed with no double/triple buffering to hide it — the dominant efficiency limiter for the on-demand-weight premise.
- **Bottom line:** The FP8 SM89 path is the real PCIe-fetch path (BF16 SM80 asserts b.is_cuda(), so it is not actually a CPU-pinned path). The FP8 kernels are largely correct, but the full-tile k>0 branch is missing a __syncthreads() before the sX LDSM in both the contiguous and masked variants — a real cross-thread smem race. Efficiency is limited by single-buffered cp.async (no latency hiding) in all variants and by redundant per-M-tile W refetch in the BF16 kernel (which abandons the documented K-outer reuse). Scale selection/application and predication are correct.

#### K3.1. 🔴 HIGH · correctness · Missing __syncthreads() before sX LDSM in FP8 full-tile k>0 path (cross-thread smem race)

- **Location:** `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh:714-721,750-753`
- **Confidence:** 0.82 · **Verification:** adversarial panel real=3 refuted=0

**Problem (context + error):** In sm89_moe_fp8_gemm_impl, the k>0 loop's full-tile branch (m_actual==BLOCK_M) issues the sX cp.async copy, fences and calls cp_async_wait<0>(), then seeds tSrO from global O — but it never calls __syncthreads() before the warp-cooperative LDSM of sX at lines 750-751. cp_async_wait<0> only guarantees the CALLING thread's own async copies have completed and are visible to that thread; it provides no cross-thread/block visibility. SM75_U32x4_LDSM_N is a warp-cooperative load in which each thread reads smem bytes that were written via cp.async by a DIFFERENT thread (the cp.async tiling Layout<(32,4),(1,16)> and the LDSM thread->address map differ). Without __syncthreads() a thread may LDSM-read sX bytes whose cp.async write (issued by another thread) has not yet landed — a read-after-write race / UB producing stale or partial X data in the MMA. Trigger: any expert whose token count for some M-tile equals BLOCK_M (=128) on a shape with k_max>1 (e.g. test cases K=512/4096/7168 with token counts 128,256,512). The k=0 path (line 668) and the partial-tile k>0 path (line 738) DO sync, and the BF16 kernel syncs at line 346 — proving the omission is an oversight.

**Evidence:** Lines 714-721: `cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX); cp_async_fence(); cp_async_wait<0>(); Tensor tSgO = thr_mma.partition_C(gO_m); ... tSrO(i) = static_cast<float>(tSgO(i));` then directly at 750-751 `cute::copy(smem_copy_A, tSsX, tSrX_view);` with NO __syncthreads() in between. Contrast k=0 path line 666-668 `cp_async_fence(); cp_async_wait<0>(); __syncthreads();` and BF16 line 343-346 which both insert the barrier.

**Fix:** Insert a __syncthreads() in the full-tile branch after the cp_async_wait<0>()/tSrO-seed block (after line 721) and before the LDSM at 750, mirroring the k=0 path. The global-O seed read at 718-721 does not need the barrier, but the sX LDSM does.

#### K3.2. 🔴 HIGH · correctness · Same missing sX-LDSM barrier in the masked FP8 kernel full-tile k>0 path

- **Location:** `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh:1048-1055,1083-1086`
- **Confidence:** 0.82 · **Verification:** adversarial panel real=3 refuted=0

**Problem (context + error):** sm89_moe_fp8_gemm_masked_impl repeats the identical structure: in the k>0 loop, the full-tile branch (m_actual==BLOCK_M) at lines 1048-1055 does cp.async sX, fence, cp_async_wait<0>(), seeds tSrO from global O, then falls through to the LDSM at 1083-1084 with no __syncthreads() guarding the sX smem read. Same cross-thread RAW race as the contiguous kernel. Triggered by any masked group whose valid row count masked_m makes a full BLOCK_M tile on a multi-K-tile shape.

**Evidence:** Lines 1048-1055 `cute::copy(gmem_tiled_copy_xw, tXgX_mk, tXsX); cp_async_fence(); cp_async_wait<0>(); Tensor tSgO = thr_mma.partition_C(gO_m); ... tSrO(i) = static_cast<float>(tSgO(i));` then 1083-1084 `cute::copy(smem_copy_A, tSsX, tSrX_view);` with no barrier. The partial-tile branch at 1072 does `__syncthreads();` before its seed read; the full-tile branch does not.

**Fix:** Add __syncthreads() after the full-tile seed block (after line 1055) and before the LDSM at 1083, mirroring the partial-tile branch and the k=0 path (line 1006).

#### K3.3. 🔴 HIGH · efficiency · BF16 sm80 kernel re-fetches the entire W tile per M-tile (abandons K-outer weight reuse)

- **Location:** `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh:307-359`
- **Confidence:** 0.85 · **Verification:** single verifier

**Problem (context + error):** The BF16 sm80_moe_gemm_impl uses an M-outer (line 307), K-inner (line 324) loop and loads W for tile (n_tile,k) inside the K-loop at lines 326-329. Because the K-loop is nested inside the M-loop, W is fetched M_tiles x K_tiles times per CTA instead of the minimal K_tiles. This directly contradicts the project's stated core idea ('each weight K-tile fetched ONCE and reused across all token M-tiles') and contrasts with the FP8 kernels, which correctly do K-outer/M-inner reuse (load W once per k-tile at lines 640-641 / 695-696, then sweep all M). For an expert with M_tiles M-tiles the redundant W traffic is (M_tiles-1)x. The BF16 API asserts b.is_cuda() (gemm.hpp:699) so W is in HBM and L2 absorbs much of the reuse, limiting the practical penalty — but the kernel is structurally unable to use CPU-pinned PCIe weights efficiently and is inconsistent with the documented architecture.

**Evidence:** Loop nest: `for (int m = 0; m < M_tiles; ++m) { ... for (int k = 0; k < K_tiles; ++k) { Tensor tWgW_k = gmem_thr_copy_xw.partition_S(gW(_, _, k)); cute::copy(gmem_tiled_copy_xw, tWgW_k, tWsW); cp_async_fence(); ...` (lines 307,324,326,329). vs FP8 line 640-641 loading W once outside the M-loop.

**Fix:** Restructure the BF16 kernel to K-outer/M-inner with HBM partial-sum read-modify-write (as the FP8 kernel already does), so each W k-tile is fetched once and reused across all M-tiles. If BF16 is only ever used with HBM weights this is lower priority, but the inconsistency should at least be documented.

#### K3.4. 🟡 MEDIUM · efficiency · Single-buffered cp.async fully exposes PCIe/HBM fetch latency (no pipelining)

- **Location:** `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh:330-346,641-668,696-717`
- **Confidence:** 0.7 · **Verification:** single verifier

**Problem (context + error):** Every cp.async load in all three kernels is immediately followed by cp_async_fence()+cp_async_wait<0>()+__syncthreads(), i.e. the kernel waits for the copy to fully complete before computing on it. There is exactly one smem stage for sX and one for sW (SMEM_X_ELEMS / SMEM_W_ELEMS, no [STAGES] dimension), so no double/triple buffering exists to overlap the next tile's fetch with the current tile's MMA. For the central premise (fetching CPU-pinned weights over PCIe, hundreds of ns to us latency), this means PCIe latency is serially exposed on the W path (FP8: one W fetch per k-tile that cannot overlap the M-sweep's compute start; BF16: one W+X fetch per (m,k)). cp_async_wait<0> waits for ALL committed groups, so even the W and X copies that are committed as separate groups (e.g. BF16 lines 330,343) are not staged to overlap each other's latency with compute.

**Evidence:** BF16: `cute::copy(...tWsW); cp_async_fence(); ... cute::copy(...tXsX); cp_async_fence(); cp_async_wait<0>(); __syncthreads();` (330-346). FP8 k>0: `cute::copy(...tWsW); cp_async_fence(); cp_async_wait<0>(); __syncthreads();` (696-699) and per-m `cute::copy(...tXsX); cp_async_fence(); cp_async_wait<0>();` (715-717). Smem layouts SmemLayoutX/W (441-444) have no stage dimension.

**Fix:** Introduce 2-3 smem stages for sW (and sX) and pipeline: prefetch tile k+1's W while MMA-ing tile k, using cp_async_wait<N-1> to keep N copies in flight. This is the primary lever to hide PCIe latency that the on-demand-weight design depends on; the current code cannot hide it.

#### K3.5. ⚪ LOW · efficiency · Per-M-tile global read-modify-write of BF16 partial sums adds O(M_tiles*K_tiles) HBM O traffic

- **Location:** `asym_gemm/include/asym_gemm/impls/sm80_moe_gemm.cuh:693-768`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** In the FP8 K-outer scheme, for k>0 every (m) iteration reads the prior BF16 partial sum back from global O (line 718-721 full / 728-744 partial) and write_output writes it back, for all k-tiles. This is the documented HBM read-modify-write trade for weight reuse, but it converts the accumulator into a BF16 round-trip per K-tile: total O traffic is ~2 * M_total * N * (k_max-1) BF16 elements, and the intermediate sums are stored in BF16 (not FP32), so each K-tile boundary loses precision via FP32->BF16->FP32 round-tripping. For large k_max (e.g. K=7168 -> 28 K-tiles) this both adds bandwidth and accumulates rounding error across 27 BF16 truncations. The error is likely within FP8-GEMM tolerance but is a real, quantifiable accuracy/bandwidth cost worth noting.

**Evidence:** `tSrO(i) = static_cast<float>(tSgO(i));` (721) seeds from BF16 global, and write_output stores `ElementOut(static_cast<float>(...)*cs)` (566-567) / moe_convert_type<ElementOut> (764) back to BF16 every k-tile. SMEM_O is BF16 (SmemLayoutO at 446-447).

**Fix:** If accuracy matters, consider an FP32 partial-sum HBM buffer for the read-modify-write (at 2x O bandwidth) or, better, keep more of the K dimension resident per M-sweep so fewer round-trips occur. At minimum document the BF16-partial-sum precision trade.

#### K3.6. ⚪ LOW · correctness · FP8 path passes b.data_ptr() raw for CPU-pinned weights; relies on UVA device-accessibility (unverified at API boundary)

- **Location:** `csrc/jit_kernels/impls/sm80_moe_gemm.hpp:153,155 and csrc/apis/gemm.hpp:761-763`
- **Confidence:** 0.55 · **Verification:** single verifier

**Problem (context + error):** The FP8 entrypoints allow b to be a CPU-pinned tensor (no b.is_cuda() assert; comment 'b may be on CPU pinned memory (PCIe)') and pass b.data_ptr() straight into params.w_ptr, which the kernel cp.async-reads as a global address. This is correct ONLY if the pinned allocation is device-accessible under UVA at the same virtual address (true for cudaHostAlloc/torch pin_memory on UVA platforms). There is no assertion that b is actually pinned (vs pageable CPU) nor any cudaHostGetDevicePointer fallback. A pageable CPU tensor passed here would yield a host pointer the device cannot dereference -> illegal address at runtime, with no early/clear host-side check. The BF16 path is stricter (asserts b.is_cuda()).

**Evidence:** gemm.hpp:762 `DG_HOST_ASSERT(a.is_cuda() && d.is_cuda());` (note: no b.is_cuda() and no b.is_pinned() check). sm80_moe_gemm.hpp:155 `.w_ptr = b.data_ptr(),` passed directly; kernel casts to const ElementIn* and cp.async-reads it.

**Fix:** Add a host-side guard: DG_HOST_ASSERT(b.is_cuda() || b.is_pinned()) in the FP8 entrypoints so a pageable-CPU b fails fast with a clear message instead of an opaque device illegal-address fault.


### [K4] asymScheduler / schedulers / TMA + common utils

- **Correctness assessment:** Largely correct, with one host-coupled risk. The CTA->(n_tile, expert) mapping, M-range derivation from offset pairs, masked-m skip, and B/A index math in asymScheduler are internally consistent and produce no double-counted or skipped tokens PROVIDED two host-side invariants hold: (a) per-expert offsets fed to the contiguous path are aligned to the kernel's BLOCK_M (satisfied by _pad_grouped_input_for_asym, which pads to a hardcoded 128), and (b) the launch sets gridDim.y to the number of ACTIVE experts (list_size-1), excluding the -1 sentinel. The scheduler itself has NO defensive guard for the sentinel or for a BLOCK_M mismatch, so correctness is delegated to launch/host code I could not see (compiled binding). TMA descriptor setup proper (encode) is host-side and not in this source tree; the device-side tma_copy / grid_constant tensormap usage is correct, and the asym kernels correctly do NOT use the device-side tensormap.replace helpers (those are only for the legacy non-asym FP8 path).
- **Efficiency assessment:** Reasonable but with a real, deliberate load-imbalance tradeoff. The static one-CTA-per-(N-tile, expert) assignment eliminates atomic contention (the stated design goal) but provides no work-stealing: with experts of very unequal token counts, CTAs owning small experts finish early and idle while CTAs on large experts serialize through many M-tiles, creating a tail-latency wave-quantization effect. The K-outer/M-inner reuse of each B tile is sound and is the main efficiency win. The masked path correctly skips zero-token groups with an immediate TMEM-freeing early-exit.
- **Bottom line:** The asymScheduler bookkeeping is correct on its own terms and consistent with the host padding/sentinel conventions, with no internal double-count or token-skip bug. The main caveats are external couplings the scheduler does not defend against: the -1 sentinel must be excluded from gridDim.y (no in-kernel guard) and per-expert offsets must be BLOCK_M(=128)-aligned (hardcoded host-side). One benign uninitialized read (current_shape_k) exists in the BF16 path, and the static expert-to-CTA mapping is a known tail-latency tradeoff.

#### K4.1. 🟡 MEDIUM · efficiency · Static one-CTA-per-(N-tile, expert) assignment causes load imbalance / tail latency across experts of unequal token counts

- **Location:** `asym_gemm/include/asym_gemm/common/asymScheduler.cuh:79-108`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** The asymScheduler assigns exactly one CTA to each (N-tile, expert) and that CTA iterates the full M extent [m_start, m_end) of its expert (see the M loops in sm100_bf16_asym_gemm.cuh:278 and :451). Unlike DeepGEMM's persistent Scheduler::get_next_block (scheduler.cuh:159) which walks a global block index across all work, here there is no work-stealing or persistence across experts. With skewed MoE routing (one hot expert with many tokens, several cold experts), CTAs owning cold experts finish after 1 M-tile and idle, while CTAs on the hot expert serialize through many M-tiles. Effective occupancy is bounded by the largest expert's M extent, not by total work / num_SMs. This is the deliberate price for eliminating atomic contention, but it is a genuine tail-latency mechanism (wave quantization on the M axis) for unbalanced expert distributions.

**Evidence:** asymScheduler.cuh:88-108 derives a single (m_start,m_end,n_idx,expert_id) per CTA from blockIdx.{x,y} with no outer loop over groups and no get_next_block-style global rebalancing; the kernel then loops `for (block_m_iter = scheduler.m_start; block_m_iter < scheduler.m_end; ...)` (sm100_bf16_asym_gemm.cuh:278) for the entire expert. Contrast scheduler.cuh:159 get_next_block which distributes blocks round-robin across kNumSMs.

**Fix:** If profiling on skewed routing shows idle SMs, consider a 1-D persistent block index over the flattened (active-expert, n-tile, m-tile) space (DeepGEMM-style) so finished CTAs pick up remaining M-tiles of hot experts, trading some atomicity/RMW complexity for better balance. At minimum, document the imbalance characteristic so callers can pad/cap per-expert token counts.

#### K4.2. ⚪ LOW · correctness · Hardcoded block_m=128 in host padding must match the kernel's actual BLOCK_M or offsets become misaligned

- **Location:** `asym_gemm/training/frozen_linear.py:444-474`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** _pad_grouped_input_for_asym pads each expert's token count up to a multiple of a hardcoded block_m=128 and produces padded_offsets that are multiples of 128. The asym kernel then derives m_start/m_end via ceil_div(offset, BLOCK_M) assuming offsets are exact multiples of the COMPILED BLOCK_M. If the JIT autotuner ever selects BLOCK_M != 128 for the BF16 contiguous kernel, ceil_div on a 128-aligned-but-not-BLOCK_M-aligned boundary would round the start UP (skipping the first tokens of the expert) or leave a partial trailing tile, producing wrong/missing rows. Today grouped-contiguous uses BLOCK_M=128 throughout (matches DeepGEMM's get_m_alignment_for_contiguous_layout and all tests), so this is latent, not active.

**Evidence:** Line 449: `block_m: int = 128`; line 462: `padded = ((count + block_m - 1) // block_m) * block_m`. Kernel side derives m_start = ceil_div_device(offsets[2y], BLOCK_M) (asymScheduler.cuh:101) where BLOCK_M is the compiled template constant, with no runtime check that 128 == BLOCK_M.

**Fix:** Either query the actual selected BLOCK_M from the kernel config and pass it into _pad_grouped_input_for_asym, or add a static/runtime assertion that the contiguous BF16 BLOCK_M is fixed at 128, to prevent a silent regression if the autotuner space is widened.

#### K4.3. ⚪ LOW · correctness · Uninitialized read of scheduler.current_shape_k in the BF16 asym kernel (UB, but result is dead)

- **Location:** `asym_gemm/include/asym_gemm/common/asymScheduler.cuh:75`
- **Confidence:** 0.85 · **Verification:** single verifier

**Problem (context + error):** asymScheduler declares `uint32_t current_shape_k, current_num_valid_groups = 0, ...` — only the members after current_shape_k get default initializers; current_shape_k is never assigned in either constructor branch. The BF16 kernel then reads it at sm100_bf16_asym_gemm.cuh:277 and :450 (`num_total_k_blocks = ceil_div_device(scheduler.current_shape_k, BLOCK_K)`). This is a read of an uninitialized uint32_t (technically UB). It is benign in practice because num_total_k_blocks is computed as `const auto&` and never used — the real K loop bound is `block_k = ceil_div_device(shape_k, BLOCK_K)` (line 229). The FP8/FP4 kernels correctly use shape_k directly and do not touch current_shape_k.

**Evidence:** asymScheduler.cuh:75 `uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0, ...` (current_shape_k has no initializer and the constructor never sets it). sm100_bf16_asym_gemm.cuh:277 `const auto& num_total_k_blocks = ceil_div_device(scheduler.current_shape_k, BLOCK_K);` and :450 same — both results unused.

**Fix:** Initialize current_shape_k = 0 in the declaration (or in the constructor), and delete the two dead `num_total_k_blocks` lines in the BF16 kernel to remove the UB and the confusion.

#### K4.4. ⚪ LOW · design · Dead n_start/n_end scheduler fields and duplicated IndexType/get_num_1d_blocks_per_group across two headers

- **Location:** `asym_gemm/include/asym_gemm/common/asymScheduler.cuh:60,96 and :8-35`
- **Confidence:** 0.85 · **Verification:** single verifier

**Problem (context + error):** asymScheduler sets n_start (line 96) and declares n_end (line 60) but no kernel reads them (grep over impls/*.cuh shows only sm80 unrelated matches), so they are dead state. Separately, both asymScheduler.cuh and scheduler.cuh define `enum class IndexType` and `get_num_1d_blocks_per_group` in the SAME namespace asym_gemm with identical bodies. No current TU includes both, so there is no ODR violation today, but if a future file ever includes both headers it will fail to compile with redefinition errors (pragma once does not help across distinct files). This is a maintainability landmine.

**Evidence:** asymScheduler.cuh:60 `uint32_t n_start, n_end;`, :96 `n_start = expert_id * blocks_perExpert;` (never read). Duplicated definitions: asymScheduler.cuh:8-12 `enum class IndexType {MN,K,SF_K};` and :23-35 get_num_1d_blocks_per_group, identical to scheduler.cuh:12-16 and :18-30, both in `namespace asym_gemm`.

**Fix:** Remove the unused n_start/n_end. Factor IndexType and get_num_1d_blocks_per_group into a shared header included by both schedulers (or guard against co-inclusion) to avoid a latent redefinition error.


### [K5] Transpose variant (transpose_b) + layout/einsum (LoRA dx path)

- **Correctness assessment:** The BF16 transpose_b path that computes the LoRA data-gradient dx = grad @ W is mathematically and dimensionally correct. I traced the index/stride algebra end-to-end: forward y = x@W^T uses the native nt contract D[M,N]=A[M,K]@B[N,K]^T; backward passes the SAME physical weight W=[N,K] with transpose_b=True, which swaps n<->k (logical n=k_phys=K, logical k=n_phys=N), sets major_b=MN, and recomputes b_outer_stride=b.stride(-2)=K. make_tma_b_desc(MN,...) then reads inner=K (stride 1) / outer=N (stride K), so the kernel reads B_logical[j,l]=W[l,j] and computes D[m,j]=sum_l grad[m,l]*W[l,j]=grad@W. Multi-expert group stride (N*K) is encoded correctly. There is ONE real gap: the dense (non-grouped) BF16 dx path is missing the k%64 contraction-alignment guard that the grouped path has. The masked BF16 wrapper has no transpose_b parameter (masked dx unsupported), which is fine since dx uses contiguous/dense layouts.
- **Efficiency assessment:** The BF16 transpose_b path is efficient: it is a pure stride/major reinterpretation with NO physical transpose or extra copy of the CPU-resident weight (confirmed by code and by the absence of any transpose copy in profiling_latest dx ranges). The quantized (fp8/fp4) dx path is the exception: it cannot reinterpret strides because per-block scales live along the K axis, so _get_quantized_host_weight(transpose=True) materializes a fully transposed + re-quantized copy of the weight, roughly DOUBLING the pinned-CPU footprint for any layer used in both forward and backward. This is cached (one-time) and is a defensible tradeoff, not a correctness bug. The layout/einsum/clean_logits kernels are not on the dx path at all.
- **Bottom line:** The core transpose_b dx path (dx = grad @ W) is mathematically correct and zero-copy for BF16, reusing the nt kernel via an MN-major reinterpretation of the frozen weight with no physical transpose. The one genuine issue is that the dense BF16 dx path omits the k%64 contraction-alignment guard that the grouped path explicitly added when it switched to transpose_b. Quantized dx necessarily re-quantizes a transposed weight copy (correct, but doubles pinned memory); the einsum/layout/clean_logits/k_grouped kernels are inherited from DeepGEMM and are not part of the LoRA dx path.

#### K5.1. 🟡 MEDIUM · correctness · Dense BF16 dx path is missing the k%64 contraction-alignment guard present on the grouped path

- **Location:** `asym_gemm/training/frozen_linear.py:282-287 (vs 329-330)`
- **Confidence:** 0.55 · **Verification:** single verifier

**Problem (context + error):** When transpose_b=True the GEMM contraction dim is k = n_phys = out_features (N). The grouped reason-checker enforces `if transpose_b and k % 64 != 0: return "transpose_b_requires_64_aligned_k"` (line 329), a guard that git history shows was added in the SAME commit (7e2d922) that switched grouped dx from a materialized transpose to the zero-copy transpose_b=True path. The dense checker `_direct_bf16_reason` (used by AsymFrozenLinearFunction.backward at frozen_linear.py:1085) only enforces `n%8` and `k%8` (lines 286-287) and has NO k%64 guard. There is also no host-side DG_HOST_ASSERT(k%64) in the C++ BF16 wrappers. So a dense frozen linear whose out_features is a multiple of 8 but not 64 (e.g. 1536/3072 are ok, but 1352, 1000, or any 8-aligned-only N) takes the transpose_b kernel with no guard; if the kernel mis-handles such k the way the grouped author evidently found, dx is silently wrong (no throw, no fallback for backend='asym').

**Evidence:** Grouped guard: `if transpose_b and k % 64 != 0: return "transpose_b_requires_64_aligned_k"`. Dense checker lacks it: it ends at `if n % 8 != 0 or k % 8 != 0: return "requires_8_aligned_nk"`. The commit that introduced the guard simultaneously changed grouped dx from `b_cpu = ctx.host_weight.transposed_tensor(); transpose_b = False` to `b_cpu = ctx.host_weight.weight; transpose_b = True`, indicating the 64-alignment requirement is intrinsic to the transpose_b kernel, not to the grouped layout.

**Fix:** Add the identical `if transpose_b and k % 64 != 0: return "transpose_b_requires_64_aligned_k"` to _direct_bf16_reason, OR add a host-side DG_HOST_ASSERT on the contraction alignment in m_grouped_bf16_asym_gemm_nt_contiguous when transpose_b is set, so an unaligned dense dx falls back to torch (auto) or raises (asym) instead of risking a silently wrong gradient. Confirm by running the dense backend='asym' dx with out_features that is 8- but not 64-aligned and diffing against torch a@W.

#### K5.2. 🟡 MEDIUM · efficiency · Quantized (fp8/fp4) dx materializes and re-quantizes a transposed weight copy, doubling pinned-CPU memory

- **Location:** `asym_gemm/training/frozen_linear.py:164-169, 216-240, 1072-1073/1248`
- **Confidence:** 0.85 · **Verification:** single verifier

**Problem (context + error):** Unlike BF16 (zero-copy stride reinterpretation), the quantized dx path calls `_get_quantized_host_weight(host_weight, precision, transpose=True)`, which runs `_transpose_source_for_quantization` -> `.t().contiguous()` (a full physical transpose of the weight) and then re-quantizes it, caching the result under key (precision, True) separately from the forward's (precision, False) entry. Any layer used in both forward and backward therefore holds TWO full quantized copies in pinned DRAM (values + scales each). This is required for correctness because per-block fp8/fp4 scales are computed along K, and transposing swaps which axis is blocked, so the scales genuinely must be recomputed along the new contraction axis - a stride trick is impossible. So this is correct but memory-expensive.

**Evidence:** `def _transpose_source_for_quantization(tensor): ... return tensor.transpose(-1, -2).contiguous()` then `_quantize_host_weight_tensor(source, ...)`; cache key is `(precision, bool(transpose))` so transpose=True is a distinct cached copy. backward calls it with transpose=True (lines 1073, 1248).

**Fix:** Document the ~2x pinned-memory cost for quantized adapters and consider (a) lazily building the transposed quantized copy only when dx is actually needed (it already is, via cache), and (b) optionally freeing the forward copy's redundant pinned buffer if memory-constrained. No code change needed for correctness; this is a design/efficiency note.

#### K5.3. ⚪ LOW · design · einsum (bmk_bnk_mn / bhr_hdr_bhd), smxx_clean_logits, and k_grouped_fp8 are inherited DeepGEMM code not used by the LoRA dx path

- **Location:** `csrc/apis/einsum.hpp:26-215; csrc/jit_kernels/impls/smxx_clean_logits.hpp:54-81; csrc/apis/gemm.hpp:153-208`
- **Confidence:** 0.9 · **Verification:** single verifier

**Problem (context + error):** The review brief asks whether these layout/einsum kernels and the k_grouped SM90-only wrapper are relevant to the dx/wgrad path. Grepping training/ and integrations/ shows none of `bmk_bnk_mn`, `einsum`, `fp8_einsum`, `smxx_clean_logits`, or `k_grouped_fp8_gemm_nt_contiguous` are referenced by the AsymGEMM training layer. smxx_clean_logits is an attention/MLA logit-masking kernel (writes -inf into [aligned_ks,ks) and [ke,aligned_ke) ranges via TMA bulk copy) entirely unrelated to GEMM. k_grouped_fp8's `DG_HOST_ASSERT(arch_major == 9)` is irrelevant because LoRA-SFT freezes the base weight (no wgrad); the only base backward is dx, which uses the m-grouped transpose_b kernels. These are dead-for-this-feature surface area.

**Evidence:** `grep transpose_b/k_grouped/einsum/clean_logits asym_gemm/training asym_gemm/integrations` returns no GEMM-einsum hits; backward() only calls `_dispatch_nt(..., transpose_b=True)` / `_dispatch_grouped_nt(..., transpose_b=True)`. gemm.hpp:203 `DG_HOST_ASSERT(arch_major == 9 and "k_grouped_fp8_gemm_nt_contiguous wrapper currently exposes the SM90 implementation")`.

**Fix:** No action required for correctness. If maintainability is a concern, mark these inherited DeepGEMM entrypoints as not-part-of-the-AsymGEMM-LoRA-surface in a comment so future readers don't assume the k_grouped SM90 limitation constrains the dx path.

#### K5.4. ⚪ LOW · correctness · Masked BF16 wrapper has no transpose_b parameter; masked-layout dx is unsupported (acceptable but undocumented)

- **Location:** `csrc/apis/gemm.hpp:620-663`
- **Confidence:** 0.8 · **Verification:** single verifier

**Problem (context + error):** m_grouped_bf16_asym_gemm_nt_masked computes major_b via get_major_type_ab(b) with no transpose_b path, unlike the two contiguous overloads. If a caller ever attempts the data-gradient through the MASKED layout it would silently compute the forward orientation (D=A@B^T) rather than grad@W. In the current training code dx always routes through the contiguous/dense overloads (transpose_b wired only there), so this is not hit today, but the asymmetry is a latent trap if MASKED dx is added later.

**Evidence:** The masked wrapper signature is `m_grouped_bf16_asym_gemm_nt_masked(a,b,d,masked_m,expected_m,compiled_dims)` with no `transpose_b`, and inside `const auto& major_b = get_major_type_ab(b);` with no MN-major override; the contiguous overloads at lines 501-563 and 565-618 both take `const bool transpose_b`.

**Fix:** Either add an explicit assert/comment that the masked BF16 wrapper does not support transpose_b, or plumb transpose_b through it for symmetry before any masked dx is wired up.


### [K6] Host launchers, tiling heuristics, JIT codegen

- **Correctness assessment:** Mostly correct. Argument validation in apis/gemm.hpp is thorough for BF16/FP8 (dtype/shape/contiguity/device, offsets/experts on CUDA, masked_m int32 on CUDA); A/D are required on CUDA and B may be CPU-pinned by design. grid_y derivation (num_groups for the tensor-list_size/capture path, list_size-1 for the int path) matches the asymScheduler device-side reads of experts[blockIdx.y] / offsets[blockIdx.y*2(+1)]. JIT codegen substitutes the right types/constants and the cache key hashes the full generated code (so transpose_b, block sizes, dtypes all produce distinct entries). SMEM budget is computed against the real hardware cap (232448 B) and stages are clamped to fit. The two genuine correctness gaps are minor/latent: (a) the SM100 BF16 launchers omit the .num_groups designated initializer so kNumGroups is baked as 0 into the template (currently harmless because the scheduler ignores kNumGroups, but inconsistent with the SM90 path and a foot-gun); (b) the FP4 contiguous dispatcher does NO validation of offsets/experts, and the int-list_size validation under-checks offsets.numel(). None corrupt results for the current matched callers.
- **Efficiency assessment:** Under-tuned for the asym (large-M weight-reuse) regime, especially the transpose-B / LoRA-backward path. Tiles are hard-coded (BF16 block_m=block_n=64; transpose default block_m=64) and stages are fixed at 2, bypassing the wave/occupancy search in get_best_config_asym. The project's own artifacts show this costs a lot: the transpose_block_sweep shows block_m=128 is ~1.6x faster than the default block_m=64 for the data-gradient GEMM, and lora_operator microbenchmarks show asym backward is ~8-15x slower than the torch baseline (forward ~4-7x). Separately, the FP4 contiguous launcher emits 19 unconditional fprintf(stderr) per launch. The end-to-end LoRA-SFT step is 0.63-0.85x of torch while saving 29-41% HBM — the HBM/latency tradeoff is real but the latency gap is larger than the tiling needs to allow.
- **Bottom line:** The launchers and heuristics are functionally correct for the wired-up callers and the JIT cache key is complete; grid/cluster/smem args are consistent with the device scheduler. The main issues are efficiency-driven: hard-coded tiles leave the transpose-B (LoRA-backward) path 1.6x+ slower than the project's own measured optimum, and there is leftover per-launch debug spam in the FP4 path. Correctness gaps (omitted kNumGroups in SM100 BF16, missing/weak offsets-experts validation in the FP4/int-list_size paths) are latent rather than live bugs.

#### K6.1. 🔴 HIGH · efficiency · Transpose-B (LoRA-backward dx) uses block_m=64 default; project's own sweep shows block_m=128 is ~1.6x faster

- **Location:** `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp:129-142`
- **Confidence:** 0.82 · **Verification:** single verifier

**Problem (context + error):** The SM100 BF16 contiguous launcher hard-codes the transpose-B tile via DG_BF16_TRANSPOSE_BLOCK_M defaulting to 64 and DG_BF16_TRANSPOSE_BLOCK_N=64, instead of running the wave/occupancy search. The transpose path is the LoRA data-gradient (dx = grad @ W via transpose_b=True), which is on the hot backward path. The repo's measured transpose_block_sweep directly contradicts the default: for every swept shape, block_m=128 beats block_m=64 by ~1.5-1.7x. End-to-end this shows up as asym backward being 8-15x slower than torch in the lora_operator microbench.

**Evidence:** Launcher: `const int block_m = (major_b == cute::UMMA::Major::MN) ? get_env<int>("DG_BF16_TRANSPOSE_BLOCK_M", 64) : get_env<int>("DG_BF16_BLOCK_M", 64);`. Measured (profiling/transpose_block_sweep, median ms, asym_bf16_transpose): t64 vs t128 -> m2048_k2048_n768: 0.0878 vs 0.0536 (1.64x); m2048_k1536_n4096: 0.3815 vs 0.2296 (1.66x); m2048_k768_n2048: 0.1958 vs 0.1115 (1.76x); m2048_k4096_n1536: 0.1586 vs 0.1243 (1.28x). lora_operator backward: asym 2.323 ms vs torch 0.156 ms (in=768,out=2048); 1.376 ms vs 0.163 ms (in=2048,out=768).

**Fix:** Set the transpose-B default block_m to 128 (the measured optimum) or, better, let get_best_config_asym pick block_m for the MN-major (transpose) case the same way it already evaluates block_n. Re-run the sweep to confirm on target shapes; keep the env override for tuning.

#### K6.2. 🟡 MEDIUM · efficiency · FP4 contiguous launcher emits 19 unconditional fprintf(stderr) on every launch

- **Location:** `csrc/jit_kernels/impls/sm100_fp4_asym_gemm_1d1d.hpp:91-223`
- **Confidence:** 0.95 · **Verification:** single verifier

**Problem (context + error):** sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d prints ~19 [FP4_LAUNCH] diagnostics to stderr on every invocation, none gated by an env var (unlike the DG_JIT_DEBUG-gated prints elsewhere). In training this dispatcher is called per-layer per-step, so this adds host-side formatting/IO overhead and floods stderr. The masked FP4 launcher has none of this, confirming it is leftover debug code.

**Evidence:** `fprintf(stderr, "[FP4_LAUNCH] Enter sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d: ...`; grep count = 19 fprintf(stderr ...) in this file, with no get_env/DG_JIT_DEBUG guard around them (only structural `if` statements at lines 101, 114, 188).

**Fix:** Delete the fprintf calls or gate them behind `if (get_env<int>("DG_JIT_DEBUG")) { ... }` like the rest of the codebase.

#### K6.3. 🟡 MEDIUM · correctness · FP4 contiguous dispatcher performs no validation of offsets/experts

- **Location:** `csrc/apis/gemm.hpp:378-435`
- **Confidence:** 0.9 · **Verification:** single verifier

**Problem (context + error):** m_grouped_fp4_asym_gemm_nt_contiguous never checks offsets/experts for is_cuda, is_contiguous, scalar_type==kInt, or numel, whereas the FP8 (lines 235-239) and BF16 (lines 531-535) contiguous dispatchers do. Its data_ptr<int>() is later taken and passed to the kernel as a raw device pointer. A caller that passes a CPU tensor, an int64 tensor, or a too-short tensor gets an illegal memory access / silently wrong indexing inside the scheduler (which reads offsets[blockIdx.y*2+1] and experts[blockIdx.y]) instead of a clean host-side assert.

**Evidence:** The function body between the shape checks and the SF transforms contains only `DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16); check_major_type_cd(d); if (m == 0) return;` — no `offsets.is_cuda()`, `experts.scalar_type() == torch::kInt`, or `numel >= ...` checks before `sm100_m_grouped_fp4_asym_gemm_contiguous_1d1d(... offsets, experts, list_size ...)`.

**Fix:** Add the same offsets/experts validation block used by the FP8/BF16 contiguous dispatchers (is_cuda, is_contiguous, scalar_type==kInt, numel bounds matching the pairs layout) to the FP4 contiguous dispatcher.

#### K6.4. ⚪ LOW · correctness · SM100 BF16 asym launchers omit .num_groups initializer; kNumGroups is baked as 0 into the JIT template

- **Location:** `csrc/jit_kernels/impls/sm100_bf16_asym_gemm.hpp:223-235`
- **Confidence:** 0.83 · **Verification:** single verifier

**Problem (context + error):** The SM100BF16AsymGemmRuntime::Args aggregate (and the masked variant at lines 419-431) is built with designated initializers that omit .num_groups, so it is value-initialized to 0. That 0 is substituted as kNumGroups (args.num_groups, used at generate_impl line 59) into the kernel template and the cache key. The SM90 path correctly sets `.num_groups = num_groups` (sm90_bf16_asym_gemm.hpp:136,199) and the SM100 FP8/FP4 paths set it too. It is currently benign because asymScheduler templates on kNumGroups but never uses it for MGroupedContiguous/MGroupedMasked (only BLOCK_M/N etc.), and no division by kNumGroups occurs in the default template args. It becomes a real bug the moment any SM100 BF16 kernel change starts using kNumGroups (e.g. for B-tensor batch indexing).

**Evidence:** Initializer: `const SM100BF16AsymGemmRuntime::Args& args = { .m = m, .n = n, .k = aligned_k, .compiled_dims = compiled_dims, .gemm_config = config, .launch_args = ..., .offsets = ..., ... };` — no `.num_groups`. Contrast sm90_bf16_asym_gemm.hpp:136 `.num_groups = num_groups,`. generate_impl substitutes `args.num_groups` as the kNumGroups template arg (line 59).

**Fix:** Add `.num_groups = num_groups,` to both SM100 BF16 Args initializers (the masked launcher already has num_groups in scope) to match the SM90/FP8/FP4 paths and remove the latent foot-gun.

#### K6.5. ⚪ LOW · correctness · int-list_size contiguous validation under-checks offsets.numel() vs the pairs layout the scheduler indexes

- **Location:** `csrc/apis/gemm.hpp:295`
- **Confidence:** 0.78 · **Verification:** single verifier

**Problem (context + error):** The int-list_size BF16 (line 594) and FP8 (line 295) contiguous dispatchers assert `offsets.numel() >= list_size`, but the scheduler treats offsets as a [start,end] pairs array, reading offsets[blockIdx.y*2+1] for blockIdx.y in [0, grid_y) with grid_y=list_size-1. The largest index accessed is 2*(list_size-1)-1, so the correct bound is offsets.numel() >= 2*(list_size-1). For list_size>=3 the existing check is weaker than required. It does not currently fire because the Python caller (_group_metadata_tensors) always builds pair_offsets of length 2*num_groups = 2*(list_size-1), so the access is in-bounds; this is a validation gap, not a live bug.

**Evidence:** C++: `DG_HOST_ASSERT(offsets.numel() >= list_size && experts.numel() >= list_size);` (gemm.hpp:295, 594). Scheduler: `uint32_t offset_pair_idx = blockIdx.y * 2; m_start = ceil_div_device(offsets[offset_pair_idx], BLOCK_M); m_end = ceil_div_device(offsets[offset_pair_idx + 1], BLOCK_M);` (asymScheduler.cuh:100-102) with grid_y=list_size-1 (gemm.hpp:304,602).

**Fix:** Change the offsets bound to `offsets.numel() >= 2 * (list_size - 1)` (or `>= 2 * num_groups`) to match the pairs layout the scheduler indexes, matching the tensor-list_size path which already checks `offsets.numel() >= 2 * num_groups` (line 239).

#### K6.6. ⚪ LOW · design · CPU-residency of B is assumed by the asym design but never enforced/asserted on the SM90/SM100 BF16/FP8 paths

- **Location:** `csrc/apis/gemm.hpp:269-318`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** The core premise is that B (the weight) lives in CPU-pinned DRAM and is fetched on-demand. The SM89 dispatchers document this in comments ('b may be on CPU pinned memory'), but the SM90/SM100 BF16 and FP8 dispatchers neither require nor forbid B's location and add no comment; the FP4 contiguous path only emits a runtime warning if B is not CUDA. Because B is fed through a TMA descriptor (make_tma_b_desc) built from b.data_ptr(), passing a host pointer where the driver cannot map it would fail at descriptor-encode or kernel time rather than at a clear host assert. Since B may legitimately be CPU-pinned, a hard is_cuda assert would be wrong; the gap is the absence of an explicit, documented contract / pinned-or-cuda check.

**Evidence:** SM89: `// b may be on CPU pinned memory (PCIe) or CUDA` plus `DG_HOST_ASSERT(a.is_cuda() && d.is_cuda());` (gemm.hpp:761-762) — note b is deliberately excluded. SM100/SM90 BF16/FP8 dispatchers assert b.is_contiguous() but never document or constrain b.device(); FP4 only warns: `[FP4_LAUNCH][WARN] Using CPU/pinned tensors for FP4 TMA path...` (sm100_fp4_asym_gemm_1d1d.hpp:102-104).

**Fix:** Add an explicit assert/comment per dispatcher that B must be either CUDA or CPU-pinned (e.g. DG_HOST_ASSERT(b.is_cuda() || b.is_pinned())) so a plain CPU tensor fails with a clear message instead of a TMA illegal-address, and document the CPU-resident contract uniformly.


---

## Q2 — LoRA-SFT modification — correctness & efficiency


### [L1] Frozen base + Asym execution core (frozen_linear.py)

- **Correctness assessment:** Mostly correct. The autograd math is sound: forward y=x@W^T via the asym GEMM, backward dx via the transpose_b asym GEMM, NO weight gradient for the frozen base, correct grad_bias (sum over M, cast to bias device/dtype), and HostWeight is data (not a Parameter) so it can never accumulate a grad or be migrated to HBM. Saved-for-backward state is minimal (no activations saved; only metadata on ctx). Two defensible defects: (1) the bf16_output_dtype=float32 contract is silently violated — the bf16 asym helpers unconditionally downcast the fp32 result back to bf16 (lines 532, 560); (2) backend=asym never silently falls back to torch (good), but as a result the entire fallback-statistics machinery (record_fallback, fallback_reasons) is dead code given VALID_BACKENDS only contains asym/torch.
- **Efficiency assessment:** The BF16 CPU-resident path is efficient and keeps the weight off HBM as designed (profiling confirms ~40% HBM reduction vs torch at a step-time cost). However the FP8/FP4 quantized path stages the FULL quantized weight CPU->HBM on every forward and dx call (_stage_quantized_tensor_for_kernel), defeating the no-HBM-staging premise for quantized precisions. The grouped path also forces a blocking D2H sync (offsets.tolist()) plus Python per-group copy loops on every call. Weight quantization itself is correctly cached once; activation quantization per-step is expected.
- **Bottom line:** The frozen-base execution core is numerically correct for the BF16 training path with proper no-weight-grad autograd and CPU-resident weight handling. The most important issues are an fp32-output contract that is silently downcast to bf16, a dead fallback-statistics subsystem, and the quantized (fp8/fp4) path staging the whole weight to HBM every call plus a per-call host-device sync in the grouped path. None of these break the primary BF16 LoRA-SFT path, but the fp32-output bug and the quantized staging undercut stated guarantees.

#### L1.1. 🔴 HIGH · correctness · bf16_output_dtype=float32 is silently downcast to bf16, violating the requested-output contract

- **Location:** `asym_gemm/training/frozen_linear.py:532,560`
- **Confidence:** 0.82 · **Verification:** adversarial panel real=3 refuted=0

**Problem (context + error):** _normalize_bf16_output_dtype explicitly accepts and returns torch.float32 (lines 138-145), and _asym_bf16_nt/_asym_grouped_bf16_nt allocate the output `d` with that dtype (lines 526, 554) so the kernel writes fp32. But both helpers then return `d if d.dtype == torch.bfloat16 else d.to(dtype=torch.bfloat16)`, unconditionally collapsing any fp32 result back to bf16. A caller that sets bf16_output_dtype=torch.float32 (e.g. to retain fp32 accumulation precision in forward y or in dx) gets bf16 anyway, with the precision silently discarded.

**Evidence:** Line 526: `d = torch.empty((m, n), device=a.device, dtype=output_dtype)` then line 532: `return d if d.dtype == torch.bfloat16 else d.to(dtype=torch.bfloat16)`. The `else` branch fires exactly when output_dtype==float32, throwing away the fp32 output. Same pattern at line 560 for the grouped helper.

**Fix:** Return `d` unchanged (it already has the requested output_dtype). If a bf16 guarantee is needed elsewhere, enforce it at the call boundary, not by silently downcasting a deliberately-fp32 buffer.

#### L1.2. 🔴 HIGH · efficiency · FP8/FP4 path stages the entire quantized weight CPU->HBM on every forward and dx call

- **Location:** `asym_gemm/training/frozen_linear.py:613,668,588`
- **Confidence:** 0.8 · **Verification:** single verifier

**Problem (context + error):** For precision in {fp8,fp4}, _asym_quantized_nt and _asym_grouped_quantized_nt call _stage_quantized_tensor_for_kernel on qweight.values and qweight.scales each call. Because the quantized cache is CPU-resident (pinned) and `a` is on CUDA, tensor.device != device always holds, so line 591 always executes a full `.to(device=...)` H2D copy of the whole quantized weight, every forward and every dx. This materializes the entire weight in HBM per call, directly contradicting the project's no-HBM-staging premise (which the BF16 path honors via TMA/cp.async on the still-CPU-resident weight). For an MoE down/gate_up projection this is a per-step full-weight H2D transfer.

**Evidence:** Line 613: `b_values = _stage_quantized_tensor_for_kernel(qweight.values, a.device)` (and 614 for scales, 668-669 grouped). _stage_quantized_tensor_for_kernel (588-591): `if tensor.device == device: return tensor.contiguous(); return tensor.to(device=device, ...)`. The acknowledging comment at 610-612 confirms this is intentional but undermines the design goal.

**Fix:** If the quantized kernels can fetch from CPU-pinned memory like the BF16 path, drop the staging. Otherwise stage ONCE and cache the device-resident quantized weight (it is frozen), rather than re-copying per call; document the resulting HBM footprint so the no-staging claim is scoped to BF16 only.

#### L1.3. 🟡 MEDIUM · efficiency · Grouped asym path forces a blocking host-device sync (offsets.tolist) plus per-group Python copy loops every call

- **Location:** `asym_gemm/training/frozen_linear.py:454,470-472,487-489`
- **Confidence:** 0.7 · **Verification:** single verifier

**Problem (context + error):** _pad_grouped_input_for_asym does `offsets.detach().to('cpu', dtype=long).tolist()` on line 454. During training offsets is typically CUDA-resident, so this is a blocking D2H copy + stream sync on every grouped forward AND dx (called from _asym_grouped_bf16_nt:553 and _asym_grouped_quantized_nt:666), stalling the pipeline. When padding is required it also allocates a zero-filled padded activation buffer and runs a Python loop of per-group `.copy_()` (470-472), and on output a matching unpad loop (487-489) — extra full activation copies and per-group kernel launches in the hot path.

**Evidence:** Line 454: `offsets_cpu = offsets.detach().to(device='cpu', dtype=torch.long).tolist()`. Lines 469-472 build `padded` and loop `padded[...].copy_(a[start:end])`; 486-489 mirror it for unpad. Both run unconditionally per grouped call.

**Fix:** Compute padded offsets on-device (or pass a precomputed CPU offsets copy alongside the CUDA one) to avoid the sync; if the kernel can consume cumulative offsets with per-group M directly, skip the pad/unpad activation copies entirely.

#### L1.4. ⚪ LOW · efficiency · _SINGLE_GROUP_LAUNCH_TENSOR_CACHE grows unbounded with variable token count M

- **Location:** `asym_gemm/training/frozen_linear.py:26,496-507`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** The module-global launch-tensor cache is keyed by (str(device), int(m)). In SFT the flattened token count m varies per step (variable batch x seq after packing/masking), so a new (offsets, experts) pair is allocated and retained for essentially every distinct m seen, never evicted. Each entry is only two tiny int32 tensors, so the practical leak is small, but it is unbounded over a long run and the cache hit rate for these single-group tensors is low precisely because m keeps changing.

**Evidence:** Line 26: `_SINGLE_GROUP_LAUNCH_TENSOR_CACHE: dict[tuple[str,int], ...] = {}`. Lines 504-506 insert `torch.tensor([0, m])` / `[0,-1]` keyed by m with no eviction. Used by both _asym_bf16_nt (527) and _asym_quantized_nt (617).

**Fix:** Build the two-element offsets/experts tensors inline per call (negligible cost vs the GEMM) or cap the cache with an LRU/size bound; keying on m provides little reuse benefit when m is dynamic.

#### L1.5. ⚪ LOW · design · Forward does not enforce/cast x to bf16; relies on caller and a hard raise under fp32 input

- **Location:** `asym_gemm/training/frozen_linear.py:1031,269`
- **Confidence:** 0.62 · **Verification:** single verifier

**Problem (context + error):** AsymFrozenLinearFunction.forward reshapes/contiguous-ifies x but never casts it to bf16. Custom autograd Functions are NOT auto-cast by torch.autocast, so if a caller passes fp32 x under autocast (or otherwise), _direct_bf16_reason returns 'requires_bf16' (line 269) and, for backend=asym, _dispatch_nt raises rather than degrading gracefully. This is correct-by-failing-loud, but it pushes a dtype invariant onto every caller and is easy to trip when integrating with autocast-based training loops.

**Evidence:** Line 1031: `x_2d = x.reshape(-1, host_weight.in_features).contiguous()` (no dtype handling). Line 269 in _direct_bf16_reason: `if a.dtype != torch.bfloat16 or b_cpu.dtype != torch.bfloat16: return 'requires_bf16'`, which for backend=asym becomes a hard RuntimeError.

**Fix:** Either document the bf16-input precondition prominently or cast x to the weight dtype (bf16) inside forward before dispatch, matching nn.Linear/autocast ergonomics.


### [L2] LoRA linear + dense autograd (lora.py, dense.py)

- **Correctness assessment:** Correct. The LoRA composition y = base(x) + dropout(x)·A·B·scaling is implemented faithfully, scaling = alpha/rank (no rslora — rslora is unimplemented anywhere, so it cannot be silently applied), dropout is applied only to the LoRA branch input, the base path is the frozen CPU-resident asym GEMM (no weight grad, dx via transpose_b=True), and only A/B are trainable. AsymLoRALinear.forward and TorchLoRALinear.forward are byte-identical, so the asym path is designed to match the torch reference within bf16 tolerance by construction. The custom autograd Functions in frozen_linear.py (used by dense.py) are correct: backward computes only dX (grad@W via transpose_b), bias grad sums over the M axis, host weight is non-differentiable data, and shapes/dtypes are handled with reshape/contiguous. Both LoRA and base dx contributions are summed at `base + lora` with nothing dropped or double-counted; this is corroborated by the gradcheck and parity tests in test_cpu_resident_frozen_base.py.
- **Efficiency assessment:** Efficient and on-design. A/B are small GPU-resident nn.Linear params; the base dx is computed via the asym transpose_b kernel rather than materializing a dense weight matmul; backward does not recompute the base forward (the frozen weight is re-fetched on demand, and under non-reentrant checkpointing the recompute is expected). The CPU-resident base trades latency for HBM (profiling_latest/lat.md: asym 6.35 GiB vs torch 10.76 GiB peak at b8/s1024), which is the documented tradeoff. Only minor, low-impact micro-inefficiencies (an unconditional dtype cast and a non-deterministic peft init) were found.
- **Bottom line:** The LoRA math is correct and the dense asym autograd Functions are correct and efficient; A/B are the only trainable params, base stays frozen CPU-resident, and dx uses the transpose_b asym kernel. The only genuine issues are low-severity: the default 'asym' init makes B nonzero so the initial LoRA delta is NOT zero (violates the B=0 invariant, but only affects the dense.py parity/showcase harness — the production lf.py path correctly uses init_lora_weights='peft' with B=0), the 'peft' init ignores the supplied generator (reproducibility, not math), and a redundant unconditional dtype cast in the LoRA forward. No correctness/race/UB defects found.

#### L2.1. ⚪ LOW · design · Default 'asym' LoRA init makes B nonzero, so the initial LoRA delta is not zero

- **Location:** `asym_gemm/training/lora.py:100-105`
- **Confidence:** 0.9 · **Verification:** single verifier

**Problem (context + error):** The default init_lora_weights='asym' (the default for both AsymLoRALinear and TorchLoRALinear, used throughout dense.py) initializes BOTH A and B with randn*0.01. This means the initial low-rank delta A·B·scaling is NOT zero at step 0, which deviates from the standard LoRA invariant 'B=0 so the adapter starts as a no-op'. The production LlamaFactory path (asym_gemm/integrations/lf.py:222) correctly overrides this with init_lora_weights='peft' (B=0), so real training is unaffected; the impact is confined to the dense.py micro/showcase parity harness, where it is intentional (both asym and torch models share identical init via copy_adapter_state, so parity still holds). Flagging because the project description states the expected init is 'B=0 so initial delta is 0', and the library default does not honor that.

**Evidence:** lora.py:100-105 `if init_lora_weights == "asym": a = _randn(... scale=0.01); b = _randn(... scale=0.01); a_weight.copy_(a...); b_weight.copy_(b...)`. The PEFT-correct branch at lines 106-109 does `nn.init.kaiming_uniform_(a_weight, a=math.sqrt(5)); nn.init.zeros_(b_weight)`. AsymLoRALinear.__init__ signature (lora.py:192) defaults `init_lora_weights: Literal["asym", "peft"] = "asym"`.

**Fix:** Either change the library default to 'peft' (matching the production path and the stated B=0 invariant), or document explicitly that 'asym' init is a parity-testing convenience that yields a nonzero initial delta. No code change is required for correctness of the LF SFT flow since lf.py already forces 'peft'.

#### L2.2. ⚪ LOW · correctness · 'peft' LoRA init ignores the supplied lora_generator (non-deterministic A init)

- **Location:** `asym_gemm/training/lora.py:106-109`
- **Confidence:** 0.85 · **Verification:** single verifier

**Problem (context + error):** In the 'peft' init branch, nn.init.kaiming_uniform_(a_weight, a=math.sqrt(5)) and nn.init.zeros_(b_weight) are called without passing the `generator` that is threaded through _reset_lora_weights. As a result, A is seeded from the global torch RNG, not the per-module lora_generator, so two models built with the same lora_seed will NOT get identical A matrices on the 'peft' path (unlike the 'asym' path at lines 101-102 which honors the generator). This is a reproducibility gap, not a math error: B=0 keeps the initial delta zero and gradients are still correct. It would, however, break exact reproducibility/parity if anyone built a model pair on the 'peft' path expecting deterministic A from lora_seed.

**Evidence:** lora.py:106-109 `if init_lora_weights == "peft": nn.init.kaiming_uniform_(a_weight, a=math.sqrt(5)); nn.init.zeros_(b_weight); return`. The function receives `generator` (line 98) and uses it only in the 'asym' branch. nn.init.kaiming_uniform_ accepts a `generator=` kwarg in recent torch but it is not passed here.

**Fix:** Pass the generator through, e.g. seed via the generator or use torch.nn.init.kaiming_uniform_(a_weight, a=math.sqrt(5), generator=generator) where supported, so the 'peft' path is as deterministic as the 'asym' path. Low priority since B=0 makes the adapter a no-op at init and production parity is validated on the 'asym' path.

#### L2.3. ⚪ LOW · efficiency · Unconditional dtype cast of the LoRA branch input on every forward

- **Location:** `asym_gemm/training/lora.py:262 (and 333 in TorchLoRALinear)`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** `lora_input = self.lora_dropout(x).to(dtype=self.lora_dtype)` always issues a .to(dtype=...) on the full activation tensor even when x is already lora_dtype (the common case: hidden states and lora_dtype are both bf16). When dropout is nn.Identity, .to() with a matching dtype is a near-no-op (returns the same tensor) so the cost is negligible; when dropout is active it allocates anyway. Minor: it can add an extra elementwise kernel/allocation when x.dtype != lora_dtype (e.g. fp32 activations with bf16 adapters). Not a correctness issue.

**Evidence:** lora.py:262 `lora_input = self.lora_dropout(x).to(dtype=self.lora_dtype)` then line 263 `lora = self.lora_B[...](self.lora_A[...](lora_input)) * self.scaling` and line 264 `return base + lora.to(dtype=base.dtype)`.

**Fix:** Optionally guard the cast (`x if x.dtype == self.lora_dtype else x.to(self.lora_dtype)`) after dropout, or rely on nn.Linear's internal dtype handling. Impact is negligible in the bf16/bf16 default; only worth doing if mixed-dtype activations are common.


### [L3] MoE expert path + recompute policies (moe.py)

- **Correctness assessment:** Correct. The grouped MoE expert LoRA path is numerically and gradient-correct across all dimensions I checked: the expert-sorted permutation (stable_keys = expert*num_routes + route) and its inverse (route_indices for scatter-back), the weighted gather/scatter with index_add_ and its autograd, the two-matmul gate_up -> SwiGLU -> down chain with LoRA added per projection, and the recompute/split/util/act subset orchestration. The subset index math in _grouped_subset is exactly right (I traced offsets=[0,16,26,30,32] selecting groups {1,3} -> route_indices=[16..25,30,31]), the four subset masks are provably disjoint and exhaustive over active rows so the new_empty output buffer is fully written, and dense_experts=False subset GEMMs select the correct original expert weights/LoRA banks. Recompute is bit-identical to forward by construction (offsets/experts captured outside the checkpoint, no dropout in the expert path, deterministic SiLU; preserve_rng_state=False is therefore safe). Zero-token experts, the -1 sentinel, masked padding (multiply-by-mask then boolean-compact), and empty-packed early returns are all handled. This matches the test contract (recompute output and LoRA grads must equal the no-recompute path to 1e-5) in tests/training/test_toy_moe_lora_sft.py and test_lf_qwen3_asym_backend.py.
- **Efficiency assessment:** Mostly sound, with minor host-device-sync overhead. The K-outer single-fetch property is preserved (each expert lands in exactly one subset, so its CPU weight is fetched once across the subset split). The recompute/act policies achieve their stated memory/compute tradeoff (profiling_latest activation_recompute sweep: s=2048 peak 17354->10659 MiB, ~39% less, at 259->359 ms step, ~38% slower). The defects are: (1) per-layer-forward GPU->host syncs in _validate_route_inputs (min/max .item()), which also hit the production Qwen3 path via build_contiguous_route_metadata; (2) up to ~9 syncs per layer (.any().item()/.sum().item()) in the recompute subset orchestration; (3) splitting one grouped GEMM into up to 4 smaller grouped GEMMs reduces per-kernel M and occupancy. None change results.
- **Bottom line:** moe.py is correct: routing/sort/offsets, the gather-scatter gradient, the gate_up->SwiGLU->down LoRA chain, and all four recompute policies are numerically and autograd-sound, with disjoint/exhaustive subset masks and bit-identical recomputation. I found no correctness bugs. The only issues are efficiency: redundant validation .item() syncs in the per-layer routing hot path (also affecting the production Qwen3 forward), several .item() syncs per layer in the recompute subset path, and reduced GEMM occupancy from per-subset splitting. Profiling artifacts confirm the recompute memory/compute tradeoff is real and as intended.

#### L3.1. ⚪ LOW · efficiency · Per-forward GPU->host syncs in route-input validation (also on production Qwen3 path)

- **Location:** `asym_gemm/training/moe.py:286-287`
- **Confidence:** 0.8 · **Verification:** single verifier

**Problem (context + error):** _validate_route_inputs runs min().item() and max().item() on the routing index tensor on every route-metadata build, i.e. every MoE layer forward. These are two GPU->host synchronizations per layer that serialize the launch stream. The check is redundant when routing comes from topk over num_experts logits (topk_routing_from_logits), since indices are guaranteed in [0,num_experts). It is reachable from the production LoRA-SFT path: qwen3_moe.py:499 calls build_contiguous_route_metadata each forward, which calls _validate_route_inputs.

**Evidence:** min_expert = int(topk_indices.min().item())\n    max_expert = int(topk_indices.max().item())\n    if min_expert < 0 or max_expert >= num_experts:\n        raise ValueError(...)  # build_contiguous_route_metadata -> _validate_route_inputs, called per layer (moe.py:1709) and per Qwen3 forward (qwen3_moe.py:499)

**Fix:** Guard the min/max device sync behind an opt-in debug flag, or skip it when indices originate from an internal top-k (pass a trusted=True flag from topk_routing_from_logits). Range validation can also be folded into a non-syncing assert via a single combined logical_and reduction kept on-device, or simply dropped on the hot path.

#### L3.2. ⚪ LOW · efficiency · Multiple GPU->host syncs per layer in the recompute subset orchestration

- **Location:** `asym_gemm/training/moe.py:1383-1388`
- **Confidence:** 0.78 · **Verification:** single verifier

**Problem (context + error):** When a per-expert recompute/split/act policy is active, _run_grouped_compact_checkpointed first does selected_groups.any().item() (line 1542), then constructs up to four subsets, each calling _grouped_subset which does group_mask.any().item() (1383) and selected_counts.sum().item() (1388). That is up to ~9 GPU->host synchronizations per MoE layer per routed-expert group, on top of the base routing syncs. For a many-layer model these stall the launch pipeline. counts derives from metadata.expert_offsets which lives on the GPU, so each .item() forces a sync.

**Evidence:** if not bool(group_mask.any().item()):\n    return None\n...\nselected_counts = counts.index_select(0, group_indices)\ntotal = int(selected_counts.sum().item())   # plus selected_groups.any().item() at line 1542, x4 subsets

**Fix:** Compute the four group masks and their counts once on-device, move a single small CPU copy of the per-group counts/masks to host (one sync), then derive subset membership and totals from the host copy without further per-subset .item() calls. Skip subsets with zero host-side count without launching a device reduction.

#### L3.3. ⚪ LOW · design · Per-subset GEMM splitting lowers grouped-GEMM M and occupancy

- **Location:** `asym_gemm/training/moe.py:1554-1593`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** With a recompute policy enabled, the single dense grouped GEMM over all experts is replaced by up to four separate grouped GEMMs (kept, split_control, recompute, activation_drop), each over a disjoint subset of experts and therefore a smaller M extent. Each subset launches its own gate/up/down base GEMMs and LoRA GEMMs. This is inherent to checkpointing a subset of experts (you cannot wrap part of one fused grouped GEMM in torch.utils.checkpoint), but it reduces the M dimension per kernel and thus the asym scheduler's per-CTA token reuse and occupancy, and multiplies Python/launch overhead by up to ~4x for the affected layers. Correctness is unaffected (experts are partitioned, so each weight is still fetched once overall).

**Evidence:** pieces = (self._run_grouped_subset_body(self._grouped_subset(packed, offsets, experts, kept_groups), ...),\n self._run_grouped_subset_body(self._grouped_subset(packed, offsets, experts, split_control_groups), ...),\n self._run_grouped_subset_body(self._grouped_subset(packed, offsets, experts, recompute_groups), ..., checkpoint_body=True),\n self._run_grouped_subset_body(self._grouped_subset(packed, offsets, experts, activation_drop_groups), ..., checkpoint_activation_down=True))

**Fix:** Document the occupancy tradeoff next to the policy selection; consider only splitting into two groups (kept vs recomputed) when split_control/activation_drop are not requested, and skip building empty subsets early using host-side counts (see the sync finding) to cut launch overhead.


### [L4] Qwen3 expert wrapping + MLP / host-weight offload

- **Correctness assessment:** Correct. wrap_qwen3_experts faithfully reproduces HF Qwen3MoeExperts.forward semantics: gate_up_proj fused linear split via chunk(2) into gate/up with matching column order, SiLU(gate)*up activation using the source's own act_fn, down_proj, and the per-route top_k_weights weighted combine applied in scatter_contiguous with HF-matching dtype/accumulation. The router (softmax/top-k/norm_topk_prob) lives in HF's unwrapped Qwen3MoeTopKRouter (self.gate), so renormalization is untouched and stays frozen; lf.py both freezes it via freeze_non_lora_params and asserts it is non-trainable. CPU/GPU residency accounting is correct: cpu_resident_base_bytes counts only AsymGroupedFrozenLinear weight_hbm_saved_bytes, gpu_resident_base_bytes falls through getattr(default 0) for the offloaded asym base (which lacks gpu_resident_weight_bytes) and reports real bytes only for the GPU-resident torch base. Recompute-policy masks are mutually exclusive and partition all active routes, with float/bf16 accumulation matching the reference; on-GPU parity tests (output + LoRA grads + dx) back this up.
- **Efficiency assessment:** Efficient and as-designed. Packed base weights are quantized/pinned/cloned exactly once at wrap time (HostWeight init), not per step; the gate/up A and B LoRA projections are fused into single grouped_mm calls (cat over rank dim) to minimize launches. Measured artifacts (profiling_latest/memory.md, lat.md) confirm the intended HBM-vs-compute tradeoff: asym offload saves 4.5 GiB HBM (29-41%) at ~30-60% higher step time vs the all-HBM torch baseline, with pinned CPU bytes matching the saved HBM. One coarseness note: in the asym+offload branch, expert-recompute thresholds degrade to all-or-nothing (recompute the full dense body over all tokens), so the fine-grained per-expert activation-memory savings available on the non-offload path are not realized when offloading.
- **Bottom line:** The Qwen3 packed-expert wrapper is correct and matches the HF reference op-for-op, including routing-weight combine, gate/up split ordering, activation, and the frozen router. CPU/GPU offload accounting is sound and confirmed by measured 4.5 GiB HBM savings. The only substantive issues are an efficiency coarseness in the asym+offload recompute path and a minor (dropout>0 only) shared-dropout-mask deviation from per-module PEFT semantics.

#### L4.1. 🟡 MEDIUM · efficiency · asym+offload recompute is all-or-nothing over all tokens, ignoring per-expert token thresholds

- **Location:** `asym_gemm/training/qwen3_moe.py:472-477`
- **Confidence:** 0.82 · **Verification:** single verifier

**Problem (context + error):** When expert_recompute is enabled and backend=='asym' with offload, the policy branch recomputes (or activation-drops) the ENTIRE dense expert body over ALL tokens/experts as soon as ANY expert is selected by the token-threshold/util mask. It calls _run_dense_checkpoint_body(packed, offsets, experts) / _run_dense_checkpoint_activation_down(packed, offsets, experts) on the full packed buffer. The non-offload branch (lines 479-495) instead uses _select_subset to checkpoint only the selected expert rows, leaving large experts on the cheap save-all path. Thus, for the primary CPU-offload training configuration, a policy like 'tok256-ckpt' meant to checkpoint only small-token experts effectively recomputes every expert. Numerically this is still correct (recompute is transparent, confirmed by test_asym_qwen3_experts_sm100_recompute_policies_match_none), but the activation-memory-vs-compute tradeoff the policy is supposed to tune is collapsed.

**Evidence:** if self.backend == "asym" and self.offload:
    if bool(recompute_groups.any().item()):
        return self._run_dense_checkpoint_body(packed, offsets, experts)
    if bool(activation_drop_groups.any().item()):
        return self._run_dense_checkpoint_activation_down(packed, offsets, experts)
    return self._forward_expert_body(packed, offsets, experts, dense_experts=True)
— compare with the subset-partitioned non-offload path that builds _select_subset(packed, metadata, recompute_groups) etc.

**Fix:** If selective recompute is intended under offload, route the offload path through the same _select_subset partitioning used in the non-offload branch (it already supports checkpoint_body / checkpoint_activation_down on subsets). If whole-body recompute under offload is intentional (e.g., to avoid extra index_select/index_copy traffic on the CPU-fetch path), document it explicitly so users do not expect token-threshold granularity to matter when offloading.

#### L4.2. ⚪ LOW · correctness · gate and up LoRA share a single dropout draw (minor deviation from per-adapter PEFT dropout)

- **Location:** `asym_gemm/training/qwen3_moe.py:210-213`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** _forward_gate_up_lora applies self.lora_dropout(x) once and reuses the resulting x_lora for BOTH the gate and up low-rank projections (via the concatenated gate_up_a). In PEFT, gate_proj and up_proj are independent LoRA modules, each with its own lora_dropout module and therefore its own independent dropout mask per forward. Here gate and up see the identical dropped mask. This only affects results when lora_dropout > 0 (default is 0.0, where dropout is nn.Identity and there is no difference), and it is a defensible design choice for the fused gate/up GEMM, but it is a behavioral mismatch versus a stock PEFT Qwen3 LoRA run and could cause small train-time divergence and non-reproducibility against a PEFT baseline at nonzero dropout.

**Evidence:** x_lora = self.lora_dropout(x).to(dtype=self.lora_dtype)
gate_up_a = torch.cat((self.gate_lora_A, self.up_lora_A), dim=1)
low_rank = grouped_expert_lora(x_lora, gate_up_a, offsets, experts, metadata=metadata)
— a single x_lora feeds both gate and up; PEFT would draw two independent masks.

**Fix:** Document that the fused gate/up LoRA uses a shared dropout mask, or (if exact PEFT parity at dropout>0 is required) apply two independent dropout draws for the gate and up inputs before the fused projection. Low priority since the default and all current tests use lora_dropout=0.0.


### [L5] LoRA-SFT efficiency from profiling artifacts

- **Correctness assessment:** The profile scripts are numerically sound and the correctness checker (check_lora_ops.py) validates base output, dx, x/A/B grads against a torch reference with L2-relative tolerances and asserts the host weight never receives grad — this is well-designed. The transpose accuracy gates (transpose_correct PASS in every block_sweep JSON) also confirm the dx path is mathematically correct. No correctness defects found in this area.
- **Efficiency assessment:** The measured artifacts show AsymGEMM is NOT efficient for LoRA SFT in the speed dimension — it is consistently SLOWER than the torch GPU-resident baseline. Its only win is HBM savings (29-41% peak HBM). At the operator level asym fwd is ~7x and bwd ~8x slower than torch (CUDA-graph captured, so this is real kernel/fetch cost, not launch overhead). The dx/transpose path is highly shape-sensitive (1.3-2.4x slower than nontranspose for several real Qwen3 shapes) and the configured BLOCK_K tile is suboptimal. End-to-end the asym step is 1.18-1.58x slower than torch with ~12-16% of step time lost to host/runtime no-kernel gaps per AsymGEMM dispatch.
- **Bottom line:** The profiling area is methodologically solid and the conclusions it supports are honest: AsymGEMM trades speed for HBM. The data clearly shows asym is slower than torch at every level measured (operator 7-8x, e2e 1.2-1.6x), with the value being 29-41% peak HBM reduction. The dx transpose path and per-dispatch host overhead are the concrete bottlenecks; the data points to block-tile autotuning and dispatch-overhead reduction as the highest-value improvements. A few measurement caveats (no L2 flush, CUDA-graph capturing the per-call fetch path, FP8 baseline asymmetry) could modestly distort absolute numbers but do not change the qualitative verdict.

#### L5.1. 🔴 HIGH · efficiency · Measured operator data shows AsymGEMM is 7-8x slower than torch baseline; speed is not the value proposition

- **Location:** `profiling/lora_ops_bf16/full_lora__b1_s2048_r16/2048x768__asym__fw/result.csv vs 2048x768__torch__fw/result.csv`
- **Confidence:** 0.93 · **Verification:** single verifier

**Problem (context + error):** The isolated full_lora operator benchmark (profile_lora_ops.py, CUDA-graph captured to remove launch overhead) shows the asym path is dramatically slower than torch at the very shapes used for LoRA SFT. Because this is graph-replay timing, the slowdown is real GPU-side kernel + CPU-resident-weight-fetch cost, not Python/launch overhead. The asym path's only measured advantage is lower peak HBM.

**Evidence:** 2048-token in2048/out768: asym fw median_ms=0.172384 vs torch fw 0.039680 (4.3x); asym bw 0.280800 vs torch 0.075152 (3.7x). The larger b8_s1024 (8192 tokens) case in profiling_latest/lora_operator is worse: asym fw 0.575056 vs torch 0.078112 (7.4x), asym bw 1.375568 vs torch 0.163200 (8.4x). HBM is lower for asym (e.g. fw 0.137 vs 0.140 GiB) but the speed gap is large.

**Fix:** Frame and report AsymGEMM for LoRA SFT explicitly as an HBM/capacity optimization, not a latency optimization. The summary table already reports Torch/Candidate speedup < 1.0 (0.632, 0.849) and HBM saved — keep that framing prominent. For latency, the fetch-bound base GEMM needs kernel-level work (overlap fetch with compute, larger reuse) before it can approach torch.

#### L5.2. 🔴 HIGH · efficiency · End-to-end asym step is 1.18-1.58x slower than torch; trade is HBM (29-41%) for time

- **Location:** `profiling_latest/bf16_lora-sft_summary.md:18-23 (Comparisons table)`
- **Confidence:** 0.9 · **Verification:** single verifier

**Problem (context + error):** The e2e LoRA-SFT summary confirms the operator-level picture at full-model scale: for Qwen3-30B-A3B (4 layers, b8) the asym backend's step time is consistently larger than torch, with the benefit being a substantial peak-HBM reduction. This is the central efficiency question and the data answers it: asym is slower but more memory-frugal.

**Evidence:** s1024: asym step 2622.379ms vs torch 1657.642ms (Torch/Candidate speedup 0.632, asym +964.7ms) but peak HBM 6.35 vs 10.76 GiB (41.49% saved). s2048: asym 2326.767ms vs torch 1975.033ms (0.849, +351.7ms) peak HBM 10.85 vs 15.17 GiB (29.32% saved). Note the s1024 asym run is SLOWER in absolute ms than the s2048 asym run, indicating per-step fixed overhead dominates at small seq.

**Fix:** Report the HBM-vs-time Pareto explicitly per workload. The s1024-slower-than-s2048 anomaly (2622 vs 2327 ms) strongly suggests a fixed per-step host/fetch overhead that does not amortize at small token counts — investigate and reduce that fixed cost (see dispatch-overhead finding).

#### L5.3. 🔴 HIGH · efficiency · dx/transpose base GEMM is shape-sensitive and 1.3-2.4x slower than nontranspose for several real Qwen3-30B shapes

- **Location:** `profiling/transpose_block_sweep/t64_64_256_m2048_k768_n2048.json (ratios.asym_bf16_transpose_over_nontranspose)`
- **Confidence:** 0.88 · **Verification:** single verifier

**Problem (context + error):** profile_transpose.py measures the dx path (G@W used for backward data-gradient). The measured ratios show the true transpose_b path is substantially slower than the nontranspose path for some shapes that occur in Qwen3-30B (k768/n2048, k1536/n4096), and is itself many times slower than torch.mm in every shape. Because dx runs in every backward, this is a real e2e bottleneck.

**Evidence:** k768/n2048 (t64 block): asym_bf16_transpose 0.1958ms vs nontranspose 0.0960ms -> ratio 2.04 ('asym_transpose_much_slower': True). k1536/n4096: transpose 0.3815ms vs nontranspose 0.1618ms -> ratio 2.358. Across all shapes torch_transpose/nontranspose stays ~1.0 (0.95-1.02) while asym transpose ranges 0.38x-2.37x and asym is 3-9x slower than the ~0.035ms torch baseline.

**Fix:** The transpose_b in-kernel path reads W with a strided/MN-major access pattern that is poorly tiled for these N/K. Either autotune the block tile per (transpose, N, K) or pre-store W^T on the host for hot dx shapes (stored_transpose is faster than true transpose for k1536/n4096: 0.2829 vs 0.3815). The data already shows stored-transpose wins for tall-K shapes and loses for short-K shapes, so a per-shape choice is warranted.

#### L5.4. 🔴 HIGH · improvement · Configured SM100 BF16 block tile is suboptimal; BLOCK_K sweep shows up to ~1.8x swing and the script's default fixed tile leaves performance on the table

- **Location:** `profiling/transpose_block_sweep/ (t64_* vs t128_* JSONs, same shapes)`
- **Confidence:** 0.82 · **Verification:** single verifier

**Problem (context + error):** The block-sweep artifacts directly compare BLOCK_M 64 vs 128 (and BLOCK_K 256 vs 512) on the same shapes and show the transpose latency is very sensitive to the tile, yet the production path appears to use a single compiled tile. Choosing the right tile per shape would materially cut the dx bottleneck above, narrowing the gap to torch.

**Evidence:** k768/n2048 transpose: t64_64_256 = 0.1958ms vs t128_64_256 = 0.1115ms (1.76x faster with BLOCK_M=128). k1536/n4096 transpose: t64_64_256 = 0.3815ms vs t128_64_256 = 0.2296ms (1.66x). The benefit is shape-dependent and sometimes reverses, so a fixed tile is leaving ~1.5-1.8x on the table for the worst shapes.

**Fix:** Add per-shape block-tile autotuning (or a small lookup table keyed on transpose/M/N/K) for the SM100 BF16 asym GEMM, seeded from these sweep JSONs. This is the single highest-leverage efficiency change the data supports for the dx path.

#### L5.5. 🟡 MEDIUM · efficiency · ~12-16% of e2e step time is GPU no-kernel gap, attributed per-AsymGEMM-dispatch to host/autograd/Python + CUDA runtime API

- **Location:** `profiling/lora_e2e_bf16/qwen3-235b-a22b-instruct-2507-l4__b8_s2048_r64_a128/asym__nsys__norecomp__polnone/s2048/profile.json (stage_breakdown gpu_no_kernel_time)`
- **Confidence:** 0.78 · **Verification:** single verifier

**Problem (context + error):** The nsys postprocessor reports a large no-kernel gap whose attribution rows point at per-AsymGEMM-call host work (Python dispatch, autograd, and CUDA runtime/API) plus route metadata. This is the host-side stall the CPU-resident-fetch design risks, and it does not overlap GPU compute well, so it directly inflates step time.

**Evidence:** step.forward total 247.18ms with gpu_no_kernel_time 38.57ms (15.6%) and sync API 39.86ms; step.backward no_kernel 29.48ms (11.5%). Gap attribution rows: 'No-kernel host/autograd/Python: routed up base AsymGEMM' 6.448ms, '...down' 5.576ms, '...gate' 5.303ms, plus 'No-kernel CUDA runtime/API: routed ... AsymGEMM' 3.96/2.67/2.57ms and 'route metadata' 2.887ms.

**Fix:** Reduce per-dispatch host cost: cache JIT-compiled kernel handles and TMA descriptors per shape, batch the gate/up/down grouped calls, precompute route-metadata tensors, and ensure the weight-fetch (cp.async/TMA) is issued early enough to overlap the previous kernel. The s1024-slower-than-s2048 anomaly (prior finding) corroborates a fixed host overhead per step.


---

## Q3 — LlamaFactory backend integration (vs ktransformers)


### [I1] LF integration module + adapter parity (lf.py)

- **Correctness assessment:** Mostly correct for the constrained first-pass flow, with ONE real defect: the standard lora_target=all path passes the MoE router leaf name `gate` into dense_target_modules, so apply_lf_asym_lora wraps the router and then its own validation raises RuntimeError, making target=all unusable on packed Qwen3 MoE. The module-tree surgery (_matches_target/_parent_and_child/_replace_child, ModuleList index replacement, expert-prefix exclusion via _is_under, double-wrap guards) is otherwise sound, and freezing/validation logic is robust. The fp32-cast skip from the early return is NOT a bug because parser.py forces pure_bf16, which makes adapter.py also skip the cast — true parity.
- **Efficiency assessment:** No efficiency problems specific to this integration layer. Setup does two full model.named_modules() passes (O(modules), one-time at load) which is negligible. No host-device syncs, redundant copies, or launch-overhead concerns introduced here.
- **Bottom line:** apply_lf_asym_lora is structurally consistent with the use_kt/PEFT branches and the early return is safe: gradient checkpointing is wired in patch_model BEFORE init_adapter, valuehead/modules_to_save are excluded by parser guards (SFT-only, no additional_target), and the fp32 upcast is intentionally skipped because pure_bf16 is enforced. The one genuine correctness issue is that the conventional lora_target=all flow routes the MoE router (leaf `gate`) into the dense targets; the router gets LoRA-wrapped and _validate_trainable_params then raises, so target=all crashes on real packed Qwen3 MoE even though the unit test passes by hand-omitting `gate`. Freezing and the lora_/router name checks are otherwise robust.

#### I1.1. ⚪ LOW · design · Early return bypasses the shared fp32-cast block but is safe only because parser enforces pure_bf16 (implicit coupling)

- **Location:** `asym_gemm/integrations/lf.py:243 (via adapter.py:272), adapter.py:307-309`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** The asym branch returns the model at adapter.py:272, skipping the cast_trainable_params_to_fp32 loop at adapter.py:307-309 that the use_kt/PEFT branches fall through to. This is correct ONLY because parser.py:355-356 hard-requires pure_bf16=true for asym, which sets cast_trainable_params_to_fp32=False at adapter.py:340, so the PEFT path would not upcast either. The correctness of keeping LoRA in bf16 is thus enforced by a validation in a different file; if that pure_bf16 guard were ever relaxed, the asym path would silently diverge from the documented LF fp32-LoRA behavior. Gradient checkpointing, valuehead, and modules_to_save parity are fine (GC set in patch_model before init_adapter; valuehead is SFT-excluded; additional_target forbidden by parser).

**Evidence:** adapter.py:272 `return model` precedes adapter.py:307 `if is_trainable and cast_trainable_params_to_fp32:`. lf.py builds adapters with `lora_dtype=torch.bfloat16` (lf.py:181,221). parser.py:355-356 raises unless pure_bf16, and adapter.py:340 maps pure_bf16 -> cast_trainable_params_to_fp32=False.

**Fix:** Document the pure_bf16 dependency in apply_lf_asym_lora (or assert bf16 LoRA dtype explicitly inside the function) so the early-return invariant is self-contained rather than relying on a remote parser guard.


### [I2] LF arg parsing, gating, guards, distributed launcher

- **Correctness assessment:** Largely correct. The AsymGEMMArguments definition, the ~40-line validation block in parser.py (lines 337-378), the DeepSpeed ZeRO-3 guard (445-446), the launcher single-process gating (launcher.py:62), the workflow predict/accuracy guards (workflow.py:84-88), the modelcard tag (trainer_utils.py:108), and the dependency check (parser.py:199-200) are all present, internally consistent, and faithfully mirror the use_kt pattern. The recompute-policy help text matches the actual parser, and check_version("asym_gemm") resolves to the real dist name. No crash- or wrong-result-inducing guard gap was found within this area.
- **Efficiency assessment:** Not an efficiency-sensitive area (one-time arg parsing). No efficiency concerns.
- **Bottom line:** The AsymGEMM LF wiring is complete and consistent with use_kt; guards cover quant/DoRA/rslora/PiSSA/LoRA+/resume/ZeRO-3/KT/Unsloth/non-SFT/non-LoRA/bf16. The main real issue is a launcher footgun shared with KT: the single-process gating keys off the USE_ASYM_GEMM env var, not the use_asym_gemm yaml flag, so a yaml-only enable on a multi-GPU box silently auto-spawns DDP, which the asym CPU-resident path is not validated for and the parser never forbids. A secondary issue is a dead/redundant bf16 check. Both are low/medium severity; the area is otherwise sound.

#### I2.1. 🟡 MEDIUM · correctness · Launcher gating reads USE_ASYM_GEMM env, not the use_asym_gemm yaml flag; yaml-only enable on multi-GPU silently spawns DDP

- **Location:** `src/llamafactory/launcher.py:62`
- **Confidence:** 0.75 · **Verification:** single verifier

**Problem (context + error):** The single-process gate uses use_asym_gemm() which reads the USE_ASYM_GEMM environment variable (misc.py:347-348), but the launcher runs as the cli.py entry point BEFORE any yaml is parsed, so it can never observe a yaml-level `use_asym_gemm: true`. There is no code anywhere that derives USE_ASYM_GEMM from the model arg (confirmed: no os.environ['USE_ASYM_GEMM'] set in src). Consequently, a user who follows the KT-example convention of setting `use_asym_gemm: true` in the config yaml (mirroring `use_kt: true`) but forgets `export USE_ASYM_GEMM=1` will, on a box with >1 visible GPU, fall into the torchrun branch and auto-launch multi-process DDP. The parser's line-380 check then does NOT fire (parallel_mode is DISTRIBUTED), and nothing in this review area forbids asym under DDP, even though asym keeps base weights CPU-resident and the lf integration has no world_size/DDP handling. The provided shell script (run_lf_lora_sft.sh:212) does set USE_ASYM_GEMM=1 and pins one GPU, so the happy path is safe; the risk is the yaml-driven invocation. This is the same failure mode as use_kt, so it is a shared design hazard rather than an asym-specific regression.

**Evidence:** launcher.py:62 `or (get_device_count() > 1 and not use_ray() and not use_kt() and not use_asym_gemm())`; misc.py:347 `def use_asym_gemm(): return is_env_enabled("USE_ASYM_GEMM")`. cli.py launches launcher.launch() before parser.get_train_args, so model_args.use_asym_gemm is unavailable at gating time. grep found no place setting the USE_ASYM_GEMM env from the flag.

**Fix:** Either (a) auto-export USE_ASYM_GEMM from the parsed flag and re-exec, or (b) add an explicit parser guard that raises when model_args.use_asym_gemm and training_args.parallel_mode == ParallelMode.DISTRIBUTED (i.e. forbid DDP) until multi-GPU asym is validated, and document that USE_ASYM_GEMM=1 must be exported. Same fix should be applied symmetrically to use_kt.

#### I2.2. ⚪ LOW · design · asym bypasses the distributed-launch requirement but is never affirmatively forbidden under actual DDP

- **Location:** `src/llamafactory/hparams/parser.py:380`
- **Confidence:** 0.6 · **Verification:** single verifier

**Problem (context + error):** Line 380 exempts use_asym_gemm (and use_kt) from the 'Please launch distributed training' error, which is correct for the intended single-process design. However, no guard anywhere asserts that asym must run single-process: if asym ends up under ParallelMode.DISTRIBUTED (via FORCE_TORCHRUN=1 or the env-var omission described in the other finding), training proceeds with no error. Because the asym base modules are custom CPU-resident wrappers (apply_lf_asym_lora) rather than standard nn.Linear, DDP gradient bucketing / parameter replication semantics are unvalidated for this path. This is a design/robustness gap, low severity because the shipped launch script pins a single GPU.

**Evidence:** parser.py:380 `if not (model_args.use_kt or model_args.use_asym_gemm) and training_args.parallel_mode == ParallelMode.NOT_DISTRIBUTED:` — asym is excused from the single-process error but there is no complementary `if model_args.use_asym_gemm and parallel_mode == DISTRIBUTED: raise`. grep for asym/kt next to world/gpu/ddp/parallel found only this line.

**Fix:** Add an explicit positive assertion for the first implementation (e.g. raise if use_asym_gemm and parallel_mode == ParallelMode.DISTRIBUTED, or if world_size > 1) so the unsupported multi-process case fails loudly instead of running with undefined DDP behavior.


### [I3] Tests + LF run scripts (coverage adequacy)

- **Correctness assessment:** The CPU-executable torch-backend tests are genuinely strong: forward parity, dx parity, LoRA-grad parity and recompute-policy parity are all asserted against an eager HF-style reference within tight tolerances (test_toy_moe/dense, test_lf_qwen3 torch path, test_cpu_resident gradcheck). The fp64 gradcheck and the per-element grad comparisons mean a wrong torch-backend gradient WOULD be caught on CPU. However, every assertion that exercises the actual asym CUDA kernel (BF16/FP8/FP4 forward, dx via transpose_b, recompute policies on the kernel, the asym-vs-torch backend parity) is gated behind `torch.cuda.is_available()` / SM90/SM100 skips, so on a CI runner without a Blackwell/Hopper GPU the kernels are never executed and a numerical regression in the kernel itself passes silently. The kernel correctness therefore rests entirely on a developer manually running on the right hardware.
- **Efficiency assessment:** Not the focus of this area; the tests do assert efficiency-adjacent invariants (staged_calls==0, torch_calls==0, asym_forward/dx call counts, HBM-saved >= 0.8x expected, pinned-CPU bytes) which would catch a silent fallback to a slow path or accidental HBM staging. These are reasonable guards. No efficiency defect in the tests themselves.
- **Bottom line:** On CPU/CI the torch reference path is well covered (forward+backward+grad parity within tolerance), but the asym CUDA kernel path is entirely skip-gated and never runs without a Hopper/Blackwell GPU, so kernel regressions are not caught in CI. The end-to-end LlamaFactory parity story is the weakest link: compare_lf_smoke_losses.py implements a loss-curve check but is wired into nothing (no CI, no script, no doc), and run_lf_lora_sft.sh only verifies that asym_forward/dx call counts are >0 (the kernel ran) without ever comparing the asym loss curve to an HF/torch baseline. There is no automated test that asym and HF/torch produce matching losses end-to-end on a real Qwen3 config.

#### I3.1. 🔴 HIGH · correctness · All asym-CUDA-kernel correctness assertions are skip-gated; CI never executes the kernel

- **Location:** `tests/training/test_lf_qwen3_asym_backend.py:291-313, tests/training/test_cpu_resident_frozen_base.py:468-509, tests/training/test_toy_moe_lora_sft.py:1191-1192`
- **Confidence:** 0.95 · **Verification:** adversarial panel real=3 refuted=0

**Problem (context + error):** Every test that drives the real asym GEMM (backend="asym") and compares its forward/dx/LoRA grads against the torch backend or a quantized reference is guarded by skipif on torch.cuda.is_available() and device capability (SM90/SM100). On a CPU CI runner all of these are skipped, so a numerical bug in the kernel (wrong K-outer accumulation, wrong transpose_b dx, wrong FP8/FP4 scaling) would not fail any collected test. The torch-backend tests that DO run cannot catch a kernel bug because they never call the kernel.

**Evidence:** test_lf_qwen3_asym_backend.py:312 `@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 10, reason="requires SM100-class CUDA")` guards the only asym-vs-torch backward-parity test (`test_asym_qwen3_experts_sm100_backward_matches_torch_backend`). Identically test_cpu_resident_frozen_base.py:468 `@pytest.mark.skipif(not _direct_bf16_available(), ...)` and :549 `if not _direct_precision_available(precision): pytest.skip(...)` gate the BF16/FP8/FP4 kernel parity. The fp8/fp4 reference at test_cpu_resident_frozen_base.py:550 only runs when `_direct_precision_available` is true (arch in {9,10}).

**Fix:** Make the GPU-required parity tests run in a hardware CI job (a Hopper/Blackwell runner) and fail the merge if they are universally skipped, or add a clearly-labeled 'kernel parity not exercised on this runner' xfail/marker that a release gate inspects. At minimum, document that a green CPU test run does NOT validate the kernel.

#### I3.2. 🔴 HIGH · correctness · compare_lf_smoke_losses.py is dead code: never invoked by any script, CI, or doc

- **Location:** `scripts/lf/compare_lf_smoke_losses.py:52-74, scripts/lf/run_lf_lora_sft.sh:243-269`
- **Confidence:** 0.92 · **Verification:** adversarial panel real=3 refuted=0

**Problem (context + error):** compare_lf_smoke_losses.py is the only artifact that would assert the asym backend's LF loss curve matches a torch/HF baseline (first-step rel-tol 0.02, max rel-tol 0.10 over min_steps). A repo-wide grep finds zero references to it: not from run_lf_lora_sft.sh, not from profile_lora_lf.sh, not from any CI (the project has no .github of its own), and not from any doc. The end-to-end numerical-parity guard therefore never runs automatically; a regression that makes asym losses diverge from HF would not be caught.

**Evidence:** grep for 'compare_lf_smoke_losses' across *.sh/*.py/*.md returns only the file itself. run_lf_lora_sft.sh:243-265 only verifies `asym_forward_calls=(\d+), asym_dx_calls=(\d+)` are >0 ('Verified AsymGEMM runtime calls') — it confirms the kernel RAN, not that the loss is correct. `find . -name .github` shows no project CI; only vendored third-party workflows exist.

**Fix:** Wire compare_lf_smoke_losses.py into run_lf_lora_sft.sh (run BACKEND=hf and BACKEND=asym, then diff the two trainer_log.jsonl dirs) or into a documented smoke-runner, and add it to a CI/release gate. As-is it provides false assurance.

#### I3.3. 🟡 MEDIUM · correctness · run_lf_lora_sft.sh CHECK_ASYM_CALLS only proves the kernel ran, not that results are correct

- **Location:** `scripts/lf/run_lf_lora_sft.sh:243-269`
- **Confidence:** 0.85 · **Verification:** single verifier

**Problem (context + error):** The only post-run validation for BACKEND=asym is a regex that asserts asym_forward_calls>0 and asym_dx_calls>0. This is a 'did it execute' check, not a correctness check: a kernel returning garbage (NaN-free but wrong) or a wrong-scale FP8 path would still report positive call counts and the script would print 'Verified AsymGEMM runtime calls' and exit 0. There is no loss-finiteness threshold, no baseline comparison, and no NaN guard on the emitted loss values in this script.

**Evidence:** run_lf_lora_sft.sh:260 `if forward_calls <= 0 or dx_calls <= 0: raise SystemExit(...)` then :265 `print(f"Verified AsymGEMM runtime calls: ...")`. The trainer_log.jsonl is only copied (`cp` at :272), never inspected for finite/decreasing loss.

**Fix:** After the run, parse trainer_log.jsonl for finite, non-increasing loss and (ideally) invoke compare_lf_smoke_losses.py against a cached hf baseline run. Rename the current check to reflect it only confirms kernel dispatch.

#### I3.4. 🟡 MEDIUM · correctness · LF integration apply_lf_asym_lora is only tested for wiring on CPU, never forward/backward numerically

- **Location:** `tests/training/test_lf_qwen3_asym_backend.py:262-288`
- **Confidence:** 0.8 · **Verification:** single verifier

**Problem (context + error):** test_apply_lf_asym_lora_wraps_experts_dense_and_freezes_router is the only CPU-runnable test of the actual LF entry point apply_lf_asym_lora. It asserts module-replacement counts, types, frozen router, and that trainable params are all LoRA — but it never calls model(...) forward or backward, so it cannot detect a wiring bug that produces wrong activations/gradients through the wrapped FakeModel. The numerical end-to-end LF path (apply_lf_asym_lora + forward + backward) is only covered by test_apply_lf_asym_lora_sm100_accumulates... at :427, which is SM100-skip-gated and itself only checks grad finiteness + optimizer-updates-only-lora, not parity against an HF reference.

**Evidence:** Lines 262-288 contain only structural asserts: `assert report.qwen3_experts_wrapped == 2`, `assert isinstance(model.layers[0].mlp.experts, AsymQwen3Experts)`, `assert not model.layers[0].mlp.gate.weight.requires_grad`, `assert all("lora_" in name ... for name in trainable)` — there is no `model(...)`, no `.backward()`, no tolerance comparison. The asym e2e variant at :427 is `@pytest.mark.skipif(not ... capability < 10)` and asserts only `torch.isfinite(grad)` and `changed` param names.

**Fix:** Add a CPU torch-backend e2e test that runs FakeModel forward+backward through apply_lf_asym_lora and compares loss/grads to the unwrapped FakeModel within tolerance (the building blocks exist via _assert_grad_close). That would catch LF-glue regressions without a GPU.

#### I3.5. 🟡 MEDIUM · improvement · Recompute-policy parity on the real kernel, FP8/FP4, and multi-layer Qwen3 are only covered behind GPU skips or not at all

- **Location:** `tests/training/test_lf_qwen3_asym_backend.py:367-423, tests/training/test_cpu_resident_frozen_base.py:549-637`
- **Confidence:** 0.75 · **Verification:** single verifier

**Problem (context + error):** The recompute policies (split/tok-ckpt/tok-act-ckpt) are parity-tested against 'none' on CPU only with backend=torch (test_lf_qwen3:209-259) and on the kernel only behind SM100 skip (:367-423). FP8 and FP4 precisions are tested ONLY on GPU (test_cpu_resident:549-637, gated by _direct_precision_available), so there is no CPU coverage that fp8/fp4 quantization math (the python-side per_token_cast / scale-expansion reference) stays correct. There is also no test on a real multi-layer Qwen3 HF config or the offload path with a real checkpoint — only the 2-layer FakeModel and synthetic FakeQwen3Experts.

**Evidence:** test_lf_qwen3_asym_backend.py:209 `@pytest.mark.parametrize("policy", ["split2", "tok2-ckpt", "tok2-act-ckpt"])` on the torch path vs :368 same parametrize but `@pytest.mark.skipif(... capability < 10)`. test_cpu_resident_frozen_base.py:551 `if not _direct_precision_available(precision): pytest.skip(...)` means the fp8/fp4 reference comparison never executes on CPU.

**Fix:** Add CPU tests that validate the fp8/fp4 quantize/dequantize reference path (per_token_cast_to_fp8/nvfp4) for round-trip error independent of the kernel, and (if feasible) a small real-config Qwen3MoE single-layer forward/backward parity using the torch backend.

#### I3.6. ⚪ LOW · design · No pytest config / markers to distinguish CPU-safe vs GPU-required, making skip coverage invisible

- **Location:** `tests/training/ (repo has no pytest.ini/pyproject [tool.pytest]/conftest.py)`
- **Confidence:** 0.7 · **Verification:** single verifier

**Problem (context + error):** There is no pytest.ini, setup.cfg, tox.ini, conftest.py, or [tool.pytest.ini_options] in the project, and no custom markers (e.g. @pytest.mark.gpu). Consequently a `pytest` run on CPU reports a sea of green with silently-skipped GPU tests, and there is no mechanism to fail a release if the GPU-required parity suite was entirely skipped. This is what makes findings 1 and 5 dangerous in practice — the skip is invisible to a reader of the pass/fail summary.

**Evidence:** `cat conftest.py pytest.ini setup.cfg tox.ini` returns nothing; `grep pytest pyproject.toml` returns nothing. Skips are ad-hoc `pytest.mark.skipif(not torch.cuda.is_available() ...)` scattered per-test with no shared marker.

**Fix:** Introduce a conftest with a `gpu`/`sm100` marker and a CI/release gate that asserts the GPU suite was collected-and-run (not skipped) on the appropriate hardware; surface skip counts in the merge gate.


---

## ⚠️ Uncertain — needs a GPU run to settle

#### U. 🟡 MEDIUM · efficiency · CPU-resident weight fetch (CUDA memcpy/transfer) is a real, measured e2e cost (~6.7% of step)

- **Location:** `profiling/lora_e2e_bf16/qwen3-235b-a22b-instruct-2507-l4__b8_s2048_r64_a128/asym__nsys__norecomp__polnone/s2048/table.md (CUDA memcpy / transfer row)`
- **Confidence:** 0.72 · **Verification:** single verifier

**Problem (context + error):** The asym design fetches B from CPU-pinned DRAM each call. The nsys trace quantifies this as a distinct memcpy/transfer line that is non-trivial and does not fully hide behind compute, adding to the step time gap vs torch (which has weights resident in HBM and pays zero such transfer).

**Evidence:** 'CUDA memcpy / transfer' = 33.6607ms total (6.69% listed): FWD 14.4245ms, BWD 19.2361ms. Torch baseline pays 0 weight transfer (weights are CUDA buffers). This transfer is intrinsic to the CPU-resident design and partly explains the e2e slowdown.

**Fix:** Quantify achieved interconnect bandwidth vs theoretical (NVLink-C2C / PCIe) for these transfers and verify double-buffering/overlap with the K-outer reuse loop. If transfers are not overlapping the GEMM compute, prefetching the next K-tile while computing the current one is the key win.

> **Verifier note:** [uncertain/low] The cited numbers are accurate but the finding's mechanistic claims are not substantiated by the code/data I read.

VERIFIED FACTS:
- asym table line 12 matches exactly: `CUDA memcpy / transfer | 14.4245 | - | 19.2361 | - | 33.6607 | 6.69%` (FWD 14.4245, BWD 19.2361, total 33.6607ms). File: profiling/lora_e2e_bf16/qwen3-235b-a22b-instruct-2507-l4__b8_s2048_r64_a128/asym__nsys__norecomp__polnone/s2048/table.md.
- CPU-resident design is real: asym memory table shows `CPU pinned W = 18432.00 MiB`, `GPU buffers = 704.16 MiB`; torch shows `CPU pinned W = 0`, `GPU buffers = 19136.16 MiB`. asym_gemm/training/frozen_linear.py guards on `b_cpu.is_pinned()` / `weight_not_cpu` / `weight_not_pinned`, confirming B lives on CPU pinned DRAM.

DECISIVE PROBLEMS WITH THE FINDING:
1. "Torch baseline pays 0 weight transfer" is FALSE as stated. The torch table (torch__nsys__norecomp__polnone/s2048/table.md, line 20) shows `CUDA memcpy / transfer = 4.7963 ms` (0.0612 FWD + 4.7350 BWD). So the incremental transfer cost of the CPU-resident design is ~29ms, not the full 33.66ms. The finding's "0" is contradicted by a non-zero memcpy line.

2. The postprocess script (scripts/lora/postprocess_nsys_lora.py) does NOT distinguish memcpy direction: there is no HtoD/DtoD/DtoH split (grep found only the single undifferentiated "CUDA memcpy / transfer" label). So the 33.66ms cannot be confirmed to be weight-B fetches; it is an undifferentiated memcpy bucket. The AsymGEMM design passes the pinned B directly into the GEMM kernel, which streams the weight inside the kernel prologue — meaning the weight-streaming cost is most plausibly folded into the ~311ms of "AsymGEMM" kernel rows, NOT this memcpy line. The finding asserts this memcpy line IS the weight fetch without evidence.

3. "does not fully hide behind compute" is an ARTIFACT of the report's accounting, not a measurement. Line 755: `gpu_no_kernel = max(0, total - kernel_union - memcpy_union)` computes kernel_union and memcpy_union as INDEPENDENT unions (lines 745-746) and subtracts both, implicitly assuming memcpy is disjoint from kernels. Whether the transfer genuinely failed to overlap compute requires inspecting actual kernel-vs-memcpy interval overlap in the trace (line 911 merges them for gap computation, but the headline number does not). The finding presents non-overlap as a measured fact, which overreaches.

CONCLUSION: The directional intuition (CPU-resident weights add some transfer cost) is plausible and partly supported, but the specific evidentiary claims (zero torch transfer; this memcpy line = the weight fetch; proven non-overlap with GEMM) are unsubstantiated or contradicted. The recommendation (prefetch next K-tile to overlap the memcpy) may target the wrong line entirely, since weight streaming is likely inside the AsymGEMM kernel, not this memcpy bucket. Settling it would require a GPU run inspecting per-memcpy CopyKind (HtoD vs DtoD) and actual kernel/memcpy interval overlap in the nsys trace.


---

## ❌ Refuted false-alarms (NOT bugs — recorded so they aren't re-raised)


### R1. [I1] ~~lora_target=all wraps the MoE router (gate) as a dense LoRA target, then validation crashes setup~~ (claimed high/correctness)

- **Location cited:** `asym_gemm/integrations/lf.py:205, lf.py:128-138`
- **Original claim:** When LlamaFactory resolves lora_target=all, find_all_linear_modules returns leaf linear names that INCLUDE the Qwen3 MoE router `gate` (a nn.Linear sibling of the experts module), and patch_target_modules returns it unchanged for non-composite text models. apply_lf_asym_lora's dense loop matches the router via _matches_target (child=='gate'); the router is NOT under expert_prefixes, so it is wrapped as a trainable TorchLoRALinear. _validate_trainable_params then detects '.mlp.gate.' in the new '...mlp.gate.lora_A.default.weight' name and raises RuntimeError, aborting setup. Trigger: real LF run with lora_target=all (or any target list containing 'gate') on packed-format Qwen3 MoE.

**Why it is NOT a bug:** [refuted/none] The finding's causal chain rests on the false premise that the Qwen3 MoE router `gate` is an `nn.Linear`. In the packed format the finding explicitly names as the trigger (transformers/models/qwen3_moe/modeling_qwen3_moe.py in both the LF venv and third_party/transformers), the router is `self.gate = Qwen3MoeTopKRouter(config)` (line 279), a custom nn.Module holding `self.weight = nn.Parameter(...)` and using `F.linear` functionally — NOT an nn.Linear. The finding's own cited evidence "modeling_qwen3_moe.py:223 self.gate = nn.Linear(...)" is wrong; line 223 is actually `self.gate_up_proj = nn.Parameter(...)`. Verified empirically: `issubclass(Qwen3MoeTopKRouter, nn.Linear)` is False and "Linear" is not in its class name.

Path A (lora_target=all): find_all_linear_modules (misc.py:48) only collects leaves where `"Linear" in module.__class__.__name__`. `Qwen3MoeTopKRouter` fails this, so `gate` is never added. I built a real packed Qwen3MoeForCausalLM and ran the exact logic: result is ['k_proj','o_proj','q_proj','v_proj'] — gate absent. patch_target_modules (visual.py:213) returns target_modules unchanged for non-composite Qwen3 MoE. So the dense loop in lf.py never even matches the gate.

Path B (a target list explicitly containing 'gate', the finding's "any target list containing gate" claim): even then, lf.py:207 `if not isinstance(module, nn.Linear): report.skipped.append(...); continue` catches the packed Qwen3MoeTopKRouter (not nn.Linear), so it is appended to report.skipped with `:not_nn_linear:Qwen3MoeTopKRouter` and never wrapped as TorchLoRALinear. No `.mlp.gate.lora_A.` parameter is ever created, so `_validate_trainable_params` (lf.py:135) finds nothing to flag and raises no RuntimeError. Verified empirically: with dense_target_modules=['q_proj','gate'], `_matches_target` matches the gate modules but all are Qwen3MoeTopKRouter with isinstance(., nn.Linear)==False.

The integration is purpose-built for the packed format: is_qwen3_experts (qwen3_moe.py:51) requires 3D `gate_up_proj`/`down_proj` nn.Parameters, which only the packed Qwen3MoeExperts provides; in that format the sibling router is never nn.Linear. The non-packed ktransformers variant that does use `self.gate = nn.Linear` (archive/.../modeling_qwen3_moe.py:258) uses a ModuleList of separate MLP experts that is_qwen3_experts rejects, so it is not the model this code runs on.

In short, the `isinstance(module, nn.Linear)` guard at lf.py:207 is exactly the exclusion the finding claims is missing, and find_all_linear_modules never surfaces the router as a target. No setup crash occurs. The cited unit test does not "hide" a bug; the packed router is guarded regardless of whether 'gate' is in the dense target list. Could not execute apply_lf_asym_lora end-to-end because the CUDA _C extension fails to import on this CPU box (GLIBC_2.38 mismatch), but every branch in the relevant pure-Python control flow was reproduced verbatim against a real packed Qwen3 MoE model, which is sufficient to settle this statically.

**Why it is NOT a bug:** [refuted/none] The finding's core premise is false for the packed-format Qwen3 MoE that AsymGEMM actually targets and that is installed in this repo (transformers 5.6.0). I traced the full call chain and empirically tested it.

Call chain (LlamaFactory/src/llamafactory/model/adapter.py): when lora_target=all, line 206 calls find_all_linear_modules, line 213 calls patch_target_modules, and the result is passed to apply_lf_asym_lora as dense_target_modules (line 261).

find_all_linear_modules (model_utils/misc.py:48-49) only collects a module's leaf name if `"Linear" in module.__class__.__name__ and "Embedding" not in ...`. The finding assumes the router is `self.gate = nn.Linear(...)` (the OLD unpacked transformers format, modeling_qwen3_moe.py:223 in those installs). But is_qwen3_experts (qwen3_moe.py:50-62) only matches the PACKED format (3D nn.Parameter gate_up_proj/down_proj + forward signature hidden_states/top_k_index/top_k_weights). In the installed packed format (.venv transformers 5.6.0, modeling_qwen3_moe.py:254-279), the router is `self.gate = Qwen3MoeTopKRouter(config)` — NOT an nn.Linear. Qwen3MoeTopKRouter stores `self.weight = nn.Parameter(...)` (line 261) and calls `F.linear` functionally (line 265); its class name contains no "Linear" and it has no nn.Linear children. So find_all_linear_modules never adds `gate`.

I confirmed empirically by building a real tiny Qwen3MoeForCausalLM with the installed transformers and replicating find_all_linear_modules: it returned ['k_proj','o_proj','q_proj','v_proj'] with `gate in targets? False`, and `model.layers.0.mlp.gate -> Qwen3MoeTopKRouter | is nn.Linear: False`. patch_target_modules (visual.py:213) returns target_modules unchanged for non-composite qwen3_moe, but that's moot since `gate` was never in the list.

Therefore the dense loop's _matches_target (lf.py:205, 92-97) never matches the router: child `gate` is not in dense_target_modules, and the module name `...mlp.gate` does not end with any target. The router is never wrapped as TorchLoRALinear, stays frozen via freeze_non_lora_params, and _validate_trainable_params (lf.py:128-138) does not raise.

The finding is internally inconsistent: in the only format where the router IS an nn.Linear (old nn.ModuleList format), `experts` is a ModuleList of Qwen3MoeMLP which is_qwen3_experts does NOT detect, so with strict=True apply_lf_asym_lora raises at lf.py:188-189 ("found no Qwen3 packed expert modules") BEFORE the dense loop — again no router-wrapping crash via the described path.

The unit test (test_lf_qwen3_asym_backend.py:262-288) passes dense_target_modules WITHOUT 'gate' precisely because that mirrors what find_all_linear_modules really returns; it is faithful, not hiding a bug. It even uses a HARDER FakeModel router that IS nn.Linear (test line 49) and still asserts the router stays frozen (line 285).

No GPU run needed; resolved statically plus a CPU model-construction check.

**Why it is NOT a bug:** [refuted/none] The finding's core premise is false for the packed-format Qwen3 MoE it explicitly names as the trigger. I traced the full resolution chain through the real code.

(1) PACKED ROUTER IS NOT nn.Linear. The transformers version actually installed in both relevant Python environments (kevin_asymgemm and the LF env), matching /third_party/transformers/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py:279, defines `self.gate = Qwen3MoeTopKRouter(config)`. Qwen3MoeTopKRouter (lines 254-272) stores its weight as a raw `self.weight = nn.Parameter(...)` (line 261) and computes `F.linear(...)` directly. There is NO nn.Linear submodule named `gate`. find_all_linear_modules (misc.py:48) only adds a module when `"Linear" in module.__class__.__name__ and "Embedding" not in ...`; `"Linear"` is NOT in `"Qwen3MoeTopKRouter"`. So `lora_target=all` does NOT produce `gate` in the target list for packed-format Qwen3 MoE. patch_target_modules (visual.py:213) returns target_modules unchanged for non-composite text models, but `gate` was never in the list to begin with. The evidence chain breaks at find_all_linear_modules.

(2) THE CITED EVIDENCE IS THE WRONG MODEL LAYOUT. The finding cites `modeling_qwen3_moe.py:223 self.gate = nn.Linear(...)`, which is an OLDER/non-packed layout. That layout is inconsistent with the finding's own stated trigger ('packed-format Qwen3 MoE'). Moreover, in the non-packed layout the experts are not 3D-packed parameters, so is_qwen3_experts (qwen3_moe.py:51, which requires 3D `gate_up_proj` AND `down_proj` params) returns False and the AsymGEMM expert-wrapping path does not engage as the finding describes.

(3) EVEN THE SECONDARY TRIGGER DOES NOT CRASH. If a user explicitly passes a target list containing `gate` on a packed model, _matches_target (lf.py:97) would match the router module by name, but lf.py:207 `if not isinstance(module, nn.Linear): report.skipped.append(...); continue` skips the packed Qwen3MoeTopKRouter (not an nn.Linear). It is never wrapped as TorchLoRALinear, so no `...mlp.gate.lora_A...` parameter is ever created and _validate_trainable_params (lf.py:128-138) does not fire.

(4) ROUTER IS CORRECTLY FROZEN. freeze_non_lora_params (lf.py:237 -> lora.py:348-350) sets requires_grad=False on every non-LoRA param including the packed router's gate.weight, which is the intended behavior; validation then passes.

(5) THE TEST IS NOT HIDING A BUG. test_apply_lf_asym_lora_wraps_experts_dense_and_freezes_router models the router as nn.Linear in its FakeBlock but passes dense_target_modules WITHOUT `gate` (line 267) — which matches what the real packed-format find_all_linear_modules actually produces (no `gate`). It also asserts `not model.layers[0].mlp.gate.weight.requires_grad`, confirming the router-frozen behavior. The test is consistent with reality, not a cover-up.

The claimed crash cannot occur on the packed-format model named as the trigger, and is already structurally prevented (isinstance guard + freeze) for the explicit-target case. Refuted.


### R2. [I1] ~~Asym MoE path is incompatible with the installed (ModuleList) Qwen3 MoE format and hard-fails under strict~~ (claimed medium/correctness)

- **Location cited:** `asym_gemm/training/qwen3_moe.py:50-62, asym_gemm/integrations/lf.py:188-189`
- **Original claim:** is_qwen3_experts requires packed 3D gate_up_proj/down_proj Parameters plus num_experts/hidden_dim/intermediate_dim attrs. The transformers version present in this environment uses the legacy nn.ModuleList of per-expert Qwen3MoeMLP (gate_proj/up_proj/down_proj), for which is_qwen3_experts returns False. With wrap_experts True (target hits experts/all) and strict=True, no experts are found and lf.py:189 raises 'found no Qwen3 packed expert modules', so the asym MoE path cannot run on this transformers build. This is a version-coupling fragility, not a logic bug, but it silently depends on a transformers version exposing the packed format.

**Why it is NOT a bug:** [refuted/none] The finding's central factual premise is false. It claims the transformers version present in this environment uses the legacy nn.ModuleList of per-expert Qwen3MoeMLP (with gate_proj/up_proj/down_proj) for which is_qwen3_experts returns False, causing lf.py:189 to raise. I verified the actual installed code in both relevant builds:

1. LlamaFactory .venv transformers (5.6.0) at /home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory/.venv/lib/python3.11/site-packages/transformers/models/qwen3_moe/modeling_qwen3_moe.py
2. Vendored third_party/transformers (5.8.0.dev0) at /home/shutianluo/kevin/AsymGEMM-SFT/third_party/transformers/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py

Both use the PACKED Qwen3MoeExperts format, not a ModuleList of MLPs. The MoE block wires it directly: line 278 `self.experts = Qwen3MoeExperts(config)`. Inside Qwen3MoeExperts (lines 215-251):
- line 223 `self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))` and line 224 `self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))` -> both 3D nn.Parameter, so _is_3d_parameter returns True.
- lines 220-222 set num_experts/hidden_dim/intermediate_dim as int attrs -> passes is_qwen3_experts int checks.
- line 225 `self.act_fn = ACT2FN[config.hidden_act]` -> callable check passes.
- lines 227-231 `def forward(self, hidden_states, top_k_index, top_k_weights)` -> all three required param names present.

Therefore is_qwen3_experts(module) returns True for the installed format, experts ARE found, and lf.py:188-189 `if not expert_replacements and strict: raise ValueError(...)` does NOT trigger. The asym MoE path runs on this build.

The finding's evidence quote is itself wrong: it states "modeling_qwen3_moe.py:205-207 defines per-expert self.gate_proj/up_proj/down_proj inside a ModuleList (line 224)". In reality lines 204-206 belong to the UNUSED Qwen3MoeMLP class (line 198), and line 224 is `self.down_proj = nn.Parameter(...)`, a packed 3D parameter, the exact opposite of the claim. Qwen3MoeMLP is only instantiated at line 319 as the dense MLP for non-sparse layers, never as a per-expert ModuleList; the only nn.ModuleList (line 459) is the decoder layers list. No conditional/legacy ModuleList expert path exists in either build.

The cited AsymGEMM code (qwen3_moe.py:50-62, lf.py:188-189) matches as described, but the version-incompatibility claim that makes it a finding is contradicted by the actual installed code. Refuted.


### R3. [I2] ~~Redundant/unreachable bf16 check: the `bf16 or pure_bf16` guard is dead given the stricter pure_bf16 mandate two lines below~~ (claimed low/improvement)

- **Location cited:** `src/llamafactory/hparams/parser.py:353`
- **Original claim:** Lines 353-354 raise when NOT (training_args.bf16 or finetuning_args.pure_bf16), but lines 355-356 immediately raise when NOT finetuning_args.pure_bf16. Since pure_bf16=True implies (bf16 or pure_bf16)=True, the only way the line-353 condition can be True is when pure_bf16 is False — in which case line 355 already raises. Thus line 353 can never be the line that actually fires; it is dead code and its error message (which implies bf16 alone suffices) is misleading because asym in fact mandates pure_bf16. Not a correctness bug, but it muddies the intended contract.

**Why it is NOT a bug:** [refuted/none] I read the cited code at /home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory/src/llamafactory/hparams/parser.py:353-356 and verified pure_bf16/bf16 are plain bools (finetuning_args.py:456 `pure_bf16: bool = field(default=False, ...)`; training_args.bf16 is HF's bool). The exact code:

353  if not (training_args.bf16 or finetuning_args.pure_bf16):
354      raise ValueError("AsymGEMM BF16 LoRA-SFT requires `bf16=true` or `pure_bf16=true`.")
355  if not finetuning_args.pure_bf16:
356      raise ValueError("AsymGEMM smoke training requires `pure_bf16=true` ...")

The finding's central, load-bearing claim is that line 353 is "dead code" that "can never be the line that actually fires" because line 355 "already raises." This is a misreading of control-flow ORDER. Line 353 PRECEDES line 355. Truth table over B=bf16, P=pure_bf16:
- B=F,P=F: line 353 condition `not(F or F)` = True -> line 353 RAISES FIRST; line 355 never reached. So line 353 IS reachable and IS the firing line here.
- B=T,P=F: line 353 `not(T or F)`=F skip; line 355 `not F`=True raises. Caught only by 355.
- B=F,P=T: both skip (valid). B=T,P=T: both skip (valid).

Thus line 353 fires for the both-false input, and line 355 fires for the bf16-only input — both lines are reachable and each catches a distinct input region. Line 353 is NOT dead code, contradicting the finding's stated mechanism. I also confirmed (grep) there is no earlier guard inside the `use_asym_gemm` block (entered at line 337 for any bf16/pure_bf16 values) that would prevent the B=F,P=F state from reaching line 353.

The only residual real issue is cosmetic: line 353's error message ("requires bf16=true OR pure_bf16=true") is slightly misleading because line 355's stricter mandate means bf16-alone is in fact rejected. That is a minor clarity nit, not dead/unreachable code. Because the finding's core technical assertion (unreachable dead code that can never fire) is factually incorrect, I refute it. The cleanup recommendation (collapse to the single pure_bf16 mandate and fix the message) is harmless but the justification given is wrong.


### R4. [I2] ~~asym arg-validation block is monolithic and not factored like KT's apply_kt_config, increasing drift risk~~ (claimed low/design)

- **Location cited:** `src/llamafactory/hparams/parser.py:337`
- **Original claim:** All asym validation lives as an inline ~40-line block in get_train_args, whereas KT factors its config handling into KTransformersArguments.apply_kt_config / get_kt_config_dict (model_args.py:494-538) invoked at parser.py:564. The asym checks are correct and complete for the listed unsupported features, but co-locating them with the dataclass (as KT does) would keep the two backends symmetric and reduce the chance that a future edit updates one guard set and not the other. Pure maintainability note; no behavioral issue.

**Why it is NOT a bug:** [refuted/none] The finding's line-number citations are literally accurate, but its core design premise is a misreading that compares two different categories of code. Verified facts: (1) parser.py:337-378 is an inline `if model_args.use_asym_gemm:` validation block of ~40 lines; (2) `apply_kt_config` is defined at model_args.py:508 and called at parser.py:564 `if model_args.use_kt: model_args.apply_kt_config(...)`; (3) `AsymGEMMArguments` (model_args.py:541-568) holds only fields.

However, `apply_kt_config` is NOT validation — its body (model_args.py:513-538) sets `os.environ[...]` keys and mutates `hf_kt._kt_config`; it is config/integration application with side-effects, which is why it lives on the dataclass and runs late at parser.py:564 after `transformers.set_seed`. KT's actual validation/guards are themselves inline in `get_train_args`, scattered exactly alongside the asym block: version checks at parser.py:194-197, the lora-reward-model guard at 326-327, the AsymGEMM/KT incompatibility guard at 357-358, and the DeepSpeed ZeRO-3 guard at 442. So KT does NOT factor its validation into a dataclass method — both backends already keep validation inline in `get_train_args`, i.e. they are symmetric with respect to validation.

The finding asserts an asymmetry ("KT factors its config handling ... whereas all asym validation lives as an inline block") and treats `apply_kt_config` as the symmetric counterpart of the asym validation block. That equivalence is false: apple (asym validation) vs orange (KT config-application). Following the recommendation — moving asym validation onto an `AsymGEMMArguments.validate_train_args` method — would actually BREAK the existing inline-validation symmetry with KT, not restore it. Additionally, asym has no need for an `apply_kt_config` analogue: unlike KT (which injects env vars for accelerate/transformers), asym reads its args directly at the consumption site (adapter.py:252-265), so there is no config-application step to factor out.

Because the claimed asymmetry/drift-risk rests on a category error and a false factual premise ("KT factors validation"), the finding is refuted even as a maintainability note.


### R5. [K2] ~~Single-buffered B serializes next-K-block fetch behind the full M-sweep (no B prefetch overlap)~~ (claimed medium/efficiency)

- **Location cited:** `asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh:112,471`
- **Original claim:** B (and SFB) use a single shared-memory slot with a single full/empty barrier pair (smem_b[0], full_barriers_b[0]/empty_barriers_b[0]). The TMA-load warp waits empty_barriers_b before loading the next K-block's B, but empty_barriers_b is only released by the MMA warp after the ENTIRE inner M-loop for the current K-block completes (umma_arrive on empty_barriers_b after the m-loop). With B fetched over NVLink-C2C / PCIe from CPU DRAM (the whole point of asym), this means the next K-tile's B fetch latency is not overlapped with compute of the current K-tile, exposing the slow host fetch on the critical path between K-blocks.

**Why it is NOT a bug:** [refuted/low] I read the cited FP8 kernel (asym_gemm/include/asym_gemm/impls/sm100_fp8_asym_gemm_1d1d.cuh) and the FP4 analog in full, plus the scheduler, the host JIT wrapper, and the Python integration that drives these kernels.

STRUCTURAL CLAIMS ARE ACCURATE:
- B is single-buffered. FP8 line 112 `constexpr uint32_t SMEM_B_SIZE = SMEM_B_SIZE_PER_STAGE;` (not *kNumStages), with a single barrier triple initialized at lines 202-204 (`full_barriers_b[0]->init(1); empty_barriers_b[0]->init(1);`). The PatternVisitors for full/empty_barriers_b (lines 178-179) always return index [0]. FP4 mirrors this (line 137, init at 233).
- The MMA warp releases `empty_barriers_b[0]` only AFTER the inner M-loop: line 471 `umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers_b[0]));` sits after the `for (m_block_idx ...)` loop closing brace. FP4 identical at line 505.
- The TMA-load warp waits `empty_barriers_b[0]->wait(phase_b ^ 1)` (line 264 FP8 / 299 FP4) at the top of each K-block before loading B. So the next K-block's B TMA genuinely cannot be issued until the whole current-K M-sweep finishes — B is not prefetched across K-blocks. That reading is correct.

BUT THE HARM PREMISE IS FALSE FOR THE CITED KERNELS:
The finding's severity rests entirely on B being TMA-loaded directly from CPU DRAM over NVLink-C2C/PCIe during the kernel, so the un-prefetched B fetch exposes slow host-fetch latency on the per-K-block critical path. That is not what happens for the FP8/FP4 asym kernels under review. In asym_gemm/training/frozen_linear.py, the only callers of m_grouped_fp8/fp4_asym_gemm_nt_contiguous (lines 621/633/679/691) first stage B to device HBM: lines 613-614 (and 668-669) `b_values = _stage_quantized_tensor_for_kernel(qweight.values, a.device)`, with `_stage_quantized_tensor_for_kernel` (line 588-591) doing `tensor.to(device=device, ...)` where device == a.device (CUDA during training). The accompanying comment (lines 610-612) is explicit: "Quantized kernels are currently more reliable when the packed cache tensors are staged to CUDA before launch; the source cache remains CPU-resident and frozen." The host JIT wrapper (csrc/jit_kernels/impls/sm100_fp8_asym_gemm_1d1d.hpp:133) builds tensor_map_b straight from this already-on-device `b`. So the TMA reads B from HBM, not from host memory. The host-resident-B path is the BF16 kernel (sm100_bf16_asym_gemm.cuh, called via _asym_bf16_nt which passes b_cpu directly) — a different file the finding does not cite.

Given B comes from HBM, the residual cost is only the small, occasional un-hidden HBM B-tile load at the start of each K-block: a single 64KB tile (FP8: LOAD_BLOCK_N=128 * BLOCK_K=512 * 1B) consumed once per K-block, with BLOCK_K=512 (host wrapper line 106) giving only a handful of K-blocks for typical K, while each K-block's M-sweep runs many MMA tiles over the entire expert M range (A is properly multi-buffered with kNumStages). That latency is negligible relative to compute and is not shown to be on the critical path; the finding itself concedes "no such measurement currently exists."

So the code observation is real but the medium-severity host-fetch bottleneck it describes does not exist on these code paths (B is HBM-resident, not host-fetched). At most this is a minor HBM-prefetch micro-optimization of unproven benefit, not a confirmed medium efficiency defect. Refuted as stated.


### R6. [K4] ~~asymScheduler has no guard for the -1 sentinel expert; correctness depends entirely on host launching gridDim.y = num_active (list_size-1)~~ (claimed medium/correctness)

- **Location cited:** `asym_gemm/include/asym_gemm/common/asymScheduler.cuh:95-107`
- **Original claim:** In the contiguous/MGrouped path the constructor unconditionally does expert_id = experts[blockIdx.y]; n_start = expert_id*blocks_perExpert; m_start=ceil_div(offsets[blockIdx.y*2],BLOCK_M); m_end=ceil_div(offsets[blockIdx.y*2+1],BLOCK_M). There is NO check for the -1 sentinel. The metadata convention (see _dense_offsets / _group_metadata_tensors) gives experts.numel()=num_active+1 (last = -1) but offsets only holds 2*num_active pair entries. If the compiled binding ever launches gridDim.y = list_size (sentinel included) instead of list_size-1, blockIdx.y=num_active reads experts[num_active]=-1 -> expert_id=0xFFFFFFFF and offsets[2*num_active], offsets[2*num_active+1] which are OUT OF BOUNDS; n_idx = blockIdx.x*BLOCK_N + shape_n*0xFFFFFFFF overflows and the CTA would issue OOB TMA loads/stores. The kernel-visible code cannot self-protect against this; it is silently delegated to the host grid computation.

**Why it is NOT a bug:** [refuted/low] I read the cited scheduler and every host launch path. The finding's factual code claims are accurate, but the described correctness failure (OOB reads from a sentinel CTA) does NOT occur in the actual code; it is purely hypothetical ("IF the compiled binding ever launches gridDim.y = list_size").

What is true:
- asymScheduler.cuh:95-107 unconditionally does `expert_id = experts[blockIdx.y]; n_start = expert_id*blocks_perExpert; m_start = ceil_div(offsets[blockIdx.y*2],BLOCK_M); m_end = ceil_div(offsets[blockIdx.y*2+1],BLOCK_M); n_idx = blockIdx.x*BLOCK_N + shape_n*expert_id;` with no `-1`/sentinel guard.
- The metadata convention (frozen_linear.py:417-441 `_group_metadata_tensors`, moe.py:2407-2423 `build_contiguous_metadata`, test `_dense_offsets`) indeed uses experts.numel()=num_active+1 (last=-1) with 2*num_active offset entries.
- The kernel's `m_start >= m_end` early-exit exists (sm100_bf16_asym_gemm.cuh:211), which the recommendation correctly leverages.

Why it is refuted as a correctness defect: every host path computes gridDim.y to exclude the sentinel, so a sentinel CTA never launches and the OOB read never happens:
- Tensor-list_size overloads (gemm.hpp:254-263, 544-562): `grid_y = num_groups` (= b.shape[0], the real expert count); blockIdx.y in [0,num_groups) so the sentinel at index num_groups is never read.
- Int-list_size overloads (gemm.hpp:304, 602) and FP4 host (sm100_fp4_asym_gemm_1d1d.hpp:195): `grid_y = list_size - 1`; with list_size = num_active+1 the sentinel sits at index list_size-1 = grid_y, never reached. The FP8/BF16 1d1d host functions merely pass grid_y through (sm100_fp8_asym_gemm_1d1d.hpp:87,150; sm100_bf16_asym_gemm.hpp:227) and don't recompute it.
- Production callers set list_size = experts.numel() = num_active+1 consistently (frozen_linear.py:555,675; moe.py:2422 list_size=num_experts+1).

So the OOB/overflow scenario requires a host grid-computation bug that does not exist in the current tree. The claimed "medium correctness" defect is not present. The residual point — the kernel cannot self-protect and relies on an unverified host invariant — is a legitimate but minor defensive-coding/robustness suggestion (add a sentinel guard or assert gridDim.y == num_active), not a correctness bug, hence corrected_severity 'low'.


### R7. [L1] ~~Fallback statistics (record_fallback / fallback_reasons) are dead code; fallback bookkeeping never populates~~ (claimed medium/correctness)

- **Location cited:** `asym_gemm/training/frozen_linear.py:906,984,56`
- **Original claim:** VALID_BACKENDS is only ('asym','torch'). In _dispatch_nt/_dispatch_grouped_nt the `record_fallback` calls (906, 984) are reachable only inside the `backend != 'torch'` block, i.e. only when backend=='asym'. But on every code path that reaches record_fallback in that block, the very next statement raises (907-916 / 985-994) because `backend == 'asym'`. On a direct RuntimeError the code re-raises at 904/982 before recording. Consequently fallback_reasons is always empty, and torch_forward_calls/torch_dx_calls only increment for a deliberate full-torch run, never as an asym->torch fallback. The AsymExecutionStats fallback machinery therefore reports nothing meaningful.

**Why it is NOT a bug:** [refuted/none] The finding's core claim is that record_fallback (frozen_linear.py:906/984) is dead because the very next statement raises (907/985) while backend=='asym', so "record_fallback's effect is discarded by the exception" and "fallback_reasons is always empty." This is a misreading of Python exception semantics.

Tracing _dispatch_nt (lines 871-923): the `if backend != "torch":` block is entered only for backend=='asym' (VALID_BACKENDS=("asym","torch"), _check_backend enforces this). Line 873 computes `reason` via _direct_bf16_reason, which returns a non-None string (e.g. "requires_8_aligned_nk", line 287) when the direct kernel is unusable. When reason is non-None, the `if reason is None:` try-block (877-916... actually 877-904) is skipped, and control reaches:
  905    if stats is not None:
  906        stats.record_fallback(f"{phase}:{reason}")
  907    if backend == "asym":
  908        raise RuntimeError(...)

record_fallback (line 56-57) does `self.fallback_reasons[reason] = ... + 1` — an in-place mutation of the caller-supplied stats object. That mutation COMPLETES at line 906 before the raise at line 908. An exception raised by a subsequent statement does NOT undo a mutation already applied to a caller-held mutable object. The caller (asym_frozen_linear / AsymFrozenLinearFunction, which threads `stats=stats` in at line 1039 and stores ctx.stats at 1052) still holds that same stats reference after the exception unwinds.

This is proven empirically by the repo's own tests:
  - test_cpu_resident_frozen_base.py:819-827: calls asym_frozen_linear(..., backend="asym", stats=stats), expects RuntimeError("...requires_8_aligned_nk"), and asserts `stats.fallback_reasons == {"forward:requires_8_aligned_nk": 1}`.
  - test_cpu_resident_frozen_base.py:844-851: backward path, asserts `stats.fallback_reasons == {"dx:transpose_b_requires_64_aligned_k": 1}`.
Both assert fallback_reasons IS populated — the exact opposite of "always empty." fallback_reasons is also surfaced in production via lf.py:45-57 runtime_log_string.

So record_fallback is live, reached on a real and tested code path (asym backend + direct-kernel-unavailable), and populates fallback_reasons before the informative RuntimeError is raised. This is a deliberate design: record WHY asym was unavailable, then raise to surface it.

The only sub-claim that is technically accurate is that torch_forward_calls/torch_dx_calls increment only on the deliberate `backend=="torch"` path (918-922) or the separate moe.py:1817 reference path, never as an asym->torch fallback — but that is by design (no auto/hybrid backend exists), not a bug, and does not make the fallback machinery "report nothing meaningful." The recommendation (add an 'auto' backend) is a design preference, not a correctness defect.

Verdict: refuted. The cited code does not exhibit the claimed problem; the premise that the trailing raise nullifies record_fallback is false and contradicted by passing tests.


### R8. [L5] ~~Operator benchmark methodology gaps: no L2 flush, CUDA-graph may freeze the per-call fetch path, and FP8 baseline is asymmetric — can distort absolute asym-vs-torch numbers~~ (claimed medium/design)

- **Location cited:** `scripts/lora/profile_lora_ops.py:196-249 (measure / measure_cuda_graph) and scripts/lora/profile_transpose.py:183-212 (measure)`
- **Original claim:** Neither measure() nor measure_cuda_graph() flushes the L2 cache between iterations, so the small repeated operands stay cache-resident and timings are optimistic for both backends — but more so for the small torch baseline weight, which biases the asym/torch ratio. CUDA-graph capture of the asym path can also bake in a single static fetch/quantization, potentially under-counting the real per-call CPU-fetch cost the design pays in training. In profile_transpose.py the FP8 comparison times torch in BF16 vs asym in FP8 and re-quantizes activations inside the timed asym call but not for torch.

**Why it is NOT a bug:** [refuted/low] I read both cited files in full plus the asym backend (asym_gemm/training/frozen_linear.py), the kernel header (csrc/apis/gemm.hpp), and both driver scripts. The finding bundles three sub-claims; assessed individually:

CLAIM 2 (headline, novel): "CUDA-graph capture of the asym path can bake in a single static fetch/quantization, under-counting the real per-call CPU-fetch cost." REFUTED on the merits. The "asym" path performs NO Python/host-side H2D weight copy. In _asym_bf16_nt (frozen_linear.py:510-532) the CPU-pinned weight b_cpu is handed straight to asym_gemm.m_grouped_bf16_asym_gemm_nt_contiguous; the host->device fetch happens device-side inside the GEMM kernel (TMA from pinned host memory). These kernels are explicitly designed to be capture-compatible — see csrc/apis/gemm.hpp:104-106: "Validate the list_size tensor ... without copying it to host. The host can no longer read the value (capture-mode incompatibility)". Therefore graph.replay() re-runs the kernel, INCLUDING its device-side per-call host fetch, on every replay; nothing is "baked in." Activation quantization only exists for fp8/fp4 (_quantize_activation_for_precision, line 563-572) and is itself a device-side CUDA kernel that is captured and replayed each iter. So the graph does not under-count fetch/quant cost. This sub-claim rests on a misreading of how AsymGEMM moves weights.

CLAIM 1 (no L2 flush): Factually TRUE — measure() (profile_lora_ops.py:208-215), measure_cuda_graph() (218-249), and profile_transpose.py measure() (197-205) record events back-to-back with no torch.cuda._sleep/L2 scratch flush (grep for _sleep/l2/flush/scratch finds nothing; the lone empty_cache at line 152 is input setup). But the claimed directional bias on the asym/torch ratio is weakly supported and partly contradicted: default driver shapes are large (profile_transpose.sh default SHAPES=8192|8192|8192 → BF16 weight = 128 MiB, far exceeding any L2 (~50 MiB); profile_lora_ops down-style up to 1536x4096 ≈ 12 MiB), both backends reuse the same activation each iter, and the asym weight is re-fetched from host regardless of L2. This is generic microbenchmark hygiene, not a result-distorting defect.

CLAIM 3 (fp8 asymmetry): Factually TRUE — asym_fp8_call (profile_transpose.py:159-180) calls quantize_fp8_activation(a) inside the timed lambda while torch_nt times plain bf16 torch.mm. But this is EXPLICITLY DISCLOSED by the script's own output at lines 569 ("precision: torch=bf16, asym={precision}"), 571, and 573 ("fp8 note: AsymGEMM quantizes activations per call ..."), only triggers under --precision fp8 (both drivers default to bf16), and faithfully reflects the real training cost (training's _asym_quantized_nt also quantizes per call). The finding itself concedes this is "already partially noted."

Net: the one genuinely novel, load-bearing claim (CUDA-graph baking in static fetch) is wrong; the other two are true facts that are either low-value hygiene with speculative/possibly-wrong directional impact or already documented and intentional. No undisclosed defect that distorts asym-vs-torch numbers. The recommendation's only defensible residual is "add an L2 flush," a minor niceness.


### R9. [L5] ~~E2E profiling uses synthetic 'learned'/'balanced' route patterns and 1-layer (--profile-layers 1/4); real-workload generality of the asym-vs-torch ratio is unconfirmed~~ (claimed low/design)

- **Location cited:** `profiling/lora_e2e_bf16/qwen3-235b-a22b-instruct-2507-l4__b8_s2048_r64_a128/asym__nsys__norecomp__polnone/command.txt (--moe-route-pattern learned --profile-layers 1)`
- **Original claim:** The MoE e2e runs profile a single (or 4) layer(s) with a synthetic route pattern rather than real router logits on real data. Expert token counts drive AsymGEMM tile occupancy and the per-expert M-extent, so a synthetic balanced/learned distribution can over- or under-state both the asym fetch-reuse benefit and the torch baseline's grouped-GEMM efficiency. The conclusions are directionally trustworthy but the exact ratio may shift with real routing skew.

**Why it is NOT a bug:** [refuted/none] The finding's central technical claims are contradicted by the code I read.

1) "synthetic 'learned' route patterns ... rather than real router logits": FALSE for the run actually performed. The inner command (job.log line 1) and command.txt both use `--moe-route-pattern learned`. In profile_lora_e2e.py:3175-3176, `learned` sets `static_routes = None`. With `static_routing=None`, moe.py `_route` (lines 1789-1793) runs the REAL router: `logits = F.linear(flat.float(), self.router_weight)` then `topk_routing_from_logits(...)`. The `router_weight` is the genuine pretrained Qwen3 gate: profile_lora_e2e.py:3834 reads `model.layers.{idx}.mlp.gate.weight` from the HF safetensors snapshot and assigns it at line 3878 (`layer_state["router_weight"] = checked(...)`). Expert weights (gate_proj/up_proj/down_proj) are also real HF tensors (lines 3837-3897). So the router logits are real, not synthetic.

2) "synthetic per-expert median_tokens ... rather than measured router output": FALSE. The per-expert counts come from `metadata.expert_counts` of the real MoE execution: record_route_metadata (lines 481-485) does `counts = metadata.expert_counts.tolist()`, called from the real layer forward at lines 2817-2819 using `details["metadata"]` produced inside `_run_moe`. route_summary (lines 586-597) computes median_tokens from these MEASURED counts. The job.log shows a clearly non-uniform real distribution (median_tokens 611-1358), exactly what a real router yields. Moreover, the evidence values cited in the finding (408.5, 343.0, 543.0, 601.0) do NOT appear in this workload's log (which shows 781.0, 1035.0, 900.5, 1239.5, ...), so the cited evidence does not match the cited location.

3) "--profile-layers 1" and "extrapolates a single-layer profile to per-layer cost": The driver command.txt has `--profile-layers 1`, but the actually-executed inner command used `--profile-layers 4` (job.log line 1). The script profiles N real stacked layers (config.num_layers, line 3816/3827) extracted from the real model; there is no code that multiplies a single-layer time up to the full model's layer count. The 'extrapolation' claim is unsupported by any artifact (the combined/ dir contains plots and a sweep index, no summary doc making such an extrapolation).

The only genuinely accurate residual is narrow and weaker than stated: the INPUT activations are synthetic random Gaussian (`x = torch.randn(...) * 0.5`, line 3191), so the real router is driven by random inputs rather than real tokenized text, and only a few layers are profiled. That is a minor design caveat, but the finding's framing (synthetic routes, synthetic logits, single-layer extrapolation) misrepresents the mechanism, which uses real pretrained router/expert weights and measured router outputs. Because the described problem is contradicted by the code, the verdict is refuted; the surviving real-data-generality caveat is at most low/none severity and is already partially mitigated (real router, real weights, real measured token distribution).

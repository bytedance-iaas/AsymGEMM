# AsymGEMM notes: DeepGEMM lineage, C2C reality, e2e bottleneck

## 1. DeepGEMM → AsymGEMM kernel transformation
DeepGEMM ships both fp8 AND bf16 SM100 grouped kernels (not fp8-only). AsymGEMM = one
"asym-ification" applied to both (fp8, bf16, fp4). AsymGEMM replaced DeepGEMM's SM100 fp8/bf16
kernels (originals gone).

File map (SM100): `sm100_fp8_gemm_1d1d.cuh`→`sm100_fp8_asym_gemm_1d1d.cuh` (567→681);
`sm100_bf16_gemm.cuh`→`sm100_bf16_asym_gemm.cuh` (437→1036); new `sm100_bf16_cpu_left_asym_gemm.cuh`
(454); `common/scheduler.cuh`(`Scheduler`)→`common/asymScheduler.cuh`(`asymScheduler`).

The shared transformation (identical fp8 & bf16):
1. Signature `int* grouped_layout` → `uint32_t* offsets, experts` (per-expert token ranges + weight-group ids; masked mode reuses `offsets` as `masked_m`, experts=nullptr).
2. Grid: persistent `num_sms` 1D → static 2D `(ceil(n/BLOCK_N) × grid_y=experts)`; `blockIdx.x`=N-tile, `blockIdx.y`=expert. (host: `sm100_fp8_asym_gemm_1d1d.hpp:160` vs DeepGEMM `:132`.)
3. Scheduler: `get_next_block()` tile-queue → `asymScheduler` computes `m_start/m_end/n_idx` from blockIdx (asymScheduler.cuh:79-109).
4. Loop inversion: K-inner/persistent → **K-outer, operand-stationary** (load one operand once per K, stream the other across the segment).
5. Consequence: K outer ⇒ partial sums can't stay in TMEM ⇒ epilogue `k==0 ? STORE : REDUCE_ADD` into HBM (fp8 `:648-657`, bf16 `:984-990`). DeepGEMM used plain store (K-inner, full TMEM reduce).
6. Empty-expert early-exit (+TMEM free); flexible BLOCK_K via per-atom desc rebasing.

FP8-specific kept: UE8M0 scale factors, UTCCP transpose warp (warp 2), block-scaled MMA. → 4 warp roles.
BF16 = same transform on DeepGEMM-bf16, OR fp8-asym minus all SF machinery (drop warp-2, SF TMAs/barriers; `make_instr_desc_block_scaled`→`make_instr_desc`; `SM100_MMA_MXF8F6F4_SS`→`SM100_MMA_F16BF16_SS`). → 3 warp roles. "asym" = CPU×HBM placement, NOT precision.

Two bf16 flavors differ only in which operand is stationary:
- `sm100_bf16_asym_gemm.cuh`: **B-stationary** (weight loaded once, tokens stream) — MoE forward.
- `sm100_bf16_cpu_left_asym_gemm.cuh`: **A-stationary, CPU-left** (A=host activation fetched once, B=GPU streamed). The offload kernel.
Backward variants: `cpu_right` / `cpu_source` — same idea, offloaded activation on the other side.

## 2. CPU→SMEM mechanism (no HBM bounce)
TMA bulk-tensor copy whose descriptor's "global" address is a **pinned host pointer**.
- Host: `a.is_cpu() && a.is_pinned()` enforced (gemm.hpp:635-636). `cuTensorMapEncodeTiled(..., a.data_ptr())` bakes the pinned host addr as global base (runtime_utils.hpp:116-120) — no host/device branch. Valid via UVA/NVLink-C2C.
- Device: `tma_copy` → `cute::SM90_TMA_LOAD_2D::copy` → PTX `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` (copy_sm90_tma.hpp:125). TMA streams CPU→SMEM directly, signals mbarrier. No cudaMemcpy, no HBM staging.
- A loaded ONCE per (M-tile,K-block), reused across all N; B (GPU) double-buffered.

## 3. C2C bandwidth — MEASURED on GB200 (not spec)
Spec ~900 GB/s = bidirectional aggregate (→ ~450/dir theoretical). REAL achievable:
- H2D one-way, local Grace: **~195 GB/s**. Remote Grace: **~117**. D2H: ~179. Bidirectional together: **~314**.
- Concurrency does NOT help one direction: 1/2/4/8 streams all flat ~196 → hard ceiling (Grace LPDDR read bw + copy path, not link/engines).
- NUMA: GB200 = 2 Grace nodes (490GB each); GPU0/1↔node0, GPU2/3↔node1. Pin local → 195; remote/unbound → 117 (1.7×). Memory-heavy runs spill to both nodes (`membind=0,1`) → blended down.
- 450 was spec; **use ~195 local / ~117 remote**.

## 4. Regimes — where AsymGEMM helps
Regime = arithmetic intensity (FLOP/byte) vs machine balance (rate/C2C ≈ ~5000 FLOP/byte).
- LoRA rank=64 (default `--lora-rank`). cpu_left runs **LoRA-A** (`x@A.T`, N=rank=64) → intensity≈64 ≪ 5000 → **transfer-bound**.
- Offloaded activations consumed in backward by `dA=dSᵀ@x` (cpu_right; dS[M,r],x[M,K]→dA[r,K], exp_act_offload_lora.py:204-212). x reused only r=64× → intensity=64 → **transfer-bound** (compute ~0.4% of transfer). Activation huge but compute tiny ⇒ transfer-bound (size cancels in FLOP/byte; rank sets it).
- Base expert/attn weight GEMMs: N=hidden/inter (768-8192) → large-N, **compute-bound** (cuBLAS/B-stationary; AsymGEMM value there = capacity not speed).
- Dense full-FT: offloaded x feeds `dW=dyᵀ@x`, reuse=out_features → **compute-bound**, transfer hides → offload ~free.

## 5. Benchmarks (GB200, bf16 grouped; bench_cpu_left_vs_staged.py, bench_v2.py, bench_headroom.py)
cpu_left vs staged (copy A→HBM then GPU grouped GEMM), speedup=staged/cpu_left:
- Small N (16-512, LoRA regime): cpu_left wins **1.2-1.3×** (cpu_left ≈ bulk-copy time; matmul hidden). Staged = copy + matmul serial.
- Crossover N≈1024. Large N (≥2048): staged wins up to ~2× (cpu_left A-stationary low-occupancy ~16 CTAs = weak matmul; copy cheap fixed tax).
- cpu_left ≈ bulk-copy bandwidth at scale (m=16384,n=64: 0.542 vs 0.537ms) ⇒ link already ~saturated in its regime.
Headroom (B-stationary asym vs cuBLAS, compute-bound): asym **~30% of cuBLAS (~3× slower)**, ~25% HW peak (kNumStages=2, reduce-add epilogue, static grid). Irrelevant — use cuBLAS for compute-bound; AsymGEMM real use is transfer-bound. Don't optimize kernel compute.

## 6. E2E investigation (3 agents) — CORE BOTTLENECK
**C2C is 100% on the critical path; ZERO overlap with compute.**
- Forward D2H: `copy_(non_blocking)` on the **compute stream**, event on compute stream — early-return only, not overlapped (activation_offload.py:191-196).
- Backward fetch: blocking staging copy before consumer (activation_offload.py:235 `event.synchronize()` + :248), OR cpu_left/cpu_right reads pinned CPU **inside the kernel** on `getCurrentCUDAStream()` (exp_act_offload_kernels.cu:228) — C2C intrinsic to compute stream.
- Compute-bound `_grouped_base_dx` (pure-HBM matmul, qwen3_moe.py:438-488) runs in **separate serial blocks**, nothing prefetched (qwen3_moe.py:1245-1254, 1364-1369).
- Stream/prefetch scaffolding all **dead**: weight_offload.py:94 "Stage 3 (prefetch)... unused"; zero `with cuda.stream`, zero `record_stream`, no `cp.async.bulk.prefetch` (dead in cute headers). lf.py no stream refs.

**Irony:** fused cpu_left wins the GEMM microbenchmark (zero HBM) but fusing C2C inside the compute kernel FORBIDS overlap → loses e2e. Microbenchmark-optimal / e2e-suboptimal.

Offload policy (is it "selective"?): coarse, NOT cost-based. No cost model anywhere.
- Per-module-type env flags: `ASYMM_EXPERT_ACT_OFFLOAD`, `ASYMM_ATTN_ACT_OFFLOAD`, `ASYMM_LAYER_ACT_OFFLOAD`, `ASYMM_LAYER_GC` (lf.py:1306-1321). Operator picks; harness sweeps.
- Two proxies: 1MiB size gate + dtype + skip-Parameters (decoder_activation_offload.py:159-181, never recomputes); per-expert token-count threshold `tok-leN/geN/A-B` (moe.py:462-498, qwen3_moe.py:2557-2568) — MoE-only, MUTUALLY EXCLUSIVE with ASYMM_EXPERT_ACT_OFFLOAD (qwen3_moe.py:2420-2421).

Tiling (cpu_left): BLOCK_M=64,N=64,K=512 (env DG_BF16_*, default; host hpp:314-316). A-tile=64×512×2=**64KiB**. **A single-buffered (`SMEM_A_NUM_TILES=1`)**, B double-buffered (kNumStages=2). No data prefetch; only descriptor prefetch + L2_256B promotion + EVICT_NORMAL.

## 7. Improvement levers (ranked, e2e)
1. **Separate stream for C2C-grad vs dx (the money).** Within a layer, `dA=dSᵀ@x` (transfer-bound) and `dx=dy@W` (compute-bound, pure HBM) are INDEPENDENT. Put cpu_left/cpu_right on a `fetch_stream`, `_grouped_base_dx` on default, event only at join. Low contention: cpu_left transfer-bound (~0.4% cores) + low occupancy (~16 CTAs leave SMs free); H2D write ~2.5% of HBM BW. fetch≈0.6× dx compute → C2C ~hidden. No HBM cost. Currently all serial on default stream.
2. **fp8/fp4 activations** — halve/quarter C2C bytes (only way past ~195 ceiling; can't widen pipe). Multiplies with #1; reduces NUMA spill.
3. **Cost-based selective offload + allow mixing.** Per tensor: `offload_cost=bytes/C2C_eff`, `recompute_cost=FLOPs/rate`, keep=HBM budget. Offload cheaper ⟺ `recompute_FLOPs/bytes > rate/C2C` (~5000 FLOP/byte; matmul output = contraction K, so K>~5000 offload else recompute; tiny/elementwise always recompute). Knapsack on HBM budget for keep. Lift act-offload↔token-recompute mutual exclusion to mix. NOTE: after #1, offload_cost→~0 (hidden) ⇒ offload almost always wins, policy simplifies.
4. **Double-buffer A (`SMEM_A_NUM_TILES 1→2`, +64KiB, fits 232KiB).** Current double-buffer is on B (cheap HBM operand); A (expensive C2C operand) single-buffered ⇒ A[k+1] C2C fetch stalls until A[k] N-sweep done. SMALL win (kernel already ~saturated; BLOCK_K=512 often 1 K-block). NOT L2 (A read-once → caching useless; fix is SMEM prefetch).
5. **Bidirectional pipelining** (195→314): overlap fwd-offload D2H of microbatch N+1 with bwd-fetch H2D of N. ~16% contention, net +60%. Needs microbatch/pipeline structure.
6. **NUMA local pinning** (117→195, 1.7×), fades on spill in memory-heavy runs.

Stop: tuning kernel compute (useless for transfer-bound; cuBLAS for compute-bound).
Next: prototype #1 (stream + event ordering vs `_grouped_base_dx`), measure e2e backward delta.

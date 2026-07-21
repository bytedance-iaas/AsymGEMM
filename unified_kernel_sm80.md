# unified_kernel_sm80 — CPU + A100 (SM80) INT8 unified MoE, mirroring the SM90 stack

**Status:** Phases 0–2 implemented & gate-green (2026-07-17, H200 validation +
sm_80 compile gates; see §6 checklist). Phase 3 (deep HBM kernel) and
Phase 4 (fused hybrid) not started. Uncommitted.
**Implementation notes vs the original plan:**
- The SM80 kernel adopts the SM90 asymScheduler segment convention
  (pair-style offsets `[start,end]` per segment, `-1`-terminated experts,
  grid Y = list_size − 1) instead of the SM80-BF16 kernel's cumulative-ends
  convention — required for facade drop-in compatibility with
  `unified_moe`'s `_build_layout` output.
- Direct test entry points `m_grouped_int8_asym_gemm_sm80_{contiguous,masked}`
  are pybind-exported so H100/H200 boxes can exercise the exact A100 code
  path (arch guard is `>= 800`).
- CPU bucket needed no work: `cg_int8_rm_backend_name` already auto-selects
  amx_int8_rm → avx512_vnni_int8_rm → none.
**Goal:** bring the unified CPU+GPU INT8 MoE stack (`asym_gemm.unified_moe.Layer`)
to A100/SM80: CPU bucket (AMX or AVX512-VNNI) + GPU bucket with pinned-host
expert streaming and HBM-cached experts, adaptive dispatch on top — the same
architecture that runs on H100/SM90 today.
**Non-goal:** FP8 anywhere. A100 has no FP8 hardware; INT8
(`mma.sync.m16n8k32.s8.s8.s32`, 624 TOPS dense) is the SM80 quantized format.
**Prerequisite:** the SM80/SM89 file split (Phase 0 below, planned separately):
`sm80_moe_gemm.cuh` must be SM80-clean before new SM80 kernels build on it.
**Dev environment:** H100 boxes. Every kernel here is written against
`__CUDA_ARCH__ >= 800` features only, so it executes bit-identically on SM90;
an `-arch=sm_80` compile-only gate guarantees no SM89/SM90 instruction leaks.

---

## 1. What "unified" means today (SM90 inventory)

The SM90 stack `unified_kernel_sm80` must mirror, component by component:

| # | SM90 component | Where | Role |
|---|---|---|---|
| 1 | CPU INT8 grouped GEMM | `csrc/cpu/cpu_gemm` (AMX + `kernels/avx512/int8_gemm_rm.h`) | `m_e <= m_cpu` bucket; stride-aware, reads the same pinned slabs |
| 2 | asym INT8 kernel | `sm90_int8_asym_gemm_1d1d.cuh` | K-outer, streams B from **pinned host** over PCIe via TMA; FP32 partials via `TMA_REDUCE_ADD` |
| 3 | deep INT8 kernel | `sm90_int8_gemm.cuh` (hybridGEMM Phase A) | persistent M-outer pipeline for **HBM-cached** experts |
| 4 | fused hybrid kernel | `sm90_int8_hybrid_gemm.cuh` (hybridGEMM Phase B) | one launch, disjoint SM ranges: asym side + hbm side, tile stealing |
| 5 | arch facades | `csrc/apis/gemm.hpp:591/622/656` | `m_grouped_int8_{asym_,}gemm_nt_{contiguous,masked}` — route by `arch_major`, currently `DG_HOST_UNREACHABLE` off SM90 |
| 6 | Python runtime | `unified_moe/runtime.py` (`Layer`, `ExpertSlab`, `_StageRing`, `_hbm_grouped_gemm`) | cached/staged/slab partitions, pinned slabs, layout build |
| 7 | dispatch model | `unified_moe/dispatch_model.py` | online-refit makespan partition CPU vs GPU |
| 8 | graph path | `unified_moe/capturable.py` | CUDA-graph capturable decode via masked variants |

Components 1, 6, 7, 8 are **GPU-arch-agnostic by construction** (the slab
format `[G, N, K]` int8 + `[G, N, Kb]` FP32 scales feeds every backend; the
dispatch model refits its rates online). The work is components 2–5.

## 2. Hardware delta: SM90 → SM80 (A100)

| Capability | H100 (SM90) | A100 (SM80) | Consequence |
|---|---|---|---|
| Bulk async copy | TMA (tensor maps, swizzle, `REDUCE_ADD`) | `cp.async` 16 B/thread, no reduce | replace all TMA with `cp.async` tiled copies; partial sums via in-CTA seed/read-modify-write |
| Tensor core | `wgmma` (warp-group) | `mma.sync.m16n8k32.s8s8.s32` + `ldmatrix` | 4-warp TiledMMA, S32 accumulators in registers |
| Host-pinned reads from kernel | TMA over PCIe | UVA zero-copy `cp.async` from pinned host | **already proven in-repo**: `sm89_moe_fp8_gemm_impl` streams W from pinned host this way; A100 is the same PCIe Gen4 ×16 (~21.5 GB/s effective) as our current boxes |
| Warp specialization | producer TMA warp + `setmaxnreg` | none needed | all threads issue `cp.async`; pipeline via commit groups (mbarrier optional, exists on SM80) |
| Smem/SM | 227 KB | 164 KB (163 KB/block opt-in) | heuristics in `csrc/jit_kernels/heuristics/sm80.hpp` already model this |
| HBM BW | 3.35 TB/s | 2.0 (80 GB) / 1.55 (40 GB) TB/s | deep-side tile sizing + dispatch priors change; PCIe:HBM ratio grows, asym-side economics unchanged |
| INT8 peak | ~1979 TOPS (wgmma) | 624 TOPS | roofline ridge ≈ 312 int8-op/B (80 GB part) |
| Host CPU (typical pairing) | SPR (AMX) | Ice Lake (AVX512-VNNI, **no AMX**) or Milan (no AVX512) | CPU bucket must select the AVX512 kernel via `caps()`; Milan hosts degrade to no/AVX2 CPU bucket — dispatch model shifts work to GPU automatically |

Key simplification vs the FP8 precedent: **within one 128-wide K-block, INT8
S32 accumulation is exact** — no inter-group `cs_g/cs_{g+1}` rescale gymnastics.
Dequant happens once per K-block at the S32→F32 boundary.

## 3. Scale contract (must not fork)

The runtime produces, and the facades consume, the *natural* layout
(`gemm.hpp:585-588`): `sfa [M, Kb]` per-token, `sfb [G, N, Kb]` per-channel,
`Kb = K/128`. SM90 transposes to K-major before its kernel. The SM80 kernels
should consume the **natural layout directly** (no transpose in the facade):
per K-block `kb`, the CTA loads `sfa[m0:m0+BM, kb]` (BM floats) and
`sfb[e, n0:n0+BN, kb]` (BN floats) into smem and applies
`d[m,n] += float(acc_s32) * sfa[m,kb] * sfb[e,n,kb]` at the block boundary.
`BLOCK_K` is constrained to 128 (== scale granularity) in v1; wider tiles are
a later optimization (two scale applications per tile).

## 4. Phases

### Phase 0 — prerequisite: SM80/SM89 split + compile gate

From the separate split plan: `smxx_moe_utils.cuh` (shared helpers),
`sm80_moe_gemm.cuh` (BF16/FP16 only, no `mma_sm89.hpp`),
`sm89_fp8_moe_gemm.cuh` (FP8 kernels), params header split, generated-code
include strings in `sm89_fp8_asym_gemm.hpp:34/130` updated, and a CI step that
JIT-compiles every `sm80_*` device header with `-arch=sm_80` (compile-only).
**Gate 0:** existing SM89/SM80 tests bit-identical; sm_80 compile gate green.

### Phase 1 — `sm80_int8_asym_moe_gemm.cuh`: the host-streaming asym kernel

The core enabler. Clone the *structure* of `sm89_moe_fp8_gemm_impl`
(K-outer, M-inner, W once per K-tile from pinned host, partials in HBM
between K-tiles) with these substitutions:

1. **Types/MMA.** `ElementIn = int8_t`; `MMA_Op = SM80_16x8x32_S32S8S8S32_TN`;
   `TiledMMA<..., Tile<16*NWARPS, _16, _32>>`; accumulator fragment S32.
   Smem layouts, 16-byte `cp.async` vectors, and the `SM75_U32x4_LDSM_N` plan
   carry over unchanged (int8 and FP8 are both 1 byte).
2. **Scales.** Drop the FP8 kernel's three modes (scalar / per-token /
   block-scale-with-ratio-folding). One mode: per K-block, stage
   `sfa[·, kb]` (BM floats) + `sfb[e, ·, kb]` (BN floats) in smem.
3. **Partial sums: FP32 in HBM** (not BF16 as the FP8 kernel uses). Sequence
   per (m-tile, k-block): seed `tSrF32` from `gO` (k>0), run the K-block's
   MMAs into a *separate* S32 fragment, then
   `tSrF32 += float(tSrS32) * sfa[m]*sfb[n]`, write back. FP32 partials match
   the SM90 asym kernel's output convention (`TMA_STORE`/`TMA_REDUCE_ADD` on
   FP32 D) so `runtime.py`'s downstream epilogue is unchanged. The K-outer
   loop keeps one CTA per (n-block, expert) → the read-modify-write is
   race-free without atomics.
4. **Params/wrapper.** `SM80MoEInt8Params` in a new `sm80_int8_moe_params.h`
   (x/w/o ptrs, expert_list/index_list as in `SM80MoEParams`, plus
   `sfa_ptr [M,Kb]`, `sfb_ptr [G,N,Kb]`, `kb_stride`). JIT wrapper
   `csrc/jit_kernels/impls/sm80_int8_asym_gemm.hpp` cloned from
   `sm89_fp8_asym_gemm.hpp`; heuristics: extend `sm80.hpp` with
   `select_sm80_int8_config` (smem = `(BM+BN)*BK + BM*BN*4 (fp32 sO) +
   (BM+BN)*4` scale floats; `BLOCK_K = 128` fixed in v1).
5. **Masked variant** `sm80_int8_asym_moe_gemm_masked_impl` cloned the same
   way from `sm89_moe_fp8_gemm_masked_impl` (constant grid, reads
   `masked_m[blockIdx.y]`) — required by `capturable.py`'s CUDA-graph decode.
6. **Facades.** `m_grouped_int8_asym_gemm_nt_{contiguous,masked}`
   (`gemm.hpp:591/622`): add `arch_major == 8 && arch_minor == 0` branches
   passing natural-layout scales (no K-major transpose). Leave 8.6/8.7
   asserting until someone needs them.

**Gate 1:** new `tests/test_sm80_int8_asym.py` — parity vs a reference
dequant GEMM (`(a_int8.float()@b_int8.float().T) * sfa * sfb` per K-block)
across the `test_sm90_int8.py` shape grid, W in pinned host memory; runs on
H100 (identical instruction path). Tolerance: tight — error only from FP32
partial-sum adds across Kb blocks.

### Phase 2 — Python runtime enablement

1. `runtime.py`: `Layer.gpu_backend` becomes arch-aware
   (`"asym_gemm_sm80_int8"` on 8.0); the call sites already go through the
   facades, so no other changes — quantization (`quantize_per_token_int8_gpu`,
   `quantize_per_channel_int8`), slab pinning, `_build_layout`, `_StageRing`
   staging are backend-neutral.
2. CPU bucket: assert-and-log backend selection from `caps()`
   (`has_amx_int8` → AMX; else `has_avx512_vnni` → `avx512/int8_gemm_rm.h`;
   else CPU bucket disabled, `m_cpu = 0`). Verify the AVX512 kernel is wired
   into the same `cpu_gemm` runtime entry the Layer calls (it exists; confirm
   selection is automatic, not compile-time).
3. Dispatch model: no code change (rates refit online), but seed per-arch
   priors — A100 GPU rates ≈ 0.6× H100 for HBM-bound terms, PCIe terms
   unchanged, CPU rates from the VNNI kernel's bench. Add an
   `arch` field to `DispatchModel.snapshot()` so calibrations don't cross-pollinate.
4. `capturable.py`: masked variants are graph-capturable by construction
   (constant grid, no host-side tensor reads) — verify capture on the SM80
   path once Phase 1's masked kernel lands.

**Gate 2:** `test_unified_moe.py` green with the SM80 backend forced
(H100 functional proxy), including the CUDA-graph decode path.

### Phase 3 — deep-side HBM INT8 kernel (`sm80_int8_gemm.cuh`)

SM80 analog of hybridGEMM Phase A, for the VRAM-cached partition. The
existing `sm80_moe_gemm_impl` already has the right loop order for HBM
(M-outer, full-K sweep, register accumulator) — extend rather than port
DeepGEMM:

1. INT8 + scales version of `sm80_moe_gemm_impl`: S32 accumulator, per-128
   K-block dequant into an FP32 register accumulator (no HBM partials —
   full K is swept in-CTA), BF16 output.
2. Double-buffer smem (2-stage `cp.async` ring for X and W) so gmem latency
   overlaps MMA — the single-buffer `__syncthreads()` pattern in the current
   kernel leaves HBM idle during math and is the main perf risk vs roofline.
3. Register in the `m_grouped_int8_gemm_nt_contiguous` facade
   (`gemm.hpp:656`) under `arch_major == 8`; `runtime.py:_hbm_grouped_gemm`
   then picks it up with **zero Python changes** (it already
   getattr-falls-back to the asym kernel).

**Gate 3:** parity as Gate 1; perf on A100-class roofline ≥ 60% of
min(624 TOPS, BW-bound bound) for prefill shapes — measured on H100 via
`ncu` pipe/DRAM classification, scaled per §5. If it misses, ship Phases 1–2
anyway: the runtime falls back to the asym kernel for the cached partition,
matching pre-hybridGEMM SM90 behavior.

### Phase 4 (stretch) — fused hybrid kernel (`sm80_int8_hybrid_gemm.cuh`)

SM80 analog of hybridGEMM Phase B: one persistent launch,
`gridDim.x = 108`, CTA-rank split — ranks `< S_host` run the Phase 1 asym
pipeline over host-resident segments, the rest run the Phase 3 pipeline over
HBM segments; on own-side exhaustion, host-side CTAs steal HBM tiles.
Simpler than SM90: no clusters, no warp-role asymmetry — both sides are
plain cp.async pipelines, so the merge is a top-level `if (cta_rank < s_host)`
over two device functions plus the shared segment scheduler
(port `asymScheduler` segment enumeration; add an atomic tile counter for
stealing). The SM split knob: the asym side is PCIe-bound — expect 2–8 SMs
to saturate ~21.5 GB/s; size from Phase 1 microbench.
**Gate 4:** ≥ the two-launch makespan of Phases 1+3 on the decode/prefill
grid; CUDA-graph capturable.

### Phase 5 — validation & A100 estimation without A100

1. **Compile gate** (Phase 0) is the hard correctness line for "runs on A100".
2. **Functional**: full suite on H100 — SM80 kernels execute the identical
   `mma.sync`/`cp.async`/`ldmatrix` path there.
3. **PCIe microbench**: `cp.async`-from-pinned-host achieved BW at the Phase 1
   access pattern (16 B/thread, BLOCK_N×128 W tiles). Target ≥ 18 GB/s
   (~85% of the 21.5 GB/s TMA baseline). This is the one number TMA-vs-cp.async
   could plausibly regress; measure before writing Phase 4.
4. **Roofline transfer**: per shape, classify (DRAM / PCIe / tensor-pipe
   bound) with `ncu` on H100, then scale: DRAM ×0.60 (80 GB) / ×0.47 (40 GB);
   PCIe ×1.0; tensor-pipe bracket ×0.32–0.58. Record per-shape estimates in
   `bench_unified_moe.py` output so they're falsifiable on real hardware.
5. **On first A100 access**: run `bench_unified_moe.py`, compare against the
   recorded estimates, recalibrate dispatch priors (§Phase 2.3).

## 5. Risks / open questions

- **cp.async over PCIe efficiency** (Phase 5.3). The SM89 FP8 kernel proves
  functionality on Ada; sustained BW at A100's Gen4 link with 108 SMs' worth
  of outstanding transactions is unmeasured. Mitigation: the K-outer loop
  already amortizes W over all M-tiles; if BW disappoints, raise BLOCK_N
  (fewer, larger tiles) before restructuring.
- **Host CPU capability on A100 boxes.** Ice Lake → VNNI path (exists);
  AMD Milan/Rome → no AVX512: CPU bucket contribution shrinks to ~0 and the
  stack degrades to GPU-only gracefully via the dispatch model. Confirm the
  actual fleet's CPUs before promising CPU-bucket speedups.
- **FP32-partials bandwidth tax** (Phase 1.3): FP32 partials double the
  per-K-block D traffic vs BF16. At Kb=56 (K=7168) this is
  `2 × M×N×4 × (Kb−1)` bytes of HBM traffic — fine while PCIe-bound, but
  re-evaluate if a profile shows D-traffic dominating; BF16 partials are the
  proven fallback (FP8 kernel ships them).
- **`BLOCK_K = 128` lock-in** (§3) halves the max K-tile vs the BF16 kernel's
  256; acceptable while PCIe-bound, revisit for the deep side (Phase 3 sweeps
  K in-CTA and can use BLOCK_K=128 with more stages instead).
- **JIT cache**: every facade/heuristic change alters generated source →
  recompiles and, on shared boxes, stale-cache confusion. Clear the cache in
  CI and before every gate run (standing project trap).

## 6. Deliverables checklist

- [x] Phase 0: file split (`smxx_moe_utils.cuh`, trimmed `sm80_moe_gemm.cuh`,
      `sm89_fp8_moe_gemm.cuh`, `sm89_fp8_moe_params.h`) — BF16/FP16 parity
      smoke green on H200, `sm89_fp8_asym_gemm.hpp` generated-code includes
      updated (JIT cache keys change: "v4")
- [x] Phase 0b: `tests/test_arch_compile_gates.py` — compile-only
      `-arch=sm_80` (BF16 + INT8 headers) and `-arch=sm_89` (FP8) gates
- [x] `asym_gemm/include/asym_gemm/impls/sm80_int8_moe_params.h`
- [x] `asym_gemm/include/asym_gemm/impls/sm80_int8_asym_moe_gemm.cuh`
      (contiguous + masked, shared body; W LDSM hoisted per K-tile)
- [x] `csrc/jit_kernels/impls/sm80_int8_asym_gemm.hpp` + pybind exports
- [x] `csrc/jit_kernels/heuristics/sm80.hpp`: `select_sm80_int8_config`
      (BLOCK_K=128 locked; 33 KB smem at 128/128 — multi-CTA/SM occupancy)
- [x] `csrc/apis/gemm.hpp`: SM80 (8,0) branches in the two INT8 asym facades,
      natural scale layout (no K-major transform)
- [x] `unified_moe/runtime.py`: arch-aware `gpu_backend` property,
      `_hbm_grouped_gemm` SM90 guard (SM80 cached partition → asym kernel)
- [x] `tests/test_sm80_int8_asym.py`: 12 tests — contiguous/masked/padded
      parity (pinned + HBM B, NaN canaries) + end-to-end unified `Layer`
      forward on the SM80 kernel vs native backend (rel < 5e-3, cos > 0.9999)
- [x] Regressions green on H200: `test_unified_moe.py` (7+2 adaptive),
      `test_sm90_int8.py`
- [x] A100 runner + benchmark: `scripts/run_sm80_a100.sh` (env → build →
      cache clear → compile gates → parity → unified tests → bench) and
      `tests/bench_sm80_int8.py` (parity sanity, copy-engine PCIe ceiling,
      streamed + HBM shapes, TOPS/GB-s, `--save/--baseline` JSON diffing);
      `scripts/test.sh` gained the `sm 80` row. H200 baseline recorded at
      `bench_results/sm80_int8_h200.json` (2026-07-21).
- [x] Deployment-artifact proof: `tests/test_sm80_ptx_deploy.py` — the kernel
      compiled to virtual-arch `compute_80` PTX (the exact artifact an A100
      driver JITs to sm_80 SASS) is loaded on the dev box via
      `cuModuleLoadData` (driver JIT → sm_90 SASS) and launched through the
      raw driver API: **bitwise equal** to the production cubin path,
      rel 7.5e-8 vs float64 reference (H200, 2026-07-21). Wired into
      `scripts/run_sm80_a100.sh` step 5.
- [ ] Phase 3: `sm80_int8_gemm.cuh` (deep HBM) + wrapper + facade branch
- [ ] Phase 4 (stretch): `sm80_int8_hybrid_gemm.cuh`
- [ ] Run the suite on a real A100 (`bash scripts/run_sm80_a100.sh`) and
      diff against the H200 baseline

## 7. First measured numbers (H200 baseline, 2026-07-21)

`tests/bench_sm80_int8.py` on the dev box (H200, Gen5 PCIe, copy-engine
ceiling 47.4 GB/s): streamed (pinned-B) shapes reach **8–12 GB/s kernel PCIe
throughput ≈ 26% of ceiling** — well under the ≥85% gate — while HBM-B shapes
reach ~35 INT8 TOPS / ~1.1 TB/s. Diagnosis: the v1 kernel single-buffers W
(`cp_async_wait<0>` then a full M-loop of MMA/epilogue before the next W
tile), so the PCIe link idles for the entire M-sweep — the §5 "cp.async over
PCIe efficiency" risk realized as a duty-cycle problem, not a per-transaction
one. First lever before Phase 4: double-buffer W (prefetch W(k+1) into a
second smem slot during the k M-loop); X-side double-buffering is the second.
On A100's Gen4 link (~25 GB/s ceiling) the absolute 8–12 GB/s may be a larger
fraction of ceiling, but the same fix applies. Absolute numbers live in
`bench_results/sm80_int8_h200.json`; re-run with `--baseline` on the A100 to
get per-shape ratios.

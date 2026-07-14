# KT ARM BF16 SFT Fix Plan v7

**KT remains native ARM BF16 CPU code. DeepSpeed is not used. No ZeRO/DeepSpeed
routing assumptions apply to KT kernels.**

This plan continues from `agent/kt/fix_arm_v6.md` (completed through the Stage 7
full same-config acceptance). v7 keeps the v6 grouped/sharded backward layout and
focuses exclusively on **backward grouped kernel compute efficiency**, which the
v6 acceptance profile shows is the dominant remaining cost.

Keep this work isolated:

- Kernel/runtime edits: `../ktransformers/kt-kernel/**`
- KT-only LF launcher/profile edits: `agent/kt/scripts/**` and `scripts/kt/**`
- Do not edit `scripts/lf/run_lf_lora_sft.sh` or `scripts/lf/profile_lora_lf_test_{source,both}.sh`
- KT validation must use physical GPU 1 first, physical GPU 2 only as fallback.
  Never GPU 0 or GPU 3 for accepted KT results.

## Accepted v6 Baseline (the row to beat)

Artifact:
`profiling_results/profiling_kt_codex_smoke/v6_accept_qwen3_s4096_b4_r64_w5_s10_t64_source/.../b4_s4096`

Shape: `Qwen/Qwen3-30B-A3B`, `seq_len=4096`, `batch=4`, `rank=64`,
`dropout=0.00`, `warmup=5`, `measure=10`, `trainer_max_steps=15`, GPU 1,
`KT_NUM_THREADS=64`, affinity `0-143`.

| Metric | v6_accept (before) |
|---|---:|
| avg_step | 276.813 s |
| avg_forward | 75.246 s |
| avg_backward | 199.535 s |
| peak_allocated | 34.479 GiB |
| peak_reserved | 44.029 GiB |
| process_rss | 184.467 GiB |
| expert_schedule_wall_ms | 1078.092 ms/layer |
| backward_grouped_tile_ms | 2457.110 ms/layer |
| backward_tile_recompute_ms | 44131.841 task-ms/layer |
| backward_route_grad_accum_ms | 82977.276 task-ms/layer |
| backward_base_grad_ms | 66154.476 task-ms/layer |
| backward_lora_grad_ms | 15930.130 task-ms/layer |
| sparse_backward_scratch_bytes | 3.043 GiB |
| loss max/last/train | 1.8324 / 1.4692 / 1.6074 |

Hardware/runtime facts confirmed from the v6_accept native log:
- SVE vector width = 16 bytes (128-bit): `sve_vector_bytes=16` → 4 f32 lanes,
  8 bf16 lanes, 4 bf16 pairs per `svbfdot_f32`.
- `kt_arm_effective_route_qlen=16384` (batch*seq), `top_k=8` →
  ~131072 routes/layer, ~1024 routes/expert average.
- High route skew: `max_local_routes` 3431-6170 vs `min_local_routes` 11-19,
  `route_skew_ratio` 3.35-6.03.
- Forward base kernels already use BFDOT:
  `base_kernel=sve_bfdot_blocked`, `down_kernel=bf16_bfdot_blocked`.
- Backward base kernels use FP32 FMLA: `backward_base_kernel=grouped_sve_tile`.

## Root-Cause / Opportunity Analysis

`backward_base_grad_ms` (66154 task-ms/layer) is the single largest backward
compute component and is the top mission priority. It is two FP32-FMLA matmuls:

1. `grad_act[M,I] = dy[M,H] @ down[H,I]` (contract H), `down_proj_bf16_`
2. `grad_x[M,H] = grad_gate[M,I] @ gate[I,H] + grad_up[M,I] @ up[I,H]`
   (contract I), `gate_proj_bf16_` / `up_proj_bf16_`

Both use `svmla_n_f32_m` (FP32 FMLA, broadcast scalar from the FP32 activation,
BF16 weight widened to FP32). FMLA does 4 MACs/instruction on 128-bit SVE.

Decisive evidence that BFDOT is the faster path for these exact shapes: the
**forward** base does the same total FLOP as backward base (3 matmuls of
`M*I*H`) and runs in **57921 task-ms** (`base_gate_up_ms` 43306 + `base_down_ms`
14616) using BFDOT, versus **66154 task-ms** for the backward FMLA. BFDOT does
8 MACs/instruction (4 lanes x 2 pairs), ~2x FMLA, and the forward proves it is
compute-favorable here (the forward scalar-M BFDOT reloads weights ~8x more than
the FMLA M8 backward yet is still faster → compute-bound, not memory-bound).

Critically, the transposed BF16 base weights needed for a BFDOT backward are
**already built every backward repack** (`transpose_base_weights()` populates
`down_proj_t_bf16_` = [E,I,H] contiguous over H, and `gate_proj_t_bf16_` /
`up_proj_t_bf16_` = [E,H,I] contiguous over I) but are **currently unused by any
compute kernel**. The repack is fully overlapped (`backward_repack_wait_ms`
~0.003 ms/layer), so these transposed weights are free to consume.

With the transposed weights, both base-grad matmuls become
`C[M,N] = A[M,K] @ B[N,K]^T` with A and B both BF16 and contiguous over the
contract dim K — exactly the form of the proven forward `arm_bf16_matmul_blocked4`
BFDOT kernel:
- grad_act: A=dy[M,H], B=down_t[I,H], K=H, N=I
- grad_x (gate): A=grad_gate[M,I], B=gate_t[H,I], K=I, N=H
- grad_x (up):   A=grad_up[M,I],   B=up_t[H,I],   K=I, N=H

## Stage 1: BFDOT register-blocked base backward grad (PRIMARY)

Status: **implemented / measured / ACCEPTED.** Full same-config 15-step LF source
acceptance on GPU 1 passed strict validation. See the Results log for numbers.

Scope: `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
- New kernel `arm_bf16_grad_matmul_reg(M,N,K,a,lda,b,ldb,c,ldc,accumulate)`:
  register-blocked (M-block 4 x N-block 4) `svbfdot_f32` GEMM, A@B^T, contract K.
  Reuses each B (weight) row across 4 A rows to cut weight traffic below both the
  forward scalar-M BFDOT and the backward FMLA-M8.
- `BackwardTileScratch`: add bf16 staging buffers `dy_bf16`, `grad_gate_bf16`,
  `grad_up_bf16`; allocate in `init_backward_tile_scratch`.
- `backward_tile_accumulate_grouped`:
  - grad_act path: convert `scratch.dy` (f32) → `dy_bf16`, call new kernel with
    `down_proj_t_bf16_` (`down_t_idx`). Output grad_act, accumulate=false (LoRA
    down grad still accumulates into grad_act afterward, unchanged).
  - grad_x path: convert `grad_gate`/`grad_up` (f32) → bf16, call new kernel with
    `gate_proj_t_bf16_` (accumulate=false) then `up_proj_t_bf16_`
    (accumulate=true). LoRA gate/up grad still accumulate into grad_x afterward.
- Keep FP32 `scratch.dy/grad_gate/grad_up` because the LoRA backward kernels
  still read them as FP32 (no LoRA change in this stage).
- Update `backward_base_kernel_name_runtime()` label to
  `grouped_sve_bfdot_tile`.

Precision note: dy is derived from `grad_output` which is already BF16 in the
model; grad_gate/grad_up move to BF16 only for the base grad_x matmul. This
matches the all-BF16 forward and is validated against the reference tests.

Risks/watches:
- New microkernel correctness (m/n tail handling) → guarded unit micro-check
  plus the per-commit reference tests before any LF run.
- If register pressure spills on 128-bit SVE, drop N-block to 2.

Validation: rebuild, run reference + dropout per-commit tests, then short LF
(warmup1/measure2) on GPU 1; accept only if `backward_base_grad_ms` and avg step
both improve with no correctness regression, then run full acceptance.

## Post-Stage-1 backward hotspot ranking (next-iteration handoff)

After Stage 1, the measured backward compute order (task-ms/layer, 10-step avg)
changed. The base grad is no longer dominant:

1. `backward_tile_recompute_ms` 47707.5 — NEW #1. This is the per-tile forward
   recompute (gate/up base BFDOT, gate/up LoRA, activation, down base BFDOT,
   down LoRA). The base parts already use BFDOT; cutting this further means
   either caching activations (a memory tradeoff the team moved away from) or
   speeding the LoRA-forward helpers used inside recompute.
2. `backward_lora_grad_ms` 16740.2 — #2. The six FP32-FMLA LoRA grad matmuls.
3. `backward_base_grad_ms` 13119.8 — was #1 (66154), now small after BFDOT.

So the highest-value next target is the recompute path (#1), then LoRA grads
(#2). The +8% recompute regression from Stage 1's BF16 staging buffers is a
small, separable cleanup (tighter scratch / fuse conversions into the producing
loop) worth folding into whichever path is optimized next.

## Stage 2 (conditional): BFDOT LoRA backward grads

`backward_lora_grad_ms` is now #2 at 16740 task-ms. The LoRA grad kernels are
FP32 FMLA on small dims (rank R=64) with mixed FP32 operands; BFDOT applies
cleanly to `grad_u = grad_y @ B` and (with an A transpose) `grad_input`, but
`grad_a`/`grad_b` contract over M (routes) and need different handling. Estimated
upside ~2% of step time — worthwhile but lower priority than the recompute path.
Mirror the Stage 1 approach: stage BF16 copies of the FP32 operands and reuse a
register-blocked `svbfdot_f32` microkernel; validate with the per-commit tests
then a short LF profile before any full acceptance.

## Stage 3: Remove dead FMLA base-grad path + stale knobs

Status: **implemented.** After Stage 1 acceptance, removed the now-unused FP32
FMLA base-grad kernels and their only-callers:
- `arm_tile_dy_down_to_grad_act_sve` + `..._blocked`
- `arm_tile_grad_gate_up_to_grad_x_sve` + `..._blocked`
- `backward_grad_m_tile()` / `backward_grad_k_tile()` (only fed the above)
- README env knobs `KT_ARM_SFT_BACKWARD_GRAD_M_TILE` /
  `KT_ARM_SFT_BACKWARD_GRAD_K_TILE` (only tuned the removed path)

Kept `backward_lora_rank_tile()` / `KT_ARM_SFT_BACKWARD_LORA_R_TILE` — still used
by the live LoRA backward blocked kernels. The removed functions were confirmed
unreferenced before deletion (grep), so the Stage 1 accepted performance carries
to the cleaned binary; re-verified by the per-commit reference + dropout tests
after the cleanup rebuild (no LF re-run needed because no executed code changed).

## Canonical commands

Build:
```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation
```

Unit tests:
```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest \
  ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

Short LF profile + full acceptance: see the v6 Stage 1 / Stage 7 command blocks
in `fix_arm_v6.md` (same shape, GPU 1, `KT_NUM_THREADS=64`).

## Results log

**KT remains native ARM BF16; DeepSpeed not used.**

### Stage 1 ACCEPTED — full 15-step LF source profile (GPU 1, KT_NUM_THREADS=64)

Artifact:
`profiling_results/profiling_kt_codex_smoke/v7_accept_bfdot_qwen3_s4096_b4_r64_w5_s10_t64_source/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a128_drop000/kt_armbf16__source__recomp__polnone__routerhf__expact0/b4_s4096`

Strict validation: `PASS KT ARM profile: gpu_id=1 affinity_count=144 wrappers=48 fw=1440 bw=720`
(all native KV labels verified, including `backward_base_kernel=grouped_sve_bfdot_tile`).

Shape: Qwen/Qwen3-30B-A3B, seq=4096, batch=4, rank=64, dropout=0.00, warmup=5,
measure=10, trainer_max_steps=15, GPU 1, KT_NUM_THREADS/OMP/BACKWARD=64,
affinity 0-143. Counter averages over the 10 measured steps (480 backward / 480
forward layer emissions), same methodology that reproduces the v6 numbers exactly.

| Metric | v6_accept (before) | v7_accept (after) | Delta |
|---|---:|---:|---:|
| avg_step (measured e2e) | 276.813 s | 244.648 s | -32.165 s (-11.6%) |
| avg_forward | 75.246 s | 76.320 s | +1.074 s (+1.4%) |
| avg_backward | 199.535 s | 166.303 s | -33.232 s (-16.7%) |
| peak_allocated | 34.479 GiB | 34.479 GiB | = |
| peak_reserved | 44.029 GiB | 44.029 GiB | = |
| process_rss_peak | 184.467 GiB | 156.862 GiB | -27.6 GiB (run variance; change adds ~117 MB) |
| expert_schedule_wall_ms | 1078.092 | 951.831 | -11.7% (forward path unchanged; variance) |
| backward_grouped_tile_ms (WALL) | 2457.110 | 1649.741 | -32.9% |
| backward_route_grad_accum_ms | 82977.276 | 30836.631 | -62.8% |
| backward_base_grad_ms | 66154.476 | 13119.780 | -80.2% (5.0x) |
| backward_lora_grad_ms | 15930.130 | 16740.179 | +5.1% (unchanged path; variance) |
| backward_tile_recompute_ms | 44131.841 | 47707.530 | +8.1% (see note) |
| backward_activation_grad_ms | 430.547 | 442.316 | +2.7% |
| sparse_backward_scratch_bytes | 3.043 GiB | 3.044 GiB | = |
| loss max/last/train | 1.8324/1.4692/1.6074 | 1.831/1.4686/(n/a) | match |

Correctness: per-commit reference + dropout tests 50/50 pass; warmup-step loss is
bit-identical to v6 (2.2104), measured losses track v6 within bf16-grad noise.

Notes:
- `backward_grouped_tile_ms` is the per-layer wall of the sharded backward
  compute region; -32.9% is the real per-layer backward kernel win. e2e backward
  -16.7% because the e2e also includes unchanged flush/scatter/sync/non-MoE work.
- `backward_tile_recompute_ms` rose ~8%. The recompute code is unchanged; the
  most likely cause is the three added BF16 staging buffers in
  `BackwardTileScratch` (~1.8 MB/thread) slightly raising L2 pressure during the
  recompute that shares the same per-thread scratch. This is dwarfed by the
  -53034 task-ms base-grad saving, so net is strongly positive. Reproduced in
  both the short and full runs, so it is a real (small) shift, not pure noise.
- HBM peaks are identical; no GPU memory regression. The RSS swing is run
  variance (the change can only add memory, not remove 27 GiB).

Command lines (build / unit tests / short / full acceptance) are in the
"Canonical commands" section above; the full acceptance used
`scripts/kt/profile_lora_lf_kt.sh` with
`OUTPUT_ROOT=profiling_results/profiling_kt_codex_smoke/v7_accept_bfdot_qwen3_s4096_b4_r64_w5_s10_t64_source`,
`SEQ_LENS=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 BACKEND_SPECS='kt_armbf16|recomp'
GPU_POOL=1 PROFILERS=source WARMUP_STEPS=5 MAX_STEPS=10 LORA_DROPOUT=0.00
KT_NUM_THREADS=64`.

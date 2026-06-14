# KT ARM BF16 SFT Fix Plan v4

This is the next implementation plan after v3. The native KT ARM BF16 path is real and not DeepSpeed. Low HBM use is expected because routed expert compute stays on CPU. The old "stuck" symptom was mainly bad CPU affinity plus a dense backward reducer that created one worker task per float. Those are fixed. The remaining problem is that the current ARM SFT path is still loop-shaped code around BF16 dot products, not a useful GEMM kernel design.

Rules for all stages:

- Use physical GPU 1 first and physical GPU 2 as fallback. Do not use GPU 0 or GPU 3 for KT validation.
- KT implementation edits stay under `../ktransformers/kt-kernel/**`.
- KT launcher/profile edits stay under `agent/kt/scripts/**`; do not edit `scripts/lf/run_lf_lora_sft.sh` or `scripts/lf/profile_lora_lf.sh` for this work.
- Profiles must be saved under `profiling_kt_codex_smoke/v4_*`.
- Do not move to the next stage until the validation gate for the current stage passes.
- NCU is not useful for CPU kernels. Use it only to sanity-check CUDA/GPU launch behavior. Use native timers, `perf`, disassembly, and source profiles for CPU kernel work.

## Conclusions Driving Priority

- Current hot code is in `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`.
- `arm_bf16_matmul_tiled()` still loops over `m x n` and calls one SVE BF16 dot per output element.
- `compute_gate_up_base_by_expert()` calls that dot-loop for gate and up.
- `compute_down_base_by_expert()` is a scalar FP32 loop over `H x I`.
- `compute_gate_up_lora_by_expert()` and `compute_down_lora_by_expert()` are per-route scalar loops.
- `forward_impl_packed()` processes active experts serially.
- `backward_impl_packed()` still allocates dense per-thread all-expert LoRA gradient buffers. The reducer is now chunked, but the dense buffer design remains wrong for sparse routed experts.
- `../ktransformers/kt-kernel/operators/moe-sft-tp.hpp` already has the right backward lesson: use active-expert sparse FP32 partials for reduce-type gradients and direct/sliced writes where ownership is disjoint.
- This host reports `sve`, `sve2`, `svebf16`, `bf16`, `i8mm`, and `svei8mm`, but not `bfmmla`, `sme`, or `sme2`. First target SVE BF16 BFDOT. Keep BFMMLA/SME paths feature-gated only.

External sources checked and used for design:

- Arm KleidiAI: https://github.com/ARM-software/kleidiai
- KleidiAI BF16 example: `examples/matmul_clamp_f32_bf16p_bf16p/matmul_clamp_f32_bf16p_bf16p.cpp`
- Arm Compute Library: https://github.com/ARM-software/ComputeLibrary
- Arm ACLE: https://arm-software.github.io/acle/main/acle.html
- Arm BFloat16 processing blog: https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bfloat16-processing-for-neural-networks-on-armv8_2d00_a
- oneDNN MatMul guide: https://uxlfoundation.github.io/oneDNN/dev_guide_matmul.html
- llama.cpp MoE gather/GEMM/scatter reference path: https://github.com/ggerganov/llama.cpp

The relevant related-work lesson is not "add more loops". It is: define packed layouts, expose microkernel tile sizes, pack stable RHS weights once, pack/gather LHS by active expert/block, run a blocked kernel, and validate each shape with PMU counters.

## Common Validation Commands

Use these from each stage unless the stage says otherwise.

```bash
export ASYM=/workspace/AsymGEMM-SFT/third_party/AsymGEMM
export KT=/workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
export PY=$ASYM/.venv/bin/python
export CUDA_VISIBLE_DEVICES=1
export NVIDIA_VISIBLE_DEVICES=1
export GPU_ID=1
export NUM_GPUS=1
export PROFILE_NSYS_GPU_METRICS_DEVICES=1
export KT_NUM_THREADS=8
export KT_ARM_OMP_NUM_THREADS=8
export KT_ARM_OMP_PROC_BIND=false
export KT_ARM_SFT_BACKWARD_THREADS=8
export KT_ARM_SFT_PROFILE=1
export KT_ARM_SFT_POOL_LOG=1
```

```bash
cd "$KT"
CPUINFER_FORCE_REBUILD=1 \
CPUINFER_BUILD_TYPE=RelWithDebInfo \
CPUINFER_PARALLEL=16 \
"$PY" -m pip install -e . -v --no-build-isolation

cd "$ASYM"
bash -n agent/kt/scripts/run_lf_lora_sft_kt.sh
bash -n agent/kt/scripts/profile_lora_lf_kt.sh
"$PY" -m py_compile \
  agent/kt/scripts/validate_kt_arm_profile.py \
  ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  ../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py

"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py -q
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

CPU feature and disassembly checks:

```bash
cd "$ASYM"
mkdir -p profiling_kt_codex_smoke/v4_cpu_preflight
python3 - <<'PY' | tee profiling_kt_codex_smoke/v4_cpu_preflight/cpu_features.txt
from pathlib import Path
import re
text = Path("/proc/cpuinfo").read_text(errors="replace")
features = set()
for m in re.finditer(r"^Features\s*:\s*(.*)$", text, re.M):
    features.update(m.group(1).split())
for key in ["sve", "sve2", "svebf16", "bf16", "bfmmla", "sme", "sme2", "i8mm", "svei8mm"]:
    print(f"{key}={int(key in features)}")
PY

EXT_PATH="$("$PY" - <<'PY'
import kt_kernel.kt_kernel_ext as ext
print(ext.__file__)
PY
)"
objdump -d "$EXT_PATH" > profiling_kt_codex_smoke/v4_cpu_preflight/objdump.txt
rg -n "bfdot|bfmmla|bfmlal|mopa|fmopa" profiling_kt_codex_smoke/v4_cpu_preflight/objdump.txt \
  > profiling_kt_codex_smoke/v4_cpu_preflight/bf16_instruction_hits.txt
test -s profiling_kt_codex_smoke/v4_cpu_preflight/bf16_instruction_hits.txt
```

Synthetic benchmark shape commands:

```bash
cd "$ASYM"
taskset -c 0-143 env \
  KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend both --qlen 128 --topk 8 --rank 8 \
  --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 3 --arm-profile \
  --output-json profiling_kt_codex_smoke/v4_stageX_q128_r8.json

taskset -c 0-143 env \
  KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness --qlen 2048 --topk 8 --rank 64 \
  --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 2 --arm-profile \
  --output-json profiling_kt_codex_smoke/v4_stageX_q2048_r64.json
```

LF source smoke command:

```bash
cd "$ASYM"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT \
  ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  PROFILE_NSYS_GPU_METRICS_DEVICES=1 BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=1 \
  OUT_DIR=profiling_kt_codex_smoke/v4_stageX_qwen3_s64_b1_r8 \
  agent/kt/scripts/run_lf_lora_sft_kt.sh \
  2>&1 | tee profiling_kt_codex_smoke/v4_stageX_qwen3_s64_b1_r8/console.log

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json profiling_kt_codex_smoke/v4_stageX_qwen3_s64_b1_r8/profile.json \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 --expected-batch 1 --expected-rank 8 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --require-final
```

PMU command:

```bash
cd "$ASYM"
mkdir -p profiling_kt_codex_smoke/v4_stageX_pmu
perf stat -r 3 \
  -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,context-switches,cpu-migrations \
  -o profiling_kt_codex_smoke/v4_stageX_pmu/perf_stat_q128.txt -- \
  taskset -c 0-143 env KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --qlen 128 --topk 8 --rank 8 --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 5 --skip-correctness \
  --output-json profiling_kt_codex_smoke/v4_stageX_pmu/bench_q128.json
```

If `perf` is blocked by host permissions, save the error under the stage artifact directory and use native phase timers plus disassembly for that stage.

## Stage 1: Phase, Layout, and Artifact Harness

Priority: highest. Do this before changing math. The current bench is too coarse to prove why a kernel helps or hurts.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `struct ArmSFTProfileStats`
  - `print_profile_stats()`
  - `ARM_BF16_SFT_MOE::base_projection_kernel_name_static()`
  - new public method `layout_report_json() const`
  - route stats in `build_packed_routes()`
  - buffer stats in `ensure_forward_buffers()` and `ensure_backward_buffers()`
- `../ktransformers/kt-kernel/ext_bindings.cpp`
  - bind `layout_report_json()`
  - bind any new counter getters
- `../ktransformers/kt-kernel/python/sft/arm.py`
  - expose `layout_report_json` and parsed `layout_report`
- `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
  - add `--artifact-dir`
  - write `layout_report.json`
  - write `native_profile_summary.json`
  - capture CPU features and relevant env vars in output JSON
- `../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py`
  - same JSON artifact fields for compare runs
- `agent/kt/scripts/validate_kt_arm_profile.py`
  - optionally accept and check the new layout fields when present

Intended changes:

- Add a JSON layout report with:
  - `sve_vector_bytes`
  - CPU feature booleans for `svebf16`, `bfmmla`, `sme`, `sme2`
  - `route_tile_m`, `lora_rank`, `padded_lora_rank`
  - `hidden_size`, `intermediate_size`, `expert_num`, `top_k`
  - current base weight layouts: `gate_proj_bf16_`, `gate_proj_t_bf16_`, `up_proj_t_bf16_`, `down_proj_t_bf16_`
  - pointer alignment for base weights, LoRA weights, packed input, gate/up/act/down buffers
  - `active_expert_count`, `valid_routes`, `padded_routes`, `max_local_routes`, `min_local_routes`, route skew
  - estimated bytes and FLOPs per phase
- Split profile timers enough to distinguish:
  - route pack
  - input gather/pack
  - base gate/up pack vs base gate/up compute
  - base down pack vs base down compute
  - LoRA A, LoRA B, and dropout
  - route merge
  - backward allocation/zero, route compute, thread reduce, flush
- Keep full forward/backward correctness as the only acceptance test. Phase masks can be used for profiling, but they must not become the correctness path.

Pseudocode:

```cpp
std::string ARM_BF16_SFT_MOE::layout_report_json() const {
  JsonLike out;
  out["base_kernel"] = base_projection_kernel_name_static();
  out["sve_vector_bytes"] = sve_vector_bytes_static();
  out["gate_proj_t_stride"] = {H, I};
  out["down_proj_t_stride"] = {I, H};
  out["aligned_weights"] = aligned_weights();
  out["route_stats"] = last route stats;
  out["buffer_bytes"] = forward/backward/cache pools;
  return out.dump();
}
```

Risks and watch items:

- Avoid adding a heavy JSON dependency to the native extension. A small hand-built JSON string is enough.
- Phase-only execution can hide dependencies. Use it only as a profiling helper, not as proof of correctness.
- The host SVE vector length is currently 16 bytes; all kernels must remain vector-length aware and not assume only 128-bit SVE.

Validation gate:

- Common validation commands pass.
- `bench_armbf16_sft.py` writes `layout_report.json` and `native_profile_summary.json`.
- `layout_report.json` proves `svebf16=1`, `aligned_weights=1`, `base_kernel` is not scalar, and route skew fields are present.
- `bf16_instruction_hits.txt` is non-empty.
- `v4_stage1_q128_r8.json`, `v4_stage1_q2048_r64.json`, and the Qwen3 LF smoke artifact exist and pass the validator.

## Stage 2: Replace Base Gate/Up Dot Loop With Packed SVE BF16 GEMM

Priority: highest compute fix. This is the core reason KT is still too slow.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - replace the implementation behind `arm_bf16_matmul_tiled()`
  - add native helpers near the current dot routines:
    - `arm_svebf16_gemm_f32()`
    - `arm_svebf16_pack_rhs_kxn()`
    - `arm_svebf16_pack_lhs_mxk()` only if the measured shape benefits from LHS packing
    - `arm_svebf16_gemm_tile_sizes()`
  - modify `transpose_base_weights()` or add `pack_base_gate_up_weights()` to build packed RHS once at `load_weights()`
  - modify `compute_gate_up_base_by_expert()` to call the new kernel for gate and up
  - add profile counters from Stage 1: pack bytes, GEMM tiles, tail tiles, backend name
- `../ktransformers/kt-kernel/CMakeLists.txt`
  - keep `-march=armv8.2-a+fp16+dotprod+sve+bf16`
  - add feature guards only if the kernel introduces new instructions
- `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
  - add output fields for selected tile sizes and backend

Intended changes:

- First implementation target is native SVE BF16 BFDOT because this host has `svebf16=1` and no `bfmmla/sme/sme2`.
- Pack stable RHS base weights once per layer/expert. Do not repack base weights in every forward.
- Use route-packed LHS rows by expert from `packed_input`; only add LHS packing if Stage 1 shows the kernel needs it.
- Compute gate and up in the same expert/block schedule so input rows stay hot, but keep gate and up accumulators separate.
- Tile across M routes and N intermediate columns. Use K chunks aligned to BF16 vector lanes and handle K tails with predicates.
- Keep a scalar/dot fallback under an env flag only for debugging:
  - `KT_ARM_SFT_GEMM_BACKEND=sve_bfdot|dot_loop`
  - default must be `sve_bfdot` after validation.

Pseudocode:

```cpp
for expert in active_experts:
  X = packed_input[begin:end, H]
  Wg = packed_gate_rhs[expert]
  Wu = packed_up_rhs[expert]
  for m0 in blocks(M, MT):
    for n0 in blocks(I, NT):
      acc_g[MT][NT] = 0
      acc_u[MT][NT] = 0
      for k0 in blocks(H, KT):
        load/predicate X tile
        load packed Wg/Wu tile
        svbfdot_f32 into acc_g and acc_u
      store gate/up fp32
```

Risks and watch items:

- BFDOT lane packing is easy to get subtly wrong. Validate odd `H`, odd `I`, and route tails.
- If M per expert is very small, LHS packing can cost more than it saves. Stage 1 route histograms decide whether to pack LHS or use direct gathered rows.
- Combining gate and up may increase register pressure. If PMU shows spills, split into two GEMM calls while preserving packed RHS.
- Do not spend time on BFMMLA or SME on this host unless CPU preflight changes.

Validation gate:

- Common correctness tests pass.
- Add or extend unit coverage in `test_armbf16_sft_reference.py` for:
  - odd `hidden_size`
  - odd `intermediate_size`
  - `qlen=1`
  - all routes to one expert
  - invalid/negative expert routes
- Run synthetic q128 and q2048 commands with output names `v4_stage2_*`.
- `native_profile_summary.json` shows lower `base_gate_up_ms` than Stage 1 at both q128 and q2048.
- `perf_stat_q128.txt` improves instructions/output element and does not show a cache-miss explosion.
- Disassembly still contains BF16 instructions.
- Qwen3 LF smoke passes validator on GPU 1 or 2.

## Stage 3: Replace Base Down Scalar Loop With Packed GEMM

Priority: high. After gate/up, base down is the next large forward phase.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `transpose_base_weights()` or new `pack_base_down_weights()`
  - `compute_down_base_by_expert()`
  - shared `arm_svebf16_gemm_f32()` helpers from Stage 2
  - `ForwardBuffers` if down requires an additional packed/converted activation buffer
- `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
  - add separate base-down tile and pack fields

Intended changes:

- Use `down_proj_t_bf16_` or a new packed down RHS layout so the kernel computes `act[M, I] x down[I, H] -> down[M, H]`.
- Convert or pack `act` to BF16 only if the measured loss and correctness tolerance are acceptable. Otherwise keep FP32 activation input and use an FP32 x BF16 path for down.
- Reuse Stage 2 scheduling and tile report fields.

Pseudocode:

```cpp
for expert in active_experts:
  A = act[begin:end, I]
  Wd = packed_down_rhs[expert]  // I x H packed by N tile
  gemm(A, Wd, down[begin:end, H])
```

Risks and watch items:

- Current down input is FP32 activation. BFDOT needs BF16 operands. A BF16 conversion of `act` changes numerics and costs bandwidth.
- If FP32 activation must be preserved, a different FP32 x BF16 kernel is needed and may be slower than expected.
- Down output width `H` is large; N tiling and cache behavior matter more than for small-rank LoRA.

Validation gate:

- Common correctness tests pass with the same tolerances as before, including dropout tests.
- Add a down-specific correctness test comparing old and new down for random `act`.
- Synthetic q128/q2048 profiles show lower `base_down_ms` without regressing `base_gate_up_ms`.
- PMU shows reduced instructions/output element for base down.
- Qwen3 LF smoke passes validator.

## Stage 4: Parallel Active-Expert and Block Scheduling

Priority: high after useful kernels exist. Current forward serializes active experts, which wastes cores when per-expert route counts are uneven.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `forward_impl_packed()`
  - `run_arm_sft_tasks()`
  - add a task descriptor near `PackedRoutes`, for example `struct ForwardExpertBlockTask`
  - `compute_gate_up_base_by_expert()`, `compute_down_base_by_expert()`, LoRA functions if they need block-level entry points

Intended changes:

- Build task lists from active experts:
  - `(expert, begin, end, phase, m_block_begin, m_block_end)`
- Parallelize base gate/up and base down by expert/block using the existing WorkerPool subpool.
- Keep activation and LoRA phases correct by adding barriers between dependent phases unless a fused per-block pipeline is proven safe.
- Keep route merge token-parallel. Do not parallelize route merge by route unless accumulation ownership is redesigned.
- Add `KT_ARM_SFT_FORWARD_TASK_MIN_ROUTES` to avoid too many tiny tasks.

Pseudocode:

```cpp
tasks = []
for expert in active_experts:
  for [b,e) in route_blocks(expert):
    tasks.push({expert,b,e})
run_arm_sft_tasks(tasks.size(), qlen, [&](int i) {
  compute_gate_up_base_block(tasks[i]);
});
barrier
run_arm_sft_tasks(tasks.size(), qlen, [&](int i) {
  compute_lora_and_activation_block(tasks[i]);
});
barrier
run_arm_sft_tasks(tasks.size(), qlen, [&](int i) {
  compute_down_block(tasks[i]);
});
```

Risks and watch items:

- WorkerPool task overhead can dominate for small `qlen` or high expert count with few routes per expert.
- Buffer writes must stay disjoint by packed route range.
- Timers must remain meaningful when phases are parallel.

Validation gate:

- Common correctness tests pass.
- Bench random, skewed, all-to-one, single-expert, and round-robin routing:

```bash
for pattern in random skewed all_to_one single_expert round_robin invalid_duplicate; do
  taskset -c 0-143 env KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
    "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
    --backend both --qlen 128 --topk 8 --rank 8 --hidden 2048 --intermediate 768 --experts 128 \
    --routing-pattern "$pattern" --threads 8 --warmup 1 --iters 3 --arm-profile \
    --output-json "profiling_kt_codex_smoke/v4_stage4_${pattern}.json"
done
```

- Profiles show effective worker use without affinity collapse.
- Skewed routing improves or at least does not regress materially.
- Qwen3 LF smoke passes validator.

## Stage 5: Group LoRA Forward Into Small GEMMs

Priority: medium-high. Base kernels should land first because they dominate and define the layout utilities.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `compute_gate_up_lora_by_expert()`
  - `compute_down_lora_by_expert()`
  - `prepare_lora_forward_weights()`
  - `transpose_lora_a_weights_for_backward()`
  - `transpose_lora_b_weights()`
  - `ForwardBuffers`
- `../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py`
  - add ARM grouped-LoRA dropout cases if not already covered by existing parametrization

Intended changes:

- Replace per-route scalar loops with per-expert small GEMMs:
  - gate/up A projection: `X[M,H] x A^T[H,R] -> U[M,R]`
  - gate/up B projection: `U[M,R] x B^T[R,I] -> delta[M,I]`
  - down A projection: `act[M,I] x A_down^T[I,R] -> U_down[M,R]`
  - down B projection: `U_down[M,R] x B_down^T[R,H] -> delta_down[M,H]`
- Use Stage 2 GEMM helpers where BF16 operands are acceptable. Use a small FP32 kernel for rank paths if conversion costs dominate.
- Keep a fast path for `dropout=0.0`.
- For dropout, generate/apply the deterministic mask before A projection and preserve `(seed, layer, projection, expert, token, route, feature)` semantics exactly.

Risks and watch items:

- Rank can be small enough that a vectorized GEMV/GEMM hybrid beats a full GEMM pack.
- Dropout determinism is non-negotiable.
- Accumulating LoRA deltas into base gate/up/down must preserve existing tolerances.

Validation gate:

- Common correctness tests pass.
- `test_sft_lora_dropout.py` passes with dropout enabled.
- Synthetic q128/q2048 profiles show lower `lora_gate_up_ms` and `lora_down_ms`.
- Run both `LORA_RANK=8` and `LORA_RANK=64` LF smokes before accepting.

## Stage 6: Sparse Active-Expert Backward Buffers

Priority: medium-high. This is the main remaining memory and backward scaling fix after the reducer bug.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `struct BackwardBuffers`
  - `backward_impl_packed()`
  - `backward_route_accumulate()`
  - `reduce_lora_grads_by_disjoint_tiles()`
  - `reduce_vector_fields()`
  - `flush_lora_grad_accum()`
  - add sparse active-expert maps based on `cache.routes.active_experts`
- `../ktransformers/kt-kernel/operators/moe-sft-tp.hpp`
  - read only; use as the local design reference
- `../ktransformers/kt-kernel/python/sft/arm.py`
  - expose sparse scratch counters if native getters are added
- `../ktransformers/kt-kernel/ext_bindings.cpp`
  - bind sparse scratch counters

Intended changes:

- Port the `moe-sft-tp.hpp` memory idea to ARM SFT:
  - Build `active_expert_to_sparse` from `cache.routes.active_experts`.
  - Allocate per-thread or per-task sparse FP32 partials for reduce-type gradients only:
    - `gate_lora_a`: `[active_count, R, H]`
    - `up_lora_a`: `[active_count, R, H]`
    - `down_lora_b`: `[active_count, H, R]`
  - Keep `grad_input_accum` per thread/token.
  - Write copy-type gradients directly only when the work partition gives disjoint ownership. Otherwise keep sparse partials and merge.
  - Flush sparse active experts into the final dense BF16 gradient tensors, leaving inactive experts zero.
- Add profile fields:
  - `active_backward_experts`
  - `dense_backward_scratch_bytes_old_estimate`
  - `sparse_backward_scratch_bytes`
  - `sparse_grad_merge_ms`

Risks and watch items:

- Direct writes to final grad tensors are safe only for disjoint slices. ARM SFT is not TP-partitioned, so do not copy the TP direct-write trick blindly.
- Accumulating duplicate routes for the same token/expert must still sum correctly.
- The final optimizer expects dense tensors with inactive experts zeroed.

Validation gate:

- Common correctness tests pass.
- Add explicit tests in `test_armbf16_sft_reference.py`:
  - inactive experts have exactly zero LoRA grads
  - duplicate routes sum correctly
  - invalid expert ids do not touch sparse buffers
  - all-to-one routing matches reference
- Synthetic q128/q2048 profiles show lower `backward_local_alloc_zero_ms` and lower scratch bytes.
- Qwen3 LF smoke passes validator.

## Stage 7: Backward Route Compute as Expert-Grouped GEMMs

Priority: medium. Do this after sparse buffers, so backward compute does not write into dense all-expert memory.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `recompute_route_forward()`
  - `backward_route_accumulate()`
  - `backward_impl_packed()`
  - Stage 2/3 GEMM helpers
  - sparse gradient merge helpers from Stage 6

Intended changes:

- Replace per-route backward scalar loops with per-expert grouped matrix operations:
  - Recompute base gate/up/down using Stage 2/3 kernels where possible.
  - `dB += U^T dY`
  - `dU += dY B^T`
  - `dA += X^T dU`
  - `dX += dU A`
  - base down/gate/up gradient-to-input paths use packed base weights.
- Keep recompute rather than storing huge forward intermediates unless Stage 1 proves cache saves are cheaper for the target shape.

Risks and watch items:

- Backward has more dependencies than forward; partial gradients and `grad_input` can race if partitioning is wrong.
- Storing intermediate `gate/up/act/down/U` can reduce compute but increase memory traffic. Benchmark both for q128 and q2048 before choosing.
- Dropout backward must reuse the exact same counter mask as forward.

Validation gate:

- Common correctness tests pass.
- Dropout tests pass.
- Synthetic q128/q2048 profiles show lower `backward_route_loop_ms` without increasing scratch bytes beyond Stage 6.
- Qwen3 LF smoke passes validator.

## Stage 8: NUMA, First-Touch, Prefetch, and Tile Autotuning

Priority: lower. Do this only after stages 2-7 remove the main structural issues.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - allocation helpers: `grow_pool()`, `ensure_forward_buffers()`, `ensure_backward_buffers()`
  - pack helpers from stages 2/3/5
  - `run_arm_sft_tasks()`
- `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
  - add tile sweep options

Intended changes:

- First-touch large buffers on the same worker set that will use them.
- Keep packed weights persistent and 64-byte aligned.
- Add prefetch only after PMU shows cache misses in kernel loops.
- Sweep tile sizes:
  - M tile: 1, 2, 4, 8, 16
  - N tile: 8, 16, 32, 64
  - reduce chunk: 4096, 8192, 16384, 32768
- Save one JSON per tile choice.

Risks and watch items:

- Tile choices can overfit q128 and regress q2048 or skewed routing.
- Manual prefetch can hurt on small route counts.
- NUMA placement is less useful while `threadpool_count=1`; do not introduce TP/NUMA partitioning in this stage.

Validation gate:

- Common correctness tests pass.
- Tile sweep artifacts include q128, q2048, random, skewed, and all-to-one routes.
- Selected tile is best or near-best across all accepted shapes, not just one shape.
- Qwen3 LF smoke passes validator.

## Stage 9: Large LF Acceptance and Upstream Readiness

Priority: final acceptance.

Modify:

- `agent/kt/scripts/run_lf_lora_sft_kt.sh`
  - only KT-specific launcher defaults or profile artifact capture
- `agent/kt/scripts/profile_lora_lf_kt.sh`
  - only KT-specific sweep defaults
- `agent/kt/scripts/validate_kt_arm_profile.py`
  - require any new profile fields needed for acceptance
- Do not modify main AsymGEMM scripts in this stage.

Intended changes:

- Run a small validated source profile first, then the large shape.
- Save:
  - `git status --short`
  - KT kernel diff
  - CPU feature report
  - disassembly BF16 hit report
  - profile JSON
  - native profile summary
  - PMU outputs where permitted
- Confirm KT script isolation. It is safe to edit main AsymGEMM scripts concurrently if KT validation uses `agent/kt/scripts/*`, but the Python runtime still imports the live AsymGEMM checkout. For publishable comparisons, freeze the AsymGEMM worktree or record all diffs.

Risks and watch items:

- The KT scripts are isolated, but the runtime still imports the live AsymGEMM checkout through `PYTHONPATH`.
- Large-shape profiles can be dominated by batch/sequence/rank/token-chunk work. Compare against the v3 post-reducer baseline, not against the original pinned/reducer-bug run.
- GPU 1 or 2 might not have enough free memory at run time. If GPU 1 fails for capacity, rerun the same command with `GPU_ID=2`, `CUDA_VISIBLE_DEVICES=2`, `NVIDIA_VISIBLE_DEVICES=2`, and `PROFILE_NSYS_GPU_METRICS_DEVICES=2`.

Validation commands:

```bash
cd "$ASYM"
mkdir -p profiling_kt_codex_smoke/v4_final_large
git status --short > profiling_kt_codex_smoke/v4_final_large/git_status.txt
(cd "$KT" && git status --short > "$ASYM/profiling_kt_codex_smoke/v4_final_large/kt_kernel_status.txt")
(cd "$KT" && git diff > "$ASYM/profiling_kt_codex_smoke/v4_final_large/kt_kernel.diff") || true
git diff -- scripts/lf/run_lf_lora_sft.sh scripts/lf/profile_lora_lf.sh \
  > profiling_kt_codex_smoke/v4_final_large/main_asym_scripts.diff || true
```

Large run, GPU 1 first:

```bash
cd "$ASYM"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT \
  ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  PROFILE_NSYS_GPU_METRICS_DEVICES=1 BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=4 \
  OUT_DIR=profiling_kt_codex_smoke/v4_final_large/qwen3_s7168_b4_r64_gpu1 \
  agent/kt/scripts/run_lf_lora_sft_kt.sh \
  2>&1 | tee profiling_kt_codex_smoke/v4_final_large/qwen3_s7168_b4_r64_gpu1/console.log
```

Validation gate:

- Validator passes.
- Profile proves physical GPU 1 or 2, not 0 or 3.
- CPU affinity count is at least `KT_NUM_THREADS`.
- Native logs prove `path=packed`, `task_dispatch=worker_pool`, `compiled_sve_bf16=1`, and BF16 instruction hits exist.
- Forward phase totals improve materially versus v3 after reducer fix.
- Backward route, allocation/zero, and scratch bytes do not regress.
- The run completes and progress counters match expected token chunks/layers.

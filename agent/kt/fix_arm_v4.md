# KT ARM BF16 SFT Fix Plan v4

This is the next implementation plan after v3. The native KT ARM BF16 path is real and not DeepSpeed. Low HBM use is expected because routed expert compute stays on CPU. The old "stuck" symptom was mainly bad CPU affinity plus a dense backward reducer that created one worker task per float. Those are fixed. Current v4 forward now uses useful ARM SVE BF16 kernels for base gate/up, base down, and dropout-0 LoRA forward. Sparse active-expert backward buffers and tile-local grouped backward recompute are implemented. The largest remaining unsolved work is true grouped backward gradient accumulation/outer-product kernels and final LF acceptance.

Rules for all stages:

- Use physical GPU 1 first and physical GPU 2 as fallback. Do not use GPU 0 or GPU 3 for KT validation.
- KT implementation edits stay under `../ktransformers/kt-kernel/**`.
- KT launcher/profile edits stay under `agent/kt/scripts/**`; do not edit `scripts/lf/run_lf_lora_sft.sh` or `scripts/lf/profile_lora_lf.sh` for this work.
- Profiles must be saved under `profiling_kt_codex_smoke/v4_*`.
- Do not move to the next stage until the validation gate for the current stage passes.
- NCU is not useful for CPU kernels. Use it only to sanity-check CUDA/GPU launch behavior. Use native timers, `perf`, disassembly, and source profiles for CPU kernel work.

## Conclusions Driving Priority

- Current hot code is in `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`.
- `compute_gate_up_base_by_expert()` defaults to `sve_bfdot_blocked` and keeps `KT_ARM_SFT_GEMM_BACKEND=dot_loop` only as a debug fallback.
- `compute_down_base_by_expert()` defaults to `bf16_bfdot_blocked`. `KT_ARM_SFT_DOWN_BACKEND=scalar` and `KT_ARM_SFT_DOWN_BACKEND=sve_fmla` remain comparison fallbacks.
- `compute_gate_up_lora_by_expert()` and `compute_down_lora_by_expert()` default to `sve_bfdot_fmla` when LoRA dropout is off. Dropout-enabled forward stays on the scalar path to preserve deterministic mask semantics.
- `forward_impl_packed()` parallelizes active experts through the existing WorkerPool for `qlen > arm_sft_small_qlen_threshold`.
- Native profile fields now include `expert_schedule_wall_ms`; per-phase expert fields such as `base_down_ms` are task-sums when expert parallelism is enabled.
- `backward_impl_packed()` now allocates sparse active-expert FP32 LoRA partials and flushes active slices into dense BF16 output gradients.
- Backward route compute is tiled by active expert. Each tile now recomputes gate/up, activation, down, and dropout-0 LoRA with the same grouped forward kernels used by the forward path. This removed the worst scalar down-recompute cost.
- The remaining backward issue is not expert scheduling; it is the per-route gradient accumulation math inside each tile: base gradient-to-input and LoRA outer products still need grouped/tiled kernels.
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

## Execution Status After Current Pass

Implemented files:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - layout/profile JSON, selected kernel names, CPU feature flags, route stats, buffer bytes, and `expert_schedule_wall_ms`
  - blocked SVE BF16 gate/up kernel: `sve_bfdot_blocked`
  - active-expert WorkerPool parallel scheduling with route blocks, so one hot expert still produces many tasks
  - dropout-0 LoRA forward fast path: `sve_bfdot_fmla`
  - BF16-converted base-down kernel: `bf16_bfdot_blocked`
  - sparse active-expert backward LoRA partial buffers, sparse-to-dense final merge, and old-vs-sparse scratch counters
  - backward route tiles, tile-local grouped recompute, route-local `grad_x` scatter, and backward worker cap to avoid idle tile buffers
  - debug fallbacks: `KT_ARM_SFT_GEMM_BACKEND=dot_loop`, `KT_ARM_SFT_DOWN_BACKEND=scalar|sve_fmla`, `KT_ARM_SFT_LORA_BACKEND=scalar`, `KT_ARM_SFT_BACKWARD_BASE_BACKEND=sve`, `KT_ARM_SFT_DISABLE_PARALLEL_EXPERTS=1`
- `../ktransformers/kt-kernel/ext_bindings.cpp`
  - bound `layout_report_json()`
- `../ktransformers/kt-kernel/python/sft/arm.py`
  - exposed parsed `layout_report` and sparse backward scratch properties
- `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
  - added `--artifact-dir`, `--forward-only`, layout/native summary artifacts, CPU/env capture
- `../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py`
  - added layout/native summary artifacts and env capture
- `../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py`
  - extended route edge-case backward checks for duplicate routes, invalid expert ids, and sparse active-expert counters

Validation already passed:

```bash
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  .venv/bin/python -m pip install -e ../ktransformers/kt-kernel -v --no-build-isolation

.venv/bin/python -m py_compile \
  ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  ../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py

.venv/bin/python -m pytest ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py -q
.venv/bin/python -m pytest ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

Latest test result:

- `test_armbf16_sft_reference.py`: 30 passed
- `test_sft_lora_dropout.py`: 13 passed

Key profiling artifacts:

- `profiling_kt_codex_smoke/v4_stage4_q128_serial_wall/`
- `profiling_kt_codex_smoke/v4_stage4_q128_parallel_wall/`
- `profiling_kt_codex_smoke/v4_stage5_q128_lora_scalar/`
- `profiling_kt_codex_smoke/v4_stage5_q128_lora_sve/`
- `profiling_kt_codex_smoke/v4_stage5_q2048_lora_scalar/`
- `profiling_kt_codex_smoke/v4_stage5_q2048_lora_sve/`
- `profiling_kt_codex_smoke/v4_stage6_q128_down_scalar/`
- `profiling_kt_codex_smoke/v4_stage6_q128_down_bf16/`
- `profiling_kt_codex_smoke/v4_stage6_q2048_down_bf16/`
- `profiling_kt_codex_smoke/v4_stage6_sparse_backward_q128_all_to_one/`
- `profiling_kt_codex_smoke/v4_stage6_sparse_backward_q2048_all_to_one/`
- `profiling_kt_codex_smoke/v4_stage7_route_block_q2048_all_to_one/`
- `profiling_kt_codex_smoke/v4_stage7_route_tile_scalar_bwd_q2048_all_to_one/`
- `profiling_kt_codex_smoke/v4_stage7_grouped_recompute_capped_q128_all_to_one/`
- `profiling_kt_codex_smoke/v4_stage7_grouped_recompute_capped_q2048_all_to_one/`
- `profiling_kt_codex_smoke/v4_stage7_grouped_recompute_split_q2048_all_to_one/`
- `profiling_kt_codex_smoke/v4_stage7_grouped_recompute_sve_bwd_q2048_all_to_one/`

Note: `v4_stage6_q*_down_*` artifact names are the BF16-down follow-up to this plan's Stage 3. `v4_stage6_sparse_backward_*` are the sparse-backward Stage 6 validation artifacts.

Measured forward-only improvements on physical GPU 1 with CPU taskset `0-15`:

| Shape | Kernel state | latency mean | expert wall | base gate/up task-sum | base down task-sum | LoRA gate/up task-sum | LoRA down task-sum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q128 r8 | serial experts, scalar down/LoRA | 510.0 ms | 482.2 ms | 179.3 ms | 259.9 ms | 28.7 ms | 12.2 ms |
| q128 r8 | parallel experts, scalar down/LoRA | 86.5 ms | 64.3 ms | 186.3 ms | 257.7 ms | 28.7 ms | 12.2 ms |
| q128 r8 | parallel, SVE LoRA, scalar down | 82.9 ms | 60.6 ms | 185.9 ms | 258.6 ms | 6.8 ms | 5.1 ms |
| q128 r8 | parallel, SVE LoRA, BF16 down | 62.7 ms | 40.1 ms | 196.1 ms | 95.9 ms | 7.1 ms | 5.2 ms |
| q2048 r64 | parallel, scalar down/LoRA | 2041.9 ms | 1711.7 ms | 2011.4 ms | 3799.5 ms | 4766.9 ms | 2852.0 ms |
| q2048 r64 | parallel, SVE LoRA, scalar down | 1248.5 ms | 925.0 ms | 2030.1 ms | 3815.5 ms | 772.3 ms | 582.1 ms |
| q2048 r64 | parallel, SVE LoRA, BF16 down | 923.4 ms | 570.0 ms | 2053.4 ms | 1010.9 ms | 786.9 ms | 586.9 ms |

Measured sparse-backward validation on physical GPU 1 with CPU taskset `0-15`:

| Shape | Routing | active backward experts | old dense scratch estimate | sparse scratch estimate | backward local alloc | backward route loop | sparse merge |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q128 r8 | all-to-one | 1 | 321.3 MB | 12.3 MB | 0.58 ms | 205.1 ms | 0.09 ms |
| q2048 r64 | all-to-one | 1 | 2642.9 MB | 170.9 MB | 8.22 ms | 4376.2 ms | 0.69 ms |

Measured Stage 7 grouped-recompute validation on physical GPU 1 with CPU taskset `0-15`:

| Shape | Routing | change | backward threads | old dense scratch estimate | sparse scratch estimate | forward total | backward route loop | total fwd+bwd |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q2048 r64 | all-to-one | route-block forward before grouped recompute | 8 | 2642.9 MB | 154.2 MB | 911.7 ms | 4412.6 ms | 5338.9 ms |
| q2048 r64 | all-to-one | grouped tile recompute, scalar base-backward | 8 | 2691.7 MB | 202.9 MB | 878.9 ms | 2517.2 ms | 3409.4 ms |
| q2048 r64 | all-to-one | grouped tile recompute, `KT_ARM_SFT_BACKWARD_BASE_BACKEND=sve` | 8 | 2691.7 MB | 202.9 MB | 892.0 ms | 5523.4 ms | 6429.8 ms |
| q128 r8 | all-to-one | grouped tile recompute, worker cap | 4 requested 8 | 202.2 MB | 33.7 MB | 74.8 ms | 192.2 ms | 267.1 ms |

Conclusion: grouped recompute is a real improvement for the hot-expert long-sequence case, but the SVE FP32xBF16 backward-base helper remains slower than the scalar default on this host. Keep it opt-in only. The next useful optimization is grouped/tiled gradient accumulation, not adding loops over experts.

The split-counter q2048 artifact (`v4_stage7_grouped_recompute_split_q2048_all_to_one`) shows why: `backward_tile_recompute_ms=4371.5` task-sum, while `backward_route_grad_accum_ms=15549.6` task-sum. The wall-clock `backward_route_loop_ms=2595.7` because 8 workers run tiles in parallel, but the task-sum split still identifies gradient accumulation as the dominant remaining CPU work.

Remaining high-priority work:

- Stage 6 sparse active-expert backward buffers is implemented and validated on synthetic q128/q2048 all-to-one routing.
- Stage 7 tile-local grouped recompute is implemented and validated on synthetic q128/q2048 all-to-one routing.
- Stage 7 grouped gradient accumulation remains open and is now the largest measured CPU issue.
- Route merge is still a visible wall-time component at long sequence (`~279 ms` in latest q2048 all-to-one full profile).
- Large LF acceptance is still pending; use GPU 1 first and GPU 2 only as fallback.

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

Status: implemented and validated. This stage removes dense per-thread all-expert LoRA partials, but it does not solve the route-loop scalar backward compute. Do not treat Stage 6 as a grouped-kernel solution; Stage 7 grouped recompute is implemented and Stage 7 grouped gradient accumulation is still required.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `struct BackwardBuffers`
  - `backward_impl_packed()`
  - `backward_route_accumulate()`
  - `reduce_lora_grads_by_disjoint_tiles()`
  - `reduce_vector_fields()`
  - `flush_sparse_lora_grad_accum()`
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
  - Allocate per-thread sparse FP32 partials for active experts:
    - `gate_lora_a`: `[active_count, R, H]`
    - `gate_lora_b`: `[active_count, I, R]`
    - `up_lora_a`: `[active_count, R, H]`
    - `up_lora_b`: `[active_count, I, R]`
    - `down_lora_a`: `[active_count, R, I]`
    - `down_lora_b`: `[active_count, H, R]`
  - Keep `grad_input_accum` per thread/token.
  - Do not directly write final dense gradient tensors from the route loop. ARM SFT is not TP-sliced, so all LoRA gradient partials stay sparse until final merge.
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

- Common correctness tests pass:
  - `CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 GPU_ID=1 NUM_GPUS=1 .venv/bin/python -m pytest ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py -q`
  - `CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 GPU_ID=1 NUM_GPUS=1 .venv/bin/python -m pytest ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q`
- Explicit tests in `test_armbf16_sft_reference.py` cover:
  - inactive experts have exactly zero LoRA grads
  - duplicate routes sum correctly
  - invalid expert ids do not touch sparse buffers
  - all-to-one routing matches reference
- Synthetic q128/q2048 profiles show lower scratch bytes:
  - `profiling_kt_codex_smoke/v4_stage6_sparse_backward_q128_all_to_one/`
  - `profiling_kt_codex_smoke/v4_stage6_sparse_backward_q2048_all_to_one/`
- Qwen3 LF smoke is still pending and should be run after Stage 7 grouped gradient accumulation, because Stage 6 exposed `backward_route_loop_ms` as the dominant unsolved bottleneck.

## Stage 7: Backward Route Compute as Expert-Grouped GEMMs

Priority: medium. Do this after sparse buffers, so backward compute does not write into dense all-expert memory.

Status: partially implemented and validated. Backward no longer performs the expensive forward recompute as a per-route scalar path: each `BackwardExpertTile` now runs the existing grouped forward kernels for base gate/up, dropout-0 LoRA gate/up, activation, BF16 base down, and dropout-0 LoRA down before per-route gradient accumulation. This fixes the worst hot-expert recompute problem without adding serial expert loops. The remaining work in this stage is grouped/tiled gradient accumulation.

Modify:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - tile-local recompute helpers:
    - `init_backward_recompute_tile_buffers()`
    - `fill_backward_recompute_tile()`
    - existing grouped forward helpers called from backward
  - `backward_route_accumulate()`
  - `backward_impl_packed()`
  - Stage 2/3 GEMM helpers
  - sparse gradient merge helpers from Stage 6

Implemented changes:

- Build `BackwardExpertTile` records from packed active-expert routes.
- Cap backward OpenMP workers to `min(KT_ARM_SFT_BACKWARD_THREADS, backward_route_tiles)` so small all-to-one cases do not allocate idle tile buffers.
- Allocate one `ForwardBuffers` tile per active backward worker with bounded route count `KT_ARM_SFT_BACKWARD_ROUTE_TILE` (default 256).
- Fill tile-local packed BF16 inputs and route metadata from the saved cache.
- Run existing grouped forward kernels on the tile:
  - `compute_gate_up_base_by_expert()`
  - `compute_gate_up_lora_by_expert()`
  - `apply_activation_by_expert()`
  - `compute_down_base_by_expert()`
  - `compute_down_lora_by_expert()`
- Keep route-local `grad_x` storage and scatter to token gradients after tile compute so duplicate routes are correct without atomics.
- Keep `KT_ARM_SFT_BACKWARD_BASE_BACKEND=sve` as opt-in only; it regressed q2048 all-to-one from `2517.2 ms` to `5523.4 ms` in `backward_route_loop_ms`.
- Add split profile counters:
  - `backward_tile_recompute_ms`
  - `backward_route_grad_accum_ms`

Remaining intended changes:

- Replace per-route gradient accumulation loops with tile/grouped operations:
  - `dB += U^T dY`
  - `dU += dY B^T`
  - `dA += X^T dU`
  - `dX += dU A`
  - base down/gate/up gradient-to-input paths use packed base weights.
- Keep recompute rather than storing huge forward intermediates unless a later profile proves cache saves are cheaper for the target LF shape.

Risks and watch items:

- Backward has more dependencies than forward; partial gradients and `grad_input` can race if partitioning is wrong.
- Tile recompute stores `gate/up/act/down/U` only for one worker tile, not for the full layer cache. This adds bounded scratch (`~48.8 MB` for 8 q2048 workers at tile 256) and avoids a full saved-forward cache.
- Grouped gradient accumulation can race on sparse LoRA partials if tile ownership is not preserved. Either keep per-worker sparse partials and reduce, or partition output gradient tiles by sparse expert and parameter slice.
- Dropout backward must reuse the exact same counter mask as forward.

Validation gate:

- Common correctness tests pass: `test_armbf16_sft_reference.py` 30 passed.
- Dropout tests pass: `test_sft_lora_dropout.py` 13 passed.
- Synthetic q128/q2048 profiles show lower `backward_route_loop_ms`:
  - q128 all-to-one: `205.1 ms` Stage 6 to `192.2 ms` grouped recompute with worker cap.
  - q2048 all-to-one: `4412.6 ms` route-tile scalar recompute to `2517.2 ms` grouped recompute.
- q2048 split profile shows `backward_route_grad_accum_ms` task-sum is roughly 3.6x `backward_tile_recompute_ms`, so the next kernel work should group gradient accumulation and LoRA outer products.
- Watch: sparse scratch estimate rose for q2048 all-to-one from `154.2 MB` route-tile scalar recompute to `202.9 MB` grouped recompute because tile-local forward buffers are now counted. This is still far below the old dense estimate (`2691.7 MB`) and bounded by tile size.
- Qwen3 LF smoke is still pending and should run before Stage 9 acceptance.

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

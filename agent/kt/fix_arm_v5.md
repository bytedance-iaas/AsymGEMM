# KT ARM BF16 SFT Fix Plan v5

This is the remaining implementation plan after v4. The KT ARM BF16 path is already native ARM CPU code, not DeepSpeed. v4 fixed the worst dense backward memory issue and moved forward plus backward recompute onto grouped route tiles. The measured remaining CPU bottleneck is still inside backward route gradient accumulation, especially LoRA outer products and base gradient-to-input math that still run per route.

The plan below is implementation-focused and intentionally keeps KT work isolated:

- Kernel/runtime edits stay under `../ktransformers/kt-kernel/**`.
- KT LF launcher/profile edits stay under `agent/kt/scripts/**` and the thin wrappers in `scripts/kt/**`.
- Do not edit `scripts/lf/run_lf_lora_sft.sh` or `scripts/lf/profile_lora_lf.sh` for this work.
- Use physical GPU 1 first and physical GPU 2 only as fallback. Do not use GPU 0 or GPU 3 for KT testing.
- Do not accept performance work from toy profiling alone. Small synthetic tests are only a correctness/debug gate; meaningful kernel stages must run the LF KT source profile before moving on.
- Profiles for this plan go under `profiling_kt_codex_smoke/v5_*`.

## Current Ground Truth

Completed in v4:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - active-expert route-block forward scheduling
  - SVE BF16/BFDOT forward gate/up and BF16-down paths
  - dropout-0 SVE LoRA forward path
  - sparse active-expert FP32 backward LoRA partials
  - backward route tiles
  - tile-local grouped forward recompute inside backward
  - split counters `backward_tile_recompute_ms` and `backward_route_grad_accum_ms`
  - SVE backward-base opt-in through `KT_ARM_SFT_BACKWARD_BASE_BACKEND=sve`; keep it non-default because the current helper regressed
- `../ktransformers/kt-kernel/python/sft/arm.py`
  - parsed layout report and sparse backward scratch properties
- `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
  - artifact output, profile summaries, env capture
- `agent/kt/scripts/run_lf_lora_sft_kt.sh`
- `agent/kt/scripts/profile_lora_lf_kt.sh`
- `scripts/kt/run_lf_lora_sft_kt.sh`
- `scripts/kt/profile_lora_lf_kt.sh`

Latest known synthetic evidence from v4:

- q2048/r64 all-to-one route-block before grouped recompute: total about 5339 ms, `backward_route_loop_ms` about 4413 ms.
- q2048/r64 grouped recompute with scalar backward base: total about 3409 ms, `backward_route_loop_ms` about 2517 ms.
- q2048/r64 grouped recompute split counters: `backward_tile_recompute_ms` task-sum about 4372 ms, `backward_route_grad_accum_ms` task-sum about 15550 ms.
- q2048/r64 SVE backward-base opt-in regressed: total about 6430 ms, `backward_route_loop_ms` about 5523 ms.
- q128/r8 grouped recompute with worker cap: total about 267 ms, `backward_route_loop_ms` about 192 ms.

Known gaps from v4:

- Full LF e2e KT profiling has not been completed for the v4 kernel state.
- `agent/kt/scripts/profile_lora_lf_kt.sh` is KT-isolated, but its current default backend sweep is still the Asym CPUAdamW profile default. For KT work, either pass `BACKEND_SPECS='kt_armbf16|recomp'` every time or change the KT-specific script default in Stage 1.
- The current backward route loop still calls `backward_route_accumulate()` once per route. That is the main remaining performance issue.
- `scatter_route_grad_x_to_tokens()` and forward `merge_routes_to_output()` still perform scalar per-token/per-route/per-hidden loops and are visible at long sequence.

## Lessons From Original KTransformers Code

Local code paths checked:

- `../ktransformers/kt-kernel/operators/amx/la/avx_kernels.hpp`
  - `lora_bf16_matmul_t4r4()`: processes multiple tokens and rank lanes together, loads each weight row once, reuses it across token rows.
  - `lora_backward_matmul_transposed()`: uses pre-transposed LoRA B as `[rank, hidden]` or `[rank, intermediate]` so each rank dot reads contiguous memory.
- `../ktransformers/kt-kernel/operators/amx/sft_moe.hpp`
  - LoRA backward is split into tile tasks: compute `u = x @ A^T`, accumulate `grad_B`, compute `grad @ B^T`, add LoRA grad-input, and compute `grad_A` using hidden-dimension blocks.
  - Large per-expert work is broken into token/rank/hidden/intermediate tiles; it does not serialize a whole expert on one thread.
- `../ktransformers/kt-kernel/operators/moe-sft-tp.hpp`
  - active experts are snapshotted once before backward.
  - reduce-type LoRA gradients use sparse FP32 partials scoped to active experts.
  - copy/disjoint slices are written directly where ownership is clear.

Design conclusion:

- Do not add a serial loop over experts for compute. It is acceptable to iterate active experts only to build route tiles and task descriptors. Actual math must run as grouped route/expert tiles through OpenMP or the KT WorkerPool.
- The next kernels should work on `M` routes at a time for one expert tile, where `M = tile.end - tile.begin`, not one route at a time.
- Prefer layouts already present in the ARM path:
  - base gate/up: `[expert, intermediate, hidden]`
  - base down: `[expert, hidden, intermediate]` plus existing transposed down layout where useful
  - LoRA A: `[expert, rank, hidden_or_intermediate]`
  - LoRA B transposed helpers already exist for forward: `[expert, padded_rank, output_dim]`
- Keep SVE BF16/BFDOT for BF16 x BF16 forward-style dot products. For backward FP32 x BF16 paths, the first v4 SVE helper was slower because its loop order did not reuse weights across route rows. v5 must use route-tile kernels that reuse RHS weights across multiple `M` rows.

External references to keep nearby during implementation:

- Arm ACLE SVE/SVE BF16 intrinsics: https://arm-software.github.io/acle/main/acle.html
- Arm SVE vector-length model and 128-to-2048-bit bound: https://developer.arm.com/documentation/102476/0101/Introducing-SVE
- Arm BF16 instruction behavior, including BFDOT/BFMMLA FP32 accumulation: https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bfloat16-processing-for-neural-networks-on-armv8_2d00_a
- Arm Neon/SVE/SME matmul blocking comparison: https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/matrix-matrix-multiplication-neon-sve-and-sme-compared
- Arm KleidiAI matmul kernel layouts and tile-driven API style: https://github.com/ARM-software/kleidiai
- Arm Compute Library CPU kernels for layout/packing examples: https://github.com/ARM-software/ComputeLibrary
- BLIS microkernel design notes for small `m,n,k` edge-aware kernels: https://github.com/flame/blis/blob/master/docs/KernelsHowTo.md

Online/code-search refinements from this pass:

- ACLE exposes vector-length queries such as `svcntb()` and `svcntw()` and uses predicated SVE loads/stores. The existing ARM code's `load_bf16_low_as_f32()` widens the low half of a BF16 vector to FP32, so FP32 x BF16 helper loops must advance by `svcntw()` FP32 lanes and use the same `svwhilelt_b16(idx, n)` pattern already used in `arm_fp32_bf16_dot_sve()`.
- Arm's BF16 dot/matrix instructions are relevant to BF16 x BF16 forward/recompute paths because they take BF16 inputs and accumulate into FP32. They do not directly solve the Stage 3/4 backward helpers where the left operand is already FP32 (`dy`, `grad_gate`, `grad_up`, or `grad_u`), so the immediate win should come from route-tile reuse, SVE predicated contiguous BF16 loads, and reduced loop overhead.
- KleidiAI's public docs emphasize micro-kernels that process only a tile of output and leave scheduling/packing decisions to the caller. That matches the v5 direction: keep route scheduling in `backward_impl_packed()` and make the new helpers stateless tile kernels.
- BLIS also treats optimized GEMM as a small edge-aware microkernel called by higher-level blocking code. That supports the same split here: `backward_impl_packed()` owns route/expert scheduling, while the new helpers own only `M x N` or `M x R` tile math.
- The current ARM header confirms these exact contiguous dimensions:
  - `gate_up_idx(expert, i, h)` and `up_proj_bf16_` are contiguous over `h`.
  - `down_idx(expert, h, i)` is contiguous over `i`.
  - `lora_a_h_idx(expert, r, h)` and `down_lora_a_idx(expert, r, i)` are contiguous over the feature dimension.
  - `lora_b_i_idx(expert, i, r)` and `down_lora_b_idx(expert, h, r)` are contiguous over rank `r`.
  - forward `gate_u/up_u/down_u` are `[M, PR]`, while scratch `grad_*_u` and sparse LoRA gradient tensors are `[M, R]` or `[K, R]`.

## Common Commands

Run these from the AsymGEMM root:

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

Build and unit gate:

```bash
cd "$KT"
CPUINFER_FORCE_REBUILD=1 \
CPUINFER_BUILD_TYPE=RelWithDebInfo \
CPUINFER_PARALLEL=16 \
"$PY" -m pip install -e . -v --no-build-isolation

cd "$ASYM"
bash -n agent/kt/scripts/run_lf_lora_sft_kt.sh
bash -n agent/kt/scripts/profile_lora_lf_kt.sh
bash -n scripts/kt/run_lf_lora_sft_kt.sh
bash -n scripts/kt/profile_lora_lf_kt.sh
"$PY" -m py_compile \
  agent/kt/scripts/validate_kt_arm_profile.py \
  ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  ../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py -q
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

Baseline synthetic profile template:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stageX_synth_q128_r8
mkdir -p "$ART"
taskset -c 0-143 env \
  KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend both --qlen 128 --topk 8 --rank 8 \
  --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 3 --arm-profile \
  --artifact-dir "$ART" \
  --output-json "$ART/result.json" \
  2>&1 | tee "$ART/console.log"

ART=profiling_kt_codex_smoke/v5_stageX_synth_q2048_r64
mkdir -p "$ART"
taskset -c 0-143 env \
  KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness --qlen 2048 --topk 8 --rank 64 \
  --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 2 --arm-profile \
  --artifact-dir "$ART" \
  --output-json "$ART/result.json" \
  2>&1 | tee "$ART/console.log"
```

LF KT source smoke:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stageX_qwen3_s64_b1_r8_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT \
  ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  PROFILE_NSYS_GPU_METRICS_DEVICES=1 BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=1 \
  OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh \
  2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 --expected-batch 1 --expected-rank 8 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --require-final
```

LF KT long source profile:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stageX_qwen3_s7168_b4_r64_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT \
  ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  PROFILE_NSYS_GPU_METRICS_DEVICES=1 BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 \
  KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=4 \
  OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh \
  2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 7168 --expected-batch 4 --expected-rank 64 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --allow-unvalidated-route-rank 1 --require-final
```

KT profile sweep wrapper:

```bash
cd "$ASYM"
GPU_POOL=1,2 \
BACKEND_SPECS='kt_armbf16|recomp' \
PROFILERS=source \
SEQ_LENS=7168 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
LORA_RANK=64 \
LORA_DROPOUT=0.00 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
MAX_SAMPLES=4 \
KT_NUM_THREADS=8 \
KT_ARM_OMP_NUM_THREADS=8 \
KT_ARM_OMP_PROC_BIND=false \
KT_ARM_SFT_BACKWARD_THREADS=8 \
KT_ARM_SFT_PROFILE=1 \
KT_ARM_SFT_POOL_LOG=1 \
KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 \
KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
OUTPUT_ROOT=profiling_kt_codex_smoke/v5_stageX_profile_sweep \
scripts/kt/profile_lora_lf_kt.sh
```

## Stage 1: Lock KT-Isolated Profiling Defaults And Baseline

Scope:

- `agent/kt/scripts/profile_lora_lf_kt.sh`
- `agent/kt/scripts/run_lf_lora_sft_kt.sh`
- `scripts/kt/README.md`
- `agent/kt/scripts/validate_kt_arm_profile.py`
- no kernel changes

Implementation:

- Change the KT-specific profile script default backend from the shared Asym default to KT:

```bash
BACKEND_SPECS=${BACKEND_SPECS:-"kt_armbf16|recomp"}
```

- Keep `GPU_POOL=${GPU_POOL:-1,2}`.
- Keep `KT_ARM_OMP_PROC_BIND=false` in `run_lf_lora_sft_kt.sh`; if `profile_lora_lf_kt.sh` still defaults it to `close`, change only the KT script default to `false`.
- Extend `validate_kt_arm_profile.py` to optionally require native split fields. Add arguments:

```python
parser.add_argument("--require-native-field", action="append", default=[])
parser.add_argument("--max-native-field-ms", action="append", default=[])
```

- Implement profile field lookup against the KT rows and native log lines. Accept either JSON-native fields or parsed `KT_ARM_SFT_PROFILE` log keys.

Pseudocode:

```python
def collect_native_numbers(source, log_text):
    values = {}
    for row in source.get("kt", {}).get("rows", []):
        if row.get("method") == "ARMBF16_SFT":
            for key, value in row.items():
                maybe_add_float(values, key, value)
    for line in log_text.splitlines():
        if "KT_ARM_SFT_PROFILE" not in line:
            continue
        for key, value in re.findall(r"([a-zA-Z0-9_]+)=([0-9.]+)", line):
            maybe_add_float(values, key, value)
    return values

for key in args.require_native_field:
    if key not in values:
        fail(f"missing native field {key}")

for spec in args.max_native_field_ms:
    key, limit = spec.split("=", 1)
    if values.get(key, inf) > float(limit):
        fail(f"{key} exceeds limit")
```

- Update `scripts/kt/README.md` with the exact KT command and a warning that `scripts/lf/**` is intentionally untouched.

Risks to watch:

- The LF profile JSON may not carry every native timing field yet. If so, the validator must read the train log next to the profile JSON and should not require schema changes in the shared LF profiler.
- Changing the KT script default can surprise old Asym comparison sweeps if someone used the KT script as a generic wrapper. That is acceptable because `scripts/kt` is explicitly KT-specific.

Validation before Stage 2:

```bash
cd "$ASYM"
for script in \
  agent/kt/scripts/profile_lora_lf_kt.sh \
  agent/kt/scripts/run_lf_lora_sft_kt.sh \
  scripts/kt/profile_lora_lf_kt.sh \
  scripts/kt/run_lf_lora_sft_kt.sh; do
  bash -n "$script"
done
"$PY" -m py_compile agent/kt/scripts/validate_kt_arm_profile.py
```

Run the LF KT source smoke with Stage 1 artifact names:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage1_qwen3_s64_b1_r8_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=1 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 --expected-batch 1 --expected-rank 8 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --require-final \
  --require-native-field backward_route_grad_accum_ms \
  --require-native-field backward_tile_recompute_ms
```

## Stage 2: Refactor Backward Route Accumulation Into Tile APIs

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - class `ARM_BF16_SFT_MOE`
  - `struct BackwardBuffers`
  - `struct BackwardExpertTile`
  - `struct ArmSFTProfileStats`
  - `append_profile_stats_json()`
  - `backward_impl_packed()`
  - `backward_route_accumulate()`
  - new helpers listed below
- `../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py`

Implementation:

- Split the current monolithic `backward_route_accumulate()` into route-local helper pieces, then rebuild `backward_route_accumulate()` as the scalar reference composition. This is required so Stage 3 can replace base math without recomputing LoRA and Stage 4 can replace LoRA math without touching base math.

```cpp
struct BackwardRouteView {
  int expert = 0;
  int sparse_expert = 0;
  int token = 0;
  int logical_token = 0;
  int route = 0;
  float route_weight = 0.0f;
  const float* x = nullptr;       // [H], converted from BF16 tile input
  const float* gate = nullptr;    // [I], recomputed forward values
  const float* up = nullptr;      // [I]
  const float* act = nullptr;     // [I]
  const float* down = nullptr;    // [H]
  const float* gate_u = nullptr;  // [PR], first R valid
  const float* up_u = nullptr;    // [PR], first R valid
  const float* down_u = nullptr;  // [PR], first R valid
};

void backward_route_prepare_common(const BackwardRouteView& v,
                                   const ggml_bf16_t* grad_output,
                                   float* grad_weight,
                                   float* dy /* [H] */) const {
  float gw = 0.0f;
  for (int h = 0; h < H; ++h) {
    float gy = bf16_to_f32(grad_output[static_cast<size_t>(v.token) * H + h]);
    gw += gy * v.down[h];
    dy[h] = gy * v.route_weight;
  }
  *grad_weight = gw;
}

void backward_route_base_down_scalar(const BackwardRouteView& v,
                                     const float* dy,
                                     float* grad_act /* [I], accumulate */) const {
  for (int h = 0; h < H; ++h) {
    for (int i = 0; i < I; ++i) {
      grad_act[i] += bf16_to_f32(down_proj_bf16_[down_idx(v.expert, h, i)]) * dy[h];
    }
  }
}

void backward_route_down_lora_scalar(const BackwardRouteView& v,
                                     const float* dy,
                                     float* grad_act /* [I], accumulate */,
                                     float* grad_down_u /* [R] */,
                                     BackwardBuffers& local,
                                     bool dropout_enabled,
                                     uint64_t dropout_seed) const {
  fill grad_down_u[0:R] with 0
  for h in 0..H:
    for r in 0..R:
      b = bf16_to_f32(down_lora_b_[down_lora_b_idx(v.expert, h, r)])
      grad_down_u[r] += scale * dy[h] * b
      local.grad_down_lora_b_accum[sparse_down_lora_b_idx(v.sparse_expert, h, r)] +=
          scale * dy[h] * v.down_u[r]
  for r in 0..R:
    for i in 0..I:
      mask = dropout_enabled ? lora_dropout_scale(dropout_seed, config_.layer_idx, KT_LORA_DROPOUT_DOWN,
                                                  v.expert, v.logical_token, v.route, i, config_.lora_dropout) : 1
      a = bf16_to_f32(down_lora_a_[down_lora_a_idx(v.expert, r, i)])
      grad_act[i] += grad_down_u[r] * a * mask
      local.grad_down_lora_a_accum[sparse_down_lora_a_idx(v.sparse_expert, r, i)] +=
          grad_down_u[r] * v.act[i] * mask
}

void backward_route_activation_scalar(const float* gate, const float* up, const float* grad_act,
                                      float* grad_gate, float* grad_up) const {
  for (int i = 0; i < I; ++i) {
    grad_gate[i] = grad_act[i] * up[i] * silu_grad(gate[i]);
    grad_up[i] = grad_act[i] * silu(gate[i]);
  }
}

void backward_route_base_gate_up_scalar(const BackwardRouteView& v,
                                        const float* grad_gate,
                                        const float* grad_up,
                                        float* grad_x /* [H], accumulate */) const {
  for i in 0..I:
    for h in 0..H:
      grad_x[h] += bf16_to_f32(gate_proj_bf16_[gate_up_idx(v.expert, i, h)]) * grad_gate[i]
      grad_x[h] += bf16_to_f32(up_proj_bf16_[gate_up_idx(v.expert, i, h)]) * grad_up[i]
}

void backward_route_gate_up_lora_scalar(const BackwardRouteView& v,
                                        const float* grad_gate,
                                        const float* grad_up,
                                        float* grad_x /* [H], accumulate */,
                                        float* grad_gate_u /* [R] */,
                                        float* grad_up_u /* [R] */,
                                        BackwardBuffers& local,
                                        bool dropout_enabled,
                                        uint64_t dropout_seed) const {
  fill grad_gate_u[0:R] and grad_up_u[0:R] with 0
  for i in 0..I:
    for r in 0..R:
      gate_b = bf16_to_f32(gate_lora_b_[lora_b_i_idx(v.expert, i, r)])
      up_b = bf16_to_f32(up_lora_b_[lora_b_i_idx(v.expert, i, r)])
      grad_gate_u[r] += scale * grad_gate[i] * gate_b
      grad_up_u[r] += scale * grad_up[i] * up_b
      local.grad_gate_lora_b_accum[sparse_lora_b_i_idx(v.sparse_expert, i, r)] += scale * grad_gate[i] * v.gate_u[r]
      local.grad_up_lora_b_accum[sparse_lora_b_i_idx(v.sparse_expert, i, r)] += scale * grad_up[i] * v.up_u[r]
  for r in 0..R:
    for h in 0..H:
      gate_mask = dropout_enabled ? lora_dropout_scale(dropout_seed, config_.layer_idx, KT_LORA_DROPOUT_GATE,
                                                       v.expert, v.logical_token, v.route, h, config_.lora_dropout) : 1
      up_mask = dropout_enabled ? lora_dropout_scale(dropout_seed, config_.layer_idx, KT_LORA_DROPOUT_UP,
                                                     v.expert, v.logical_token, v.route, h, config_.lora_dropout) : 1
      gate_a = bf16_to_f32(gate_lora_a_[lora_a_h_idx(v.expert, r, h)])
      up_a = bf16_to_f32(up_lora_a_[lora_a_h_idx(v.expert, r, h)])
      grad_x[h] += grad_gate_u[r] * gate_a * gate_mask
      grad_x[h] += grad_up_u[r] * up_a * up_mask
      local.grad_gate_lora_a_accum[sparse_lora_a_h_idx(v.sparse_expert, r, h)] += grad_gate_u[r] * v.x[h] * gate_mask
      local.grad_up_lora_a_accum[sparse_lora_a_h_idx(v.sparse_expert, r, h)] += grad_up_u[r] * v.x[h] * up_mask
}
```

- Rebuild `backward_route_accumulate()` from those helpers. It remains the scalar truth implementation and must produce byte-identical behavior to the current code within existing BF16 tolerances.

```cpp
void backward_route_accumulate(int expert, int sparse_expert, int token, int logical_token, int route, const float* x,
                               const float* gate, const float* up, const float* act, const float* down,
                               const float* gate_u, const float* up_u, const float* down_u,
                               const ggml_bf16_t* grad_output, float route_weight, float* grad_weight,
                               std::vector<float>& grad_x,
                               std::vector<float>& grad_act, std::vector<float>& grad_gate,
                               std::vector<float>& grad_up, std::vector<float>& route_grad_x,
                               std::vector<float>& grad_gate_u, std::vector<float>& grad_up_u,
                               std::vector<float>& grad_down_u, std::vector<float>& dy,
                               std::vector<float>& grad_gate_lora_a_accum,
                               std::vector<float>& grad_gate_lora_b_accum,
                               std::vector<float>& grad_up_lora_a_accum,
                               std::vector<float>& grad_up_lora_b_accum,
                               std::vector<float>& grad_down_lora_a_accum,
                               std::vector<float>& grad_down_lora_b_accum,
                               bool dropout_enabled, uint64_t dropout_seed, bool atomic_lora_accum) const {
  BackwardRouteView v;
  v.expert = expert;
  v.sparse_expert = sparse_expert;
  v.token = token;
  v.logical_token = logical_token;
  v.route = route;
  v.route_weight = route_weight;
  v.x = x;
  v.gate = gate;
  v.up = up;
  v.act = act;
  v.down = down;
  v.gate_u = gate_u;
  v.up_u = up_u;
  v.down_u = down_u;
  fill grad_act[0:I] with 0
  fill route_grad_x[0:H] with 0
  backward_route_prepare_common(v, grad_output, grad_weight, dy.data())
  backward_route_base_down_scalar(v, dy.data(), grad_act.data())
  backward_route_down_lora_scalar(v, dy.data(), grad_act.data(), grad_down_u.data(), local, dropout_enabled, dropout_seed)
  backward_route_activation_scalar(v.gate, v.up, grad_act.data(), grad_gate.data(), grad_up.data())
  backward_route_base_gate_up_scalar(v, grad_gate.data(), grad_up.data(), route_grad_x.data())
  backward_route_gate_up_lora_scalar(v, grad_gate.data(), grad_up.data(), route_grad_x.data(), grad_gate_u.data(), grad_up_u.data(), local, dropout_enabled, dropout_seed)
  for h in 0..H:
    grad_x[h] += route_grad_x[h]
}
```

- Add `BackwardTileScratch` separate from `ForwardBuffers`. These buffers are route-tile shaped and are reused by each OpenMP worker.

```cpp
struct BackwardTileScratch {
  int capacity_routes = 0;
  std::vector<float> x_f32;        // [M, H]
  std::vector<float> dy;           // [M, H]
  std::vector<float> grad_act;     // [M, I]
  std::vector<float> grad_gate;    // [M, I]
  std::vector<float> grad_up;      // [M, I]
  std::vector<float> grad_x;       // [M, H]
  std::vector<float> grad_gate_u;  // [M, R]
  std::vector<float> grad_up_u;    // [M, R]
  std::vector<float> grad_down_u;  // [M, R]
};
```

```cpp
void init_backward_tile_scratch(BackwardTileScratch& s, int max_routes) const {
  const int H = config_.hidden_size;
  const int I = config_.intermediate_size;
  const int R = config_.lora_rank;
  const int M = std::max(1, max_routes);
  s.capacity_routes = M;
  s.x_f32.assign(static_cast<size_t>(M) * H, 0.0f);
  s.dy.assign(static_cast<size_t>(M) * H, 0.0f);
  s.grad_act.assign(static_cast<size_t>(M) * I, 0.0f);
  s.grad_gate.assign(static_cast<size_t>(M) * I, 0.0f);
  s.grad_up.assign(static_cast<size_t>(M) * I, 0.0f);
  s.grad_x.assign(static_cast<size_t>(M) * H, 0.0f);
  s.grad_gate_u.assign(static_cast<size_t>(M) * R, 0.0f);
  s.grad_up_u.assign(static_cast<size_t>(M) * R, 0.0f);
  s.grad_down_u.assign(static_cast<size_t>(M) * R, 0.0f);
}
```

- Add a behavior-preserving tile wrapper. For Stage 2, both scalar and grouped dispatch may call this wrapper; the goal is stable control flow and scratch ownership before adding kernels.

```cpp
void backward_tile_accumulate_scalar_reference(
    const CacheEntry& cache,
    const BackwardExpertTile& tile,
    const PackedRoutes& tile_routes,
    const ForwardBuffers& fwd,
    const ggml_bf16_t* grad_output,
    float* grad_weights,
    float* route_grad_x_accum,
    BackwardTileScratch& scratch,
    BackwardBuffers& local) const {
  const int M = tile.routes;
  for (int local_route = 0; local_route < M; ++local_route) {
    int packed = tile.begin + local_route;
    int token = tile_routes.packed_token_ids[local_route];
    int route = tile_routes.packed_route_slots[local_route];
    int logical_token = cache.token_offset + token;
    float* x = scratch.x_f32.data() + static_cast<size_t>(local_route) * H;
    float* dy = scratch.dy.data() + static_cast<size_t>(local_route) * H;
    float* grad_act = scratch.grad_act.data() + static_cast<size_t>(local_route) * I;
    float* grad_gate = scratch.grad_gate.data() + static_cast<size_t>(local_route) * I;
    float* grad_up = scratch.grad_up.data() + static_cast<size_t>(local_route) * I;
    float* grad_x = scratch.grad_x.data() + static_cast<size_t>(local_route) * H;
    float* grad_gate_u = scratch.grad_gate_u.data() + static_cast<size_t>(local_route) * R;
    float* grad_up_u = scratch.grad_up_u.data() + static_cast<size_t>(local_route) * R;
    float* grad_down_u = scratch.grad_down_u.data() + static_cast<size_t>(local_route) * R;
    for (int h = 0; h < H; ++h) {
      x[h] = bf16_to_f32(fwd.packed_input[static_cast<size_t>(local_route) * H + h]);
      grad_x[h] = 0.0f;
    }
    backward_route_accumulate(tile.expert, tile.sparse_expert, token, logical_token, route, x,
        fwd.gate.data() + static_cast<size_t>(local_route) * I,
        fwd.up.data() + static_cast<size_t>(local_route) * I,
        fwd.act.data() + static_cast<size_t>(local_route) * I,
        fwd.down.data() + static_cast<size_t>(local_route) * H,
        fwd.gate_u.data() + static_cast<size_t>(local_route) * PR,
        fwd.up_u.data() + static_cast<size_t>(local_route) * PR,
        fwd.down_u.data() + static_cast<size_t>(local_route) * PR,
        grad_output, cache.weights[static_cast<size_t>(token) * cache.k + route],
        grad_weights + static_cast<size_t>(token) * cache.k + route,
        grad_x, grad_act, grad_gate, grad_up, grad_x, grad_gate_u, grad_up_u, grad_down_u, dy,
        local.grad_gate_lora_a_accum, local.grad_gate_lora_b_accum,
        local.grad_up_lora_a_accum, local.grad_up_lora_b_accum,
        local.grad_down_lora_a_accum, local.grad_down_lora_b_accum,
        cache.dropout_enabled, cache.dropout_seed, false);
    std::copy(grad_x, grad_x + H, route_grad_x_accum + static_cast<size_t>(packed) * H);
  }
}
```

- Replace the direct `for local_route` block in `backward_impl_packed()` with a tile dispatch:

```cpp
if (use_grouped_backward_grad_backend()) {
  backward_tile_accumulate_grouped(cache, tile, tile_routes, tile_buffers, grad_output,
                                  grad_weights, route_grad_x_accum.data(),
                                  tile_scratch, local);
} else {
  backward_tile_accumulate_scalar_reference(cache, tile, tile_routes, tile_buffers, grad_output,
                                           grad_weights, route_grad_x_accum.data(),
                                           tile_scratch, local);
}
```

- Add selector:

```cpp
bool use_grouped_backward_grad_backend() const {
  const char* backend = std::getenv("KT_ARM_SFT_BACKWARD_GRAD_BACKEND");
  return backend != nullptr && std::strcmp(backend, "grouped") == 0;
}
```

- Default remains scalar until Stage 6 explicitly promotes grouped kernels after LF e2e acceptance.
- Update scratch estimates in `backward_impl_packed()`:

```cpp
const size_t backward_tile_scratch_float_elems =
    static_cast<size_t>(max_backward_tile_routes) *
    (3 * static_cast<size_t>(H) + 3 * static_cast<size_t>(I) + 3 * static_cast<size_t>(R));
estimated_temp_scratch_bytes += backward_tile_scratch_float_elems * sizeof(float) * static_cast<size_t>(threads);
```

- Extend `test_armbf16_sft_reference.py`:
  - Parametrize `test_armbf16_sft_forward_backward_matches_reference` over `KT_ARM_SFT_BACKWARD_GRAD_BACKEND` unset and `grouped`.
  - Parametrize `test_armbf16_sft_route_edge_cases_match_reference` over the same values.
  - Add a metadata assertion that grouped dispatch keeps sparse scratch below the old dense estimate.

Risks to watch:

- `grad_x` and `route_grad_x` alias in the scalar wrapper above. This preserves current behavior only if the row is zeroed before each route call. If this is too fragile during implementation, allocate a separate `[H]` route-local buffer.
- Scratch memory estimate must include `BackwardTileScratch` in addition to existing `ForwardBuffers` recompute bytes and global `route_grad_x_scratch_bytes`.
- `route_grad_x_accum` is indexed by global packed route. Tile code must write `tile.begin + local_route`.
- Dropout seed inputs must continue using logical token and route slot from `tile_routes`.

Validation before Stage 3:

```bash
cd "$KT"
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 "$PY" -m pip install -e . -v --no-build-isolation

cd "$ASYM"
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py -q
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

Synthetic correctness/perf smoke:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage2_synth_q128_r8_grouped_dispatch
mkdir -p "$ART"
taskset -c 0-143 env \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped \
  KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend both --qlen 128 --topk 8 --rank 8 \
  --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 3 --arm-profile \
  --artifact-dir "$ART" --output-json "$ART/result.json" \
  2>&1 | tee "$ART/console.log"
```

LF smoke is required because the refactor touches backward control flow:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage2_qwen3_s64_b1_r8_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=1 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 --expected-batch 1 --expected-rank 8 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

## Stage 3: Implement Grouped Base Backward Kernels

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - class `ARM_BF16_SFT_MOE`
  - `struct BackwardTileScratch`
  - `struct ArmSFTProfileStats`
  - `append_profile_stats_json()`
  - `prepare_backward_tile_common()`
  - new SVE grouped base helpers
  - `backward_tile_accumulate_grouped()`
  - `backward_impl_packed()` scratch estimate
- `../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py`

Target code changes:

- Add profile fields before changing math:

```cpp
struct ArmSFTProfileStats {
  // Keep existing fields above this block unchanged.
  double backward_prepare_common_ms = 0.0;
  double backward_base_grad_ms = 0.0;
  double backward_lora_grad_ms = 0.0;
  double backward_activation_grad_ms = 0.0;
  double backward_grad_weight_ms = 0.0;
};
```

Add these fields to `append_profile_stats_json()` and to the human-readable `KT_ARM_SFT_PROFILE` line printed near the existing split counters.

- Add `prepare_backward_tile_common()`. It should do only route metadata, BF16 input conversion, `dy`, and `grad_weights`; no base or LoRA projection math.

```cpp
void prepare_backward_tile_common(const CacheEntry& cache,
                                  const BackwardExpertTile& tile,
                                  const PackedRoutes& tile_routes,
                                  const ForwardBuffers& fwd,
                                  const ggml_bf16_t* grad_output,
                                  float* grad_weights,
                                  BackwardTileScratch& scratch) const {
  for (int local = 0; local < tile.routes; ++local) {
    int token = tile_routes.packed_token_ids[local];
    int route = tile_routes.packed_route_slots[local];
    const ggml_bf16_t* x_bf16 = fwd.packed_input.data() + local * H;
    const float* down = fwd.down.data() + local * H;
    float* x = scratch.x_f32.data() + local * H;
    float* dy = scratch.dy.data() + local * H;
    float route_weight = cache.weights[token * cache.k + route];
    float gw = 0.0f;
    for (int h = 0; h < H; ++h) {
      float gy = bf16_to_f32(grad_output[token * H + h]);
      x[h] = bf16_to_f32(x_bf16[h]);
      gw += gy * down[h];
      dy[h] = gy * route_weight;
    }
    grad_weights[token * cache.k + route] = gw;
  }
}
```

- Add a tile-view builder so scalar fallback helpers and grouped LoRA helpers share the same metadata.

```cpp
BackwardRouteView make_backward_tile_route_view(const CacheEntry& cache,
                                                const BackwardExpertTile& tile,
                                                const PackedRoutes& tile_routes,
                                                const ForwardBuffers& fwd,
                                                const BackwardTileScratch& scratch,
                                                int m) const {
  BackwardRouteView v;
  v.expert = tile.expert;
  v.sparse_expert = tile.sparse_expert;
  v.token = tile_routes.packed_token_ids[static_cast<size_t>(m)];
  v.logical_token = cache.token_offset + v.token;
  v.route = tile_routes.packed_route_slots[static_cast<size_t>(m)];
  v.route_weight = cache.weights[static_cast<size_t>(v.token) * cache.k + v.route];
  v.x = scratch.x_f32.data() + static_cast<size_t>(m) * H;
  v.gate = fwd.gate.data() + static_cast<size_t>(m) * I;
  v.up = fwd.up.data() + static_cast<size_t>(m) * I;
  v.act = fwd.act.data() + static_cast<size_t>(m) * I;
  v.down = fwd.down.data() + static_cast<size_t>(m) * H;
  v.gate_u = fwd.gate_u.data() + static_cast<size_t>(m) * PR;
  v.up_u = fwd.up_u.data() + static_cast<size_t>(m) * PR;
  v.down_u = fwd.down_u.data() + static_cast<size_t>(m) * PR;
  return v;
}
```

- Implement grouped base math for all `M` routes in a tile:
  - `dy[M,H] = grad_output[token,H] * route_weight`
  - `grad_act[M,I] += dy[M,H] @ down_proj[H,I]`
  - activation split:
    - `grad_gate[M,I] = grad_act[M,I] * up[M,I] * silu_grad(gate[M,I])`
    - `grad_up[M,I] = grad_act[M,I] * silu(gate[M,I])`
  - `grad_x[M,H] += grad_gate[M,I] @ gate_proj[I,H] + grad_up[M,I] @ up_proj[I,H]`

- Add tile kernels that reuse weight rows across multiple route rows. Do not use the current v4 `arm_fp32_bf16_matmul_kn_sve()` loop order as the main path because it only handles one row efficiently and was slower in profiling.

New helpers:

```cpp
void arm_tile_dy_down_to_grad_act_sve(
    int M, int H, int I,
    const float* dy, int dy_ld,
    const ggml_bf16_t* down_hi, int down_ld_i,
    float* grad_act, int grad_act_ld,
    bool accumulate) const;

void arm_tile_grad_gate_up_to_grad_x_sve(
    int M, int I, int H,
    const float* grad_gate, int grad_gate_ld,
    const ggml_bf16_t* gate_ih, int gate_ld_h,
    const float* grad_up, int grad_up_ld,
    const ggml_bf16_t* up_ih, int up_ld_h,
    float* grad_x, int grad_x_ld,
    bool accumulate) const;
```

Pseudocode for `dy @ down`:

```cpp
void arm_tile_dy_down_to_grad_act_sve(int M, int H, int I,
                                      const float* dy, int dy_ld,
                                      const ggml_bf16_t* down_hi, int down_ld_i,
                                      float* grad_act, int grad_act_ld,
                                      bool accumulate) const {
  lanes = svcntw()
  for i0 in range(0, I, lanes):
    pg_i32 = svwhilelt_b32(i0, I)
    pg_i16 = svwhilelt_b16(i0, I)
    for m0 in range(0, M, 4):
      acc0 = accumulate ? load grad_act[m0+0, i0:i0+lanes] : 0
      acc1 = (m0+1<M && accumulate) ? load grad_act[m0+1, i0:i0+lanes] : 0
      acc2 = (m0+2<M && accumulate) ? load grad_act[m0+2, i0:i0+lanes] : 0
      acc3 = (m0+3<M && accumulate) ? load grad_act[m0+3, i0:i0+lanes] : 0
      for h in range(0, H):
        w = load_bf16_low_as_f32(pg_i16, down_hi + h * down_ld_i + i0)
        acc0 += w * dy[(m0+0)*dy_ld + h]
        if m0+1 < M: acc1 += w * dy[(m0+1)*dy_ld + h]
        if m0+2 < M: acc2 += w * dy[(m0+2)*dy_ld + h]
        if m0+3 < M: acc3 += w * dy[(m0+3)*dy_ld + h]
      store acc0 to grad_act[(m0+0)*grad_act_ld + i0]
      if m0+1 < M: store acc1 to grad_act[(m0+1)*grad_act_ld + i0]
      if m0+2 < M: store acc2 to grad_act[(m0+2)*grad_act_ld + i0]
      if m0+3 < M: store acc3 to grad_act[(m0+3)*grad_act_ld + i0]
}
```

Pseudocode for `grad_gate/up @ gate/up`:

```cpp
void arm_tile_grad_gate_up_to_grad_x_sve(int M, int I, int H,
                                         const float* grad_gate, int grad_gate_ld,
                                         const ggml_bf16_t* gate_ih, int gate_ld_h,
                                         const float* grad_up, int grad_up_ld,
                                         const ggml_bf16_t* up_ih, int up_ld_h,
                                         float* grad_x, int grad_x_ld,
                                         bool accumulate) const {
  lanes = svcntw()
  for h0 in range(0, H, lanes):
    pg_h32 = svwhilelt_b32(h0, H)
    pg_h16 = svwhilelt_b16(h0, H)
    for m0 in range(0, M, 4):
      acc0 = accumulate ? load grad_x[m0+0, h0:h0+lanes] : 0
      acc1 = (m0+1<M && accumulate) ? load grad_x[m0+1, h0:h0+lanes] : 0
      acc2 = (m0+2<M && accumulate) ? load grad_x[m0+2, h0:h0+lanes] : 0
      acc3 = (m0+3<M && accumulate) ? load grad_x[m0+3, h0:h0+lanes] : 0
      for i in range(0, I):
        gw = load_bf16_low_as_f32(pg_h16, gate_ih + i * gate_ld_h + h0)
        uw = load_bf16_low_as_f32(pg_h16, up_ih + i * up_ld_h + h0)
        gg0 = grad_gate[(m0+0)*grad_gate_ld + i]
        ug0 = grad_up[(m0+0)*grad_up_ld + i]
        acc0 += gw * gg0 + uw * ug0
        if m0+1 < M:
          acc1 += gw * grad_gate[(m0+1)*grad_gate_ld + i] + uw * grad_up[(m0+1)*grad_up_ld + i]
        if m0+2 < M:
          acc2 += gw * grad_gate[(m0+2)*grad_gate_ld + i] + uw * grad_up[(m0+2)*grad_up_ld + i]
        if m0+3 < M:
          acc3 += gw * grad_gate[(m0+3)*grad_gate_ld + i] + uw * grad_up[(m0+3)*grad_up_ld + i]
      store acc0 to grad_x[(m0+0)*grad_x_ld + h0]
      if m0+1 < M: store acc1 to grad_x[(m0+1)*grad_x_ld + h0]
      if m0+2 < M: store acc2 to grad_x[(m0+2)*grad_x_ld + h0]
      if m0+3 < M: store acc3 to grad_x[(m0+3)*grad_x_ld + h0]
}
```

- Add grouped activation split. This is scalar over `I`, but route-tile contiguous and cheap relative to GEMM.

```cpp
void compute_activation_grads_grouped(int M, int I,
                                      const float* gate, int gate_ld,
                                      const float* up, int up_ld,
                                      const float* grad_act, int grad_act_ld,
                                      float* grad_gate, int grad_gate_ld,
                                      float* grad_up, int grad_up_ld) const {
  for (int m = 0; m < M; ++m) {
    for (int i = 0; i < I; ++i) {
      float g = gate[m * gate_ld + i];
      float u = up[m * up_ld + i];
      float ga = grad_act[m * grad_act_ld + i];
      grad_gate[m * grad_gate_ld + i] = ga * u * silu_grad(g);
      grad_up[m * grad_up_ld + i] = ga * silu(g);
    }
  }
}
```

- In `backward_tile_accumulate_grouped()`:

```cpp
void backward_tile_accumulate_grouped(const CacheEntry& cache,
                                      const BackwardExpertTile& tile,
                                      const PackedRoutes& tile_routes,
                                      const ForwardBuffers& fwd,
                                      const ggml_bf16_t* grad_output,
                                      float* grad_weights,
                                      float* route_grad_x_accum,
                                      BackwardTileScratch& scratch,
                                      BackwardBuffers& local) const {
  const int M = tile.routes;
  prepare_backward_tile_common(cache, tile, tile_routes, fwd, grad_output, grad_weights, scratch);
  fill scratch.grad_act[0:M*I] with 0
  fill scratch.grad_gate[0:M*I] with 0
  fill scratch.grad_up[0:M*I] with 0
  fill scratch.grad_x[0:M*H] with 0

  // Base down: dy[M,H] x down_proj[H,I] -> grad_act[M,I].
  arm_tile_dy_down_to_grad_act_sve(
      M, H, I,
      scratch.dy.data(), H,
      down_proj_bf16_.data() + down_idx(tile.expert, 0, 0), I,
      scratch.grad_act.data(), I,
      false);

  // Until Stage 4, use the scalar LoRA-only helper from Stage 2 to add LoRA contribution to grad_act.
  if (!use_grouped_lora_backward(cache.dropout_enabled)) {
    for (int m = 0; m < M; ++m) {
      BackwardRouteView v = make_backward_tile_route_view(cache, tile, tile_routes, fwd, scratch, m);
      backward_route_down_lora_scalar(v,
                                      scratch.dy.data() + static_cast<size_t>(m) * H,
                                      scratch.grad_act.data() + static_cast<size_t>(m) * I,
                                      scratch.grad_down_u.data() + static_cast<size_t>(m) * R,
                                      local,
                                      cache.dropout_enabled,
                                      cache.dropout_seed);
    }
  }

  compute_activation_grads_grouped(
      M, I,
      fwd.gate.data(), I,
      fwd.up.data(), I,
      scratch.grad_act.data(), I,
      scratch.grad_gate.data(), I,
      scratch.grad_up.data(), I);

  // Base gate/up: grad_gate/up[M,I] x gate/up_proj[I,H] -> grad_x[M,H].
  arm_tile_grad_gate_up_to_grad_x_sve(
      M, I, H,
      scratch.grad_gate.data(), I,
      gate_proj_bf16_.data() + gate_up_idx(tile.expert, 0, 0), H,
      scratch.grad_up.data(), I,
      up_proj_bf16_.data() + gate_up_idx(tile.expert, 0, 0), H,
      scratch.grad_x.data(), H,
      false);

  // Until Stage 4, use scalar LoRA-only helper to add gate/up LoRA contribution to grad_x.
  if (!use_grouped_lora_backward(cache.dropout_enabled)) {
    for (int m = 0; m < M; ++m) {
      BackwardRouteView v = make_backward_tile_route_view(cache, tile, tile_routes, fwd, scratch, m);
      backward_route_gate_up_lora_scalar(v,
                                         scratch.grad_gate.data() + static_cast<size_t>(m) * I,
                                         scratch.grad_up.data() + static_cast<size_t>(m) * I,
                                         scratch.grad_x.data() + static_cast<size_t>(m) * H,
                                         scratch.grad_gate_u.data() + static_cast<size_t>(m) * R,
                                         scratch.grad_up_u.data() + static_cast<size_t>(m) * R,
                                         local,
                                         cache.dropout_enabled,
                                         cache.dropout_seed);
    }
  }

  for (int m = 0; m < M; ++m) {
    int packed = tile.begin + m;
    std::copy(scratch.grad_x.data() + static_cast<size_t>(m) * H,
              scratch.grad_x.data() + static_cast<size_t>(m + 1) * H,
              route_grad_x_accum + static_cast<size_t>(packed) * H);
  }
}
```

- Add selector for grouped LoRA but keep it off until Stage 4:

```cpp
bool use_grouped_lora_backward(bool dropout_enabled) const {
  const char* backend = std::getenv("KT_ARM_SFT_BACKWARD_LORA_BACKEND");
  if (dropout_enabled) {
    return backend != nullptr && std::strcmp(backend, "grouped_dropout") == 0;
  }
  return backend != nullptr && std::strcmp(backend, "grouped") == 0;
}
```

- Time the grouped base and scalar LoRA parts separately using local per-thread accumulators, then merge into `profile_stats_` after the OpenMP loop. Do not update shared profile fields inside the parallel region.

Risks to watch:

- This SVE kernel uses FP32 accumulators with BF16 weights converted to FP32. It will not use BFDOT because the left side is FP32. The win must come from route-tile reuse and vectorized contiguous RHS loads.
- `down_proj_bf16_` is indexed by `down_idx(expert, h, i)` and is contiguous over `i`; that matches `arm_tile_dy_down_to_grad_act_sve()`.
- `gate_proj_bf16_` and `up_proj_bf16_` are indexed by `gate_up_idx(expert, i, h)` and contiguous over `h`; that matches `arm_tile_grad_gate_up_to_grad_x_sve()`.
- The existing `load_bf16_low_as_f32()` helper should be reused for FP32 x BF16 SVE helpers. It loads BF16 and widens to FP32; increment the logical index by `svcntw()` lanes, not by BF16 lane count.
- Long `I` loops can thrash caches. If perf counters show high LLC misses, add `I_TILE=128/256` and `H_TILE=256` env-tunable blocking.

Validation before Stage 4:

```bash
cd "$KT"
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 "$PY" -m pip install -e . -v --no-build-isolation

cd "$ASYM"
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py -q
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

Synthetic profile:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage3_synth_q2048_r64_grouped_base
mkdir -p "$ART"
taskset -c 0-143 env \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped \
  KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness --qlen 2048 --topk 8 --rank 64 \
  --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 2 --arm-profile \
  --artifact-dir "$ART" --output-json "$ART/result.json" \
  2>&1 | tee "$ART/console.log"
```

Perf counter spot check:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage3_perf_q2048_r64
mkdir -p "$ART"
taskset -c 0-143 perf stat -d -d -o "$ART/perf_stat.txt" -- \
  env KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness --qlen 2048 --topk 8 --rank 64 \
  --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 1 --arm-profile \
  --artifact-dir "$ART" --output-json "$ART/result.json"
```

LF long profile is required before moving on:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage3_qwen3_s7168_b4_r64_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped \
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=4 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 7168 --expected-batch 4 --expected-rank 64 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --allow-unvalidated-route-rank 1 --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

Accept Stage 3 only if:

- correctness tests pass;
- LF source profile is final and validator passes;
- `backward_route_grad_accum_ms` decreases materially versus v4 split profile;
- if it regresses, keep grouped base behind `KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped` and continue Stage 4 against the scalar default only if needed.

## Stage 4: Implement Grouped LoRA Backward Kernels

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - class `ARM_BF16_SFT_MOE`
  - `struct BackwardTileScratch`
  - `backward_tile_accumulate_grouped()`
  - new LoRA tile kernels
  - existing sparse index helpers: `sparse_lora_a_h_idx()`, `sparse_lora_b_i_idx()`, `sparse_down_lora_a_idx()`, `sparse_down_lora_b_idx()`
- `../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py`
- `../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py`

Implementation:

- Implement dropout-0 grouped LoRA first. Keep dropout-enabled grouped LoRA disabled until mask semantics are proven. Use the selector introduced in Stage 3:

```cpp
bool use_grouped_lora_backward(bool dropout_enabled) const {
  const char* backend = std::getenv("KT_ARM_SFT_BACKWARD_LORA_BACKEND");
  if (dropout_enabled) {
    return backend != nullptr && std::strcmp(backend, "grouped_dropout") == 0;
  }
  return backend != nullptr && std::strcmp(backend, "grouped") == 0;
}
```

- Add helpers for rank-tile reductions. Target `R=8` and `R=64`; handle arbitrary `R` tail. The helper names below are concrete and should live near the existing ARM SVE helpers.

```cpp
void lora_bwd_grad_y_b_to_u_grouped_sve(
    int M, int K, int R,
    const float* grad_y, int grad_y_ld,          // [M,K], FP32
    const ggml_bf16_t* lora_b_kr, int b_ld_r,   // [K,R], BF16, contiguous R
    float scale,
    float* grad_u, int grad_u_ld) const;         // [M,R], FP32, output is zeroed by helper

void lora_bwd_grad_b_grouped_sve(
    int M, int K, int R,
    const float* grad_y, int grad_y_ld,          // [M,K], FP32
    const float* u, int u_ld,                    // [M,PR] for forward u or [M,R] for scratch u
    float scale,
    float* grad_b_accum, int grad_b_ld) const;   // [K,R], FP32 sparse local

void lora_bwd_grad_a_and_input_grouped_sve(
    int M, int R, int K,
    const float* grad_u, int grad_u_ld,          // [M,R], already includes LoRA scale
    const float* input_or_act, int input_ld,     // [M,K], FP32
    const ggml_bf16_t* lora_a_rk, int a_ld_k,   // [R,K], BF16
    float* grad_a_accum, int grad_a_ld,          // [R,K], FP32 sparse local
    float* grad_input_or_act, int grad_input_ld, // [M,K], FP32 accumulates
    bool accumulate_grad_input) const;
```

Down LoRA math:

```cpp
// forward: down += scale * down_u @ down_lora_b^T
// backward:
grad_down_u[M,R] = scale * dy[M,H] @ down_lora_b[H,R]     // scale is applied here
grad_down_lora_b[H,R] += scale * dy[M,H]^T @ down_u[M,R]
grad_act[M,I] += grad_down_u[M,R] @ down_lora_a[R,I]      // do not apply scale again
grad_down_lora_a[R,I] += grad_down_u[M,R]^T @ act[M,I]    // do not apply scale again
```

Gate/up LoRA math:

```cpp
// forward:
gate_u[M,R] = x[M,H] @ gate_lora_a[R,H]^T
up_u[M,R]   = x[M,H] @ up_lora_a[R,H]^T
gate[M,I] += scale * gate_u[M,R] @ gate_lora_b[I,R]^T
up[M,I]   += scale * up_u[M,R] @ up_lora_b[I,R]^T

// backward:
grad_gate_u[M,R] = scale * grad_gate[M,I] @ gate_lora_b[I,R] // scale is applied here
grad_up_u[M,R]   = scale * grad_up[M,I] @ up_lora_b[I,R]
grad_gate_lora_b[I,R] += scale * grad_gate[M,I]^T @ gate_u[M,R]
grad_up_lora_b[I,R]   += scale * grad_up[M,I]^T @ up_u[M,R]
grad_x[M,H] += grad_gate_u[M,R] @ gate_lora_a[R,H]           // do not apply scale again
grad_x[M,H] += grad_up_u[M,R] @ up_lora_a[R,H]
grad_gate_lora_a[R,H] += grad_gate_u[M,R]^T @ x[M,H]
grad_up_lora_a[R,H]   += grad_up_u[M,R]^T @ x[M,H]
```

Concrete kernel pseudocode for `grad_u = scale * grad_y @ B`:

```cpp
void lora_bwd_grad_y_b_to_u_grouped_sve(int M, int K, int R,
                                        const float* grad_y, int grad_y_ld,
                                        const ggml_bf16_t* B, int b_ld_r,
                                        float scale,
                                        float* grad_u, int grad_u_ld) {
  lanes = svcntw()
  for m in 0..M:
    zero grad_u[m, 0:R]
  for r0 in range(0, R, lanes):
    pg_r32 = svwhilelt_b32(r0, R)
    pg_r16 = svwhilelt_b16(r0, R)
    for m0 in range(0, M, 4):
      acc0 = 0; acc1 = 0; acc2 = 0; acc3 = 0
      for k in 0..K:
        b_vec = load_bf16_low_as_f32(pg_r16, B + k * b_ld_r + r0)
        acc0 += b_vec * (grad_y[(m0+0)*grad_y_ld + k] * scale)
        if m0+1<M: acc1 += b_vec * (grad_y[(m0+1)*grad_y_ld + k] * scale)
        if m0+2<M: acc2 += b_vec * (grad_y[(m0+2)*grad_y_ld + k] * scale)
        if m0+3<M: acc3 += b_vec * (grad_y[(m0+3)*grad_y_ld + k] * scale)
      svst1_f32(pg_r32, grad_u + (m0+0) * grad_u_ld + r0, acc0)
      if m0+1<M: svst1_f32(pg_r32, grad_u + (m0+1) * grad_u_ld + r0, acc1)
      if m0+2<M: svst1_f32(pg_r32, grad_u + (m0+2) * grad_u_ld + r0, acc2)
      if m0+3<M: svst1_f32(pg_r32, grad_u + (m0+3) * grad_u_ld + r0, acc3)
}
```

Concrete kernel pseudocode for `grad_B += grad_y^T @ u`:

```cpp
void lora_bwd_grad_b_grouped_sve(int M, int K, int R,
                                 const float* grad_y, int grad_y_ld,
                                 const float* u, int u_ld,
                                 float scale,
                                 float* out, int out_ld_r) {
  // out layout [K, R], contiguous R for each K.
  lanes = svcntw()
  for k in range(0, K):
    for r0 in range(0, R, lanes):
      pg_r = svwhilelt_b32(r0, R)
      acc = svld1_f32(pg_r, out + k * out_ld_r + r0)
      for m in range(0, M):
        gy = grad_y[m * grad_y_ld + k]
        uv = svld1_f32(pg_r, u + m * u_ld + r0)
        acc = svmla_n_f32_m(pg_r, acc, uv, gy * scale)
      svst1_f32(pg_r, out + k * out_ld_r + r0, acc)
}
```

Concrete kernel pseudocode for `grad_A += grad_u^T @ x` and `grad_x += grad_u @ A`:

```cpp
void lora_bwd_grad_a_and_input_grouped_sve(int M, int R, int K,
                                           const float* grad_u, int grad_u_ld,
                                           const float* x, int x_ld,
                                           const ggml_bf16_t* A, int a_ld_k,
                                           float* out_grad_a, int grad_a_ld,
                                           float* out_grad_x, int grad_x_ld,
                                           bool accumulate_grad_input) {
  // grad_A layout [R, K]; A layout [R, K].
  lanes = svcntw()
  for r in range(0, R):
    for k0 in range(0, K, lanes):
      pg_k32 = svwhilelt_b32(k0, K)
      pg_k16 = svwhilelt_b16(k0, K)
      acc_a = svld1_f32(pg_k32, out_grad_a + r * grad_a_ld + k0)
      a_vec = load_bf16_low_as_f32(pg_k16, A + r * a_ld_k + k0)
      for m in range(0, M):
        gu = grad_u[m * grad_u_ld + r]
        x_vec = svld1_f32(pg_k32, x + m * x_ld + k0)  // x_f32 or act
        acc_a += x_vec * gu
        if accumulate_grad_input:
          gx = svld1_f32(pg_k32, out_grad_x + m * grad_x_ld + k0)
          gx = svmla_n_f32_m(pg_k32, gx, a_vec, gu)
          svst1_f32(pg_k32, out_grad_x + m * grad_x_ld + k0, gx)
      svst1_f32(pg_k32, out_grad_a + r * grad_a_ld + k0, acc_a)
}
```

- Wire these helpers inside `backward_tile_accumulate_grouped()` only when `use_grouped_lora_backward(cache.dropout_enabled)` is true. The exact pointer bases are:

```cpp
// Down LoRA.
float* down_u_grad = scratch.grad_down_u.data();                 // [M,R]
const float* down_u_forward = fwd.down_u.data();                 // [M,PR]
float* sparse_down_b = local.grad_down_lora_b_accum.data()
    + sparse_down_lora_b_idx(tile.sparse_expert, 0, 0);          // [H,R]
float* sparse_down_a = local.grad_down_lora_a_accum.data()
    + sparse_down_lora_a_idx(tile.sparse_expert, 0, 0);          // [R,I]
lora_bwd_grad_y_b_to_u_grouped_sve(M, H, R, scratch.dy, H,
    down_lora_b_.data() + down_lora_b_idx(tile.expert, 0, 0), R,
    scale, down_u_grad, R);
lora_bwd_grad_b_grouped_sve(M, H, R, scratch.dy, H, down_u_forward, PR,
    scale, sparse_down_b, R);
lora_bwd_grad_a_and_input_grouped_sve(M, R, I, down_u_grad, R,
    fwd.act.data(), I,
    down_lora_a_.data() + down_lora_a_idx(tile.expert, 0, 0), I,
    sparse_down_a, I,
    scratch.grad_act.data(), I,
    true);

// Gate LoRA.
float* gate_u_grad = scratch.grad_gate_u.data();                 // [M,R]
const float* gate_u_forward = fwd.gate_u.data();                 // [M,PR]
float* sparse_gate_b = local.grad_gate_lora_b_accum.data()
    + sparse_lora_b_i_idx(tile.sparse_expert, 0, 0);             // [I,R]
float* sparse_gate_a = local.grad_gate_lora_a_accum.data()
    + sparse_lora_a_h_idx(tile.sparse_expert, 0, 0);             // [R,H]
lora_bwd_grad_y_b_to_u_grouped_sve(M, I, R, scratch.grad_gate, I,
    gate_lora_b_.data() + lora_b_i_idx(tile.expert, 0, 0), R,
    scale, gate_u_grad, R);
lora_bwd_grad_b_grouped_sve(M, I, R, scratch.grad_gate, I, gate_u_forward, PR,
    scale, sparse_gate_b, R);
lora_bwd_grad_a_and_input_grouped_sve(M, R, H, gate_u_grad, R,
    scratch.x_f32.data(), H,
    gate_lora_a_.data() + lora_a_h_idx(tile.expert, 0, 0), H,
    sparse_gate_a, H,
    scratch.grad_x.data(), H,
    true);

// Up LoRA is identical to gate, using up_lora_b_, up_lora_a_, fwd.up_u,
// scratch.grad_up, scratch.grad_up_u, and sparse up accum buffers.
```

- Use `PR` for forward `gate_u/up_u/down_u` strides and `R` for scratch gradient-u strides. This is not optional; the forward buffers are padded rank, but sparse gradient tensors are dense rank.
- Every grouped LoRA helper must honor the explicit leading-dimension arguments above. Do not use shorthand like `m * R`, `m * K`, or `r * K` in the implementation unless the passed leading dimension is exactly that value at the call site. This is the main place where a correct-looking grouped kernel can silently corrupt rank-64 runs.

- Avoid atomics. Each OpenMP worker writes into its own `BackwardBuffers local`. Reduction happens after route tiles, as in v4. Do not write directly to final dense output from multiple route tiles.
- Optional improvement after correctness: split copy-type and reduce-type gradients like `moe-sft-tp.hpp`, but only if reduction appears in profiles after grouped LoRA kernels.

Dropout handling:

- Stage 4A supports dropout 0.0 only and falls back to scalar route accumulation when `cache.dropout_enabled`.
- Stage 4B adds masked grouped dropout only after Stage 4A passes LF e2e:

```cpp
if (dropout_enabled && grouped_dropout_enabled) {
  use lora_dropout_scale(dropout_seed, config_.layer_idx, KT_LORA_DROPOUT_GATE/UP/DOWN,
                         expert, logical_token, route, feature, config_.lora_dropout)
  inside the M,K loops before accumulating grad_A or grad_input/grad_act
} else if (dropout_enabled) {
  call scalar LoRA-only helpers for this whole tile
}
```

Risks to watch:

- LoRA A/B gradients are FP32 partials but flushed to BF16. Compare tolerances against current scalar path, not full FP32 PyTorch.
- Rank 64 increases route-rank work sharply. If the grouped kernel is memory-bound, use `K_TILE=128/256` and `M_TILE=16/32/64` sweeps rather than increasing route tile blindly.
- Nonzero dropout requires exact deterministic mask indexing by `(layer, kind, expert, logical_token, route, feature)`. Do not enable grouped dropout until tests cover duplicate routes and token offsets.
- Scale must be applied exactly once. `grad_u` helpers apply `scale`; `grad_A` and LoRA grad-input helpers consume already-scaled `grad_u` and must not multiply by `scale` again.
- Sparse pointer bases are only contiguous for the target expert slice. Never write through dense expert ids inside the grouped tile; dense flush remains the job of `flush_sparse_lora_grad_accum()`.

Validation before Stage 5:

```bash
cd "$KT"
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 "$PY" -m pip install -e . -v --no-build-isolation

cd "$ASYM"
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py -q
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

Synthetic grouped LoRA:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage4_synth_q2048_r64_grouped_lora
mkdir -p "$ART"
taskset -c 0-143 env \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped \
  KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
  KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
  "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness --qlen 2048 --topk 8 --rank 64 \
  --hidden 2048 --intermediate 768 --experts 128 \
  --threads 8 --warmup 1 --iters 2 --arm-profile \
  --artifact-dir "$ART" --output-json "$ART/result.json" \
  2>&1 | tee "$ART/console.log"
```

LF smoke and long source profiles:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage4_qwen3_s64_b1_r8_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=1 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 --expected-batch 1 --expected-rank 8 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms

ART=profiling_kt_codex_smoke/v5_stage4_qwen3_s7168_b4_r64_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=4 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 7168 --expected-batch 4 --expected-rank 64 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --allow-unvalidated-route-rank 1 --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

Accept Stage 4 only if LF e2e completes and `backward_route_grad_accum_ms` is no longer the dominant native timing by a large margin. If it remains dominant, use `perf stat` plus the split fields to decide whether `grad_B`, `grad_A`, or LoRA grad-input is the next specific kernel to retile.

## Stage 5: Optimize Route Merge And Grad-Input Scatter

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - include list: add `#include <array>` if using the fixed-lane temporary helper below
  - class `ARM_BF16_SFT_MOE`
  - `merge_routes_to_output()`
  - `scatter_route_grad_x_to_tokens()`
  - new helper `store_f32_sve_to_bf16_tail()`
  - profile stats fields for merge/scatter if missing
- `../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py`

Implementation:

- Replace scalar hidden loops with SVE vector hidden loops.
- Keep token-level parallelism; no expert loop is needed here.
- Add one local conversion helper before replacing the loops. Arm SVE vector length is runtime-selected and bounded by the architecture, so use a fixed 64-lane FP32 scratch buffer, which covers a 2048-bit SVE implementation. This keeps the first implementation simple and avoids inventing a BF16 store intrinsic path before profiles prove conversion is a bottleneck.

```cpp
static constexpr int ARM_SFT_MAX_SVE_F32_LANES = 64;

void store_f32_sve_to_bf16_tail(svbool_t pg,
                                svfloat32_t values,
                                ggml_bf16_t* dst,
                                int remaining) const {
  const int lanes = static_cast<int>(svcntw());
  if (lanes > ARM_SFT_MAX_SVE_F32_LANES) {
    throw std::runtime_error("ARMBF16_SFT unexpected SVE FP32 lane count");
  }
  alignas(64) std::array<float, ARM_SFT_MAX_SVE_F32_LANES> tmp{};
  svst1_f32(pg, tmp.data(), values);
  const int active = std::min(lanes, remaining);
  for (int j = 0; j < active; ++j) {
    dst[j] = f32_to_bf16(tmp[static_cast<size_t>(j)]);
  }
}
```

- Forward merge:

```cpp
void merge_routes_to_output_sve(int qlen, int k,
                                const PackedRoutes& routes,
                                const float* weights,
                                const ForwardBuffers& buffers,
                                ggml_bf16_t* output) {
  const int H = config_.hidden_size;
  const int lanes = static_cast<int>(svcntw());
  direct_or_parallel(qlen, qlen, [&](int token) {
    for (int h0 = 0; h0 < H; h0 += lanes) {
      svbool_t pg = svwhilelt_b32(h0, H);
      svfloat32_t acc = svdup_f32(0.0f);
      for (int route = 0; route < k; ++route) {
        int packed = routes.flat_route_to_packed[static_cast<size_t>(token) * k + route];
        if (packed >= 0) {
          float w = weights[static_cast<size_t>(token) * k + route];
          svfloat32_t down = svld1_f32(pg, buffers.down.data() + static_cast<size_t>(packed) * H + h0);
          acc = svmla_n_f32_m(pg, acc, down, w);
        }
      }
      store_f32_sve_to_bf16_tail(pg, acc, output + static_cast<size_t>(token) * H + h0, H - h0)
    }
  });
}
```

- Backward scatter:

```cpp
void scatter_route_grad_x_to_tokens_sve(const PackedRoutes& routes,
                                        const std::vector<float>& route_grad_x,
                                        ggml_bf16_t* grad_input) {
  const int H = config_.hidden_size;
  const int lanes = static_cast<int>(svcntw());
  direct_or_parallel(routes.qlen, routes.qlen, [&](int token) {
    for (int h0 = 0; h0 < H; h0 += lanes) {
      svbool_t pg = svwhilelt_b32(h0, H);
      svfloat32_t acc = svdup_f32(0.0f);
      for (int route = 0; route < routes.k; ++route) {
        int packed = routes.flat_route_to_packed[static_cast<size_t>(token) * routes.k + route];
        if (packed >= 0) {
          acc = svadd_f32_m(pg, acc,
              svld1_f32(pg, route_grad_x.data() + static_cast<size_t>(packed) * H + h0));
        }
      }
      store_f32_sve_to_bf16_tail(pg, acc, grad_input + static_cast<size_t>(token) * H + h0, H - h0)
    }
  });
}
```

- Only optimize FP32->BF16 conversion after merge/scatter wall time remains visible. The first pass should not use non-obvious BF16 narrowing intrinsics unless they are covered by unit tests and the compiler emits the expected instructions.

Risks to watch:

- Duplicate routes for the same token must sum exactly once per route.
- Invalid expert routes are represented by `packed < 0`; preserve skip behavior.
- `store_f32_sve_to_bf16_tail()` relies on the architectural maximum SVE vector length. Keep the runtime lane-count guard so this fails loudly if built for an unexpected target.
- If route merge is not significant after Stage 4, skip implementation and leave it as low priority.

Validation before Stage 6:

```bash
cd "$KT"
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 "$PY" -m pip install -e . -v --no-build-isolation

cd "$ASYM"
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py -q
"$PY" -m pytest ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

LF long source profile:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage5_qwen3_s7168_b4_r64_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=4 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 7168 --expected-batch 4 --expected-rank 64 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --allow-unvalidated-route-rank 1 --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

Accept Stage 5 only if merge/scatter counters decrease without raising total e2e time. If the long LF profile is noisy, run the profile twice on GPU 1 or once on GPU 1 and once on GPU 2 and compare source timing medians.

## Stage 6: Final LF Acceptance Sweep And Regression Lock

Scope:

- `agent/kt/scripts/profile_lora_lf_kt.sh`
- `agent/kt/scripts/validate_kt_arm_profile.py`
- `scripts/kt/README.md`
- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `use_grouped_backward_grad_backend()`
  - `use_grouped_lora_backward()`
- `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py` only if extra native fields need artifact capture
- no new kernels unless validation reveals a blocker

Implementation:

- Make sure `profile_lora_lf_kt.sh` emits enough artifacts for comparison:
  - env capture
  - `profile.json`
  - train log
  - native KT profile lines
  - memory summary
  - wrapper summary
- Add validator checks for:
  - GPU id in `{1,2}`
  - `NUM_GPUS=1`
  - `KT_ARM_OMP_PROC_BIND=false`
  - `ARMBF16_SFT` forward and backward calls present
  - native ARM evidence present: packed path, worker pool dispatch, SVE BF16 compiled, aligned weights
  - split fields present: `backward_tile_recompute_ms`, `backward_route_grad_accum_ms`
- Document accepted commands in `scripts/kt/README.md`.
- Only after the small and long LF source profiles pass with explicit grouped envs, promote dropout-0 grouped kernels to defaults:

```cpp
bool use_grouped_backward_grad_backend() const {
  const char* backend = std::getenv("KT_ARM_SFT_BACKWARD_GRAD_BACKEND");
  if (backend != nullptr && std::strcmp(backend, "scalar") == 0) {
    return false;
  }
  return true;  // default after Stage 6 acceptance
}

bool use_grouped_lora_backward(bool dropout_enabled) const {
  const char* backend = std::getenv("KT_ARM_SFT_BACKWARD_LORA_BACKEND");
  if (dropout_enabled) {
    return backend != nullptr && std::strcmp(backend, "grouped_dropout") == 0;
  }
  if (backend != nullptr && std::strcmp(backend, "scalar") == 0) {
    return false;
  }
  return true;  // dropout-0 grouped LoRA default after Stage 6 acceptance
}
```

- Keep scalar escape hatches documented:
  - `KT_ARM_SFT_BACKWARD_GRAD_BACKEND=scalar`
  - `KT_ARM_SFT_BACKWARD_LORA_BACKEND=scalar`
  - `KT_ARM_SFT_BACKWARD_BASE_BACKEND=sve` remains opt-in only and should not be promoted based on v4 data.

Risks to watch:

- Nsight Systems does not profile CPU SVE kernels directly. Use `PROFILE_PROFILER=source` as the acceptance source; use `nsys` only to verify GPU-side behavior after source profiles are good.
- Long LF profiles can fail from route-rank and scratch safety guards. For this validation, explicit `KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1` and `KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1` are acceptable because the purpose is kernel profiling, not production guard validation.

Final validation small LF source:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage6_final_qwen3_s64_b1_r8_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=1 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 --expected-batch 1 --expected-rank 8 \
  --expected-top-k 8 --expected-cache-depth 2 \
  --expected-dropout 0.0 --expected-recompute false \
  --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

Final validation long LF source:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage6_final_qwen3_s7168_b4_r64_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=4 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 7168 --expected-batch 4 --expected-rank 64 \
  --expected-top-k 8 --expected-cache-depth 2 \
  --expected-dropout 0.0 --expected-recompute false \
  --allow-unvalidated-route-rank 1 --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

Optional GPU sanity after source acceptance:

```bash
cd "$ASYM"
GPU_POOL=1,2 \
BACKEND_SPECS='kt_armbf16|recomp' \
PROFILERS=nsys \
SEQ_LENS=7168 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
LORA_RANK=64 \
LORA_DROPOUT=0.00 \
MAX_STEPS=1 \
WARMUP_STEPS=0 \
MAX_SAMPLES=4 \
KT_NUM_THREADS=8 \
KT_ARM_OMP_NUM_THREADS=8 \
KT_ARM_OMP_PROC_BIND=false \
KT_ARM_SFT_BACKWARD_THREADS=8 \
KT_ARM_SFT_PROFILE=1 \
KT_ARM_SFT_POOL_LOG=1 \
KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped \
KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 \
KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
KT_ARM_ALLOW_NSYS_WITHOUT_SOURCE_OK=0 \
OUTPUT_ROOT=profiling_kt_codex_smoke/v5_stage6_final_nsys \
scripts/kt/profile_lora_lf_kt.sh
```

Default-path validation after selector promotion:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage6_default_qwen3_s64_b1_r8_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=1 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 --expected-batch 1 --expected-rank 8 \
  --expected-top-k 8 --expected-cache-depth 2 \
  --expected-dropout 0.0 --expected-recompute false \
  --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

Default-path long validation after selector promotion:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage6_default_qwen3_s7168_b4_r64_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=4 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 7168 --expected-batch 4 --expected-rank 64 \
  --expected-top-k 8 --expected-cache-depth 2 \
  --expected-dropout 0.0 --expected-recompute false \
  --allow-unvalidated-route-rank 1 --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

Acceptance criteria:

- The unit gates pass.
- The LF KT source smoke passes on GPU 1 or GPU 2, never GPU 0 or GPU 3.
- The long LF KT source profile is final and validates.
- Native log evidence shows `ARMBF16_SFT`, packed path, worker pool dispatch, SVE BF16 compiled, and aligned weights.
- `backward_route_grad_accum_ms` is materially lower than the v4 split artifact and no longer explains most of the route loop wall time.
- HBM memory remains low relative to DeepSpeed ZeRO-3 offload because routed expert compute is on CPU; low HBM alone is not suspicious.

## Stage 7: Autotune Only After Kernel Wins Are Proven

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `env_int_clamped()` or equivalent helper
  - `backward_grad_m_tile()`
  - `backward_grad_k_tile()`
  - `backward_lora_rank_tile()`
  - grouped base/LoRA helper loops that use these tunables
- `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
  - optional sweep args/artifact summary
- `agent/kt/scripts/profile_lora_lf_kt.sh`
  - optional env passthrough documentation

Implementation:

- Add env tunables only for tile sizes that showed sensitivity in Stage 3/4:

```cpp
int env_int_clamped(const char* name, int fallback, int min_value, int max_value) const {
  long value = env_long_or(name, fallback);
  if (value < min_value) {
    value = min_value;
  }
  if (value > max_value) {
    value = max_value;
  }
  return static_cast<int>(value);
}

int backward_grad_m_tile() const {
  return env_int_clamped("KT_ARM_SFT_BACKWARD_GRAD_M_TILE", 16, 4, 256);
}
int backward_grad_k_tile() const {
  return env_int_clamped("KT_ARM_SFT_BACKWARD_GRAD_K_TILE", 256, 64, 2048);
}
int backward_lora_rank_tile() const {
  return env_int_clamped("KT_ARM_SFT_BACKWARD_LORA_R_TILE", 8, 4, 64);
}
```

- Use the tunables only in the loop blocking of grouped helpers. Do not change tensor layouts or sparse gradient indexing in this stage.

```cpp
const int m_tile = std::min(backward_grad_m_tile(), M);
for (int m_begin = 0; m_begin < M; m_begin += m_tile) {
  int mb = std::min(m_tile, M - m_begin);
  run the existing grouped helper on the [m_begin, m_begin + mb) route slice
}

const int k_tile = backward_grad_k_tile();
for (int k_begin = 0; k_begin < K; k_begin += k_tile) {
  int kb = std::min(k_tile, K - k_begin);
  run the same accumulation math over this K block
}
```

- Keep defaults conservative and tested.
- Do not tune NUMA/affinity before grouped kernels are validated. The v4 issue was not just affinity; the dominant remaining issue is math shape.

Risks to watch:

- Tuning can improve synthetic all-to-one routing while hurting real LF routing. Promote only LF-proven defaults.
- Larger `M_TILE` can increase scratch reuse but hurt cache locality for LoRA A/B updates. If LLC misses rise in `perf stat`, prefer smaller `M_TILE`.
- `K_TILE` must not change numerical accumulation order enough to fail existing BF16 tolerances. If it does, keep the previous default and leave the tunable opt-in.

Validation:

```bash
cd "$ASYM"
for M_TILE in 8 16 32 64; do
  for K_TILE in 128 256 512; do
    ART="profiling_kt_codex_smoke/v5_stage7_tune_m${M_TILE}_k${K_TILE}"
    mkdir -p "$ART"
    taskset -c 0-143 env \
      KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped \
      KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
      KT_ARM_SFT_BACKWARD_GRAD_M_TILE="$M_TILE" \
      KT_ARM_SFT_BACKWARD_GRAD_K_TILE="$K_TILE" \
      KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
      KT_ARM_SFT_BACKWARD_THREADS=8 OMP_NUM_THREADS=8 OMP_PROC_BIND=false \
      "$PY" ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
      --backend arm --skip-correctness --qlen 2048 --topk 8 --rank 64 \
      --hidden 2048 --intermediate 768 --experts 128 \
      --threads 8 --warmup 1 --iters 2 --arm-profile \
      --artifact-dir "$ART" --output-json "$ART/result.json" \
      2>&1 | tee "$ART/console.log"
  done
done
```

Confirm the selected default with LF, not only the synthetic sweep:

```bash
cd "$ASYM"
ART=profiling_kt_codex_smoke/v5_stage7_selected_tiles_qwen3_s7168_b4_r64_source
mkdir -p "$ART"
taskset -c 0-143 env \
  SFT_ROOT=/workspace/AsymGEMM-SFT ROOT=/workspace/AsymGEMM-SFT/third_party/AsymGEMM \
  GPU_ID=1 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_NUM_THREADS=8 KT_ARM_OMP_NUM_THREADS=8 KT_ARM_OMP_PROC_BIND=false \
  KT_ARM_SFT_BACKWARD_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_BACKWARD_GRAD_BACKEND=grouped KT_ARM_SFT_BACKWARD_LORA_BACKEND=grouped \
  KT_ARM_ALLOW_UNVALIDATED_ROUTE_RANK_WORK=1 KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.0 \
  MAX_STEPS=1 WARMUP_STEPS=0 MAX_SAMPLES=4 OUT_DIR="$ART" \
  scripts/kt/run_lf_lora_sft_kt.sh 2>&1 | tee "$ART/console.log"

"$PY" agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "$ART/profile.json" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 7168 --expected-batch 4 --expected-rank 64 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute false --allow-unvalidated-route-rank 1 --require-final \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms
```

Only promote tuned defaults if the LF long profile improves. If synthetic improves but LF does not, leave tunables available but keep the previous default.

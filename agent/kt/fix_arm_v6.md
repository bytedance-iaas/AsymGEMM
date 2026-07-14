# KT ARM BF16 SFT Fix Plan v6

This document replaces the remaining v5 queue. It keeps only the work that is
still useful after the v5 implementation pass and the same-config partial LF
profile.

The current KT path is native ARM BF16 CPU code, not DeepSpeed. The stale scalar
backend selectors and route-serial fallback paths should not be revived. Loops
are allowed for building route/expert tile descriptors and for normal vector
dimension blocking inside a kernel. Compute must not be implemented as a serial
loop over experts or routes.

Keep this work isolated:

- Kernel/runtime edits: `../ktransformers/kt-kernel/**`
- KT-only LF launcher/profile edits: `agent/kt/scripts/**` and `scripts/kt/**`
- Do not edit `scripts/lf/run_lf_lora_sft.sh` or `scripts/lf/profile_lora_lf.sh`
- KT validation must use physical GPU 1 first, physical GPU 2 only as fallback.
  Do not use GPU 0 or GPU 3 for accepted KT results.

## Baseline To Trust

Do not treat the same-config run below as final acceptance because it was
interrupted after the first complete warmup step. It is still the best guide for
the next optimization priority.

- Artifact: `profiling_results/profiling_kt_codex_smoke/v5_sameconfig_qwen3_s4096_b4_r64_w5_s10_source`
- Shape: `Qwen/Qwen3-30B-A3B`, `seq_len=4096`, `batch=4`, `rank=64`,
  `dropout=0.00`
- Run state: physical GPU 1, `CUDA_VISIBLE_DEVICES=1`, `KT_NUM_THREADS=8`
- Trainer step 1: `1171.55 s/it`, loss `2.1505`
- Source step sample: forward `273.793 s`, backward `895.032 s`,
  optimizer/update side `2.175 s`
- HBM: peak allocated `85.604 GiB`, peak reserved about `90.920 GiB`
- Process RSS peak: `170.344 GiB`
- Native forward rows, first 48 layers:
  - `expert_schedule_wall_ms`: total `253.695 s`, average `5.285 s/layer`
  - route merge average `26.8 ms/layer`
- Native backward rows, first 48 layers:
  - `backward_grouped_tile_ms`: total `843.744 s`, average `17.578 s/layer`
  - `backward_route_scatter_ms`: average `23.3 ms/layer`
  - `backward_repack_wait_ms`: average about `0.798 s/layer`
  - `backward_base_grad_ms`: task-sum average `72.752 s/layer`
  - `backward_lora_grad_ms`: task-sum average `31.927 s/layer`
- Native labels were correct:
  - `base_kernel=sve_bfdot_blocked`
  - `down_kernel=bf16_bfdot_blocked`
  - `lora_forward_kernel=sve_bfdot_fmla`
  - `backward_base_kernel=grouped_sve_tile`
  - `backward_lora_kernel=grouped_sve_tile_dropout0`
  - `compiled_sve_bf16=1`

Priority conclusion:

1. The core issue is CPU expert math, not DeepSpeed, not GPU HBM, and not route
   merge/scatter.
2. The first optimization target is grouped base backward math
   (`backward_base_grad_ms`).
3. The second target is grouped LoRA backward math (`backward_lora_grad_ms`).
4. Forward/recompute LoRA kernels are still important because forward wall time
   is already `273.793 s` at the acceptance shape and backward recompute also
   calls the same forward helpers.
5. Scratch/RSS tuning matters only if 16/32/64 thread scaling is blocked by
   memory or reduction overhead.

Accepted comparison shape for all meaningful LF profiling from here:

- `Qwen/Qwen3-30B-A3B`
- `seq_len=4096`
- `per_device_train_batch_size=4`
- `lora_rank=64`
- `lora_dropout=0.00`
- `warmup_steps=5`
- `measure_steps=10`
- total trainer `max_steps=15`
- `PROFILE_PROFILER=source`
- `PROFILE_LEVEL=module`
- physical GPU 1 first, GPU 2 fallback

## Current Implementation Status

Completed and validated through Stage 4 on the accepted short e2e shape:

- Artifact:
  `profiling_results/profiling_kt_codex_smoke/v6_stage4_tilebalanced_qwen3_s4096_b4_r64_t64_source`
- Validation:
  `PASS KT ARM profile: gpu_id=1 affinity_count=144 wrappers=48 fw=288 bw=144`
- Shape: `Qwen/Qwen3-30B-A3B`, `seq_len=4096`, `batch=4`,
  `rank=64`, `dropout=0.00`, `warmup_steps=1`, `measure_steps=2`,
  `trainer_max_steps=3`
- Device/threading: physical GPU 1, `CUDA_VISIBLE_DEVICES=1`,
  `KT_NUM_THREADS=64`, CPU affinity `0-143`
- E2E trainer timing:
  - measured e2e step `275.536 s`
  - total e2e step `273.330 s`
  - measured forward avg `74.657 s`
  - measured backward avg `199.304 s`
- Memory:
  - peak allocated HBM `34.478 GiB`
  - peak reserved HBM `40.111 GiB`
  - process RSS peak `179.638 GiB`
- Native counters, all 144 backward rows:
  - `backward_grouped_tile_ms` avg `2444.008 ms/layer`
  - `backward_tile_recompute_ms` avg `44108.256 task-ms/layer`
  - `backward_route_grad_accum_ms` avg `82858.797 task-ms/layer`
  - `backward_base_grad_ms` avg `66050.799 task-ms/layer`
  - `backward_lora_grad_ms` avg `15915.865 task-ms/layer`
  - `backward_local_alloc_zero_ms` avg `60.352 ms/layer`
  - `backward_thread_reduce_ms` avg `0.000 ms/layer`
  - `backward_lora_grad_flush_ms` avg `26.373 ms/layer`
  - `backward_repack_wait_ms` avg `0.004 ms/layer`
  - `backward_route_scatter_ms` avg `14.398 ms/layer`
  - `sparse_backward_scratch_bytes` avg `3.043 GiB`, max `3.066 GiB`
- Native counters, last 96 backward rows:
  - `backward_grouped_tile_ms` avg `2452.849 ms/layer`
  - `backward_local_alloc_zero_ms` avg `60.647 ms/layer`
  - `backward_thread_reduce_ms` avg `0.000 ms/layer`
  - `sparse_backward_scratch_bytes` avg `3.043 GiB`
- Native labels stayed correct:
  - `base_kernel=sve_bfdot_blocked`
  - `down_kernel=bf16_bfdot_blocked`
  - `lora_forward_kernel=sve_bfdot_fmla`
  - `backward_base_kernel=grouped_sve_tile`
  - `backward_lora_kernel=grouped_sve_tile_dropout0`
  - `compiled_sve_bf16=1`
- Losses: warmup `2.2104`, measured `1.6572`, `1.6701`,
  trainer `1.8459`

Same short e2e comparison against the Stage 1/2/3 grouped artifact
`profiling_results/profiling_kt_codex_smoke/v6_stage123_grouped_qwen3_s4096_b4_r64_t64_source`:

- measured e2e step improved from `311.745 s` to `275.536 s`
  (`-36.209 s`, `-11.6%`)
- measured backward avg improved from `235.798 s` to `199.304 s`
  (`-36.494 s`, `-15.5%`)
- process RSS peak improved from `194.852 GiB` to `179.638 GiB`
- `sparse_backward_scratch_bytes` avg improved from `18.193 GiB` to
  `3.043 GiB`
- `backward_local_alloc_zero_ms` avg improved from `936.237 ms/layer` to
  `60.352 ms/layer`
- `backward_thread_reduce_ms` avg improved from `263.423 ms/layer` to
  `0.000 ms/layer`
- `backward_grouped_tile_ms` regressed from `2048.579 ms/layer` to
  `2444.008 ms/layer`; this is acceptable for now because scratch/allocation
  savings still improve e2e time materially.

Decision after Stage 4:

- Keep tile-balanced compact partials as the default. The earlier expert-owned
  attempt lowered scratch but serialized skewed hot experts and was rejected.
- The KT profile wrapper now defaults
  `KT_ARM_SFT_DEFAULT_MAX_BACKWARD_SCRATCH_BYTES=34359738368` and validates it
  as positive, matching `agent/kt/scripts/run_lf_lora_sft_kt.sh`. This avoids a
  preflight failure where the wrapper passed `0` even when
  `KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1`.
- Full profile sweep commands must use fixed dropout labels such as
  `LORA_DROPOUT=0.00`; the profile wrapper rejects `0.0` because the value is
  also used in artifact labels.
- Skip Stage 5 for now. `backward_repack_wait_ms` is about `0.004 ms/layer`,
  so repack overlap is not a meaningful bottleneck in the current profile.
- Skip Stage 6 for now. Route scatter/merge remains tens of milliseconds per
  layer, while CPU grouped expert math remains seconds per layer.
Stage 7 full same-config LF acceptance is complete:

- Artifact:
  `profiling_results/profiling_kt_codex_smoke/v6_accept_qwen3_s4096_b4_r64_w5_s10_t64_source/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a128_drop000/kt_armbf16__source__recomp__polnone__routerhf__expact0/b4_s4096`
- Validation:
  `PASS KT ARM profile: gpu_id=1 affinity_count=144 wrappers=48 fw=1440 bw=720`
- Shape: `Qwen/Qwen3-30B-A3B`, `seq_len=4096`, `batch=4`,
  `rank=64`, `dropout=0.00`, `warmup_steps=5`, `measure_steps=10`,
  `trainer_max_steps=15`
- Device/threading: physical GPU 1, `CUDA_VISIBLE_DEVICES=1`,
  `KT_NUM_THREADS=64`, `KT_ARM_OMP_NUM_THREADS=64`,
  `KT_ARM_SFT_BACKWARD_THREADS=64`, CPU affinity `0-143`
- E2E trainer timing:
  - measured e2e step `276.813 s`
  - total e2e step `277.079 s`
  - measured forward avg `75.246 s`
  - measured backward avg `199.535 s`
- Memory:
  - peak allocated HBM `34.479 GiB`
  - peak reserved HBM `44.029 GiB`
  - process RSS peak `184.467 GiB`
- Native counters, measured last 10 steps:
  - `expert_schedule_wall_ms` avg `1078.092 ms/layer`
  - `backward_grouped_tile_ms` avg `2457.110 ms/layer`
  - `backward_tile_recompute_ms` avg `44131.841 task-ms/layer`
  - `backward_route_grad_accum_ms` avg `82977.276 task-ms/layer`
  - `backward_base_grad_ms` avg `66154.476 task-ms/layer`
  - `backward_lora_grad_ms` avg `15930.130 task-ms/layer`
  - `backward_local_alloc_zero_ms` avg `56.582 ms/layer`
  - `backward_thread_reduce_ms` avg `0.000 ms/layer`
  - `backward_repack_wait_ms` avg `0.003 ms/layer`
  - `backward_route_scatter_ms` avg `13.292 ms/layer`
  - `sparse_backward_scratch_bytes` avg `3.043 GiB`, max `3.074 GiB`
- Losses: measured max `1.8324`, measured last `1.4692`,
  trainer `1.6074`

Stage 7 script fixes:

- `agent/kt/scripts/profile_lora_lf_kt.sh` now passes the sibling
  `train.log` to `validate_kt_arm_profile.py` during `existing_profile_complete`.
  The first completed full run exited nonzero after training because this
  wrapper validation call omitted `--train-log`; the artifact itself was
  complete and strict validation passed once the log was supplied.
- `agent/kt/scripts/validate_kt_arm_profile.py` now infers `train.log` and
  `lf_run/train.log` before falling back to `train_*.log` patterns.
- `agent/kt/scripts/lf/run_lf_profiled_train.py` removes the live
  `source_profile.partial.json` sidecar after a successful final
  `source_profile.json` write.
- Wrapper validation was rechecked with `COLLECT_EXISTING=true`; it found the
  completed artifact without rerunning training.

## Stage 0: Fix KT Profile Metadata And Thread Sweep

Scope:

- `agent/kt/scripts/profile_lora_lf_kt.sh`
  - defaults near `SEQ_LENS`, `PER_DEVICE_TRAIN_BATCH_SIZE`, `MAX_STEPS`,
    `PROFILE_LEVEL`, `KT_NUM_THREADS`, `KT_ARM_OMP_NUM_THREADS`
  - job env construction around `PROFILE_WARMUP_STEPS`,
    `PROFILE_MEASURE_STEPS`, `PROFILE_TOTAL_STEPS`, `KT_NUM_THREADS`
- `scripts/kt/profile_lora_lf_kt.sh`
  - thin wrapper should remain a wrapper only
- `agent/kt/scripts/run_lf_lora_sft_kt.sh`
  - validate GPU 1/2 guard, KT thread propagation, and config logging
- `scripts/kt/run_lf_lora_sft_kt.sh`
  - thin wrapper should remain a wrapper only
- `agent/kt/scripts/lf/run_lf_profiled_train.py`
  - profile config fields around warmup/measured/total trainer steps

Implementation:

- Make the KT profile script default to the same comparison shape:
  - `SEQ_LENS=${SEQ_LENS:-4096}`
  - `PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}`
  - `PROFILE_LEVEL=${PROFILE_LEVEL:-module}`
  - keep `MAX_STEPS` as measured steps in `profile_lora_lf_kt.sh`
  - continue computing `TOTAL_STEPS=$((MAX_STEPS + WARMUP_STEPS))`
- Keep the run script default `CUTOFF_LEN=4096`. Do not change the generic
  non-profile default batch unless there is a KT-specific reason.
- Fix profile metadata so automated comparisons cannot confuse measured steps
  with trainer total steps.

Concrete pseudocode:

```python
# agent/kt/scripts/lf/run_lf_profiled_train.py
trainer_max_steps = int(args.max_steps)
warmup_steps = int(env_config.get("warmup_steps", 0))
profile_total_steps = int(env_config.get("total_steps", trainer_max_steps))
measure_steps = env_config.get("measure_steps")
if measure_steps is None:
    measure_steps = max(0, profile_total_steps - warmup_steps)
measure_steps = int(measure_steps)

profile_config.update({
    "trainer_max_steps": trainer_max_steps,
    "warmup_steps": warmup_steps,
    "measure_steps": measure_steps,
    "profile_total_steps": profile_total_steps,
    "measured_step_start": warmup_steps + 1,
    "measured_step_end": warmup_steps + measure_steps,
    # Keep only if downstream scripts still read it.
    "max_steps": measure_steps,
})
```

- Add a same-shape thread sweep before touching kernels. The previous
  same-config run used only 8 threads, which is not enough evidence for a CPU
  kernel path. Sweep `8,16,32,64` with one measured step first, then use the
  best thread count for later full acceptance.

Risks and watches:

- `KT_NUM_THREADS=64` may raise RSS too much because current LoRA gradient
  partial buffers are per thread. If 32/64 cannot run or spend too much time in
  reduction/allocation, move Stage 4 before Stage 1.
- Do not accept a kernel change from the one-step sweep. It only chooses the
  best thread setting for the next e2e profiles.

Validation before Stage 1:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

bash -n agent/kt/scripts/profile_lora_lf_kt.sh
bash -n agent/kt/scripts/run_lf_lora_sft_kt.sh
bash -n scripts/kt/profile_lora_lf_kt.sh
bash -n scripts/kt/run_lf_lora_sft_kt.sh

.venv/bin/python -m py_compile \
  agent/kt/scripts/lf/run_lf_profiled_train.py \
  agent/kt/scripts/validate_kt_arm_profile.py
```

One-step same-shape sweep:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

for T in 8 16 32 64; do
  ART="profiling_results/profiling_kt_codex_smoke/v6_stage0_threads_t${T}_qwen3_s4096_b4_r64_1step"
  taskset -c 0-143 env \
    GPU_ID=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
    BACKEND=kt_armbf16 KT_BACKEND=armbf16 KT_PRECISION=bf16 \
    KT_NUM_THREADS="${T}" KT_ARM_OMP_NUM_THREADS="${T}" KT_ARM_SFT_BACKWARD_THREADS="${T}" \
    KT_ARM_OMP_PROC_BIND=false KT_ARM_SFT_PROFILE=1 \
    KT_ARM_SFT_MAX_ROUTE_RANK_WORK=8388608 \
    KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
    MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B \
    DATASET=asym_long_sft_smoke__qwen3-30b-a3b__s4096 \
    CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 \
    LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=0.00 \
    GRADIENT_CHECKPOINTING=true \
    MAX_STEPS=1 MAX_SAMPLES=4 \
    PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=module PROFILE_SYNC=1 \
    PROFILE_WARMUP_STEPS=0 PROFILE_MEASURE_STEPS=1 PROFILE_TOTAL_STEPS=1 \
    PROFILE_OUTPUT_DIR="${ART}" PROFILE_JSON="${ART}/profile.json" \
    PROFILE_SOURCE_JSON="${ART}/source_profile.json" PROFILE_SUMMARY_MD="${ART}/summary.md" \
    PROFILE_HEARTBEAT_JSON="${ART}/heartbeat.json" \
    OUT_DIR="${ART}/lf_run" LOG_FILE="${ART}/train.log" RUN_ID="v6_stage0_t${T}_recomp" \
    scripts/kt/run_lf_lora_sft_kt.sh

  .venv/bin/python agent/kt/scripts/validate_kt_arm_profile.py \
    --profile-json "${ART}/profile.json" \
    --train-log "${ART}/train.log" \
    --expected-seq-len 4096 --expected-batch 4 --expected-rank 64 \
    --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
    --expected-recompute true --expected-route-rank-limit 8388608 \
    --require-native-field expert_schedule_wall_ms \
    --require-native-field backward_grouped_tile_ms \
    --require-native-field backward_base_grad_ms \
    --require-native-field backward_lora_grad_ms \
    --require-native-field sparse_backward_scratch_bytes
done
```

## Stage 1: Retile Grouped Base Backward Kernels

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `backward_grad_m_tile()`
  - `backward_grad_k_tile()`
  - `arm_tile_dy_down_to_grad_act_sve()`
  - `arm_tile_dy_down_to_grad_act_sve_blocked()`
  - `arm_tile_grad_gate_up_to_grad_x_sve()`
  - `arm_tile_grad_gate_up_to_grad_x_sve_blocked()`
  - `backward_tile_accumulate_grouped()`

Implementation:

- Add M-unrolled SVE helpers for the two base backward GEMM-like operations.
  Current helpers process 4 routes per inner tile. Add a compile-time
  `M_UNROLL=8` helper and keep the existing M4 path as the tail/fallback.
- Keep the existing layout. Do not add a new packing stage yet:
  - down base weights are `[expert, hidden, intermediate]`, contiguous over
    `intermediate`
  - gate/up base weights are `[expert, intermediate, hidden]`, contiguous over
    `hidden`
- Reuse `KT_ARM_SFT_BACKWARD_GRAD_K_TILE` and
  `KT_ARM_SFT_BACKWARD_GRAD_M_TILE`. Do not add new selector env vars that can
  select old scalar code.
- Add counters only if they separate the two base kernels:
  - `backward_base_down_to_act_ms`
  - `backward_base_gate_up_to_x_ms`
  These can be added inside `ArmSFTProfileStats` only if the split is needed to
  decide the next kernel.

Concrete pseudocode for `dy @ down -> grad_act`:

```cpp
void arm_tile_dy_down_to_grad_act_sve_m8(
    int M, int H, int I,
    const float* dy, int dy_ld,
    const ggml_bf16_t* down_hi, int down_ld_i,
    float* grad_act, int grad_act_ld,
    bool accumulate) {
  int lanes = svcntw();
  for (int i0 = 0; i0 < I; i0 += lanes) {
    pg32 = svwhilelt_b32(i0, I);
    pg16 = svwhilelt_b16(i0, I);
    for (int m0 = 0; m0 < M; m0 += 8) {
      mb = min(8, M - m0);
      acc[0:mb] = accumulate ? load grad_act rows : 0;
      for (int h = 0; h < H; ++h) {
        w = load_bf16_low_as_f32(pg16, down_hi + h * down_ld_i + i0);
        for (int mm = 0; mm < mb; ++mm) {
          acc[mm] = svmla_n_f32_m(
              pg32, acc[mm], w, dy[(m0 + mm) * dy_ld + h]);
        }
      }
      store acc[0:mb] to grad_act rows;
    }
  }
}
```

Concrete pseudocode for `grad_gate/up @ W -> grad_x`:

```cpp
void arm_tile_grad_gate_up_to_grad_x_sve_m8(
    int M, int I, int H,
    const float* grad_gate, int grad_gate_ld,
    const ggml_bf16_t* gate_ih, int gate_ld_h,
    const float* grad_up, int grad_up_ld,
    const ggml_bf16_t* up_ih, int up_ld_h,
    float* grad_x, int grad_x_ld,
    bool accumulate) {
  int lanes = svcntw();
  for (int h0 = 0; h0 < H; h0 += lanes) {
    pg32 = svwhilelt_b32(h0, H);
    pg16 = svwhilelt_b16(h0, H);
    for (int m0 = 0; m0 < M; m0 += 8) {
      mb = min(8, M - m0);
      acc[0:mb] = accumulate ? load grad_x rows : 0;
      for (int i = 0; i < I; ++i) {
        gw = load_bf16_low_as_f32(pg16, gate_ih + i * gate_ld_h + h0);
        uw = load_bf16_low_as_f32(pg16, up_ih + i * up_ld_h + h0);
        for (int mm = 0; mm < mb; ++mm) {
          acc[mm] = svmla_n_f32_m(
              pg32, acc[mm], gw, grad_gate[(m0 + mm) * grad_gate_ld + i]);
          acc[mm] = svmla_n_f32_m(
              pg32, acc[mm], uw, grad_up[(m0 + mm) * grad_up_ld + i]);
        }
      }
      store acc[0:mb] to grad_x rows;
    }
  }
}
```

Risks and watches:

- SVE register pressure may make M8 slower than M4 on this CPU. Keep the M4
  helper available as the tail path and profile M4 vs M8 with the real LF run.
- `BFDOT` does not directly apply here because the left operand is FP32
  (`dy`, `grad_gate`, `grad_up`). The win must come from route-row reuse,
  contiguous BF16 RHS loads, better unroll, and less loop overhead.
- Accumulation across K tiles must preserve `accumulate || tile_begin > 0`.
  This is the easiest correctness bug to introduce.

Validation before Stage 2:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest \
  ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

LF validation, using the best thread count from Stage 0:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
BEST_T=64
ART="profiling_results/profiling_kt_codex_smoke/v6_stage1_base_qwen3_s4096_b4_r64_t${BEST_T}_source"

taskset -c 0-143 env \
  GPU_ID=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  BACKEND=kt_armbf16 KT_BACKEND=armbf16 KT_PRECISION=bf16 \
  KT_NUM_THREADS="${BEST_T}" KT_ARM_OMP_NUM_THREADS="${BEST_T}" KT_ARM_SFT_BACKWARD_THREADS="${BEST_T}" \
  KT_ARM_OMP_PROC_BIND=false KT_ARM_SFT_PROFILE=1 \
  KT_ARM_SFT_MAX_ROUTE_RANK_WORK=8388608 \
  KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B \
  DATASET=asym_long_sft_smoke__qwen3-30b-a3b__s4096 \
  CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 \
  LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=0.00 \
  GRADIENT_CHECKPOINTING=true \
  MAX_STEPS=3 MAX_SAMPLES=12 \
  PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=module PROFILE_SYNC=1 \
  PROFILE_WARMUP_STEPS=1 PROFILE_MEASURE_STEPS=2 PROFILE_TOTAL_STEPS=3 \
  PROFILE_OUTPUT_DIR="${ART}" PROFILE_JSON="${ART}/profile.json" \
  PROFILE_SOURCE_JSON="${ART}/source_profile.json" PROFILE_SUMMARY_MD="${ART}/summary.md" \
  PROFILE_HEARTBEAT_JSON="${ART}/heartbeat.json" \
  OUT_DIR="${ART}/lf_run" LOG_FILE="${ART}/train.log" RUN_ID="v6_stage1_base_t${BEST_T}" \
  scripts/kt/run_lf_lora_sft_kt.sh

.venv/bin/python agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "${ART}/profile.json" \
  --train-log "${ART}/train.log" \
  --expected-seq-len 4096 --expected-batch 4 --expected-rank 64 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute true --expected-route-rank-limit 8388608 \
  --require-native-field backward_base_grad_ms \
  --require-native-field backward_lora_grad_ms \
  --require-native-field backward_grouped_tile_ms
```

Accept Stage 1 only if the LF source profile shows a material reduction in
`backward_base_grad_ms` and average step time does not regress. If the kernel
improves synthetic tests but not this LF profile, revert or keep it behind a
compile-time helper that is not the default.

## Stage 2: Retile Grouped LoRA Backward Kernels

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `backward_lora_rank_tile()`
  - `lora_bwd_grad_y_b_to_u_grouped_sve()`
  - `lora_bwd_grad_y_b_to_u_grouped_sve_blocked()`
  - `lora_bwd_grad_b_grouped_sve()`
  - `lora_bwd_grad_b_grouped_sve_blocked()`
  - `lora_bwd_grad_a_and_input_grouped_sve()`
  - `lora_bwd_grad_a_and_input_grouped_sve_blocked()`
  - `backward_tile_accumulate_grouped()`

Implementation:

- Split `lora_bwd_grad_a_and_input_grouped_sve()` into two helpers:
  - `lora_bwd_grad_a_grouped_sve()`
  - `lora_bwd_grad_input_grouped_sve()`
  This lets each helper use the best loop order and makes profiling clearer.
- Add M8 route unroll to `grad_y @ B -> grad_u` and LoRA grad-input.
- Add small K-row unroll to `grad_B += grad_y^T @ u` so each loaded `u[m, r:r+VL]`
  vector contributes to multiple contiguous K rows before moving on.
- Add small R unroll to `grad_A += grad_u^T @ input` so one input vector load
  contributes to several rank rows.
- Keep dropout-enabled native ARM unsupported for now. This acceptance shape is
  `dropout=0.00`; do not spend this stage on dropout masks.

Concrete pseudocode for `grad_u = grad_y @ B`:

```cpp
for (int r0 = 0; r0 < R; r0 += lanes) {
  pg = svwhilelt_b32(r0, R);
  for (int m0 = 0; m0 < M; m0 += 8) {
    mb = min(8, M - m0);
    acc_u[0:mb] = 0;
    for (int k = 0; k < K; ++k) {
      b = load_bf16_low_as_f32(pg16, b_kr + k * b_ld_r + r0);
      for (int mm = 0; mm < mb; ++mm) {
        gy = grad_y[(m0 + mm) * grad_y_ld + k] * scale;
        acc_u[mm] = svmla_n_f32_m(pg, acc_u[mm], b, gy);
      }
    }
    store acc_u[0:mb] to grad_u rows;
  }
}
```

Concrete pseudocode for `grad_B += grad_y^T @ u`:

```cpp
const int K_UNROLL = 4;
for (int k0 = 0; k0 < K; k0 += K_UNROLL) {
  kb = min(K_UNROLL, K - k0);
  for (int r0 = 0; r0 < R; r0 += lanes) {
    pg = svwhilelt_b32(r0, R);
    acc_b[0:kb] = load grad_b[k0 + kk, r0:r0+lanes];
    for (int m = 0; m < M; ++m) {
      uv = load u[m, r0:r0+lanes];
      for (int kk = 0; kk < kb; ++kk) {
        gy = grad_y[m, k0 + kk] * scale;
        acc_b[kk] = svmla_n_f32_m(pg, acc_b[kk], uv, gy);
      }
    }
    store acc_b[0:kb];
  }
}
```

Concrete pseudocode for split `grad_A`:

```cpp
const int R_UNROLL = 4;
for (int r0 = 0; r0 < R; r0 += R_UNROLL) {
  rb = min(R_UNROLL, R - r0);
  for (int k0 = 0; k0 < K; k0 += lanes) {
    pg = svwhilelt_b32(k0, K);
    acc_a[0:rb] = load grad_a[r0 + rr, k0:k0+lanes];
    for (int m = 0; m < M; ++m) {
      x = load input[m, k0:k0+lanes];
      for (int rr = 0; rr < rb; ++rr) {
        gu = grad_u[m, r0 + rr];
        acc_a[rr] = svmla_n_f32_m(pg, acc_a[rr], x, gu);
      }
    }
    store acc_a[0:rb];
  }
}
```

Concrete pseudocode for split `grad_input += grad_u @ A`:

```cpp
for (int k0 = 0; k0 < K; k0 += lanes) {
  pg32 = svwhilelt_b32(k0, K);
  pg16 = svwhilelt_b16(k0, K);
  for (int m0 = 0; m0 < M; m0 += 8) {
    mb = min(8, M - m0);
    acc_x[0:mb] = load grad_input rows;
    for (int r = 0; r < R; ++r) {
      a = load_bf16_low_as_f32(pg16, lora_a_rk + r * a_ld_k + k0);
      for (int mm = 0; mm < mb; ++mm) {
        gu = grad_u[(m0 + mm) * grad_u_ld + r];
        acc_x[mm] = svmla_n_f32_m(pg32, acc_x[mm], a, gu);
      }
    }
    store acc_x[0:mb];
  }
}
```

Risks and watches:

- Rank is small (`R=64`) but appears in six LoRA backward kernels per expert
  tile. Do not over-unroll rank until M/K reuse is proven by LF profile.
- FP32 accumulation ordering changes will not be bitwise identical. Unit tests
  should use the existing tolerance; investigate only material loss drift.
- The LoRA B layout is contiguous over rank. The proposed `K_UNROLL` keeps that
  property and reuses `u` vectors; do not switch to a layout that makes rank
  gathers necessary.

Validation before Stage 3:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest \
  ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

LF validation:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
BEST_T=64
ART="profiling_results/profiling_kt_codex_smoke/v6_stage2_lora_bwd_qwen3_s4096_b4_r64_t${BEST_T}_source"

taskset -c 0-143 env \
  GPU_ID=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  BACKEND=kt_armbf16 KT_BACKEND=armbf16 KT_PRECISION=bf16 \
  KT_NUM_THREADS="${BEST_T}" KT_ARM_OMP_NUM_THREADS="${BEST_T}" KT_ARM_SFT_BACKWARD_THREADS="${BEST_T}" \
  KT_ARM_OMP_PROC_BIND=false KT_ARM_SFT_PROFILE=1 \
  KT_ARM_SFT_MAX_ROUTE_RANK_WORK=8388608 \
  KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B \
  DATASET=asym_long_sft_smoke__qwen3-30b-a3b__s4096 \
  CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 \
  LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=0.00 \
  GRADIENT_CHECKPOINTING=true \
  MAX_STEPS=3 MAX_SAMPLES=12 \
  PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=module PROFILE_SYNC=1 \
  PROFILE_WARMUP_STEPS=1 PROFILE_MEASURE_STEPS=2 PROFILE_TOTAL_STEPS=3 \
  PROFILE_OUTPUT_DIR="${ART}" PROFILE_JSON="${ART}/profile.json" \
  PROFILE_SOURCE_JSON="${ART}/source_profile.json" PROFILE_SUMMARY_MD="${ART}/summary.md" \
  PROFILE_HEARTBEAT_JSON="${ART}/heartbeat.json" \
  OUT_DIR="${ART}/lf_run" LOG_FILE="${ART}/train.log" RUN_ID="v6_stage2_lora_bwd_t${BEST_T}" \
  scripts/kt/run_lf_lora_sft_kt.sh

.venv/bin/python agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "${ART}/profile.json" \
  --train-log "${ART}/train.log" \
  --expected-seq-len 4096 --expected-batch 4 --expected-rank 64 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute true --expected-route-rank-limit 8388608 \
  --require-native-field backward_lora_grad_ms \
  --require-native-field backward_base_grad_ms \
  --require-native-field backward_grouped_tile_ms
```

Accept Stage 2 only if `backward_lora_grad_ms` and average step time improve on
the Stage 1 LF artifact.

## Stage 3: Retile Forward And Backward-Recompute LoRA Helpers

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `compute_gate_up_lora_by_expert()`
  - `compute_down_lora_by_expert()`
  - `fill_backward_recompute_tile()`
  - `backward_impl_packed()` route-tile recompute call sequence
  - optional new helpers:
    - `lora_forward_a_to_u_bf16_grouped_sve()`
    - `lora_forward_a_to_u_fp32_grouped_sve()`
    - `lora_forward_b_accum_grouped_sve()`

Implementation:

- Replace per-packed-route LoRA forward loops with grouped route-tile helpers.
- Keep LoRA A in its existing contiguous-feature layout:
  - gate/up LoRA A: `[expert, rank, hidden]`
  - down LoRA A: `[expert, rank, intermediate]`
- Use M-unroll across route rows for each rank. Loading one LoRA A row should
  update several route rows before moving to the next rank.
- Keep existing LoRA B transposed buffers for output accumulation:
  - `gate_lora_b_t_`
  - `up_lora_b_t_`
  - `down_lora_b_t_`
- Continue writing `gate_u`, `up_u`, and `down_u` because backward needs them
  for LoRA gradient accumulation.

Concrete pseudocode for gate/up LoRA A:

```cpp
void lora_forward_gate_up_a_to_u_grouped(
    int M, int H, int R, int PR,
    const ggml_bf16_t* x_mh,
    const ggml_bf16_t* gate_a_rh,
    const ggml_bf16_t* up_a_rh,
    float* gate_u_mpr,
    float* up_u_mpr) {
  zero gate_u/up_u for M * PR;
  for (int r = 0; r < R; ++r) {
    for (int m0 = 0; m0 < M; m0 += 8) {
      mb = min(8, M - m0);
      acc_gate[0:mb] = 0;
      acc_up[0:mb] = 0;
      for (int h0 = 0; h0 < H; h0 += bf16_vector_lanes) {
        x[0:mb] = load bf16 x rows at h0 and widen;
        ga = load bf16 gate_a[r, h0:h0+lanes] and widen;
        ua = load bf16 up_a[r, h0:h0+lanes] and widen;
        acc_gate[mm] += horizontal_sum(x[mm] * ga);
        acc_up[mm] += horizontal_sum(x[mm] * ua);
      }
      gate_u[(m0 + mm) * PR + r] = acc_gate[mm];
      up_u[(m0 + mm) * PR + r] = acc_up[mm];
    }
  }
}
```

Concrete pseudocode for LoRA B accumulation:

```cpp
void lora_forward_b_accum_grouped(
    int M, int N, int R, int PR,
    const float* u_mpr,
    const ggml_bf16_t* b_t_rn,
    float scale,
    float* out_mn) {
  for (int n0 = 0; n0 < N; n0 += lanes) {
    pg = svwhilelt_b32(n0, N);
    for (int m0 = 0; m0 < M; m0 += 8) {
      mb = min(8, M - m0);
      acc[0:mb] = load out rows;
      for (int r = 0; r < R; ++r) {
        b = load_bf16_low_as_f32(pg16, b_t_rn + r * N + n0);
        for (int mm = 0; mm < mb; ++mm) {
          coeff = scale * u_mpr[(m0 + mm) * PR + r];
          acc[mm] = svmla_n_f32_m(pg, acc[mm], b, coeff);
        }
      }
      store acc[0:mb] to out rows;
    }
  }
}
```

Concrete pseudocode for down LoRA A:

```cpp
for (int r = 0; r < R; ++r) {
  for (int m0 = 0; m0 < M; m0 += 8) {
    acc_down_u[0:mb] = 0;
    for (int i0 = 0; i0 < I; i0 += lanes) {
      act_vec[0:mb] = load fp32 act rows;
      a_vec = load_bf16_low_as_f32(pg16, down_lora_a[r, i0:i0+lanes]);
      acc_down_u[mm] += horizontal_sum(act_vec[mm] * a_vec);
    }
    down_u[(m0 + mm) * PR + r] = acc_down_u[mm];
  }
}
lora_forward_b_accum_grouped(M, H, R, PR, down_u, down_lora_b_t, scale, down);
```

Risks and watches:

- Horizontal reductions over feature vectors can become the bottleneck. If M8
  spills, keep M4 or split gate and up into separate helpers.
- The tile route builder may include padded invalid routes in some edge cases.
  Preserve the existing token validity checks or guarantee `tile.routes` counts
  only valid entries.
- Backward recompute uses the same helpers. A forward improvement must not
  break saved `gate_u/up_u/down_u` values consumed by Stage 2 kernels.

Validation before Stage 4:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest \
  ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

LF validation:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
BEST_T=64
ART="profiling_results/profiling_kt_codex_smoke/v6_stage3_lora_fwd_qwen3_s4096_b4_r64_t${BEST_T}_source"

taskset -c 0-143 env \
  GPU_ID=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  BACKEND=kt_armbf16 KT_BACKEND=armbf16 KT_PRECISION=bf16 \
  KT_NUM_THREADS="${BEST_T}" KT_ARM_OMP_NUM_THREADS="${BEST_T}" KT_ARM_SFT_BACKWARD_THREADS="${BEST_T}" \
  KT_ARM_OMP_PROC_BIND=false KT_ARM_SFT_PROFILE=1 \
  KT_ARM_SFT_MAX_ROUTE_RANK_WORK=8388608 \
  KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B \
  DATASET=asym_long_sft_smoke__qwen3-30b-a3b__s4096 \
  CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 \
  LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=0.00 \
  GRADIENT_CHECKPOINTING=true \
  MAX_STEPS=3 MAX_SAMPLES=12 \
  PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=module PROFILE_SYNC=1 \
  PROFILE_WARMUP_STEPS=1 PROFILE_MEASURE_STEPS=2 PROFILE_TOTAL_STEPS=3 \
  PROFILE_OUTPUT_DIR="${ART}" PROFILE_JSON="${ART}/profile.json" \
  PROFILE_SOURCE_JSON="${ART}/source_profile.json" PROFILE_SUMMARY_MD="${ART}/summary.md" \
  PROFILE_HEARTBEAT_JSON="${ART}/heartbeat.json" \
  OUT_DIR="${ART}/lf_run" LOG_FILE="${ART}/train.log" RUN_ID="v6_stage3_lora_fwd_t${BEST_T}" \
  scripts/kt/run_lf_lora_sft_kt.sh

.venv/bin/python agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "${ART}/profile.json" \
  --train-log "${ART}/train.log" \
  --expected-seq-len 4096 --expected-batch 4 --expected-rank 64 \
  --expected-dropout 0.0 --expected-top-k 8 --expected-cache-depth 2 \
  --expected-recompute true --expected-route-rank-limit 8388608 \
  --require-native-field expert_schedule_wall_ms \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_grouped_tile_ms
```

Accept Stage 3 only if `expert_schedule_wall_ms`, `backward_tile_recompute_ms`,
or average step time improves on the Stage 2 LF artifact.

## Stage 4: Reduce Per-Thread LoRA Scratch And Thread-Reduction Cost

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `BackwardBuffers`
  - `backward_impl_packed()`
  - `reduce_vector_fields()`
  - `reduce_lora_grads_by_disjoint_tiles()`
  - scratch estimate and `KT_ARM_SFT_BACKWARD_SCRATCH` logging

Implementation:

- Do this stage only if Stage 0 or later LF profiles show 16/32/64 threads are
  blocked by RSS, scratch allocation time, or `backward_thread_reduce_ms`.
- Replace full per-thread active-expert LoRA gradient buffers with expert-owned
  or shard-owned gradient buffers.
- Preserve grouped tile math. The scheduler may group tile descriptors by
  expert/shard; the compute kernel must still operate on route tiles, not one
  route at a time.

Concrete pseudocode for expert-owned scheduling:

```cpp
struct BackwardShard {
  std::vector<int> tile_indices;
  BackwardBuffers grads;  // only experts owned by this shard
};

int shard_count = min(threads, active_count);
std::vector<BackwardShard> shards(shard_count);

for (int tile_idx = 0; tile_idx < backward_tiles.size(); ++tile_idx) {
  const auto& tile = backward_tiles[tile_idx];
  int owner = tile.sparse_expert % shard_count;
  shards[owner].tile_indices.push_back(tile_idx);
}

#pragma omp parallel num_threads(shard_count)
{
  int sid = omp_get_thread_num();
  allocate grads only for experts owned by sid;
  for (int tile_idx : shards[sid].tile_indices) {
    recompute tile;
    backward_tile_accumulate_grouped(..., shards[sid].grads, ...);
  }
}

// If each sparse expert has one owner, flush directly.
// If heavy experts are split into multiple owners later, reduce only those
// split-expert partials instead of all active experts times all threads.
flush_sparse_lora_grad_accum(sharded_grads, ...);
```

- If one expert owns too many routes and load balance suffers, add
  `KT_ARM_SFT_BACKWARD_EXPERT_SHARDS` with default `1`, then split heavy
  experts into 2 or 4 shards and reduce only those experts.

Risks and watches:

- Expert-owned scheduling may reduce parallelism on skewed routing. Use the
  route histogram from the profile log before making it default.
- Per-expert locks around FP32 LoRA gradient accumulation are simpler but likely
  too expensive. Prefer owner/shard reduction first.
- This stage should lower RSS and reduction time. It is not accepted if it only
  lowers scratch bytes but raises average step time.

Validation before Stage 5:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest \
  ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

Thread scaling validation:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

for T in 16 32 64; do
  ART="profiling_results/profiling_kt_codex_smoke/v6_stage4_shards_t${T}_qwen3_s4096_b4_r64_source"
  taskset -c 0-143 env \
    GPU_ID=1 CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
    BACKEND=kt_armbf16 KT_BACKEND=armbf16 KT_PRECISION=bf16 \
    KT_NUM_THREADS="${T}" KT_ARM_OMP_NUM_THREADS="${T}" KT_ARM_SFT_BACKWARD_THREADS="${T}" \
    KT_ARM_OMP_PROC_BIND=false KT_ARM_SFT_PROFILE=1 \
    KT_ARM_SFT_MAX_ROUTE_RANK_WORK=8388608 \
    KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
    MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B \
    DATASET=asym_long_sft_smoke__qwen3-30b-a3b__s4096 \
    CUTOFF_LEN=4096 PER_DEVICE_TRAIN_BATCH_SIZE=4 \
    LORA_RANK=64 LORA_ALPHA=128 LORA_DROPOUT=0.00 \
    GRADIENT_CHECKPOINTING=true \
    MAX_STEPS=3 MAX_SAMPLES=12 \
    PROFILE=1 PROFILE_PROFILER=source PROFILE_LEVEL=module PROFILE_SYNC=1 \
    PROFILE_WARMUP_STEPS=1 PROFILE_MEASURE_STEPS=2 PROFILE_TOTAL_STEPS=3 \
    PROFILE_OUTPUT_DIR="${ART}" PROFILE_JSON="${ART}/profile.json" \
    PROFILE_SOURCE_JSON="${ART}/source_profile.json" PROFILE_SUMMARY_MD="${ART}/summary.md" \
    PROFILE_HEARTBEAT_JSON="${ART}/heartbeat.json" \
    OUT_DIR="${ART}/lf_run" LOG_FILE="${ART}/train.log" RUN_ID="v6_stage4_shards_t${T}" \
    scripts/kt/run_lf_lora_sft_kt.sh
done
```

Accept Stage 4 only if the best thread count improves e2e time without
unacceptable RSS growth and `sparse_backward_scratch_bytes` or
`backward_thread_reduce_ms` decreases materially.

## Stage 5: Improve Backward Weight Repack Overlap Only If Still Visible

Scope:

- `../ktransformers/kt-kernel/python/sft/autograd.py`
  - `KTMoEFunction.backward()`
  - calls to `ctx.wrapper.wait_backward_repack()`
  - calls to `next_bwd.submit_backward_repack()`
- `../ktransformers/kt-kernel/python/sft/base.py`
  - `submit_backward_repack()`
  - `wait_backward_repack()`
- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `submit_backward_repack()`
  - `wait_backward_repack()`
  - `prepare_backward_weights_from_forward()`
  - `transpose_base_weights()`
  - profile fields `backward_repack_submit_ms`, `backward_repack_wait_ms`

Implementation:

- Do this stage only if `backward_repack_wait_ms` remains above about 5 percent
  of backward wall time after Stages 1-3. In the current partial profile it is
  lower priority than base/LoRA math.
- Verify whether `submit_backward_repack()` is being triggered early enough for
  the next layer in backward order. The Python path already calls
  `next_bwd.submit_backward_repack()` after one MoE backward; the first backward
  layer may still pay synchronous wait.
- If the wait is real, make the submit point earlier only when the target
  wrapper's forward weights are stable and its shared backward buffer is not
  currently owned by another layer.

Concrete pseudocode:

```python
# python/sft/autograd.py
if next_bwd is not None and next_bwd.share_backward_bb:
    # Submit immediately after this wrapper no longer needs the shared pool.
    # Do not submit if next_bwd already has an active repack thread.
    if not next_bwd.backward_repack_async_active():
        next_bwd.submit_backward_repack()
```

```cpp
// bf16_sft_moe.hpp
void submit_backward_repack() {
  if (backward_repack_async_active()) {
    return;  // avoid submit -> wait -> serialize
  }
  if (backward_weights_prepared_ && shared_backward_weights_owned_by_this_layer()) {
    return;
  }
  launch thread that calls prepare_backward_weights_from_forward();
}
```

Risks and watches:

- Shared backward weight pool ownership is subtle. A wrong overlap change can
  corrupt weights across layers. Keep this behind correctness tests and do not
  prioritize it before math kernels.
- If `backward_repack_wait_ms` falls after thread/kernel changes, skip this
  stage.

Validation before Stage 6:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest \
  ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

LF validation must include:

- `backward_repack_wait_ms`
- `backward_repack_submit_ms`
- `async_submit`
- `async_wait`
- average step time

Use the same Stage 1 LF command shape and artifact naming:
`profiling_results/profiling_kt_codex_smoke/v6_stage5_repack_qwen3_s4096_b4_r64_t${BEST_T}_source`.

## Stage 6: Low-Priority Cleanup Only After Math Wins

Scope:

- `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
  - `scatter_route_grad_x_to_tokens()`
  - `merge_routes_to_output()` if it is still visible in the profile
  - `compute_activation_grads_grouped()`
  - dropout path in `backward_tile_accumulate_grouped()`
- `agent/kt/scripts/**`
  - stale experimental env vars should remain rejected or absent

Implementation:

- Route scatter/merge is currently tens of milliseconds per layer, not the main
  problem. Only optimize after Stages 1-3 materially reduce CPU math.
- If needed, keep the current SVE route scatter style and only tune scheduling
  or token blocking. Do not reintroduce route-serial compute.
- Native ARM dropout greater than zero can stay unsupported for the acceptance
  shape. Add grouped dropout only if later LoRA runs require it.
- Remove or reject any new experimental selector that lets users select stale
  scalar fallback code.

Concrete pseudocode for route scatter cleanup, if still needed:

```cpp
direct_or_parallel(token_blocks, qlen, [&](int block) {
  for token in block:
    for h0 in 0..H step lanes:
      acc = 0;
      for route in 0..topk:
        packed = flat_route_to_packed[token * topk + route];
        if packed >= 0:
          acc += load route_grad_x[packed, h0:h0+lanes];
      store bf16 grad_input[token, h0:h0+lanes];
});
```

Risks and watches:

- Spending time here before base/LoRA math improves will not close the gap with
  the Asym activation-offload table.
- Dropout mask semantics affect correctness. Do not add dropout kernels without
  explicit dropout tests and LF validation.

Validation before final acceptance:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m pytest \
  ../ktransformers/kt-kernel/test/per_commit/test_armbf16_sft_reference.py \
  ../ktransformers/kt-kernel/test/per_commit/test_sft_lora_dropout.py -q
```

Run the same short LF source profile as previous stages and accept only if
average step time does not regress.

## Stage 7: Full Same-Config LF Acceptance Profile

Scope:

- No new implementation in this stage.
- Required scripts:
  - `scripts/kt/profile_lora_lf_kt.sh`
  - `agent/kt/scripts/profile_lora_lf_kt.sh`
  - `agent/kt/scripts/validate_kt_arm_profile.py`
  - `agent/kt/scripts/lf/postprocess_lf_profile_artifacts.py` if needed for
    table generation

Implementation:

- Run the full accepted profile only after the short LF source profile shows the
  target counters improved.
- Use the best thread count from the Stage 0/Stage 4 sweep.
- Compare only against same shape (`batch=4`, `seq=4096`, `rank=64`,
  `dropout=0.00`). The old partial KT profile is not accepted comparison data.

Full acceptance command:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
BEST_T=64  # validated winner from Stage 0/Stage 4 short profiles

SEQ_LENS=4096 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 \
BACKEND_SPECS='kt_armbf16|recomp' \
GPU_POOL=1 \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES='none|false' \
CONTINUE_ON_ERROR=false \
RUN_POST=false \
KT_NUM_THREADS="${BEST_T}" \
KT_ARM_OMP_NUM_THREADS="${BEST_T}" \
KT_ARM_SFT_BACKWARD_THREADS="${BEST_T}" \
KT_ARM_OMP_PROC_BIND=false \
KT_ARM_SFT_PROFILE=1 \
KT_ARM_SFT_MAX_ROUTE_RANK_WORK=8388608 \
KT_ARM_ALLOW_UNVALIDATED_BACKWARD_SCRATCH=1 \
MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B \
LORA_RANK=64 \
LORA_ALPHA=128 \
LORA_DROPOUT=0.00 \
WARMUP_STEPS=5 \
MAX_STEPS=10 \
MAX_SAMPLES=60 \
PROFILE=1 \
PROFILE_LEVEL=module \
PROFILE_SYNC=1 \
OUTPUT_ROOT=profiling_results/profiling_kt_codex_smoke/v6_accept_qwen3_s4096_b4_r64_w5_s10_t${BEST_T}_source \
scripts/kt/profile_lora_lf_kt.sh
```

If GPU 1 is unavailable, rerun with `GPU_POOL=2`; do not use GPU 0 or GPU 3 for
accepted numbers.

Acceptance validation:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
ART="profiling_results/profiling_kt_codex_smoke/v6_accept_qwen3_s4096_b4_r64_w5_s10_t${BEST_T}_source"
PROFILE_JSON="$(find "${ART}" -path '*/b4_s4096/profile.json' -print -quit)"
TRAIN_LOG="$(dirname "${PROFILE_JSON}")/train.log"

.venv/bin/python agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json "${PROFILE_JSON}" \
  --train-log "${TRAIN_LOG}" \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 4096 \
  --expected-batch 4 \
  --expected-rank 64 \
  --expected-dropout 0.0 \
  --expected-warmup-steps 5 \
  --expected-measure-steps 10 \
  --expected-profile-total-steps 15 \
  --expected-trainer-max-steps 15 \
  --expected-measured-step-start 6 \
  --expected-measured-step-end 15 \
  --min-measured-step-samples 10 \
  --expected-top-k 8 \
  --expected-cache-depth 2 \
  --expected-recompute true \
  --expected-route-rank-limit 8388608 \
  --require-final \
  --require-native-kv base_kernel=sve_bfdot_blocked \
  --require-native-kv down_kernel=bf16_bfdot_blocked \
  --require-native-kv lora_forward_kernel=sve_bfdot_fmla \
  --require-native-kv backward_base_kernel=grouped_sve_tile \
  --require-native-kv backward_lora_kernel=grouped_sve_tile_dropout0 \
  --require-native-kv compiled_sve_bf16=1 \
  --require-native-field expert_schedule_wall_ms \
  --require-native-field backward_grouped_tile_ms \
  --require-native-field backward_tile_recompute_ms \
  --require-native-field backward_route_grad_accum_ms \
  --require-native-field backward_base_grad_ms \
  --require-native-field backward_lora_grad_ms \
  --require-native-field sparse_backward_scratch_bytes
```

Final report must include:

- artifact path
- GPU id and thread count
- peak allocated and reserved HBM
- process RSS peak
- average step, forward, backward
- `expert_schedule_wall_ms`
- `backward_grouped_tile_ms`
- `backward_base_grad_ms`
- `backward_lora_grad_ms`
- `backward_tile_recompute_ms`
- `backward_repack_wait_ms`
- `backward_route_scatter_ms`
- `sparse_backward_scratch_bytes`
- loss max/last/train
- native kernel labels

Do not compare estimates against the Asym activation-offload table. Compare only
full accepted KT source artifacts from the same `batch=4`, `seq=4096`,
`rank=64`, `dropout=0.00`, `warmup=5`, `measure=10` shape.

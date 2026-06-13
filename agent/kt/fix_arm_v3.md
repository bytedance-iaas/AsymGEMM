# KT ARM BF16 SFT Fix Plan v3

This file is the staged execution plan for the slow `kt_armbf16` SFT path. Do not treat the large profiling workload itself as a bug: batch 4, sequence 7168, rank 64, 14 token chunks, 48 layers, and activation checkpointing is expected to be heavy. The remaining issues are launch/threading correctness, incomplete profiling, and native kernel structure.

Rule: do not move to the next stage until the validation gate for the current stage passes and the profile artifact is saved under `profiling_kt_codex_smoke/`.

## Current Facts

- The completed rank-64 run entered native KT ARM BF16, not DeepSpeed and not a scalar fallback. `train.log` reports `path=packed`, `task_dispatch=worker_pool`, `base_kernel=sve_bfdot`, and `aligned_weights=1`.
- The same run recorded `cpu_affinity_count=1`, `cpu_affinity.cpus=0`, while the host shell can see 144 CPUs. This is the most suspicious correctness issue.
- KT totals for the completed rank-64 source profile: 48 wrappers, 1344 forward native calls, 672 backward native calls.
- Per 2048-token chunk, native forward averages approximately: base gate/up 5.6s, LoRA gate/up 6.1s, base down 3.8s, LoRA down 3.0s, route merge 0.31s. Copies to/from CUDA are small.
- Backward aggregate logs time only `lora_grad_reduce_ms` and `grad_flush_ms`; wrapper backward sync shows much more time than those two fields explain. Allocation, zeroing, thread reduction, grad-input flush, and some recompute work need explicit timers.
- Low GPU memory is not suspicious by itself. KT keeps routed expert work CPU-side; memory comparison against ZeRO-3 offload is not one-to-one.

Useful source locations:

- Launcher and profiling env: `scripts/lf/run_lf_lora_sft.sh`
- Sweep wrapper: `scripts/lf/profile_lora_lf.sh`
- ARM Python wrapper: `../ktransformers/kt-kernel/python/sft/arm.py`
- Autograd/checkpoint hook: `../ktransformers/kt-kernel/python/sft/autograd.py`
- LF KT layer bridge: `../ktransformers/kt-kernel/python/sft/layer.py`
- Native ARM SFT kernel: `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
- Worker pool: `../ktransformers/kt-kernel/cpu_backend/worker_pool.cpp`

## Stage 0: Baseline Repro and Artifact Parser

Goal: make every later stage compare against the same small and large shapes.

Commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

python3 - <<'PY'
import os
print("affinity_count", len(os.sched_getaffinity(0)))
print("affinity", sorted(os.sched_getaffinity(0))[:8], "...", sorted(os.sched_getaffinity(0))[-8:])
PY

python3 - <<'PY'
import json, pathlib, re, statistics
run = pathlib.Path("profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_drop010_chunk2048_detached")
profile = json.loads((run / "profile.json").read_text())
print("config", {k: profile["config"].get(k) for k in [
    "batch_size", "seq_len", "logical_qlen", "lora_rank", "activation_recompute",
    "kt_arm_sft_token_chunk_size", "kt_arm_token_chunks", "kt_num_threads",
    "kt_arm_omp_num_threads", "cpu_affinity", "cpu_affinity_count",
]})
print("kt", {k: profile["kt"].get(k) for k in [
    "wrapper_count", "total_forward_calls", "total_backward_calls",
]})
rows = []
for line in (run / "train.log").read_text(errors="replace").splitlines():
    if "KT_ARM_SFT_PROFILE" not in line:
        continue
    rec = {}
    for m in re.finditer(r"(\w+)=([^\s]+)", line):
        key, value = m.groups()
        try:
            rec[key] = float(value)
        except ValueError:
            rec[key] = value
    rows.append(rec)
fwd = [r for r in rows if float(r.get("lora_grad_reduce_ms", 0)) == 0]
bwd = [r for r in rows if float(r.get("lora_grad_reduce_ms", 0)) > 0 or float(r.get("grad_flush_ms", 0)) > 0]
for name, data, keys in [
    ("fwd", fwd, ["base_gate_up_ms", "lora_gate_up_ms", "base_down_ms", "lora_down_ms", "route_merge_ms"]),
    ("bwd", bwd, ["lora_grad_reduce_ms", "grad_flush_ms"]),
]:
    print(name, len(data))
    for key in keys:
        xs = [float(r.get(key, 0)) for r in data]
        if xs:
            print(" ", key, "mean_ms", round(sum(xs) / len(xs), 3), "sum_ms", round(sum(xs), 3))
PY
```

Validation gate:

- The parser prints the known completed artifact values above.
- All future profiles must be summarized with the same parser or a checked-in equivalent script.

## Stage 1: Fix Launch CPU Affinity

Problem: the completed run requested 8 KT threads but recorded only CPU `0` in its affinity mask. This can make KT appear stuck and invalidates performance conclusions.

Implementation:

- Add a launcher preflight for `BACKEND=kt_armbf16` in `scripts/lf/run_lf_lora_sft.sh`.
- Compute available CPU affinity with Python `os.sched_getaffinity(0)`.
- Fail early when `affinity_count < KT_NUM_THREADS` or `affinity_count < KT_ARM_OMP_NUM_THREADS`.
- Log `taskset -pc $$`, `/proc/self/status` `Cpus_allowed_list`, `OMP_NUM_THREADS`, `KT_NUM_THREADS`, `KT_ARM_OMP_NUM_THREADS`, and numactl settings into `train.log`.
- Add an explicit override variable, for example `KT_ARM_ALLOW_LOW_CPU_AFFINITY=1`, only for negative tests.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

# Unit/script tests after adding the guard.
.venv/bin/python -m pytest tests/lf/test_superoffload_backend_scripts.py -k "kt_arm" -q

# Negative guard: this should fail before training starts.
taskset -c 0 env \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 \
  MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=1 \
  OUT_DIR=/tmp/kt_affinity_negative \
  scripts/lf/run_lf_lora_sft.sh

# Positive smoke: this must complete and profile must report enough CPUs.
taskset -c 0-143 env \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_affinity_positive \
  scripts/lf/run_lf_lora_sft.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_affinity_positive.log

python3 - <<'PY'
import json, pathlib
p = pathlib.Path("profiling_kt_codex_smoke/kt_armbf16_affinity_positive/profile.json")
profile = json.loads(p.read_text())
cfg = profile["config"]
assert cfg["cpu_affinity_count"] >= int(cfg["kt_num_threads"]), cfg
print("PASS affinity", cfg["cpu_affinity"])
PY
```

Profile gate:

- Positive smoke `profile.json` has `cpu_affinity_count >= KT_NUM_THREADS`.
- `train.log` shows `base_kernel=sve_bfdot`.
- No large profile may be accepted if `cpu_affinity_count < KT_NUM_THREADS`.

## Stage 2: Fix Worker-Pool Placement

Problem: `WorkerPool` uses worker 0 as the caller thread, while persistent worker threads start at index 1. If the caller is badly pinned, every native task includes a badly placed worker.

Implementation:

- In `cpu_backend/worker_pool.cpp`, ensure the caller path for worker 0 is bound or validated before `process_tasks(0)`.
- Prefer a minimal guard first: when entering `do_work_stealing_job_async`, verify the caller CPU is within the intended subpool NUMA cpuset and log/fail if not.
- Then implement robust binding: set `WorkerPool::thread_local_id=0` and bind the caller thread to the same NUMA/core policy used for worker 0 before processing tasks.
- Avoid changing multi-subpool semantics for other KT paths; ARM SFT currently requires `threadpool_count=1`.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 env \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
  CUTOFF_LEN=128 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_worker_place \
  scripts/lf/run_lf_lora_sft.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_worker_place.log

# During a longer local run, sample thread placement.
PID=<python_or_torchrun_pid>
taskset -pc "$PID"
grep Cpus_allowed_list /proc/"$PID"/status
ps -L -o pid,tid,psr,pcpu,comm -p "$PID" | sort -k4 -nr | head -40
numastat -p "$PID"
```

Profile gate:

- Worker-pool log reports the expected effective worker count.
- `ps -L` shows KT worker threads running across the intended CPUs, not all parked on CPU 0.
- Source smoke time must not regress versus the Stage 1 positive smoke.

## Stage 3: Complete Native Timing Coverage

Problem: backward wrapper sync is much larger than the current `KT_ARM_SFT_PROFILE` backward fields explain.

Implementation:

- Extend `ArmSFTProfileStats` in `bf16_sft_moe.hpp` with explicit backward phase fields:
  - `backward_ensure_buffers_ms`
  - `backward_grad_weights_zero_ms`
  - `backward_local_alloc_zero_ms`
  - `backward_route_loop_ms`
  - `backward_thread_reduce_ms`
  - `backward_grad_input_flush_ms`
  - `backward_lora_grad_flush_ms`
- Keep existing `lora_grad_reduce_ms` for compatibility, but make `backward_route_loop_ms` the full route-loop timer.
- Add forward subphase counters for calls and bytes moved where possible: input pack bytes, base gate/up bytes, base down bytes, LoRA A/B bytes, merge bytes.
- Print all fields in one `KT_ARM_SFT_PROFILE` line so the existing log parser can consume it.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 env \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_NATIVE_PROGRESS=1 \
  MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
  CUTOFF_LEN=128 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_timer_coverage \
  scripts/lf/run_lf_lora_sft.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_timer_coverage.log

rg "backward_(ensure_buffers|local_alloc_zero|thread_reduce|grad_input_flush)_ms" \
  profiling_kt_codex_smoke/kt_armbf16_timer_coverage.log
```

Profile gate:

- For each backward call, the sum of detailed native backward phases should be within 10 percent of wrapper `backward_sync_ms_last`, excluding explicit Python/CUDA copy fields.
- If the gap is larger than 10 percent, do not optimize kernels yet; add missing timers first.

## Stage 4: Add a Native Microbenchmark Harness

Problem: the end-to-end LF run is too expensive to use as the only kernel development loop.

Implementation:

- Add a small C++ or Python-bound microbench that constructs one ARM SFT MoE wrapper/layer with synthetic BF16 inputs, synthetic top-k IDs/weights, and rank/top-k/hidden/intermediate matching the model.
- Benchmark at least these shapes:
  - `qlen=128, topk=8, rank=8`
  - `qlen=512, topk=8, rank=8`
  - `qlen=2048, topk=8, rank=64`
- Emit JSON with phase times, effective bandwidth estimates, thread count, affinity, NUMA placement, route skew, and checksum.
- Add a correctness mode comparing native output/grads against a simple Torch or scalar reference for tiny shapes.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 .venv/bin/python scripts/kt/bench_arm_sft_moe.py \
  --qlen 128 --top-k 8 --rank 8 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --threads 8 --check --json profiling_kt_codex_smoke/bench_arm_sft_q128_r8.json

taskset -c 0-143 .venv/bin/python scripts/kt/bench_arm_sft_moe.py \
  --qlen 2048 --top-k 8 --rank 64 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --threads 8 --json profiling_kt_codex_smoke/bench_arm_sft_q2048_r64.json
```

Profile gate:

- Tiny correctness mode passes against reference within BF16/FP32 tolerance.
- JSON includes all Stage 3 timing fields.
- Microbench timings trend in the same direction as LF source-profile timings.

## Stage 5: Replace Base Gate/Up With Real Grouped BF16 GEMM

Problem: `arm_bf16_matmul_tiled` is currently nested loops around a single `svbfdot` dot product. That is not a high-throughput GEMM.

Implementation options, in order:

1. Use a proven ARM BF16 GEMM library if available on the target host: Arm Performance Libraries, oneDNN ACL backend, OpenBLAS only if it has relevant BF16 support, or KML if usable through existing build knobs.
2. If no library is acceptable, implement a blocked SVE BF16 microkernel for `MxN x K` with register blocking across multiple output columns and rows, not one dot per output element.
3. Batch by expert: route packing already groups routes per expert, so call one grouped GEMM per active expert for gate and up.

Validation commands:

```bash
# Build after kernel change.
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

# Microbench before LF.
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 .venv/bin/python scripts/kt/bench_arm_sft_moe.py \
  --qlen 2048 --top-k 8 --rank 64 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --threads 8 --json profiling_kt_codex_smoke/bench_stage5_gate_up.json

# Optional CPU PMU profile if perf is installed and permitted.
command -v perf && perf stat -d -d -d -- \
  taskset -c 0-143 .venv/bin/python scripts/kt/bench_arm_sft_moe.py \
    --qlen 2048 --top-k 8 --rank 64 --hidden-size 2048 --intermediate-size 768 \
    --num-experts 128 --threads 8 --json /tmp/bench_stage5_gate_up.json
```

Profile gate:

- Correctness mode still passes.
- `base_gate_up_ms` improves materially on `qlen=2048, rank=64`; target first milestone is at least 2x faster than the existing dot-loop baseline.
- No regression in input packing or route merge.

## Stage 6: Replace Base Down Projection

Problem: base down currently loops over packed routes and does scalar FP32 accumulation over `H x I`; it does not use the SVE BF16 dot primitive as a real GEMM.

Implementation:

- Store or expose down weights in a layout suitable for BF16 GEMM.
- Compute `act[M, I] x down_weight[I, H] -> down[M, H]` by active expert using the same grouped-GEMM machinery as Stage 5.
- Keep FP32 accumulation and BF16 output conversion semantics unchanged.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 .venv/bin/python scripts/kt/bench_arm_sft_moe.py \
  --qlen 2048 --top-k 8 --rank 64 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --threads 8 --json profiling_kt_codex_smoke/bench_stage6_down.json
```

Profile gate:

- Correctness mode still passes.
- `base_down_ms` improves materially; target first milestone is at least 2x faster than scalar baseline.
- Total forward chunk time improves after Stage 5 plus Stage 6, not just the isolated phase.

## Stage 7: Batch LoRA Forward

Problem: LoRA gate/up/down forward currently loops per packed route and per rank. With rank 64 this dominates.

Implementation:

- For each active expert, compute LoRA A projections as grouped GEMMs:
  - gate/up: `X[M, H] x A_gate/up[H, R] -> U[M, R]`
  - down: `Act[M, I] x A_down[I, R] -> U_down[M, R]`
- Then compute LoRA B projections as grouped GEMMs:
  - gate/up: `U[M, R] x B_gate/up[R, I] -> delta[M, I]`
  - down: `U_down[M, R] x B_down[R, H] -> delta[M, H]`
- Preserve dropout exactly. If dropout masks prevent simple GEMM fusion, add two paths:
  - no-dropout fast path
  - dropout path with vectorized mask application into packed temporary buffers before GEMM
- The current run uses nonzero LoRA dropout, so the dropout path must be measured, not left as a slow fallback.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
for dropout in 0.00 0.10; do
  taskset -c 0-143 .venv/bin/python scripts/kt/bench_arm_sft_moe.py \
    --qlen 2048 --top-k 8 --rank 64 --hidden-size 2048 --intermediate-size 768 \
    --num-experts 128 --threads 8 --lora-dropout "$dropout" \
    --json "profiling_kt_codex_smoke/bench_stage7_lora_drop${dropout}.json"
done
```

Profile gate:

- Correctness passes for dropout 0.00 and 0.10.
- `lora_gate_up_ms + lora_down_ms` improves materially; target first milestone is at least 2x faster than current per-route loops.
- Dropout 0.10 is not more than 25 percent slower than dropout 0.00 unless profiling proves RNG/mask generation dominates.

## Stage 8: Redesign Backward Scratch and Reductions

Problem: backward allocates dense full-expert LoRA gradient buffers per thread, then reduces dense buffers. This creates large memory traffic and hides time outside the existing timers.

Implementation:

- Reuse per-wrapper backward scratch across calls instead of allocating/zeroing large vectors every chunk.
- Replace dense per-thread full-expert gradients with per-expert or tiled reductions:
  - accumulate only active expert tiles
  - reduce per tile, not full dense arrays
  - flush only touched ranges
- Use grouped GEMM formulations for LoRA gradient math:
  - `dB += U^T x dY`
  - `dU += dY x B^T`
  - `dA += X^T x dU`
  - `dX += dU x A^T`
- Keep the exact gradient accumulation semantics across token microchunks: the first reversed chunk zeros LoRA grads, earlier chunks accumulate.
- Keep route-weight gradient correctness.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 .venv/bin/python scripts/kt/bench_arm_sft_moe.py \
  --qlen 128 --top-k 8 --rank 8 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --threads 8 --check --backward \
  --json profiling_kt_codex_smoke/bench_stage8_backward_check.json

taskset -c 0-143 .venv/bin/python scripts/kt/bench_arm_sft_moe.py \
  --qlen 2048 --top-k 8 --rank 64 --hidden-size 2048 --intermediate-size 768 \
  --num-experts 128 --threads 8 --backward \
  --json profiling_kt_codex_smoke/bench_stage8_backward_q2048_r64.json
```

Memory and CPU sampling during a longer run:

```bash
PID=<python_or_torchrun_pid>
while kill -0 "$PID" 2>/dev/null; do
  date
  grep -E 'VmRSS|VmHWM|Cpus_allowed_list|Threads' /proc/"$PID"/status
  ps -L -o pid,tid,psr,pcpu,pmem,comm -p "$PID" | sort -k4 -nr | head -30
  numastat -p "$PID"
  sleep 5
done | tee profiling_kt_codex_smoke/stage8_proc_sampling.log
```

Profile gate:

- Tiny backward correctness passes.
- `backward_local_alloc_zero_ms`, `backward_thread_reduce_ms`, and `backward_grad_input_flush_ms` are visible and reduced.
- Wrapper `backward_sync_ms_last` is explained by native detailed timers within 10 percent.
- Peak CPU RSS does not increase versus baseline for the same shape.

## Stage 9: NUMA and Memory-Flow Optimization

Problem: once compute kernels improve, memory placement and bandwidth can become the limiter.

Implementation:

- Keep route buffers, packed inputs, forward caches, and LoRA weights on the NUMA node used by worker threads.
- Avoid first-touch by the wrong Python/caller thread.
- Add optional log fields for current CPU, NUMA node, and allocation sizes for major buffers.
- Evaluate huge pages only after correctness and affinity are fixed.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 numactl --cpunodebind=0 --membind=0 env \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
  CUTOFF_LEN=512 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_numa_node0 \
  scripts/lf/run_lf_lora_sft.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_numa_node0.log

taskset -c 0-143 numactl --cpunodebind=0,1 --membind=0,1 env \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
  CUTOFF_LEN=512 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_numa_all \
  scripts/lf/run_lf_lora_sft.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_numa_all.log
```

Profile gate:

- Chosen NUMA policy is faster or equal on source-profile timing.
- `numastat -p` does not show accidental remote-memory dominance for the selected policy.
- No affinity regression.

## Stage 10: Re-evaluate Chunking and Checkpointing

Problem: chunking and checkpointing are not bugs, but after native kernels improve they determine the end-to-end tradeoff between HBM, CPU memory, and runtime.

Implementation:

- Keep the route-rank guard. Do not remove `KT_ARM_SFT_MAX_ROUTE_RANK_WORK`.
- Sweep token chunk sizes after kernel stages: 1024, 2048, 4096, and full logical qlen only if route-rank and scratch gates permit it.
- Sweep checkpointing on/off if HBM allows. Checkpointing doubles forward native calls in this shape.
- Preserve correctness of native cache stack depth for microchunk backward.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
for chunk in 1024 2048 4096; do
  taskset -c 0-143 env \
    BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
    KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 \
    KT_ARM_SFT_TOKEN_CHUNK_SIZE="$chunk" \
    MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
    CUTOFF_LEN=1024 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
    OUT_DIR="profiling_kt_codex_smoke/kt_armbf16_chunk_${chunk}" \
    scripts/lf/run_lf_lora_sft.sh 2>&1 | tee "profiling_kt_codex_smoke/kt_armbf16_chunk_${chunk}.log"
done
```

Profile gate:

- Cache stack depth returns to zero after backward.
- `native_cache_save_count == native_cache_pop_count`.
- Best chunk size is selected by measured source-profile time and memory, not by assumption.

## Stage 11: GPU-Side Sanity Profiling

Problem: the slow ARM BF16 MoE path is CPU-side, but GPU profiling still helps verify that CUDA work is not the hidden blocker.

Use Nsight Systems for timeline sanity after a completed matching source profile:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 env \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=nsys \
  KT_ARM_SOURCE_OK_PROFILE_JSON=profiling_kt_codex_smoke/kt_armbf16_source_ok/profile.json \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 \
  MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
  CUTOFF_LEN=512 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_nsys_sanity \
  scripts/lf/run_lf_lora_sft.sh
```

Use NCU only for CUDA kernels outside the ARM CPU MoE hot path:

```bash
ncu --target-processes all --set full --launch-count 20 --force-overwrite \
  --export profiling_kt_codex_smoke/kt_armbf16_cuda_sanity \
  taskset -c 0-143 env BACKEND=kt_armbf16 PROFILE=0 \
    MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
    CUTOFF_LEN=128 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 \
    scripts/lf/run_lf_lora_sft.sh
```

Profile gate:

- Nsight shows CPU KT regions are the dominant MoE wait source only when native CPU timers also say so.
- NCU is not used to approve ARM BF16 kernel changes; it is only a guard against CUDA-side regressions.

## Stage 12: Full Regression and Large-Shape Acceptance

Run only after Stages 1 through 10 pass on small shapes.

Commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

.venv/bin/python -m pytest tests/lf/test_superoffload_backend_scripts.py -k "kt_arm" -q

taskset -c 0-143 env \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_TOKEN_CHUNK_SIZE=2048 KT_ARM_FIRST_STEP_TIMEOUT_SECONDS=0 \
  MODEL_NAME_OR_PATH=meta-llama/Llama-4-Scout-17B-16E \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.10 \
  MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final \
  scripts/lf/run_lf_lora_sft.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final.log
```

Acceptance gate:

- `profile.json` exists and is non-partial.
- `cpu_affinity_count >= KT_NUM_THREADS`.
- `base_kernel=sve_bfdot` appears in `train.log`.
- `kt.total_forward_calls` and `kt.total_backward_calls` match the expected microchunk/checkpoint count for the shape.
- Backward detailed timers explain wrapper backward time within 10 percent.
- Total measured step time improves materially versus the baseline 9.5 hour run.
- GPU memory remains explainable by KT CPU expert offload; do not require it to match ZeRO-3 offload memory.

## Optimization Backlog After v3 Acceptance

- Route-skew scheduling: split hottest experts into smaller tasks to improve load balance.
- Async overlap: overlap CPU KT forward/backward chunks with GPU shared expert/attention work only after single-stage timings are clean.
- Weight layout persistence: prepack all base and LoRA weights into the final GEMM-friendly layout once at load time.
- Touched-expert gradient format: store sparse/touched LoRA gradient updates internally, then flush dense gradients only for the optimizer contract.
- Optional ArmPL/oneDNN backend selection: make backend visible in `KT_ARM_SFT_PROFILE`, for example `base_kernel=armpl_bf16_gemm` or `base_kernel=sve_bfdot_microkernel`.
- CI smoke: add a tiny synthetic ARMBF16 SFT test that validates affinity guard, forward/backward correctness, cache depth, and profile field presence.

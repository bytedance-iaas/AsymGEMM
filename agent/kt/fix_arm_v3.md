# KT ARM BF16 SFT Fix Plan v3

This file is the staged execution plan for the slow `kt_armbf16` SFT path. Do not treat the large profiling workload itself as a bug: batch 4, sequence 7168, rank 64, 14 token chunks, 48 layers, and activation checkpointing is expected to be heavy. The remaining issues are launch/threading correctness, incomplete profiling, and native kernel structure.

Rule: do not move to the next stage until the validation gate for the current stage passes and the profile artifact is saved under `profiling_kt_codex_smoke/`.

## Current Facts

- The completed rank-64 run entered native KT ARM BF16, not DeepSpeed and not a scalar fallback. `train.log` reports `path=packed`, `task_dispatch=worker_pool`, `base_kernel=sve_bfdot`, and `aligned_weights=1`.
- The original accepted-looking rank-64 run recorded `cpu_affinity_count=1`, `cpu_affinity.cpus=0`, while the host shell can see 144 CPUs. This was a real launch/threading correctness issue, not a profiler artifact.
- The CPU-affinity collapse was reproduced with Qwen3 and traced to `OMP_PROC_BIND=close`: the process leader became pinned to CPU 0 after OpenMP initialized. The KT-only launcher now defaults `KT_ARM_OMP_PROC_BIND=false` and monitors the child process affinity while it runs.
- The fixed-affinity Qwen3 smoke records `cpu_affinity_count=144`, `OMP_PROC_BIND=false`, 48 wrappers, 48 forward calls, and 48 backward calls.
- KT totals for the completed rank-64 source profile: 48 wrappers, 1344 forward native calls, 672 backward native calls.
- Per 2048-token chunk, native forward averages approximately: base gate/up 5.6s, LoRA gate/up 6.1s, base down 3.8s, LoRA down 3.0s, route merge 0.31s. Copies to/from CUDA are small.
- Stage 3 timers now explain native backward sync within measurement noise: before the reducer fix, detailed native phases accounted for 99.96 percent of wrapper backward sync on `profiling_kt_codex_smoke/kt_armbf16_timer_coverage_qwen3`.
- The dominant validated bug was the dense backward reducer dispatching one WorkerPool task per float element. For Qwen3 rank-8 smoke this produced task counts of 131,072, 786,432, and 2,097,152 per reduction field and `backward_thread_reduce_ms ~= 4108 ms/layer`.
- Stage 8A changed dense reduction dispatch to contiguous chunks (`KT_ARM_SFT_REDUCE_CHUNK_ELEMS`, default 16384). The same Qwen3 smoke dropped `backward_thread_reduce_ms` to `8.45 ms/layer`, backward sync from `4233.50 ms/layer` to `129.63 ms/layer`, and one-step train runtime from `266.6s` to `83.1s`.
- Accepted KT profiles are now checked by `agent/kt/scripts/validate_kt_arm_profile.py`, which rejects partial profiles, wrong backend, wrong physical GPU, low CPU affinity, OpenMP bind collapse, missing native SVE BF16 log evidence, and missing KT forward/backward counters.
- KT microbench coverage exists under `../ktransformers/kt-kernel/bench/`, but it is still coarse. The existing benches now run and respect `--threads` for both WorkerPool and ARM backward OpenMP, but they do not yet provide per-phase masks or a full layout/PMU JSON report.
- Low GPU memory is not suspicious by itself. KT keeps routed expert work CPU-side; memory comparison against ZeRO-3 offload is not one-to-one.
- GPU policy for KT tests: use physical GPU 1 first, physical GPU 2 as fallback. Do not use GPU 0 or GPU 3 for KT validation. Current snapshot on 2026-06-12 showed GPU 1 with about 140 GiB free and GPU 2 with about 132 GiB free, which is enough for KT smoke tests and should be enough for the previously observed large KT profile shape.
- Model policy for KT-native SFT validation: use a model architecture supported by `kt_kernel.sft.arch`, currently DeepSeekV2/V3, Qwen2Moe/Qwen3Moe/Qwen3_5Moe, or Mixtral. `meta-llama/Llama-4-Scout-17B-16E` currently fails the KT SFT architecture gate with `KTAMXModelNotSupportedError`; do not use it as a positive KT-native smoke until explicit Llama4 support is implemented.

## Executed Validation Artifacts

- `profiling_kt_codex_smoke/kt_armbf16_gpu1_affinity_fixed_qwen3`: GPU 1, Qwen3, `CUTOFF_LEN=64`, batch 1, rank 8, max step 1. Validated native KT ARM BF16 path with `base_kernel=sve_bfdot`, `compiled_sve_bf16=1`, `sve_vector_bytes=16`, `aligned_weights=1`, and `cpu_affinity_count=144`.
- `profiling_kt_codex_smoke/kt_armbf16_timer_coverage_qwen3`: pre-reducer-fix timing coverage. Backward sync mean `4233.499 ms`, detailed native timing sum mean `4231.648 ms`, coverage `99.96%`. Root cause was `backward_thread_reduce_ms` mean `4108.213 ms`.
- `profiling_kt_codex_smoke/kt_armbf16_reduce_chunk_qwen3`: post-reducer-fix validation. Train runtime `83.1s`, backward sync mean `129.632 ms`, `backward_thread_reduce_ms` mean `8.454 ms`, `backward_route_loop_ms` mean `87.916 ms`, `backward_local_alloc_zero_ms` mean `31.345 ms`.
- `profiling_kt_codex_smoke/kt_armbf16_layout_pmu`: build/disassembly evidence. `bf16_instruction_hits.txt` is non-empty and contains BF16 dot instructions.
- `profiling_kt_codex_smoke/bench_armbf16_tiny_smoke.json`: `bench_armbf16_sft.py` tiny ARM smoke after bench thread fix. `KT_ARM_SFT_BACKWARD_SCRATCH` reports `threads=1` when `--threads 1`.
- `profiling_kt_codex_smoke/bench_arm_sft_compare_tiny_smoke.json`: `bench_arm_sft_compare.py` tiny correctness/latency smoke after fixing the stale `wrapper` reference in latency payload.
- `profiling_kt_codex_smoke/bench_arm_sft_compare_invalid_tiny_smoke.json`: invalid-route-pattern correctness/latency smoke with batched Torch enabled; verifies all reference modes skip invalid expert IDs consistently.
- Script regression: `bash -n agent/kt/scripts/run_lf_lora_sft_kt.sh agent/kt/scripts/profile_lora_lf_kt.sh` passes. `.venv/bin/python -m pytest tests/lf/test_superoffload_backend_scripts.py -k "kt_arm" -q` passes with `20 passed, 28 deselected`.
- Python syntax regression: `.venv/bin/python -m py_compile agent/kt/scripts/validate_kt_arm_profile.py ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py ../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py` passes.

## Remaining Bugs and Optimization Targets

1. Forward base projections still use a dot-loop shape rather than a blocked SVE BF16 GEMM. In the post-reducer Qwen3 smoke, forward is now the largest wall-time contributor: base gate/up mean `263.341 ms/layer`, base down mean `161.890 ms/layer`, route merge mean `52.512 ms/layer`.
2. Backward route recompute is now the main native backward compute target: `backward_route_loop_ms` mean `87.916 ms/layer`. It still recomputes route forward and backward with per-route loops.
3. Dense backward scratch still exists. Stage 8A fixed reduction task granularity, but per-thread dense full-expert gradient buffers still allocate/zero about `281 MB/layer` local scratch across 8 threads and `35 MB/layer` reduced scratch for the Qwen3 rank-8 smoke. Large rank-64 shapes will amplify this.
4. Backward-weight repack wait is visible but mostly overlapped outside current backward sync after the reducer fix. It still reports `backward_repack_wait_ms` around `0.94s/layer` in profile lines, so Stage 9/10 should verify overlap and avoid treating it as a hidden synchronous blocker.
5. `profile.json` now includes KT config and affinity fields, and the KT-only launcher has been patched to include physical `gpu_id`, `num_gpus`, `cuda_visible_devices`, `nvidia_visible_devices`, and `profile_nsys_gpu_metrics_devices` in future source profiles. Older artifacts still require checking `train.log` for `GPU_ID=1` or `GPU_ID=2`.
6. The shared AsymGEMM sweep script `scripts/lf/profile_lora_lf.sh` still defaults to GPU 0 and should not be used for accepted KT ARM profiles. Use `agent/kt/scripts/profile_lora_lf_kt.sh`, whose default `GPU_POOL` is `1,2`.
7. Script isolation does not freeze all runtime code: KT script copies still import the live AsymGEMM checkout through `PYTHONPATH`. For accepted performance comparisons, record `git status --short`, relevant diffs, and the `../ktransformers/kt-kernel` source revision or use a separate frozen worktree.
8. Llama4 remains unsupported by KT SFT. Continue using Qwen3 for positive KT-native tests unless `kt_kernel.sft.arch` is extended and validated.

Useful source locations:

- Original launcher and profiling env: `scripts/lf/run_lf_lora_sft.sh`
- Original sweep wrapper: `scripts/lf/profile_lora_lf.sh`
- KT-only launcher copy to edit during this plan: `agent/kt/scripts/run_lf_lora_sft_kt.sh`
- KT-only sweep copy to edit during this plan: `agent/kt/scripts/profile_lora_lf_kt.sh`
- KT accepted-profile validator: `agent/kt/scripts/validate_kt_arm_profile.py`
- ARM Python wrapper: `../ktransformers/kt-kernel/python/sft/arm.py`
- Autograd/checkpoint hook: `../ktransformers/kt-kernel/python/sft/autograd.py`
- LF KT layer bridge: `../ktransformers/kt-kernel/python/sft/layer.py`
- Native ARM SFT kernel: `../ktransformers/kt-kernel/operators/arm/bf16_sft_moe.hpp`
- Worker pool: `../ktransformers/kt-kernel/cpu_backend/worker_pool.cpp`
- Existing coarse KT ARM bench: `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py`
- Existing correctness/latency compare bench: `../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py`

Root causes this plan targets:

1. Invalid launch placement: prior accepted-looking profiles had `cpu_affinity_count=1`.
2. Worker 0 placement: the caller thread participates in worker-pool jobs and can be badly pinned.
3. Backward reduction task granularity: dense reductions previously dispatched one worker-pool task per float element. Stage 8A fixed this with chunked contiguous reductions, but dense scratch remains.
4. Base projection shape: `arm_bf16_matmul_tiled` is currently a nested loop around one SVE BF16 dot per output element, not a blocked GEMM.
5. Base down shape: down projection is scalar FP32 accumulation over `H x I`.
6. LoRA shape: gate/up/down LoRA paths are per-route scalar loops instead of grouped rank GEMMs.
7. Backward memory traffic: dense full-expert per-thread gradient buffers and dense reductions create large allocation, zeroing, cache, and memory-bandwidth costs.
8. Layout/NUMA uncertainty: existing logs do not prove first-touch placement, contiguous kernel access, or per-phase bytes/FLOPs strongly enough.

Required evidence before any optimization is accepted:

- Correctness: tiny-shape Torch/scalar reference match for forward and backward when the stage changes math.
- Validator: `agent/kt/scripts/validate_kt_arm_profile.py` passes for any LF source profile accepted as complete.
- Launch: `train.log` proves `GPU_ID=1` or `GPU_ID=2`, `NUM_GPUS=1`, `cpu_affinity_count >= KT_NUM_THREADS`, and no accepted run uses GPU 0 or GPU 3.
- Instruction path: build manifest, `cpu_features.txt`, `objdump.txt`, and `bf16_instruction_hits.txt` prove SVE BF16 code is compiled and present.
- Layout: `layout_report.json` records SVE vector bytes, padded LoRA rank, route tile shape, base/LoRA strides, 64-byte alignment, route skew, and padding overhead.
- Phase timing: source profile or microbench JSON records each target phase separately, not only total wrapper time.
- PMU profile: `perf_stat_*.txt` and `perf_*_report.txt` show whether the phase is instruction-bound, cache-bound, branch-heavy, or dominated by scalar conversion/dot-loop symbols.
- Memory placement: NUMA artifacts and native allocation logs show first-touch CPU/node, pointer alignment, major buffer sizes, and reuse.
- Final LF validation: the same large-shape source profile improves materially versus the baseline, with KT forward/backward call counts matching expected chunk/checkpoint counts.

## Stage 0A: Isolate AsymGEMM Launcher Scripts

Goal: keep KT profiling and launcher edits out of the normal AsymGEMM script path while KT ARM work is active.

Isolation contract:

- KT implementation edits belong under `../ktransformers/kt-kernel/**`.
- KT launcher/profile edits belong under ignored `agent/kt/**`.
- KT run artifacts belong under `profiling_kt_codex_smoke/kt_armbf16_*`.
- Do not edit the original AsymGEMM launcher/sweep files during KT work: `scripts/lf/run_lf_lora_sft.sh` and `scripts/lf/profile_lora_lf.sh`.
- Do not use `scripts/lf/profile_lora_lf.sh` for accepted KT ARM profiles. Its default GPU pool is not the KT-safe `1,2` policy.
- Do not edit shared AsymGEMM helper scripts for KT unless the change is intentionally part of the main AsymGEMM work. If those helpers change while KT profiling is running, KT profiles may reflect those runtime changes.
- For file-change isolation, the ignored `agent/kt/scripts` copies are enough. For hard runtime isolation from all main AsymGEMM changes, run KT from a separate frozen AsymGEMM worktree/clone and point `ASYM_DIR`/`ROOT` there.
- For accepted performance profiles in this checkout, save `git status --short`, relevant `git diff` snippets, and the KT kernel source diff beside the profile so concurrent AsymGEMM edits are visible.

Implementation:

- Copy `scripts/lf/run_lf_lora_sft.sh` to `agent/kt/scripts/run_lf_lora_sft_kt.sh`.
- Copy `scripts/lf/profile_lora_lf.sh` to `agent/kt/scripts/profile_lora_lf_kt.sh`.
- In `profile_lora_lf_kt.sh`, set `RUN_LF_SCRIPT="${ASYM_DIR}/agent/kt/scripts/run_lf_lora_sft_kt.sh"` so KT sweeps call the KT-only launcher copy.
- Optionally copy helper scripts into `agent/kt/scripts/...` and patch the KT copies to call those helper copies. This is recommended if the main AsymGEMM helper scripts are actively changing.
- Make all launcher/profiling changes from this plan in the KT copies first.
- Do not edit the original `run_lf_lora_sft.sh` or `profile_lora_lf.sh` until the KT-only path passes the acceptance gates and is ready to upstream.

Commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

mkdir -p agent/kt/scripts/lf agent/kt/scripts/lora agent/kt/scripts/plotting
cp scripts/lf/run_lf_lora_sft.sh agent/kt/scripts/run_lf_lora_sft_kt.sh
cp scripts/lf/profile_lora_lf.sh agent/kt/scripts/profile_lora_lf_kt.sh

# Optional helper copies for stronger runtime isolation from scripts/ changes.
cp scripts/lf/run_lf_profiled_train.py agent/kt/scripts/lf/run_lf_profiled_train.py
cp scripts/lf/postprocess_lf_profile_artifacts.py agent/kt/scripts/lf/postprocess_lf_profile_artifacts.py
cp scripts/lf/check_superoffload_run.py agent/kt/scripts/lf/check_superoffload_run.py
cp scripts/lf/check_deepspeed_cpuadam_run.py agent/kt/scripts/lf/check_deepspeed_cpuadam_run.py
cp scripts/lf/build_lf_sft_eval_pair.py agent/kt/scripts/lf/build_lf_sft_eval_pair.py
cp scripts/lf/validate_lf_memory_capacity_schema.py agent/kt/scripts/lf/validate_lf_memory_capacity_schema.py
cp scripts/lora/postprocess_nsys_lora.py agent/kt/scripts/lora/postprocess_nsys_lora.py
cp scripts/plotting/plot_activation_recompute_sweep.py agent/kt/scripts/plotting/plot_activation_recompute_sweep.py
cp scripts/plotting/plot_lf_memory_breakdown.py agent/kt/scripts/plotting/plot_lf_memory_breakdown.py
cp scripts/plotting/plot_lf_interconnect_ctc.py agent/kt/scripts/plotting/plot_lf_interconnect_ctc.py

python3 - <<'PY'
from pathlib import Path
replacements = {
    Path("agent/kt/scripts/profile_lora_lf_kt.sh"): {
        'RUN_LF_SCRIPT="${ASYM_DIR}/scripts/lf/run_lf_lora_sft.sh"':
            'RUN_LF_SCRIPT="${ASYM_DIR}/agent/kt/scripts/run_lf_lora_sft_kt.sh"',
        'GPU_POOL=${GPU_POOL:-0}':
            'GPU_POOL=${GPU_POOL:-1,2}',
        'BUILD_DATASET_SCRIPT="${ASYM_DIR}/scripts/lf/build_lf_sft_eval_pair.py"':
            'BUILD_DATASET_SCRIPT="${ASYM_DIR}/agent/kt/scripts/lf/build_lf_sft_eval_pair.py"',
        'PROFILE_POSTPROCESS_SCRIPT="${ASYM_DIR}/scripts/lf/postprocess_lf_profile_artifacts.py"':
            'PROFILE_POSTPROCESS_SCRIPT="${ASYM_DIR}/agent/kt/scripts/lf/postprocess_lf_profile_artifacts.py"',
        'MEMORY_SCHEMA_VALIDATOR="${ASYM_DIR}/scripts/lf/validate_lf_memory_capacity_schema.py"':
            'MEMORY_SCHEMA_VALIDATOR="${ASYM_DIR}/agent/kt/scripts/lf/validate_lf_memory_capacity_schema.py"',
        'PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_activation_recompute_sweep.py"':
            'PLOT_SCRIPT="${ASYM_DIR}/agent/kt/scripts/plotting/plot_activation_recompute_sweep.py"',
        'MEMORY_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_memory_breakdown.py"':
            'MEMORY_PLOT_SCRIPT="${ASYM_DIR}/agent/kt/scripts/plotting/plot_lf_memory_breakdown.py"',
        'INTERCONNECT_PLOT_SCRIPT="${ASYM_DIR}/scripts/plotting/plot_lf_interconnect_ctc.py"':
            'INTERCONNECT_PLOT_SCRIPT="${ASYM_DIR}/agent/kt/scripts/plotting/plot_lf_interconnect_ctc.py"',
    },
    Path("agent/kt/scripts/run_lf_lora_sft_kt.sh"): {
        'CHECK_SUPEROFFLOAD_SCRIPT=${CHECK_SUPEROFFLOAD_SCRIPT:-${ASYM_DIR}/scripts/lf/check_superoffload_run.py}':
            'CHECK_SUPEROFFLOAD_SCRIPT=${CHECK_SUPEROFFLOAD_SCRIPT:-${ASYM_DIR}/agent/kt/scripts/lf/check_superoffload_run.py}',
        'CHECK_CPUADAM_SCRIPT=${CHECK_CPUADAM_SCRIPT:-${ASYM_DIR}/scripts/lf/check_deepspeed_cpuadam_run.py}':
            'CHECK_CPUADAM_SCRIPT=${CHECK_CPUADAM_SCRIPT:-${ASYM_DIR}/agent/kt/scripts/lf/check_deepspeed_cpuadam_run.py}',
        'PROFILE_LAUNCHER=${PROFILE_LAUNCHER:-${ASYM_DIR}/scripts/lf/run_lf_profiled_train.py}':
            'PROFILE_LAUNCHER=${PROFILE_LAUNCHER:-${ASYM_DIR}/agent/kt/scripts/lf/run_lf_profiled_train.py}',
        'PROFILE_NSYS_POSTPROCESS_SCRIPT=${PROFILE_NSYS_POSTPROCESS_SCRIPT:-${ASYM_DIR}/scripts/lora/postprocess_nsys_lora.py}':
            'PROFILE_NSYS_POSTPROCESS_SCRIPT=${PROFILE_NSYS_POSTPROCESS_SCRIPT:-${ASYM_DIR}/agent/kt/scripts/lora/postprocess_nsys_lora.py}',
        'PROFILE_POSTPROCESS_SCRIPT=${PROFILE_POSTPROCESS_SCRIPT:-${ASYM_DIR}/scripts/lf/postprocess_lf_profile_artifacts.py}':
            'PROFILE_POSTPROCESS_SCRIPT=${PROFILE_POSTPROCESS_SCRIPT:-${ASYM_DIR}/agent/kt/scripts/lf/postprocess_lf_profile_artifacts.py}',
    },
}
for path, reps in replacements.items():
    text = path.read_text()
    for old, new in reps.items():
        if old not in text:
            raise SystemExit(f"missing expected line in {path}: {old}")
        text = text.replace(old, new, 1)
    path.write_text(text)
PY

bash -n agent/kt/scripts/run_lf_lora_sft_kt.sh
bash -n agent/kt/scripts/profile_lora_lf_kt.sh
.venv/bin/python -m py_compile \
  agent/kt/scripts/validate_kt_arm_profile.py \
  agent/kt/scripts/lf/build_lf_sft_eval_pair.py \
  agent/kt/scripts/plotting/plot_lf_memory_breakdown.py \
  agent/kt/scripts/plotting/plot_activation_recompute_sweep.py
```

Validation gate:

- `bash -n` passes for both KT-only scripts.
- `rg -n "run_lf_lora_sft_kt.sh" agent/kt/scripts/profile_lora_lf_kt.sh` shows the sweep copy calls the KT launcher copy.
- `rg -n "agent/kt/scripts" agent/kt/scripts/run_lf_lora_sft_kt.sh agent/kt/scripts/profile_lora_lf_kt.sh` shows helper paths point to KT copies if helper-copy mode is used.
- `rg -n 'GPU_POOL=\$\{GPU_POOL:-1,2\}' agent/kt/scripts/profile_lora_lf_kt.sh` proves the KT sweep defaults to physical GPU 1/2, not GPU 0.
- `git check-ignore -v --no-index agent/kt/scripts/run_lf_lora_sft_kt.sh agent/kt/scripts/profile_lora_lf_kt.sh agent/kt/scripts/validate_kt_arm_profile.py` proves new KT script copies are ignored by normal git status.
- `py_compile` passes for the validator and copied helper scripts.
- If the original AsymGEMM scripts are already dirty from unrelated work, leave them dirty and do not touch them. Record `git diff -- scripts/lf/run_lf_lora_sft.sh scripts/lf/profile_lora_lf.sh` before and after KT work; the KT pass must not add new hunks to those files.
- All commands in this document use `agent/kt/scripts/run_lf_lora_sft_kt.sh` after this stage.
- Any final upstreaming into the original scripts must be a separate integration step after Stage 12.

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

## Stage 0B: GPU and ARM Instruction Preflight

Goal: every KT test uses GPU 1 or GPU 2 only, and every native result proves the ARM SVE BF16 instruction path and layout assumptions before timing is trusted.

Implementation:

- In `agent/kt/scripts/run_lf_lora_sft_kt.sh`, add a KT-only GPU guard:
  - for `BACKEND=kt_armbf16`, require `GPU_ID` to be `1` or `2`
  - require `NUM_GPUS=1`
  - require `PROFILE_NSYS_GPU_METRICS_DEVICES` to match `GPU_ID` for Nsight runs
  - allow a temporary override only with `KT_ARM_ALLOW_GPU_0_OR_3=1`, and never use that override for accepted profiles
- In `agent/kt/scripts/profile_lora_lf_kt.sh`, default `GPU_POOL=1,2`.
- For direct Python microbenches that import torch, set `CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1` unless explicitly testing GPU 2.
- Add native profile fields or startup logs for:
  - `sve_vector_bytes=svcntb()`
  - `base_kernel=sve_bfdot`
  - `compiled_sve_bf16=1`
  - base and LoRA weight layouts/strides
  - `aligned_weights=1`
  - packed route order, active expert count, hottest expert, and route skew

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader,nounits

# Accept GPU 1 or GPU 2 only for KT runs. This should fail once the guard exists.
taskset -c 0-143 env \
  GPU_ID=0 NUM_GPUS=1 BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=/tmp/kt_bad_gpu0 \
  agent/kt/scripts/run_lf_lora_sft_kt.sh

# This is the default KT GPU path.
taskset -c 0-143 env \
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_gpu1_preflight \
  agent/kt/scripts/run_lf_lora_sft_kt.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_gpu1_preflight.log

rg "GPU_ID|CUDA_VISIBLE_DEVICES|base_kernel=sve_bfdot|aligned_weights=1|sve_vector_bytes" \
  profiling_kt_codex_smoke/kt_armbf16_gpu1_preflight.log

.venv/bin/python agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json profiling_kt_codex_smoke/kt_armbf16_gpu1_preflight/profile.json \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 \
  --expected-batch 1 \
  --expected-rank 8 \
  --expected-dropout 0.0 \
  --expected-top-k 8 \
  --expected-cache-depth 2 \
  --expected-recompute false \
  --require-final

# Verify the extension was built for ARM SVE BF16 and contains the expected instruction text.
SO="$(
  .venv/bin/python - <<'PY'
from kt_kernel import kt_kernel_ext
print(kt_kernel_ext.__file__)
PY
)"
echo "$SO"
if command -v llvm-objdump >/dev/null 2>&1; then
  llvm-objdump -d "$SO" | rg -i "bfdot|bfmmla|svbfdot" | head -20
else
  objdump -d "$SO" | rg -i "bfdot|bfmmla|svbfdot" | head -20
fi
```

Profile gate:

- Accepted KT source profiles must have `GPU_ID=1` or `GPU_ID=2` in `train.log`; default is GPU 1.
- `nvidia-smi` before launch shows the selected GPU has enough free memory for the shape.
- `train.log` has `base_kernel=sve_bfdot`, `aligned_weights=1`, and `sve_vector_bytes` once that field is added.
- `validate_kt_arm_profile.py` passes. It is allowed to infer physical GPU from `train.log` for older artifacts, but new artifacts should also contain `config.gpu_id`, `config.num_gpus`, and CUDA visibility fields.
- Disassembly or build log proves the extension contains SVE BF16 code. If disassembly cannot be used, the build log must show `-march=...+sve+bf16` or `-march=native` on a host with `svebf16`.

## Stage 1: Fix Launch CPU Affinity

Problem: the completed run requested 8 KT threads but recorded only CPU `0` in its affinity mask. This can make KT appear stuck and invalidates performance conclusions.

Implementation:

- Add a launcher preflight for `BACKEND=kt_armbf16` in `agent/kt/scripts/run_lf_lora_sft_kt.sh`.
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
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 MAX_STEPS=1 \
  OUT_DIR=/tmp/kt_affinity_negative \
  agent/kt/scripts/run_lf_lora_sft_kt.sh

# Positive smoke: this must complete and profile must report enough CPUs.
taskset -c 0-143 env \
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=64 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_affinity_positive \
  agent/kt/scripts/run_lf_lora_sft_kt.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_affinity_positive.log

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

Current status:

- The reproduced placement failure was caused by `OMP_PROC_BIND=close`, which pinned the process leader to CPU 0 after OpenMP initialization.
- Stage 1 fixed the accepted LF launcher path by defaulting `KT_ARM_OMP_PROC_BIND=false` and monitoring child affinity during training.
- No WorkerPool caller-thread binding change has been accepted yet. Treat this stage as a targeted follow-up only if `ps -L`, WorkerPool logs, or source timings still show caller-thread placement imbalance after the Stage 1 fix.

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
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=128 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_worker_place \
  agent/kt/scripts/run_lf_lora_sft_kt.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_worker_place.log

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
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_NATIVE_PROGRESS=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=128 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_timer_coverage \
  agent/kt/scripts/run_lf_lora_sft_kt.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_timer_coverage.log

rg "backward_(ensure_buffers|local_alloc_zero|thread_reduce|grad_input_flush)_ms" \
  profiling_kt_codex_smoke/kt_armbf16_timer_coverage.log
```

Profile gate:

- For each backward call, the sum of detailed native backward phases should be within 10 percent of wrapper `backward_sync_ms_last`, excluding explicit Python/CUDA copy fields.
- This gate passed on `profiling_kt_codex_smoke/kt_armbf16_timer_coverage_qwen3`: wrapper backward sync mean `4233.499 ms`, detailed native backward phase sum mean `4231.648 ms`, coverage `99.96%`.
- The same artifact identified `backward_thread_reduce_ms` mean `4108.213 ms/layer` as the validated blocker before Stage 8A.
- If the gap is larger than 10 percent, do not optimize kernels yet; add missing timers first.

## Stage 4: Add a Native Microbenchmark Harness

Problem: the end-to-end LF run is too expensive to use as the only kernel development loop.

Current status:

- Use `../ktransformers/kt-kernel/bench/bench_armbf16_sft.py` for coarse ARM/Torch wrapper correctness and latency.
- Use `../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py` for scalar reference, grouped Torch, KT Torch BF16, and KT ARM BF16 comparisons.
- `bench_arm_sft_compare.py` had a stale `wrapper` reference in `_latency_payload`; this is fixed.
- Both benches now set `KT_ARM_SFT_BACKWARD_THREADS`, `OMP_NUM_THREADS`, and `OMP_PROC_BIND=false` from `--threads` while running the ARM backend, so `--threads` controls both WorkerPool and ARM backward OpenMP.
- Remaining harness gap: neither existing bench has per-phase masks, dedicated layout JSON, bytes/FLOPs estimates, or PMU artifact orchestration. Do not accept Stage 5/6/7/8B kernel rewrites on these coarse bench latencies alone.

Implementation:

- Extend one existing bench, preferably `bench_arm_sft_compare.py`, or add a small C++/Python-bound harness that constructs one ARM SFT MoE wrapper/layer with synthetic BF16 inputs, synthetic top-k IDs/weights, and rank/top-k/hidden/intermediate matching the model.
- The harness must support phase masks so each slow component can be isolated:
  - base gate/up only
  - base down only
  - LoRA gate/up only
  - LoRA down only
  - route merge only
  - backward route loop only
  - backward reduction/flush only
- Benchmark at least these shapes:
  - `qlen=128, topk=8, rank=8`
  - `qlen=512, topk=8, rank=8`
  - `qlen=2048, topk=8, rank=64`
- Emit JSON with phase times, effective bandwidth estimates, estimated FLOPs, achieved GFLOP/s, bytes touched, arithmetic intensity, thread count, affinity, NUMA placement, route skew, and checksum.
- Add a correctness mode comparing native output/grads against a simple Torch or scalar reference for tiny shapes.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
.venv/bin/python -m py_compile \
  ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  ../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py

taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  .venv/bin/python ../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py \
  --qlen 2 --experts 2 --topk 1 --hidden 16 --intermediate 8 --rank 2 \
  --threads 1 --warmup 0 --iters 1 \
  --output-json profiling_kt_codex_smoke/bench_arm_sft_compare_tiny_smoke.json

taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 KT_ARM_SFT_PROFILE=1 \
  .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend both \
  --qlen 128 --topk 8 --rank 8 --hidden 2048 --intermediate 768 \
  --experts 128 --threads 8 --warmup 1 --iters 3 \
  --output-json profiling_kt_codex_smoke/bench_armbf16_q128_r8.json

taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 KT_ARM_SFT_PROFILE=1 \
  .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness \
  --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
  --experts 128 --threads 8 --warmup 1 --iters 3 \
  --output-json profiling_kt_codex_smoke/bench_armbf16_q2048_r64.json
```

Profile gate:

- Tiny correctness mode passes against reference within BF16/FP32 tolerance.
- `KT_ARM_SFT_BACKWARD_SCRATCH` logs `threads=<--threads>`, not host CPU count, for bench runs.
- JSON includes native diagnostic fields such as `base_projection_kernel`, `sve_bf16_compiled`, `last_forward_path`, `last_task_dispatch`, `active_expert_count`, route counts, and cache depth.
- Existing coarse microbench timings trend in the same direction as LF source-profile timings.
- Before accepting a per-phase kernel optimization, the harness must be extended to emit the missing phase-mask/layout/bytes/FLOPs fields, or an equivalent source-profile plus `perf` artifact set must be saved.

## Stage 4A: Instruction, Layout, and PMU Baseline

Problem: `base_kernel=sve_bfdot` only proves the dot primitive is compiled. It does not prove the hot loops are using a high-throughput SVE BF16 GEMM shape or a cache-friendly layout.

Implementation:

- Add a layout report mode to the microbench or native profile. It must emit:
  - `sve_vector_bytes`
  - `padded_lora_rank`
  - `route_tile_m`
  - base gate/up layout and strides
  - base down layout and strides
  - LoRA A/B layouts and strides
  - whether each phase consumes contiguous rows, transposed weights, and 64-byte aligned buffers
  - active expert histogram, hottest expert, route skew, padded routes, and padding overhead
- Add a build manifest artifact for every native change:
  - compiler ID/version
  - `CMAKE_BUILD_TYPE`
  - full CXX flags
  - whether `-march=...+sve+bf16` or `-march=native` was used
  - `/proc/cpuinfo` feature lines proving `sve` and `svebf16`
- Add `perf stat` and `perf record` runs on the phase microbench. Use generic events first because vendor-specific ARM PMU event names vary by kernel:
  - `cycles`
  - `instructions`
  - `cache-references`
  - `cache-misses`
  - `branches`
  - `branch-misses`
  - `task-clock`
  - `context-switches`
  - `cpu-migrations`
- If vendor events are available, add SVE/FP pipeline and memory bandwidth events after checking `perf list`.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM

mkdir -p profiling_kt_codex_smoke/kt_armbf16_layout_pmu
lscpu > profiling_kt_codex_smoke/kt_armbf16_layout_pmu/lscpu.txt
grep -m1 -i '^Features' /proc/cpuinfo > profiling_kt_codex_smoke/kt_armbf16_layout_pmu/cpu_features.txt
perf list > profiling_kt_codex_smoke/kt_armbf16_layout_pmu/perf_list.txt 2>&1 || true

SO="$(
  .venv/bin/python - <<'PY'
from kt_kernel import kt_kernel_ext
print(kt_kernel_ext.__file__)
PY
)"
echo "$SO" > profiling_kt_codex_smoke/kt_armbf16_layout_pmu/kt_kernel_ext_so.txt
readelf -A "$SO" > profiling_kt_codex_smoke/kt_armbf16_layout_pmu/readelf_A.txt 2>&1 || true
if command -v llvm-objdump >/dev/null 2>&1; then
  llvm-objdump -d "$SO" > profiling_kt_codex_smoke/kt_armbf16_layout_pmu/objdump.txt
else
  objdump -d "$SO" > profiling_kt_codex_smoke/kt_armbf16_layout_pmu/objdump.txt
fi
rg -i "bfdot|bfmmla|svbfdot" profiling_kt_codex_smoke/kt_armbf16_layout_pmu/objdump.txt \
  | tee profiling_kt_codex_smoke/kt_armbf16_layout_pmu/bf16_instruction_hits.txt

# Temporary coarse layout/diagnostic probe until the Stage 4 phase/layout extension exists.
taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 KT_ARM_SFT_PROFILE=1 \
  .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
    --backend arm --skip-correctness \
    --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
    --experts 128 --threads 8 --warmup 1 --iters 3 \
    --output-json profiling_kt_codex_smoke/kt_armbf16_layout_pmu/layout_probe.json \
    2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_layout_pmu/layout_probe.log

command -v perf && perf stat -r 3 \
  -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,context-switches,cpu-migrations \
  -o profiling_kt_codex_smoke/kt_armbf16_layout_pmu/perf_stat_armbf16_wrapper.txt -- \
  taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 KT_ARM_SFT_PROFILE=1 \
    .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
      --backend arm --skip-correctness \
      --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
      --experts 128 --threads 8 --warmup 1 --iters 3 \
      --output-json profiling_kt_codex_smoke/kt_armbf16_layout_pmu/armbf16_wrapper.json

command -v perf && perf record -g --call-graph dwarf \
  -o profiling_kt_codex_smoke/kt_armbf16_layout_pmu/perf_armbf16_wrapper.data -- \
  taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 KT_ARM_SFT_PROFILE=1 \
    .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
      --backend arm --skip-correctness \
      --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
      --experts 128 --threads 8 --warmup 1 --iters 3 \
      --output-json /tmp/kt_armbf16_wrapper_profile.json

command -v perf && perf report --stdio \
  -i profiling_kt_codex_smoke/kt_armbf16_layout_pmu/perf_armbf16_wrapper.data \
  > profiling_kt_codex_smoke/kt_armbf16_layout_pmu/perf_armbf16_wrapper_report.txt
```

Profile gate:

- `cpu_features.txt` contains `sve` and `svebf16`.
- `bf16_instruction_hits.txt` is non-empty.
- The current coarse `layout_probe.log` must at least show `base_kernel=sve_bfdot`, `compiled_sve_bf16=1`, `sve_vector_bytes`, `route_tile_m`, `padded_lora_rank`, route counts, and `aligned_weights=1`.
- The full Stage 4A gate is not complete until `layout_report.json` exists and proves 64-byte alignment plus each phase's memory layout and strides.
- Before Stage 5, `perf` should show the dot-loop/matmul functions near the top for the ARM BF16 wrapper; after Stage 5 they should no longer dominate as one-dot-per-output loops.
- Each kernel optimization stage must save its `layout_report.json` or temporary `layout_probe.log`, `perf_stat_*.txt`, and `perf_*_report.txt` next to the phase JSON.

## Stage 5: Replace Base Gate/Up With Real Grouped BF16 GEMM

Problem: `arm_bf16_matmul_tiled` is currently nested loops around a single `svbfdot` dot product. That is not a high-throughput GEMM.

Implementation options, in order:

1. Use a proven ARM BF16 GEMM library if available on the target host: Arm Performance Libraries, oneDNN ACL backend, OpenBLAS only if it has relevant BF16 support, or KML if usable through existing build knobs.
2. If no library is acceptable, implement a blocked SVE BF16 microkernel for `MxN x K` with register blocking across multiple output columns and rows, not one dot per output element.
3. Batch by expert: route packing already groups routes per expert, so call one grouped GEMM per active expert for gate and up.
4. Parallelize across active experts and output blocks for large route sets; the current serial active-expert loop leaves most CPU cores unused in small-per-expert route distributions.
5. Consider fusing gate/up input packing or at least reusing the same packed input stream for both projections so the two base projections do not reload the same route input independently.

Validation commands:

```bash
# Build after kernel change.
cd /workspace/AsymGEMM-SFT/third_party/ktransformers/kt-kernel
CPUINFER_FORCE_REBUILD=1 CPUINFER_BUILD_TYPE=RelWithDebInfo CPUINFER_PARALLEL=16 \
  /workspace/AsymGEMM-SFT/third_party/AsymGEMM/.venv/bin/python -m pip install -e . -v --no-build-isolation

# Microbench before LF.
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  KT_ARM_SFT_PROFILE=1 \
  .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness \
  --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
  --experts 128 --threads 8 --warmup 1 --iters 3 \
  --output-json profiling_kt_codex_smoke/bench_stage5_gate_up.json \
  2>&1 | tee profiling_kt_codex_smoke/bench_stage5_gate_up.log

# Optional CPU PMU profile if perf is installed and permitted.
command -v perf && perf stat -d -d -d -- \
  taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
    KT_ARM_SFT_PROFILE=1 \
    .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
    --backend arm --skip-correctness \
    --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
    --experts 128 --threads 8 --warmup 1 --iters 3 \
    --output-json /tmp/bench_stage5_gate_up.json
```

Profile gate:

- Correctness mode still passes.
- `base_gate_up_ms` improves materially on `qlen=2048, rank=64`; target first milestone is at least 2x faster than the existing dot-loop baseline.
- `perf_stat_base_gate_up.txt` shows fewer instructions and cycles per output element than Stage 4A.
- `perf_base_gate_up_report.txt` no longer attributes most time to one-dot-per-output `arm_bf16_dot` calls.
- Disassembly of the new hot symbol shows a blocked SVE BF16 kernel using vector-width work, not scalar BF16-to-FP32 inner loops.
- `layout_report.json` shows gate/up weights and packed inputs are consumed with contiguous or deliberately packed strides for the new kernel.
- Source profile proves active-expert scheduling is not serially bottlenecked by one hot or many tiny expert batches.
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
taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  KT_ARM_SFT_PROFILE=1 \
  .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness \
  --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
  --experts 128 --threads 8 --warmup 1 --iters 3 \
  --output-json profiling_kt_codex_smoke/bench_stage6_down.json \
  2>&1 | tee profiling_kt_codex_smoke/bench_stage6_down.log
```

Profile gate:

- Correctness mode still passes.
- `base_down_ms` improves materially; target first milestone is at least 2x faster than scalar baseline.
- `perf_stat_base_down.txt` shows fewer instructions and cycles per output element than Stage 4A.
- `perf_base_down_report.txt` shows the new grouped down GEMM path, not scalar `bf16_to_f32` loops over `H x I`, as the dominant symbol.
- `layout_report.json` proves the down weight layout matches the kernel access pattern and avoids per-output strided gathers.
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
  taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
    KT_ARM_SFT_PROFILE=1 \
    .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
    --backend arm --skip-correctness \
    --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
    --experts 128 --threads 8 --lora-dropout "$dropout" \
    --warmup 1 --iters 3 \
    --output-json "profiling_kt_codex_smoke/bench_stage7_lora_drop${dropout}.json" \
    2>&1 | tee "profiling_kt_codex_smoke/bench_stage7_lora_drop${dropout}.log"
done
```

Profile gate:

- Correctness passes for dropout 0.00 and 0.10.
- `lora_gate_up_ms + lora_down_ms` improves materially; target first milestone is at least 2x faster than current per-route loops.
- Dropout 0.10 is not more than 25 percent slower than dropout 0.00 unless profiling proves RNG/mask generation dominates.
- `perf_stat_lora_*.txt` shows lower instruction count per route-rank operation than the per-route baseline.
- `layout_report.json` proves LoRA A/B are consumed in GEMM-friendly orientation and that padded rank does not create excessive padding overhead.
- The profile reports separate `dropout_mask_ms` or equivalent if dropout remains a meaningful cost.

## Stage 8: Redesign Backward Scratch and Reductions

Problem: backward allocates dense full-expert LoRA gradient buffers per thread, then reduces dense buffers. This creates large memory traffic and hides time outside the existing timers.

Stage 8A completed:

- `reduce_vector_fields()` now reduces contiguous chunks instead of dispatching one WorkerPool task per float.
- The chunk size is controlled by `KT_ARM_SFT_REDUCE_CHUNK_ELEMS` and defaults to `16384`.
- Validation artifact: `profiling_kt_codex_smoke/kt_armbf16_reduce_chunk_qwen3`.
- Result on the same Qwen3 smoke: `backward_thread_reduce_ms` dropped from `4108.213 ms/layer` to `8.454 ms/layer`; wrapper `backward_sync_ms` dropped from `4233.499 ms/layer` to `129.632 ms/layer`; one-step runtime dropped from `266.6s` to `83.1s`.
- This fixed the “stuck after increasing length/batch” symptom caused by pathological task granularity in backward reduction.

Remaining Stage 8B work:

- Dense per-thread full-expert scratch still exists.
- Backward route loop is now the primary native backward compute target.
- The next optimization must reduce only active expert slices or tiles and use grouped GEMM-style LoRA gradient math.

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
taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  .venv/bin/python ../ktransformers/kt-kernel/bench/bench_arm_sft_compare.py \
  --qlen 4 --experts 4 --topk 2 --hidden 64 --intermediate 32 --rank 4 \
  --threads 4 --warmup 0 --iters 1 \
  --output-json profiling_kt_codex_smoke/bench_stage8_backward_check.json

taskset -c 0-143 env CUDA_VISIBLE_DEVICES=1 NVIDIA_VISIBLE_DEVICES=1 \
  KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_REDUCE_CHUNK_ELEMS=16384 \
  .venv/bin/python ../ktransformers/kt-kernel/bench/bench_armbf16_sft.py \
  --backend arm --skip-correctness \
  --qlen 2048 --topk 8 --rank 64 --hidden 2048 --intermediate 768 \
  --experts 128 --threads 8 --warmup 1 --iters 3 \
  --output-json profiling_kt_codex_smoke/bench_stage8_backward_q2048_r64.json \
  2>&1 | tee profiling_kt_codex_smoke/bench_stage8_backward_q2048_r64.log

.venv/bin/python agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json profiling_kt_codex_smoke/kt_armbf16_reduce_chunk_qwen3/profile.json \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 64 \
  --expected-batch 1 \
  --expected-rank 8 \
  --expected-dropout 0.0 \
  --expected-top-k 8 \
  --expected-cache-depth 2 \
  --expected-recompute false \
  --require-final
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
- Stage 8A gate passed on `profiling_kt_codex_smoke/kt_armbf16_reduce_chunk_qwen3`: `validate_kt_arm_profile.py` passes, native profile evidence is present, and `backward_thread_reduce_ms` is no longer the dominant phase.
- Stage 8B gate is still open: `backward_local_alloc_zero_ms`, dense scratch bytes, and `backward_route_loop_ms` must be reduced without correctness loss.
- Wrapper `backward_sync_ms_last` is explained by native detailed timers within 10 percent.
- Peak CPU RSS does not increase versus baseline for the same shape.
- `perf_stat_backward_route_loop.txt` and `perf_stat_backward_reduce.txt` show reductions in instructions, cycles, and cache misses versus Stage 4A/Stage 3 baseline.
- Backward profile artifacts prove dense full-expert per-thread gradient buffers are no longer allocated and reduced for untouched experts.

## Stage 9: NUMA and Memory-Flow Optimization

Problem: once compute kernels improve, memory placement and bandwidth can become the limiter.

Implementation:

- Keep route buffers, packed inputs, forward caches, and LoRA weights on the NUMA node used by worker threads.
- Avoid first-touch by the wrong Python/caller thread.
- Add optional log fields for current CPU, NUMA node, and allocation sizes for major buffers.
- Add native allocation/layout logs for each major buffer:
  - pointer address modulo 64
  - NUMA node if available
  - byte size
  - first-touch thread id / CPU
  - owner layer and reuse count
- Add route-layout counters:
  - active experts
  - padded routes / valid routes
  - hottest expert routes
  - route skew ratio
  - per-expert local route min/p50/max
- Save CPU topology artifacts for every accepted NUMA profile.
- Evaluate huge pages only after correctness and affinity are fixed.

Validation commands:

```bash
cd /workspace/AsymGEMM-SFT/third_party/AsymGEMM
mkdir -p profiling_kt_codex_smoke/kt_armbf16_numa_artifacts
numactl --hardware > profiling_kt_codex_smoke/kt_armbf16_numa_artifacts/numactl_hardware.txt
lscpu -e=CPU,NODE,SOCKET,CORE,CACHE > profiling_kt_codex_smoke/kt_armbf16_numa_artifacts/lscpu_extended.txt
command -v hwloc-ls && hwloc-ls --whole-system > profiling_kt_codex_smoke/kt_armbf16_numa_artifacts/hwloc_whole_system.txt || true

taskset -c 0-143 numactl --cpunodebind=0 --membind=0 env \
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=512 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_numa_node0 \
  agent/kt/scripts/run_lf_lora_sft_kt.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_numa_node0.log

taskset -c 0-143 numactl --cpunodebind=0,1 --membind=0,1 env \
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=512 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_numa_all \
  agent/kt/scripts/run_lf_lora_sft_kt.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_numa_all.log
```

Profile gate:

- Chosen NUMA policy is faster or equal on source-profile timing.
- `numastat -p` does not show accidental remote-memory dominance for the selected policy.
- `lscpu_extended.txt` and `numactl_hardware.txt` are stored next to the profile.
- Native logs prove first-touch happens on the intended NUMA node for packed inputs, forward work buffers, backward work buffers, cache buffers, and LoRA weight buffers.
- Route-layout counters are present and can explain load imbalance separately from kernel throughput.
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
    GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
    BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
    KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 \
    KT_ARM_SFT_TOKEN_CHUNK_SIZE="$chunk" \
    MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
    CUTOFF_LEN=1024 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
    OUT_DIR="profiling_kt_codex_smoke/kt_armbf16_chunk_${chunk}" \
    agent/kt/scripts/run_lf_lora_sft_kt.sh 2>&1 | tee "profiling_kt_codex_smoke/kt_armbf16_chunk_${chunk}.log"
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
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=nsys \
  KT_ARM_SOURCE_OK_PROFILE_JSON=profiling_kt_codex_smoke/kt_armbf16_source_ok/profile.json \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=512 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_nsys_sanity \
  agent/kt/scripts/run_lf_lora_sft_kt.sh
```

Use NCU only for CUDA kernels outside the ARM CPU MoE hot path:

```bash
ncu --target-processes all --set full --launch-count 20 --force-overwrite \
  --export profiling_kt_codex_smoke/kt_armbf16_cuda_sanity \
  taskset -c 0-143 env GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 BACKEND=kt_armbf16 PROFILE=0 \
    MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
    CUTOFF_LEN=128 PER_DEVICE_TRAIN_BATCH_SIZE=1 LORA_RANK=8 MAX_STEPS=1 \
    agent/kt/scripts/run_lf_lora_sft_kt.sh
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
  GPU_ID=1 NUM_GPUS=1 PROFILE_NSYS_GPU_METRICS_DEVICES=1 \
  BACKEND=kt_armbf16 PROFILE=1 PROFILE_PROFILER=source \
  KT_ARM_OMP_NUM_THREADS=8 KT_NUM_THREADS=8 KT_ARM_SFT_PROFILE=1 KT_ARM_SFT_POOL_LOG=1 \
  KT_ARM_SFT_TOKEN_CHUNK_SIZE=2048 KT_ARM_FIRST_STEP_TIMEOUT_SECONDS=0 \
  MODEL_NAME_OR_PATH=Qwen/Qwen3-30B-A3B TEMPLATE=qwen3_nothink \
  CUTOFF_LEN=7168 PER_DEVICE_TRAIN_BATCH_SIZE=4 LORA_RANK=64 LORA_DROPOUT=0.10 \
  MAX_STEPS=1 WARMUP_STEPS=0 \
  OUT_DIR=profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final \
  agent/kt/scripts/run_lf_lora_sft_kt.sh 2>&1 | tee profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final.log

git status --short > profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final/git_status_short.txt
git diff -- scripts/lf/run_lf_lora_sft.sh scripts/lf/profile_lora_lf.sh \
  > profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final/main_script_diff.txt
git diff -- agent/kt/fix_arm_v3.md \
  > profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final/fix_plan_diff.txt
git -c safe.directory=/workspace/AsymGEMM-SFT/third_party/ktransformers -C ../ktransformers diff \
  > profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final/ktransformers_diff.txt || true

.venv/bin/python agent/kt/scripts/validate_kt_arm_profile.py \
  --profile-json profiling_kt_codex_smoke/kt_armbf16_source_b4_s7168_r64_v3_final/profile.json \
  --expected-model Qwen/Qwen3-30B-A3B \
  --expected-seq-len 7168 \
  --expected-batch 4 \
  --expected-rank 64 \
  --expected-dropout 0.10 \
  --expected-top-k 8 \
  --expected-cache-depth 2 \
  --expected-recompute true \
  --expected-token-chunk 2048 \
  --require-final
```

Acceptance gate:

- `profile.json` exists and is non-partial.
- `train.log` shows `GPU_ID=1` or `GPU_ID=2`; default accepted GPU is 1. No accepted profile may use GPU 0 or GPU 3.
- `cpu_affinity_count >= KT_NUM_THREADS`.
- `base_kernel=sve_bfdot`, `aligned_weights=1`, and `sve_vector_bytes` appear in `train.log`.
- `validate_kt_arm_profile.py` passes for the final `profile.json`.
- Provenance artifacts exist: `git_status_short.txt`, `main_script_diff.txt`, `fix_plan_diff.txt`, and `ktransformers_diff.txt`.
- Stage 4A artifacts exist for the final build: CPU features, extension disassembly/build manifest, layout report, perf stat, and perf report.
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

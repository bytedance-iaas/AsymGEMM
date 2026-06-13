# Qwen3 Gate/Up Windowed Backward Kernel Progress

Canonical implementation plan:

```text
agent/fused_kernels/kernel.md
```

This file is the required progress ledger. Every implementation, correctness,
latency, NSYS, NCU, and E2E result for the staged kernel work must be recorded
here. Do not rely only on terminal output, chat history, or profile directories.

## Current Decision

```text
current_stage: stage7_cache_first_profile
current_status: hard_stop_reached_stage7_failed_for_e2e
last_updated: 2026-06-05
stage7_decision: failed_ncu_or_traffic
explicit_user_approval_after_stage7: false
next_action: stop for user review; do not start Stage 8, Stage 9, or Stage 10.
  The next approved kernel iteration should replace scalar recompute/dX with
  tensor-core tiled kernels, add qwen_shape_routed and aggregate profile gates,
  then rerun Stage 7 before any LF E2E claim.
```

## Stage Entry Template

Copy this template for every stage attempt.

```text
stage:
attempt:
status:
date:
git_or_worktree_note:

changed_files:
changed_functions:
new_public_apis:

commands:
validation_json:
validation_md:
latency_csv:
nsys_artifacts:
ncu_artifacts:
profile_jsons:

correctness_summary:
latency_summary:
memory_traffic_summary:
before_after_summary:

passed_gates:
failed_gates:
missing_artifacts:

decision:
next_action:
```

## Stage 0: Baseline Instrumentation

```text
status: not_started
decision: pending
```

## Stage 1: Metadata And W-Cache Fill

```text
status: implemented_in_direct_native_api
decision: passed_op_level_correctness_counters
date: 2026-06-05
changed_files:
  csrc/apis/qwen3_moe.hpp
  csrc/qwen3/qwen3_gate_up_windowed_bwd.cu
  csrc/python_api.cpp
  setup.py
  CMakeLists.txt
  asym_gemm/__init__.py
  scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
  tests/training/test_qwen3_gate_up_windowed_bwd.py
commands:
  TORCH_CUDA_ARCH_LIST=10.0 python setup.py build_ext --inplace
  CUDA_VISIBLE_DEVICES=1 python -m pytest -q tests/training/test_qwen3_gate_up_windowed_bwd.py -rs
  CUDA_VISIBLE_DEVICES=1 python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage4_native_direct_e2e --op native_e2e --case tiny --device cuda:0 --warmup-iters 1 --latency-iters 3 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage4_native_direct_e2e_tiny
validation_json:
  profiling/qwen3_gate_up_windowed_bwd/stage4_native_direct_e2e_tiny/validation.json
correctness_summary:
  metadata row mapping and w_cache fill pass through native_e2e validation
memory_traffic_summary:
  tiny: cpu_weight_stream_multiplier=1.0, hbm_w_cache_valid_write_bytes=expected_cpu_weight_bytes_min
```

## Stage 2: Recompute And Activation Backward

```text
status: implemented_in_direct_native_api
decision: passed_op_level_correctness
date: 2026-06-05
validation_json:
  profiling/qwen3_gate_up_windowed_bwd/stage4_native_direct_e2e_tiny/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage4_ragged_groups/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage4_partial_window/validation.json
correctness_summary:
  grad_gate_sel and grad_up_sel match BF16-contract references on tiny,
  one_group, ragged_groups, and partial_window cases
missing_artifacts:
  focused NCU activation/local-memory report not collected yet
```

## Stage 3: Selected dX Window

```text
status: implemented_in_direct_native_api
decision: passed_op_level_correctness
date: 2026-06-05
validation_json:
  profiling/qwen3_gate_up_windowed_bwd/stage4_native_direct_e2e_tiny/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage4_one_group/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage4_ragged_groups/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage4_partial_window/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage4_qwen_shape_smallM/validation.json
correctness_summary:
  grad_x_base_sel matches BF16-contract reference by exact check on tiny and
  by BF16 allclose on larger reduction-order-sensitive cases
memory_traffic_summary:
  dx reads w_cache and reports dx_reads_w_cache_bytes; no second CPU W stream in
  native direct API
missing_artifacts:
  dX implementation is scalar CUDA, not final tensor-core grouped GEMM
  focused NCU dX report not collected yet
```

## Stage 4: Native Direct End-To-End

```text
status: implemented_and_validated_op_level
decision: passed_correctness_not_stage7_ready
date: 2026-06-05
new_public_apis:
  asym_gemm.qwen3_gate_up_recompute_bwd_sm100_bf16_windowed
commands:
  CUDA_VISIBLE_DEVICES=1 python -m pytest -q tests/training/test_qwen3_gate_up_windowed_bwd.py -rs
  CUDA_VISIBLE_DEVICES=1 python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage4_native_direct_e2e --op native_e2e --case qwen_shape_smallM --device cuda:0 --warmup-iters 0 --latency-iters 1 --max-abs-tol 32 --max-rel-tol 0.05 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage4_qwen_shape_smallM
correctness_summary:
  pytest exact tiny case passed
  qwen_shape_smallM passed BF16 allclose with max_abs_error=16.0 and all close_checks=true
latency_summary:
  qwen_shape_smallM native_total_ms about 13.45 ms for one direct op call
  this is not a Stage 7 comparison and not an E2E speedup claim
memory_traffic_summary:
  qwen_shape_smallM cpu_weight_stream_multiplier=1.0
  w_cache_bytes_allocated_peak=16 MiB for E=8,H=2048,I=768,Q=8
missing_artifacts:
  no NSYS artifacts
  no NCU artifacts
  no current-path before/after comparison
```

## Stage 5: Native Down-LoRA-A Contract Repair

```text
status: implemented_and_validated_op_level
decision: native_contract_repaired_python_integration_pending
date: 2026-06-05
changed_files:
  csrc/apis/qwen3_moe.hpp
  csrc/qwen3/qwen3_gate_up_windowed_bwd.cu
  asym_gemm/__init__.py
  scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
  scripts/lf/profile_qwen3_gate_up_windowed_ops.sh
  tests/training/test_qwen3_gate_up_windowed_bwd.py
new_public_apis:
  asym_gemm.qwen3_gate_up_recompute_bwd_sm100_bf16_windowed old overload:
    returns grad_x_base_sel, grad_gate_sel, grad_up_sel, stats
  asym_gemm.qwen3_gate_up_recompute_bwd_sm100_bf16_windowed integrated overload:
    consumes dS_down_sel, down_mask_packed, down_dropout_p
    returns grad_x_base_sel, grad_gate_sel, grad_up_sel,
    grad_down_lora_A_sel, stats
commands:
  python setup.py build_ext --inplace
  CUDA_VISIBLE_DEVICES=1 python -m pytest -q tests/training/test_qwen3_gate_up_windowed_bwd.py -rs
  CUDA_VISIBLE_DEVICES=1 python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage5_integrated_down_lora_A --op native_e2e_down_lora_A --case tiny --device cuda:0 --with-down-lora-a --r-down 3 --down-dropout-p 0.25 --warmup-iters 1 --latency-iters 3 --max-abs-tol 1.0 --max-rel-tol 0.05 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage5_integrated_down_lora_A_tiny
  CUDA_VISIBLE_DEVICES=1 python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage5_integrated_down_lora_A --op native_e2e_down_lora_A --case qwen_shape_smallM --device cuda:0 --with-down-lora-a --r-down 64 --down-dropout-p 0.0 --warmup-iters 0 --latency-iters 1 --max-abs-tol 128 --max-rel-tol 0.2 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage5_integrated_down_lora_A_qwen_shape_smallM
validation_json:
  profiling/qwen3_gate_up_windowed_bwd/stage5_integrated_down_lora_A_tiny/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage5_integrated_down_lora_A_qwen_shape_smallM/validation.json
correctness_summary:
  resolved the prior activation dependency by moving selected dA_down into the
  native op while keeping dact_lora and final dact outside native
  tiny case with down_dropout_p=0.25 passed exact BF16-contract checks for
  grad_x_base_sel, grad_gate_sel, grad_up_sel, and grad_down_lora_A_sel
  qwen_shape_smallM passed configured BF16 allclose checks for all four outputs
latency_summary:
  tiny median_latency_ms=0.2963, native_total_ms=0.1694
  qwen_shape_smallM median_latency_ms=14.0569, native_total_ms=13.4337
  these are scalar-prototype timings and not a Stage 7 latency decision
memory_traffic_summary:
  tiny cpu_weight_stream_multiplier=1.0, down_lora_A_atomic_adds=150
  qwen_shape_smallM cpu_weight_stream_multiplier=1.0,
  w_cache_bytes_allocated_peak=16 MiB,
  grad_down_lora_A_bytes_allocated_peak=1.5 MiB,
  down_lora_A_atomic_adds=25,165,824
passed_gates:
  old four-output overload still works without dS_down_sel
  integrated five-output overload works with dS_down_sel and packed down mask
  validator writes dA_down correctness, latency, and traffic counters
failed_gates:
  Python training backward is not integrated yet
  scalar CUDA recompute and atomic dA_down are not Stage 7 performance-ready
  no NSYS artifacts
  no NCU artifacts
next_action:
  completed by Stage 5B and Stage 6 Python integration
```

## Stage 5B: Python Integration Dropout 0.00

```text
status: implemented_and_validated_python_integration
decision: passed_correctness_not_latency_gate
date: 2026-06-05
changed_files:
  asym_gemm/training/qwen3_moe.py
  scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
changed_functions:
  _ThresholdedQwen3ExpertFunction.backward
  _grouped_down_lora_backward_split_loop_free
  _selected_rows_for_mode
  _select_packed_mask_rows
  run_python_integration_case
new_public_controls:
  ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1
  ASYM_QWEN3_GATE_UP_WINDOWED_BWD_P
  ASYM_QWEN3_GATE_UP_WINDOWED_BWD_Q
  ASYM_QWEN3_GATE_UP_WINDOWED_BWD_BM
  ASYM_QWEN3_GATE_UP_WINDOWED_BWD_BK
  ASYM_QWEN3_GATE_UP_WINDOWED_BWD_G_WORK
commands:
  CUDA_VISIBLE_DEVICES=1 ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1 python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_recompute_policies_match_none --tb=short -x -rs
  CUDA_VISIBLE_DEVICES=1 python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage6_python_integration_drop000 --op python_integration --case one_group --device cuda:0 --expert-policy tok-le1024 --lora-dropout 0.00 --warmup-iters 0 --latency-iters 1 --max-abs-tol 0.25 --max-rel-tol 0.10 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage6_python_integration_drop000
validation_json:
  profiling/qwen3_gate_up_windowed_bwd/stage6_python_integration_drop000/validation.json
validation_md:
  profiling/qwen3_gate_up_windowed_bwd/stage6_python_integration_drop000/validation.md
correctness_summary:
  pytest passed for recompute policy parity with native integration enabled
  validator passed output, input grad, and LoRA parameter grad comparisons
  old_selected_base_dx_rows=0 and new_selected_base_dx_rows=8, confirming that
  selected base dX moved to the native path for this one_group validator case
latency_summary:
  native_kernel_total_ms=0.7382 in validator stats
  Python wall-time selected-region numbers are rough one-iteration integration
  timings and are not a Stage 7 latency decision
memory_traffic_summary:
  cpu_weight_stream_multiplier=1.0
  w_cache_bytes_allocated_peak=262144
  grad_down_lora_A_bytes_allocated_peak=32768
passed_gates:
  native API is invoked from Python backward
  selected down-LoRA-A is merged from native selected contribution plus existing
  nonselected path
  selected old Python base-dX work is skipped
failed_gates:
  no NSYS profile in this stage
  no NCU profile in this stage
decision:
  passed_correctness_not_latency_gate
next_action:
  validate dropout 0.10 mask consumption and then run Stage 7 profiles
```

## Stage 6: Python Integration Dropout 0.10

```text
status: implemented_and_validated_python_integration
decision: passed_correctness_not_latency_gate
date: 2026-06-05
changed_files:
  asym_gemm/training/qwen3_moe.py
  scripts/lf/validate_qwen3_gate_up_windowed_bwd.py
commands:
  CUDA_VISIBLE_DEVICES=1 ASYM_QWEN3_GATE_UP_WINDOWED_BWD=1 python -m pytest -q tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_recompute_lora_dropout_matches_none tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_recompute_lora_dropout_backward_consumes_no_rng --tb=short -x -rs
  CUDA_VISIBLE_DEVICES=1 python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage6_python_integration_drop010 --op python_integration --case one_group --device cuda:0 --expert-policy tok-le1024 --lora-dropout 0.10 --warmup-iters 0 --latency-iters 1 --max-abs-tol 0.25 --max-rel-tol 0.10 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage6_python_integration_drop010
validation_json:
  profiling/qwen3_gate_up_windowed_bwd/stage6_python_integration_drop010/validation.json
validation_md:
  profiling/qwen3_gate_up_windowed_bwd/stage6_python_integration_drop010/validation.md
correctness_summary:
  pytest passed dropout parity and no-RNG-consumption cases with native
  integration enabled
  validator passed output, input grad, and LoRA parameter grad comparisons
  native_kernel_consumed_down_dropout_masks=true
latency_summary:
  native_kernel_total_ms=0.6675 in validator stats
  Python wall-time selected-region numbers are rough one-iteration integration
  timings and are not a Stage 7 latency decision
memory_traffic_summary:
  cpu_weight_stream_multiplier=1.0
  down_mask_reads_bytes=4096
  w_cache_bytes_allocated_peak=262144
passed_gates:
  saved packed down dropout mask is consumed by native selected dA_down path
  backward does not rerun dropout RNG
  selected old Python base-dX work is skipped
failed_gates:
  no NSYS profile in this stage
  no NCU profile in this stage
decision:
  passed_correctness_not_latency_gate
next_action:
  run Stage 7 NSYS/NCU cache-first profile gate
```

## Stage 7: NSYS/NCU Cache-First Profile Gate

Hard stop stage. After this section is updated with the aggregate Stage 7
decision, stop and ask the user to review before starting Stage 8, Stage 9, or
Stage 10.

Allowed decisions:

```text
passed_for_e2e
failed_correctness
failed_latency
failed_ncu_or_traffic
blocked_missing_artifacts
```

```text
status: completed_failed_profile_gate
decision: failed_ncu_or_traffic
explicit_user_approval_to_continue_after_stage7: false
date: 2026-06-05
changed_files:
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/validation.md
commands:
  CUDA_VISIBLE_DEVICES=1 nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none -o profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/native_one_group python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage7_cache_first_profile --op native_e2e --case one_group --device cuda:0 --with-down-lora-a --r-down 8 --down-dropout-p 0.10 --warmup-iters 1 --latency-iters 3 --max-abs-tol 1.0 --max-rel-tol 0.10 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/native_one_group_validation
  CUDA_VISIBLE_DEVICES=1 ncu --target-processes all --kernel-name 'regex:.*(fill_w_cache_kernel|recompute_activation_kernel|dx_window_kernel).*' --launch-count 6 --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats --force-overwrite --export profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu/native_one_group --csv --log-file profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu/native_one_group.csv python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage7_cache_first_profile --op native_e2e --case one_group --device cuda:0 --with-down-lora-a --r-down 8 --down-dropout-p 0.10 --warmup-iters 0 --latency-iters 1 --max-abs-tol 8.0 --max-rel-tol 0.20 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu/native_one_group_validation
  ncu --import profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu/native_one_group.ncu-rep --csv --page raw > profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu/native_one_group_metrics.csv
  CUDA_VISIBLE_DEVICES=1 python scripts/lf/validate_qwen3_gate_up_windowed_bwd.py --stage stage7_cache_first_profile --op native_e2e --case qwen_shape_smallM --device cuda:0 --with-down-lora-a --r-down 64 --down-dropout-p 0.10 --warmup-iters 1 --latency-iters 3 --max-abs-tol 256 --max-rel-tol 0.25 --output-dir profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/native_qwen_shape_smallM_validation
validation_json:
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/native_one_group_validation/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/native_qwen_shape_smallM_validation/validation.json
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu/native_one_group_validation/validation.json
validation_md:
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/validation.md
nsys_artifacts:
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile/native_one_group.nsys-rep
ncu_artifacts:
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu/native_one_group.ncu-rep
  profiling/qwen3_gate_up_windowed_bwd/stage7_cache_first_profile_ncu/native_one_group_metrics.csv
correctness_summary:
  Stage 6 Python integration correctness passed for dropout 0.00 and 0.10
  Stage 7 native one_group and qwen_shape_smallM correctness passed under the
  configured BF16 allclose thresholds
latency_summary:
  one_group native_total_ms=0.7822 outside NCU
  qwen_shape_smallM native_total_ms=13.0103, median_latency_ms=13.2301
  qwen_shape_smallM recompute_down_lora_activation_ms=11.7699, so the scalar
  recompute/activation/down-LoRA-A phase dominates the prototype
memory_traffic_summary:
  qwen_shape_smallM cpu_weight_stream_multiplier=1.0
  qwen_shape_smallM w_cache_bytes_allocated_peak=16 MiB
  qwen_shape_smallM grad_down_lora_A_bytes_allocated_peak=1.5 MiB
  qwen_shape_smallM recompute_reads_w_cache_bytes=3 GiB
  qwen_shape_smallM dx_reads_w_cache_bytes=3 GiB
  NCU one_group tensor_pipe_pct=0.0 for fill_w_cache_kernel,
  recompute_activation_kernel, and dx_window_kernel
before_after_summary:
  integration validator confirms the old selected base-dX rows are zero when
  native is enabled, but Stage 7 paired same-shape current/native NSYS/NCU
  profiles were not completed, so no LF E2E speedup claim is allowed
passed_gates:
  Stage 6 correctness passed
  native selected path executes from Python integration
  native direct op correctness passed for one_group and qwen_shape_smallM
  CPU W stream multiplier is 1.0 in native validations
  at least one NCU report exists for the scalar native kernels
failed_gates:
  implementation reaching NCU is scalar fill/recompute/dX, not tensor-core
  recompute/dX
  tensor-pipe utilization is 0.0 percent in the captured kernels
  qwen_shape_smallM native_total_ms is too high for an E2E integration pass
  required paired same-shape current/native Stage 7 profiles are incomplete
  qwen_shape_routed Stage 7 case is missing
  required qwen3_gate_up_windowed/* NSYS ranges are missing
missing_artifacts:
  paired current-path NSYS/NCU profile for qwen_shape_smallM
  paired native-path NSYS/NCU profile for qwen_shape_smallM
  paired current/native NSYS/NCU profiles for qwen_shape_routed
  separate or explicitly mapped qwen_shape_smallM NCU artifacts for
  fill_w_cache/recompute/down_lora/dx
  cache_first_profile aggregate validator op in the validation script
decision:
  failed_ncu_or_traffic
next_action:
  HARD STOP. Do not start Stage 8, Stage 9, or Stage 10 LF E2E without explicit
  user approval. Proposed next iteration is to replace scalar recompute/dX with
  tensor-core tiled kernels, add qwen_shape_routed and aggregate Stage 7 gates,
  then rerun Stage 7.
```

## Stage 8: Optional seed_group_direct

Do not start until Stage 7 is written here and the user explicitly approves.

```text
status: not_started
decision: waiting_for_stage7_and_user_approval
```

## Stage 9: Optional all_rows_direct

Do not start until Stage 7 is written here and the user explicitly approves.

```text
status: not_started
decision: waiting_for_stage7_and_user_approval
```

## Stage 10: LF 50-Step End-To-End Gate

Do not start until Stage 7 is written here with `passed_for_e2e` and the user
explicitly approves E2E.

```text
status: not_started
decision: waiting_for_stage7_pass_and_user_approval
```

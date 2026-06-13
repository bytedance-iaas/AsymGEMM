# SuperOffload Method Notes

## Core Idea

- SuperOffload is a ZeRO-3 subclass, not a replacement for ZeRO-3.
- Keeps normal ZeRO-3 parameter partition/fetch/release and gradient reduce path.
- Main change: start CPU Adam updates during backward as soon as a subgroup's gradients are ready.
- Normal ZeRO-3 offload waits until `optimizer.step()` to run CPU Adam.
- SuperOffload overlaps GPU backward compute with CPU optimizer work.

## What Runs Where

- GPU: forward, backward, gradient production.
- ZeRO-3: local subgroup/partition management, grad partition buffers, param fetch/release.
- CPU worker: Adam update for ready fp32 master-weight subgroup.
- Single-GPU case: "shard" mostly means local ZeRO subgroup/tile, not multi-GPU sharding.

## CPU Adam Inputs

- CPU Adam updates fp32 master weight shards/subgroups.
- Inputs needed for each subgroup:
  - fp32 master weight
  - gradient shard/subgroup
  - Adam states: `exp_avg`, `exp_avg_sq`, step
- Updated fp32 master weight is copied/cast back to the bf16/fp16 compute weight shard.

## Precision Reminder

- bf16/fp16 weights are for GPU forward/backward compute.
- fp32 master weights are for stable Adam updates.
- This is why CPU Adam may update fp32 even when training compute is bf16.

## Speculation/Validation

- CPU update can start before the full backward pass is done.
- It is speculative because overflow/global norm/clipping may not be known yet.
- At `step()`, DeepSpeed waits for async CPU results and validates.
- If overflow occurs, CPU Adam subgroup state can be rolled back.
- If gradient clipping is active, SuperOffload uses synchronous CPU updates, so the async overlap benefit is reduced/disabled.

## Compared With ZeRO-3 CPU Offload

- Same ZeRO-3 offload machinery for params and grads.
- Same CPU offload target in current configs.
- Different optimizer schedule:
  - ZeRO-3 offload: backward finishes, then CPU Adam in `step()`.
  - SuperOffload: CPU Adam starts during backward for completed subgroups.
- Small code surface, important runtime scheduling change.

## Current LlamaFactory Config Facts

- `ds_z3_superoffload_config.json` uses CPU offload only.
- `offload_optimizer.device = "cpu"`.
- `offload_param.device = "cpu"`.
- `pin_memory = true` for both.
- `super_offload = true`.
- `cpuadam_cores_perc = 0.8`.
- No `nvme_path`; NVMe offload is disabled.
- Normal `ds_z3_offload_config.json` also uses CPU only, not NVMe.

## LoRA SFT Reminder

- Frozen base weights are not optimized by Adam.
- CPU optimizer offload applies to trainable LoRA weights only.
- `offload_param: cpu` can still move frozen base params through CPU offload because forward/backward needs them.
- LoRA weights are small; they may stay persistent depending on `stage3_param_persistence_threshold`.
- With HF `"auto"`: `stage3_param_persistence_threshold = 10 * hidden_size`.
- Lower threshold: more small params partitioned/offloaded.
- Higher threshold: more small params kept resident, less fetch overhead.

## Limitations To Remember

- CUDA-only assertion in current SuperOffload implementation.
- No separate SuperOffload C++/CUDA kernel; it reuses `DeepSpeedCPUAdam`.
- CPUAdam worker is a separate spawned process.
- Worker-local optimizer state makes checkpoint/cleanup behavior worth auditing.
- No in-tree end-to-end SuperOffload tests found; CPUAdam subgroup/rollback tests exist.

## Code Anchors

- `deepspeed/runtime/zero/offload_config.py`: `super_offload`, `cpuadam_cores_perc`.
- `deepspeed/runtime/engine.py`: swaps ZeRO-3 class to `SuperOffloadOptimizer_Stage3`.
- `deepspeed/runtime/superoffload/superoffload_stage3.py`: ZeRO-3 subclass and async scheduling.
- `deepspeed/runtime/superoffload/superoffload_utils.py`: CPUAdam worker process and queues.
- `deepspeed/ops/adam/cpu_adam.py`: `step_subgroup()` and `rollback_subgroup()`.

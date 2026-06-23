# Fix / enable SM100 BF16 + v14 GPU SwiGLU-backward for the long profiling sweep

Follow-up to `port_sm100_bf16.md`. Records the v14 GPU expert-SwiGLU-backward
extension to Llama4, the profiling enablement, and the remaining test fixes.

**Testing point / script to modify: `scripts/lf/profile_lora_lf_long.sh`.**
**Config: `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=cpu`** (script default, line 77).

## Runtime changes (applied)

### 1. Shared GPU SwiGLU backward — `asym_gemm/training/qwen3_moe.py`

- Refactored `_silu_backward_gpu(ctx, grad_act, manager)` →
  `_silu_backward_gpu(gate_cpu, up_cpu, grad_act, manager)` (symmetric with the CPU
  `_activation_offload_cpu_silu_backward`) so Qwen3 and Llama4 share one implementation;
  updated the Qwen3 call site to pass `ctx.gate_cpu, ctx.up_cpu`. Behavior identical for Qwen3.
- Gated by `ASYMM_EXPERT_SILU_BWD_GPU` (`_use_gpu_silu_bwd()`, unset → OFF).
- Math (bf16): `grad_up = grad_act * silu(gate)`, `grad_gate = silu_backward(grad_act * up, gate)`.
  Keeps `grad_act` on GPU (skips the `dact` D2H), stages gate/up H2D, releases immediately.

### 2. Llama4 wiring — `asym_gemm/training/llama4_experts.py`

- Import `_use_gpu_silu_bwd, _silu_backward_gpu` from `qwen3_moe`.
- `_ActivationOffloadLlama4ExpertFunction.backward`: branch on `_use_gpu_silu_bwd()` exactly
  like Qwen3 — keep `grad_act` on GPU, get gate/up via `ensure_gate_up_cpu()`, call
  `_silu_backward_gpu(gate_handle, up_handle, grad_act, manager)`, set CPU handles `None`.
- Pure Python — no rebuild.

### 3. Profiling toggles — `scripts/lf/profile_lora_lf_long.sh` (TESTING POINT)

- Top of script (~lines 120-121), default ON, flip to `0` for A/B; forwarded via `run_env` (~2534-2535):
  - `ASYMM_EXPERT_SILU_BWD_GPU=${ASYMM_EXPERT_SILU_BWD_GPU:-1}`
  - `DG_BF16_CPU_LEFT_COMPACT_GRID=${DG_BF16_CPU_LEFT_COMPACT_GRID:-1}`

## What is exercised under the default (cpu) sweep

- **Tiled atomic-free LoRA-A grad** — default ON (atomic only via `ASYMM_LORA_A_GRAD_ATOMIC=1`). Both models.
- **GPU silu bwd** — ON, exercised (expert act-offload on). Qwen3 + Llama4 both validated.
- **Compact grid** — ON and **LIVE under `cpu`**: the expert cpu-left LoRA-A forward is multi-group, so
  `grid.x = max-group-blocks` instead of `total-blocks` → real launched-CTA savings.
- **Native gate/up pair** (`ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE`) — **OFF** (two-call gate/up path).
  To also measure it: `export ASYMM_CPU_LEFT_LORA_A_PAIR_NATIVE=1` before the run (the script launches
  training via `env …`, so an exported var is inherited) and rerun. Validated bit-exact to two calls.

## Validation (done)

- Llama4 offload backward, env=0 (CPU) vs env=1 (GPU): **bit-exact** — input grad + all 6 LoRA grads.
- Qwen3 offload backward, env=1: numerical L2 checks vs the torch backend pass.
- `bash -n scripts/lf/profile_lora_lf_long.sh`: OK.

## Remaining test fixes (for a green suite; NOT blocking the profiling run)

1. `tests/training/test_lf_qwen3_asym_backend.py::test_asym_qwen3_experts_sm100_activation_offload_matches_torch_backend`
   (8 params) fail under silu-bwd=ON with `KeyError: 'dgate_up_for_gate_up_base'` — a CPU-path
   staging-stat assert (numerics pass). Make it path-aware: when `_use_gpu_silu_bwd()`, assert the GPU
   stage tags (`gate_for_silu_bwd` / `up_for_silu_bwd`) instead of the CPU concat tag.
2. Add a Llama4 offload-vs-torch numerical test mirroring the Qwen3 one.
3. Port the kernel-port test update `tests/training/test_cpu_left_lora.py` (stale native-pair test:
   `atol=0` + call-count `== 3`).

## Run for final numbers

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
bash scripts/lf/profile_lora_lf_long.sh
```

Defaults: `Qwen/Qwen3-30B-A3B|1` + `meta-llama/Llama-4-Scout-17B-16E|1`, backend
`asym_cpuadamwds|norecomp|ligerloss1`, workload `4096|8|1`, warmup 3 / measure 3, profilers both,
`LORA_A_FWD=cpu`. Outputs per run: `profile.json`, `summary.md`, memory-breakdown, plots.

A/B: rerun with `ASYMM_EXPERT_SILU_BWD_GPU=0` and/or `DG_BF16_CPU_LEFT_COMPACT_GRID=0`.

CAUTION — stale results: the new toggles are NOT encoded in the per-run output directory name, so a
prior run for the same config would be treated as complete and skipped (reusing pre-change numbers).
For fresh numbers, run with `OVERWRITE=true` or a fresh `RUN_NAME=`/`OUTPUT_ROOT=`.

Ops: run sequentially; stop with `kill -TERM` / `term` (never `-9` — corrupts the DeepSpeed
cpu_adam JIT and hangs); heavy host RAM.

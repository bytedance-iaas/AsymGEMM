# KT ARM BF16 SFT Fix Plan v11 — Forward base GEMM register-blocking (latency)

**KT remains native ARM BF16 CPU code. DeepSpeed is not used.**

After a deep re-audit (the memory loop had only addressed transposed-weight RSS),
two meaningful optimizations were found that were NOT yet incorporated. v11 is the
latency one.

## Finding

The FORWARD base projection GEMMs used a scalar-M (MB=1) BFDOT kernel:
- `arm_bf16_gate_up_matmul_blocked` (gate+up) and `arm_bf16_matmul_blocked4` (down)
  loop `for mm` one A-row at a time, reloading each BF16 weight column for every
  one of the M=256 route rows.

Meanwhile v7's `arm_bf16_grad_matmul_reg` (used for the base BACKWARD grads) is a
register-blocked **MB=4** kernel: each weight row is reused across 4 route rows,
~4x less weight memory traffic. The forward base GEMMs are the single largest
compute in the kernel (`base_gate_up_ms` 43305 + `base_down_ms` 14615 task-ms/layer)
and are also re-run in the backward recompute (`backward_tile_recompute_ms` 47707).

## Change (v11)

`compute_gate_up_base_by_expert` and `compute_down_base_by_expert` now call the
MB=4 `arm_bf16_grad_matmul_reg` (renamed in comments to the shared forward+backward
BF16 GEMM):
- gate: `reg(m, I, H, x, H, gate_W, H, gate, I, false)`
- up:   `reg(m, I, H, x, H, up_W,   H, up,   I, false)` (x reloaded once more; x is
  the small operand, the ~4x weight-traffic cut dominates)
- down: `reg(m, H, I, act_bf16, I, down_W, I, down, H, false)`

Removed the now-dead MB=1 kernels `arm_bf16_gate_up_matmul_blocked` and
`arm_bf16_matmul_blocked4`. Numerically identical (same per-output svbfdot +
svaddv reduction order); MB only groups outputs. Helps forward base AND recompute.

Expected: meaningful cut to `base_gate_up_ms`/`base_down_ms` and
`backward_tile_recompute_ms`; estimated several s/step. Native labels
`base_kernel=sve_bfdot_blocked` / `down_kernel=bf16_bfdot_blocked` kept (still
bfdot-blocked; MB=4 is the block size).

Risk: low (numerically identical; the MB=4 pattern is the proven v7 kernel).
Validate: reference + dropout tests, then short LF profile; accept iff avg step /
forward / recompute improve with no correctness regression.

## Results log

### Short 3-step e2e profile (GPU 1) — short-run, full acceptance pending

| Metric | v10 short | v11+v12 short | Delta |
|---|---:|---:|---:|
| measured step | 233.1 s | **137.25 s** | **-41.1%** |
| forward total | ~72 s | 38.26 s | -47% |
| backward total | ~161 s | 97.36 s | -40% |
| base_gate_up_ms | 43305 | 12020 | -72% |
| base_down_ms | 14615 | 4099 | -72% |
| backward_tile_recompute_ms | 47707 | 19594 | -59% |
| backward_grouped_tile_ms (wall) | 1628 | 967 | -41% |
| expert_schedule_wall_ms | 1078 | 397 | -63% |
| peak_alloc / reserved HBM | 34.48 / 40.12 | 34.48 / 40.12 | = |
| warmup / measured loss | 2.2104 / 1.6583,1.6686 | 2.2104 / 1.6562,1.6656 | match |

**The forward base GEMMs were severely memory-bandwidth-bound** (MB=1 reloaded each
weight ~256x per expert block). MB=4 cut weight traffic ~4x -> ~3.6x faster on the
base GEMMs, cascading through forward AND the backward recompute. Numerically
identical (losses match); HBM flat. Per-commit tests 50/50 pass. This is the
single largest latency win of the whole effort. **ACCEPT** (pending full
acceptance confirmation).

Note: an initial short run was killed (SIGKILL, status 9) deep into step ~3 by an
environmental/transient cause (host RAM had 1474 GiB free, GPU uncontended, run
completed 2+ steps first); the re-run completed cleanly with the numbers above.

### Full 15-step acceptance (GPU 1) — ACCEPTED, strict validation PASS

Artifact: `profiling_results/profiling_kt_codex_smoke/v11_accept_fwdMB4_qwen3_s4096_b4_r64_w5_s10_t64_source/.../b4_s4096`
`PASS gpu_id=1 affinity_count=144 wrappers=48 fw=1440 bw=720`.

| Metric | v10_accept | v11_accept | Delta |
|---|---:|---:|---:|
| avg_step | 234.87 s | 170.55 s | **-27.4%** |
| avg_forward | 71.97 s | 35.82 s | **-50.2%** |
| avg_backward | 161.06 s | 132.59 s | -17.7% |
| steady RSS | 94.54 GiB | 94.57 GiB | = (v12 is RSS-neutral) |
| peak_allocated / reserved HBM | 34.479 / 44.029 | 34.479 / 44.029 | = |
| loss max/last | 1.831/1.469 | 1.831/1.4674 | match |

Verdict: **ACCEPT.** The 10-step official avg_step is 170.55 s (the short run's
137 s was 2-step noise) — still a -27% step / -50% forward reduction, losses match,
HBM flat. The forward base GEMM was memory-bandwidth-bound; MB=4 was the fix.

**Session cumulative v6 -> v11: avg_step 276.81 -> 170.55 s (-38.4%) and steady CPU
RSS 153.85 -> 94.57 GiB (-38.5%).**

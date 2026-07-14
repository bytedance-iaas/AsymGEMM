# KT ARM BF16 SFT Fix Plan v10 — Memory reduction (cont.)

**KT remains native ARM BF16 CPU code. DeepSpeed is not used.**

Continues the memory loop from `fix_arm_v9.md`. Acceptance rule unchanged: keep a
change only if it cuts **steady-state working RSS**
(`optimizer_memory.process_memory_after_step.rss_bytes`) meaningfully without
blowing up latency. (Do NOT use `rss_peak_bytes` — it carries a ~14 GiB one-time
init transient that masks small persistent-buffer removals.)

## Stage 1 (v10): Share the live transposed LoRA-B weights

Status: implemented / measured / ACCEPTED (short e2e).

The transposed LoRA-B weights (`*_lora_b_t_`, forward-used and trainable) were
per-layer member vectors (~58.7 MiB/layer x 48 = ~2.82 GiB). Moved them to a
single process-shared buffer, re-transposed (parallel over experts) for the
current layer at the start of every forward and backward. LoRA-B is trainable,
so the buffer is rebuilt every time (no caching flag) — it always reflects the
latest weights. Forward and backward are sequential per layer and the transpose
runs single-threaded before the parallel compute, so one shared buffer is
race-free and always holds the layer about to compute. Removed the per-layer
members and the `lora_forward_prepared_` / `lora_backward_prepared_` /
`lora_b_transposed_` caching flags.

| Metric | v9 short | v10 short | Delta |
|---|---:|---:|---:|
| steady RSS | 96.70 GiB | 94.27 GiB | **-2.43 GiB (-2.5%)** |
| peak_allocated / reserved HBM | 34.48 / 40.12 | 34.48 / 40.12 | = |
| measured step | 235.9 s | 233.1 s | -1.2% (noise) |
| warmup / measured loss | 2.2104 / 1.6532,1.6706 | 2.2104 / 1.6583,1.6686 | match |

Verdict: **ACCEPT.** Meaningful steady-RSS cut (-2.43 GiB), latency within noise
(the ~28 ms/step extra transpose work is masked), losses match (forward-path
correctness confirmed). Per-commit tests 50/50 pass.

## Cumulative memory result (v7 -> v10, steady RSS)

153.85 -> 100.42 (v8) -> 96.70 (v9) -> 94.27 (v10) GiB = **-59.58 GiB (-38.7%)**,
with avg step latency flat/slightly better and losses matching throughout.

## Convergence / remaining targets (marginal or out-of-scope)

After v10 the KT-controllable per-layer-persistent transposed buffers are all
shared or removed. Remaining items, all judged marginal or unsafe:
- `route_grad_x_accum` FP32->BF16: ~0.5 GiB — trivial.
- LoRA-grad sharded scratch: 1.14 GiB FP32 — accumulation precision-sensitive.
- per-thread tile temp: 0.89 GiB — needed for recompute + BF16 staging.
- Backward route tile size: shrinking trades a little scratch for scheduling
  latency — not worth it.

The dominant steady RSS is now params + grads + optimizer state (~78 GiB; ~58 GiB
of it the frozen base experts, the rest the LoRA AdamW state in
`asym_gemm/training/cpu_adam.py`) — irreducible or AsymGEMM-side (outside the KT
hard boundary). The KT memory loop has therefore converged: no remaining
KT-only change cuts steady RSS meaningfully without latency or precision cost.

## Results log

**KT remains native ARM BF16; DeepSpeed not used.**

### Consolidated v8+v9+v10 full 15-step acceptance (GPU 1, 64 threads) — ACCEPTED

Artifact: `profiling_results/profiling_kt_codex_smoke/v10_accept_sharedLoraB_qwen3_s4096_b4_r64_w5_s10_t64_source/.../b4_s4096`
Strict validation: `PASS gpu_id=1 affinity_count=144 wrappers=48 fw=1440 bw=720`
(BFDOT label intact, `backward_base_transpose_ms` present).

| Metric | v7_accept (before mem work) | v10 stack (after) | Delta |
|---|---:|---:|---:|
| steady RSS | 153.85 GiB | 94.54 GiB | **-59.31 GiB (-38.6%)** |
| peak_allocated HBM | 34.479 GiB | 34.479 GiB | = |
| peak_reserved HBM | 44.029 GiB | 44.029 GiB | = |
| avg_step (measured e2e) | 244.65 s | 234.87 s | -9.78 s (-4.0%, faster) |
| avg_forward | 76.32 s | 71.97 s | within noise |
| avg_backward | 166.30 s | 161.06 s | within noise |
| loss max/last | 1.8324/1.4692 | 1.8335/1.4687 | match |

Verdict: **ACCEPT.** The three transpose-sharing/removal changes cut steady CPU
RSS by ~59 GiB (-38.6%) while keeping HBM flat, losses matching at 15 steps, and
average step latency flat-to-slightly-better. This frees substantial CPU RAM for
activation offload (the binding constraint) at zero latency cost.

## Loop conclusion (CONVERGED)

The KT-side memory loop is complete. All KT-controllable per-layer-persistent
transposed buffers are now shared (base, LoRA-B) or removed (dead LoRA-A). The
remaining reducible items are marginal or unsafe, and the dominant steady RSS is
out of KT scope:
- route_grad_x FP32->BF16: ~0.5 GiB (trivial -> reject).
- LoRA-grad sharded scratch: 1.14 GiB FP32 (accumulation precision -> unsafe).
- per-thread tile temp: 0.89 GiB (needed for recompute/BF16 staging).
- ~78 GiB params+grad+optimizer state: ~58 GiB frozen base experts (the model,
  irreducible) + LoRA AdamW state in `asym_gemm/training/cpu_adam.py`
  (AsymGEMM-side, outside the KT hard boundary).

No remaining KT-only change cuts steady RSS meaningfully without latency or
precision cost, so the loop stops here per the acceptance rule.

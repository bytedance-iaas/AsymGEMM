# KT ARM BF16 SFT Fix Plan v9 — Memory reduction (cont.)

**KT remains native ARM BF16 CPU code. DeepSpeed is not used.**

Continues the memory loop from `fix_arm_v8.md` (shared transposed base weights,
−45 GiB RSS, accepted). Same acceptance rule: keep a change only if it cuts memory
meaningfully (not trivially) without blowing up latency, decided by e2e profiling.

## Diagnosis (post-v8)

Post-v8 RSS ~111 GiB. Remaining KT-controllable per-layer-persistent buffers:
- **Transposed LoRA-A weights (`gate/up/down_lora_a_t_`): ~79.8 MB/layer x 48 =
  ~3.8 GiB, and they are DEAD** — written by `transpose_lora_a_weights_for_backward`
  and only referenced by the alignment check; no compute kernel reads them (the
  backward LoRA grads use the original `*_lora_a_` pointers). Same shape of waste
  the base transposed weights had before v7, but here the buffers are simply unused.
- Transposed LoRA-B weights (`*_lora_b_t_`): ~58.7 MB/layer x 48 = ~2.82 GiB, and
  these ARE live (forward LoRA reads them). Sharing them like v8 is possible but
  they are forward-used and trainable (re-transposed each optimizer step) — higher
  risk, smaller win; tracked as a later candidate, may be marginal.
- Backward scratch items (LoRA-grad 1.14 GiB, route_grad_x 1.0 GiB fp32, per-thread
  temp 0.89 GiB) are each <1.2 GiB — trivial by the acceptance bar.

## Stage 1 (v9): Remove dead transposed LoRA-A weights

Status: implemented / measured / accepted — (in progress)

Delete `transpose_lora_a_weights_for_backward()`, the `*_lora_a_t_` member
vectors, the `lora_a_transposed_` flag, and the alignment-check references; drop
the call from `prepare_lora_backward_weights` (keep `transpose_lora_b_weights`).
This is both a memory win (~3.8 GiB) and a dead-path removal. Zero correctness
risk (the buffers are never read); confirmed by per-commit tests + e2e loss; the
freed transpose work is a tiny latency improvement, not a regression.

Expected: RSS −~3.5 GiB (111→~107.5 GiB); latency flat or slightly better; HBM
unchanged.

## Metric note (important)

Use **steady-state working RSS** (`optimizer_memory.process_memory_after_step.rss_bytes`),
NOT `memory.process.rss_peak_bytes`. The peak includes a ~14 GiB one-time init
transient that masks removals of small persistent buffers. Steady RSS is what
constrains CPU capacity for activation offload. Re-stated with steady RSS:
- v7 -> v8 (shared base transpose): 153.85 -> 100.42 GiB = **-53.4 GiB (-34.7%)**
  (the `rss_peak` view under-reported this as -46 GiB).

## Results log

**KT remains native ARM BF16; DeepSpeed not used.**

### Stage 1 (v9) ACCEPTED — short 3-step e2e profile (GPU 1, 64 threads)

Artifact: `profiling_kt_codex_smoke/v9_deadAt_qwen3_s4096_b4_r64_t64_source`

| Metric | v8 short | v9 short | Delta |
|---|---:|---:|---:|
| steady RSS | 100.34 GiB | 96.70 GiB | **-3.64 GiB (-3.6%)** |
| rss_peak (init transient) | 111.20 GiB | 110.98 GiB | -0.23 (noise; not the right metric) |
| peak_allocated / reserved HBM | 34.48 / 40.12 | 34.48 / 40.12 | = |
| measured step | 239.96 s | 235.91 s | -1.7% (run noise; dead-code removal is latency-neutral/positive) |
| backward total_ms | 162.5 s | 161.2 s | within noise |
| losses | 1.6544/1.6722 | 1.6532/1.6706 | match |

Verdict: **ACCEPT.** Removing the dead transposed LoRA-A buffers cut steady RSS by
3.64 GiB (matches the ~3.8 GiB allocation), with no latency cost (removing dead
work) and matching losses. Per-commit tests 50/50 pass. This is both a memory win
and a dead-path removal. Accepted on the short e2e profile (dead-code removal is
provably safe); a consolidated full acceptance will lock official numbers after
the next stage.

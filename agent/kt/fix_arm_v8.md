# KT ARM BF16 SFT Fix Plan v8 — Memory reduction

**KT remains native ARM BF16 CPU code. DeepSpeed is not used.**

Continues from `fix_arm_v7.md` (BFDOT base-grad latency win, accepted). v8 pivots
to the new objective: **meaningfully reduce memory without blowing up latency**,
decided by e2e profiling. A change is kept only if it cuts memory in a
non-trivial way while latency does not blow up; reject memory-neutral
latency-regressions and trivial memory wins.

## Acceptance rule (per user)
- KEEP iff: memory drops meaningfully (not trivially) AND latency not blown up.
- REJECT: memory same + latency up; memory down only trivially.
- Decide with e2e profiling (short 3-step gate; full 15-step to finalize).

## Diagnosis (from v7_accept profile + native scratch/pool logs)

Process RSS ~157 GiB. KT-controllable breakdown:
- Forward base expert weights (frozen, bf16): ~58 GB — the model, irreducible.
- **Transposed base backward weights (`gate/up/down_proj_t_bf16_`): per-layer
  member vectors, 1.208 GB/layer x 48 layers = ~58 GB.** `.assign()`-ed on every
  backward repack, never freed; invisible to PyTorch param accounting but real in
  RSS. The base weights are FROZEN so this transpose is constant data, re-stored
  per layer purely as storage. A 1.2 GB shared `bwd_weight` pool +
  `bwd_weight_owner_layer` already exist for exactly this but hold nothing the
  BFDOT reads.
- Backward scratch (`estimated_total_scratch_bytes`) ~3.26 GB: LoRA-grad sharded
  1.14 GB + route_grad_x_accum 1.0 GB (fp32) + per-thread temp 0.89 GB.

## Stage 1 (v8): Share transposed base backward weights

Status: implemented / measured / accepted — (in progress)

Collapse the per-layer transposed base weights to a single process-shared buffer
(~1.2 GB), transposed synchronously and in parallel at the start of each layer's
backward. Backward is strictly sequential per layer (PyTorch autograd), so one
shared buffer is race-free once the async base-transpose repack is removed.

- `gate/up/down_proj_t_bf16_` member vectors → static-shared accessors
  (`shared_gate_proj_t()` etc.), one instance for all 48 layers.
- `transpose_base_weights()`: resize the shared buffers once, parallel transpose
  (`omp parallel for` over experts), set `shared_bwd_t_owner_layer`.
- `backward_impl_packed()`: synchronous `transpose_base_weights()` at the start
  if the shared buffer is not already owned by this layer. Timed via a new
  `backward_base_transpose_ms` counter.
- Disable the async base-transpose (`prepare_backward_weights_from_forward` no
  longer transposes; the synchronous path owns it) so nothing writes the shared
  buffer concurrently with a backward read.

Expected: RSS −~56 GB (157→~101 GiB). Latency: the transpose (~2.4 GB
read+write/layer, bandwidth-bound) moves onto the critical path; predicted <1-3%.
If it blows up, upgrade to an async double-buffer (2 x 1.2 GB) to restore overlap.

Risk/watch: correctness (shared buffer must hold the active layer's weights
during its BFDOT) — guarded by sequential backward + owner check + per-commit
reference/dropout tests + e2e loss. HBM peaks must stay flat.

## Results log

**KT remains native ARM BF16; DeepSpeed not used.**

### Stage 1 (v8) ACCEPTED — full 15-step LF source acceptance (GPU 1, 64 threads)

Artifact: `profiling_kt_codex_smoke/v8_accept_sharedT_qwen3_s4096_b4_r64_w5_s10_t64_source/.../b4_s4096`
Strict validation: `PASS gpu_id=1 affinity_count=144 wrappers=48 fw=1440 bw=720`
(BFDOT label intact, `backward_base_transpose_ms` present).

| Metric | v7_accept (before) | v8_accept (after) | Delta |
|---|---:|---:|---:|
| process_rss_peak | 156.862 GiB | 110.873 GiB | **-45.99 GiB (-29.3%)** |
| peak_allocated | 34.479 GiB | 34.479 GiB | = |
| peak_reserved | 44.029 GiB | 44.029 GiB | = |
| avg_step (measured e2e) | 244.648 s | 238.746 s | -5.9 s (run noise) |
| avg_forward | 76.320 s | 76.492 s | +0.2% |
| avg_backward | 166.303 s | 160.407 s | -5.9 s (run noise) |
| backward_base_transpose_ms | n/a | 22.25 ms/layer (1.07 s/step) | new sync transpose |
| backward_grouped_tile_ms | 1649.741 | 1628.442 | -1.3% (noise) |
| backward_base_grad_ms | 13119.780 | 13075.462 | = |
| backward_tile_recompute_ms | 47707.530 | 47004.986 | -1.5% |
| loss max/last | 1.8324/1.4692 | 1.8313/1.4683 | match |

Verdict: **ACCEPT.** Memory dropped meaningfully (-46 GiB / -29% RSS); the only
real added latency is the 1.07 s/step synchronous parallel transpose (+0.45%),
which is within e2e run noise (the measured step was actually lower this run);
HBM flat; losses match v7 (cross-layer shared-buffer correctness confirmed at 48
layers). Per-commit reference + dropout tests 50/50 pass.

Note: the synchronous transpose replaced the async overlap. The measured cost
(22 ms/layer) is small enough that an async double-buffer (2x1.2 GiB, would recover
~1 s/step at +1.2 GiB) is not worth the complexity. Kept synchronous single-buffer.

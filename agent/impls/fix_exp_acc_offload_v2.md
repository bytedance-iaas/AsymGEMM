# Expert Activation Offload V2 Historical Record

This document is frozen as a record of prior V2 attempts, baselines, failures,
and profiling evidence. Do not add new implementation design here. The active
implementation plan lives in `agent/impls/fix_exp_acc_offload_v3.md`.

Goal: reduce peak HBM during Qwen3 MoE LoRA SFT by offloading almost all
forward routed expert activations to pinned CPU memory, then consuming those
activations in backward through grouped CPU-source AsymGEMM/native kernels.

The fair claim is narrow:

```text
With model, optimizer, LoRA config, precision, routing, batch, sequence length,
profiler, and global recompute mode held fixed, replacing expert activation
recompute with expert activation offload plus grouped CPU-source AsymGEMM
fetchback reduces peak HBM.
```

Do not use unrelated memory optimizers to prove this claim. LoRAFusion,
alternate fused MoE stacks, different loss kernels, different checkpointing, or
optimizer changes are useful design references only if both sides of the
comparison get the same change. The main comparison remains:

```text
BACKEND_SPECS=asym_cpuadamwds|norecomp
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false
```

The recorded comparison workflow for these V2 measurements was
`/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`
workflow. `none|true` was the target implementation. `gc-exp|false` was the
expert checkpoint/recompute baseline in this workflow. `none|false` is the
no-expact/no-expert-recompute control that shows whether the workload fits
without either expert memory strategy. Global transformer recompute remains a
useful lower bound, not the primary comparison.

Historical acceptance rule used when rejecting V2 experiments:

1. Preserve and increase the HBM reduction of `none|true`.
2. Improve `none|true` latency whenever the change does not materially increase
   peak HBM or only increases it within a clearly justified, roughly flat
   budget.
3. Reject latency optimizations that erase the memory win, even if they make
   the path faster.

Rollback status, `b4_s6144`, dropout `0.00`, `asym_cpuadamwds|norecomp`:

- The v2 code experiment that reached `none|true` peak allocated
  `143.368 GiB`, peak reserved `158.543 GiB`, forward+backward `62.479 s`,
  and measured step `64.689 s` was rejected and reverted because it did not
  reduce peak allocated HBM versus the earlier activation-offload baseline and
  made latency much worse.
- The current known target baseline to beat is the earlier `none|true`
  activation-offload run: peak allocated `143.368 GiB`, peak reserved
  `158.055 GiB`, average step `42.842 s`, average forward `9.323 s`, average
  backward `33.430 s`.
- The recompute comparison baseline is `gc-exp|false`: peak allocated
  `167.462 GiB`, peak reserved `181.883 GiB`, average step `3.649 s`, average
  forward `1.464 s`, average backward `2.135 s`.
- Post-revert validation on 2026-06-13 with the same LF workflow:
  - `none|true`: peak allocated `146,809.33 MiB` (`143.368 GiB`), peak
    reserved `161,848.00 MiB` (`158.055 GiB`), `lf.training_step.total`
    `44.447 s`, forward+backward `44.359 s`, forward `11.038 s`, backward
    `33.322 s`
  - `gc-exp|false`: peak allocated `171,481.33 MiB` (`167.462 GiB`), peak
    reserved `186,248.00 MiB` (`181.883 GiB`), `lf.training_step.total`
    `4.155 s`, forward+backward `4.105 s`, forward `1.642 s`, backward
    `2.463 s`
  - current memory delta: `none|true` saves `24,672.00 MiB` (`24.094 GiB`)
    allocated HBM and `24,400.00 MiB` (`23.828 GiB`) reserved HBM versus
    `gc-exp|false`
  - current latency reality: `none|true` is still about `10.7x` slower by
    `lf.training_step.total`; memory is the win, latency remains the major
    optimization target
- Any future change is acceptable only if it either lowers peak HBM without
  blowing up latency, or improves latency while keeping peak HBM roughly flat
  against the current `none|true` baseline. Same memory with worse latency is a
  regression and was the reason the V2 experiment was reverted.

The V2 LF profiling records used the workflow script:

```bash
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh
```

Direct unit tests and `scripts/testing/profile_qwen3_activation_offload.py`
were used only for fast kernel or wrapper checks. They did not replace the LF
workflow comparison. The canonical LF profile command shape was:

```bash
cd /home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM
OUTPUT_ROOT="$PWD/outputs/<run_name>" \
GPU_POOL=<gpu_id_or_pool> \
MODEL_SPECS="Qwen/Qwen3-30B-A3B|1" \
BACKEND_SPECS="asym_cpuadamwds|norecomp" \
PROFILERS=source \
ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false \
ASYM_OFFLOAD_MODULES=all \
LORA_DROPOUT=0.00 \
SEQ_LENS=<seq_len> \
PER_DEVICE_TRAIN_BATCH_SIZE=<batch_size> \
MAX_STEPS=<measured_steps> \
WARMUP_STEPS=<warmup_steps> \
PLOT=false \
RUN_POST=false \
/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh --gpus <gpu_id_or_pool>
```

For the recorded canonical `b4_s6144` validation, the intended settings were
`SEQ_LENS=6144`, `PER_DEVICE_TRAIN_BATCH_SIZE=4`, `LORA_DROPOUT=0.00`, and the
same policy axis above. `RUN_POST=false` was used for targeted or single-policy
runs so the script did not launch unrelated follow-up sweeps.

## Preserved Evidence And Baseline Context

These earlier findings remain important implementation context:

- Current canonical workflow, Qwen3-30B-A3B `b4_s6144`, dropout `0.00`,
  backend `asym_cpuadamwds|norecomp`, router `whole`:
  - `gc-exp|false`, implementation `torch checkpoint`:
    - peak allocated `167.462 GiB`
    - peak reserved `181.883 GiB`
    - average step `3.649 s`
    - average forward `1.464 s`
    - average backward `2.135 s`
  - `none|true`, implementation `activation offload`:
    - peak allocated `143.368 GiB`
    - peak reserved `158.055 GiB`
    - average step `42.842 s`
    - average forward `9.323 s`
    - average backward `33.430 s`
  - interpretation: current `none|true` already saves about `24.094 GiB`
    allocated HBM versus `gc-exp|false`, but it is about `11.7x` slower by
    average step time. The staged plan must preserve the memory win first while
    removing the avoidable latency from full staging, CPU idle regions, full
    FP32 workspaces, and repeated CPU-source passes.

- Rejected v2 experiment observations, kept only as evidence for future
  redesign:
  - these code changes were reverted from the hot path because the full LF
    workflow kept peak allocated HBM at the earlier `none|true` baseline while
    worsening latency; do not re-land any of these paths unless the LF
    acceptance gate passes
  - replacing the old full CUDA `act_for_down_base` and
    `dgate_up_for_gate_up_base` stages with CPU-source base kernels removes the
    target wide stage tags in small SM100 tests
  - switching gate/up LoRA-B backward from staged CUDA gradient halves to the
    CPU-source LoRA-B helper reduces small-profile H2D stage traffic from about
    `90 MB` to about `98 KB` and reduces `max_stage_bytes_live` from about
    `45 MB` to low-rank size
  - splitting target-shape LoRA-B backward so `dS` uses grouped CPU-left
    AsymGEMM and `grad_b` uses the native CPU-source reducer improves the
    isolated representative helper shape
    `m=1024,n=11008,rank=8,groups=8` from about `25.77 ms` for the old
    combined helper to about `0.90 ms` for the auto split path, while preserving
    the low-rank-only HBM stage profile
  - the canonical LF target uses LoRA rank `64` and Qwen3 expert width `768`;
    the first Stage 2 auto guard (`out_dim >= 1024 && rank <= 16`) missed this
    shape, so `expact_lora_b_ds_cpu_left_calls=0` in the first full LF run
  - focused standalone timing for the canonical rank-64 LoRA-B region shows the
    split is still faster when width is at least `512`:
    - `m=24576,n=768,rank=64,groups=128`: forced split about `7.99 ms`, current
      auto/old wrapper about `43.9 ms`, raw old native helper about `20.0 ms`
    - `m=196608,n=768,rank=64,groups=128`: forced split about `72.3 ms`, raw
      old native helper about `157.5 ms`
    - `m=256,n=512,rank=64,groups=8`: forced split about `0.275 ms`, current
      auto/old wrapper about `0.374 ms`
    - `m=256,n=256,rank=64,groups=8`: forced split about `0.330 ms`, current
      auto/old wrapper about `0.255 ms`; this remains a fallback case
  - after the LoRA-B split, the small Qwen3 expert profile
    `tokens=1024,hidden=4096,intermediate=11008,rank=8` reports activation
    offload peak allocated `345,288,704` bytes versus current-asym
    `412,918,272` bytes, `max_stage_bytes_live=32,768` bytes, `h2d_bytes=98,304`
    bytes, and average activation-offload step `149.98 ms`
  - first complete Stage 1/2A canonical LF b4_s6144 source run, before enabling
    rank-64 split auto:
    - `none|true`: peak allocated `143.368 GiB`, peak reserved `158.543 GiB`,
      forward+backward `69.067 s`, measured step `71.375 s`
    - `gc-exp|false`: peak allocated `167.462 GiB`, peak reserved `181.883 GiB`,
      forward+backward `8.374 s`, measured step `10.687 s`
    - `none|false`: OOM in first forward after peak allocated `182.746 GiB`
    - `none|true` saved about `24.094 GiB` allocated HBM versus `gc-exp|false`,
      but latency regressed versus the earlier `42.842 s` offload baseline
  - activation attribution for the same `none|true` LF run reported `48`
    activation-offload rows, `max_stage_bytes_live=16,842,752` bytes, aggregate
    `d2h_bytes=57,400,098,816`, aggregate `h2d_bytes=2,425,356,288`, and stage
    tags limited to `X`, `gate`, `up`, `S_*`, and `dact`; no old full
    gate/up-gradient stage tags reappeared
  - after enabling rank-64 LoRA-B split auto:
    - `expact_lora_b_ds_cpu_left_calls=576`
    - peak allocated/reserved HBM unchanged at `146,809.33 MiB` /
      `162,348.00 MiB`
    - forward+backward improved from `69.067 s` to `64.016 s`
    - backward improved from `52.804 s` to `47.841 s`
  - after Stage 4A paired LoRA-A forward via one grouped CPU-left call over
    concatenated gate/up LoRA-A weights:
    - `expact_lora_a_forward_grouped_calls` dropped from `864` to `576` while
      `expact_lora_a_pair_forward_grouped_calls` stayed `288`
    - `expact_cpu_source_kernel_bytes` dropped from `1,337,437,716,480` to
      `1,182,214,914,048`
    - peak allocated/reserved HBM stayed `146,809.33 MiB` / `162,348.00 MiB`
    - forward+backward improved from `64.016 s` to `62.479 s`; forward improved
      from `16.174 s` to `13.619 s`, while backward regressed from `47.841 s`
      to `48.860 s`
    - this is not an acceptable result because it remains slower than the
      earlier `42.842 s` `none|true` baseline at the same peak allocated HBM
  - this observation does not invalidate the memory-first direction; it means
    future work must first recover or improve the accepted baseline, then only
    keep CPU-source/pairing changes that lower memory or improve latency under
    the same memory budget

### Failed V2 Attempt: Why It Did Not Lower HBM

This attempt is rejected. It must stay recorded so the same change is not
reintroduced under a different name.

Measured baseline to beat, from
`outputs/expact_vs_gc_exp_b4s6144_drop000_20260613T074927Z/.../polnone__routerwhole__expact1/b4_s6144`:

- peak allocated HBM: `146,809.33 MiB` (`143.368 GiB`)
- peak reserved HBM: `161,848.00 MiB` (`158.055 GiB`)
- `lf.training_step.total`: `42,841.621 ms`
- `step.forward + step.backward`: `42,753.172 ms`
- `step.forward`: `9,323.455 ms`
- `step.backward`: `33,429.717 ms`

Rejected v2 Stage 4 result, from
`outputs/expact_v2_stage4_pair_lora_a_b4s6144/.../polnone__routerwhole__expact1/b4_s6144`:

- peak allocated HBM: `146,809.33 MiB` (`143.368 GiB`)
- peak reserved HBM: `162,348.00 MiB` (`158.543 GiB`)
- `lf.training_step.total`: `62,669.398 ms`
- `step.forward + step.backward`: `62,479.118 ms`
- `step.forward`: `13,618.883 ms`
- `step.backward`: `48,860.235 ms`

That is a strict regression: same peak allocated HBM, higher reserved HBM, and
about `19.7 s` slower forward+backward than the accepted activation-offload
baseline. Stage 2 and Stage 3 had the same peak allocated HBM and were also
slower (`64.016 s` and `69.067 s` forward+backward respectively), so Stage 4
only improved a bad experiment; it did not beat the real baseline.

The HBM failure is explained by the measured peak attribution, not by a guess:

- The v2 memory breakdown reports actual peak allocated HBM
  `153,940,737,024` bytes, exactly matching `146,809.33 MiB`.
- The largest HBM rows at that peak are not the removed H2D stage buffers:
  - norms saved activations: `55,095,582,720` bytes (`51.312 GiB`)
  - attention saved activations: `36,104,545,024` bytes (`33.625 GiB`)
  - routed-expert saved activations: `25,933,726,464` bytes (`24.153 GiB`)
  - loss saved activations: `9,996,304,908` bytes (`9.310 GiB`)
  - routed-expert inferred temporary workspace: `6,793,698,932` bytes
    (`6.327 GiB`)
  - routed-expert trainable weights: `6,643,777,536` bytes (`6.188 GiB`)
  - routed-expert gradients: `6,643,777,536` bytes (`6.188 GiB`)
- The v2 expact stats show `full_activation_stage_bytes=0`, but the maximum
  live H2D stage bytes were only `808,452,096` bytes (`0.753 GiB`) and limited
  to low-rank tags `S_down_for_dB`, `S_gate_for_dB`, and `S_up_for_dB`.

Therefore removing `act_for_down_base`, `dgate_up_for_gate_up_base`, and wide
LoRA-B staging did not move the global allocator peak because those buffers were
not the dominant live HBM owner at the measured peak after the rewrite. The
attempt improved some local stage counters, but the full workflow peak was
already controlled by larger saved-activation/workspace groups. A future memory
optimization must first identify which exact tensors in the peak stack are
movable under the fair comparison. If the peak remains norms/attention/loss,
that is outside the expert-only offload claim unless the same non-expert change
is applied to the recompute baseline too.

There is also an attribution risk that must be resolved before the next memory
change: the v2 breakdown still labels about `24.153 GiB` as routed-expert saved
activations while expact stats say no full activation stage is live. This may be
real final/low-rank expert state, stale saved-tensor labeling, or an attribution
bucket that is too coarse. Do not optimize based only on the component label;
inspect the concrete tensor names/shapes/lifetimes or improve attribution until
the row explains which tensors are actually live.

The latency failure is also measured:

- v2 Stage 4 reports aggregate expact CPU-source kernel bytes
  `1,182,214,914,048` bytes, about `1.10 TiB`.
- Activation-offload stats report aggregate D2H bytes `57,400,098,816` bytes
  (`53.458 GiB`), CPU-source stream-wait bytes `54,166,290,432` bytes
  (`50.446 GiB`), and CPU-read wait bytes `29,104,275,456` bytes
  (`27.105 GiB`).
- The paired LoRA-A forward experiment reduced its own call count and improved
  the bad Stage 2 forward time, but the final v2 forward was still
  `13.619 s` versus the accepted baseline `9.323 s`, and backward was
  `48.860 s` versus `33.430 s`.

The likely mechanism, supported by these counters, is that the CPU-source base
and LoRA rewrites traded small or non-peak HBM stages for much more host-memory
traffic, stream waits, CPU-side concat/read waits, and extra CPU-source work.
That is exactly the tradeoff the acceptance rule rejects when peak HBM does not
drop.

Next implementation rule from this failure:

- Do not re-land CPU-source base or paired LoRA changes just because they remove
  a local stage tag.
- First run a diagnostic memory-attribution profile on the current accepted
  baseline and identify the actual peak-owner tensors by name/shape/lifetime.
- A candidate memory change must reduce the global LF peak or a clearly
  reported expert-block peak. If it only changes `max_stage_bytes_live` while
  the global peak stays flat, it is not a memory win.
- A candidate latency change must be compared against the accepted
  `42.842 s` `none|true` baseline, not against another rejected v2 stage.

- Qwen3 `4x8192`, recompute on:
  - `expact0`: peak allocated `73.546 GB`, measured step `14.708 s`
  - `expact1`: peak allocated `73.546 GB`, measured step `107.291 s`
- Qwen3 `4x8192`, recompute off:
  - `expact0`: OOM in first forward, peak allocated about `196.158 GB`
  - `expact1`: OOM in first forward, peak allocated about `195.853 GB`
- Qwen3 `2x8192`, recompute off:
  - `expact0`: OOM in first forward/loss, peak allocated `193.732 GB`; the
    failing allocation was cross-entropy trying to allocate about `9.27 GiB`
  - `expact1`: completed; peak allocated `135.628 GB`, peak reserved
    `141.503 GB`, measured step `44.950 s`, forward `10.578 s`,
    backward `32.939 s`

Interpretation to preserve:

- current expact can reduce enough activation HBM to make smaller no-recompute
  runs fit
- current expact does not yet beat the matched recompute path because it
  recreates wide HBM stages and full FP32 workspaces
- the `4x8192` no-recompute failure is still dominated by allocations outside
  the intended CPU-source schedule
- loss/cross-entropy peaks can hide the expert result, so expert-block peaks
  and loss peaks must be reported separately

Keep the three baseline categories separate:

- canonical workflow comparison:
  `BACKEND_SPECS=asym_cpuadamwds|norecomp` and
  `ASYMM_EXP_ACT_POLICIES=none|true,gc-exp|false,none|false`
- global recompute lower bound:
  `BACKEND_SPECS=asym_cpuadamwds|recomp` and
  `ASYMM_EXP_ACT_POLICIES=none|false`
- optional global-plus-expert recompute sanity check:
  `BACKEND_SPECS=asym_cpuadamwds|recomp` and
  `ASYMM_EXP_ACT_POLICIES=gc-exp|false`

Older thresholded token-policy examples are not the default workflow for this
plan. Use them only as explicit extra experiments if we need to compare custom
thresholded recompute against `gc-exp`.

### Current-Baseline Attribution Probe: Profiler Perturbation

On 2026-06-13, two diagnostic runs were attempted on the reverted current
`none|true` implementation at `b4_s6144` to answer whether remaining expert
lifetime issues are real:

1. Full live-at-peak breakdown:
   `outputs/diag_current_expact_attr_b4s6144_20260613T204923Z/.../polnone__routerwhole__expact1/b4_s6144`
   with `PROFILE_MEMORY_ATTRIBUTION=true` and
   `PROFILE_MEMORY_BREAKDOWN=true`.
2. Lighter saved-tensor attribution:
   `outputs/diag_current_expact_saved_attr_b4s6144_20260613T205424Z/.../polnone__routerwhole__expact1/b4_s6144`
   with `PROFILE_MEMORY_ATTRIBUTION=true` and
   `PROFILE_MEMORY_BREAKDOWN=false`.

Both runs OOMed during backward with the same failure:

- attempted allocation: `13.91 GiB`
- PyTorch allocated at failure: about `172.35 GiB`
- partial profile peak allocated: `192,629,625,856` bytes
  (`183,705.93 MiB`, `179.400 GiB`)
- partial profile peak reserved: `193,722,318,848` bytes
  (`184,748.00 MiB`, `180.418 GiB`)

This is not the accepted runtime baseline. The accepted reverted run without
saved-tensor hooks is still `146,809.33 MiB` (`143.368 GiB`) peak allocated.
The profiler inflated the peak by `36.03 GiB`, which exactly matches
`48 * 768 MiB`. The saved-tensor partial profile identifies these as one
bf16 `[196608, 2048]` tensor per layer from
`forward.layers.N.mlp.experts.scatter_combine`:

- per-layer routed expert output before scatter: `768 MiB`
- all 48 layers: `36.0 GiB`
- top saved-tensor owner from the partial run: `lf.forward_loss`
  `118,920.24 MiB`
- expert scatter owners: `48` rows of `768.38 MiB`
- aggregate expert-owned saved tensors in the partial run: about
  `43,290.09 MiB`

Do not treat the hook-inflated `183.706 GiB` as a real baseline. Treat it as a
profiling failure mode. The evidence still matters because it explains a
remaining lifetime risk: `scatter_contiguous` currently uses ordinary autograd,
so the routed expert output `down` can be saved for scatter backward. That
tensor is outside the internal gate/up/act offload path and is route-expanded
by top-k. For Qwen3-30B-A3B `b4_s6144`, it is `768 MiB` per layer.

Follow-up design and new profiling conclusions from this point forward belong
in `agent/impls/fix_exp_acc_offload_v3.md`.

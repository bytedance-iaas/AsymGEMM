# LF LoRA-SFT Memory Status

Date: 2026-06-14

Model: `meta-llama/Llama-4-Scout-17B-16E`

Primary workload:

- backend: `asym`
- precision: `bf16`
- profiler: `source`
- recompute: enabled
- router mode: `whole`
- batch and sequence: `b4_s8192`
- LoRA: `r64_a16`

## Current AsymGEMM System

These are the latest Llama4 runs through:

`/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`

Config: `b4_s8192`, `drop008`, `max_steps=10`, `warmup_steps=5`.

| Offload modules | Peak allocated HBM | Peak reserved HBM | Activations at peak | Temp/workspace | Persistent HBM residual | Runtime | Trainable params | CPU base weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `routed_experts` | 127.95 GiB | 144.00 GiB | 66.42 GiB | 21.22 GiB | 40.31 GiB | 223.7s | 2,186,280,960 | 180.00 GiB |
| `all` | 105.20 GiB | 122.32 GiB | 66.11 GiB | 21.64 GiB | 17.45 GiB | 301.6s | 2,210,791,424 | 201.23 GiB |

`ASYM_OFFLOAD_MODULES=all` versus `routed_experts`:

| Delta | Value |
|---|---:|
| Peak allocated HBM | -22.75 GiB |
| Peak reserved HBM | -21.68 GiB |
| Activations at peak | -0.31 GiB |
| Temp/workspace | +0.41 GiB |
| Persistent HBM residual | -22.86 GiB |
| Runtime | +77.9s, +34.8% |
| CPU base weights | +21.24 GiB |

Extra CPU-resident base weights in `all`:

| Component | CPU base weight |
|---|---:|
| `shared_experts` | 11.25 GiB |
| `attention` | 6.13 GiB |
| `embed_tokens` | 1.93 GiB |
| `lm_head` | 1.93 GiB |
| `router` | 0.01 GiB |
| `norms` | 0.00 GiB |

Runtime verification:

| Offload modules | Asym forward calls | Asym dx calls | Torch fallback | Reference fallback |
|---|---:|---:|---:|---:|
| `routed_experts` | 2,880 | 1,440 | 0 | 0 |
| `all` | 16,455 | 6,495 | 0 | 0 |

Conclusion for the current system:

- `all` materially reduces HBM on Llama4 because Llama4 has large `shared_experts`, attention, embedding, and lm_head weights outside routed experts.
- The saving is almost entirely persistent HBM, not activations: activations stay about 66 GiB.
- The cost is runtime: `all` is about 35% slower than `routed_experts` for this source-profile run.
- The GEMM-bearing offloaded modules are being fetched through AsymGEMM: no torch fallback and no reference fallback were reported.
- `embed_tokens` is still CPU gather plus activation copy, not GEMM.
- Stateless Llama4 `qk_norm` has no weight to offload; it is now allowed under `norms` instead of failing strict `all`.

Artifacts:

- `routed_experts`: `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling/asym_long_sft_smoke__llama-4-scout-17b-16e__s8192__lora__lf__bf16/compare_llama4_routed_20260611_103001__drop008/asym__source__recomp__polnone__routerwhole/b4_s8192/source_profile.json`
- `all`: `/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/profiling/asym_long_sft_smoke__llama-4-scout-17b-16e__s8192__lora__lf__bf16/compare_llama4_all_fixed_20260611_103917__drop008/asym__source__recomp__polnone__routerwhole/b4_s8192/source_profile.json`

## CPUAdamW Timing Artifacts

Run: `Llama-4-Scout-17B-16E`, `b4_s8192`, `drop008`, `warmup=5`, `measure=10`, `ASYM_OFFLOAD_MODULES=all`, source profiler, memory attribution/breakdown disabled.

Timing source: `heartbeat_dataloader_interval`

Artifact root:

`/tmp/asym_cpuadam_compare_e2e_nomem_kevinni_1781220484/asym_long_sft_smoke__lora__lf__bf16/llama-4-scout-17b-16e__gpus1__b4_s8192_w5_s10_r64_a16_drop008`

| Backend | E2E step incl optimizer | Fwd+Bwd | Update/loop side | Optimizer substage | Train runtime | Peak alloc HBM | Peak reserved HBM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `asym` | 17.043s | 16.262s | 0.782s | 0.079s | 257.98s | 105.20 GiB | 122.32 GiB |
| `asym_cpuadamwds` | 18.294s | 16.309s | 1.984s | 1.200s | 281.20s | 97.06 GiB | 110.84 GiB |

| Delta: `asym_cpuadamwds` - `asym` | Value |
|---|---:|
| E2E step incl optimizer | +1.250s, +7.3% |
| Fwd+Bwd | +0.047s, +0.3% |
| Update/loop side | +1.203s |
| Optimizer substage | +1.122s |
| Train runtime | +23.22s, +9.0% |
| Peak alloc HBM | -8.15 GiB, -7.7% |
| Peak reserved HBM | -11.48 GiB, -9.4% |

| CPUAdamW detail | Value |
|---|---:|
| trainable LoRA params | 2,210,791,424 |
| grad copy to CPU | 0.778s |
| CPU Adam step | 0.642s |
| weight copyback | 0.024s |
| CPU master weights | 8.24 GiB |
| CPU optimizer state | 16.29 GiB |

| Artifact | Status |
|---|---|
| `lat.md` | optimizer-inclusive e2e rows present |
| `summary.md` | optimizer-inclusive e2e rows present |
| `profile.json` | `trainer.timing.source=heartbeat_dataloader_interval` |
| `source_profile.json` | `trainer.timing.source=heartbeat_dataloader_interval` |
| `step_samples.csv/json` | `step_milliseconds_source`, `forward_backward_milliseconds`, `trainer_e2e_step_milliseconds`, `optimizer_update_side_milliseconds` present |

## Zero3 and SuperOffload Baselines

These are existing baseline artifacts from the profiling tree. They are not fully apples-to-apples with the current `all` run:

- baselines use `drop010`; current `all`/`routed_experts` use `drop008`;
- Zero3/SuperOffload use `routerhf`; AsymGEMM uses `routerwhole`;
- Zero3/SuperOffload trained 223,346,688 LoRA params;
- AsymGEMM trained about 2.19B to 2.21B LoRA params because its wrapper/module exposure makes `lora_target=all` cover a much larger graph.

### Historical b4_s7168, drop010

| Backend | Peak allocated HBM | Peak reserved HBM | Activations at peak | Temp/workspace | Persistent HBM residual | Runtime | Trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zero3_offload` | 72.63 GiB | 80.83 GiB | 59.49 GiB | 13.14 GiB | ~0 GiB | 375.0s | 223,346,688 |
| `superoffload` | 72.63 GiB | 80.83 GiB | 59.49 GiB | 13.14 GiB | ~0 GiB | 377.2s | 223,346,688 |
| old `asym` routed experts | 116.59 GiB | 131.38 GiB | 58.26 GiB | 18.02 GiB | 40.31 GiB | 184.6s | 2,186,280,960 |

Old `asym` routed experts versus `superoffload` at `b4_s7168`:

| Delta | Value |
|---|---:|
| Peak allocated HBM | +43.96 GiB |
| Activations at peak | -1.23 GiB |
| Temp/workspace | +4.88 GiB |
| Persistent HBM residual | +40.31 GiB |
| Runtime | -192.6s, 51.1% faster |

### Historical b4_s8192, drop010

| Backend | Peak allocated HBM | Peak reserved HBM | Activations at peak | Temp/workspace | Persistent HBM residual | Runtime | Trainable params |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zero3_offload` | 82.23 GiB | 116.82 GiB | 66.95 GiB | 15.28 GiB | ~0 GiB | 422.1s | 223,346,688 |
| old `asym` routed experts | 128.07 GiB | 164.46 GiB | 66.42 GiB | 21.34 GiB | 40.31 GiB | 259.5s | 2,186,280,960 |
| current `asym` `routed_experts` | 127.95 GiB | 144.00 GiB | 66.42 GiB | 21.22 GiB | 40.31 GiB | 223.7s | 2,186,280,960 |
| current `asym` `all` | 105.20 GiB | 122.32 GiB | 66.11 GiB | 21.64 GiB | 17.45 GiB | 301.6s | 2,210,791,424 |

Comparison to `zero3_offload` at `b4_s8192`:

| System | Peak allocated delta vs `zero3_offload` | Runtime delta vs `zero3_offload` |
|---|---:|---:|
| old `asym` routed experts | +45.84 GiB | -162.6s, 38.5% faster |
| current `asym` `routed_experts` | +45.72 GiB | -198.4s, 47.0% faster |
| current `asym` `all` | +22.97 GiB | -120.5s, 28.5% faster |

There is no matching `b4_s8192` SuperOffload source artifact in the current profiling tree. The available SuperOffload source artifact is `b4_s7168`.

## Qwen3 LF/ZeRO Expert LoRA Surface Fix

Run: `Qwen/Qwen3-30B-A3B`, `zero3_offload|recomp`, `b4_s4096`, `drop000`, `warmup=5`, `measure=10`, source profiler, memory attribution/breakdown/snapshot disabled.

This fixes the earlier LF/ZeRO trainable-surface mismatch. The accepted default is now `split-target-parameters`: it keeps the split gate/up/down expert LoRA parameter surface while preserving the original Qwen grouped expert execution path.

| Qwen expert LoRA impl | Peak allocated HBM | Peak reserved HBM | Avg step | Avg forward | Avg backward | Trainable params | PEFT expert params | Qwen split expert params | Fallback | Loss max/last/train |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `peft-target-parameters` | 33.107 GiB | 38.756 GiB | 4.932s | 1.436s | 3.496s | 2,570,059,776 | 2,516,582,400 | 0 | 0 | 2.326 / 1.273 / 1.914 |
| `split-target-parameters` | 33.138 GiB | 38.268 GiB | 6.026s | 1.671s | 4.355s | 3,375,366,144 | 3,321,888,768 | 3,321,888,768 | 0 | 2.273 / 1.231 / 1.874 |

Result:

- `split-target-parameters` is accepted: it has the corrected expert LoRA coverage while preserving the fast grouped expert path.
- `peft-target-parameters` remains a useful fast PEFT baseline, but it trains fewer expert LoRA parameters and is not the corrected all-expert surface.
- Only `split-target-parameters`, `peft-target-parameters`, and `off` remain selectable.
- A default-mode smoke without `LF_EXPERT_LORA_IMPLS` selected `split-target-parameters`, reported `3,321,888,768` expert LoRA params, and had `reference_fallback_count=0`.

Artifacts:

- A/B root: `profiling/lf_lora_split_accept/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000`
- default smoke: `profiling/lf_lora_split_default_smoke/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s2_r64_a16_drop000/zero3_offload__source__recomp__polnone__routerhf__expact0__attnact0__layeract0__loraafwdcpu__qwenexpertsplit-target-parameters/b4_s4096/source_profile.json`

## Qwen3 Attention and Expert Activation Offload Snapshot

Run: `Qwen/Qwen3-30B-A3B`, `b4_s4096`, `drop000`, `warmup=5`, `measure=10`, source profiler, memory attribution/breakdown/snapshot disabled.

Artifact root:

`/home/kevinni/AsymGEMM-SFT/third_party/AsymGEMM/reports/attn_act_offload/lf_memory_b4_post_s4_compare/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s4096_w5_s10_r64_a16_drop000`

| Backend spec | Policy | Implementation | Peak allocated HBM | Peak reserved HBM | Avg step | Avg forward | Avg backward |
|---|---|---|---:|---:|---:|---:|---:|
| `asym_cpuadamwds|norecomp` | `none|true|true` | expert + attention activation offload | 58.343 GiB | 63.406 GiB | 46.161s | 11.393s | 34.623s |
| `asym_cpuadamwds|norecomp` | `none|true|false` | expert activation offload only | 102.312 GiB | 107.547 GiB | 42.197s | 9.626s | 32.462s |
| `asym_cpuadamwds|norecomp` | `gc-exp|false|false` | expert checkpoint baseline | 126.312 GiB | 131.414 GiB | 3.835s | 1.430s | 2.342s |
| `asym_cpuadamwds|norecomp` | `none|false|false` | no expact/no recompute | 170.525 GiB | 179.990 GiB | 3.037s | 1.429s | 1.551s |
| `asym_cpuadamwds|recomp` | `none|false|false` | global gradient checkpointing | 37.422 GiB | 43.016 GiB | 4.356s | 1.413s | 2.890s |
| `zero3_offload|recomp` | `none|false|false` | ZeRO-3 offload + global GC | 33.009 GiB | 38.229 GiB | 2.575s | 0.944s | 1.579s |

Validation notes:

- The `asym_cpuadamwds|recomp` row is the real AsymGEMM CPU-Adam/global-GC baseline for `none|false|false`.
- The `zero3_offload|recomp` row in this older table completed and wrote a source profile, but the post-run trainable-surface guard failed. It logged only attention LoRA modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with 53,477,376 trainable params and no captured expert LoRA. Treat this row as a measured historical artifact. Use the accepted `split-target-parameters` row above for current LF/ZeRO Qwen3 comparisons.
- The attention activation offload rows materially reduce HBM, but their current latency is not acceptable: they reduce memory by tens of GiB while increasing step time to about 42-46s. They should not be accepted as production changes until the fetch/backward path is redesigned.

## Qwen3 Expert Activation-Offload LoRA-A A/B

Run: `Qwen/Qwen3-30B-A3B`, `b4_s4096`, `drop000`, `warmup=5`, `measure=10`, source profiler, memory attribution/breakdown/snapshot disabled.

Backend/policy: `asym_cpuadamwds|norecomp`, `none|true|true|true` (`expert`, `attention`, and `layer` activation offload enabled).

| Forward LoRA-A mode | Peak allocated HBM | Peak reserved HBM | Source fwd+bwd step | Avg forward | Avg backward | Forward-end HBM | Saved CPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cpu` | 34.593 GiB | 39.676 GiB | 44.930s | 11.318s | 33.612s | 16.046 GiB | 1.469 GiB |
| `hbm` | 34.593 GiB | 39.676 GiB | 36.631s | 2.831s | 33.800s | 16.046 GiB | 1.469 GiB |

| Forward LoRA-A mode | Trainer E2E step | AsymGEMM fwd/dx | Expert CPU-left LoRA-A calls | Expert HBM LoRA-A calls | Generic CPU-left LoRA-A calls | Loss max / last / train |
|---|---:|---:|---:|---:|---:|---:|
| `cpu` | 46.864s | 242,640 / 344,160 | 103,680 | 0 | 241,920 | 2.335 / 1.267 / 1.916 |
| `hbm` | 38.506s | 242,640 / 344,160 | 0 | 69,120 | 138,240 | 2.326 / 1.261 / 1.915 |

Result:

- `hbm` passed the hard memory acceptance gate: peak allocated HBM delta is `0.000 GiB`, below the `<0.5 GiB` cap, and peak reserved HBM also did not increase.
- `hbm` improved source forward+backward step by `8.299s` (`18.47%`), forward by `8.488s` (`74.99%`), and trainer E2E step by `8.357s` (`17.83%`). Backward increased by `0.188s` (`0.56%`).
- The default remains `cpu` in this patch because the implementation plan's default-promotion latency gate asks for at least `20%` matched avg-step improvement. `hbm` is validated and selectable with `ASYMM_EXPERT_ACT_OFFLOAD_LORA_A_FWD=hbm`.
- Both profiles reported `attention+expert LoRA`, `3,375,366,144` trainable params, `3,321,888,768` expert LoRA params, and `reference_fallback_count=0`.

Artifacts:

The artifact directory names are historical from the run that produced the
numbers; the current LoRA-A selector values are only `cpu` and `hbm`.

## Interpretation

The old AsymGEMM memory gap versus Zero3/SuperOffload was mostly not activation memory. Activations were similar across systems. The gap came from persistent GPU state that Zero3/SuperOffload shard or offload:

- frozen dense model weights outside routed experts;
- LoRA params;
- LoRA grads;
- optimizer state;
- extra temporary/workspace.

With `ASYM_OFFLOAD_MODULES=all`, the frozen dense model weight part is much smaller:

- current `routed_experts` persistent HBM residual: 40.31 GiB;
- current `all` persistent HBM residual: 17.45 GiB.

So `all` closes about 22.9 GiB of the HBM gap on Llama4. The remaining gap versus Zero3/SuperOffload is still expected because AsymGEMM keeps the large trainable LoRA surface, gradients, optimizer state, and some workspace on GPU. The comparison is still not clean until the LoRA target surface and router mode are aligned across backends.

Bottom line:

- `routed_experts` is fastest, but high HBM.
- `all` is the current best AsymGEMM memory mode for Llama4: much lower HBM than `routed_experts`, still faster than the old Zero3-Offload source run at `b4_s8192`, but slower than `routed_experts`.
- Zero3/SuperOffload still have the lowest HBM in the available baselines, partly because they train a much smaller LoRA surface.

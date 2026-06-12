# Llama4 LF LoRA-SFT Memory Status

Date: 2026-06-11

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

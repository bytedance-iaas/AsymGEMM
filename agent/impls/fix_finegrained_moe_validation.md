# Fine-Grained MoE Offload Validation Log

This log is the companion artifact for `agent/impls/fix_finegrained_moe.md`.
Update it before and after every stage. Do not use undocumented runs as evidence.

## Global Rules

- Run validation jobs serially. Do not run experiments in parallel while debugging this
  path.
- Keep `UNSLOTH_GC_OUTER_HBM_EVERY_N=0` for all target and baseline comparisons.
- Keep the external policy tuple `none|false|false|false|false|false`.
- For the target, old `ASYMM_EXPERT_ACT_OFFLOAD` must be false.
- For the target, the new Qwen3 MoE fine-grained flag/counters must prove the path.
- Compare `asym_cpuadamwds` primarily against `superoffload_mem`, not
  `*_nocpuadamw`.

## Scoreboard Template

```text
Model: qwen3-30b-a3b    LoRA: r64/a16/d0.00    CPUAdamW family: yes
Workload   Backend            Recompute          Config                  fwd_s  bwd_s  opt_s  step_s  fwd_H  bwd_H  step_H    RAM  Status
---------  -----------------  -----------------  ----------------------  -----  -----  -----  ------  -----  -----  ------  -----  --------
s80000.b8  superoffload_mem   unsloth            none + ligerloss1        29.6  130.3    0.0   162.4   91.9  176.9   176.9  359.9  complete
s80000.b8  superoffload_mem   unsloth-off        none + ligerloss1                                                 PENDING
s80000.b8  asym_cpuadamwds    recomp-off-full-fg moefg1 + ligerloss1                                              PENDING
```

Known complete `superoffload_mem|unsloth` artifact:

```text
profiling_q3_30b_a3b_s80000/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b8_s80000_ga1_w0_s1_r64_a16_drop000/superoffload_mem__source__unsloth__polnone__routerhf__expact0__attnact0__layeract0__layergc0__sdparecomp0__loraafwdhbm__actrecomp0__xunpack0__ligerloss1/b8_s80000_ga1/source_profile.json
```

Known `superoffload_mem|unsloth` peak decomposition:

```text
saved_activations/routed_experts      69.75 GiB
temporary_workspace/routed_experts    55.78 GiB
saved_activations/norms               20.84 GiB
saved_activations/attention           13.99 GiB
saved_activations/router               2.81 GiB
live_activation/embed_tokens           2.44 GiB
live_activation/router                 0.16 GiB
actual peak allocated                 176.95 GiB
actual peak reserved                  180.69 GiB
```

## Stage 0: Config Truth And Path Ownership

stage: 0

expected run labels:

- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

expected config:

- `recomp_off_stage=full-fg`
- `use_unsloth_gc=true`
- `unsloth_gc_recompute_save_on_cpu=true`
- `unsloth_gc_outer_hbm_every_n=0`
- `asymm_qwen3_moe_finegrained_offload=true`
- `asymm_expert_act_offload=false`
- `asymm_attn_act_offload=true`
- CPUAdamW grad/weight offload enabled by `asym_cpuadamwds`

expected path labels:

- includes `moefg1`
- includes `expact0`
- includes `attnact1`
- does not hide MoE-fg state only inside `recomp-off-full-fg`

expected wrappers/counters:

- setup report has `qwen3_moe_finegrained_offload_enabled=true`
- setup report has `qwen3_moe_finegrained_offload_wrapped > 0`
- old expact counters are zero until a diagnostic run explicitly enables old expact
- dense fine-grained counters are zero for MoE

observed:

- PENDING

conclusion:

- PENDING

## Stage 1: s80000 Baselines

stage: 1

run labels:

- `superoffload_mem|unsloth|ligerloss1`
- `superoffload_mem|unsloth-off|ligerloss1`

commands:

```bash
RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm

RUNS='q3-30b-a3b|1 ; superoffload_mem|unsloth-off|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm
```

expected:

- both run serially;
- both use `superoffload_mem`;
- both use `ligerloss1`;
- both use the same model, LoRA, batch, sequence, and grad accumulation;
- unsloth-off has `UNSLOTH_GC_RECOMPUTE_SAVE_ON_CPU=true`;
- no Asym counters fire.

observed:

- `superoffload_mem|unsloth`: complete; metrics recorded in scoreboard.
- `superoffload_mem|unsloth-off`: PENDING.

conclusion:

- PENDING until unsloth-off is complete or has an audited failure.

## Stage 2: Current Asym MoE Control

stage: 2

run labels:

- `asym_cpuadamwds|recomp-off-base|ligerloss1`
- `asym_cpuadamwds|recomp-off-attn|ligerloss1`
- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

workload:

```text
s8192.b8 first; larger only after config truth passes
```

expected:

- base/attn prove outer Unsloth GC and CPUAdamW offload;
- current `full-fg` is not accepted as MoE-fg unless new MoE counters fire;
- old expact path stays off;
- dense fine-grained counters stay zero for MoE.

observed:

- PENDING

conclusion:

- PENDING

## Stage 3: New MoE Fine-Grained Unit/Parity Tests

stage: 3

test cases:

- balanced static routes,
- skewed static routes,
- repeated experts,
- empty expert groups,
- learned router if the new wrapper owns routing,
- bf16 CUDA,
- LoRA r64/a16/drop0.00,
- CPUAdamW weight gather/release if touched.

expected:

- outputs finite;
- gradients finite;
- LoRA gradients close to existing Qwen3 MoE reference;
- no fused `gate_up [R,2I]` saved for backward;
- no fused `grad_gate_up [R,2I]`;
- `stage_concat_columns` count is zero.

observed:

- PENDING

conclusion:

- PENDING

## Stage 4: Small LF Smoke

stage: 4

run labels:

- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

workloads:

```text
s2048.b8
s8192.b8
```

expected:

- complete one measured step;
- new MoE fine-grained wrapper count > 0;
- `qwen3_moe_finegrained_forward_calls > 0`;
- `qwen3_moe_finegrained_backward_calls > 0`;
- `qwen3_moe_finegrained_stage_concat_columns_calls == 0`;
- old expact counters == 0;
- dense fine-grained counters == 0.

observed:

- PENDING

conclusion:

- PENDING

## Stage 5: Memory-Shape Gate

stage: 5

run labels:

- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

workloads:

```text
s30000.b8
s45000.b8
```

expected:

- routed-expert saved activations and temporary workspace move in the expected
  direction versus baselines;
- if peak is still live gate/up/act overlap, memory details identify exact shapes and
  owners before any sequential-staging change;
- no conclusion from peak number alone.

observed:

- PENDING

conclusion:

- PENDING

## Stage 6: Final s80000 Target

stage: 6

run label:

- `asym_cpuadamwds|recomp-off-full-fg|ligerloss1`

command:

```bash
RUNS='q3-30b-a3b|1 ; asym_cpuadamwds|recomp-off-full-fg|ligerloss1 ; 80000|8|1 ; none|false|false|false|false|false' \
  scripts/lf/profile_lora_lf_test_source.sh --gpus 3 --overwrite true --plot false --max-steps 1 --warmup-steps 0 --output-root profiling_fix_fgm
```

expected:

- complete or produce an audited capacity/failure point;
- new MoE-fg counters prove dispatch;
- lower `step_H` than `superoffload_mem|unsloth-off`, or a precise audited reason why
  not;
- meaningful routed-expert activation/workspace reduction versus
  `superoffload_mem|unsloth`;
- final table includes `fwd_s`, `bwd_s`, `opt_s`, `step_s`, `fwd_H`, `bwd_H`,
  `step_H`, `RAM`.

observed:

- PENDING

conclusion:

- PENDING

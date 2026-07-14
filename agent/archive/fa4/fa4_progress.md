# FA4 LlamaFactory LoRA-SFT Progress

Date: 2026-06-04

## Metrics / Results

Current result: FA4 is integrated and functional in the isolated
`LlamaFactory-fa4` lab, but it does not currently improve speed or memory over
torch SDPA for the tested LF LoRA-SFT workloads. Torch SDPA is already an
optimized fused attention backend on this stack.

### Validation

Artifact:
`third_party/LlamaFactory-fa4/fa4_reports/validate_lf_fa4_rerun_20260604/report.json`

Status: `pass`.

| Probe | Result |
|---|---|
| FA4 import | `flash_attn_func` and `flash_attn_varlen_func` found |
| Direct FA4 attention | 5 cases pass, finite gradients, reported max/relative diffs `0.0` |
| Transformers FA4 probe | `attn_implementation=flash_attention_4`, loss `5.5479`, 16 LoRA grad tensors, peak `70.09 MiB` |
| LF smoke SDPA | loss `5.676`, clean SDPA log marker |
| LF smoke FA4 | loss `5.679`, clean FA4 log marker, relative loss diff `0.000529` |

### Microbench: Attention / MoE Block

Artifact:
`third_party/LlamaFactory-fa4/fa4_reports/bench_fa4_vs_sdpa_20260604/benchmark_report.json`

BF16 CUDA event timing after warmup.

| Test | Accuracy | SDPA median | FA4 median | FA4/SDPA |
|---|---:|---:|---:|---:|
| core attn `b2_s128_qh8_kvh8_d64` | max diff `0.003906`, grad finite | `0.0759 ms` | `0.0997 ms` | `0.761x` |
| core attn `b2_s256_qh16_kvh8_d64` | max diff `0.007812`, grad finite | `0.0670 ms` | `0.0948 ms` | `0.707x` |
| attention module `b2_s256_h1024_qh16_kvh8_d64` | max diff `0.001953` | `0.3008 ms` | `0.3627 ms` | `0.829x` |
| synthetic MoE block | max diff `0.03125`, same routing | `2.0803 ms` | `2.1494 ms` | `0.968x` |

### Qwen3-30B-Like Attention Sweep

Artifacts:

- `third_party/LlamaFactory-fa4/fa4_reports/attention_sweep_qwen3_30b_bs_seq_20260604/attention_sweep_report.json`
- `third_party/LlamaFactory-fa4/fa4_reports/attention_sweep_qwen3_30b_s8192_20260604/attention_sweep_report.json`
- `third_party/LlamaFactory-fa4/fa4_reports/attention_sweep_qwen3_30b_s16384_20260604/attention_sweep_report.json`

Shape: BF16 causal GQA, `q_heads=32`, `kv_heads=4`, `head_dim=128`.

| Shape | Fwd SDPA | Fwd FA4 | Fwd speed | Fwd+Bwd SDPA | Fwd+Bwd FA4 | Fwd+Bwd speed | Peak alloc SDPA -> FA4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b1_s1024` | `0.078 ms` | `0.123 ms` | `0.637x` | `0.580 ms` | `0.886 ms` | `0.655x` | `0.131 -> 0.141 GiB` |
| `b4_s2048` | `0.168 ms` | `0.215 ms` | `0.782x` | `1.070 ms` | `1.128 ms` | `0.949x` | `1.048 -> 1.126 GiB` |
| `b4_s4096` | `0.440 ms` | `0.497 ms` | `0.885x` | `2.552 ms` | `2.610 ms` | `0.978x` | `2.096 -> 2.252 GiB` |
| `b1_s8192` | `0.409 ms` | `0.444 ms` | `0.921x` | `1.967 ms` | `2.052 ms` | `0.958x` | `1.048 -> 1.126 GiB` |
| `b2_s8192` | `0.760 ms` | `0.800 ms` | `0.950x` | `3.770 ms` | `3.688 ms` | `1.022x` | `2.096 -> 2.252 GiB` |
| `b4_s8192` | `1.450 ms` | `1.440 ms` | `1.007x` | `7.207 ms` | `7.149 ms` | `1.008x` | `4.191 -> 4.504 GiB` |
| `b8_s8192` | `2.746 ms` | `3.092 ms` | `0.888x` | `14.009 ms` | `14.638 ms` | `0.957x` | `8.383 -> 9.008 GiB` |
| `b1_s16384` | `1.383 ms` | `1.419 ms` | `0.974x` | `6.233 ms` | `6.243 ms` | `0.998x` | `2.096 -> 2.252 GiB` |
| `b2_s16384` | `2.735 ms` | `2.660 ms` | `1.028x` | `12.307 ms` | `12.608 ms` | `0.976x` | `4.191 -> 4.504 GiB` |
| `b4_s16384` | `5.231 ms` | `5.713 ms` | `0.916x` | `23.296 ms` | `25.915 ms` | `0.899x` | `8.383 -> 9.008 GiB` |
| `b8_s16384` | `10.137 ms` | `12.729 ms` | `0.796x` | `51.438 ms` | `52.612 ms` | `0.978x` | `16.766 -> 18.016 GiB` |

Best isolated forward-only win was `1.028x` at `b2_s16384`, but that same
shape was slower for fwd+bwd (`0.976x`). Best fwd+bwd win was only `1.022x` at
`b2_s8192`. FA4 peak allocation was consistently slightly higher than SDPA.

### Small LoRA-SFT Steady State

Artifact:
`third_party/LlamaFactory-fa4/fa4_reports/steady_state_lora_sft_after_cu13_20260604/steady_state_report.json`

Workload: `Qwen/Qwen3-0.6B`, `cutoff_len=256`, fixed 192-token dataset,
12 total steps, first 5 discarded.

| Backend | Median step | Mean step | P90 step | Loss | First step |
|---|---:|---:|---:|---:|---:|
| SDPA | `0.3198 s` | `0.3197 s` | `0.3212 s` | `4.655` | `0.6844 s` |
| FA4 | `0.3356 s` | `0.3364 s` | `0.3391 s` | `4.648` | `24.5450 s` |

FA4 steady-state speed: `0.953x` vs SDPA, about `4.7%` slower. Relative loss
diff: `0.00150`. FA4 first step includes large FA4/CUTLASS compile/codegen
overhead.

### 30B / 4k LoRA-SFT: Prompt-Heavy Run

Artifact:
`third_party/LlamaFactory-fa4/fa4_reports/profile_lora_30b_s4096_20260604/profile_workload_report.json`

Dataset has 4096 input tokens after LF preprocessing but only 18 non-ignore
label tokens, so it is valid for attention shape but under-represents long SFT
backward work.

| Backend | Median step | Mean step | P90 step | Peak alloc | Loss | First step |
|---|---:|---:|---:|---:|---:|---:|
| SDPA | `0.4118 s` | `0.4086 s` | `0.4181 s` | `91.214 GiB` | `4.146` | `1.0683 s` |
| FA4 | `0.4307 s` | `0.4308 s` | `0.4454 s` | `91.230 GiB` | `4.160` | `18.1746 s` |

FA4 steady-state speed: `0.956x`, about `4.4%` slower. Relative loss diff:
`0.00338`. Memory was effectively unchanged; FA4 was higher by about
`0.017 GiB`.

### 30B / 4k LoRA-SFT: Corrected Long-Label Run

Summary artifact:
`third_party/LlamaFactory-fa4/fa4_reports/heavy_lora_30b_s4096_summary_20260604.json`

Dataset correction: raw template tokens `4091`, source tokens `2041`,
target/supervised non-ignore label tokens `2050`.

| Backend | Median step | Mean step | P90 step | Peak alloc | Peak reserved | Loss | First step |
|---|---:|---:|---:|---:|---:|---:|---:|
| SDPA | `0.4726 s` | `0.4780 s` | `0.4866 s` | `95.344 GiB` | `97.166 GiB` | `0.04432` | `1.1902 s` |

FA4 did not produce a valid post-warmup steady-state comparison. It reached the
FA4 backend and training loop, then emitted one timing row only: step 1 took
`453.224 s` with peak allocated `90.784 GiB`. A retry also reached the FA4
backend/training loop but was stopped before step 1 completed. No OOM marker
was present; behavior was CPU-bound FA4/CUTLASS DSL compile/codegen overhead.

## Progress

### Scope / Environment

- Lab checkout only:
  `/home/shutianluo/kevin/AsymGEMM-SFT/third_party/LlamaFactory-fa4`.
- Reference production workload:
  `third_party/AsymGEMM/scripts/lf/profile_lora_lf.sh`.
- Target heavy workload: `Qwen/Qwen3-30B-A3B`, `cutoff_len=4096`, LoRA rank
  `64`, alpha `16`, dropout `0.0`, batch `1`, grad accumulation `1`, `15`
  total steps with first `5` discarded.
- Test GPU: `CUDA_VISIBLE_DEVICES=2`, reported as `NVIDIA GB200`.
- Lab env: `third_party/LlamaFactory-fa4/.conda-lf-fa4`.
- Validated versions: `torch==2.12.0+cu130`, CUDA runtime `13.0`,
  `transformers==5.6.0`, `flash-attn-4==4.0.0b16`.

### Integration

- FA4 backend plumbed in lab LlamaFactory:
  `src/llamafactory/extras/constants.py`,
  `src/llamafactory/model/model_utils/attention.py`,
  `src/llamafactory/data/collator.py`.
- Transformers FA4 compatibility patch applied inside lab env:
  `transformers/integrations/flash_attention.py`.
- Patch detail: `s_aux=s_aux.to(query.dtype) if s_aux is not None else None`.
- Checker: `fa4_probes/patch_transformers_fa4.py --check`.
- Main harnesses:
  `fa4_probes/validate_lf_fa4.py`,
  `fa4_probes/benchmark_fa4_vs_sdpa.py`,
  `fa4_probes/benchmark_attention_sweep.py`,
  `fa4_probes/benchmark_lf_steady_state.py`,
  `fa4_probes/benchmark_lf_profile_workload.py`.

### Interpretation

- Functional status is good: imports, direct attention probes, Transformers
  FA4 probe, and LF smoke runs pass.
- Numerics are acceptable in tested attention/MoE/LF smoke cases.
- Performance is not better than torch SDPA for this LF LoRA-SFT setup.
- Larger batch/sequence attention sweeps narrow the gap in a few cases, but do
  not show a robust win; FA4 peak allocation is slightly higher in the sweep.
- Completed 30B/4k runs do not show memory reduction. Attention workspace is
  not the dominant peak component for these measurements.
- Corrected 30B/4k long-label FA4 is blocked by first-shape FA4/CUTLASS
  compile/codegen overhead before steady-state can be measured.

### Remaining Work

- If FA4 remains required, investigate persistent FA4/CUTLASS compile cache or
  precompile for the exact 30B/4k shapes.
- Run Nsight on a post-compile steady-state FA4 window only after first-step
  compile cost can be amortized or bypassed.
- Recompare against torch SDPA on the corrected long-label dataset once FA4 can
  complete at least `5+10` steps.
- Keep FA4 experimentation under `third_party/LlamaFactory-fa4` until stable
  enough to port.

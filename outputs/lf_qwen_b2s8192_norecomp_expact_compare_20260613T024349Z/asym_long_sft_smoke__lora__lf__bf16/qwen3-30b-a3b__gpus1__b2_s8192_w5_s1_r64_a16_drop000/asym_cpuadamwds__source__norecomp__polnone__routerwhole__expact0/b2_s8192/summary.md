# LF LoRA-SFT Source Profile

Workload: `qwen3-30b-a3b`  
Backend: `asym_cpuadamwds`  
Router mode: `whole`  
Expert activation offload: `false`  
Precision: `bf16`  
Seq len: `8192`
Steps: `5` warmup + `1` measured

| Stage | host ms | avg allocated start MiB | avg allocated end MiB | avg peak allocated MiB | avg peak reserved MiB | samples |
|---|---:|---:|---:|---:|---:|---:|
| lf.training_step.total | 0.000 | - | - | - | - | - |
| step.forward + step.backward | 0.000 | - | - | - | - | - |
| step.forward | 0.000 | - | - | - | - | - |
| step.backward | 0.000 | - | - | - | - | - |

Peak allocated HBM: `184756.89 MiB`
Peak reserved HBM: `185548.00 MiB`
Reserved but unallocated: `791.11 MiB`
Timing source: `-`
Trainer log: `/workspace/AsymGEMM-SFT/third_party/AsymGEMM/outputs/lf_qwen_b2s8192_norecomp_expact_compare_20260613T024349Z/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b2_s8192_w5_s1_r64_a16_drop000/asym_cpuadamwds__source__norecomp__polnone__routerwhole__expact0/b2_s8192/lf_run/trainer_log.jsonl`

## Process Memory

| Metric | Value |
|---|---:|
| available | True |
| RSS bytes | 173596803072 |
| RSS MiB | 165554.81 |
| RSS peak bytes | 214348595200 |
| RSS peak MiB | 204418.75 |
| virtual memory bytes | 726246096896 |
| source | /proc/self/status |

| Sample | RSS bytes | RSS peak bytes | virtual memory bytes | RSS delta bytes |
|---|---:|---:|---:|---:|
| report | 173596803072 | 214348595200 | 726246096896 | - |

## KT / LoRA Counters

| Metric | Value |
|---|---:|
| KT backend | - |
| KT wrappers | missing |
| KT forward calls | missing |
| KT backward calls | missing |
| trainable params | 3375366144 |
| PEFT LoRA params | 3375366144 |
| LF fused expert LoRA params | 0 |
| LF fused expert LoRA tensors | 0 |
| KT expert LoRA params | 0 |
| KT PEFT-view expert LoRA params | 0 |
| KT fused expert LoRA params | 0 |
| KT fused expert LoRA tensors | 0 |
| KT fused expert LoRA sidecar expected tensors | 0 |
| KT fused expert LoRA sidecar expected params | 0 |
| trainable surface | attention+expert LoRA |
| non-expert PEFT LoRA params | 53477376 |
| PEFT expert LoRA params | 3321888768 |
| expert LoRA params | 3321888768 |
| backend comparison note | requires a baseline that also trains expert LoRA |

## Optimizer Memory Preflight

| Metric | Value |
|---|---:|
| available | True |
| source | lora_counters_pre_optimizer_step |
| reason | - |
| assumed param dtype | bf16 |
| logical qlen | 16384 |
| LoRA rank | 64 |
| KT ARM top-k | None |
| KT ARM token chunk size | None |
| KT ARM effective route qlen | None |
| KT ARM token chunks | None |
| KT ARM route-rank work | None |
| KT ARM route-rank cap | None |
| trainable params | 3375366144 |
| KT fused expert LoRA params | 0 |
| KT expert LoRA params | 0 |
| non-expert PEFT LoRA params | 53477376 |
| BF16 param bytes | 6750732288 |
| BF16 grad bytes | 6750732288 |
| AdamW BF16 moments bytes | 13501464576 |
| AdamW FP32 moments bytes | 27002929152 |
| FP32 master bytes | 13501464576 |
| total BF16 params/grads + BF16 moments bytes | 27002929152 |
| total BF16 params/grads + FP32 moments bytes | 40504393728 |
| total with FP32 moments + master bytes | 54005858304 |
| large surface warning | True |

## KT Fused LoRA Update Health

| Metric | Value |
|---|---:|
| available | - |
| passed | False |
| reason | KT fused LoRA update health missing after-step snapshot fields: after_sampled_tensors, after_total_fused_tensors, missing_after_tensors, unexpected_after_tensors |
| total fused tensors | - |
| sampled tensors | - |
| after total fused tensors | - |
| after sampled tensors | - |
| exhaustive | False |
| exhaustive elements | False |
| requested max tensors | - |
| requested max elements | - |
| compared tensors | - |
| missing after tensors | - |
| unexpected after tensors | - |
| nonzero-gradient sampled tensors | - |
| nonzero-gradient tensors changed | 0 |
| nonzero-gradient tensors unchanged | 0 |
| changed sampled tensors | - |

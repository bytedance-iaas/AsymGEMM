# LF LoRA-SFT Source Profile

Workload: `qwen3-30b-a3b`  
Backend: `asym_cpuadamwds`  
Router mode: `whole`  
Expert activation offload: `true`  
Precision: `bf16`  
Seq len: `4096`
Steps: `5` warmup + `1` measured

| Stage | host ms | avg allocated start MiB | avg allocated end MiB | avg peak allocated MiB | avg peak reserved MiB | samples |
|---|---:|---:|---:|---:|---:|---:|
| trainer.e2e.measured_step | 16020.641 | - | - | - | - | - |
| trainer.e2e.total_step | 55900.132 | - | - | - | - | - |
| lf.training_step.total | 14660.179 | 6502.10 | 12940.10 | 14458.15 | 19388.00 | 1 |
| step.forward + step.backward | 14619.005 | - | - | - | - | - |
| step.forward | 600.945 | 6502.10 | 9710.19 | 13287.19 | 19388.00 | 1 |
| step.backward | 14018.059 | 9710.19 | 12974.13 | 14458.15 | 19388.00 | 1 |
| lf.grad_clip | 5.600 | 12940.10 | 12940.10 | 12941.08 | 19388.00 | 1 |
| lf.optimizer.step | 994.396 | 12940.10 | 12940.10 | 13004.10 | 19388.00 | 7 |
| lf.scheduler.step | 0.279 | 12940.10 | 12940.10 | 12940.10 | 19388.00 | 1 |

Peak allocated HBM: `14458.15 MiB`
Peak reserved HBM: `19388.00 MiB`
Reserved but unallocated: `4929.85 MiB`
Timing source: `heartbeat_dataloader_interval`
Trainer log: `/workspace/AsymGEMM-SFT/third_party/AsymGEMM/outputs/lf_expact_compare_20260613T003448Z/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b1_s4096_w5_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp__polnone__routerwhole__expact1/b1_s4096/lf_run/trainer_log.jsonl`

## Process Memory

| Metric | Value |
|---|---:|
| available | True |
| RSS bytes | 244115636224 |
| RSS MiB | 232806.81 |
| RSS peak bytes | 244115636224 |
| RSS peak MiB | 232806.81 |
| virtual memory bytes | 743132626944 |
| source | /proc/self/status |

| Sample | RSS bytes | RSS peak bytes | virtual memory bytes | RSS delta bytes |
|---|---:|---:|---:|---:|
| report | 244115636224 | 244115636224 | 743132626944 | - |
| lf.data.next | 244096761856.0 | 244096761856 | 743065452544.0 | 0.0 |
| lf.grad_clip | 244115636224.0 | 244115636224 | 743065452544.0 | 0.0 |
| lf.inputs.prepare | 244096761856.0 | 244096761856 | 743065452544.0 | 0.0 |
| lf.log_save_eval | 244115636224.0 | 244115636224 | 743066697728.0 | 0.0 |
| lf.optimizer.step | 244056316781.7143 | 244115636224 | 743065480630.8572 | 0.0 |
| lf.scheduler.step | 244115636224.0 | 244115636224 | 743066697728.0 | 0.0 |
| lf.step.total | 244115636224.0 | 244115636224 | 743065452544.0 | 18874368.0 |
| step.backward | 244115636224.0 | 244115636224 | 743065452544.0 | 0.0 |
| step.forward | 244115636224.0 | 244115636224 | 743065452544.0 | 18874368.0 |
| optimizer_step_start | 244115636224 | 244115636224 | 743065649152 | - |
| optimizer_step_before | 244115636224 | 244115636224 | 743066697728 | - |
| optimizer_step_after | 244115636224 | 244115636224 | 743066697728 | 0 |

## Gradient Clipping

| Metric | Value |
|---|---:|
| available | True |
| path | default |
| operation | norm |
| method | CustomSeq2SeqTrainer._get_grad_norm |
| max norm | - |
| total norm | - |
| result norm | 0.263671875 |
| clip coefficient | - |
| clipped | - |
| nonfinite | - |
| elapsed ms | 33.286 |
| unique grad tensors | - |
| duplicate grad tensors | - |
| CPU grad tensors | - |
| CPU grad elements | - |
| CUDA grad tensors | - |
| CUDA grad elements | - |
| chunk elements | - |

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
| logical qlen | 4096 |
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
| available | False |
| passed | False |
| reason | KT fused LoRA update health unavailable |
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

## Measured Losses

| Step | Raw step | Loss |
|---:|---:|---:|
| 1 | 6 | 1.4556084871292114 |

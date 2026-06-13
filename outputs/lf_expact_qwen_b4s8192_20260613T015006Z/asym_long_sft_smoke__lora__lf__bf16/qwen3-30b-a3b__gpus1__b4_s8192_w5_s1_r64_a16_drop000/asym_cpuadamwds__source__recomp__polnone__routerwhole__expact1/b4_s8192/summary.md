# LF LoRA-SFT Source Profile

Workload: `qwen3-30b-a3b`  
Backend: `asym_cpuadamwds`  
Router mode: `whole`  
Expert activation offload: `true`  
Precision: `bf16`  
Seq len: `8192`
Steps: `5` warmup + `1` measured

| Stage | host ms | avg allocated start MiB | avg allocated end MiB | avg peak allocated MiB | avg peak reserved MiB | samples |
|---|---:|---:|---:|---:|---:|---:|
| trainer.e2e.measured_step | 107290.856 | - | - | - | - | - |
| trainer.e2e.total_step | 132472.753 | - | - | - | - | - |
| lf.training_step.total | 105935.268 | 6502.75 | 12940.75 | 70138.95 | 81268.00 | 1 |
| step.forward + step.backward | 105891.594 | - | - | - | - | - |
| step.forward | 1961.852 | 6502.75 | 32155.20 | 60771.45 | 81268.00 | 1 |
| step.backward | 103929.742 | 32155.20 | 13200.82 | 70138.95 | 81268.00 | 1 |
| lf.grad_clip | 5.866 | 12940.75 | 12940.75 | 12941.74 | 81268.00 | 1 |
| lf.optimizer.step | 1008.193 | 12940.75 | 12940.75 | 13004.75 | 81268.00 | 7 |
| lf.scheduler.step | 0.260 | 12940.75 | 12940.75 | 12940.75 | 81268.00 | 1 |

Peak allocated HBM: `70138.95 MiB`
Peak reserved HBM: `81268.00 MiB`
Reserved but unallocated: `11129.05 MiB`
Timing source: `heartbeat_dataloader_interval`
Trainer log: `/workspace/AsymGEMM-SFT/third_party/AsymGEMM/outputs/lf_expact_qwen_b4s8192_20260613T015006Z/asym_long_sft_smoke__lora__lf__bf16/qwen3-30b-a3b__gpus1__b4_s8192_w5_s1_r64_a16_drop000/asym_cpuadamwds__source__recomp__polnone__routerwhole__expact1/b4_s8192/lf_run/trainer_log.jsonl`

## Process Memory

| Metric | Value |
|---|---:|
| available | True |
| RSS bytes | 219013054464 |
| RSS MiB | 208867.12 |
| RSS peak bytes | 220406284288 |
| RSS peak MiB | 210195.81 |
| virtual memory bytes | 749051510784 |
| source | /proc/self/status |

| Sample | RSS bytes | RSS peak bytes | virtual memory bytes | RSS delta bytes |
|---|---:|---:|---:|---:|
| report | 219013054464 | 220406284288 | 749051510784 | - |
| lf.data.next | 219101855744.0 | 220406284288 | 748984532992.0 | 0.0 |
| lf.grad_clip | 220239167488.0 | 220406284288 | 748985581568.0 | 0.0 |
| lf.inputs.prepare | 219101855744.0 | 220406284288 | 748984532992.0 | 0.0 |
| lf.log_save_eval | 219013054464.0 | 220406284288 | 748985581568.0 | 0.0 |
| lf.optimizer.step | 216114462720.0 | 220406284288 | 748984551716.5714 | -1234651428.5714285 |
| lf.scheduler.step | 219013054464.0 | 220406284288 | 748985581568.0 | 0.0 |
| lf.step.total | 220239167488.0 | 220406284288 | 748984532992.0 | 1137311744.0 |
| step.backward | 220239167488.0 | 220406284288 | 748984532992.0 | 1126629376.0 |
| step.forward | 219112538112.0 | 220406284288 | 748984532992.0 | 10682368.0 |
| optimizer_step_start | 220239167488 | 220406284288 | 748985581568 | - |
| optimizer_step_before | 220239167488 | 220406284288 | 748985581568 | - |
| optimizer_step_after | 219013054464 | 220406284288 | 748985581568 | -1226113024 |

## Gradient Clipping

| Metric | Value |
|---|---:|
| available | True |
| path | default |
| operation | norm |
| method | CustomSeq2SeqTrainer._get_grad_norm |
| max norm | - |
| total norm | - |
| result norm | 0.177734375 |
| clip coefficient | - |
| clipped | - |
| nonfinite | - |
| elapsed ms | 35.011 |
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
| logical qlen | 32768 |
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
| 1 | 6 | 1.5764399766921997 |

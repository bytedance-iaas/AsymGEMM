# AsymGEMM LoRA-SFT + Profiling Harness

This harness runs **LoRA supervised fine-tuning (SFT)** of large MoE and dense models through
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), using AsymGEMM's CPU-resident weight
and activation offload. Expert weights, gradients, and optimizer state stay in CPU DRAM and are
fetched on demand, so long-context LoRA training of 30B–235B models fits on a
single GPU or 2 GPUs via expert parallelism or data parallelism. 
The driver sweeps a `RUNS` list of model / backend / recompute / sequence configs and
profiles each run's memory, timing, throughput, and CPU↔GPU interconnect.

## Models

Select a model by its shorthand in a `RUNS` row (e.g. `q3-30b-a3b|1`), or pass a full HF path.

| Shorthand | Model | Type |
| --- | --- | --- |
| `q3-30b-a3b` | Qwen/Qwen3-30B-A3B | MoE |
| `q3-235b-a22b` | Qwen/Qwen3-235B-A22B | MoE |
| `q3.5-35b-a3b` | Qwen/Qwen3.5-35B-A3B | MoE |
| `q3.5-122b-a10b` | Qwen/Qwen3.5-122B-A10B | MoE |
| `llama4-scout` | meta-llama/Llama-4-Scout-17B-16E | MoE |
| `q3-32b` | Qwen/Qwen3-32B | dense |
| `q3.5-27b` | Qwen/Qwen3.5-27B | dense |
| `q2.5-32b` | Qwen/Qwen2.5-32B-Instruct | dense |
| `q2.5-72b` | Qwen/Qwen2.5-72B-Instruct | dense |
| `llama3.3-70b` | meta-llama/Llama-3.3-70B-Instruct | dense |

## Install

```bash
cd third-party
git clone https://github.com/deepspeedai/DeepSpeed.git deepspeed
git clone https://github.com/linkedin/Liger-Kernel.git Liger-Kernel
cd ..
bash scripts/lf/bootstrap_lf_venv.sh
```

## Run

```bash
bash scripts/lf/profile_lora_lf_test_both.sh
```

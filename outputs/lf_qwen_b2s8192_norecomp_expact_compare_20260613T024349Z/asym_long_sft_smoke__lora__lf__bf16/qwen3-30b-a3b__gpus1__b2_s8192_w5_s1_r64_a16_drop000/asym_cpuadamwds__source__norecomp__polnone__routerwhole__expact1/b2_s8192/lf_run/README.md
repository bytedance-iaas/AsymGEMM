---
library_name: transformers
license: other
base_model: Qwen/Qwen3-30B-A3B
tags:
- llama-factory
- lora
- asym-gemm
- generated_from_trainer
model-index:
- name: lf_run
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lf_run

This model is a fine-tuned version of [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) on the asym_long_sft_smoke__qwen3-30b-a3b__s8192 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: constant
- training_steps: 6

### Training results



### Framework versions

- Transformers 5.6.0
- Pytorch 2.12.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2

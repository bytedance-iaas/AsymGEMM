# Raw manual commands for debugging DeepSpeed/SuperOffload GPU visibility.
# These are intentionally not wrapped in functions. Copy/run the block you want.

export SFT_ROOT=/home/kevinni/AsymGEMM-SFT
export ASYMGEMM_DIR="${SFT_ROOT}/third_party/AsymGEMM"
export LF_DIR="${SFT_ROOT}/third_party/LlamaFactory"
export DEEPSPEED_DIR="${SFT_ROOT}/third_party/deepspeed"

cd "${ASYMGEMM_DIR}"
source "${ASYMGEMM_DIR}/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${LF_DIR}/src:${PYTHONPATH:-}"

mkdir -p "${ASYMGEMM_DIR}/profiling/raw_deepspeed_zero3_offload/lf_run"

# DeepSpeed ZeRO-3 offload, one physical GPU: CUDA_VISIBLE_DEVICES=0 -> logical cuda:0.
CUDA_VISIBLE_DEVICES=0 \
deepspeed \
  --num_gpus 1 \
  --master_addr 127.0.0.1 \
  --master_port 29591 \
  "${LF_DIR}/src/train.py" \
  --model_name_or_path Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --trust_remote_code true \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_rank 64 \
  --lora_alpha 16 \
  --lora_dropout 0.10 \
  --lora_target all \
  --dataset asym_long_sft_smoke__qwen3-235b-a22b-instruct-2507__s8192 \
  --dataset_dir "${LF_DIR}/data" \
  --template qwen3_nothink \
  --cutoff_len 8192 \
  --max_samples 128 \
  --overwrite_cache true \
  --preprocessing_num_workers 4 \
  --dataloader_num_workers 2 \
  --output_dir "${ASYMGEMM_DIR}/profiling/raw_deepspeed_zero3_offload/lf_run" \
  --logging_steps 1 \
  --logging_first_step true \
  --save_strategy no \
  --eval_strategy no \
  --report_to none \
  --plot_loss false \
  --overwrite_output_dir true \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --max_steps 15 \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --seed 42 \
  --bf16 true \
  --pure_bf16 false \
  --gradient_checkpointing true \
  --disable_gradient_checkpointing false \
  --deepspeed "${LF_DIR}/examples/deepspeed/ds_z3_offload_config.json"

# mkdir -p "${ASYMGEMM_DIR}/profiling/raw_superoffload/lf_run"

# # SuperOffload. Keep local DeepSpeed first on PYTHONPATH so the SuperOffload
# # runtime is used instead of the venv/site-packages DeepSpeed.
# PYTHONPATH="${DEEPSPEED_DIR}:${LF_DIR}/src:${PYTHONPATH:-}" \
# deepspeed \
#   --num_gpus 1 \
#   --master_addr 127.0.0.1 \
#   --master_port 29592 \
#   "${LF_DIR}/src/train.py" \
#   --model_name_or_path Qwen/Qwen3-235B-A22B-Instruct-2507 \
#   --trust_remote_code true \
#   --stage sft \
#   --do_train true \
#   --finetuning_type lora \
#   --lora_rank 64 \
#   --lora_alpha 16 \
#   --lora_dropout 0.10 \
#   --lora_target all \
#   --dataset asym_long_sft_smoke__qwen3-235b-a22b-instruct-2507__s8192 \
#   --dataset_dir "${LF_DIR}/data" \
#   --template qwen3_nothink \
#   --cutoff_len 8192 \
#   --max_samples 128 \
#   --overwrite_cache true \
#   --preprocessing_num_workers 4 \
#   --dataloader_num_workers 2 \
#   --output_dir "${ASYMGEMM_DIR}/profiling/raw_superoffload/lf_run" \
#   --logging_steps 1 \
#   --logging_first_step true \
#   --save_strategy no \
#   --eval_strategy no \
#   --report_to none \
#   --plot_loss false \
#   --overwrite_output_dir true \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 1 \
#   --learning_rate 1e-4 \
#   --max_steps 15 \
#   --lr_scheduler_type constant \
#   --warmup_steps 0 \
#   --seed 42 \
#   --bf16 true \
#   --pure_bf16 false \
#   --gradient_checkpointing true \
#   --disable_gradient_checkpointing false \
#   --deepspeed "${LF_DIR}/examples/deepspeed/ds_z3_superoffload_config.json"

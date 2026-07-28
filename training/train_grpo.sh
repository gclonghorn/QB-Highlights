#!/bin/bash
# Stage 2 — Trace-Aware GRPO (LoRA, paper config), portable multi-GPU launcher via torchrun.
#
# Usage:
#   MODEL_PATH=<Stage1 SFT ckpt> NGPUS=8 bash train_grpo.sh
#   # multi-node:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<host> MASTER_PORT=29500 NGPUS=8 bash train_grpo.sh
set -e

# ---- paths (edit) ----
MS_SWIFT=./ms-swift-3.11.0
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Stage-1 SFT checkpoint dir}"
DATASET_PATH="${DATASET_PATH:-../dataset/trainset/grpo_49k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qcr_grpo}"
EXTERNAL_PLUGINS="./reward_functions.py"

# ---- launch (edit) ----
NGPUS=${NGPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}

# ---- env ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NPROC_PER_NODE=$NGPUS
export FPS=1.0
export FPS_MAX_FRAMES=180
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
# optional: export WANDB_PROJECT=... WANDB_RUN_NAME=... WANDB_API_KEY=...

echo "GRPO: nnodes=$NNODES node_rank=$NODE_RANK ngpus=$NGPUS num_generations=$NUM_GENERATIONS"
echo "  model (Stage-1 SFT ckpt) : $MODEL_PATH"
echo "  dataset                  : $DATASET_PATH"
echo "  output                   : $OUTPUT_DIR"

# Paper effective batch size = 1 * grad_accum(16) * world_size(48) / num_generations(8) = 96.
# Paper reward:  R = λ_qual · R_qual + λ_comp · R_comp + λ_ground · R_ground,
#                R_qual = α · R_global + (1-α) · R_local,
#                (λ_qual, λ_comp, λ_ground) = (0.6, 0.2, 0.2), α = 0.5, γ = 1.0.
# --reward_weights below is expanded to the four registered rewards (order = --reward_funcs):
#   r-global = λ_qual · α       = 0.30 (R_global, union-IoU)
#   r-local  = λ_qual · (1-α)   = 0.30 (R_local, NGIoU + boundary)
#   r-ground = λ_ground         = 0.20 (R_ground)
#   r-comp   = λ_comp           = 0.20 (R_comp)
torchrun \
  --nnodes=$NNODES --node_rank=$NODE_RANK \
  --nproc_per_node=$NGPUS \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $MS_SWIFT/swift/cli/rlhf.py \
    --rlhf_type grpo \
    --model $MODEL_PATH \
    --tuner_type lora \
    --model_type qwen3_vl \
    --offload_optimizer true --offload_model true \
    --use_vllm true --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 --sleep_level 1 \
    --dataset $DATASET_PATH \
    --load_from_cache_file true \
    --output_dir $OUTPUT_DIR \
    --logging_steps 10 --save_steps 100 --save_total_limit 10000 \
    --vllm_max_model_len 32768 --max_length 32768 --max_completion_length 4096 \
    --padding_free true \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 --attn_impl flash_attn \
    --learning_rate 5e-6 \
    --num_train_epochs 1 \
    --target_modules all-linear \
    --freeze_vit true --freeze_aligner false \
    --lora_rank 64 --lora_alpha 128 \
    --num_generations $NUM_GENERATIONS \
    --beta 0.1 --temperature 1.0 \
    --dataloader_num_workers 2 --dataset_num_proc 2 \
    --dataset_shuffle false --train_dataloader_shuffle false \
    --external_plugins $EXTERNAL_PLUGINS \
    --reward_funcs r-global r-local r-ground r-comp \
    --reward_weights 0.30 0.30 0.20 0.20 \
    --deepspeed zero3 \
    --report_to tensorboard

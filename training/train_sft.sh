#!/bin/bash
# Stage 1 — SFT.
#
# Usage:
#   NGPUS=8 bash train_sft.sh
#   # multi-node:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<host> MASTER_PORT=29500 NGPUS=8 bash train_sft.sh
set -e

# ---- paths (edit) ----
MS_SWIFT=./ms-swift-3.11.0                        # clone: git clone --branch v3.11.0 https://github.com/modelscope/ms-swift
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"   # HF id or local dir
DATASET_PATH="${DATASET_PATH:-../dataset/trainset/sft_cot_17k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qcr_sft}"

# ---- launch (edit) ----
NGPUS=${NGPUS:-8}                 # per-node GPU count
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# ---- env ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NPROC_PER_NODE=$NGPUS
export FPS=1.0

echo "SFT: nnodes=$NNODES node_rank=$NODE_RANK ngpus=$NGPUS"
echo "  model   : $MODEL_PATH"
echo "  dataset : $DATASET_PATH"
echo "  output  : $OUTPUT_DIR"

# Paper effective batch size = 1 * grad_accum(8) * world_size(16 with sp=2 -> 128).
# Adjust grad_accum / sequence_parallel_size for your NGPUS.
torchrun \
  --nnodes=$NNODES --node_rank=$NODE_RANK \
  --nproc_per_node=$NGPUS \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  $MS_SWIFT/swift/cli/sft.py \
    --model $MODEL_PATH \
    --model_type qwen3_vl \
    --train_type full \
    --sequence_parallel_size 2 \
    --dataset $DATASET_PATH \
    --load_from_cache_file false \
    --output_dir $OUTPUT_DIR \
    --logging_steps 10 \
    --save_steps 200 \
    --num_train_epochs 3 \
    --max_length 49152 \
    --deepspeed zero3 \
    --gradient_checkpointing true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --aligner_lr 5e-5 \
    --warmup_ratio 0.05 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --padding_free true \
    --freeze_vit true --freeze_llm false --freeze_aligner false \
    --dataloader_num_workers 4 --dataset_num_proc 24 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --loss_scale timestamp

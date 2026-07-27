#!/bin/bash
# QCR inference (paper setting)
export CUDA_VISIBLE_DEVICES=0,1

python infer.py \
    --experiment-name 'qcr_preds' \
    --tasks-jsonl-path '../dataset/testset/test_0.9k.jsonl' \
    --video-dir '<VIDEO_DIR>' \
    --output-root './results' \
    --vlm-model-path '<QCR_MERGED_CHECKPOINT>' \
    --vlm-model-name 'qwen3-vl-8b' \
    --num-gpus 2 \
    --fps 1.0

echo "Script Executed."

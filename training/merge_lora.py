import os

import torch
from swift import Swift
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def merge_lora_into_qwen3vl(
    base_model_path: str,
    lora_adapter_path: str,
    merged_model_path: str,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    safe_serialization: bool = True,
):
    """Merge a Swift-trained LoRA adapter into a Qwen3-VL base model and save as HF format."""
    print("Loading base model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_path)

    print("Injecting LoRA adapter...")
    model = Swift.from_pretrained(model, lora_adapter_path)

    print("Merging LoRA into base model...")
    merged_model = model.merge_and_unload()

    os.makedirs(merged_model_path, exist_ok=True)

    print("Saving merged model and processor...")
    merged_model.save_pretrained(merged_model_path, safe_serialization=safe_serialization)
    processor.save_pretrained(merged_model_path)

    print(f"Merged model saved to: {merged_model_path}")


if __name__ == "__main__":
    merge_lora_into_qwen3vl(
        base_model_path="<PATH_TO_STAGE1_SFT_CHECKPOINT>",
        lora_adapter_path="<PATH_TO_STAGE2_GRPO_LORA_CHECKPOINT>",
        merged_model_path="<PATH_TO_MERGED_OUTPUT_DIR>",  # e.g. checkpoints/QCR-checkpoint
    )

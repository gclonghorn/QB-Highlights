import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from swift import Swift
import os

def merge_lora_into_qwen3vl(
    base_model_path: str,
    lora_adapter_path: str,
    merged_model_path: str,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
    safe_serialization: bool = True
):
    """
    将 LoRA 适配器合并到 Qwen3-VL 基础模型中，并保存为标准 Hugging Face 格式。

    参数:
        base_model_path (str): 基础模型路径（Qwen3-VL）。
        lora_adapter_path (str): LoRA 适配器路径（由 Swift 训练得到）。
        merged_model_path (str): 合并后模型的保存路径。
        torch_dtype: 模型加载的数据类型，默认为 torch.bfloat16。
        device_map: 设备映射策略，默认 "auto"。
        attn_implementation: 注意力实现方式，需与训练时一致，默认 "flash_attention_2"。
        safe_serialization (bool): 是否使用 safetensors 格式保存，默认 True。
    """
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

    # 确保保存目录存在
    os.makedirs(merged_model_path, exist_ok=True)

    print("Saving merged model and processor...")
    merged_model.save_pretrained(merged_model_path, safe_serialization=safe_serialization)
    processor.save_pretrained(merged_model_path)

    print(f"✅ Merged model saved to: {merged_model_path}")


if __name__ == "__main__":
    merge_lora_into_qwen3vl(
        base_model_path="<PATH_TO_STAGE1_SFT_CHECKPOINT>",
        lora_adapter_path="<PATH_TO_STAGE2_GRPO_LORA_CHECKPOINT>",
        merged_model_path="<PATH_TO_MERGED_OUTPUT_DIR>"      # e.g. checkpoints/QCR-checkpoint
    )

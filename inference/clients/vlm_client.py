# vlm_client.py — Qwen3-VL inference for QCR.

import logging
import os
import torch
from typing import Any, Dict, Optional

from prompts.temporal_grounding_prompts import (
    DIRECT_HIGHLIGHT_DETECT_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class VLMClient:
    def __init__(self, model_path: str, model_name: str, device: str = "cuda:0", lora_path: Optional[str] = None):
        self.model_name = model_name
        self.device = device

        if "qwen3-vl" not in model_name.lower():
            raise ValueError(f"Unsupported local VLM model name: {model_name}")

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading Qwen3-VL from {model_path} onto {self.device}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device,
        )

        if lora_path:
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA path not found: {lora_path}")
            from peft import PeftModel
            logger.info(f"Merging LoRA adapter from {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path, is_trainable=False).merge_and_unload()

        self.model = model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _inference_with_video(self, video_path: str, user_prompt: str, fps: float,
                              use_system_prompt: bool = True) -> str:
        messages = []
        if use_system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": user_prompt},
            ],
        })

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=4096)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        logger.debug(f"Local VLM ({self.device}) output:\n{output_text}")
        return output_text

    def direct_highlight_detect(self, video_url: str, metadata: Dict[str, Any], query: str,
                                fps: float, use_system_prompt: bool = True, **kwargs) -> str:
        user_prompt = DIRECT_HIGHLIGHT_DETECT_PROMPT_TEMPLATE.format(query=query)
        return self._inference_with_video(video_url, user_prompt, fps=fps, use_system_prompt=use_system_prompt)

# -*- coding: utf-8 -*-
"""Thin VLM/LLM client wrappers for the teacher pipeline.

- GeminiClient: video-in, JSON-out (candidate selection + composition), Gemini-2.5-Pro.
- QwenClient:   text-in, JSON-out (reasoning-trace rewrite), Qwen3-Max via DashScope.

Both are intentionally minimal reference wrappers. Swap in your own SDK calls /
retry-and-rate-limit logic as needed; the pipeline only relies on the two public
methods `generate_json(...)`.
"""
import json
import os
import re
import time
from typing import Any, Dict, Optional


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    # strip ```json ... ``` fences
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        # first {...} or [...] block
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if m:
            text = m.group(1)
    return json.loads(text)


class GeminiClient:
    """Gemini-2.5-Pro client for video understanding (candidate selection / composition).

    Uses the google-generativeai SDK. The video is uploaded once and reused across
    the coarse/fine calls if desired.
    """

    def __init__(self, model: str = "gemini-2.5-pro", api_key: Optional[str] = None):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self._genai = genai
        self.model = genai.GenerativeModel(model)

    def upload_video(self, video_path: str):
        """Upload a video file and wait until it is ACTIVE."""
        f = self._genai.upload_file(path=video_path)
        while f.state.name == "PROCESSING":
            time.sleep(2)
            f = self._genai.get_file(f.name)
        if f.state.name != "ACTIVE":
            raise RuntimeError(f"video upload failed: {f.state.name}")
        return f

    def generate_json(self, video_file, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Run one prompt against an uploaded video, return parsed JSON."""
        last = None
        for _ in range(max_retries):
            try:
                resp = self.model.generate_content(
                    [video_file, prompt],
                    generation_config={"response_mime_type": "application/json"},
                )
                return _extract_json(resp.text)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(3)
        raise RuntimeError(f"Gemini generate_json failed: {last}")


class QwenClient:
    """Qwen3-Max text client (DashScope, OpenAI-compatible endpoint) for the
    reasoning-trace rewrite stage."""

    def __init__(self, model: str = "qwen3-max", api_key: Optional[str] = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ["DASHSCOPE_API_KEY"],
                             base_url=base_url)

    def generate_json(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        last = None
        for _ in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                return _extract_json(resp.choices[0].message.content)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(3)
        raise RuntimeError(f"Qwen generate_json failed: {last}")

# -*- coding: utf-8 -*-
"""Thin CLIP wrapper used by the set-level diversity filter (Stage 3)."""
from typing import List

import numpy as np
import torch


class CLIPScorer:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        import clip  # openai/CLIP
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, pil_images: List) -> np.ndarray:
        """Return L2-normalized CLIP image features (n, d)."""
        if not pil_images:
            return np.zeros((0, 512), dtype=np.float32)
        batch = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        feats = self.model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

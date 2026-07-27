# -*- coding: utf-8 -*-
"""Configuration for the QB-Highlights training-set construction pipeline.

All thresholds match paper Sec. 4 / Appendix A.3.
"""
from dataclasses import dataclass, field
from typing import Set


@dataclass
class PipelineConfig:
    # ---- teacher models ----
    coarse_model: str = "gemini-2.5-pro"     # Stage 1 candidate selection (360p)
    fine_model: str = "gemini-2.5-pro"       # Stage 2 composition (720p)
    trace_model: str = "qwen3-max"           # Stage 4 reasoning-trace rewrite

    # ---- video sampling ----
    coarse_resolution: int = 360             # coarse scan resolution
    fine_resolution: int = 720               # fine review resolution
    coarse_fps: float = 1.0
    fine_fps: float = 2.0

    # ---- budget ----
    budget_seconds: float = 10.0             # target highlight duration B
    min_segment_seconds: float = 2.0
    max_segment_seconds: float = 10.0        # candidate clips are 3–10s
    min_candidate_seconds: float = 3.0

    # ---- quality filter (Appendix A.3) ----
    mdvqa_min: float = 80.0                  # avg MD-VQA over selected segments must exceed this
    boundary_touch_max_frac: float = 0.50    # discard if target bbox touches image boundary > this frac
    area_thresh_regular: float = 0.10        # min bbox area frac for regular products
    area_thresh_small: float = 0.05          # min bbox area frac for small-object categories

    # small-object leaf/main categories (accessories, jewelry, ...)
    small_object_categories: Set[str] = field(default_factory=lambda: {
        "Jewelry", "Jewelry/Diamonds/Jadeite", "Accessories", "Jewelry & Accessories",
        "Bracelets", "Neckwear", "Rings", "Earrings", "Watches",
    })

    # ---- diversity filter (Appendix A.3) ----
    diversity_num_frames: int = 3            # frames sampled across concatenated highlight timeline
    clip_sim_max: float = 0.91               # discard if avg pairwise CLIP similarity exceeds this

    # ---- reasoning-trace subset ----
    trace_subset_size: int = 17000           # ~17K subset gets Qwen3-Max reasoning traces

    # ---- grounding ----
    grounding_dino_config: str = ""          # path to GroundingDINO config (user-provided)
    grounding_dino_ckpt: str = ""            # path to GroundingDINO weights
    clip_model: str = "ViT-B/32"


DEFAULT = PipelineConfig()

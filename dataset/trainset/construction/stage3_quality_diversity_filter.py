# -*- coding: utf-8 -*-
"""Stage 3 — Quality & diversity post-filtering (paper Appendix A.3).

Discards composed highlights that fail any of:

  QUALITY (per selected segment set)
    - clarity / temporal stability : average MD-VQA score over selected segments
      must be > 80.
    - completeness : Grounding DINO localizes the target product; discard if its
      bounding box touches the image boundary for > 50% of the highlight duration.
    - subject prominence : discard if the target bbox area is below a
      category-dependent threshold throughout the highlight
      (10% frame area for regular products, 5% for small-object categories).

  DIVERSITY (set level)
    - uniformly sample 3 frames across the concatenated highlight timeline, extract
      CLIP features, discard if the average pairwise CLIP similarity exceeds 0.91.

MD-VQA and Grounding DINO are external models. Reference hooks are provided; wire
them to your own MD-VQA service / GroundingDINO checkpoint. `--dry-run` skips the
external models and only applies the CLIP diversity filter.
"""
import argparse
import itertools

import numpy as np

from config import DEFAULT
from utils_io import (read_jsonl, write_jsonl, parse_timestamp, extract_frames,
                      concat_segment_timeline)


# ----------------------------------------------------------------------------
# External-model hooks (plug in your own implementations)
# ----------------------------------------------------------------------------
def mdvqa_score(video_path: str, start: float, end: float) -> float:
    """Return the MD-VQA perceptual-quality score in [0,100] for a segment.

    Replace with a call to your MD-VQA service / local MD-VQA repo
    (Zhang et al., CVPR'23). Must be provided to enable the quality filter.
    """
    raise NotImplementedError("wire up MD-VQA scoring here")


def grounding_dino_boxes(frame, query: str):
    """Return the best target bounding box (x1,y1,x2,y2 normalized in [0,1]) for
    `query` in `frame`, or None. Replace with GroundingDINO inference."""
    raise NotImplementedError("wire up GroundingDINO here")


# ----------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------
def quality_ok(rec, cfg) -> bool:
    clips = rec["composition"]["clips"]
    segs = [parse_timestamp(c["timestamp"]) for c in clips]
    segs = [s for s in segs if s]
    if not segs:
        return False

    # (1) clarity / temporal stability: mean MD-VQA over segments > 80
    scores, total_dur = [], 0.0
    for (s, e) in segs:
        dur = e - s
        scores.append(mdvqa_score(rec["video_path"], s, e) * dur)
        total_dur += dur
    if total_dur <= 0 or (sum(scores) / total_dur) <= cfg.mdvqa_min:
        return False

    # subject-prominence threshold depends on category
    small = (rec.get("main_category") in cfg.small_object_categories or
             rec.get("leaf_category") in cfg.small_object_categories)
    area_thresh = cfg.area_thresh_small if small else cfg.area_thresh_regular

    # (2) completeness + (3) prominence via GroundingDINO across the highlight
    boundary_touch_dur = 0.0
    prominent_anywhere = False
    for (s, e) in segs:
        for fr, t_frac in extract_frames(rec["video_path"], s, e, n=5):
            box = grounding_dino_boxes(fr, rec["query"])
            if box is None:
                continue
            x1, y1, x2, y2 = box
            # boundary touch (completeness): box edge at image border
            if x1 <= 0.01 or y1 <= 0.01 or x2 >= 0.99 or y2 >= 0.99:
                boundary_touch_dur += (e - s) / 5.0
            # prominence: bbox area fraction
            if (x2 - x1) * (y2 - y1) >= area_thresh:
                prominent_anywhere = True
    total = sum(e - s for (s, e) in segs)
    if total > 0 and boundary_touch_dur / total > cfg.boundary_touch_max_frac:
        return False   # truncated / partially-visible product
    if not prominent_anywhere:
        return False   # never prominent enough
    return True


def diversity_ok(rec, cfg, clip_model=None) -> bool:
    """CLIP pairwise-similarity diversity filter."""
    clips = rec["composition"]["clips"]
    segs = [parse_timestamp(c["timestamp"]) for c in clips]
    segs = [s for s in segs if s]
    frames = concat_segment_timeline(rec["video_path"], segs, cfg.diversity_num_frames)
    if len(frames) < 2:
        return True
    feats = clip_model.encode(frames)                      # (n, d), L2-normalized
    sims = [float(np.dot(feats[i], feats[j]))
            for i, j in itertools.combinations(range(len(feats)), 2)]
    return (sum(sims) / len(sims)) <= cfg.clip_sim_max


def run(in_path: str, out_path: str, dry_run: bool = False, cfg=DEFAULT):
    from clip_scorer import CLIPScorer          # local thin wrapper (see utils_io note)
    clip_model = CLIPScorer(cfg.clip_model)
    recs = read_jsonl(in_path)
    kept = []
    for r in recs:
        if not r.get("composition") or not r["composition"].get("clips"):
            continue
        try:
            if not dry_run and not quality_ok(r, cfg):
                r["_filter"] = "quality_reject"; continue
            if not diversity_ok(r, cfg, clip_model):
                r["_filter"] = "diversity_reject"; continue
        except NotImplementedError:
            # external model not wired: keep sample but flag
            r["_filter"] = "quality_skipped(dry)"
        r["_filter"] = r.get("_filter", "pass")
        kept.append(r)
    write_jsonl(out_path, kept)
    print(f"[stage3] kept {len(kept)}/{len(recs)} after quality+diversity filter -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="skip MD-VQA/GroundingDINO, apply CLIP diversity only")
    run(**vars(ap.parse_args()))

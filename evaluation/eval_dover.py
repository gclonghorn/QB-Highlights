# -*- coding: utf-8 -*-
"""DOVER — disentangled objective video quality (rebuttal, independent of MD-VQA).

Runs DOVER (Wu et al., ICCV'23) on each extracted highlight clip mp4 and reports the
fused overall quality score (scale [0,1]), averaged per video then over the 902 eval
videos. DOVER's training data and backbone are disjoint from both MD-VQA and our
CLIP-based pipeline, so it serves as an independent quality metric.

Reference repo: https://github.com/VQAssessment/DOVER  (local path via --dover-root)
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

import metrics_common as mc


def build_dover(dover_root: str, opt_file: str, device: str):
    sys.path.insert(0, dover_root)
    from dover.models import DOVER
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import fuse_results  # score-level fusion (0..1)

    with open(opt_file) as f:
        opt = yaml.safe_load(f)
    # resolve the (relative) test_load_path against the DOVER repo root so the
    # script works regardless of the current working directory
    ckpt = opt["test_load_path"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(dover_root, ckpt)
    model = DOVER(**opt["model"]["args"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    dopt = opt["data"]["val-l1080p"]["args"]
    samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            samplers[stype] = UnifiedFrameSampler(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
        else:
            samplers[stype] = UnifiedFrameSampler(sopt["clip_len"] // sopt["t_frag"], sopt["t_frag"],
                                                  sopt["frame_interval"], sopt["num_clips"])
    return model, dopt, samplers, spatial_temporal_view_decomposition, fuse_results


@torch.no_grad()
def dover_score(mp4, model, dopt, samplers, decomp, fuse, device) -> float:
    mean = torch.FloatTensor([123.675, 116.28, 103.53]).reshape(-1, 1, 1, 1).to(device)
    std = torch.FloatTensor([58.395, 57.12, 57.375]).reshape(-1, 1, 1, 1).to(device)
    views, _ = decomp(mp4, dopt["sample_types"], samplers)
    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (((v.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1).to(device))
    results = [r.mean().item() for r in model(views)]
    return float(fuse(results))          # fused overall score in [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--eval-jsonl", default=mc.EVAL_JSONL)
    ap.add_argument("--dover-root", required=True)
    ap.add_argument("--opt", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dopt, samplers, decomp, fuse = build_dover(args.dover_root, args.opt, device)
    ids = mc.load_eval_ids(args.eval_jsonl)

    per_video = []
    for vid in ids:
        clips = mc.clip_mp4s(args.pred_dir, vid)
        cs = []
        for mp4 in clips:
            try:
                cs.append(dover_score(mp4, model, dopt, samplers, decomp, fuse, device))
            except Exception:  # noqa: BLE001
                pass
        if cs:
            per_video.append(float(np.mean(cs)))

    print(f"videos scored : {len(per_video)}/{len(ids)}")
    print(f"DOVER         : {np.mean(per_video):.4f}" if per_video else "DOVER: n/a")


if __name__ == "__main__":
    main()

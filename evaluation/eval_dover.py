# -*- coding: utf-8 -*-
"""DOVER — third-party non-reference video quality (paper Table 3).

Runs DOVER (Wu et al., ICCV'23) on each video's predicted highlight and reports
the fused overall quality score (scale [0,1]), averaged over the ~0.9k eval
videos (videos without a prediction are skipped). DOVER's training data and
backbone are disjoint from both MD-VQA and our CLIP-based pipeline, so it
serves as an independent quality metric.

Protocol (paper): each video is scored on its *concatenated* highlight
`<pred-dir>/<id>_spot_fine.mp4`. If only per-clip mp4s are available
(`fine_clips/<id>_spot/*.mp4`), each clip is scored and the per-video mean is
used instead.

Reference repo: https://github.com/VQAssessment/DOVER  (local path via --dover-root)
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

import metrics_common as mc

# score-level fusion, verbatim from DOVER's evaluate_one_video.py
_MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
_STD = torch.FloatTensor([58.395, 57.12, 57.375])


def fuse_results(results) -> float:
    x = (results[0] - 0.1107) / 0.07355 * 0.6104 + (results[1] + 0.08285) / 0.03774 * 0.3896
    return 1.0 / (1.0 + np.exp(-x))


def build_dover(dover_root: str, opt_file: str, device: str):
    sys.path.insert(0, dover_root)
    from dover.models import DOVER
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

    with open(opt_file) as f:
        opt = yaml.safe_load(f)
    # resolve the (relative) test_load_path against the DOVER repo root so the
    # script works regardless of the current working directory
    ckpt = opt["test_load_path"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(dover_root, ckpt)
    model = DOVER(**opt["model"]["args"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    dopt = opt["data"]["val-l1080p"]["args"]
    samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            samplers[stype] = UnifiedFrameSampler(sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"])
        else:
            samplers[stype] = UnifiedFrameSampler(sopt["clip_len"] // sopt["t_frag"], sopt["t_frag"],
                                                  sopt["frame_interval"], sopt["num_clips"])
    return model, dopt, samplers, spatial_temporal_view_decomposition


@torch.no_grad()
def dover_score(mp4, model, dopt, samplers, decomp, device) -> float:
    views, _ = decomp(mp4, dopt["sample_types"], samplers)
    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (((v.permute(1, 2, 3, 0) - _MEAN) / _STD)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1).to(device))
    results = [r.mean().item() for r in model(views)]
    return float(fuse_results(results))          # fused overall score in [0,1]


def highlight_mp4s(pred_dir: str, vid: str):
    """Prefer the concatenated highlight (paper protocol); fall back to per-clip mp4s."""
    single = os.path.join(pred_dir, f"{vid}_spot_fine.mp4")
    if os.path.exists(single):
        return [single]
    return mc.clip_mp4s(pred_dir, vid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--eval-jsonl", default=mc.EVAL_JSONL)
    ap.add_argument("--dover-root", required=True)
    ap.add_argument("--opt", default=None,
                    help="DOVER config yml (default: <dover-root>/dover.yml)")
    args = ap.parse_args()

    opt_file = args.opt or os.path.join(args.dover_root, "dover.yml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dopt, samplers, decomp = build_dover(args.dover_root, opt_file, device)
    ids = mc.load_eval_ids(args.eval_jsonl)

    per_video = []
    for vid in ids:
        cs = []
        for mp4 in highlight_mp4s(args.pred_dir, vid):
            try:
                cs.append(dover_score(mp4, model, dopt, samplers, decomp, device))
            except Exception:  # noqa: BLE001
                pass
        if cs:
            per_video.append(float(np.mean(cs)))

    print(f"videos scored : {len(per_video)}/{len(ids)}")
    print(f"DOVER         : {np.mean(per_video):.4f}" if per_video else "DOVER: n/a")


if __name__ == "__main__":
    main()

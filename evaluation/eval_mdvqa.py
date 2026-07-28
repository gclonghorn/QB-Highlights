# -*- coding: utf-8 -*-
"""MD-VQA — model-based perceptual video quality (paper Table 3).

Runs MD-VQA (Zhang et al., CVPR'23) on each extracted highlight clip mp4, averages
per video, then over the ~0.9k eval videos. MD-VQA is a *third-party* model trained on
UGC videos disjoint from QB-Highlights (used as an independent quality check).

Reference repo: https://github.com/zzc-1998/MD-VQA  (local path via --mdvqa-root)
"""
import argparse
import os
import sys

import numpy as np

import metrics_common as mc


def build_scorer(mdvqa_root: str):
    sys.path.insert(0, mdvqa_root)
    from video_quality.video_quality import MDVQA
    return MDVQA(
        use_cudnn=True,
        semantic_path=os.path.join(mdvqa_root, "data/efficientnet_v2_s-dd5fe13b.pth"),
        motion_path=os.path.join(mdvqa_root, "data/r2plus1d_18-91a641e6.pth"),
        mdvqa_path=os.path.join(mdvqa_root, "data/MDVQA_LSVQ_TaoLive.pth"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--eval-jsonl", default=mc.EVAL_JSONL)
    ap.add_argument("--mdvqa-root", required=True)
    args = ap.parse_args()

    scorer = build_scorer(args.mdvqa_root)
    ids = mc.load_eval_ids(args.eval_jsonl)

    per_video = []
    for vid in ids:
        clips = mc.clip_mp4s(args.pred_dir, vid)
        clip_scores = []
        for mp4 in clips:
            try:
                res = scorer(video_path=mp4)
                clip_scores.append(float(res["score"]))
            except Exception:  # noqa: BLE001
                pass
        if clip_scores:
            per_video.append(float(np.mean(clip_scores)))

    print(f"videos scored : {len(per_video)}/{len(ids)}")
    print(f"MD-VQA        : {np.mean(per_video):.3f}" if per_video else "MD-VQA: n/a")


if __name__ == "__main__":
    main()

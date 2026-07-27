# -*- coding: utf-8 -*-
"""Score@Pred and Prec@4.0 — human-annotation metrics (paper Table 3, primary).

Score@Pred : duration-weighted average of clip-level human mean_score over the
             predicted segments.
Prec@τ     : fraction of predicted duration overlapping GT clips whose
             mean_score ≥ τ (default τ = 4.0).

GT is read from `dataset/testset/test_set_annotations.csv`; the fixed denominator
(902 ids) from `dataset/testset/test_0.9k.jsonl`. Videos with no valid
prediction contribute 0.
"""
import argparse

import metrics_common as mc


def score_at_pred(pred_segs, gt_clips) -> float:
    """Duration-weighted mean human score over predicted segments."""
    dur = mc.total_duration(pred_segs)
    if dur <= 0:
        return 0.0
    acc = 0.0
    for seg in pred_segs:
        for (cs, ce, sc) in gt_clips:
            acc += mc.overlap(seg, (cs, ce)) * sc
    return acc / dur


def prec_at_tau(pred_segs, gt_clips, tau: float) -> float:
    """Fraction of predicted duration overlapping GT clips with score ≥ tau."""
    dur = mc.total_duration(pred_segs)
    if dur <= 0:
        return 0.0
    hi = [(cs, ce) for (cs, ce, sc) in gt_clips if sc >= tau]
    hit = sum(mc.overlap(seg, g) for seg in pred_segs for g in hi)
    return hit / dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="dir of <id>_spot.json predictions")
    ap.add_argument("--eval-jsonl", default=mc.EVAL_JSONL)
    ap.add_argument("--annotations", default=mc.ANNOTATIONS_CSV)
    ap.add_argument("--tau", type=float, default=4.0)
    args = ap.parse_args()

    ids = mc.load_eval_ids(args.eval_jsonl)
    gt = mc.load_annotations(args.annotations)

    scores, precs, n_valid = [], [], 0
    for vid in ids:
        segs = mc.load_pred_segments(args.pred_dir, vid)
        clips = gt.get(vid, [])
        if not segs or not clips:
            scores.append(0.0); precs.append(0.0); continue
        n_valid += 1
        scores.append(score_at_pred(segs, clips))
        precs.append(prec_at_tau(segs, clips, args.tau))

    n = len(ids)
    print(f"videos (fixed denominator) : {n}")
    print(f"with valid prediction+GT   : {n_valid}")
    print(f"Score@Pred                 : {sum(scores)/n:.4f}")
    print(f"Prec@{args.tau:<21}: {100*sum(precs)/n:.2f}")


if __name__ == "__main__":
    main()

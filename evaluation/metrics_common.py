# -*- coding: utf-8 -*-
"""Shared helpers for QB-Highlights evaluation.

Reads the released test set from `dataset/testset/`:
  - test_0.9k.jsonl   : the 902 eval ids (+ query / categories)
  - test_set_annotations.csv  : per-video dense 2-second clip scores
                                (video_id, clip="SSS-EEE", mean_score, ...)

Predictions: one JSON per video under a --pred-dir, named `<id>_spot.json`,
containing a `clips` array whose items have a `timestamp` "MM:SS.xx-MM:SS.xx".
Extracted highlight clip mp4s (for MD-VQA / DOVER / diversity) live at
`<pred-dir>/fine_clips/<id>_spot/*.mp4`.
"""
import csv
import glob
import json
import os
import re
from typing import Dict, List, Tuple

# repo-relative default location of the released test set
_HERE = os.path.dirname(os.path.abspath(__file__))
TESTSET_DIR = os.path.abspath(os.path.join(_HERE, "..", "dataset", "testset"))
EVAL_JSONL = os.path.join(TESTSET_DIR, "test_0.9k.jsonl")
ANNOTATIONS_CSV = os.path.join(TESTSET_DIR, "test_set_annotations.csv")

_TS = re.compile(r"(\d+):(\d+(?:\.\d+)?)\s*-\s*(\d+):(\d+(?:\.\d+)?)")


# ----------------------------------------------------------------- eval ids
def load_eval_ids(eval_jsonl: str = EVAL_JSONL) -> List[str]:
    ids = []
    with open(eval_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["id"]))
    return ids


# ----------------------------------------------------------------- GT clips
def load_annotations(csv_path: str = ANNOTATIONS_CSV) -> Dict[str, List[Tuple[float, float, float]]]:
    """video_id -> list of (clip_start_sec, clip_end_sec, mean_score)."""
    gt: Dict[str, List[Tuple[float, float, float]]] = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vid = str(row["video_id"])
            s, e = row["clip"].split("-")
            gt.setdefault(vid, []).append((float(s), float(e), float(row["mean_score"])))
    for vid in gt:
        gt[vid].sort()
    return gt


# ----------------------------------------------------------------- pred clips
def parse_ts(ts: str):
    m = _TS.search(ts or "")
    if not m:
        return None
    sm, ss, em, es = m.groups()
    s = int(sm) * 60 + float(ss)
    e = int(em) * 60 + float(es)
    return (s, e) if e > s else None


def load_pred_segments(pred_dir: str, vid: str) -> List[Tuple[float, float]]:
    """Predicted highlight segments (seconds) for a video, from `<id>_spot.json`."""
    path = os.path.join(pred_dir, f"{vid}_spot.json")
    if not os.path.exists(path):
        cands = glob.glob(os.path.join(pred_dir, f"{vid}*spot*.json")) + \
                glob.glob(os.path.join(pred_dir, f"{vid}*.json"))
        if not cands:
            return []
        path = cands[0]
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    clips = d.get("clips") or (d.get("metadata", {}) or {}).get("clips") or []
    segs = [parse_ts(c.get("timestamp", "")) for c in clips]
    return [s for s in segs if s]


def clip_mp4s(pred_dir: str, vid: str) -> List[str]:
    """Extracted highlight clip mp4s for a video (for MD-VQA / DOVER / diversity)."""
    d = os.path.join(pred_dir, "fine_clips", f"{vid}_spot")
    if os.path.isdir(d):
        return sorted(glob.glob(os.path.join(d, "*.mp4")))
    # fallback: single concatenated highlight
    single = os.path.join(pred_dir, f"{vid}_spot_fine.mp4")
    return [single] if os.path.exists(single) else []


# ----------------------------------------------------------------- overlap
def total_duration(segs) -> float:
    return sum(e - s for (s, e) in segs)


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

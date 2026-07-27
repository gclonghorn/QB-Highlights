# -*- coding: utf-8 -*-
"""Stage 1 — Coarse candidate selection (360p, Gemini-2.5-Pro).

Input : a JSONL of ⟨video_id, video_path, query, product_block, ...⟩ records.
Output: same records + `candidates` = [{timestamp, description}, ...] (1–10 clips,
        each 3–10 s) produced by the CANDIDATE_SELECTION_PROMPT.

The paper scans each video at 360p to select candidate high-quality segments.
"""
import argparse
import json

from prompts import CANDIDATE_SELECTION_PROMPT
from clients import GeminiClient
from config import DEFAULT
from utils_io import read_jsonl, write_jsonl, downscale_video


def run(in_path: str, out_path: str, cfg=DEFAULT):
    client = GeminiClient(model=cfg.coarse_model)
    recs = read_jsonl(in_path)
    out = []
    for r in recs:
        # scan at 360p (paper: coarse scan resolution)
        video_360p = downscale_video(r["video_path"], cfg.coarse_resolution, cfg.coarse_fps)
        vfile = client.upload_video(video_360p)
        prompt = CANDIDATE_SELECTION_PROMPT.format(query=r["query"], video="video")
        try:
            resp = client.generate_json(vfile, prompt)
        except Exception as e:  # noqa: BLE001
            r["candidates"] = []
            r["_stage1_error"] = str(e)
            out.append(r)
            continue
        # normalize to a list of {timestamp, description}
        cands = resp.get("clips") or resp.get("candidate_clips") or resp.get("timestamps") or []
        r["candidates"] = cands
        r["candidate_analysis"] = resp.get("analysis", "")
        out.append(r)
    write_jsonl(out_path, out)
    print(f"[stage1] {len(out)} records, "
          f"{sum(1 for r in out if r.get('candidates'))} with candidates -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    run(**vars(ap.parse_args()))

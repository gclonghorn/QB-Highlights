# -*- coding: utf-8 -*-
"""Stage 2 — Fine composition (720p, Gemini-2.5-Pro).

Input : Stage-1 records with `candidates`.
Output: same records + `composition` = {analysis, clips:[{clip_id, timestamp,
        description, core_product}, ...]} — a ~10s budgeted highlight set reviewed
        at 720p via the COMPOSITION_PROMPT.

The paper reviews the candidates at 720p to compose a highlight set satisfying the
budgeted composition criteria.
"""
import argparse
import json

from prompts import COMPOSITION_PROMPT
from clients import GeminiClient
from config import DEFAULT
from utils_io import read_jsonl, write_jsonl, downscale_video


def run(in_path: str, out_path: str, cfg=DEFAULT):
    client = GeminiClient(model=cfg.fine_model)
    recs = read_jsonl(in_path)
    out = []
    for r in recs:
        if not r.get("candidates"):
            r["composition"] = None
            out.append(r)
            continue
        # review at 720p (paper: fine review resolution)
        video_720p = downscale_video(r["video_path"], cfg.fine_resolution, cfg.fine_fps)
        vfile = client.upload_video(video_720p)
        prompt = COMPOSITION_PROMPT.format(
            query=r["query"],
            candidate_segments=json.dumps(r["candidates"], ensure_ascii=False),
        )
        try:
            resp = client.generate_json(vfile, prompt)
        except Exception as e:  # noqa: BLE001
            r["composition"] = None
            r["_stage2_error"] = str(e)
            out.append(r)
            continue
        clips = resp.get("clips") or resp.get("selected_clips") or []
        r["composition"] = {"analysis": resp.get("analysis", ""), "clips": clips}
        out.append(r)
    write_jsonl(out_path, out)
    print(f"[stage2] {len(out)} records, "
          f"{sum(1 for r in out if r.get('composition'))} composed -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    run(**vars(ap.parse_args()))

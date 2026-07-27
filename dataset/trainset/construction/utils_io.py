# -*- coding: utf-8 -*-
"""Shared I/O and video utilities for the construction pipeline."""
import json
import os
import re
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- jsonl io
def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- timestamps
_TS = re.compile(r"(\d+):(\d+(?:\.\d+)?)\s*-\s*(\d+):(\d+(?:\.\d+)?)")


def parse_timestamp(ts: str) -> Optional[Tuple[float, float]]:
    """'MM:SS.xx-MM:SS.xx' -> (start_sec, end_sec)."""
    m = _TS.search(ts or "")
    if not m:
        return None
    sm, ss, em, es = m.groups()
    s = int(sm) * 60 + float(ss)
    e = int(em) * 60 + float(es)
    return (s, e) if e > s else None


# ---------------------------------------------------------------- video ops
def downscale_video(video_path: str, resolution: int, fps: float) -> str:
    """Downscale to `resolution` (short side) at `fps`. Returns a cached temp path.

    Paper uses 360p for the coarse scan and 720p for the fine review.
    """
    cache = os.path.join(tempfile.gettempdir(), "qbh_scaled")
    os.makedirs(cache, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join(cache, f"{base}_{resolution}p_{fps}fps.mp4")
    if os.path.exists(out):
        return out
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"scale=-2:{resolution},fps={fps}",
         "-c:v", "libx264", "-an", out],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return out


def extract_frames(video_path: str, start: float, end: float, n: int = 5):
    """Yield (PIL.Image, time_fraction) at n uniform times within [start, end]."""
    from PIL import Image
    import av  # PyAV
    container = av.open(video_path)
    stream = container.streams.video[0]
    times = [start + (end - start) * (i + 0.5) / n for i in range(n)]
    for t in times:
        container.seek(int(t / stream.time_base), stream=stream)
        for frame in container.decode(video=0):
            img = Image.fromarray(frame.to_ndarray(format="rgb24"))
            yield img, (t - start) / max(end - start, 1e-6)
            break
    container.close()


def concat_segment_timeline(video_path: str, segments, n_frames: int):
    """Uniformly sample `n_frames` across the *concatenated* timeline of the
    selected segments (paper's diversity filter samples 3 such frames)."""
    from PIL import Image
    import av
    if not segments:
        return []
    total = sum(e - s for (s, e) in segments)
    # positions along concatenated timeline
    targets = [total * (i + 0.5) / n_frames for i in range(n_frames)]
    frames = []
    container = av.open(video_path)
    stream = container.streams.video[0]
    for tgt in targets:
        # map concatenated offset -> absolute video time
        acc, abs_t = 0.0, segments[0][0]
        for (s, e) in segments:
            if acc + (e - s) >= tgt:
                abs_t = s + (tgt - acc)
                break
            acc += (e - s)
        container.seek(int(abs_t / stream.time_base), stream=stream)
        for frame in container.decode(video=0):
            frames.append(Image.fromarray(frame.to_ndarray(format="rgb24")))
            break
    container.close()
    return frames

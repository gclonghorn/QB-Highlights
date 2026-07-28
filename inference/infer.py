#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QCR inference. For each test item it prompts the trained model with the QCR
template, parses <think>+<answer>, and writes:
    <output>/<id>_spot.json              (reasoning trace + predicted clips)
    <output>/<id>_spot_fine.mp4          (concatenated highlight)
    <output>/fine_clips/<id>_spot/clip_NN.mp4   (individual selected clips)

Videos are resolved as <video-dir>/<id>.mp4 (source videos are not shipped;
see ../DATASHEET.md).
"""
import argparse
import json
import logging
import os
import re
import traceback
import multiprocessing as mp
from multiprocessing import Process, Manager
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from utils import save_json, parse_time_str, clip_video_from_segments, clip_single_video, is_video_valid
from clients.vlm_client import VLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def core_product_name(meta: Dict[str, Any]) -> str:
    name = meta.get("query") or meta.get("livestream_name") or ""
    leaf = meta.get("product_leaf_category")
    return f"{name} ({leaf})" if leaf else name


def extract_answer_json(response: str) -> Dict[str, Any]:
    """Parse the JSON inside <answer>...</answer>, with a last-JSON-block fallback."""
    if not response:
        return {}
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        content = re.sub(r"^```json\s*|\s*```$", "", m.group(1).strip(), flags=re.MULTILINE)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("<answer> found but JSON parse failed; trying fallback.")
    for block in reversed(re.findall(r"\{.*\}", response, re.DOTALL)):
        try:
            return json.loads(re.sub(r"^```json\s*|\s*```$", "", block, flags=re.MULTILINE))
        except Exception:
            continue
    return {}


def process_one(client: VLMClient, task: Dict[str, Any], args: argparse.Namespace):
    vid = str(task.get("id"))
    filename = f"{vid}_spot"
    output_dir = os.path.join(args.output_root, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, f"{filename}.json")

    video_path = os.path.join(args.video_dir, f"{vid}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found for id {vid}: {video_path}")

    # --- QCR highlight detection ---
    response = client.direct_highlight_detect(
        video_url=video_path,
        metadata={k: task.get(k) for k in ("query", "livestream_name",
                                           "product_main_category", "product_leaf_category") if task.get(k)},
        core_product=core_product_name(task),
        fps=args.fps,
        use_system_prompt=True,
    )
    result = extract_answer_json(response)
    for clip in result.get("clips", []) or []:
        clip["global_timestamp"] = clip.get("timestamp")
    save_json({"metadata": task, "fine_grounding_response": response, "fine_grounding_result": result},
              output_json_path)

    # --- parse segments ---
    segments = []
    for clip in result.get("clips", []) or []:
        ts = clip.get("global_timestamp") or clip.get("timestamp")
        if not ts:
            continue
        try:
            s, e = [parse_time_str(t.strip()) for t in ts.split("-")]
            segments.append((s, e))
        except (ValueError, IndexError) as ex:
            logger.error(f"Bad timestamp '{ts}' for {vid}: {ex}")
    if not segments:
        logger.warning(f"No valid segments for {vid}; skipping clip extraction.")
        return

    # --- concatenated highlight + individual clips (un-cropped) ---
    fine_video_path = os.path.join(output_dir, f"{filename}_fine.mp4")
    if not is_video_valid(fine_video_path):
        clip_video_from_segments(video_path, segments, fine_video_path)

    clips_dir = os.path.join(output_dir, "fine_clips", filename)
    os.makedirs(clips_dir, exist_ok=True)
    for i, (s, e) in enumerate(segments):
        out = os.path.join(clips_dir, f"clip_{i+1:02d}.mp4")
        if not is_video_valid(out):
            clip_single_video(video_path, out, s, e, crop_bbox=None)


def gpu_worker(worker_id: int, device: str, tasks: List[Dict], result_list, args: argparse.Namespace):
    name = f"Worker-{worker_id}({device})"
    logger.info(f"[{name}] starting with {len(tasks)} tasks")
    try:
        client = VLMClient(model_path=args.vlm_model_path, model_name=args.vlm_model_name,
                           device=device, lora_path=args.vlm_lora_path)
    except Exception as e:
        logger.error(f"[{name}] VLMClient init failed: {e}", exc_info=True)
        for t in tasks:
            result_list.append(("INIT_FAIL", str(t.get("id"))))
        return
    for task in tqdm(tasks, desc=name, position=worker_id):
        try:
            process_one(client, task, args)
            result_list.append(("SUCCESS", str(task.get("id"))))
        except Exception as e:
            logger.error(f"[{name}] task {task.get('id')} failed: {e}\n{traceback.format_exc()}")
            result_list.append(("FAILURE", str(task.get("id"))))
    logger.info(f"[{name}] done")


def main(args):
    tasks = load_jsonl(args.tasks_jsonl_path)
    start = args.start or 0
    end = args.end if args.end is not None else len(tasks)
    tasks = tasks[start:end]
    if not tasks:
        logger.info("No tasks to process.")
        return

    num_gpus = min(args.num_gpus, torch.cuda.device_count()) or 1
    shards = [[] for _ in range(num_gpus)]
    for i, t in enumerate(tasks):
        shards[i % num_gpus].append(t)
    logger.info(f"Dispatching {len(tasks)} tasks across {num_gpus} GPU(s).")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    with Manager() as manager:
        result_list = manager.list()
        procs = []
        for i in range(num_gpus):
            p = Process(target=gpu_worker, args=(i, f"cuda:{i}", shards[i], result_list, args))
            p.start(); procs.append(p)
        for p in procs:
            p.join()
        ok = sum(1 for s, _ in result_list if s == "SUCCESS")
        logger.info(f"Done. total={len(tasks)} success={ok} failed={len(tasks)-ok}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QCR inference")
    ap.add_argument("--tasks-jsonl-path", required=True, help="test_0.9k.jsonl")
    ap.add_argument("--video-dir", required=True, help="dir containing <id>.mp4 source videos")
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--vlm-model-path", required=True, help="local QCR (Qwen3-VL-8B) checkpoint")
    ap.add_argument("--vlm-model-name", default="qwen3-vl-8b")
    ap.add_argument("--vlm-lora-path", default=None)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    main(ap.parse_args())

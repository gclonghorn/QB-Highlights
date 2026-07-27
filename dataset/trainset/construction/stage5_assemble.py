# -*- coding: utf-8 -*-
"""Stage 5 — Assemble ⟨video, query, highlight set⟩ tuples into ms-swift JSONL.

Emits the two released files, with video references sanitized to `videos/<id>.mp4`:

  sft_cot_17k.jsonl   {videos, messages:[user, assistant]}
      assistant = <think>[Quality-guided Global Scan] ... Ci:<time>..</time>
                  <eval>..</eval> ... [Budgeted Set Composition] ...</think>
                  <answer>{"clips":[...]}</answer>          (structured CoT)

  grpo_49k.jsonl      {videos, video_duration, solution, messages:[system, user]}
      solution  = natural inner-monologue trace with <time>..</time>
                  <clip_eval>..</clip_eval> + set-level rationale   (reward reference)

Both render from the Stage-4 structured reasoning trace
(scan_overview / segment_qualification / budgeted_composition) and the Stage-2
composition (the final selected clips).
"""
import argparse
import json

from prompts import GRPO_SYSTEM_PROMPT, TRAIN_USER_PROMPT
from config import DEFAULT
from utils_io import read_jsonl


def _video_ref(video_id: str) -> str:
    return f"videos/{video_id}.mp4"


def _answer_json(clips) -> str:
    arr = [{
        "clip_id": i + 1,
        "timestamp": c["timestamp"],
        "description": c.get("description", ""),
        "core_product": c.get("core_product", ""),
    } for i, c in enumerate(clips)]
    return json.dumps({"clips": arr}, ensure_ascii=False, indent=2)


def _sft_think(trace, clips) -> str:
    """Structured CoT: scan + per-candidate <eval> + composition."""
    scan = (trace or {}).get("scan_overview", "").strip()
    quals = (trace or {}).get("segment_qualification", []) or []
    comp = (trace or {}).get("budgeted_composition", "").strip()
    lines = [f"[Quality-guided Global Scan] {scan} "
             f"We first perform a quality-guided global scan to identify candidate segments."]
    for i, q in enumerate(quals):
        ts = q.get("timestamp", "")
        ev = q.get("reasoning", q.get("eval", "")).strip()
        lines.append(f"C{i+1}: <time> {ts} </time>\n<eval> {ev} </eval>")
    lines.append(f"\n[Budgeted Set Composition] {comp}")
    return "<think>\n" + "\n".join(lines) + "\n</think>\n\n<answer>\n" + _answer_json(clips) + "\n</answer>"


def _grpo_solution(trace, clips) -> str:
    """Inner-monologue reference trace with <clip_eval>."""
    scan = (trace or {}).get("scan_overview", "").strip()
    quals = (trace or {}).get("segment_qualification", []) or []
    comp = (trace or {}).get("budgeted_composition", "").strip()
    body = [scan]
    for q in quals:
        ts = q.get("timestamp", "")
        ev = q.get("reasoning", q.get("eval", "")).strip()
        body.append(f"<time>{ts}</time>\n<clip_eval>\n{ev}\n</clip_eval>")
    body.append(comp)
    return "<think>\n" + "\n\n".join(body) + "\n</think>\n\n<answer>\n" + _answer_json(clips) + "\n</answer>"


def run(in_path: str, sft_out: str, grpo_out: str, cfg=DEFAULT):
    recs = read_jsonl(in_path)
    n_sft = n_grpo = 0
    with open(sft_out, "w", encoding="utf-8") as fs, open(grpo_out, "w", encoding="utf-8") as fg:
        for r in recs:
            comp = r.get("composition")
            if not comp or not comp.get("clips"):
                continue
            clips = comp["clips"]
            vids = [_video_ref(r["video_id"])]
            user = TRAIN_USER_PROMPT.format(product_block=r["product_block"])
            trace = r.get("reasoning_trace")

            # GRPO: all composed & filtered samples (solution = reference trace)
            fg.write(json.dumps({
                "videos": vids,
                "video_duration": r.get("duration"),
                "solution": _grpo_solution(trace, clips),
                "messages": [
                    {"role": "system", "content": GRPO_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            }, ensure_ascii=False) + "\n")
            n_grpo += 1

            # SFT: only the subset that received a structured reasoning trace
            if trace:
                fs.write(json.dumps({
                    "videos": vids,
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": _sft_think(trace, clips)},
                    ],
                }, ensure_ascii=False) + "\n")
                n_sft += 1
    print(f"[stage5] SFT={n_sft} -> {sft_out} | GRPO={n_grpo} -> {grpo_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--sft-out", dest="sft_out", default="sft_cot_17k.jsonl")
    ap.add_argument("--grpo-out", dest="grpo_out", default="grpo_49k.jsonl")
    run(**vars(ap.parse_args()))

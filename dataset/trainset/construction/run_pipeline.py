# -*- coding: utf-8 -*-
"""Orchestrate the full teacher-assisted coarse-to-fine construction pipeline.

    input.jsonl
      -> stage1 candidate selection (Gemini, 360p)
      -> stage2 composition          (Gemini, 720p)
      -> stage3 quality+diversity filter (MD-VQA / GroundingDINO / CLIP)
      -> stage4 reasoning-trace rewrite (Qwen3-Max, 17K subset)
      -> stage5 assemble -> sft_cot_17k.jsonl + grpo_49k.jsonl

Input JSONL record schema (one line per ⟨video, query⟩ pair):
    {
      "video_id": "1765541512769",
      "video_path": "/abs/path/1765541512769_src_360p_h264.mp4",
      "query": "present the women's double-breasted mid-length leather jacket",
      "product_block": "Women's double-breasted mid-length leather jacket Product Category: Women's Clothing",
      "duration": 82.0,
      "main_category": "Women's Clothing",
      "leaf_category": "Leather Jacket"
    }
"""
import argparse
import os

import stage1_candidate_selection as s1
import stage2_composition as s2
import stage3_quality_diversity_filter as s3
import stage4_reasoning_trace as s4
import stage5_assemble as s5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input ⟨video, query⟩ JSONL")
    ap.add_argument("--workdir", default="./_build")
    ap.add_argument("--dry-run-filter", action="store_true",
                    help="skip MD-VQA/GroundingDINO in Stage 3 (CLIP diversity only)")
    ap.add_argument("--trace-subset", type=int, default=17000)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    p = lambda n: os.path.join(args.workdir, n)  # noqa: E731

    s1.run(args.input,           p("s1_candidates.jsonl"))
    s2.run(p("s1_candidates.jsonl"), p("s2_composition.jsonl"))
    s3.run(p("s2_composition.jsonl"), p("s3_filtered.jsonl"), dry_run=args.dry_run_filter)
    s4.run(p("s3_filtered.jsonl"), p("s4_traced.jsonl"), subset=args.trace_subset)
    s5.run(p("s4_traced.jsonl"),
           sft_out=p("sft_cot_17k.jsonl"),
           grpo_out=p("grpo_49k.jsonl"))
    print("[run_pipeline] done. outputs in", args.workdir)


if __name__ == "__main__":
    main()

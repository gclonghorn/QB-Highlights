# -*- coding: utf-8 -*-
"""Stage 4 — Reasoning-trace generation (17K subset, Qwen3-Max).

For a randomly sampled ~17K subset of the filtered samples, rewrite the coarse
(candidate-selection) and fine (composition) reasoning into a concise, standardized
reasoning trace with three fields (scan_overview / segment_qualification /
budgeted_composition) via the REASONING_TRACE_PROMPT.

The structured trace is what Stage 5 renders into the SFT `<think>` block and the
GRPO `solution` reference trace.
"""
import argparse
import random

from prompts import REASONING_TRACE_PROMPT
from clients import QwenClient
from config import DEFAULT
from utils_io import read_jsonl, write_jsonl


def run(in_path: str, out_path: str, subset: int = DEFAULT.trace_subset_size,
        seed: int = 0, cfg=DEFAULT):
    client = QwenClient(model=cfg.trace_model)
    recs = read_jsonl(in_path)

    # deterministic random subset (paper: "a randomly sampled 17K subset")
    rng = random.Random(seed)
    idx = list(range(len(recs)))
    rng.shuffle(idx)
    chosen = set(idx[:subset])

    out = []
    for i, r in enumerate(recs):
        if i not in chosen:
            r["reasoning_trace"] = None
            out.append(r)
            continue
        prompt = REASONING_TRACE_PROMPT.format(
            candidate_selection=_dump(r.get("candidate_analysis"), r.get("candidates")),
            composition=_dump(r["composition"].get("analysis"), r["composition"].get("clips")),
        )
        try:
            r["reasoning_trace"] = client.generate_json(prompt)
        except Exception as e:  # noqa: BLE001
            r["reasoning_trace"] = None
            r["_stage4_error"] = str(e)
        out.append(r)
    write_jsonl(out_path, out)
    n_trace = sum(1 for r in out if r.get("reasoning_trace"))
    print(f"[stage4] generated {n_trace} reasoning traces (subset={subset}) -> {out_path}")


def _dump(analysis, clips):
    import json
    return json.dumps({"analysis": analysis, "clips": clips}, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--subset", type=int, default=DEFAULT.trace_subset_size)
    ap.add_argument("--seed", type=int, default=0)
    run(**vars(ap.parse_args()))

# -*- coding: utf-8 -*-
"""Run all six QB-Highlights metrics on a prediction directory and print a table.

    Score@Pred, Prec@4.0            (human annotations)
    MD-VQA, DOVER                   (independent model-based quality)
    Compositional Diversity (CLIP), Diversity(DINO)   (set-level diversity)
"""
import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, extra):
    cmd = [sys.executable, os.path.join(HERE, script), "--pred-dir", ARGS.pred_dir] + extra
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--skip-quality", action="store_true",
                    help="skip MD-VQA/DOVER (heavy; need external repos)")
    ARGS = ap.parse_args()

    print("=" * 60, "\nHuman metrics\n", "=" * 60)
    run("eval_score.py", [])

    print("=" * 60, "\nDiversity\n", "=" * 60)
    run("eval_diversity.py", ["--backbone", "clip"])
    run("eval_diversity.py", ["--backbone", "dino"])

    if not ARGS.skip_quality:
        print("=" * 60, "\nModel-based quality\n", "=" * 60)
        run("eval_mdvqa.py", [])
        run("eval_dover.py", [])

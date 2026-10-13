# -*- coding: utf-8 -*-
"""Run all six QB-Highlights metrics on a prediction directory and print a table.

    Score@Pred, Prec@4.0                          (human annotations)
    MD-VQA, DOVER                                 (model-based quality)
    Compositional Diversity (CLIP), Diversity(DINO)   (set-level diversity)

MD-VQA and DOVER require local clones of their official repos (see README §5);
without --mdvqa-root / --dover-root they are skipped with a note.
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
    ap.add_argument("--mdvqa-root", default=None,
                    help="local clone of github.com/zzc-1998/MD-VQA")
    ap.add_argument("--dover-root", default=None,
                    help="local clone of github.com/VQAssessment/DOVER")
    ap.add_argument("--dover-opt", default=None,
                    help="DOVER config yml (default: <dover-root>/dover.yml)")
    ap.add_argument("--skip-quality", action="store_true",
                    help="skip MD-VQA/DOVER entirely (heavy; need external repos)")
    ARGS = ap.parse_args()

    print("=" * 60, "\nHuman metrics\n", "=" * 60)
    run("eval_score.py", [])

    print("=" * 60, "\nDiversity\n", "=" * 60)
    run("eval_diversity.py", ["--backbone", "clip"])
    run("eval_diversity.py", ["--backbone", "dino"])

    if ARGS.skip_quality:
        print("\nMD-VQA / DOVER skipped (--skip-quality)")
    else:
        print("=" * 60, "\nModel-based quality\n", "=" * 60)
        if ARGS.mdvqa_root:
            run("eval_mdvqa.py", ["--mdvqa-root", ARGS.mdvqa_root])
        else:
            print("MD-VQA skipped: pass --mdvqa-root (clone github.com/zzc-1998/MD-VQA)")
        if ARGS.dover_root:
            extra = ["--dover-root", ARGS.dover_root]
            if ARGS.dover_opt:
                extra += ["--opt", ARGS.dover_opt]
            run("eval_dover.py", extra)
        else:
            print("DOVER skipped: pass --dover-root (clone github.com/VQAssessment/DOVER)")

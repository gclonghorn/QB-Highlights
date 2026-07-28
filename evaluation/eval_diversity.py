# -*- coding: utf-8 -*-
"""Set-level diversity metrics (paper Table 3 + rebuttal).

Uniformly samples `--n-frames` (default 3) frames across the *concatenated*
timeline of a video's predicted highlight clips, extracts visual features, and
computes  diversity = 1 - mean(pairwise cosine similarity).

  --backbone clip  ->  Compositional Diversity   (CLIP ViT-B/32, paper Table 3)
  --backbone dino  ->  Diversity(DINO)            (DINOv2, feature space disjoint
                                                   from CLIP; rebuttal metric)

Averaged over the ~0.9k eval videos (missing predictions contribute 0).
"""
import argparse
import itertools

import cv2
import numpy as np
import torch
from PIL import Image

import metrics_common as mc


class CLIPBackbone:
    name = "clip"
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def feat(self, img: Image.Image) -> np.ndarray:
        inp = self.proc(images=img, return_tensors="pt").to(self.device)
        f = self.model.get_image_features(**inp)
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().flatten()


class DINOBackbone:
    name = "dino"
    def __init__(self, device):
        # DINOv2 ViT-B/14 via torch.hub (feature space independent of CLIP)
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
        import torchvision.transforms as T
        self.tf = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def feat(self, img: Image.Image) -> np.ndarray:
        x = self.tf(img).unsqueeze(0).to(self.device)
        f = self.model(x)                       # (1, d) CLS feature
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().flatten()


def sample_frames(clip_paths, n):
    """Uniformly sample n frames across the concatenated clip timeline."""
    segs = []                                   # (path, frame_count)
    total = 0
    for p in clip_paths:
        cap = cv2.VideoCapture(p)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if fc > 0:
            segs.append((p, fc)); total += fc
    if total == 0:
        return []
    frames = []
    for k in range(1, n + 1):
        gidx = int((2 * k - 1) / (2 * n) * total)
        gidx = max(0, min(gidx, total - 1))
        off = 0
        for p, fc in segs:
            if gidx < off + fc:
                cap = cv2.VideoCapture(p)
                cap.set(cv2.CAP_PROP_POS_FRAMES, gidx - off)
                ok, fr = cap.read(); cap.release()
                if ok:
                    frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                break
            off += fc
    return frames


def diversity(feats) -> float:
    if len(feats) < 2:
        return 0.0
    sims = [float(np.dot(feats[i], feats[j]))
            for i, j in itertools.combinations(range(len(feats)), 2)]
    return 1.0 - sum(sims) / len(sims)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--backbone", choices=["clip", "dino"], default="clip")
    ap.add_argument("--eval-jsonl", default=mc.EVAL_JSONL)
    ap.add_argument("--n-frames", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bk = CLIPBackbone(device) if args.backbone == "clip" else DINOBackbone(device)

    ids = mc.load_eval_ids(args.eval_jsonl)
    vals = []
    for vid in ids:
        clips = mc.clip_mp4s(args.pred_dir, vid)
        frames = sample_frames(clips, args.n_frames) if clips else []
        vals.append(diversity([bk.feat(f) for f in frames]) if len(frames) >= 2 else 0.0)

    label = "Compositional Diversity" if args.backbone == "clip" else "Diversity(DINO)"
    print(f"videos            : {len(ids)}")
    print(f"{label:<18}: {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()

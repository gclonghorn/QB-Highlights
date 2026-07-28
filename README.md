# QB-Highlights

Code and data release for **QB-Highlights: Quality-Guided Budgeted Highlight Detection in
Videos with Dense Query-Relevant Moments**

## Repository layout

```
QB-Highlights-Release/
├── dataset/
│   ├── trainset/            49.1K GRPO + 17.5K SFT-CoT pseudo-labeled samples
│   │   └── construction/    teacher-assisted pseudo-label pipeline (prompts + code)
│   └── testset/             0.9k-video test set + dense per-clip human annotations
├── training/                Stage-1 SFT + Stage-2 GRPO
launchers + reward plugin
├── inference/               QCR inference over the test set -> per-video clips + json
├── evaluation/              6 metrics (see below)
├── DATASHEET.md             provenance, ethics, full release scope
└── LICENSE
```

## 1. Dataset (`dataset/`)

- **Test set** — `dataset/testset/test_0.9k.jsonl` (~0.9k items: `id`, `query`,
  categories) + `dataset/testset/test_set_annotations.csv` (dense 2 s clip scores from
  3 anonymized annotators).
- **Training set** — `dataset/trainset/{sft_cot_17k, grpo_49k}.jsonl` (MLLM-assisted
  pseudo labels + reasoning traces).
- All `videos` fields in the JSONLs are `videos/<video_id>.mp4`; raw videos will be released
  separately after paper publication + company content review. See `DATASHEET.md`.

## 2. Environment

- Python ≥ 3.10, CUDA 12.x, PyTorch ≥ 2.1, `ffmpeg` + `ffprobe` on `PATH`.
- Backbone: `Qwen3-VL-8B-Instruct` (from Hugging Face Hub / ModelScope).
- Per-component pinned deps: `training/requirements.txt`,
  `inference/requirements.txt`, `evaluation/requirements.txt`,
  `dataset/trainset/construction/requirements.txt`.

**QCR checkpoint** — `<HF link withheld for double-blind review;
will be added in the camera-ready>` — download to `checkpoints/QCR-checkpoint/` and pass
its path to `--vlm-model-path`.

## 3. Training (`training/`)

Clone official **ms-swift 3.11.0** into `training/ms-swift-3.11.0/`
(`git clone --branch v3.11.0 https://github.com/modelscope/ms-swift`), then run either
launcher via `torchrun` (single-node multi-GPU by default; multi-node via env vars):

```bash
# Stage 1 — SFT. Paper: eff. bs 128, lr 1e-5, max_len 49152, zero3.
NGPUS=8 bash training/train_sft.sh

# Stage 2 — GRPO. Paper: eff. bs 96, lr 5e-6, num_generations 8,
# vLLM colocate. Init from the Stage-1 SFT checkpoint.
MODEL_PATH=./outputs/qcr_sft/<ckpt-dir> NGPUS=8 bash training/train_grpo.sh

# multi-node: NNODES=2 NODE_RANK=0 MASTER_ADDR=<host> MASTER_PORT=29500 NGPUS=8 bash ...
```

## 4. Inference (`inference/`)

```bash
bash inference/infer.sh   # edit --video-dir and --vlm-model-path first
```

Runs QCR over `dataset/testset/test_0.9k.jsonl`. For each item it produces
`<id>_spot.json` (metadata + raw model response with `<think>…</think><answer>…</answer>`+ parsed `clips`), `<id>_spot_fine.mp4` (concatenated highlight), and
`fine_clips/<id>_spot/clip_NN.mp4` (each selected segment). Videos are resolved as
`<video-dir>/<id>.mp4`; multi-GPU dispatch via `--num-gpus`.

## 5. Evaluation (`evaluation/`)

```bash
python evaluation/run_all.py --pred-dir results/qcr_preds
```

| Metric | Script | Source |
|---|---|---|
| Score@Pred, Prec@4.0 | `eval_score.py` | human annotations |
| MD-VQA | `eval_mdvqa.py` | MD-VQA CVPR'23 (external) |
| DOVER | `eval_dover.py` | DOVER ICCV'23 (external) |
| Compositional Diversity | `eval_diversity.py --backbone clip` | CLIP |
| Diversity(DINO) | `eval_diversity.py --backbone dino` | DINOv2 |

Predictions layout expected by every metric: `<pred-dir>/<id>_spot.json` (with `clips[].timestamp`);
MD-VQA / DOVER / Diversity additionally read the extracted mp4s at
`<pred-dir>/fine_clips/<id>_spot/*.mp4` (or fallback `<pred-dir>/<id>_spot_fine.mp4`).

**External models — user must set up:**

```bash
# MD-VQA (Zhang et al., CVPR'23)
git clone https://github.com/zzc-1998/MD-VQA
# put pretrained weights into MD-VQA/data/:
#   efficientnet_v2_s-dd5fe13b.pth, r2plus1d_18-91a641e6.pth, MDVQA_LSVQ_TaoLive.pth
python evaluation/eval_mdvqa.py --pred-dir <pred> --mdvqa-root /path/to/MD-VQA

# DOVER (Wu et al., ICCV'23)
git clone https://github.com/VQAssessment/DOVER && cd DOVER && pip install -e .
# put DOVER.pth + convnext_tiny_1k_224_ema.pth into DOVER/pretrained_weights/
python evaluation/eval_dover.py --pred-dir <pred> --dover-root /path/to/DOVER --opt /path/to/DOVER/dover.yml
```
DINOv2 auto-downloads via `torch.hub` on first run; CLIP auto-downloads via `transformers`.

## Results (QB-Highlights, 0.9k test videos, budget = 10 s)

| | Score@Pred ↑ | Prec@4.0 ↑ | MD-VQA ↑ | Comp. Diversity ↑ |
|---|---|---|---|---|
| Best baseline | 3.239 | 34.15 | 86.512 | 0.095 |
| **QCR (Ours)** | **3.507** | **37.67** | **88.170** | **0.119** |

## Appendix — rebuilding the training set from scratch

Skip this if you use the released `dataset/trainset/*.jsonl`. 

To reproduce the
pseudo-label construction, see `dataset/trainset/construction/`: teacher-assisted
coarse-to-fine pipeline — Gemini-2.5-Pro candidate selection (360p) → composition
(720p) → MD-VQA + Grounding DINO + CLIP filters → Qwen3-Max reasoning traces →
`sft_cot_17k.jsonl` + `grpo_49k.jsonl`. 

The three teacher prompts (verbatim from paper
Appendix B.2) are in `dataset/trainset/construction/prompts.py`; run with
`python dataset/trainset/construction/run_pipeline.py --input <video_query.jsonl>`.

## Ethics, licensing & release scope

See **`DATASHEET.md`** for provenance, privacy/copyright/consent, and the full release
table. Annotations, pseudo-labels, reasoning traces, construction/training/inference/
evaluation code are released now; Checkpoints link will be released after double-blind review;  **raw videos** will be released after
paper publication + company content review. 

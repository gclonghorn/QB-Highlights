# QB-Highlights — Datasheet, Provenance & Release

Provenance, ethics (privacy / copyright / consent), and the full release scope for
QB-Highlights.

## Provenance

QB-Highlights is built from **50K e-commerce livestream videos** (1080p, ~80 s avg)
from a leading e-commerce platform — **publicly viewable product-presentation replays**
(a host demonstrating a product), not private recordings. Each video is paired with a
fine-grained product-presentation **query** (e.g. "present the women's double-breasted
mid-length leather jacket"). 49.1K samples carry MLLM-assisted pseudo labels; 902 test
videos carry dense human quality annotations.

## Ethics: consent, copyright, privacy

- **Consent & copyright.** Every source video was originally uploaded to the platform by
  the merchant / host under the platform's terms of use, which grant the platform the
  right to distribute and reuse it (including for internal research). We use only
  publicly viewable product-presentation replays, for **non-commercial academic
  research**. Raw-video public release requires a routine company content review after
  paper publication; we expect it to pass. Until then, raw pixels are withheld and this
  repo releases only annotations, pseudo-labels, reasoning traces, code, and
  checkpoints.
- **Privacy.** The subjects on screen are commercial livestream hosts performing
  product demos on a public platform, not private users; no bystanders or third-party
  private content is in scope. What we release:
  - **Video IDs** are the platform's product / stream identifiers only — they contain
    no host names, contacts, or other PII.
  - **Human annotations** are anonymized: worker IDs are replaced by
    `annotator_1/2/3`, and only numeric 1–5 quality scores are kept.
  - **Text fields** (queries, product categories, teacher reasoning traces) describe
    on-screen product presentation only; we screened for personal data.
- **Intended use.** Video highlight detection / temporal grounding research. **Not**
  for surveillance, person re-identification, or commercial use of the footage.

## Release scope

| # | Item | Status | Location |
|---|---|---|---|
| 1 | Raw videos (50K train + 902 test) | **After publication + company review** (expected to pass) | HF (pending) |
| 2 | Video IDs | ✅ Released | `dataset/testset/test_0.9k.jsonl`; `videos/<id>.mp4` refs in trainset |
| 3 | Test set + human annotations | ✅ Released | `dataset/testset/{test_0.9k.jsonl, test_set_annotations.csv}` |
| 4 | Training pseudo-labels + CoT reasoning traces | ✅ Released | `dataset/trainset/{grpo_49k.jsonl, sft_cot_17k.jsonl}` |
| 5 | Training-set construction pipeline + prompts | ✅ Released | `dataset/trainset/construction/` |
| 6 | Evaluation code (6 metrics) | ✅ Released | `evaluation/` |
| 7 | Trained checkpoints (SFT, GRPO-merged) | ✅ Released | HF: `<link withheld for double-blind review; camera-ready will include the URL>` — `QCR-checkpoint/` (Qwen license inherited) |
| 8 | Full dataset | ✅ Released (raw video via #1) | `dataset/` |
| + | Training code + configs | ✅ Released | `training/` |
| + | Inference code | ✅ Released | `inference/` |

All released annotations / pseudo-labels / traces / prompts / code are CC-BY-4.0 /
Apache-2.0 (see `LICENSE`); checkpoints inherit the Qwen license.

## Reproducibility

All training, inference, and evaluation in this repo run end-to-end once the videos are
obtained (item #1 after publication + review, or re-collected via the released video
IDs on the platform).

# QB-Highlights — Datasheet, Provenance & Release

Provenance, ethics (privacy / copyright / consent), release scope for QB-Highlights.

## Provenance

QB-Highlights is built from **50K e-commerce livestream videos** (1080p, ~80 s average length) from a leading e-commerce platform. The source videos are **publicly viewable product-presentation replays** rather than private recordings. Each video is paired with a fine-grained product-presentation **query** (e.g., “present the women’s double-breasted mid-length leather jacket”). The training split contains **49.1K** samples with MLLM-assisted pseudo labels, and the test split contains **0.9K** videos with dense human quality annotations.

## Ethics: privacy, copyright, and consent

Source videos are publicly viewable product-presentation replays hosted on the platform. Their collection and use for this benchmark underwent **internal legal/data review** and comply with applicable privacy regulations and platform policies. Users are informed through platform policies that public content may be used for research/AI-related purposes.

To reduce privacy risks in the release:
- **Released video identifiers are anonymized** and contain no host names, contact information, or other direct PII.
- **Human annotations are anonymized**: worker IDs are replaced by `annotator_1/2/3`, and only numeric 1–5 quality scores are retained.
- **Text fields** (e.g., queries, product categories, and teacher reasoning traces) describe on-screen product presentation only and were screened for personal data.


## Release scope

| # | Item | Status | Location |
|---|---|---|---|
| 1 | Raw videos (~50k train + ~0.9k test) | **After publication + routine company review** | HF / release page (pending) |
| 2 | Video IDs / metadata | ✅ Released | `dataset/testset/test_0.9k.jsonl`; train-set video refs |
| 3 | Test set + human annotations | ✅ Released | `dataset/testset/{test_0.9k.jsonl, test_set_annotations.csv}` |
| 4 | Training pseudo-labels + reasoning traces | ✅ Released | `dataset/trainset/{grpo_49k.jsonl, sft_cot_17k.jsonl}` |
| 5 | Training-set construction pipeline + prompts | ✅ Released | `dataset/trainset/construction/` |
| 6 | Evaluation code | ✅ Released | `evaluation/` |
| 7 | Trained checkpoints (SFT, GRPO-merged) | ✅ Released | `QCR-checkpoint/` *(Qwen license inherited; public link added after de-anonymization)* |
| 8 | Training code + configs | ✅ Released | `training/` |
| 9 | Inference code | ✅ Released | `inference/` |

**Licenses.** Code is released under **Apache-2.0**. Released dataset assets (including annotations, pseudo-labels, prompts, reasoning traces, and metadata) are released under **ODC-BY**. Checkpoints inherit the **Qwen** license. Raw-video release will occur after publication, subject to routine internal company content review.

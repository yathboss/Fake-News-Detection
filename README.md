# Multimodal Fake News Detection using RAG

## Problem Statement

Fake news often pairs real or manipulated images with misleading captions, so a
text-only classifier misses evidence that lives in the image, and an
image-only classifier misses evidence that lives in the wording of the
claim. This project trains a **multimodal** (text + image) classifier on the
**MMFakeBench** dataset and augments it with a **retrieval-augmented
generation (RAG)** step that fetches related claim text from a small
evidence store before classification, to see whether retrieved context helps
or hurts the prediction.

## Architecture Overview

```
Claim text + (optional) image
        |
        |-- SentenceTransformer (all-MiniLM-L6-v2) --> FAISS similarity search --> top-k retrieved evidence text
        |                                                                                 |
        |-- claim text + "[EVIDENCE] " + retrieved text ------------------------------>  (RAG-augmented text)
        |
        v
   CLIP (openai/clip-vit-base-patch32), frozen
        |-- text_embeds (from augmented or raw text)
        |-- image_embeds (from the image, or a blank placeholder if none given)
        v
   concat(text_embeds, image_embeds) --> MLP head (Linear -> ReLU -> Dropout -> Linear) --> Real / Fake
```

- **CLIP encoder + MLP head** (`multimodal.py`): CLIP's text and image towers
  are frozen; only a small fusion MLP is trained. This keeps training
  CPU-feasible and avoids overfitting the full CLIP backbone on a small
  dataset.
- **SentenceTransformers + FAISS retrieval** (`build_index.py`,
  `rag_retriever.py`): claim texts are embedded with `all-MiniLM-L6-v2` and
  indexed in a FAISS `IndexFlatIP` (cosine similarity via L2-normalized
  inner product). At inference time, the top-k most similar texts are
  concatenated onto the claim before it is encoded by CLIP.
- **FastAPI backend** (`backend/main.py`): exposes `POST /verify` (claim +
  optional image -> label, confidence, retrieved evidence) and `GET /health`.
- **Streamlit UI** (`app.py`): a thin demo client for the backend API.

## ⚠️ Fallback Assumptions & Known Limitations

- **No official train split.** MMFakeBench, as provided here, ships only
  `MMFakeBench_val.json` (1,020 items) and `MMFakeBench_test.json` (10,000
  items) — there is no separate `train.json`. Training uses a **stratified
  80/20 split of `MMFakeBench_val.json`** (816 train / 204 held-out
  validation items, seed 42, class-stratified so the split doesn't amplify
  the existing class imbalance). `dataset/train.json` and `dataset/test.json`
  in this repo are not used by any documented command here — they appear to
  be byproducts of an earlier Colab-compatibility pass and simply mirror
  `MMFakeBench_val.json` / `MMFakeBench_test.json` with path formatting
  differences.
- **Small, imbalanced dataset.** 816 training examples is small for
  fine-tuning even a lightweight fusion head, and both the train split and
  the test split skew ~70% "Fake" / 30% "Real". Metrics below use
  class-balanced training weights and macro-averaged precision/recall/F1 to
  avoid the majority class dominating the score, but the absolute numbers
  should be read as "what this small baseline achieves on this dataset", not
  as a general claim about multimodal fake-news detection.
- **RAG evidence store is the dataset itself, not an external knowledge
  base.** The "evidence" FAISS index is built from this project's own
  training-split claim texts (see the leakage section below) — real-world
  deployment would need a genuine fact-checking corpus. The RAG variant
  evaluated here measures whether retrieving *similar past claims* changes
  the model's prediction, not whether it has access to independently
  verified facts.
- **The multimodal classifier was not trained with RAG-augmented text.**
  "CLIP multimodal + RAG" below reuses the same checkpoint trained on plain
  claim text, with retrieved evidence appended only at inference time. It is
  a zero-shot robustness check, not an ablation of a model that was actually
  trained to use retrieval.
- **Legacy components.** `backend/pipeline.py`, `evaluation/evaluate.py` and
  `utils/{embed_index,llm_infer,ner_kg,retriever,load_data}.py` are an older
  text-only RAG + knowledge-graph prototype ("Phase 1") that predates the
  multimodal pivot. They are not wired into `backend/main.py` or any command
  below and are not covered by the evaluation or leakage audit in this
  README.
- **`train_mm.py` and `backend/inference_mm.py` are currently broken**, not
  just unused: they are older duplicates of `train.py` / `infer_single.py`
  that still import from the pre-restructuring `models.multimodal_classifier`
  path and were not fixed here, since no README command uses them. Treat
  `train.py` and `infer_single.py` as the maintained versions.

## 📁 Dataset Placement

Ensure your dataset files are present inside the `dataset/` directory:
- `dataset/MMFakeBench_val.json`
- `dataset/MMFakeBench_test.json`
- `dataset/images/` (images extracted from the dataset's image archive)

None of these (nor `checkpoints/`, nor the FAISS index files) are committed
to this repository — see `.gitignore`. You need your own copy of the
MMFakeBench data and images to run training/evaluation locally.

## 🛠️ Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

All dependencies (including `torch`, `transformers`, `pillow`, and
`python-multipart`, which used to be a separate manual `pip install` line in
this README) are now pinned with exact versions in `requirements.txt`.

## 🚀 How to Run the System

### 1. Build the Evidence Store (RAG)
Builds the FAISS index from the **training portion only** of the dataset
(see "Leakage Checks" below for why):
```bash
python build_index.py --corpus_json dataset/MMFakeBench_val.json
```

### 2. Training the Multimodal Baseline
```bash
python train.py --annotation_file dataset/MMFakeBench_val.json --epochs 5 --batch_size 16 --seed 42
```

### 3. Evaluation Script
Evaluates the CLIP multimodal model (no RAG) on the test split:
```bash
python evaluation/eval_mm.py --test_annotation dataset/MMFakeBench_test.json
```
For the full baseline comparison (majority-class, text-only, multimodal,
multimodal+RAG) used to produce the results table below:
```bash
python evaluation/run_full_evaluation.py
```

### 4. Inference Scripts

**Single Inference** via CLI (`infer_single.py`):
```bash
python infer_single.py --text "Scientists announce a new vaccine trial for the common cold" --image_path "dataset/images/example.png"
```
*(This retrieves FAISS evidence automatically and predicts a label.)*

**Batch Inference** via CLI (`infer_batch.py`):
```bash
python infer_batch.py --split test --annotation_file dataset/MMFakeBench_test.json
```
*(Outputs predictions to `outputs/batch_predictions.csv`)*

### 5. Local App / UI Demo

First, start the backend API (port 8080 avoids a Windows/Hyper-V reservation
that sometimes blocks port 8000):
```bash
uvicorn backend.main:app --reload --port 8080
```

Then, in a separate terminal, run the Streamlit demo:
```bash
python -m streamlit run app.py
```

## 🔎 Leakage Checks (Task 2 audit)

Before any metric below was computed, `evaluation/leakage_audit.py` checked
the RAG evidence store and the dataset splits for leakage. Full output is in
`evaluation/leakage_audit_results.json`; summary:

| Check | Finding |
|---|---|
| Does the FAISS index store labels or label-revealing fields? | No — `build_index.py` writes `metadata.json` as a plain list of claim **text only**; no `gt_answers`/`fake_cls`/`image_path` is stored. |
| Did the index include the held-out 20% validation items? | **Yes, before the fix.** The default corpus was the *full* `MMFakeBench_val.json`, so all 203 unique held-out-validation texts were present verbatim in the evidence store. **Fixed:** `build_index.py` now defaults to `--split_mode train`, indexing only the 80% training partition (same seed/split as training). After the fix, only 3 held-out validation texts remain in the index — these are exact-duplicate captions that legitimately occur in both partitions under different rows/images, a pre-existing data characteristic, not a code bug. |
| Did the index include test items? | 3 exact text duplicates naturally occur between `MMFakeBench_val.json` and `MMFakeBench_test.json` (pre-existing in the raw data, not introduced by the split). Out of 10,000 test items this is a 0.03% overlap. |
| Can a claim retrieve itself (or an exact duplicate) as "evidence"? | **Yes, before the fix** (a claim's exact text, present in the index, is always its own nearest neighbor). **Fixed:** `rag_retriever.py` now over-fetches from FAISS and filters out any result whose normalized text exactly matches the query. |
| Exact/near-duplicate claims across splits? | Within `MMFakeBench_val.json`: 4 distinct texts repeated across 20 of 1,020 rows. Within `MMFakeBench_test.json`: 53 distinct texts repeated across 109 of 10,000 rows. Between val and test: 3 exact duplicates; a bounded fuzzy-match sample (300 val texts, length-bucketed against test, similarity ≥ 0.9) found 0 additional near-duplicates. Between the train/val-split partitions of `MMFakeBench_val.json`: 2 texts appear in both (same wording, different row/image) — a minor, data-inherent leak that the split logic cannot remove without dropping those rows entirely. |

## 📊 Results (TEST split: `dataset/MMFakeBench_test.json`, n = 10,000, seed = 42)

| Method | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---|---|---|---|
| Majority-class baseline (always "Fake") | 0.7000 | 0.3500 | 0.5000 | 0.4118 |
| Text-only (frozen CLIP text + MLP) | 0.6773 | 0.6431 | 0.6635 | 0.6454 |
| CLIP multimodal (text + image), no RAG | 0.7853 | 0.7446 | 0.7488 | 0.7466 |
| CLIP multimodal + RAG | 0.7719 | 0.7306 | 0.7430 | 0.7358 |

How to read this:
- The test set is 70% "Fake", so the majority-class baseline already gets
  0.70 accuracy while never predicting "Real". Use **macro-F1** to compare
  methods.
- Adding the image (multimodal vs. text-only) improves macro-F1 from 0.6454
  to 0.7466 on this test set.
- **RAG did not help here.** Appending the retrieved evidence slightly lowered
  macro-F1 (0.7466 → 0.7358). Two caveats: the evidence store holds only
  training-split claims, and the classifier was never trained on
  evidence-augmented input (see Limitations).
- Each number comes from a single run with seed 42. No confidence intervals
  or multi-seed variance were computed, so small differences (such as the
  roughly 0.01 RAG gap) should not be read as conclusive.
- The text-only model was trained for this comparison
  (`checkpoints/text_only_best.pt`, best held-out-validation macro-F1 0.7436).
  The multimodal rows use the existing `checkpoints/model_best.pt`, which was
  not retrained (its recorded best validation macro-F1 is 0.8494).

Confusion matrices and the full per-method JSON/CSV are in
`evaluation/results.md`, `evaluation/results.json`, and
`evaluation/results.csv` (generated by
`python evaluation/run_full_evaluation.py`).

## Limitations

- **No official train split** (see above) — all numbers come from an 80/20
  fallback split of the validation file.
- **Small training set (816 items)** relative to the 10,000-item test set;
  results can be sensitive to the random split seed.
- **RAG here retrieves from the model's own training claims**, not an
  independent fact base, and the classifier was never trained on
  RAG-augmented input — see "Fallback Assumptions" above for what the
  CLIP+RAG result does and does not show.
- **CPU-only evaluation environment**: all numbers in this README were
  produced on a CPU (no CUDA available), which constrains how much
  retraining/hyperparameter search was practical for this resume project.
- **Known-unused/legacy code remains in the repo** (`backend/pipeline.py`,
  `evaluation/evaluate.py`, several `utils/` modules) from an earlier
  text-only prototype; it is not covered by this README's commands or
  metrics.

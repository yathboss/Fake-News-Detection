"""
Full evaluation on the held-out TEST split (dataset/MMFakeBench_test.json) for:
  1. Majority-class baseline
  2. Text-only model (frozen CLIP text tower + small MLP head, trained on the
     same 80/20 train/val split of MMFakeBench_val.json used for the
     multimodal model)
  3. CLIP multimodal (text + image), no RAG -- uses checkpoints/model_best.pt
  4. CLIP multimodal + RAG -- same checkpoint, claim text augmented at
     inference time with evidence retrieved from the leakage-safe FAISS
     index (see rag_retriever.py / build_index.py)

All four variants are evaluated on the identical test set, with the same
label derivation (dataset/mmfakebench.py:derive_mmfakebench_label) and the
same fixed random seed. No metric here is estimated or invented -- every
number comes from actually running inference against the real test data.

Outputs:
  evaluation/results.md    (human-readable report)
  evaluation/results.json  (machine-readable, per-method metrics + confusion matrix)
  evaluation/results.csv   (flat summary table)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.mmfakebench import MMFakeBenchDataset, derive_mmfakebench_label  # noqa: E402
from multimodal import MultimodalFakeNewsClassifier  # noqa: E402
from rag_retriever import RealRAGRetriever  # noqa: E402

SEED = 42
VAL_ANNOTATION = "dataset/MMFakeBench_val.json"
TEST_ANNOTATION = "dataset/MMFakeBench_test.json"
IMAGE_DIR = "dataset/images"
CLIP_NAME = "openai/clip-vit-base-patch32"
CHECKPOINT_PATH = "checkpoints/model_best.pt"
LABEL_NAMES = ["Real", "Fake"]
LABEL_ORDER = [0, 1]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split():
    """Same stratified 80/20 split (seed=42) used everywhere else in this repo."""
    train_ds = MMFakeBenchDataset(VAL_ANNOTATION, IMAGE_DIR, split_mode="train", split_ratio=0.8, seed=SEED)
    val_ds = MMFakeBenchDataset(VAL_ANNOTATION, IMAGE_DIR, split_mode="val", split_ratio=0.8, seed=SEED)
    return train_ds.data, val_ds.data


def load_test():
    with open(TEST_ANNOTATION, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
        "n": len(y_true),
    }


# ---------------------------------------------------------------------------
# 1. Majority-class baseline
# ---------------------------------------------------------------------------
def run_majority_baseline(train_data, test_data):
    train_labels = [derive_mmfakebench_label(i) for i in train_data]
    majority_label = max(set(train_labels), key=train_labels.count)
    test_labels = [derive_mmfakebench_label(i) for i in test_data]
    preds = [majority_label] * len(test_labels)
    metrics = compute_metrics(test_labels, preds)
    metrics["majority_label"] = LABEL_NAMES[majority_label]
    return metrics


# ---------------------------------------------------------------------------
# 2. Text-only model: frozen CLIP text tower + small MLP head
# ---------------------------------------------------------------------------
class TextOnlyClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, text_features):
        return self.head(text_features)


def encode_texts(clip_model, processor, texts, device, batch_size=32):
    clip_model.eval()
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            feats = clip_model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def run_text_only(train_data, val_data, test_data, device, epochs=30, lr=1e-3, out_path="checkpoints/text_only_best.pt"):
    print("[text-only] Loading frozen CLIP text tower...")
    clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
    for p in clip_model.parameters():
        p.requires_grad = False
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    embed_dim = clip_model.config.projection_dim

    train_texts = [i["text"] for i in train_data]
    val_texts = [i["text"] for i in val_data]
    test_texts = [i["text"] for i in test_data]
    train_labels = torch.tensor([derive_mmfakebench_label(i) for i in train_data])
    val_labels = torch.tensor([derive_mmfakebench_label(i) for i in val_data])
    test_labels = [derive_mmfakebench_label(i) for i in test_data]

    print(f"[text-only] Encoding {len(train_texts)} train / {len(val_texts)} val / {len(test_texts)} test texts...")
    t0 = time.time()
    train_feats = encode_texts(clip_model, processor, train_texts, device)
    val_feats = encode_texts(clip_model, processor, val_texts, device)
    test_feats = encode_texts(clip_model, processor, test_texts, device)
    print(f"[text-only] Encoding done in {time.time() - t0:.1f}s")

    model = TextOnlyClassifier(embed_dim).to(device)
    total = len(train_labels)
    counts = [int((train_labels == c).sum()) for c in LABEL_ORDER]
    class_weights = torch.tensor([total / max(1, 2 * c) for c in counts], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_f1 = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(train_feats.to(device))
        loss = criterion(logits, train_labels.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_feats.to(device))
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
        _, _, val_f1, _ = precision_recall_fscore_support(
            val_labels.numpy(), val_preds, labels=LABEL_ORDER, average="macro", zero_division=0,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[text-only] epoch {epoch+1}/{epochs} loss={loss.item():.4f} val_macro_f1={val_f1:.4f}")

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[text-only] Best val macro-F1={best_f1:.4f}, saved to {out_path}")

    model.eval()
    with torch.no_grad():
        test_logits = model(test_feats.to(device))
        test_preds = torch.argmax(test_logits, dim=1).cpu().numpy().tolist()
    return compute_metrics(test_labels, test_preds)


# ---------------------------------------------------------------------------
# 3 & 4. CLIP multimodal, no-RAG and +RAG (single pass over images for both)
# ---------------------------------------------------------------------------
def run_multimodal(test_data, device, batch_size=16, rag_top_k=3, limit=None):
    print("[multimodal] Loading CLIP processor + multimodal classifier checkpoint...")
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    model = MultimodalFakeNewsClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    retriever = RealRAGRetriever()

    data = test_data if limit is None else test_data[:limit]
    n = len(data)

    labels_all, no_rag_preds, rag_preds = [], [], []
    missing_images = 0
    t_start = time.time()

    for start in range(0, n, batch_size):
        batch = data[start:start + batch_size]
        texts = [b["text"] for b in batch]
        labels = [derive_mmfakebench_label(b) for b in batch]

        images = []
        for b in batch:
            img_path = os.path.join(IMAGE_DIR, b.get("image_path", "").lstrip("/\\"))
            if os.path.exists(img_path):
                images.append(Image.open(img_path).convert("RGB"))
            else:
                images.append(Image.new("RGB", (224, 224), color="black"))
                missing_images += 1

        rag_texts = []
        for t in texts:
            evidence = retriever.retrieve(t, top_k=rag_top_k)
            if evidence:
                rag_texts.append(t + " [EVIDENCE] " + " ".join(e["text"] for e in evidence))
            else:
                rag_texts.append(t)

        with torch.no_grad():
            inputs_no_rag = processor(text=texts, images=images, return_tensors="pt",
                                       padding=True, truncation=True, max_length=77).to(device)
            logits_no_rag = model(input_ids=inputs_no_rag.input_ids,
                                   attention_mask=inputs_no_rag.attention_mask,
                                   pixel_values=inputs_no_rag.pixel_values)
            preds_no_rag = torch.argmax(logits_no_rag, dim=1).cpu().numpy()

            inputs_rag = processor(text=rag_texts, images=images, return_tensors="pt",
                                    padding=True, truncation=True, max_length=77).to(device)
            logits_rag = model(input_ids=inputs_rag.input_ids,
                                attention_mask=inputs_rag.attention_mask,
                                pixel_values=inputs_rag.pixel_values)
            preds_rag = torch.argmax(logits_rag, dim=1).cpu().numpy()

        labels_all.extend(labels)
        no_rag_preds.extend(preds_no_rag.tolist())
        rag_preds.extend(preds_rag.tolist())

        done = start + len(batch)
        if done % (batch_size * 20) == 0 or done == n:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("inf")
            print(f"[multimodal] {done}/{n} done, {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    print(f"[multimodal] Missing images: {missing_images}/{n}")
    return compute_metrics(labels_all, no_rag_preds), compute_metrics(labels_all, rag_preds)


def write_reports(results: dict):
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    for name, m in results.items():
        rows.append({
            "method": name,
            "n": m["n"],
            "accuracy": round(m["accuracy"], 4),
            "precision_macro": round(m["precision_macro"], 4),
            "recall_macro": round(m["recall_macro"], 4),
            "f1_macro": round(m["f1_macro"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv("evaluation/results.csv", index=False)

    lines = ["# Evaluation Results (TEST split: dataset/MMFakeBench_test.json)", ""]
    lines.append(f"n = {next(iter(results.values()))['n']} test examples, fixed seed = {SEED}")
    lines.append("")
    lines.append("| Method | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |")
    lines.append("|---|---|---|---|---|")
    for name, m in results.items():
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | "
                      f"{m['recall_macro']:.4f} | {m['f1_macro']:.4f} |")
    lines.append("")
    for name, m in results.items():
        lines.append(f"## {name} - Confusion matrix")
        lines.append("")
        lines.append("| | Pred Real | Pred Fake |")
        lines.append("|---|---|---|")
        cm = m["confusion_matrix"]
        lines.append(f"| True Real | {cm[0][0]} | {cm[0][1]} |")
        lines.append(f"| True Fake | {cm[1][0]} | {cm[1][1]} |")
        lines.append("")

    with open("evaluation/results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote evaluation/results.md, evaluation/results.json, evaluation/results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on test items, for smoke-testing.")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, val_data = load_split()
    test_data = load_test()
    print(f"Train: {len(train_data)}, Val (held-out): {len(val_data)}, Test: {len(test_data)}")

    results = {}

    print("\n=== 1. Majority-class baseline ===")
    results["Majority-class baseline"] = run_majority_baseline(train_data, test_data if args.limit is None else test_data[:args.limit])

    print("\n=== 2. Text-only model ===")
    results["Text-only (frozen CLIP text + MLP)"] = run_text_only(
        train_data, val_data, test_data if args.limit is None else test_data[:args.limit], device,
    )

    print("\n=== 3 & 4. CLIP multimodal (no RAG / + RAG) ===")
    no_rag_metrics, rag_metrics = run_multimodal(test_data, device, batch_size=args.batch_size, limit=args.limit)
    results["CLIP multimodal, no RAG"] = no_rag_metrics
    results["CLIP multimodal + RAG"] = rag_metrics

    write_reports(results)

    print("\n=== Summary ===")
    for name, m in results.items():
        print(f"{name}: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()

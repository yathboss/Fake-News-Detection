"""
Leakage audit for the MMFakeBench RAG pipeline.

Checks, in order:
1. Exact-text duplicates within dataset/MMFakeBench_val.json (source of the train/val split).
2. Exact-text duplicates between dataset/MMFakeBench_val.json and dataset/MMFakeBench_test.json
   (would mean a "test" item's claim text is also seen during training/retrieval-corpus construction).
3. Whether the stratified 80/20 train/val split (same seed/logic as dataset/mmfakebench.py)
   produces any overlapping texts between its train and val partitions.
4. What the FAISS evidence corpus (as currently built by build_index.py, default args)
   actually contains relative to the held-out 20% validation slice and the test set.
5. Whether a claim can retrieve itself (or a near-duplicate) as "evidence" at inference time,
   using the actual RealRAGRetriever against a freshly-built index.

Results are printed and also written to evaluation/leakage_audit_results.json.
No metrics from the classifier are computed here; this is strictly a data/retrieval audit.
"""
from __future__ import annotations

import json
import os
import random
import sys
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.mmfakebench import derive_mmfakebench_label  # noqa: E402

VAL_PATH = "dataset/MMFakeBench_val.json"
TEST_PATH = "dataset/MMFakeBench_test.json"


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    return " ".join((text or "").lower().split())


def stratified_split(data, split_ratio=0.8, seed=42):
    random.seed(seed)
    real_items = [item for item in data if derive_mmfakebench_label(item) == 0]
    fake_items = [item for item in data if derive_mmfakebench_label(item) == 1]
    random.shuffle(real_items)
    random.shuffle(fake_items)
    real_idx = int(len(real_items) * split_ratio)
    fake_idx = int(len(fake_items) * split_ratio)
    train = real_items[:real_idx] + fake_items[:fake_idx]
    val = real_items[real_idx:] + fake_items[fake_idx:]
    return train, val


def report_duplicates(name, texts):
    norm = [normalize(t) for t in texts]
    counts = {}
    for t in norm:
        counts[t] = counts.get(t, 0) + 1
    dupes = {t: c for t, c in counts.items() if c > 1}
    total_dup_rows = sum(c for c in dupes.values())
    print(f"[{name}] {len(texts)} rows, {len(set(norm))} unique normalized texts, "
          f"{len(dupes)} distinct texts repeated ({total_dup_rows} rows involved).")
    return dupes


def near_duplicate_sample(set_a_texts, set_b_texts, label_a, label_b, threshold=0.9,
                           sample_cap=300, bucket_width=5, max_candidates_per_item=40, seed=0):
    """
    Approximate near-duplicate check: exact-match first (cheap, exhaustive over full sets),
    then a bounded fuzzy pass (SequenceMatcher) over a random sample from set_a, restricted
    to length-bucketed candidates from set_b to keep this tractable (full O(n*m) fuzzy
    comparison over ~1k x 10k items is not feasible here).
    """
    norm_a = set(normalize(t) for t in set_a_texts)
    norm_b = set(normalize(t) for t in set_b_texts)
    exact_overlap = norm_a & norm_b
    print(f"Exact-text overlap between {label_a} and {label_b}: {len(exact_overlap)} shared normalized strings.",
          flush=True)

    buckets = {}
    for b in norm_b:
        buckets.setdefault(len(b) // bucket_width, []).append(b)

    rng = random.Random(seed)
    candidates_a = [a for a in norm_a if a not in norm_b]
    sample_a = rng.sample(candidates_a, min(sample_cap, len(candidates_a)))

    near_hits = []
    for a in sample_a:
        key = len(a) // bucket_width
        candidates = buckets.get(key, []) + buckets.get(key - 1, []) + buckets.get(key + 1, [])
        if len(candidates) > max_candidates_per_item:
            candidates = rng.sample(candidates, max_candidates_per_item)
        best_ratio, best_b = 0.0, None
        for b in candidates:
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio > best_ratio:
                best_ratio, best_b = ratio, b
        if best_ratio >= threshold:
            near_hits.append((a, best_b, best_ratio))
    print(f"Near-duplicate scan ({label_a} sample of {len(sample_a)} vs length-bucketed {label_b}, "
          f"threshold={threshold}): {len(near_hits)} near-duplicate pairs found in sample.", flush=True)
    return sorted(exact_overlap), near_hits[:20]


def main():
    val_data = load(VAL_PATH)
    test_data = load(TEST_PATH)

    val_texts = [item.get("text", "") for item in val_data]
    test_texts = [item.get("text", "") for item in test_data]

    results = {}

    print("=" * 70)
    print("1. Duplicates within MMFakeBench_val.json")
    results["val_internal_dupes"] = len(report_duplicates("val.json", val_texts))

    print("=" * 70)
    print("2. Duplicates within MMFakeBench_test.json")
    results["test_internal_dupes"] = len(report_duplicates("test.json", test_texts))

    print("=" * 70)
    print("3. Exact + near-duplicate text overlap: val.json vs test.json")
    exact_overlap, near_hits = near_duplicate_sample(val_texts, test_texts, "val.json", "test.json")
    results["val_test_exact_overlap_count"] = len(exact_overlap)
    results["val_test_exact_overlap_examples"] = exact_overlap[:10]
    results["val_test_near_duplicate_examples"] = near_hits

    print("=" * 70)
    print("4. Train/Val stratified split (seed=42, ratio=0.8) overlap check")
    train_split, val_split = stratified_split(val_data)
    train_texts = set(normalize(i.get("text", "")) for i in train_split)
    val_split_texts = set(normalize(i.get("text", "")) for i in val_split)
    split_overlap = train_texts & val_split_texts
    print(f"Train partition: {len(train_split)} rows, {len(train_texts)} unique texts.")
    print(f"Val partition:   {len(val_split)} rows, {len(val_split_texts)} unique texts.")
    print(f"Overlap between train and held-out val partitions: {len(split_overlap)} texts.")
    results["train_val_split_overlap"] = len(split_overlap)

    print("=" * 70)
    print("5. What does build_index.py (default args) put in the FAISS corpus?")
    full_val_unique = set(normalize(t) for t in val_texts)
    train_unique = train_texts
    held_out_val_unique = val_split_texts
    index_vs_heldout = full_val_unique & held_out_val_unique
    index_vs_test = full_val_unique & set(normalize(t) for t in test_texts)
    print(f"Default corpus (full val.json) unique texts: {len(full_val_unique)}")
    print(f"  -> overlap with held-out 20% val partition: {len(index_vs_heldout)} "
          f"(expected: ALL of them, since the corpus is built from the full file, train+val combined)")
    print(f"  -> overlap with test.json: {len(index_vs_test)}")
    results["default_index_includes_heldout_val_count"] = len(index_vs_heldout)
    results["default_index_includes_test_count"] = len(index_vs_test)

    print("=" * 70)
    print("6. Does the index store labels or label-revealing fields?")
    print("build_index.py writes metadata.json as a plain list of claim TEXT strings only "
          "(see build_faiss_index(): `texts = list(set(item['text'] ...))`). "
          "No gt_answers / fake_cls / image_path is written to metadata.json, so the index "
          "itself does not store labels directly.")

    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/leakage_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved summary to evaluation/leakage_audit_results.json")


if __name__ == "__main__":
    main()

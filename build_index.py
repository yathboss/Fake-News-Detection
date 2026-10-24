import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install faiss-cpu and sentence-transformers: pip install faiss-cpu sentence-transformers")
    sys.exit(1)

from dataset.mmfakebench import MMFakeBenchDataset


def load_corpus_texts(corpus_path, split_mode, split_ratio, seed):
    """
    Loads the claim texts that are allowed into the RAG evidence store.

    IMPORTANT (leakage fix): by default this only uses the TRAIN partition of the
    annotation file (the same stratified 80/20 split, same seed, used by
    dataset/mmfakebench.py for training). Building the index from the full
    file would leak the held-out validation split's own claim text into the
    retriever, letting a validation/test claim retrieve an exact or
    near-duplicate copy of itself as "evidence". Pass --split_mode all only if
    you explicitly want the old (leaky) behavior, e.g. for debugging.
    """
    if split_mode == "all":
        with open(corpus_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                data = [json.loads(line) for line in f]
    else:
        # image_dir is irrelevant here: we only read .data (text/labels), we never
        # load images, so a dummy path is fine.
        dataset = MMFakeBenchDataset(
            corpus_path, image_dir="dataset/images", split_mode=split_mode,
            split_ratio=split_ratio, seed=seed,
        )
        data = dataset.data
    return data


def build_faiss_index(corpus_path, index_path, metadata_path, split_mode="train", split_ratio=0.8, seed=42):
    print(f"Loading corpus from {corpus_path} (split_mode={split_mode})")
    data = load_corpus_texts(corpus_path, split_mode, split_ratio, seed)

    # Extract unique texts from the dataset to act as the Evidence Store.
    # In a real scenario, this would be an external trusted corpus. We use the
    # dataset text as the fallback corpus, restricted to the training portion
    # only (see load_corpus_texts) so held-out val/test claims are not in the
    # evidence store verbatim.
    texts = list(set([item["text"] for item in data if "text" in item]))
    print(f"Extracted {len(texts)} unique text snippets for the evidence store.")
    
    if len(texts) == 0:
        print("No texts found to index. Aborting.")
        return
        
    print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding corpus (this might take a moment)...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype('float32')
    
    # L2 normalize for cosine similarity search
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner product = Cosine similarity if normalized
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f)
        
    print(f"FAISS index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_json", type=str, default="dataset/MMFakeBench_val.json", help="Path to json dataset to build the corpus from")
    parser.add_argument("--index_out", type=str, default="retrieval/index.faiss")
    parser.add_argument("--metadata_out", type=str, default="retrieval/metadata.json")
    parser.add_argument("--split_mode", type=str, default="train", choices=["train", "all"],
                         help="'train' (default, leakage-safe) uses only the 80%% training "
                              "partition of --corpus_json. 'all' uses the full file, which "
                              "leaks the held-out validation claims into the evidence store.")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    build_faiss_index(args.corpus_json, args.index_out, args.metadata_out,
                       split_mode=args.split_mode, split_ratio=args.split_ratio, seed=args.seed)
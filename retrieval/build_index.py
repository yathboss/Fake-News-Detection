import argparse
import sys
import os
import json
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install faiss-cpu and sentence-transformers: pip install faiss-cpu sentence-transformers")
    sys.exit(1)

def build_faiss_index(corpus_path, index_path, metadata_path):
    print(f"Loading corpus from {corpus_path}")
    data = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f]
            
    # Extract unique texts from the dataset to act as the Evidence Store
    # In a real scenario, this would be an external trusted corpus. We use the dataset text as the fallback corpus.
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
    
    args = parser.parse_args()
    build_faiss_index(args.corpus_json, args.index_out, args.metadata_out)

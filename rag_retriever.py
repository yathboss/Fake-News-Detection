import os
import json
import numpy as np

class RealRAGRetriever:
    """
    Real working retriever using FAISS and SentenceTransformers.
    """
    def __init__(self, index_path="retrieval/index.faiss", metadata_path="retrieval/metadata.json"):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.ready = True
            print("Loaded FAISS index and evidence metadata.")
        except ImportError:
            print("Missing dependencies. Please install faiss-cpu and sentence-transformers.")
            self.ready = False
        except Exception as e:
            print(f"Failed to load FAISS index: {e}. RAG retrieval will be empty.")
            self.ready = False

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieves top_k evidence snippets for a text query.
        Returns a list of dictionaries with 'text' and 'score'.

        Exact or near-exact self-matches (the query text itself, or trivial
        whitespace/case variants of it) are excluded from the results: the
        goal of RAG here is independent evidence, not a restatement of the
        claim being checked. We over-fetch from FAISS and filter, since the
        index may legitimately contain the same claim text from a different
        row (e.g. the same headline paired with a different image).
        """
        if not self.ready:
            print("Warning: RAG Retriever not ready, returning empty list.")
            return []

        normalized_query = " ".join(query.lower().split())

        # Encode query
        import faiss # lazy import to catch scoping issues if needed
        query_emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_emb)

        # Over-fetch so that filtering out self-matches still leaves top_k results.
        search_k = max(top_k * 4, top_k + 10)
        scores, indices = self.index.search(query_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            candidate_text = self.metadata[idx]
            if " ".join(candidate_text.lower().split()) == normalized_query:
                continue  # exclude self-match
            results.append({
                "text": candidate_text,
                "score": float(score)
            })
            if len(results) >= top_k:
                break
        return results
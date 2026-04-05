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
        """
        if not self.ready:
            print("Warning: RAG Retriever not ready, returning empty list.")
            return []
            
        # Encode query
        import faiss # lazy import to catch scoping issues if needed
        query_emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_emb)
        
        # Search index
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "text": self.metadata[idx],
                    "score": float(score)
                })
        return results

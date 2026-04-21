"""
FAISS vector store for embeddings.
"""
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional

class VectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, dimension: int, index_path: Optional[str] = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine after normalization)
        self.metadata: List[Dict[str, Any]] = []   # تخزين البيانات الوصفية لكل متجه
        self.index_path = index_path
        
        if index_path and os.path.exists(index_path):
            self.load(index_path)
    
    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        Add a single embedding to the index.
        Args:
            embedding: 1D numpy array of shape (dimension,)
            metadata: dict with post_id, text, etc.
        Returns:
            index position
        """
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        # تأكد من أن النوع float32
        embedding = embedding.astype(np.float32)
        self.index.add(embedding)
        self.metadata.append(metadata)
        return self.index.ntotal - 1
    
    def add_batch(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        """Add multiple embeddings at once."""
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
    
    def search(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for k nearest neighbors.
        Args:
            query: 1D embedding (dimension,) or 2D (1, dimension)
            k: number of results
        Returns:
            list of dicts with 'score' (similarity) and metadata
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)
        
        # FAISS IndexFlatIP يعطي تشابه جيب التمام (لأن المتجهات معمّمة)
        scores, indices = self.index.search(query, min(k, self.index.ntotal))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                'score': float(scores[0][i]),
                'metadata': self.metadata[idx]
            })
        return results
    
    def save(self, path: str):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path: str):
        """Load FAISS index and metadata from disk."""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
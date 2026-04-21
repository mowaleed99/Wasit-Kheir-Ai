"""
FAISS vector store with atomic persistence for Modal.
- Atomic writes: write to temp file → rename (prevents corruption)
- Logging on every load/save/commit
- No thread-safety needed: Modal's allow_concurrent_inputs=1 handles that
"""
import numpy as np
import faiss
import pickle
import os
import logging
import tempfile
import shutil
from typing import List, Dict, Any, Optional

logger = logging.getLogger("faiss-store")


class FAISSStore:
    """FAISS index with automatic persistence to Modal Volume."""

    def __init__(self, dimension: int, index_name: str = "default",
                 persist_dir: Optional[str] = None):
        self.dimension = dimension
        self.index_name = index_name
        self.persist_dir = persist_dir
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

        if persist_dir:
            self._try_load()

    # ── File paths ──

    def _index_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.index_name}.faiss")

    def _meta_path(self) -> str:
        return os.path.join(self.persist_dir, f"{self.index_name}.pkl")

    # ── Load ──

    def _try_load(self):
        """Load FAISS index + metadata from Volume if files exist."""
        idx_path = self._index_path()
        meta_path = self._meta_path()

        if os.path.exists(idx_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(idx_path)
                with open(meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(
                    f"✅ Loaded '{self.index_name}': "
                    f"{self.index.ntotal} vectors, "
                    f"{len(self.metadata)} metadata entries"
                )
            except Exception as e:
                logger.error(
                    f"❌ Failed to load '{self.index_name}': {e}. "
                    f"Starting with empty index."
                )
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []
        else:
            logger.info(
                f"📝 No existing index for '{self.index_name}'. "
                f"Starting fresh."
            )

    # ── Atomic Save ──

    def _persist(self):
        """Save FAISS index + metadata to disk ATOMICALLY.

        Strategy:
        1. Write to temp files in the same directory
        2. os.replace() atomically swaps old → new
        3. If crash happens mid-write, old files are untouched

        This prevents corruption from partial writes.
        """
        if not self.persist_dir:
            return

        os.makedirs(self.persist_dir, exist_ok=True)

        idx_path = self._index_path()
        meta_path = self._meta_path()

        # Write FAISS index to temp file, then atomic rename
        tmp_idx = idx_path + ".tmp"
        faiss.write_index(self.index, tmp_idx)
        os.replace(tmp_idx, idx_path)  # Atomic on POSIX (Linux = Modal)

        # Write metadata to temp file, then atomic rename
        tmp_meta = meta_path + ".tmp"
        with open(tmp_meta, 'wb') as f:
            pickle.dump(self.metadata, f)
        os.replace(tmp_meta, meta_path)  # Atomic

        logger.info(
            f"💾 Saved '{self.index_name}': "
            f"{self.index.ntotal} vectors"
        )

    # ── CRUD Operations ──

    def add(self, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        """Add a single embedding. Auto-persists to disk."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)
        self.index.add(embedding)
        self.metadata.append(metadata)
        self._persist()
        return self.index.ntotal - 1

    def add_batch(self, embeddings: np.ndarray,
                  metadatas: List[Dict[str, Any]]):
        """Add multiple embeddings at once. Single persist call."""
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata.extend(metadatas)
        self._persist()

    def search(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for k nearest neighbors. Returns list of {score, metadata}."""
        if self.index.ntotal == 0:
            return []

        if query.ndim == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)

        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, actual_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            if idx < len(self.metadata):
                results.append({
                    'score': float(scores[0][i]),
                    'metadata': self.metadata[idx]
                })
            else:
                logger.warning(
                    f"⚠️ Index {idx} out of metadata range "
                    f"({len(self.metadata)}). Skipping."
                )
        return results

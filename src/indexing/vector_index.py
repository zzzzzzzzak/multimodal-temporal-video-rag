"""High-performance dense vector index for multimodal embedding search.

Provides exact inner-product and cosine-similarity nearest neighbor search
with metadata filtering, ranking, and persistence.
"""

from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class SearchResult:
    """Retrieved candidate from vector index."""
    index: int
    score: float
    metadata: Dict[str, Any]
    vector: Optional[np.ndarray] = None


class DenseVectorIndex:
    """In-memory dense vector index supporting cosine similarity and dot product."""
    def __init__(self, dimension: int, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric.lower()
        self.vectors: Optional[np.ndarray] = None  # (N, D) float32
        self.metadata: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norms, 1e-8)

    def add(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Add dense vectors and corresponding metadata payloads.
        
        Args:
            vectors: 2D array of shape (N, dimension).
            metadata: Optional list of N metadata dicts.
            
        Returns:
            List of assigned integer indices.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}"
            )

        n = len(vectors)
        if self.metric == "cosine":
            vectors = self._normalize(vectors)

        start_idx = len(self)
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])

        if metadata is None:
            metadata = [{} for _ in range(n)]
        elif len(metadata) != n:
            raise ValueError(f"Metadata length ({len(metadata)}) != vectors length ({n})")

        self.metadata.extend(metadata)
        return list(range(start_idx, start_idx + n))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_fn: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Retrieve top-k nearest neighbors for a single query vector.
        
        Args:
            query_vector: 1D or 2D array of shape (dimension,) or (1, dimension).
            top_k: Maximum number of matches to return.
            filter_fn: Optional callable taking metadata dict and returning bool.
            
        Returns:
            Ranked list of SearchResult instances sorted by descending score.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if len(q) != self.dimension:
            raise ValueError(f"Query dimension {len(q)} != index dimension {self.dimension}")

        if self.metric == "cosine":
            q_norm = np.linalg.norm(q)
            q = q / max(q_norm, 1e-8)

        # Compute dot products across all indexed vectors: (N,)
        scores = np.dot(self.vectors, q)

        # Rank indices
        ranked_indices = np.argsort(-scores)

        results = []
        for idx in ranked_indices:
            meta = self.metadata[idx]
            if filter_fn is not None and not filter_fn(meta):
                continue
            
            results.append(
                SearchResult(
                    index=int(idx),
                    score=float(scores[idx]),
                    metadata=meta,
                    vector=self.vectors[idx],
                )
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: str):
        """Persist vectors and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "dimension": self.dimension,
            "metric": self.metric,
            "metadata": self.metadata,
        }
        meta_path = filepath + ".meta.json"
        vec_path = filepath + ".npy"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if self.vectors is not None:
            np.save(vec_path, self.vectors)

    @classmethod
    def load(cls, filepath: str) -> "DenseVectorIndex":
        """Load persisted index from disk."""
        meta_path = filepath + ".meta.json"
        vec_path = filepath + ".npy"
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        index = cls(dimension=data["dimension"], metric=data["metric"])
        index.metadata = data["metadata"]
        if os.path.isfile(vec_path):
            index.vectors = np.load(vec_path)
        return index

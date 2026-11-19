"""Unit tests for DenseVectorIndex."""

import os
import tempfile
import numpy as np
import pytest
from src.indexing.vector_index import DenseVectorIndex, SearchResult


def test_index_add_and_search():
    dim = 64
    index = DenseVectorIndex(dimension=dim, metric="cosine")
    
    vecs = np.random.randn(10, dim).astype(np.float32)
    meta = [{"id": f"item_{i}", "category": "A" if i < 5 else "B"} for i in range(10)]
    
    assigned = index.add(vecs, meta)
    assert len(index) == 10
    assert assigned == list(range(10))

    # Search with first vector - should return item_0 as top-1 with score approx 1.0
    results = index.search(vecs[0], top_k=3)
    assert len(results) == 3
    assert results[0].metadata["id"] == "item_0"
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


def test_index_dimension_mismatch():
    index = DenseVectorIndex(dimension=32)
    bad_vecs = np.random.randn(5, 64).astype(np.float32)
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        index.add(bad_vecs)


def test_index_metadata_filtering():
    dim = 16
    index = DenseVectorIndex(dimension=dim)
    vecs = np.random.randn(4, dim).astype(np.float32)
    meta = [{"tag": "cat"}, {"tag": "dog"}, {"tag": "cat"}, {"tag": "bird"}]
    index.add(vecs, meta)

    results = index.search(vecs[0], top_k=5, filter_fn=lambda m: m.get("tag") == "cat")
    assert len(results) == 2
    for r in results:
        assert r.metadata["tag"] == "cat"


def test_index_save_and_load():
    dim = 16
    index = DenseVectorIndex(dimension=dim)
    vecs = np.random.randn(5, dim).astype(np.float32)
    meta = [{"val": i} for i in range(5)]
    index.add(vecs, meta)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_idx")
        index.save(path)
        
        loaded = DenseVectorIndex.load(path)
        assert len(loaded) == 5
        assert loaded.dimension == dim
        assert loaded.metadata[2]["val"] == 2
        np.testing.assert_allclose(loaded.vectors, index.vectors, atol=1e-5)

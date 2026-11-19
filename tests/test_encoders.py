"""Unit tests for VisionClipEncoder, TextQueryEncoder, and TemporalCrossModalReranker."""

import pytest
import torch
from src.config import ModelConfig
from src.models.vision_encoder import VisionClipEncoder
from src.models.text_encoder import TextQueryEncoder, SimpleSubwordTokenizer
from src.models.cross_encoder import TemporalCrossModalReranker


def test_vision_clip_encoder():
    cfg = ModelConfig(embedding_dim=128, vision_feature_dim=64)
    model = VisionClipEncoder(cfg)
    
    # 2 segments, each with 4 keyframes of 3x64x64
    x = torch.rand(2, 4, 3, 64, 64)
    clip_embs, frame_embs = model(x)

    assert clip_embs.shape == (2, 128)
    assert frame_embs.shape == (2, 4, 128)
    
    # Verify L2 normalization
    norms = torch.norm(clip_embs, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4)


def test_text_query_encoder():
    cfg = ModelConfig(embedding_dim=128, text_hidden_dim=64)
    model = TextQueryEncoder(cfg)

    queries = ["find expanding circle", "car moves fast"]
    pooled, token_embs = model.encode_text(queries)

    assert pooled.shape == (2, 128)
    assert token_embs.shape[0] == 2
    assert token_embs.shape[2] == 128

    norms = torch.norm(pooled, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-4)


def test_cross_modal_reranker():
    cfg = ModelConfig(embedding_dim=128)
    reranker = TemporalCrossModalReranker(cfg)

    # Batch of 2 pairs: query tokens (2, 6, 128) and keyframes (2, 4, 128)
    q_tokens = torch.rand(2, 6, 128)
    f_embs = torch.rand(2, 4, 128)

    out = reranker(q_tokens, f_embs)
    assert "alignment_score" in out
    assert "boundary_deltas" in out
    assert "attention_weights" in out

    assert out["alignment_score"].shape == (2, 1)
    assert (out["alignment_score"] >= 0.0).all() and (out["alignment_score"] <= 1.0).all()
    assert out["boundary_deltas"].shape == (2, 2)
    assert out["attention_weights"].shape == (2, 6, 4)

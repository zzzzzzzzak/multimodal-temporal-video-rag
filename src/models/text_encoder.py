"""Text query encoder with multi-head self-attention and multimodal projection."""

import math
import re
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class SimpleSubwordTokenizer:
    """Deterministic hash-based subword and token encoder with zero external dependencies."""
    def __init__(self, vocab_size: int = 4096, max_len: int = 32):
        self.vocab_size = vocab_size
        self.max_len = max_len

    def tokenize(self, text: str) -> List[int]:
        clean = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        words = clean.split()
        if not words:
            return [1]  # <unk>
        
        tokens = [2]  # <cls> token
        for w in words:
            # Hash word and character tri-grams to fit in vocab
            h_word = (hash(w) % (self.vocab_size - 10)) + 10
            tokens.append(int(h_word))
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tri = w[i:i+3]
                    h_tri = (hash(tri) % (self.vocab_size - 10)) + 10
                    tokens.append(int(h_tri))
            if len(tokens) >= self.max_len - 1:
                break

        tokens.append(3)  # <sep> token
        return tokens[:self.max_len]

    def pad_batch(self, token_lists: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_b_len = min(self.max_len, max(len(t) for t in token_lists))
        batch_ids = []
        masks = []
        for t in token_lists:
            ids = t[:max_b_len]
            mask = [1] * len(ids)
            if len(ids) < max_b_len:
                pad_len = max_b_len - len(ids)
                ids += [0] * pad_len
                mask += [0] * pad_len
            batch_ids.append(ids)
            masks.append(mask)

        return (
            torch.tensor(batch_ids, dtype=torch.long),
            torch.tensor(masks, dtype=torch.bool),
        )


class TextQueryEncoder(nn.Module):
    """Encodes natural language queries into the shared 512-D multimodal space."""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.tokenizer = SimpleSubwordTokenizer(vocab_size=self.config.text_vocab_size)
        
        # Token Embeddings + Learned Position Embeddings
        self.token_embed = nn.Embedding(self.config.text_vocab_size, self.config.text_hidden_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, self.config.text_hidden_dim) * 0.02)
        
        # Transformer Self-Attention Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.text_hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.text_hidden_dim * 2,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.layer_norm = nn.LayerNorm(self.config.text_hidden_dim)
        
        # Projection to common multimodal embedding space
        self.proj = nn.Sequential(
            nn.Linear(self.config.text_hidden_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
        )

    def encode_text(
        self, queries: List[str], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode query strings to normalized embeddings.
        
        Args:
            queries: List of query strings.
            device: Target torch device.
            
        Returns:
            pooled_embeddings: Normalized query embeddings (B, D).
            token_embeddings: Per-token contextualized embeddings (B, L, D).
        """
        token_lists = [self.tokenizer.tokenize(q) for q in queries]
        input_ids, mask = self.tokenizer.pad_batch(token_lists)
        if device is not None:
            input_ids = input_ids.to(device)
            mask = mask.to(device)
            
        return self(input_ids, mask)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, seq_len = input_ids.shape
        x = self.token_embed(input_ids) + self.pos_embed[:, :seq_len, :]
        
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None
        hidden = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        hidden = self.layer_norm(hidden)
        
        # Project tokens to multimodal dimension
        token_embs = self.proj(hidden)
        token_embs = F.normalize(token_embs, p=2, dim=-1)
        
        # Masked average pooling
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(token_embs * expanded_mask, dim=1) / torch.clamp(expanded_mask.sum(dim=1), min=1.0)
        else:
            pooled = token_embs.mean(dim=1)

        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled, token_embs

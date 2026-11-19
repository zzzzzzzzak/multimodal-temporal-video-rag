"""Cross-modal attention re-ranker and temporal boundary regression."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class TemporalCrossModalReranker(nn.Module):
    """Fine-grained cross-attention re-ranker and boundary regression head."""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        dim = self.config.embedding_dim

        # Multi-Head Cross Attention (Query: Text Tokens, Key/Value: Video Frames)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)

        # Fine Alignment Scoring Head
        self.scoring_head = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )

        # Temporal Boundary Regression Head: outputs normalized (delta_start, delta_end)
        self.boundary_regressor = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),
            nn.Tanh(),  # Normalized delta in [-1, 1] relative to segment duration
        )

    def forward(
        self,
        query_token_embs: torch.Tensor,
        frame_embs: torch.Tensor,
        query_pooled: Optional[torch.Tensor] = None,
        clip_pooled: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute fine-grained cross-attention alignment and boundary regression.
        
        Args:
            query_token_embs: (B, L, D) text token embeddings.
            frame_embs: (B, K, D) video keyframe embeddings.
            query_pooled: Optional (B, D) pooled query embedding.
            clip_pooled: Optional (B, D) pooled clip embedding.
            
        Returns:
            Dictionary containing:
                'alignment_score': (B, 1) fine-grained relevance in [0, 1].
                'boundary_deltas': (B, 2) normalized (delta_start, delta_end) in [-1, 1].
                'attention_weights': (B, L, K) cross-modal attention map.
        """
        # Cross-attention: Text attends to Video Keyframes
        attn_out, attn_weights = self.cross_attn(
            query=query_token_embs,
            key=frame_embs,
            value=frame_embs,
        )  # (B, L, D), (B, L, K)
        
        attn_fused = self.norm(query_token_embs + attn_out).mean(dim=1)  # (B, D)

        if clip_pooled is None:
            clip_pooled = frame_embs.mean(dim=1)
            
        combined = torch.cat([attn_fused, clip_pooled], dim=-1)  # (B, 2D)

        alignment_score = self.scoring_head(combined)  # (B, 1)
        boundary_deltas = self.boundary_regressor(combined)  # (B, 2)

        return {
            "alignment_score": alignment_score,
            "boundary_deltas": boundary_deltas,
            "attention_weights": attn_weights,
        }

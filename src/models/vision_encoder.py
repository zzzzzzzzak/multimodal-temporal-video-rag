"""Vision backbone and temporal clip feature encoder."""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class ConvBlock(nn.Module):
    """Residual convolutional block for visual feature extraction."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class VisionClipEncoder(nn.Module):
    """Encodes keyframe images and aggregates them into normalized clip embeddings."""
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        
        # Convolutional feature extractor
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.vision_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        self.stage1 = ConvBlock(32, 64, stride=2)
        self.stage2 = ConvBlock(64, 128, stride=2)
        self.stage3 = ConvBlock(128, self.config.vision_feature_dim, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to common multimodal embedding space
        self.proj = nn.Sequential(
            nn.Linear(self.config.vision_feature_dim, self.config.embedding_dim),
            nn.LayerNorm(self.config.embedding_dim),
        )
        
        # Temporal attention pooling weights across keyframes
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.config.embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of individual frames (N, 3, H, W) -> (N, D)."""
        x = self.stem(frames)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        pooled = self.pool(x).flatten(1)
        emb = self.proj(pooled)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, keyframe_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode keyframe sets for video segments.
        
        Args:
            keyframe_batch: Tensor of shape (B, K, 3, H, W) where B is segments, K is keyframes.
            
        Returns:
            clip_embeddings: Aggregated segment embeddings (B, D).
            frame_embeddings: Per-frame embeddings (B, K, D).
        """
        b, k, c, h, w = keyframe_batch.shape
        flat_frames = keyframe_batch.view(b * k, c, h, w)
        frame_embs = self.forward_frames(flat_frames)  # (B*K, D)
        frame_embs = frame_embs.view(b, k, -1)         # (B, K, D)
        
        # Attention-weighted temporal pooling across K keyframes
        attn_weights = self.temporal_attn(frame_embs)   # (B, K, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, K, 1)
        
        clip_embs = torch.sum(attn_weights * frame_embs, dim=1)  # (B, D)
        clip_embs = F.normalize(clip_embs, p=2, dim=-1)
        
        return clip_embs, frame_embs

"""Typed configuration dataclasses for multimodal video RAG and temporal grounding."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SegmentationConfig:
    """Configuration for video scene and shot boundary detection."""
    min_scene_len_sec: float = 1.0
    max_scene_len_sec: float = 12.0
    scene_threshold: float = 0.035  # Combined HSV histogram & pixel distance threshold
    sampling_fps: float = 2.0      # Frames extracted per second for indexing
    downsample_size: Tuple[int, int] = (128, 128)


@dataclass
class ModelConfig:
    """Configuration for dual-encoder multimodal projection architecture."""
    embedding_dim: int = 512
    vision_channels: int = 3
    vision_feature_dim: int = 256
    text_vocab_size: int = 4096
    text_hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.15


@dataclass
class RetrievalConfig:
    """Configuration for vector similarity retrieval and candidate selection."""
    top_k: int = 5
    metric: str = "cosine"  # 'cosine', 'dot_product', 'euclidean'
    min_similarity: float = 0.25
    enable_reranking: bool = True


@dataclass
class GroundingConfig:
    """Configuration for timestamp boundary regression and temporal calibration."""
    temporal_iou_threshold: float = 0.50
    boundary_slack_sec: float = 0.25
    confidence_scale: float = 1.0


@dataclass
class PipelineConfig:
    """Master configuration for the Video RAG pipeline."""
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    device: str = "auto"

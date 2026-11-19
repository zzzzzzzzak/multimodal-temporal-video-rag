"""Temporal moment grounding and boundary refinement engine."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import GroundingConfig


@dataclass
class GroundedMoment:
    """A verified and timestamp-grounded temporal video moment."""
    moment_id: str
    video_id: str
    calibrated_start: float
    calibrated_end: float
    duration: float
    original_start: float
    original_end: float
    relevance_score: float
    confidence: float
    keyframe_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    attention_map: Optional[np.ndarray] = None


def compute_temporal_iou(
    pred: Tuple[float, float], gt: Tuple[float, float]
) -> float:
    """Compute Temporal Intersection-over-Union (tIoU) between two time intervals.
    
    Args:
        pred: (pred_start, pred_end) in seconds.
        gt: (gt_start, gt_end) in seconds.
        
    Returns:
        float tIoU in [0, 1].
    """
    p_start, p_end = sorted(pred)
    g_start, g_end = sorted(gt)

    inter_start = max(p_start, g_start)
    inter_end = min(p_end, g_end)
    intersection = max(0.0, inter_end - inter_start)

    union_start = min(p_start, g_start)
    union_end = max(p_end, g_end)
    union = max(1e-8, union_end - union_start)

    return float(intersection / union)


class TemporalGrounder:
    """Refines coarse retrieved video candidate segments into calibrated temporal bounds."""
    def __init__(self, config: Optional[GroundingConfig] = None):
        self.config = config or GroundingConfig()

    def ground_segment(
        self,
        segment_id: str,
        video_id: str,
        start_time: float,
        end_time: float,
        relevance_score: float,
        boundary_deltas: Optional[Tuple[float, float]] = None,
        keyframe_indices: Optional[List[int]] = None,
        metadata: Optional[Dict] = None,
        attention_map: Optional[np.ndarray] = None,
    ) -> GroundedMoment:
        """Calibrate timestamp bounds using cross-modal delta predictions."""
        duration = max(0.1, end_time - start_time)

        if boundary_deltas is not None:
            delta_s, delta_e = boundary_deltas
            # Bound shift by up to 25% of duration
            max_shift = 0.25 * duration * self.config.confidence_scale
            c_start = max(0.0, start_time + delta_s * max_shift)
            c_end = max(c_start + 0.5, end_time + delta_e * max_shift)
        else:
            c_start = start_time
            c_end = end_time

        c_start = round(c_start, 2)
        c_end = round(c_end, 2)
        c_duration = round(c_end - c_start, 2)

        # Calibrated confidence as function of alignment score and temporal compactness
        confidence = float(np.clip(relevance_score * 0.85 + 0.15, 0.0, 1.0))

        return GroundedMoment(
            moment_id=f"ground_{segment_id}",
            video_id=video_id,
            calibrated_start=c_start,
            calibrated_end=c_end,
            duration=c_duration,
            original_start=start_time,
            original_end=end_time,
            relevance_score=float(relevance_score),
            confidence=confidence,
            keyframe_indices=keyframe_indices or [],
            metadata=metadata or {},
            attention_map=attention_map,
        )

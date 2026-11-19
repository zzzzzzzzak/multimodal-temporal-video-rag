"""Motion and temporal keyframe sampling algorithms."""

from typing import List, Tuple
import numpy as np


def sample_keyframes_uniformly(
    num_frames: int,
    target_count: int = 4,
) -> List[int]:
    """Sample target_count frame indices uniformly spaced across a clip.
    
    Args:
        num_frames: Total number of frames in the clip.
        target_count: Desired number of keyframes.
        
    Returns:
        List of selected integer frame indices.
    """
    if num_frames <= 0:
        return []
    if num_frames <= target_count:
        return list(range(num_frames))

    indices = np.linspace(0, num_frames - 1, target_count, dtype=int)
    return sorted(list(set(indices.tolist())))


def sample_motion_keyframes(
    frames: List[np.ndarray],
    target_count: int = 4,
) -> List[int]:
    """Sample keyframes positioned at points of maximum visual motion/change."""
    n = len(frames)
    if n <= target_count:
        return list(range(n))

    # Calculate frame-to-frame difference energy
    diffs = [0.0]
    for i in range(1, n):
        f1 = frames[i - 1].astype(np.float32)
        f2 = frames[i].astype(np.float32)
        diff = float(np.mean(np.abs(f2 - f1)))
        diffs.append(diff)

    # Pick top motion peaks alongside start and end
    indices = {0, n - 1}
    ranked_motion = np.argsort(-np.array(diffs)).tolist()
    for idx in ranked_motion:
        indices.add(idx)
        if len(indices) >= target_count:
            break

    return sorted(list(indices))

"""Video scene and shot boundary segmenter."""

from dataclasses import dataclass, field
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.config import SegmentationConfig
from src.indexing.frame_sampler import sample_keyframes_uniformly


@dataclass
class VideoSegment:
    """A semantically coherent temporal slice of a video stream."""
    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    duration: float
    frame_indices: List[int]
    keyframes: List[np.ndarray] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class VideoSegmenter:
    """Partitions raw continuous video streams into indexable temporal segments."""
    def __init__(self, config: Optional[SegmentationConfig] = None):
        self.config = config or SegmentationConfig()

    def _compute_frame_difference(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Compute normalized HSV color histogram difference between adjacent frames."""
        h1 = cv2.cvtColor(f1, cv2.COLOR_RGB2HSV)
        h2 = cv2.cvtColor(f2, cv2.COLOR_RGB2HSV)
        
        hist1 = cv2.calcHist([h1], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist2 = cv2.calcHist([h2], [0, 1], None, [16, 16], [0, 180, 0, 256])
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Chi-Square or correlation distance + pixel L1 difference
        score = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR))
        l1_diff = float(np.mean(np.abs(f1.astype(np.float32) - f2.astype(np.float32)))) / 255.0
        return score + l1_diff

    def segment_frames(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        video_id: str = "video_0",
        keyframes_per_segment: int = 4,
    ) -> List[VideoSegment]:
        """Segment a sequence of video frames into coherent temporal clips.
        
        Args:
            frames: Sequence of RGB frames.
            fps: Frame rate of the input video stream.
            video_id: Unique identifier for the source video.
            keyframes_per_segment: Number of representative frames to extract.
            
        Returns:
            List of VideoSegment instances.
        """
        if not frames:
            return []

        num_frames = len(frames)
        min_frames = max(1, int(self.config.min_scene_len_sec * fps))
        max_frames = max(min_frames + 1, int(self.config.max_scene_len_sec * fps))

        cut_points = [0]
        last_cut = 0

        for i in range(1, num_frames):
            frame_dist = self._compute_frame_difference(frames[i - 1], frames[i])
            elapsed = i - last_cut

            is_threshold_cut = (frame_dist >= self.config.scene_threshold) and (elapsed >= min_frames)
            is_max_length_cut = elapsed >= max_frames

            if is_threshold_cut or is_max_length_cut:
                cut_points.append(i)
                last_cut = i

        if cut_points[-1] != num_frames:
            cut_points.append(num_frames)

        segments = []
        for idx in range(len(cut_points) - 1):
            start_f = cut_points[idx]
            end_f = cut_points[idx + 1]
            seg_frames = frames[start_f:end_f]
            
            start_t = round(start_f / fps, 3)
            end_t = round(end_f / fps, 3)
            duration = round(end_t - start_t, 3)

            # Sample keyframe indices
            sampled_rel_idx = sample_keyframes_uniformly(len(seg_frames), keyframes_per_segment)
            keyframes = [seg_frames[k] for k in sampled_rel_idx]
            abs_indices = [start_f + k for k in sampled_rel_idx]

            seg_id = f"{video_id}_seg_{idx:04d}"
            segments.append(
                VideoSegment(
                    segment_id=seg_id,
                    video_id=video_id,
                    start_time=start_t,
                    end_time=end_t,
                    duration=duration,
                    frame_indices=abs_indices,
                    keyframes=keyframes,
                    metadata={"shot_index": idx, "num_frames": len(seg_frames)},
                )
            )

        return segments

    def segment_video_file(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        sample_stride: int = 2,
    ) -> List[VideoSegment]:
        """Extract and segment frames directly from a video file."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        v_id = video_id or os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 24.0

        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % sample_stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Downsample for faster segmentation
                h, w = self.config.downsample_size
                resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
                frames.append(resized)
            count += 1
        cap.release()

        effective_fps = fps / sample_stride
        return self.segment_frames(frames, fps=effective_fps, video_id=v_id)

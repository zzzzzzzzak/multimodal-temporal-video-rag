"""Unit tests for video segmentation and keyframe sampling."""

import numpy as np
import pytest
from src.config import SegmentationConfig
from src.indexing.video_segmenter import VideoSegmenter, VideoSegment
from src.indexing.frame_sampler import sample_keyframes_uniformly, sample_motion_keyframes


def test_sample_keyframes_uniformly():
    indices = sample_keyframes_uniformly(num_frames=20, target_count=4)
    assert len(indices) == 4
    assert indices[0] == 0
    assert indices[-1] == 19


def test_sample_motion_keyframes():
    # 10 constant frames + 1 sharp transition frame
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(10)]
    frames[5] = np.ones((32, 32, 3), dtype=np.uint8) * 255  # Motion peak
    
    indices = sample_motion_keyframes(frames, target_count=3)
    assert 5 in indices or 6 in indices


def test_video_segmenter_frames():
    cfg = SegmentationConfig(min_scene_len_sec=0.5, max_scene_len_sec=2.0)
    segmenter = VideoSegmenter(cfg)

    # Create 30 frames: 15 black, 15 white (1 scene cut in middle)
    fps = 10.0
    f1 = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(15)]
    f2 = [np.ones((64, 64, 3), dtype=np.uint8) * 255 for _ in range(15)]
    frames = f1 + f2

    segments = segmenter.segment_frames(frames, fps=fps, video_id="test_vid")
    assert len(segments) >= 2
    assert all(isinstance(s, VideoSegment) for s in segments)
    assert segments[0].video_id == "test_vid"
    assert segments[0].start_time < segments[0].end_time

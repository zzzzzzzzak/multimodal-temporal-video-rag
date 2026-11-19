"""Procedural multi-scene video sequence generator with ground-truth temporal events."""

from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class SyntheticEvent:
    """Ground truth event label and temporal interval."""
    event_id: str
    action_name: str
    query_text: str
    start_time: float
    end_time: float
    duration: float


def generate_synthetic_video_stream(
    fps: int = 10,
    width: int = 128,
    height: int = 128,
    output_path: Optional[str] = None,
) -> Tuple[List[np.ndarray], List[SyntheticEvent]]:
    """Generate a multi-scene synthetic video sequence with known semantic events.
    
    Generates 4 distinct visual actions with clear temporal boundaries:
      1. Red expanding circle: [0.0s, 3.0s]
      2. Blue moving rectangle: [3.0s, 6.0s]
      3. Green static cross: [6.0s, 9.0s]
      4. Yellow corner diagonal motion: [9.0s, 12.0s]
      
    Returns:
        frames: List of RGB numpy arrays (H, W, 3) in uint8.
        events: List of ground-truth SyntheticEvent instances.
    """
    events = [
        SyntheticEvent(
            event_id="ev_01",
            action_name="red_expanding_circle",
            query_text="red circle expanding in the center of the screen",
            start_time=0.0,
            end_time=3.0,
            duration=3.0,
        ),
        SyntheticEvent(
            event_id="ev_02",
            action_name="blue_moving_rectangle",
            query_text="blue rectangle moving horizontally from left to right",
            start_time=3.0,
            end_time=6.0,
            duration=3.0,
        ),
        SyntheticEvent(
            event_id="ev_03",
            action_name="green_static_cross",
            query_text="green cross stationary in the center",
            start_time=6.0,
            end_time=9.0,
            duration=3.0,
        ),
        SyntheticEvent(
            event_id="ev_04",
            action_name="yellow_corner_motion",
            query_text="yellow circle moving along the diagonal",
            start_time=9.0,
            end_time=12.0,
            duration=3.0,
        ),
    ]

    total_duration = 12.0
    total_frames = int(total_duration * fps)
    frames = []

    for f in range(total_frames):
        t = f / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if 0.0 <= t < 3.0:
            # Red expanding circle
            radius = int(5 + (t / 3.0) * 35)
            cv2.circle(frame, (width // 2, height // 2), radius, (230, 30, 30), -1)
        elif 3.0 <= t < 6.0:
            # Blue moving rectangle
            rel_t = (t - 3.0) / 3.0
            cx = int(20 + rel_t * (width - 40))
            cv2.rectangle(frame, (cx - 15, height // 2 - 15), (cx + 15, height // 2 + 15), (30, 80, 230), -1)
        elif 6.0 <= t < 9.0:
            # Green cross
            cx, cy = width // 2, height // 2
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (30, 220, 40), 6)
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (30, 220, 40), 6)
        else:
            # Yellow diagonal circle
            rel_t = (t - 9.0) / 3.0
            pos = int(20 + rel_t * (width - 40))
            cv2.circle(frame, (pos, pos), 16, (240, 220, 30), -1)

        frames.append(frame)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for fr in frames:
            # RGB to BGR for OpenCV
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()

    return frames, events


def generate_synthetic_benchmark_suite(
    num_videos: int = 5,
    fps: int = 10,
) -> List[Tuple[List[np.ndarray], List[SyntheticEvent]]]:
    """Generate a batch of synthetic video streams for benchmark evaluation."""
    suite = []
    for _ in range(num_videos):
        frames, events = generate_synthetic_video_stream(fps=fps)
        suite.append((frames, events))
    return suite

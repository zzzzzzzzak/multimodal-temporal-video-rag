"""Procedural multimodal video sequence and benchmark generator."""

from src.data.synthetic_video_stream import (
    generate_synthetic_video_stream,
    generate_synthetic_benchmark_suite,
    SyntheticEvent,
)

__all__ = [
    "generate_synthetic_video_stream",
    "generate_synthetic_benchmark_suite",
    "SyntheticEvent",
]

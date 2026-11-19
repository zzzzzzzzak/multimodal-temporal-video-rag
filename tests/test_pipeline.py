"""End-to-end integration tests for VideoRAGPipeline."""

import pytest
from src.config import PipelineConfig
from src.pipeline import VideoRAGPipeline
from src.data.synthetic_video_stream import generate_synthetic_video_stream
from src.grounding.temporal_grounder import GroundedMoment


def test_pipeline_index_and_search():
    cfg = PipelineConfig()
    pipeline = VideoRAGPipeline(config=cfg, device="cpu")

    frames, events = generate_synthetic_video_stream(fps=10)
    indexed_count = pipeline.index_frames(frames, fps=10.0, video_id="test_stream_01")
    assert indexed_count > 0
    assert len(pipeline.index) == indexed_count

    # Execute search
    query = "red circle expanding in center"
    moments = pipeline.search_and_ground(query, top_k=2)

    assert len(moments) > 0
    assert isinstance(moments[0], GroundedMoment)
    assert moments[0].video_id == "test_stream_01"
    assert moments[0].calibrated_start >= 0.0
    assert moments[0].duration > 0.0
    assert moments[0].relevance_score >= 0.0

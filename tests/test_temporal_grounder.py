"""Unit tests for temporal grounding and tIoU calculation."""

import pytest
from src.grounding.temporal_grounder import (
    compute_temporal_iou,
    TemporalGrounder,
    GroundedMoment,
)


def test_compute_temporal_iou_perfect_match():
    iou = compute_temporal_iou((2.0, 5.0), (2.0, 5.0))
    assert iou == pytest.approx(1.0, abs=1e-5)


def test_compute_temporal_iou_disjoint():
    iou = compute_temporal_iou((1.0, 3.0), (5.0, 8.0))
    assert iou == pytest.approx(0.0, abs=1e-5)


def test_compute_temporal_iou_partial():
    # Int 1: [2, 6], Int 2: [4, 8]. Overlap: [4, 6] = 2. Union: [2, 8] = 6. IoU = 2/6 = 0.3333
    iou = compute_temporal_iou((2.0, 6.0), (4.0, 8.0))
    assert iou == pytest.approx(2.0 / 6.0, abs=1e-4)


def test_temporal_grounder_calibration():
    grounder = TemporalGrounder()
    moment = grounder.ground_segment(
        segment_id="seg_01",
        video_id="vid_A",
        start_time=2.0,
        end_time=6.0,
        relevance_score=0.88,
        boundary_deltas=(-0.1, 0.2),
    )

    assert isinstance(moment, GroundedMoment)
    assert moment.moment_id == "ground_seg_01"
    assert moment.calibrated_start <= moment.calibrated_end
    assert 0.0 <= moment.confidence <= 1.0

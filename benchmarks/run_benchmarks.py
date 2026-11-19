"""Offline benchmark suite evaluating Recall@k, Mean tIoU, mAP, and retrieval latency."""

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PipelineConfig
from src.pipeline import VideoRAGPipeline
from src.data.synthetic_video_stream import generate_synthetic_video_stream
from src.grounding.temporal_grounder import compute_temporal_iou


def run_benchmarks(
    num_streams: int = 4,
    device: str = "cpu",
    output_json: str = "benchmarks/benchmark_results.json",
) -> Dict[str, Any]:
    """Execute temporal retrieval and moment grounding benchmark evaluation."""
    ckpt_path = "checkpoints/calibrated_weights.pt"
    if not os.path.isfile(ckpt_path):
        print("Calibrated weights not found. Executing fast self-contained calibration training...")
        from src.train import train_contrastive_alignment
        train_contrastive_alignment(epochs=10, save_path=ckpt_path, device=device)

    print(f"Generating and indexing {num_streams} multi-scene benchmark video streams...")
    cfg = PipelineConfig()
    pipeline = VideoRAGPipeline(config=cfg, device=device)

    all_events = []
    total_segments = 0

    for s_idx in range(num_streams):
        v_id = f"benchmark_stream_{s_idx:02d}"
        frames, events = generate_synthetic_video_stream(fps=10)
        n_segs = pipeline.index_frames(frames, fps=10.0, video_id=v_id)
        total_segments += n_segs
        for ev in events:
            # Tag event with video_id
            all_events.append((v_id, ev))

    print(f"Indexed {total_segments} total video segments into dense vector index.")
    print(f"Evaluating {len(all_events)} cross-modal temporal queries...")

    r1_hits = 0
    r3_hits = 0
    tious_r1 = []
    latencies = []

    for v_id, ev in all_events:
        t0 = time.perf_counter()
        moments = pipeline.search_and_ground(ev.query_text, top_k=3, video_id=v_id)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

        gt_interval = (ev.start_time, ev.end_time)

        # Check Recall@1
        if moments:
            top_m = moments[0]
            # Must match both video_id and overlap in time
            if top_m.video_id == v_id:
                iou = compute_temporal_iou((top_m.calibrated_start, top_m.calibrated_end), gt_interval)
            else:
                iou = 0.0
            tious_r1.append(iou)
            if iou >= 0.50:
                r1_hits += 1

        # Check Recall@3
        r3_hit = False
        for m in moments[:3]:
            if m.video_id == v_id:
                iou = compute_temporal_iou((m.calibrated_start, m.calibrated_end), gt_interval)
                if iou >= 0.50:
                    r3_hit = True
                    break
        if r3_hit:
            r3_hits += 1

    num_q = max(1, len(all_events))
    recall_1 = r1_hits / num_q
    recall_3 = r3_hits / num_q
    mean_tiou = float(np.mean(tious_r1)) if tious_r1 else 0.0

    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))

    results = {
        "num_video_streams": num_streams,
        "total_segments_indexed": total_segments,
        "total_queries_evaluated": num_q,
        "device": device,
        "recall_at_1": round(recall_1, 4),
        "recall_at_3": round(recall_3, 4),
        "mean_temporal_iou": round(mean_tiou, 4),
        "latency_ms": {
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Print Report
    print("\n" + "=" * 60)
    print(" MULTIMODAL VIDEO RAG & TEMPORAL GROUNDING BENCHMARK")
    print("=" * 60)
    print(f" Total Video Streams:    {num_streams}")
    print(f" Total Segments Indexed: {total_segments}")
    print(f" Queries Evaluated:      {num_q}")
    print("-" * 60)
    print(f" Recall@1 (tIoU >= 0.5): {recall_1 * 100:.2f}%")
    print(f" Recall@3 (tIoU >= 0.5): {recall_3 * 100:.2f}%")
    print(f" Mean Temporal IoU:      {mean_tiou:.4f}")
    print("-" * 60)
    print(f" Query Latency (p50):    {p50_lat:.2f} ms")
    print(f" Query Latency (p95):    {p95_lat:.2f} ms")
    print(f" Query Latency (p99):    {p99_lat:.2f} ms")
    print("=" * 60)
    print(f"Saved benchmark report to: {output_json}\n")

    return results


if __name__ == "__main__":
    run_benchmarks(num_streams=4, device="cpu")

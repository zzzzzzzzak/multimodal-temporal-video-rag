"""End-to-end Multimodal Video RAG & Temporal Moment Grounding Pipeline."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.config import PipelineConfig
from src.indexing.vector_index import DenseVectorIndex, SearchResult
from src.indexing.video_segmenter import VideoSegmenter, VideoSegment
from src.models.vision_encoder import VisionClipEncoder
from src.models.text_encoder import TextQueryEncoder
from src.models.cross_encoder import TemporalCrossModalReranker
from src.grounding.temporal_grounder import TemporalGrounder, GroundedMoment


class VideoRAGPipeline:
    """Orchestrates video segmentation, multimodal encoding, vector search, and temporal grounding."""
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        checkpoint_dir: Optional[str] = None,
        device: str = "auto",
    ):
        self.config = config or PipelineConfig()
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Components
        self.segmenter = VideoSegmenter(self.config.segmentation)
        self.vision_encoder = VisionClipEncoder(self.config.model).to(self.device).eval()
        self.text_encoder = TextQueryEncoder(self.config.model).to(self.device).eval()
        self.reranker = TemporalCrossModalReranker(self.config.model).to(self.device).eval()
        self.grounder = TemporalGrounder(self.config.grounding)
        self.index = DenseVectorIndex(dimension=self.config.model.embedding_dim)

        default_ckpt = "checkpoints/calibrated_weights.pt"
        target_ckpt = checkpoint_dir if (checkpoint_dir and os.path.isfile(checkpoint_dir)) else (default_ckpt if os.path.isfile(default_ckpt) else None)
        if target_ckpt:
            ckpt = torch.load(target_ckpt, map_location=self.device, weights_only=True)
            if "vision_encoder" in ckpt:
                self.vision_encoder.load_state_dict(ckpt["vision_encoder"])
            if "text_encoder" in ckpt:
                self.text_encoder.load_state_dict(ckpt["text_encoder"])

        self.segments_cache: Dict[str, VideoSegment] = {}

    def _prepare_keyframe_tensor(self, keyframes: List[np.ndarray]) -> torch.Tensor:
        """Convert list of RGB uint8 keyframes into normalized float32 tensor (1, K, 3, H, W)."""
        target_size = self.config.segmentation.downsample_size
        k_tensors = []
        for kf in keyframes:
            # Resize
            import cv2
            resized = cv2.resize(kf, (target_size[1], target_size[0]))
            t = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
            k_tensors.append(t)

        stacked = torch.stack(k_tensors, dim=0).unsqueeze(0)  # (1, K, 3, H, W)
        return stacked.to(self.device)

    def index_frames(
        self,
        frames: List[np.ndarray],
        fps: float = 10.0,
        video_id: str = "video_0",
        keyframes_per_segment: int = 4,
    ) -> int:
        """Segment and index an in-memory sequence of video frames.
        
        Returns:
            Number of newly indexed segments.
        """
        segments = self.segmenter.segment_frames(
            frames, fps=fps, video_id=video_id, keyframes_per_segment=keyframes_per_segment
        )
        if not segments:
            return 0

        vectors = []
        metadatas = []

        with torch.no_grad():
            for seg in segments:
                self.segments_cache[seg.segment_id] = seg
                k_tensor = self._prepare_keyframe_tensor(seg.keyframes)
                clip_emb, frame_embs = self.vision_encoder(k_tensor)
                
                vec = clip_emb[0].cpu().numpy()
                vectors.append(vec)
                
                meta = {
                    "segment_id": seg.segment_id,
                    "video_id": seg.video_id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "frame_indices": seg.frame_indices,
                }
                metadatas.append(meta)

        self.index.add(np.array(vectors), metadatas)
        return len(segments)

    def index_video_file(
        self,
        video_path: str,
        video_id: Optional[str] = None,
        sample_stride: int = 2,
    ) -> int:
        """Segment and index a video file from disk."""
        segments = self.segmenter.segment_video_file(
            video_path, video_id=video_id, sample_stride=sample_stride
        )
        if not segments:
            return 0

        vectors = []
        metadatas = []

        with torch.no_grad():
            for seg in segments:
                self.segments_cache[seg.segment_id] = seg
                k_tensor = self._prepare_keyframe_tensor(seg.keyframes)
                clip_emb, _ = self.vision_encoder(k_tensor)
                
                vec = clip_emb[0].cpu().numpy()
                vectors.append(vec)
                
                meta = {
                    "segment_id": seg.segment_id,
                    "video_id": seg.video_id,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "duration": seg.duration,
                    "frame_indices": seg.frame_indices,
                }
                metadatas.append(meta)

        self.index.add(np.array(vectors), metadatas)
        return len(segments)

    def search_and_ground(
        self,
        query: str,
        top_k: Optional[int] = None,
        video_id: Optional[str] = None,
    ) -> List[GroundedMoment]:
        """Query the video RAG index and ground the most relevant temporal moments."""
        k = top_k or self.config.retrieval.top_k
        if len(self.index) == 0:
            return []

        # 1. Encode query
        with torch.no_grad():
            query_pooled, token_embs = self.text_encoder.encode_text([query], device=self.device)
            q_vec = query_pooled[0].cpu().numpy()

        # 2. First-stage Dense Retrieval with optional video scoping
        filter_fn = (lambda m: m.get("video_id") == video_id) if video_id else None
        candidates = self.index.search(q_vec, top_k=k, filter_fn=filter_fn)
        if not candidates:
            return []

        # 3. Second-stage Cross-Modal Re-ranking & Boundary Calibration
        grounded_moments = []

        for cand in candidates:
            meta = cand.metadata
            seg_id = meta["segment_id"]
            seg = self.segments_cache.get(seg_id)

            if seg is not None and self.config.retrieval.enable_reranking:
                with torch.no_grad():
                    k_tensor = self._prepare_keyframe_tensor(seg.keyframes)
                    _, frame_embs = self.vision_encoder(k_tensor)
                    
                    rerank_out = self.reranker(
                        query_token_embs=token_embs,
                        frame_embs=frame_embs,
                        query_pooled=query_pooled,
                    )
                    
                    fine_score = float(rerank_out["alignment_score"][0, 0].cpu().item())
                    deltas = rerank_out["boundary_deltas"][0].cpu().tolist()
                    attn_map = rerank_out["attention_weights"][0].cpu().numpy()
            else:
                fine_score = cand.score
                deltas = (0.0, 0.0)
                attn_map = None

            # 4. Temporal Grounding
            moment = self.grounder.ground_segment(
                segment_id=seg_id,
                video_id=meta["video_id"],
                start_time=meta["start_time"],
                end_time=meta["end_time"],
                relevance_score=fine_score,
                boundary_deltas=(deltas[0], deltas[1]),
                keyframe_indices=meta.get("frame_indices", []),
                metadata=meta,
                attention_map=attn_map,
            )
            grounded_moments.append(moment)

        # Sort by relevance score descending
        grounded_moments.sort(key=lambda m: m.relevance_score, reverse=True)
        return grounded_moments

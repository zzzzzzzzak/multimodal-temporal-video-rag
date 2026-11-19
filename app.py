"""Streamlit Web UI for Multimodal Video RAG & Moment Grounding."""

import io
import json
import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

from src.config import PipelineConfig
from src.pipeline import VideoRAGPipeline
from src.data.synthetic_video_stream import generate_synthetic_video_stream

st.set_page_config(
    page_title="Multimodal Video RAG & Temporal Grounding",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .ground-badge {
        background-color: #dbeafe;
        color: #1e40af;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 1.1rem;
        display: inline-block;
    }
    .score-badge {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 1.1rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    cfg = PipelineConfig()
    pipeline = VideoRAGPipeline(config=cfg, device="cpu")
    # Pre-index a synthetic stream for instantaneous demo
    frames, events = generate_synthetic_video_stream(fps=10)
    pipeline.index_frames(frames, fps=10.0, video_id="demo_stream_01")
    return pipeline, frames, events


def main():
    st.title("🎬 Multimodal Video RAG & Temporal Moment Grounding")
    st.markdown(
        "**Semantic Video Search and Timestamp Grounding** using dense dual-encoder embeddings, "
        "vector indexing, and cross-attention temporal re-ranking."
    )

    pipeline, demo_frames, demo_events = get_pipeline()

    # Sidebar Controls
    st.sidebar.header("Configuration & Stream Source")
    stream_choice = st.sidebar.radio(
        "Video Source",
        ["Curated Multi-Event Video (Pre-indexed)", "Upload Custom Video (MP4 / AVI)"]
    )

    top_k = st.sidebar.slider("Top Candidates (k)", 1, 5, 3)

    if stream_choice == "Upload Custom Video (MP4 / AVI)":
        uploaded = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded:
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(uploaded.name)[1], delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            with st.spinner("Segmenting and indexing custom video stream..."):
                num_segs = pipeline.index_video_file(tmp_path, video_id=uploaded.name)
            st.sidebar.success(f"Successfully indexed {num_segs} video segments!")

    # Search Bar & Query Input
    st.subheader("🔍 Natural Language Moment Retrieval")
    preset_query = st.selectbox(
        "Example Semantic Queries:",
        [
            "red circle expanding in the center of the screen",
            "blue rectangle moving horizontally from left to right",
            "green cross stationary in the center",
            "yellow circle moving along the diagonal",
        ],
    )
    custom_query = st.text_input("Or enter custom query:", value=preset_query)

    if st.button("Search Video Stream", type="primary") or custom_query:
        with st.spinner("Searching dense vector index and computing cross-modal attention..."):
            moments = pipeline.search_and_ground(custom_query, top_k=top_k)

        if not moments:
            st.warning("No matching temporal moments found for this query.")
            return

        st.markdown(f"### Found {len(moments)} Grounded Video Moments")

        for idx, m in enumerate(moments):
            with st.container():
                st.markdown(f"#### Match #{idx+1}: `{m.video_id}`")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="ground-badge">⏱️ [{m.calibrated_start:.2f}s - {m.calibrated_end:.2f}s]</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="score-badge">Relevance: {m.relevance_score * 100:.1f}%</div>', unsafe_allow_html=True)
                with c3:
                    st.metric("Duration", f"{m.duration:.2f}s")
                with c4:
                    st.metric("Calibrated Confidence", f"{m.confidence * 100:.1f}%")

                # Show Keyframes
                seg = pipeline.segments_cache.get(m.metadata.get("segment_id"))
                if seg and seg.keyframes:
                    st.markdown("**Representative Keyframes:**")
                    kf_cols = st.columns(min(4, len(seg.keyframes)))
                    for k_idx, kf in enumerate(seg.keyframes[:4]):
                        with kf_cols[k_idx]:
                            st.image(kf, caption=f"Frame #{seg.frame_indices[k_idx]}", use_container_width=True)

                # Show Cross-Attention Heatmap if available
                if m.attention_map is not None:
                    with st.expander("🔬 View Token-to-Keyframe Cross-Attention Heatmap"):
                        fig, ax = plt.subplots(figsize=(6, 3))
                        cax = ax.matshow(m.attention_map, cmap="viridis")
                        fig.colorbar(cax)
                        ax.set_ylabel("Query Token Index")
                        ax.set_xlabel("Video Keyframe Index")
                        ax.set_title("Cross-Modal Attention Matrix")
                        st.pyplot(fig)

                st.markdown("---")

        # Download Report
        st.download_button(
            "📥 Download Grounding JSON Report",
            data=json.dumps([m.__dict__ for m in moments], default=str, indent=2),
            file_name="grounded_moments.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()

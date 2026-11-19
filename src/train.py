"""Contrastive vision-language calibration training pipeline."""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ModelConfig
from src.models.vision_encoder import VisionClipEncoder
from src.models.text_encoder import TextQueryEncoder
from src.data.synthetic_video_stream import generate_synthetic_video_stream


class ContrastiveModel(nn.Module):
    """Dual-encoder model wrapped for InfoNCE contrastive training."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.vision_encoder = VisionClipEncoder(config)
        self.text_encoder = TextQueryEncoder(config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self, keyframes: torch.Tensor, queries: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clip_embs, _ = self.vision_encoder(keyframes)
        text_embs, _ = self.text_encoder.encode_text(queries, device=keyframes.device)
        return clip_embs, text_embs


def info_nce_loss(
    vision_embs: torch.Tensor, text_embs: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    """Symmetric InfoNCE contrastive cross-entropy loss."""
    scale = logit_scale.exp().clamp(max=100.0)
    logits_per_image = scale * (vision_embs @ text_embs.T)
    logits_per_text = logits_per_image.T
    
    b = vision_embs.shape[0]
    labels = torch.arange(b, device=vision_embs.device, dtype=torch.long)
    
    loss_v = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_v + loss_t) / 2.0


def train_contrastive_alignment(
    epochs: int = 15,
    lr: float = 2e-3,
    save_path: str = "checkpoints/calibrated_weights.pt",
    device: str = "auto",
) -> Tuple[ContrastiveModel, Dict[str, float]]:
    """Train dual encoders on multi-scene action streams to align vision and text spaces."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(f"Generating calibration data streams on {dev}...")
    cfg = ModelConfig()
    model = ContrastiveModel(cfg).to(dev)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Collect training action pairs
    raw_pairs = []
    for _ in range(8):
        frames, events = generate_synthetic_video_stream(fps=10)
        for ev in events:
            # Extract 4 keyframes for this event interval
            start_f = int(ev.start_time * 10)
            end_f = int(ev.end_time * 10)
            sub_frames = frames[start_f:end_f]
            indices = np.linspace(0, len(sub_frames) - 1, 4, dtype=int)
            kfs = [cv2.resize(sub_frames[k], (128, 128)) for k in indices]
            # Convert to (4, 3, 128, 128) float
            kf_t = torch.stack([
                torch.from_numpy(kf.transpose(2, 0, 1)).float() / 255.0 for kf in kfs
            ])
            raw_pairs.append((kf_t, ev.query_text))

    print(f"Starting contrastive alignment training across {len(raw_pairs)} action samples for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        # Batch by 4 unique actions
        np.random.shuffle(raw_pairs)
        batch_size = 4
        num_batches = len(raw_pairs) // batch_size

        for b in range(num_batches):
            batch = raw_pairs[b * batch_size:(b + 1) * batch_size]
            k_tensor = torch.stack([item[0] for item in batch]).to(dev)  # (B, 4, 3, 128, 128)
            queries = [item[1] for item in batch]

            optimizer.zero_grad()
            v_embs, t_embs = model(k_tensor, queries)
            loss = info_nce_loss(v_embs, t_embs, model.logit_scale)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, num_batches)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | InfoNCE Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        "vision_encoder": model.vision_encoder.state_dict(),
        "text_encoder": model.text_encoder.state_dict(),
    }, save_path)
    print(f"Contrastive alignment complete! Checkpoint saved to: {save_path}")

    return model, {"final_loss": avg_loss}


if __name__ == "__main__":
    train_contrastive_alignment(epochs=25, device="auto")

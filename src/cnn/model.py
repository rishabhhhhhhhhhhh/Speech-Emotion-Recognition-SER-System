"""Acoustic Primer CNN - Paper A's 4-conv mel architecture, modernized.

Paper A (arXiv:2503.19677): "4 2D convolutional layers with max pooling, batch
normalization, dropout regularization, and ELU activation", Adam + categorical CE.

Deviations from Paper A, and why:
  0. FREQUENCY IS FLATTENED INTO THE FEATURE AXIS, not mean-pooled. v1 averaged the
     8 surviving mel bands into one scalar per channel; that discards formant and pitch
     structure and the model underfit hard (train WA 40.2%). Measured directly.
  1. TEMPORAL ATTENTION POOLING instead of flatten. Emotion is not uniformly
     distributed across an utterance, and 94% of our clips are zero-padded to 4.0s -
     attention lets the net ignore the padding. Flatten would also blow the FC layer up.
  2. GENDER AS AN AUXILIARY HEAD, not folded into the label space. Paper A predicts
     gender x emotion jointly (male_angry, female_sad), which doubles the class count
     and dilutes every class. We keep 7 emotion classes and pass gender to the LALM
     separately as a prosody-normalisation hint.
  3. Class-balanced focal loss (see losses.py) instead of plain CE.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class AttentivePool(nn.Module):
    """Softmax-weighted mean over the time axis. Input (B, C, T) -> (B, C).

    `hidden` is a fixed small bottleneck, not channels//4 - with C=2048 the latter
    would spend ~1M params on scoring alone.
    """
    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, hidden, 1), nn.Tanh(),
            nn.Conv1d(hidden, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x), dim=-1)       # (B, 1, T)
        return (x * w).sum(dim=-1)                     # (B, C)


def _block(cin: int, cout: int, drop: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ELU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout2d(drop))


class AcousticPrimer(nn.Module):
    """(B, 1, 128, 250) log-mel -> 7 emotion logits + 2 gender logits."""

    def __init__(self, n_classes: int = 7, n_gender: int = 2, width: int = 32,
                 n_mels: int = 128):
        super().__init__()
        w = width
        # Dropout2d reduced 0.20/0.25/0.30/0.30 -> 0.10/0.10/0.15/0.15 and FC 0.5 -> 0.3.
        # v1 underfit badly (train WA 40.2%, loss flat at 0.77): it was over-regularised
        # for a 0.4M-param model, not overfitting.
        self.features = nn.Sequential(
            _block(1,      w,     0.10),   # -> (w,   64, 125)
            _block(w,      w * 2, 0.10),   # -> (2w,  32,  62)
            _block(w * 2,  w * 4, 0.15),   # -> (4w,  16,  31)
            _block(w * 4,  w * 8, 0.15),   # -> (8w,   8,  15)
        )
        feat_dim = w * 8 * (n_mels // 16)          # channels x surviving mel bands
        self.pool = AttentivePool(feat_dim)
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ELU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ELU(inplace=True), nn.Dropout(0.3))
        self.emotion = nn.Linear(128, n_classes)
        self.gender = nn.Linear(128, n_gender)

    def forward(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x.unsqueeze(1)                     # (B, 128, 250) -> (B, 1, 128, 250)
        h = self.features(x)                       # (B, 8w, 8, 15)
        B, C, F, T = h.shape
        h = h.reshape(B, C * F, T)                 # KEEP freq -> (B, 8w*8, 15)
        h = self.pool(h)                           # (B, 8w*8)
        h = self.trunk(h)
        return self.emotion(h), self.gender(h)

    @torch.no_grad()
    def probs(self, x: torch.Tensor, temperature: float = 1.0):
        """Calibrated probabilities. temperature comes from artifacts/cnn/temperature.json."""
        e, g = self(x)
        return torch.softmax(e / temperature, dim=-1), torch.softmax(g, dim=-1)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

"""Class-balanced focal loss.

MELD is ~48% neutral; our acted pool has 5:1 imbalance on `surprise` (only RAVDESS
and SAVEE carry that class - CREMA-D has none). Plain cross-entropy collapses to the
majority class and destroys UA (macro recall), which is the metric that matters.

Class-balanced weighting: Cui et al. 2019, "effective number of samples"
    w_c  = (1 - beta) / (1 - beta^{n_c}),   normalised to mean 1
Focal term: Lin et al. 2017
    FL   = -w_c * (1 - p_t)^gamma * log(p_t)
"""
from __future__ import annotations
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(counts, beta: float = 0.999) -> torch.Tensor:
    counts = np.asarray(counts, dtype=np.float64)
    eff = 1.0 - np.power(beta, np.maximum(counts, 1))
    w = (1.0 - beta) / eff
    w = w / w.mean()                       # mean 1 keeps the loss scale comparable
    return torch.tensor(w, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts, beta: float = 0.999, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", effective_number_weights(class_counts, beta))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        focal = (1.0 - p_t).pow(self.gamma)
        w = self.weight.to(logits.device)[target]
        loss = -w * focal * logp_t
        if self.label_smoothing > 0:
            smooth = -(logp.mean(dim=-1)) * w
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * smooth
        return loss.mean()

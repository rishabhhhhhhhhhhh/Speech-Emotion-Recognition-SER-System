"""Temperature scaling + Expected Calibration Error.

Why this matters here, more than in a normal classifier
-------------------------------------------------------
The CNN's softmax is rendered verbatim into the LALM prompt as evidence
("anger 0.41 | disgust 0.22 | ..."), and the confidence-gated cascade thresholds on
max-prob. A small CNN trained on ~10k clips is badly overconfident, so uncalibrated
numbers would make the LALM trust a coin flip. One scalar T, fit by NLL on a held-out
split (Guo et al. 2017), fixes most of it without touching accuracy at all
(temperature is monotonic, so argmax - and therefore UA/WA/F1 - is unchanged).
"""
from __future__ import annotations
import json, pathlib
import numpy as np, torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]


def fit_temperature(logits: torch.Tensor, labels: np.ndarray,
                    max_iter: int = 200) -> float:
    """Fit a single scalar T minimising NLL of softmax(logits / T)."""
    logits = logits.detach().float()
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)          # optimise log T > 0
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error over `n_bins` equal-width confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            rows.append((f"{lo:.1f}-{hi:.1f}", int(m.sum()),
                         round(conf[m].mean(), 3), round(correct[m].mean(), 3)))
    return rows


def save(temperature: float, meta: dict, path: pathlib.Path | None = None):
    path = path or ROOT / "artifacts/cnn/temperature.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"temperature": temperature, **meta}, open(path, "w"), indent=2)
    return path


def load(path: pathlib.Path | None = None) -> float:
    path = path or ROOT / "artifacts/cnn/temperature.json"
    return json.load(open(path))["temperature"]

"""Fit the ensemble's temperature on a held-out split and report calibration.

    .venv/bin/python src/cnn/fit_calibration.py \
        --ckpts artifacts/cnn/meld_seed0.pt,artifacts/cnn/meld_seed1.pt,artifacts/cnn/meld_seed2.pt \
        --split meld_dev

Must be fit on a split the CNN did NOT train on, and NOT on the test set - fitting T on test
would leak. Temperature is monotonic, so UA/WA/F1 are unchanged by construction; this only
fixes how much the probabilities can be believed, which is exactly what the LALM prompt and
the cascade threshold depend on.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cnn.model import AcousticPrimer
from cnn.dataset import MelDataset
from cnn.calibrate import fit_temperature, ece, reliability_table, save
from data.labels import NUM_CLASSES

ROOT = pathlib.Path(__file__).resolve().parents[2]


@torch.no_grad()
def ensemble_probs(ckpts, split, device):
    """Mean of per-model softmax. Averaging probabilities (not logits) is the correct
    ensemble for calibration - averaging logits lets one overconfident member dominate."""
    ds = MelDataset(split, train=False)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    acc, n = None, 0
    for ck in ckpts:
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        m = AcousticPrimer(NUM_CLASSES).to(device); m.load_state_dict(sd["model"]); m.eval()
        chunks = [torch.softmax(m(mel.to(device))[0].float(), -1).cpu() for mel, _, _ in dl]
        p = torch.cat(chunks)
        acc = p if acc is None else acc + p
        n += 1
        print(f"    + {pathlib.Path(ck).name}")
    return acc / n, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--split", default="meld_dev")
    ap.add_argument("--out", default="artifacts/cnn/temperature.json")
    a = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpts = [ROOT / c.strip() for c in a.ckpts.split(",")]
    print(f"ensembling {len(ckpts)} checkpoint(s) on {a.split}:")
    probs, ds = ensemble_probs(ckpts, a.split, device)
    y = ds.labels

    p0 = probs.numpy()
    logits = torch.log(probs.clamp_min(1e-12))          # recover logits for T-scaling
    T = fit_temperature(logits, y)
    p1 = torch.softmax(logits / T, dim=-1).numpy()

    e0, e1 = ece(p0, y), ece(p1, y)
    wa = accuracy_score(y, p1.argmax(1)) * 100
    ua = recall_score(y, p1.argmax(1), average="macro", zero_division=0) * 100

    print(f"\n  fitted temperature T = {T:.4f}")
    print(f"  ECE (15-bin)  before {e0:.4f}  ->  after {e1:.4f}")
    print(f"  mean confidence before {p0.max(1).mean():.4f} -> after {p1.max(1).mean():.4f}")
    print(f"  accuracy unchanged (temperature is monotonic): "
          f"{(p0.argmax(1) == p1.argmax(1)).all()}")
    print(f"  ensemble on {a.split}: UA {ua:.2f}  WA {wa:.2f}")
    print(f"\n  reliability after calibration (bin, n, mean conf, accuracy):")
    for row in reliability_table(p1, y):
        print(f"    {row[0]:>10s}  n={row[1]:5d}  conf={row[2]:.3f}  acc={row[3]:.3f}")

    gate = "PASS" if e1 < 0.10 else "FAIL"
    print(f"\n  Phase 1 calibration gate (ECE < 0.10): {gate}  ({e1:.4f})")
    path = save(T, {"split": a.split, "ece_before": e0, "ece_after": e1,
                    "n": int(len(y)), "ckpts": [str(c) for c in ckpts],
                    "ua": ua, "wa": wa}, ROOT / a.out)
    print(f"  -> {path}")


if __name__ == "__main__":
    main()

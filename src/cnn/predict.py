"""Run the calibrated CNN ensemble over a split -> priors parquet for the LALM stage.

    .venv/bin/python src/cnn/predict.py --ckpts artifacts/cnn/meld_seed0.pt,... \
        --splits meld_train,meld_dev,meld_test --temperature artifacts/cnn/temperature.json

Output columns: uid, p_<class> x7, pred, conf, top3_str, p_male, p_female, gender_pred.
`top3_str` is what gets rendered verbatim into the PC-ESR prompt, so it is produced here
once and never re-derived - the prompt builder must not recompute probabilities.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cnn.model import AcousticPrimer
from cnn.dataset import MelDataset
from data.labels import CANONICAL, NUM_CLASSES

ROOT = pathlib.Path(__file__).resolve().parents[2]


@torch.no_grad()
def ensemble_logits(ckpts, split, device, batch=256):
    """Mean of per-model SOFTMAX (not logits) - averaging probabilities is the correct
    ensemble for calibration; averaging logits lets one overconfident member dominate."""
    ds = MelDataset(split, train=False)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    acc_e, acc_g, n = None, None, 0
    for ck in ckpts:
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        m = AcousticPrimer(NUM_CLASSES).to(device)
        m.load_state_dict(sd["model"]); m.eval()
        es, gs = [], []
        for mel, _y, _g in dl:
            e, g = m(mel.to(device))
            es.append(torch.softmax(e.float(), -1).cpu())
            gs.append(torch.softmax(g.float(), -1).cpu())
        e = torch.cat(es); g = torch.cat(gs)
        acc_e = e if acc_e is None else acc_e + e
        acc_g = g if acc_g is None else acc_g + g
        n += 1
    return acc_e / n, acc_g / n, ds


def render_top3(p: np.ndarray, k: int = 3) -> str:
    idx = np.argsort(-p)[:k]
    return " | ".join(f"{CANONICAL[i]} {p[i]:.2f}" for i in idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-separated checkpoint paths")
    ap.add_argument("--splits", required=True, help="comma-separated split names")
    ap.add_argument("--temperature", default="artifacts/cnn/temperature.json")
    ap.add_argument("--out", default="artifacts/priors")
    a = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpts = [ROOT / c.strip() for c in a.ckpts.split(",")]
    T = 1.0
    tp = ROOT / a.temperature
    if tp.exists():
        T = json.load(open(tp))["temperature"]
        print(f"temperature T = {T:.3f}")
    else:
        print("!! no temperature.json - emitting UNCALIBRATED probabilities")

    outdir = ROOT / a.out; outdir.mkdir(parents=True, exist_ok=True)
    for split in a.splits.split(","):
        split = split.strip()
        probs, gprobs, ds = ensemble_logits(ckpts, split, device)
        # temperature is applied in LOGIT space; recover logits from the mean probs
        logits = torch.log(probs.clamp_min(1e-12))
        p = torch.softmax(logits / T, dim=-1).numpy()
        g = gprobs.numpy()

        df = pd.DataFrame({"uid": ds.idx.uid})
        for i, c in enumerate(CANONICAL):
            df[f"p_{c}"] = p[:, i]
        df["pred"] = [CANONICAL[i] for i in p.argmax(1)]
        df["conf"] = p.max(1)
        df["top3_str"] = [render_top3(row) for row in p]
        df["p_male"], df["p_female"] = g[:, 0], g[:, 1]
        df["gender_pred"] = np.where(g[:, 0] >= g[:, 1], "male", "female")
        out = outdir / f"{split}_priors.parquet"
        df.to_parquet(out, index=False)
        acc = (df.pred.values == ds.idx.label.values).mean() * 100
        print(f"  {split}: n={len(df)} -> {out.name}  ensemble acc {acc:.2f}%  "
              f"mean conf {df.conf.mean():.3f}")
        print(f"    example top3: {df.top3_str.iloc[0]}")


if __name__ == "__main__":
    main()

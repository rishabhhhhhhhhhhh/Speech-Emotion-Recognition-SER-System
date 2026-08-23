"""Materialise the LALM prompt datasets (jsonl) from manifests + cached CNN priors.

    .venv/bin/python src/prompts/build_dataset.py --split meld_train --mode train
    .venv/bin/python src/prompts/build_dataset.py --split meld_test  --mode eval

TRAIN MIX (the core idea of this project)
-----------------------------------------
    65%  true calibrated prior
    20%  NO prior          (base ESR schema - prior paragraph and <prior_check> both dropped)
    15%  CORRUPTED prior   (a wrong class shown with plausible confidence)

The ground-truth label is the TRUE label in all three cases. So GRPO learns three things at
once:
  * use the prior when it coheres with what you hear and read   -> the prior gains real weight
  * survive without it                                          -> no collapse into sycophancy
  * OVERRIDE it when it conflicts with strong evidence          -> the behaviour that lets the
                                                                   hybrid beat both components
Costs zero extra compute - it is entirely prompt construction.

EVAL MIX
--------
Three parallel copies of every dev/test item: `true` / `none` / `shuffled`. One trained
model, evaluated three ways. If accuracy tracks the SHUFFLED prior, the model is parroting
the CNN and the reasoning is decoration - that is the anti-sycophancy proof, and it costs
one extra eval pass instead of a second training run.

NOTE the deliberate difference between the two perturbations:
  corrupted (train) - teaches the policy to override a wrong prior
  shuffled  (eval)  - tests whether it blindly follows one
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data.labels import CANONICAL
from prompts.templates import build_prompt

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/prompts"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_MIX = {"true": 0.65, "none": 0.20, "corrupt": 0.15}
PCOLS = [f"p_{c}" for c in CANONICAL]


def render_top3(p: np.ndarray, k: int = 3) -> str:
    idx = np.argsort(-p)[:k]
    return " | ".join(f"{CANONICAL[i]} {p[i]:.2f}" for i in idx)


def corrupt_prior(p: np.ndarray, true_label: str, rng: np.random.Generator) -> np.ndarray:
    """Promote a WRONG class to the top, keeping the distribution shape plausible.

    The wrong class is drawn from the CNN's own tail (weighted by its remaining mass), so
    the corruption looks like a realistic classifier mistake rather than random noise -
    otherwise the model could learn to spot corruption from its shape instead of from the
    evidence, which would defeat the purpose.
    """
    ti = CANONICAL.index(true_label)
    w = p.copy()
    w[ti] = 0.0
    if w.sum() <= 0:
        w = np.ones(len(CANONICAL)); w[ti] = 0.0
    w = w / w.sum()
    wrong = int(rng.choice(len(CANONICAL), p=w))
    q = p.copy()
    q[wrong], q[int(np.argmax(p))] = p[int(np.argmax(p))], p[wrong]   # swap top with wrong
    return q / q.sum()


def shuffle_prior(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute the probability vector: realistic numbers, wrong class assignment."""
    return p[rng.permutation(len(p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="e.g. meld_train / meld_test")
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--priors", default=None, help="default artifacts/priors/<split>_priors.parquet")
    ap.add_argument("--prior-ua", type=float, required=True, help="CNN standalone UA on this corpus")
    ap.add_argument("--prior-wa", type=float, required=True, help="CNN standalone WA on this corpus")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--subset", type=int, default=0, help="stratified subset size (0 = all)")
    a = ap.parse_args()

    man = pd.read_csv(ROOT / f"data/manifests/{a.split}.csv")
    pri = pd.read_parquet(ROOT / (a.priors or f"artifacts/priors/{a.split}_priors.parquet"))
    df = man.merge(pri, on="uid", how="inner", validate="one_to_one")
    if len(df) != len(man):
        print(f"  !! {len(man)-len(df)} manifest rows had no prior and were dropped")

    if a.subset and a.subset < len(df):
        df = (df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(max(1, round(a.subset * len(g) / len(df))),
                                          random_state=a.seed))
                .reset_index(drop=True))
        print(f"  stratified subset -> n={len(df)}")

    rng = np.random.default_rng(a.seed)
    P = df[PCOLS].to_numpy(dtype=np.float64)

    if a.mode == "train":
        conds = rng.choice(list(TRAIN_MIX), size=len(df), p=list(TRAIN_MIX.values()))
        variants = [(conds, "train")]
    else:
        variants = [(np.full(len(df), c), c) for c in ("true", "none", "shuffled")]

    for cond_arr, tag in variants:
        path = OUT / f"{a.split}_{tag}.jsonl"
        counts = {}
        with open(path, "w") as fh:
            for i, row in enumerate(df.itertuples(index=False)):
                cond = cond_arr[i]
                counts[cond] = counts.get(cond, 0) + 1
                p = P[i]
                if cond == "none":
                    mode, top3 = "ESR", None
                else:
                    mode = "PC_ESR"
                    if cond == "corrupt":
                        p = corrupt_prior(p, row.label, rng)
                    elif cond == "shuffled":
                        p = shuffle_prior(p, rng)
                    top3 = render_top3(p)
                prompt = build_prompt(
                    mode, transcript=row.transcript, prior_top3=top3,
                    gender=getattr(row, "gender_pred", "unknown"),
                    prior_ua=a.prior_ua, prior_wa=a.prior_wa)
                fh.write(json.dumps({
                    "uid": row.uid, "audio": row.wav_path, "prompt": prompt,
                    "schema": mode, "prior_condition": cond,
                    "label": row.label, "label_id": int(row.label_id),
                    "cnn_pred": row.pred, "cnn_conf": float(row.conf),
                }) + "\n")
        print(f"  wrote {path.name:34s} n={len(df):5d}  {counts}")


if __name__ == "__main__":
    main()

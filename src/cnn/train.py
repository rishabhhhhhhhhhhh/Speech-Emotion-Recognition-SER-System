"""Train the Acoustic Primer CNN.

Stage A (pretrain on acted pool):
    .venv/bin/python src/cnn/train.py --train acted_train --val acted_val \
        --epochs 60 --lr 3e-4 --tag pretrain --seed 0

Stage B (fine-tune on MELD) adds --init artifacts/cnn/pretrain_seed0.pt --lr 1e-4
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cnn.model import AcousticPrimer, count_params
from cnn.dataset import MelDataset
from cnn.losses import ClassBalancedFocalLoss
from data.labels import CANONICAL, NUM_CLASSES

ROOT = pathlib.Path(__file__).resolve().parents[2]
CKPT = ROOT / "artifacts/cnn"; CKPT.mkdir(parents=True, exist_ok=True)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, logits_all = [], [], []
    for mel, y, _g in loader:
        e, _ = model(mel.to(device))
        logits_all.append(e.float().cpu())
        ps.append(e.argmax(-1).cpu()); ys.append(y)
    y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy()
    return dict(
        UA=recall_score(y, p, average="macro", zero_division=0) * 100,
        WA=accuracy_score(y, p) * 100,
        F1=f1_score(y, p, average="macro", zero_division=0) * 100,
    ), torch.cat(logits_all), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="acted_train")
    ap.add_argument("--val", default="acted_val")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--gender-weight", type=float, default=0.2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--cb-beta", type=float, default=0.99)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pretrain")
    ap.add_argument("--init", default=None, help="checkpoint to warm-start from")
    ap.add_argument("--freq-mask", type=int, default=8, help="SpecAugment freq mask width")
    ap.add_argument("--time-mask", type=int, default=15, help="SpecAugment time mask width")
    ap.add_argument("--resume", default=None, help="resume state file (last_*.pt)")
    ap.add_argument("--balanced-sampler", action="store_true",
                    help="draw class-rebalanced batches (WeightedRandomSampler). For severe "
                         "imbalance this is stronger than loss reweighting alone: it changes "
                         "what the model SEES, not just how errors are scored. MELD is 17.6:1.")
    ap.add_argument("--sampler-power", type=float, default=0.5,
                    help="sample weight ~ (1/n_c)^p.  p=1 full balance, p=0.5 sqrt (default), "
                         "p=0 none. FULL balance (p=1) stacked on top of class-balanced loss "
                         "over-corrects hard: MELD WA collapsed to 4.7%% by epoch 2 because the "
                         "model stopped predicting neutral at all. Use p=0.5 with cb-beta 0.99.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device()

    tr_ds = MelDataset(args.train, train=True,
                       freq_mask=args.freq_mask, time_mask=args.time_mask)
    va_ds = MelDataset(args.val, train=False)
    if args.balanced_sampler:
        from torch.utils.data import WeightedRandomSampler
        cnt = tr_ds.class_counts(NUM_CLASSES)
        per_class = np.power(1.0 / np.maximum(cnt, 1), args.sampler_power)
        sample_w = per_class[tr_ds.labels]
        sampler = WeightedRandomSampler(sample_w.tolist(), num_samples=len(tr_ds),
                                        replacement=True)
        tr = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
        share = per_class * cnt; share = share / share.sum() * 100
        print(f"rebalanced sampler ON (p={args.sampler_power}) - expected per-class share: "
              + " ".join(f"{c[:4]}:{v:.1f}%" for c, v in zip(CANONICAL, share)))
    else:
        tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0)
    va = DataLoader(va_ds, batch_size=args.batch * 2, shuffle=False,
                    num_workers=args.workers, persistent_workers=args.workers > 0)

    counts = tr_ds.class_counts(NUM_CLASSES)
    model = AcousticPrimer(NUM_CLASSES).to(device)
    if args.init:
        # weights_only=False: our checkpoints store a numpy scalar in `metrics`, and
        # torch 2.6+ defaults weights_only=True. These are our own files.
        sd = torch.load(ROOT / args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
        print(f"warm-started from {args.init}")

    crit_e = ClassBalancedFocalLoss(counts, beta=args.cb_beta, gamma=args.gamma).to(device)
    crit_g = nn.CrossEntropyLoss(ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tr), pct_start=0.15)

    print(f"device={device}  params={count_params(model):,}  "
          f"train={len(tr_ds)}  val={len(va_ds)}")
    print(f"class counts: {dict(zip(CANONICAL, counts.tolist()))}")

    # Track BOTH: best macro-F1 (early-stopping signal) and best UA. The Phase 1 gate is
    # specified on UA, and selecting the checkpoint by F1 can pick a different epoch - a
    # gap easily worth the ~0.4 UA we were missing by. Persist both, select by the metric
    # the gate actually measures.
    best, best_ep, hist, start_ep = -1.0, -1, [], 1
    best_ua, best_ua_ep = -1.0, -1
    LAST = CKPT / f"{args.tag}_seed{args.seed}_last.pt"
    if args.resume:
        st = torch.load(ROOT / args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        best, best_ep, hist, start_ep = st["best"], st["best_ep"], st["hist"], st["epoch"] + 1
        print(f"RESUMED from {args.resume} at epoch {start_ep} (best F1 {best:.2f} @ ep{best_ep})")

    for ep in range(start_ep, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for mel, y, g in tr:
            mel, y, g = mel.to(device), y.to(device), g.to(device)
            e_log, g_log = model(mel)
            loss = crit_e(e_log, y)
            if (g >= 0).any():
                loss = loss + args.gender_weight * crit_g(g_log, g)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); tot += loss.item()
        m, _, _ = evaluate(model, va, device)
        hist.append(dict(epoch=ep, loss=tot / len(tr), **m))
        star = ""
        if m["F1"] > best:
            best, best_ep, star = m["F1"], ep, "  *"
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "metrics": m, "epoch": ep},
                       CKPT / f"{args.tag}_seed{args.seed}.pt")
        if m["UA"] > best_ua:
            best_ua, best_ua_ep = m["UA"], ep
            star += "U"
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "metrics": m, "epoch": ep},
                       CKPT / f"{args.tag}_seed{args.seed}_bestua.pt")
        print(f"ep{ep:3d}  loss {tot/len(tr):.4f}  UA {m['UA']:5.2f}  "
              f"WA {m['WA']:5.2f}  F1 {m['F1']:5.2f}  {time.time()-t0:4.1f}s{star}")
        # full resume state every epoch: a stop costs at most one epoch, and the LR
        # schedule position is preserved (warm-starting from best weights alone would
        # restart the cycle and waste progress).
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best,
                    "best_ep": best_ep, "hist": hist, "args": vars(args)}, LAST)
        json.dump(hist, open(CKPT / f"{args.tag}_seed{args.seed}_history.json", "w"), indent=1)

        if ep - best_ep >= args.patience:
            print(f"early stop (no F1 gain in {args.patience} epochs)"); break

    print(f"\nBEST-F1  epoch {best_ep}     F1 {best:.2f}  -> {args.tag}_seed{args.seed}.pt")
    print(f"BEST-UA  epoch {best_ua_ep}  UA {best_ua:.2f}  -> {args.tag}_seed{args.seed}_bestua.pt")
    json.dump(hist, open(CKPT / f"{args.tag}_seed{args.seed}_history.json", "w"), indent=1)

    model.load_state_dict(torch.load(CKPT / f"{args.tag}_seed{args.seed}.pt", weights_only=False)["model"])
    m, _, _ = evaluate(model, va, device)
    print(f"final val: UA {m['UA']:.2f}  WA {m['WA']:.2f}  F1 {m['F1']:.2f}")


if __name__ == "__main__":
    main()

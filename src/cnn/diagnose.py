"""Under- vs over-fitting diagnostic + confusion matrix for a trained primer."""
import sys, json, pathlib, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from cnn.model import AcousticPrimer
from cnn.dataset import MelDataset
from data.labels import CANONICAL

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="artifacts/cnn/pretrain_seed0.pt")
ap.add_argument("--train", default="acted_train")
ap.add_argument("--val", default="acted_val")
a = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
m = AcousticPrimer(7).to(dev); m.load_state_dict(ck["model"]); m.eval()

def run(split):
    dl = DataLoader(MelDataset(split, train=False), batch_size=256,
                    shuffle=False, num_workers=0)
    ys, ps = [], []
    with torch.no_grad():
        for mel, y, _ in dl:
            ps.append(m(mel.to(dev))[0].argmax(-1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()

ytr, ptr = run(a.train)
yva, pva = run(a.val)
tr_wa, va_wa = accuracy_score(ytr, ptr), accuracy_score(yva, pva)
print(f"TRAIN (no aug) : WA {tr_wa*100:5.2f}  UA {recall_score(ytr,ptr,average='macro',zero_division=0)*100:5.2f}")
print(f"VAL            : WA {va_wa*100:5.2f}  UA {recall_score(yva,pva,average='macro',zero_division=0)*100:5.2f}")
wa_gap = (tr_wa - va_wa) * 100
tr_ua = recall_score(ytr, ptr, average="macro", zero_division=0)
va_ua = recall_score(yva, pva, average="macro", zero_division=0)
ua_gap = (tr_ua - va_ua) * 100
# Judge on BOTH axes. A model can underfit on WA while overfitting on UA - which is exactly
# what oversampling-with-replacement does: it memorises the few minority clips (train UA high)
# without generalising (test UA low). Reporting only the WA gap hid that and printed
# "UNDERFIT" for a model with a 16.7-point train->test UA gap.
if ua_gap > 12:
    verdict = f"OVERFIT ON MINORITY CLASSES (UA gap {ua_gap:+.1f}) - oversampling is memorising"
elif wa_gap > 15:
    verdict = "OVERFIT"
elif tr_wa < 0.60 and ua_gap < 8:
    verdict = "UNDERFIT - model cannot even fit its own train set"
else:
    verdict = "balanced"
print(f"train-val gap  : WA {wa_gap:+.2f} pts | UA {ua_gap:+.2f} pts  ->  {verdict}\n")

cm = confusion_matrix(yva, pva, labels=range(7))
print("val confusion (rows=true, cols=pred):")
print(f"{'':10s}" + "".join(f"{c[:4]:>6s}" for c in CANONICAL) + "   recall")
for i, c in enumerate(CANONICAL):
    tot = cm[i].sum()
    print(f"{c:10s}" + "".join(f"{v:6d}" for v in cm[i]) + f"   {cm[i,i]/max(tot,1)*100:5.1f}%")
print("\npred distribution:", {CANONICAL[i]: int((pva==i).sum()) for i in range(7)})

hp = pathlib.Path(a.ckpt.replace(".pt", "_history.json"))
if hp.exists():
    h = json.load(open(hp))
    first, last = h[0]["loss"], h[-1]["loss"]
    tail = [x["loss"] for x in h[-8:]]
    flat = (max(tail) - min(tail)) < 0.02
    print(f"\nloss ep1 {first:.3f} -> ep{len(h)} {last:.3f}  "
          f"(drop {first - last:.3f}, last-8 {'FLAT' if flat else 'still moving'})")

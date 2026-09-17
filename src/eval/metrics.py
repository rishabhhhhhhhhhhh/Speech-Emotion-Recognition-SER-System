"""Evaluation metrics for the hybrid pipeline.

Beyond UA / WA / macro-F1 (Paper B's metrics), three that are specific to this project:

  eswr_soft_accuracy  - Plutchik-weighted score. Measures HOW WRONG the errors are, not just
                        how many. Confusing anger with disgust (adjacent on the wheel) is a
                        better failure than confusing anger with joy.
  format_compliance   - fraction of responses matching the schema the prompt requested.
  override_precision  - THE money metric. When <prior_check> says CONFLICT, how often is the
                        model right? If this exceeds the CNN's standalone accuracy, the LALM
                        is adding real value on top of the primer rather than laundering it.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data.labels import CANONICAL, canonicalize
from rl.plutchik import similarity


def core_metrics(y_true, y_pred) -> dict:
    """UA (macro recall), WA (accuracy), macro-F1 - all as percentages."""
    return {
        "UA": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "WA": accuracy_score(y_true, y_pred) * 100,
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    }


def eswr_soft_accuracy(pred_labels, true_labels) -> float:
    """Mean Plutchik similarity S(pred, true). Unparseable predictions score 0."""
    vals = [0.0 if p is None else similarity(p, t)
            for p, t in zip(pred_labels, true_labels)]
    return float(np.mean(vals)) * 100


def format_compliance(rewards) -> float:
    return float(np.mean(rewards)) * 100


def parse_rate(pred_labels) -> float:
    """Fraction of responses from which a VALID canonical label could be extracted."""
    return float(np.mean([p is not None for p in pred_labels])) * 100


def override_analysis(verdicts, pred_labels, true_labels, cnn_preds) -> dict:
    """Split accuracy by the model's stated AGREE/CONFLICT verdict on the CNN prior.

    `override_precision` is accuracy on the CONFLICT subset - the cases where the model
    explicitly disagreed with the prior. Compare it against `cnn_acc_on_conflict`: if the
    model overrides the CNN and is right more often than the CNN was on those same items,
    the reasoning stage is contributing, not parroting.
    """
    verdicts = np.asarray([v if v else "NONE" for v in verdicts])
    correct = np.asarray([p is not None and p == t for p, t in zip(pred_labels, true_labels)])
    cnn_correct = np.asarray([c == t for c, t in zip(cnn_preds, true_labels)])
    out = {}
    for tag in ("AGREE", "CONFLICT", "NONE"):
        m = verdicts == tag
        out[f"n_{tag.lower()}"] = int(m.sum())
        out[f"acc_{tag.lower()}"] = float(correct[m].mean() * 100) if m.any() else float("nan")
    m = verdicts == "CONFLICT"
    out["override_precision"] = float(correct[m].mean() * 100) if m.any() else float("nan")
    out["cnn_acc_on_conflict"] = float(cnn_correct[m].mean() * 100) if m.any() else float("nan")
    out["override_rate"] = float(m.mean() * 100)
    out["lift_over_cnn_on_conflict"] = out["override_precision"] - out["cnn_acc_on_conflict"]
    return out


def cascade_curve(cnn_probs, cnn_preds, lalm_preds, true_labels, thresholds=None):
    """Accuracy vs LALM-call-rate as the CNN max-prob threshold tau sweeps.

    Below tau the CNN answers alone (free); at or above it we escalate to the LALM.
    Directly addresses Paper B's stated limitation that LALM inference is too slow for
    real-time use. Free post-hoc - no extra GPU time.
    """
    conf = np.asarray(cnn_probs).max(axis=1)
    cnn_preds = np.asarray(cnn_preds)
    lalm_preds = np.asarray([p if p is not None else "" for p in lalm_preds])
    true_labels = np.asarray(true_labels)
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.01, 0.05), 2)
    rows = []
    for tau in thresholds:
        escalate = conf < tau                       # low CNN confidence -> ask the LALM
        final = np.where(escalate, lalm_preds, cnn_preds)
        rows.append({
            "tau": float(tau),
            "lalm_call_rate": float(escalate.mean() * 100),
            "WA": float((final == true_labels).mean() * 100),
            "UA": recall_score(true_labels, final, average="macro", zero_division=0) * 100,
        })
    return rows


def confusion(y_true, y_pred, labels=None):
    labels = labels or CANONICAL
    return confusion_matrix(y_true, y_pred, labels=labels)


def format_confusion(cm, labels=None) -> str:
    labels = labels or CANONICAL
    out = [f"{'':10s}" + "".join(f"{c[:4]:>6s}" for c in labels) + "   recall"]
    for i, c in enumerate(labels):
        tot = cm[i].sum()
        out.append(f"{c:10s}" + "".join(f"{v:6d}" for v in cm[i]) +
                   f"   {cm[i, i] / max(tot, 1) * 100:5.1f}%")
    return "\n".join(out)


def canon_predictions(raw_answers, fallback_texts=None):
    """Map raw <answer> strings to canonical labels (None if unmappable).

    `fallback_texts` (the full responses) are consulted ONLY when no <answer> tag was
    produced. Zero-shot Qwen2-Audio often replies with a bare label like "angry" and no
    tags at all - that is a real prediction and scoring it 0 would understate the model
    rather than measure it. Format compliance is scored separately and stays strict, so
    the leniency here never inflates the format number.
    """
    out = []
    for i, a in enumerate(raw_answers):
        lab = canonicalize(a) if a is not None else None
        if lab is None and fallback_texts is not None:
            txt = (fallback_texts[i] or "").strip()
            lab = canonicalize(txt)
            if lab is None:
                tail = [ln for ln in txt.splitlines() if ln.strip()]
                if tail:
                    lab = canonicalize(tail[-1])
        out.append(lab)
    return out

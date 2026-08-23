"""Build speaker-disjoint manifests for the acted corpora (CNN pretraining pool).

    .venv/bin/python src/data/build_manifests.py

Design decision — RAVDESS holdout
---------------------------------
RAVDESS is used TWICE in this project: as CNN pretraining data, and as a
cross-corpus eval set in Phase 5 (mirroring Paper B Table 2). To keep that eval
honest, actors 21-24 are held out of pretraining ENTIRELY and reserved for eval.
Paper A excluded only actor 24; we exclude four so the eval set is a usable size
(240 clips) and covers both genders.
"""
from __future__ import annotations
import csv, sys, pathlib, random
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data.labels import (LABEL2ID, RAVDESS_EMOTION, RAVDESS_ORIG, RAVDESS_STATEMENT,
                         CREMAD_EMOTION, CREMAD_SENTENCE, SAVEE_EMOTION)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data/raw", ROOT / "data/manifests"
OUT.mkdir(parents=True, exist_ok=True)

RAVDESS_HOLDOUT_ACTORS = {21, 22, 23, 24}   # reserved for Phase 5 cross-corpus eval
VAL_FRACTION = 0.15
SEED = 1337
MIN_BYTES = 2000                             # below this = unfetched LFS pointer / empty


SILENCE_PEAK = 1e-6

def _probe(path: pathlib.Path):
    """Return (duration_s, samplerate) or (None, reason).

    Also rejects DIGITALLY SILENT files. CREMA-D ships at least one
    (1076_MTI_SAD_XX.wav, peak == rms == 0.0). A silent clip has zero variance, so
    per-utterance CMVN in precompute_mel divides by ~0 and emits an all-zero mel -
    a training sample with a label but no signal. Caught here so it never reaches
    the cache. Costs a full read per file (~30s over the whole pool); worth it.
    """
    try:
        info = sf.info(str(path))
        y, _ = sf.read(str(path), dtype="float32", always_2d=False)
        import numpy as _np
        if _np.abs(y).max() < SILENCE_PEAK:
            return None, "digitally silent"
        return round(info.frames / info.samplerate, 3), info.samplerate
    except Exception as e:
        return None, f"unreadable: {e!r}"


def scan_ravdess(rows, skipped):
    for wav in sorted((RAW / "RAVDESS").rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            skipped.append((str(wav), "bad filename")); continue
        if wav.stat().st_size < MIN_BYTES:
            skipped.append((str(wav), "too small")); continue
        emo, statement, actor = parts[2], parts[4], int(parts[6])
        dur, sr = _probe(wav)
        if dur is None:
            skipped.append((str(wav), sr)); continue
        rows.append(dict(
            uid=f"ravdess_{wav.stem}", wav_path=str(wav.relative_to(ROOT)),
            corpus="ravdess", speaker_id=f"ravdess_{actor:02d}",
            gender="male" if actor % 2 == 1 else "female",
            label=RAVDESS_EMOTION[emo], orig_label=RAVDESS_ORIG[emo],
            transcript=RAVDESS_STATEMENT[statement], duration=dur, samplerate=sr,
            role="holdout" if actor in RAVDESS_HOLDOUT_ACTORS else "pool",
        ))


def scan_cremad(rows, skipped):
    demo_path = RAW / "CREMA-D-repo/VideoDemographics.csv"
    sex = {}
    if demo_path.exists():
        with open(demo_path, newline="") as fh:
            for r in csv.DictReader(fh):
                sex[r["ActorID"].strip()] = r["Sex"].strip().lower()
    for wav in sorted((RAW / "CREMA-D-repo/AudioWAV").glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) != 4:
            skipped.append((str(wav), "bad filename")); continue
        if wav.stat().st_size < MIN_BYTES:
            skipped.append((str(wav), "lfs pointer / too small")); continue
        actor, sent, emo, _inten = parts
        if emo not in CREMAD_EMOTION:
            skipped.append((str(wav), f"unknown emotion {emo}")); continue
        dur, sr = _probe(wav)
        if dur is None:
            skipped.append((str(wav), sr)); continue
        rows.append(dict(
            uid=f"cremad_{wav.stem}", wav_path=str(wav.relative_to(ROOT)),
            corpus="cremad", speaker_id=f"cremad_{actor}",
            gender=sex.get(actor, "unknown"),
            label=CREMAD_EMOTION[emo], orig_label=emo,
            transcript=CREMAD_SENTENCE.get(sent, ""), duration=dur, samplerate=sr,
            role="pool",
        ))


def scan_savee(rows, skipped):
    base = RAW / "SAVEE"
    if not base.exists():
        print("  SAVEE not present - skipping (optional corpus)"); return
    for wav in sorted(base.rglob("*.wav")):
        stem, spk = wav.stem, wav.parent.name
        code = "".join(ch for ch in stem if ch.isalpha()).lower()
        code = code if code in SAVEE_EMOTION else code[:1]
        if code not in SAVEE_EMOTION:
            skipped.append((str(wav), "bad code")); continue
        dur, sr = _probe(wav)
        if dur is None:
            skipped.append((str(wav), sr)); continue
        rows.append(dict(
            uid=f"savee_{spk}_{stem}", wav_path=str(wav.relative_to(ROOT)),
            corpus="savee", speaker_id=f"savee_{spk}", gender="male",
            label=SAVEE_EMOTION[code], orig_label=code, transcript="",
            duration=dur, samplerate=sr, role="pool",
        ))


def main():
    rows, skipped = [], []
    print("scanning RAVDESS ..."); scan_ravdess(rows, skipped)
    print("scanning CREMA-D ..."); scan_cremad(rows, skipped)
    print("scanning SAVEE   ..."); scan_savee(rows, skipped)

    df = pd.DataFrame(rows)
    df["label_id"] = df["label"].map(LABEL2ID)
    assert df["label_id"].notna().all(), "unmapped label present"

    holdout = df[df.role == "holdout"].copy()
    pool = df[df.role == "pool"].copy()

    # speaker-disjoint split: whole speakers go to train or val, never both
    speakers = sorted(pool.speaker_id.unique())
    random.Random(SEED).shuffle(speakers)
    n_val = max(1, round(len(speakers) * VAL_FRACTION))
    val_speakers = set(speakers[:n_val])
    pool["split"] = pool.speaker_id.map(lambda s: "val" if s in val_speakers else "train")
    holdout["split"] = "holdout"

    for name, part in (("acted_train", pool[pool.split == "train"]),
                       ("acted_val", pool[pool.split == "val"]),
                       ("ravdess_holdout", holdout)):
        part.to_csv(OUT / f"{name}.csv", index=False)
        print(f"  wrote {name}.csv  n={len(part):5d}  speakers={part.speaker_id.nunique()}")

    # hard assertion: no speaker may appear in more than one split
    tr = set(pool[pool.split == "train"].speaker_id)
    va = set(pool[pool.split == "val"].speaker_id)
    ho = set(holdout.speaker_id)
    assert not (tr & va), f"LEAK train/val: {tr & va}"
    assert not (tr & ho) and not (va & ho), "LEAK holdout"
    print("  speaker-disjointness: OK")

    if skipped:
        pd.DataFrame(skipped, columns=["path", "reason"]).to_csv(OUT / "skipped_acted.csv", index=False)
        print(f"\n  SKIPPED {len(skipped)} files -> manifests/skipped_acted.csv")
        for reason, grp in pd.DataFrame(skipped, columns=["p", "r"]).groupby("r"):
            print(f"    {len(grp):5d}  {reason}")

    print("\n--- class distribution (pool) ---")
    print(pd.crosstab(pool.label, pool.corpus, margins=True).to_string())
    print("\n--- duration (s) ---")
    print(pool.duration.describe()[["mean", "50%", "min", "max"]].round(2).to_string())
    print(f"\n--- samplerates present: {sorted(pool.samplerate.unique())} ---")


if __name__ == "__main__":
    main()

"""Extract MELD audio (HF mirror) and build manifests.

    .venv/bin/python src/data/prepare_meld.py --extract --manifest

Source: HF `ajyy/MELD_audio` - audio-only, 1.45 GB vs 10.1 GB for the official
MELD.Raw.tar.gz. We discard video anyway (`-vn`), so the official tarball is ~85% waste.
Mirror verified against official MELD: train 9989 / dev 1109 / test 2610, identical CSV
columns, identical 7-emotion label set, ~47% neutral skew as documented.

Layout inside each tarball:  {split}/dia{D}_utt{U}.flac
FLAC is lossless and `soundfile` reads it natively, so NO ffmpeg stage is needed at all -
`precompute_mel.py` resamples to 16 kHz mono on load, exactly as it does for RAVDESS 48 kHz.

Transcripts come from the CSVs' `Utterance` column (no ASR), repaired via
`textclean.clean_transcript` - 26% of rows carry C1 control chars where curly quotes belong.
"""
from __future__ import annotations
import argparse, pathlib, sys, tarfile
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from data.labels import LABEL2ID, canonicalize
from data.textclean import clean_transcript

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/MELD_hf"
OUT = ROOT / "data/manifests"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ("train", "dev", "test")
EXPECTED = {"train": 9989, "dev": 1109, "test": 2610}
SILENCE_PEAK = 1e-6


def extract():
    for split in SPLITS:
        tgz = SRC / "archive" / f"{split}.tar.gz"
        dst = SRC / split
        if dst.exists() and any(dst.glob("*.flac")):
            print(f"  {split}/ already extracted ({len(list(dst.glob('*.flac')))} flac)")
            continue
        if not tgz.exists():
            print(f"  !! {tgz.name} missing - still downloading?")
            continue
        print(f"  extracting {tgz.name} ...")
        with tarfile.open(tgz) as t:
            t.extractall(SRC)
        print(f"    -> {len(list(dst.glob('*.flac')))} flac files")


def manifest():
    import numpy as np
    for split in SPLITS:
        csv = SRC / f"{split}.csv"
        adir = SRC / split
        if not csv.exists() or not adir.exists():
            print(f"  !! {split}: missing csv or audio dir - skipping")
            continue
        df = pd.read_csv(csv, encoding="utf-8")
        if len(df) != EXPECTED[split]:
            print(f"  !! {split}: {len(df)} rows, expected {EXPECTED[split]}")

        rows, skipped = [], []
        for _, r in df.iterrows():
            fp = adir / f"dia{int(r.Dialogue_ID)}_utt{int(r.Utterance_ID)}.flac"
            lab = canonicalize(str(r.Emotion))
            if lab is None:
                skipped.append((fp.name, f"bad label {r.Emotion}")); continue
            if not fp.exists():
                skipped.append((fp.name, "audio missing")); continue
            try:
                info = sf.info(str(fp))
                y, _ = sf.read(str(fp), dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                if np.abs(y).max() < SILENCE_PEAK:
                    skipped.append((fp.name, "digitally silent")); continue
                dur = round(info.frames / info.samplerate, 3)
            except Exception as e:                                  # noqa: BLE001
                skipped.append((fp.name, f"unreadable {e!r}")); continue

            rows.append(dict(
                uid=f"meld_{split}_dia{int(r.Dialogue_ID)}_utt{int(r.Utterance_ID)}",
                wav_path=str(fp.relative_to(ROOT)), corpus="meld",
                speaker_id=f"meld_{str(r.Speaker).strip()}", gender="unknown",
                label=lab, orig_label=str(r.Emotion), label_id=LABEL2ID[lab],
                transcript=clean_transcript(r.Utterance),
                duration=dur, samplerate=info.samplerate,
                dialogue_id=int(r.Dialogue_ID), utterance_id=int(r.Utterance_ID),
                split=split))

        out = pd.DataFrame(rows)
        out.to_csv(OUT / f"meld_{split}.csv", index=False)
        print(f"\n  meld_{split}.csv  n={len(out)}  skipped={len(skipped)}")
        if skipped:
            pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(
                OUT / f"skipped_meld_{split}.csv", index=False)
            for reason, grp in pd.DataFrame(skipped, columns=["f", "r"]).groupby("r"):
                print(f"      {len(grp):5d}  {reason}")
        vc = out.label.value_counts()
        print("      " + "  ".join(f"{k}:{v}" for k, v in vc.items()))
        print(f"      neutral share {vc.get('neutral',0)/max(len(out),1)*100:.1f}%  |  "
              f"dur mean {out.duration.mean():.2f}s median {out.duration.median():.2f}s  |  "
              f"<=4s {(out.duration<=4).mean()*100:.1f}%  |  "
              f"sr {sorted(out.samplerate.unique())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.manifest:
        manifest()
    if not (a.extract or a.manifest):
        ap.print_help()

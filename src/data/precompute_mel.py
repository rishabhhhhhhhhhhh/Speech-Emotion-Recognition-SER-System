"""Precompute log-mel spectrograms into memmapped float16 arrays.

    .venv/bin/python src/data/precompute_mel.py acted_train acted_val ravdess_holdout

Why memmap + float16
--------------------
~26k clips x (128 x 250) float32 = 3.3 GB. float16 halves that to 1.65 GB and the
values are dB in roughly [-80, 0] so fp16 precision is far more than enough. A single
memmapped array per split gives sequential reads and OS page-cache reuse, instead of
26k tiny file opens per epoch. Cast to float32 in the Dataset.

NEVER run this on MPS - librosa/STFT stays on CPU (see CLAUDE.md).
"""
from __future__ import annotations
import sys, pathlib, warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, librosa
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = pathlib.Path(__file__).resolve().parents[2]
MEL_DIR = ROOT / "data/mel"; MEL_DIR.mkdir(parents=True, exist_ok=True)

# --- feature spec: FIXED. Any change invalidates every cached mel + the CNN. ---
SR          = 16000
N_FFT       = 1024
HOP         = 256
WIN         = 1024
N_MELS      = 128
FMIN, FMAX  = 20, 8000
DUR_S       = 4.0
N_SAMPLES   = int(SR * DUR_S)          # 64000
N_FRAMES    = 250                      # 64000 // 256
WORKERS     = 8


def wav_to_logmel(rel_path: str) -> np.ndarray:
    """wav -> (128, 250) float16 log-mel, per-utterance CMVN. Resamples to 16 kHz."""
    y, _ = librosa.load(ROOT / rel_path, sr=SR, mono=True)
    if len(y) < N_SAMPLES:                       # centre-pad short clips
        pad = N_SAMPLES - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))
    else:                                        # centre-crop long clips
        start = (len(y) - N_SAMPLES) // 2
        y = y[start:start + N_SAMPLES]
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)[:, :N_FRAMES]
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-5)   # per-utterance CMVN
    return logmel.astype(np.float16)


def _worker(args):
    i, rel = args
    try:
        return i, wav_to_logmel(rel), None
    except Exception as e:                                   # noqa: BLE001
        return i, None, repr(e)


def build(split: str):
    man = ROOT / f"data/manifests/{split}.csv"
    if not man.exists():
        print(f"  !! {split}.csv missing - skipping"); return
    df = pd.read_csv(man)
    out = MEL_DIR / f"{split}.npy"
    arr = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float16, shape=(len(df), N_MELS, N_FRAMES))

    # NOTE: ThreadPool, not multiprocessing.Pool. librosa's STFT/resample release the
    # GIL, so threads parallelise fine - and multiprocessing.Pool deadlocks on teardown
    # under macOS spawn when the parent writes to a memmap. Verified: it hung at 0% CPU.
    failures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        it = ex.map(_worker, enumerate(df.wav_path))
        for i, mel, err in tqdm(it, total=len(df), desc=f"{split:16s}", ncols=78):
            if err is None:
                arr[i] = mel
            else:
                failures.append((df.wav_path.iloc[i], err))
    arr.flush()
    df[["uid", "label", "label_id", "speaker_id", "corpus", "gender"]].to_csv(
        MEL_DIR / f"{split}_index.csv", index=False)

    mb = out.stat().st_size / 1048576
    print(f"  {split}: {len(df)} clips -> {out.name} ({mb:.0f} MB)  failures={len(failures)}")
    for p, e in failures[:5]:
        print(f"      FAIL {p}: {e}")
    return failures


if __name__ == "__main__":
    splits = sys.argv[1:] or ["acted_train", "acted_val", "ravdess_holdout"]
    print(f"spec: {SR} Hz | n_fft {N_FFT} | hop {HOP} | {N_MELS} mels | "
          f"{FMIN}-{FMAX} Hz | {DUR_S}s -> ({N_MELS}, {N_FRAMES})")
    for s in splits:
        build(s)

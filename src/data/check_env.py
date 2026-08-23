"""Environment sanity check. Run after ANY install/upgrade. Exits non-zero on failure.

    .venv/bin/python src/data/check_env.py
"""
import os, sys, tempfile, importlib, importlib.util

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

FAIL = []
def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label:<34} {detail}")
    if not cond:
        FAIL.append(label)

print("=" * 64)
print("SER Project — environment sanity check")
print("=" * 64)

# --- interpreter --------------------------------------------------------
check("python 3.11.x", sys.version_info[:2] == (3, 11), sys.version.split()[0])
check("running inside project venv",
      "SER Project/.venv" in sys.prefix, sys.prefix)

# --- TensorFlow must NOT exist -----------------------------------------
tf_present = importlib.util.find_spec("tensorflow") is not None
keras_present = importlib.util.find_spec("keras") is not None
check("tensorflow ABSENT", not tf_present, "would break protobuf/numpy")
check("keras ABSENT", not keras_present)

# --- numpy anchor -------------------------------------------------------
import numpy as np
check("numpy == 1.26.4", np.__version__ == "1.26.4", np.__version__)

# --- torch / MPS --------------------------------------------------------
import torch, torchaudio
check("torch imports", True, torch.__version__)
check("torchaudio imports", True, torchaudio.__version__)
check("MPS available", torch.backends.mps.is_available())
try:
    x = torch.randn(64, 128, device="mps")
    _ = (x @ x.T).sum().item()
    conv = torch.nn.Conv2d(1, 32, 3, padding=1).to("mps")
    shape = tuple(conv(torch.randn(2, 1, 128, 250, device="mps")).shape)
    check("MPS matmul + conv2d", shape == (2, 32, 128, 250), str(shape))
except Exception as e:                                    # noqa: BLE001
    check("MPS matmul + conv2d", False, repr(e))
check("no float64 on MPS (expected)", torch.zeros(2, device="mps").dtype == torch.float32)

# --- librosa round trip (numba JIT is the usual failure point) ----------
import librosa, soundfile as sf, numba
check("numba", numba.__version__.startswith("0.60"), numba.__version__)
check("librosa", librosa.__version__.startswith("0.10.2"), librosa.__version__)
try:
    sr = 16000
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sr * 2) / sr)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        sf.write(fh.name, tone, sr)
        y, got_sr = librosa.load(fh.name, sr=16000)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=256, win_length=1024,
            n_mels=128, fmin=20, fmax=8000)
        logmel = librosa.power_to_db(mel)
    os.unlink(fh.name)
    check("librosa load @16k", got_sr == 16000 and y.dtype == np.float32, f"{y.shape}")
    check("log-mel (128, ~125)", logmel.shape[0] == 128, f"{logmel.shape} {logmel.dtype}")
except Exception as e:                                    # noqa: BLE001
    check("librosa mel pipeline", False, repr(e))

# --- the rest -----------------------------------------------------------
for name in ("scipy", "sklearn", "pandas", "pyarrow", "matplotlib",
             "seaborn", "tqdm", "audiomentations"):
    try:
        m = importlib.import_module(name)
        check(f"{name}", True, getattr(m, "__version__", "ok"))
    except Exception as e:                                # noqa: BLE001
        check(f"{name}", False, repr(e))

# --- ffmpeg (needed for MELD mp4 -> wav) --------------------------------
check("ffmpeg on PATH", os.system("ffmpeg -version >/dev/null 2>&1") == 0)

print("=" * 64)
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED: {', '.join(FAIL)}")
    sys.exit(1)
print("ALL CHECKS PASSED")

"""Mel dataset with online SpecAugment. Reads the memmapped float16 cache.

Augmentation is ONLINE and cheap (mask/roll/gain on the cached mel) rather than
offline waveform augmentation, because recomputing mels every epoch on an M5 would
dominate wall-clock. Paper A used no augmentation at all - a likely reason its
accuracy stalled at 68.88% on 1,440 clips.
"""
from __future__ import annotations
import pathlib
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

ROOT = pathlib.Path(__file__).resolve().parents[2]
GENDER2ID = {"male": 0, "female": 1}          # anything else -> -1 (ignored in loss)


class MelDataset(Dataset):
    def __init__(self, split: str, train: bool = False,
                 n_freq_mask: int = 2, freq_mask: int = 8,
                 n_time_mask: int = 2, time_mask: int = 15,
                 max_roll: int = 20, gain_db: float = 0.4):
        mel_path = ROOT / f"data/mel/{split}.npy"
        idx_path = ROOT / f"data/mel/{split}_index.csv"
        if not mel_path.exists():
            raise FileNotFoundError(f"{mel_path} - run src/data/precompute_mel.py first")
        self.mels = np.load(mel_path, mmap_mode="r")      # (N, 128, 250) float16
        self.idx = pd.read_csv(idx_path)
        assert len(self.mels) == len(self.idx), "mel/index length mismatch"
        self.labels = self.idx.label_id.to_numpy(np.int64)
        self.genders = self.idx.gender.map(GENDER2ID).fillna(-1).to_numpy(np.int64)
        self.train = train
        self.n_freq_mask, self.freq_mask = n_freq_mask, freq_mask
        self.n_time_mask, self.time_mask = n_time_mask, time_mask
        self.max_roll, self.gain_db = max_roll, gain_db

    def __len__(self):
        return len(self.idx)

    def class_counts(self, n_classes: int = 7):
        return np.bincount(self.labels, minlength=n_classes)

    def _augment(self, m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_mels, n_frames = m.shape
        if self.max_roll:                                   # small time shift
            m = np.roll(m, rng.integers(-self.max_roll, self.max_roll + 1), axis=1)
        if self.gain_db:                                    # level jitter (post-CMVN)
            m = m + rng.normal(0.0, self.gain_db)
        for _ in range(self.n_freq_mask):                   # SpecAugment freq masks
            f = rng.integers(0, self.freq_mask + 1)
            if f:
                f0 = rng.integers(0, n_mels - f)
                m[f0:f0 + f, :] = 0.0
        for _ in range(self.n_time_mask):                   # SpecAugment time masks
            t = rng.integers(0, self.time_mask + 1)
            if t:
                t0 = rng.integers(0, n_frames - t)
                m[:, t0:t0 + t] = 0.0
        return m

    def __getitem__(self, i: int):
        m = np.asarray(self.mels[i], dtype=np.float32)      # fp16 cache -> fp32
        if self.train:
            m = self._augment(m, np.random.default_rng())
        return (torch.from_numpy(np.ascontiguousarray(m)),
                int(self.labels[i]), int(self.genders[i]))

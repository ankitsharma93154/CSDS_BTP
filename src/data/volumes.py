"""Generic 3D-volume dataset + k-fold splitting, shared by MedMNIST3D and LIDC."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset


def _augment(vol: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light 3D augmentation (flip / rot90 / gamma), ~50% chance each."""
    if rng.random() < 0.5:
        vol = np.flip(vol, axis=int(rng.integers(0, 3)) + 1)
    if rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        ax = tuple(int(a) + 1 for a in rng.choice(3, 2, replace=False))
        vol = np.rot90(vol, k=k, axes=ax)
    if rng.random() < 0.5:
        vol = np.clip(vol, 0, 1) ** float(rng.uniform(0.7, 1.5))
    return np.ascontiguousarray(vol)


class Volume3DDataset(Dataset):
    """images: (N, D, H, W) or (N, 1, D, H, W); any dtype.  Values scaled to
    [0, 1] (÷255 if they look like uint8).  Resized to ``size`` by centre
    pad/crop.  ``train`` enables augmentation."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 size: int = 32, train: bool = False, seed: int = 0):
        self.images = images
        self.labels = np.asarray(labels).astype(np.int64).reshape(-1)
        self.size = size
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def _resize(self, vol: np.ndarray) -> np.ndarray:
        d = vol.shape[-1]
        if d == self.size:
            return vol
        if d < self.size:
            pad = self.size - d
            lo = pad // 2
            p = (lo, pad - lo)
            return np.pad(vol, ((0, 0), p, p, p), mode="constant")
        c = (d - self.size) // 2
        return vol[:, c:c + self.size, c:c + self.size, c:c + self.size]

    def __getitem__(self, i):
        vol = np.asarray(self.images[i], dtype=np.float32)
        if vol.ndim == 3:
            vol = vol[None]
        if vol.max() > 1.5:
            vol = vol / 255.0
        vol = self._resize(vol)
        if self.train:
            vol = _augment(vol, self.rng)
        return torch.from_numpy(np.ascontiguousarray(vol)).float(), int(self.labels[i])


# back-compat alias
MedMNIST3DArray = Volume3DDataset


def make_datasets(images, labels, train_idx, test_idx, size=32, seed=0):
    tr = Volume3DDataset(images[train_idx], labels[train_idx], size, train=True, seed=seed)
    te = Volume3DDataset(images[test_idx], labels[test_idx], size, train=False, seed=seed)
    return tr, te


def kfold_splits(y: np.ndarray, folds: int, repeats: int, seed: int,
                 val_fraction_of_dev: float = 1 / 8):
    """Yield (rep, fold, tr_idx, val_idx, test_idx).

    Matches the paper: 5-fold CV over the whole set (test = one fold, 20%);
    the remaining 80% is split 87.5:12.5 -> ~70:10 train:val overall.
    """
    y = np.asarray(y).reshape(-1)
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        for fold, (dev_idx, test_idx) in enumerate(skf.split(np.zeros_like(y), y)):
            tr_idx, val_idx = train_test_split(
                dev_idx, test_size=val_fraction_of_dev,
                stratify=y[dev_idx], random_state=seed + rep)
            yield rep, fold, tr_idx, val_idx, test_idx

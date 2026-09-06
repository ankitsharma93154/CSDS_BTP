"""Load the preprocessed LIDC-IDRI nodule archive produced by src/data/lidc.py."""
from __future__ import annotations

import numpy as np


def load_lidc_npz(path: str):
    """Return (images, labels): images (N, 1, 75, 75, 75) float32 in [0, 1]."""
    d = np.load(path, allow_pickle=True)
    images = d["images"].astype(np.float32)
    labels = d["labels"].astype(np.int64).reshape(-1)
    if images.ndim == 4:
        images = images[:, None]
    return images, labels

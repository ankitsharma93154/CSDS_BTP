"""MedMNIST3D loaders (VesselMNIST3D, SynapseMNIST3D) for pipeline validation.

The paper uses these as external validation sets (Table 4): 28^3 volumes,
resized to 32^3.  Both are binary.  The dataset/splitting machinery lives in
``src.data.volumes``.
"""
from __future__ import annotations

import numpy as np

import medmnist
from medmnist import INFO

from .volumes import (MedMNIST3DArray, Volume3DDataset, kfold_splits,  # noqa: F401
                      make_datasets)

FLAGS = {"vessel": "vesselmnist3d", "synapse": "synapsemnist3d"}


def load_pooled(name: str, size: int = 32, download: bool = True, root: str = "data"):
    """(images, labels, info) with train+val+test pooled -- the paper runs k-fold
    CV across the entire dataset rather than the official split."""
    flag = FLAGS.get(name, name)
    info = INFO[flag]
    cls = getattr(medmnist, info["python_class"])
    parts = [cls(split=s, download=download, root=root, size=28)
             for s in ("train", "val", "test")]
    images = np.concatenate([p.imgs for p in parts], axis=0)
    labels = np.concatenate([p.labels for p in parts], axis=0)
    return images, labels, info

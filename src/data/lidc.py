"""LIDC-IDRI preprocessing — reproduces Section 3 of the paper.

Runs where the DICOM archive lives (Kaggle: attach the "LIDC-IDRI" dataset;
or a machine with the TCIA download). Not runnable on the dev laptop.

Pipeline (paper Section 3, following Zhao et al. 2020):
  1. pylidc scan iteration; drop scans with slice thickness > 3 mm.
  2. Keep nodules annotated by >= 3 of 4 radiologists, diameter >= 3 mm.
  3. Ground truth = mean malignancy over annotations:
        < 3 -> benign (0),  > 3 -> malignant (1),  == 3 -> discard.
  4. DBSCAN on per-nodule feature vectors to drop outliers.
  5. Crop 40^3 voxels at the consensus centroid; resample to 1.0 mm^3 isotropic
     (spline); rescale to 75^3.
  6. Clip HU to [-1000, 400] and min-max normalise to [0, 1].
Target: 1013 nodules (561 benign / 452 malignant).

Usage:
    python -m src.data.lidc --out data/lidc_nodules.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HU_MIN, HU_MAX = -1000.0, 400.0
CROP = 40
TARGET = 75
MIN_ANNOTATIONS = 3
MIN_DIAMETER_MM = 3.0
MAX_SLICE_THICKNESS_MM = 3.0


def _lazy_imports():
    import pylidc as pl
    from scipy.ndimage import zoom
    from sklearn.cluster import DBSCAN
    return pl, zoom, DBSCAN


def _consensus_label(anns) -> int | None:
    mal = np.mean([a.malignancy for a in anns])
    if mal == 3:
        return None
    return int(mal > 3)


def _nodule_feature(anns) -> np.ndarray:
    """Semantic-feature vector used for DBSCAN outlier detection."""
    attrs = ("subtlety", "internalStructure", "calcification", "sphericity",
             "margin", "lobulation", "spiculation", "texture", "malignancy")
    return np.array([np.mean([getattr(a, x) for a in anns]) for x in attrs], dtype=np.float64)


def _crop_resample(vol_hu: np.ndarray, spacing: np.ndarray, centroid: np.ndarray, zoom):
    half = CROP // 2
    c = np.round(centroid).astype(int)
    lo = np.clip(c - half, 0, np.array(vol_hu.shape) - 1)
    hi = np.clip(lo + CROP, 0, np.array(vol_hu.shape))
    lo = hi - CROP
    patch = vol_hu[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    if patch.shape != (CROP, CROP, CROP):
        pad = [(0, CROP - s) for s in patch.shape]
        patch = np.pad(patch, pad, mode="constant", constant_values=HU_MIN)
    # resample to 1.0 mm^3 isotropic, then to TARGET^3
    patch = zoom(patch, spacing, order=3)
    patch = zoom(patch, np.array([TARGET] * 3) / np.array(patch.shape), order=3)
    patch = np.clip(patch, HU_MIN, HU_MAX)
    return (patch - HU_MIN) / (HU_MAX - HU_MIN)


def build(out_path: str, dbscan_eps: float = 1.6, dbscan_min_samples: int = 8,
          limit: int | None = None):
    pl, zoom, DBSCAN = _lazy_imports()

    vols, labels, feats, uids = [], [], [], []
    scans = pl.query(pl.Scan)
    total = scans.count()
    for si, scan in enumerate(scans):
        if scan.slice_thickness is None or scan.slice_thickness > MAX_SLICE_THICKNESS_MM:
            continue
        try:
            clusters = scan.cluster_annotations()
        except Exception:
            continue
        vol = None
        for anns in clusters:
            if len(anns) < MIN_ANNOTATIONS:
                continue
            if np.mean([a.diameter for a in anns]) < MIN_DIAMETER_MM:
                continue
            lab = _consensus_label(anns)
            if lab is None:
                continue
            if vol is None:
                vol = scan.to_volume().astype(np.float32)          # (x, y, z) HU
                spacing = np.array([scan.pixel_spacing, scan.pixel_spacing,
                                    scan.slice_thickness], dtype=np.float64)
            centroid = np.mean([a.centroid for a in anns], axis=0)
            try:
                patch = _crop_resample(vol, spacing, centroid, zoom)
            except Exception:
                continue
            vols.append(patch.astype(np.float32))
            labels.append(lab)
            feats.append(_nodule_feature(anns))
            uids.append(f"{scan.patient_id}:{int(centroid[2])}")
        if (si + 1) % 25 == 0:
            print(f"  scan {si+1}/{total}  kept {len(vols)} nodules")
        if limit and len(vols) >= limit:
            break

    vols = np.stack(vols)[:, None]           # (N, 1, 75, 75, 75)
    labels = np.array(labels, dtype=np.int64)
    feats = np.stack(feats)

    # DBSCAN outlier removal (label == -1 -> outlier)
    fz = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(fz)
    keep = db.labels_ != -1
    print(f"DBSCAN kept {keep.sum()}/{len(keep)}  "
          f"(benign {int((labels[keep]==0).sum())}, malignant {int((labels[keep]==1).sum())})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=vols[keep], labels=labels[keep],
                        uids=np.array(uids)[keep])
    print(f"saved -> {out_path}  shape {vols[keep].shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lidc_nodules.npz")
    ap.add_argument("--dbscan-eps", type=float, default=1.6)
    ap.add_argument("--dbscan-min-samples", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    build(a.out, a.dbscan_eps, a.dbscan_min_samples, a.limit)


if __name__ == "__main__":
    main()

"""Uncertainty quantification and evaluation metrics.

Implements the measurement/assessment machinery from Zahari, Cox & Obara (2024),
"Uncertainty-aware image classification on 3D CT lung", CIBM 172:108324.

Conventions
-----------
* ``member_probs``: array of shape (S, N, C) -- per-sample softmax probability
  vectors from S stochastic realisations.  S is:
      MCD  -> number of forward passes M
      DE   -> number of ensemble members
      EMCD -> members * passes  (flattened)
* ``mean_probs``:   (N, C) predictive distribution, Eq. (1):  mean over S.
* ``labels``:       (N,) integer ground-truth class ids.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Predictive distribution
# --------------------------------------------------------------------------- #
def mean_probs_from_members(member_probs: np.ndarray) -> np.ndarray:
    """Eq. (1): p(y*|x*) = (1/S) sum_s p*_s."""
    return np.asarray(member_probs, dtype=np.float64).mean(axis=0)


def predictions(mean_probs: np.ndarray) -> np.ndarray:
    return np.asarray(mean_probs).argmax(axis=-1)


# --------------------------------------------------------------------------- #
# Uncertainty measures
# --------------------------------------------------------------------------- #
def predictive_entropy(mean_probs: np.ndarray, base: str = "e") -> np.ndarray:
    """Eq. (3): H(p) = -sum_c p_c log p_c   (per sample)."""
    p = np.clip(np.asarray(mean_probs, dtype=np.float64), _EPS, 1.0)
    h = -np.sum(p * np.log(p), axis=-1)
    if base == "2":
        h /= np.log(2.0)
    return h


def predictive_std(member_probs: np.ndarray,
                   mean_probs: np.ndarray | None = None,
                   mode: str = "pred_class") -> np.ndarray:
    """Eq. (4): sigma = sqrt( (1/S) sum_s (p*_s - p)^2 ).

    ``mode``:
        "pred_class" -> std of the winning-class probability (Asgharnezhad et al.)
        "mean"       -> mean of the per-class std
        "max"        -> max per-class std
    """
    m = np.asarray(member_probs, dtype=np.float64)          # (S, N, C)
    if mean_probs is None:
        mean_probs = m.mean(axis=0)
    var = np.mean((m - mean_probs[None]) ** 2, axis=0)      # (N, C)
    std = np.sqrt(var)
    if mode == "pred_class":
        idx = np.asarray(mean_probs).argmax(axis=-1)
        return std[np.arange(std.shape[0]), idx]
    if mode == "mean":
        return std.mean(axis=-1)
    if mode == "max":
        return std.max(axis=-1)
    raise ValueError(mode)


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1] across the evaluation set (H_norm, sigma_norm)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.min(x), np.max(x)
    if hi - lo < _EPS:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# Uncertainty ratio  (Section 3.4)
# --------------------------------------------------------------------------- #
def uncertainty_ratio(uncertainty: np.ndarray, correct_mask: np.ndarray) -> float:
    """mean uncertainty of incorrect predictions / mean of correct predictions."""
    u = np.asarray(uncertainty, dtype=np.float64)
    c = np.asarray(correct_mask, dtype=bool)
    if c.all() or (~c).all():
        return float("nan")
    return float(u[~c].mean() / (u[c].mean() + _EPS))


# --------------------------------------------------------------------------- #
# Uncertainty-aware confusion matrix  (Fig. 3, Eqs. 5-8)
# --------------------------------------------------------------------------- #
def uncertainty_confusion(correct_mask: np.ndarray,
                          norm_uncertainty: np.ndarray,
                          threshold: float) -> dict:
    """Split predictions into certain/uncertain at ``threshold`` and score.

    certain  := norm_uncertainty <  threshold
    uncertain:= norm_uncertainty >= threshold
    """
    correct = np.asarray(correct_mask, dtype=bool)
    uncertain = np.asarray(norm_uncertainty, dtype=np.float64) >= threshold

    TC = int(np.sum(correct & ~uncertain))
    FU = int(np.sum(correct & uncertain))
    FC = int(np.sum(~correct & ~uncertain))
    TU = int(np.sum(~correct & uncertain))

    def _safe(n, d):
        return float(n / d) if d else float("nan")

    return {
        "threshold": float(threshold),
        "TC": TC, "FC": FC, "TU": TU, "FU": FU,
        "U_Accuracy":    _safe(TU + TC, TU + TC + FU + FC),   # Eq. 5
        "U_Precision":   _safe(TU, TU + FU),                  # Eq. 6
        "U_Recall":      _safe(TU, TU + FC),                  # Eq. 7
        "U_Specificity": _safe(TC, TC + FU),                  # Eq. 8
    }


# --------------------------------------------------------------------------- #
# Data referral  (Section 4.4, Table 5)
# --------------------------------------------------------------------------- #
def referral_metrics(labels: np.ndarray,
                     preds: np.ndarray,
                     norm_uncertainty: np.ndarray,
                     threshold: float,
                     positive_label: int = 1) -> dict:
    """Refer (drop) samples with norm_uncertainty >= threshold; score the rest."""
    y = np.asarray(labels)
    yhat = np.asarray(preds)
    keep = np.asarray(norm_uncertainty, dtype=np.float64) < threshold
    n = len(y)
    n_keep = int(keep.sum())

    out = {
        "threshold": float(threshold),
        "pct_retained": 100.0 * n_keep / n if n else float("nan"),
        "pct_referral": 100.0 * (n - n_keep) / n if n else float("nan"),
        "n_retained": n_keep,
    }
    if n_keep == 0:
        out.update(accuracy=float("nan"), precision=float("nan"),
                   recall=float("nan"), f1=float("nan"))
        return out

    yk, yhk = y[keep], yhat[keep]
    tp = np.sum((yhk == positive_label) & (yk == positive_label))
    fp = np.sum((yhk == positive_label) & (yk != positive_label))
    fn = np.sum((yhk != positive_label) & (yk == positive_label))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out.update(
        accuracy=float(np.mean(yk == yhk)),
        precision=float(prec), recall=float(rec), f1=float(f1),
    )
    return out


def sweep_thresholds(correct_mask: np.ndarray,
                     norm_uncertainty: np.ndarray,
                     labels: np.ndarray | None = None,
                     preds: np.ndarray | None = None,
                     start: float = 0.01, stop: float = 0.99, step: float = 0.01):
    """Return a list of per-threshold metric dicts over [start, stop]."""
    rows = []
    for t in np.round(np.arange(start, stop + step / 2, step), 4):
        row = uncertainty_confusion(correct_mask, norm_uncertainty, float(t))
        if labels is not None and preds is not None:
            ref = referral_metrics(labels, preds, norm_uncertainty, float(t))
            row.update({f"ref_{k}": v for k, v in ref.items() if k != "threshold"})
        rows.append(row)
    return rows

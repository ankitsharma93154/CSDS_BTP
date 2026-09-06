"""Standard classification metrics (accuracy, AUC, precision, recall, F1)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true, y_prob, positive_label: int = 1) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    p1 = y_prob[:, positive_label]
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        out["auc"] = roc_auc_score(y_true, p1)
    except ValueError:
        out["auc"] = float("nan")
    return out


def aggregate(dicts: list[dict]) -> dict:
    """mean / std across folds or repeats."""
    keys = dicts[0].keys()
    return {k: (float(np.mean([d[k] for d in dicts])),
               float(np.std([d[k] for d in dicts]))) for k in keys}

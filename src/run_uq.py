"""UQ phase (Sections 4.3-4.4): given trained base model(s), run MCD / DE / EMCD,
then reproduce Table 3 (uncertainty summary) and Table 5 (referral at a
threshold) style numbers.

Expects per-fold checkpoints laid out as::

    <ckpt_dir>/r{rep}f{fold}.pth          # single model  (MCD)
    <ckpt_dir>/r{rep}f{fold}_m{k}.pth     # ensemble members (DE / EMCD)

Falls back to the single-model file for every member if member files are absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.medmnist3d import load_pooled, make_datasets
from src.eval_utils import aggregate, classification_metrics
from src.models.factory import build_model
from src.train import set_seed
from src.uncertainty.metrics import (minmax_normalize, predictive_entropy,
                                     predictive_std, referral_metrics,
                                     sweep_thresholds, uncertainty_confusion,
                                     uncertainty_ratio)
from src.uncertainty.predict import de_predict, emcd_predict, mcd_predict


def _load(ckpt, model_name, size, dropout):
    m = build_model(model_name, 1, 2, dropout=dropout, input_size=size)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m


def _member_paths(ckpt_dir: Path, rep, fold, n_members):
    base = ckpt_dir / f"r{rep}f{fold}.pth"
    members = [ckpt_dir / f"r{rep}f{fold}_m{k}.pth" for k in range(n_members)]
    if all(p.exists() for p in members):
        return members
    return [base] * n_members


def summarise(member_probs, labels, std_mode="pred_class"):
    from src.uncertainty.metrics import mean_probs_from_members
    mean_p = mean_probs_from_members(member_probs)
    preds = mean_p.argmax(1)
    correct = preds == labels
    ent = predictive_entropy(mean_p)
    std = predictive_std(member_probs, mean_p, mode=std_mode)
    cm = classification_metrics(labels, mean_p)
    return {
        **cm,
        "ET_Co": float(ent[correct].mean()), "ET_Inc": float(ent[~correct].mean()),
        "STD_Co": float(std[correct].mean()), "STD_Inc": float(std[~correct].mean()),
        "Unc_Ratio": uncertainty_ratio(ent, correct),
        "_arrays": dict(mean_p=mean_p, preds=preds, correct=correct,
                        ent_norm=minmax_normalize(ent), std_norm=minmax_normalize(std),
                        labels=labels),
    }


def run(dataset, model_name, ckpt_dir, *, folds=5, repeats=1, size=32,
        mc_passes=50, n_members=3, dropout=0.5, seed=0, thresholds=(0.5, 0.75),
        out_dir="results/uq", device="cpu"):
    images, labels_all, _ = load_pooled(dataset, size=size)
    y = labels_all.reshape(-1)
    ckpt_dir = Path(ckpt_dir)
    out = Path(out_dir) / f"{dataset}_{model_name}"
    out.mkdir(parents=True, exist_ok=True)

    per_fold = {"MCD": [], "DE": [], "EMCD": []}
    sweeps = {"MCD": [], "DE": [], "EMCD": []}

    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        for fold, (dev_idx, test_idx) in enumerate(skf.split(np.zeros_like(y), y)):
            set_seed(seed + rep * 100 + fold)
            _, test_ds = make_datasets(images, labels_all, test_idx, test_idx, size, seed)
            mpaths = _member_paths(ckpt_dir, rep, fold, n_members)
            single = _load(mpaths[0], model_name, size, dropout)
            ens = [_load(p, model_name, size, dropout) for p in mpaths]

            runs = {
                "MCD": mcd_predict(single, test_ds, passes=mc_passes, device=device),
                "DE": de_predict(ens, test_ds, device=device),
                "EMCD": emcd_predict(ens, test_ds, passes=mc_passes, device=device),
            }
            for name, (members, lab) in runs.items():
                s = summarise(members, lab)
                arr = s.pop("_arrays")
                per_fold[name].append({**s, "rep": rep, "fold": fold})
                sweeps[name].append(sweep_thresholds(
                    arr["correct"], arr["ent_norm"], arr["labels"], arr["preds"]))
                np.savez(out / f"{name}_r{rep}f{fold}.npz", **arr)

    report = {}
    for name in per_fold:
        keys = ("accuracy", "auc", "precision", "recall", "f1",
                "ET_Co", "ET_Inc", "STD_Co", "STD_Inc", "Unc_Ratio")
        report[name] = {"table3": aggregate([{k: d[k] for k in keys} for d in per_fold[name]])}
        for t in thresholds:
            rows = []
            for fold_rows, fold_meta in zip(sweeps[name], per_fold[name]):
                r = min(fold_rows, key=lambda z: abs(z["threshold"] - t))
                rows.append({
                    "U_Accuracy": r["U_Accuracy"], "U_Precision": r["U_Precision"],
                    "U_Recall": r["U_Recall"], "U_Specificity": r["U_Specificity"],
                    "acc": r["ref_accuracy"], "f1": r["ref_f1"],
                    "pct_retained": r["ref_pct_retained"], "pct_referral": r["ref_pct_referral"],
                })
            report[name][f"table5_th{t}"] = aggregate(rows)

    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vessel", choices=["vessel", "synapse"])
    ap.add_argument("--model", default="inceptionresnetv2")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--mc-passes", type=int, default=50)
    ap.add_argument("--n-members", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default="results/uq")
    a = ap.parse_args()
    run(a.dataset, a.model, a.ckpt_dir, folds=a.folds, repeats=a.repeats, size=a.size,
        mc_passes=a.mc_passes, n_members=a.n_members, dropout=a.dropout, seed=a.seed,
        out_dir=a.out_dir, device=a.device)


if __name__ == "__main__":
    main()

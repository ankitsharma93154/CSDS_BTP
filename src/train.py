"""Training + k-fold cross-validation for the model-selection phase (Section 4.2).

Protocol from the paper: Adam, lr 1e-5, batch 8, up to 500 epochs with early
stopping, 5-fold stratified CV with a 70:10:20 train:val:test ratio, each
configuration repeated 3x (mean +/- std).  Defaults here are lighter; override
via CLI for the full run on GPU.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

from src.data.medmnist3d import load_pooled, make_datasets
from src.eval_utils import aggregate, classification_metrics
from src.models.factory import build_model


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _predict_probs(model, ds, batch_size, device):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        ys.append(y.numpy().reshape(-1))
    return np.concatenate(probs), np.concatenate(ys)


def train_model(model, train_ds, val_ds, *, epochs=50, lr=1e-5, batch_size=8,
                patience=15, device="cpu", class_weights=None, log_every=5, tag="",
                select_metric="f1_macro"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        vp, vy = _predict_probs(model, val_ds, batch_size, device)
        vm = classification_metrics(vy, vp)
        if vm[select_metric] > best:
            best = vm[select_metric]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % log_every == 0 or ep == 1:
            print(f"  [{tag}] ep{ep:3d} loss={tot/len(train_ds):.4f} "
                  f"val_{select_metric}={vm[select_metric]:.4f} val_acc={vm['accuracy']:.4f} "
                  f"val_auc={vm['auc']:.4f} best={best:.4f} ({time.time()-t0:.1f}s)")
        if bad >= patience:
            print(f"  [{tag}] early stop at ep{ep} (best val_{select_metric}={best:.4f})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def run_cv(dataset="vessel", model_name="inceptionresnetv2", *, folds=5, repeats=1,
           epochs=50, lr=1e-5, batch_size=8, size=32, patience=15, device="cpu",
           seed=0, out_dir="results", save_models=False, dropout=0.0, members=1,
           select_metric="f1_macro"):
    images, labels, info = load_pooled(dataset, size=size)
    y = labels.reshape(-1)
    print(f"{dataset}: {len(y)} samples, classes {np.bincount(y).tolist()}, input {size}^3")

    out = Path(out_dir) / f"{dataset}_{model_name}"
    out.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        for fold, (dev_idx, test_idx) in enumerate(skf.split(np.zeros_like(y), y)):
            set_seed(seed + rep * 100 + fold)
            tr_idx, val_idx = train_test_split(
                dev_idx, test_size=1 / 8, stratify=y[dev_idx], random_state=seed + rep)
            train_ds, _ = make_datasets(images, labels, tr_idx, tr_idx, size, seed)
            _, val_ds = make_datasets(images, labels, val_idx, val_idx, size, seed)
            _, test_ds = make_datasets(images, labels, test_idx, test_idx, size, seed)

            cw = len(y[tr_idx]) / (2 * np.bincount(y[tr_idx]))
            in_ch = images[0].shape[0] if images.ndim == 5 else 1
            # train `members` independent models per fold (for Deep Ensemble); the
            # first one doubles as the single model used by MCD.
            member_probs = []
            for mi in range(members):
                set_seed(seed + rep * 1000 + fold * 10 + mi)
                model = build_model(model_name, in_channels=in_ch, num_classes=2,
                                    dropout=dropout, input_size=size)
                tag = f"r{rep}f{fold}" + (f"_m{mi}" if members > 1 else "")
                model, _ = train_model(model, train_ds, val_ds, epochs=epochs, lr=lr,
                                       batch_size=batch_size, patience=patience, device=device,
                                       class_weights=cw.tolist(), tag=tag,
                                       select_metric=select_metric)
                tp, ty = _predict_probs(model, test_ds, batch_size, device)
                member_probs.append(tp)
                if save_models:
                    torch.save(model.state_dict(), out / f"{tag}.pth")
            tp = np.mean(member_probs, axis=0)
            m = classification_metrics(ty, tp)
            m.update(rep=rep, fold=fold)
            all_metrics.append(m)
            btag = f"r{rep}f{fold}"
            print(f"  [{btag}] TEST acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
                  f"auc={m['auc']:.4f} f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f}")
            np.savez(out / f"{btag}_preds.npz", probs=tp, labels=ty, test_idx=test_idx)

    _keys = ("accuracy", "balanced_accuracy", "auc", "precision", "recall", "f1",
             "f1_macro", "precision_macro", "recall_macro")
    summary = aggregate([{k: d[k] for k in _keys} for d in all_metrics])
    (out / "summary.json").write_text(json.dumps(
        {"per_fold": all_metrics, "summary": summary}, indent=2))
    print("\n=== SUMMARY (mean +/- std) ===")
    for k, (mu, sd) in summary.items():
        print(f"  {k:10s} {mu:.4f} (+/- {sd:.4f})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="vessel", choices=["vessel", "synapse"])
    ap.add_argument("--model", default="inceptionresnetv2")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="head dropout; set 0.5 for the UQ base model")
    ap.add_argument("--members", type=int, default=1,
                    help="independent models per fold for Deep Ensemble")
    ap.add_argument("--select-metric", default="f1_macro",
                    help="validation metric for early stopping / checkpointing")
    a = ap.parse_args()
    run_cv(a.dataset, a.model, folds=a.folds, repeats=a.repeats, epochs=a.epochs,
           lr=a.lr, batch_size=a.batch_size, size=a.size, patience=a.patience,
           device=a.device, seed=a.seed, out_dir=a.out_dir, save_models=a.save_models,
           dropout=a.dropout, members=a.members, select_metric=a.select_metric)


if __name__ == "__main__":
    main()

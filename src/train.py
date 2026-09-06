"""Training + k-fold cross-validation for the model-selection phase (Section 4.2).

Protocol from the paper: Adam, lr 1e-5, batch 8, up to 500 epochs with early
stopping, 5-fold stratified CV with a 70:10:20 train:val:test ratio, each
configuration repeated 3x (mean +/- std).  Defaults here are lighter; override
via CLI for the full run on GPU.

``run_cv_arrays`` is the core (works on in-memory volumes -> used for LIDC).
``run_cv`` is a thin wrapper that pools a MedMNIST3D dataset first.
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
from torch.utils.data import DataLoader

from src.data.volumes import Volume3DDataset, kfold_splits
from src.eval_utils import aggregate, classification_metrics
from src.models.factory import build_model

_SUMMARY_KEYS = ("accuracy", "balanced_accuracy", "auc", "precision", "recall",
                 "f1", "f1_macro", "precision_macro", "recall_macro")


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
                select_metric="f1_macro", warmup=3, use_amp=True):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # linear warmup then cosine decay -- stabilises the deep Inception nets and
    # lets them converge in far fewer than the paper's 500 epochs.
    def _lr_factor(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        prog = (ep - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_factor)
    w = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    amp = use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        vp, vy = _predict_probs(model, val_ds, batch_size, device)
        vm = classification_metrics(vy, vp)
        if vm[select_metric] > best:
            best = vm[select_metric]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % log_every == 0 or ep == 1:
            print(f"  [{tag}] ep{ep:3d} loss={tot/max(len(train_ds),1):.4f} "
                  f"val_{select_metric}={vm[select_metric]:.4f} val_acc={vm['accuracy']:.4f} "
                  f"val_auc={vm['auc']:.4f} best={best:.4f} ({time.time()-t0:.1f}s)", flush=True)
        if bad >= patience:
            print(f"  [{tag}] early stop at ep{ep} (best val_{select_metric}={best:.4f})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def run_cv_arrays(images, labels, model_name="inceptionresnetv2", *, size,
                  folds=5, repeats=1, epochs=50, lr=1e-5, batch_size=8, patience=15,
                  device="cpu", seed=0, out_dir="results", save_models=False,
                  dropout=0.0, members=1, select_metric="f1_macro", tag_prefix="run"):
    y = np.asarray(labels).reshape(-1)
    in_ch = images.shape[1] if images.ndim == 5 else 1
    print(f"{tag_prefix}/{model_name}: {len(y)} samples, classes {np.bincount(y).tolist()}, "
          f"input {size}^3", flush=True)
    out = Path(out_dir) / f"{tag_prefix}_{model_name}"
    out.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for rep, fold, tr_idx, val_idx, test_idx in kfold_splits(y, folds, repeats, seed):
        set_seed(seed + rep * 100 + fold)
        train_ds = Volume3DDataset(images[tr_idx], y[tr_idx], size, train=True, seed=seed)
        val_ds = Volume3DDataset(images[val_idx], y[val_idx], size, train=False, seed=seed)
        test_ds = Volume3DDataset(images[test_idx], y[test_idx], size, train=False, seed=seed)
        cw = (len(y[tr_idx]) / (2 * np.bincount(y[tr_idx]))).tolist()

        member_probs = []
        for mi in range(members):
            set_seed(seed + rep * 1000 + fold * 10 + mi)
            model = build_model(model_name, in_channels=in_ch, num_classes=2,
                                dropout=dropout, input_size=size)
            tag = f"r{rep}f{fold}" + (f"_m{mi}" if members > 1 else "")
            model, _ = train_model(model, train_ds, val_ds, epochs=epochs, lr=lr,
                                   batch_size=batch_size, patience=patience, device=device,
                                   class_weights=cw, tag=tag, select_metric=select_metric)
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
              f"auc={m['auc']:.4f} f1={m['f1']:.4f} f1_macro={m['f1_macro']:.4f}", flush=True)
        np.savez(out / f"{btag}_preds.npz", probs=tp, labels=ty, test_idx=test_idx)

    summary = aggregate([{k: d[k] for k in _SUMMARY_KEYS} for d in all_metrics])
    (out / "summary.json").write_text(json.dumps({"per_fold": all_metrics, "summary": summary}, indent=2))
    print("\n=== SUMMARY (mean +/- std) ===")
    for k, (mu, sd) in summary.items():
        print(f"  {k:18s} {mu:.4f} (+/- {sd:.4f})")
    return summary


def run_cv(dataset="vessel", model_name="inceptionresnetv2", *, size=32, **kw):
    from src.data.medmnist3d import load_pooled
    images, labels, _ = load_pooled(dataset, size=size)
    kw.setdefault("out_dir", "results")
    return run_cv_arrays(images, labels, model_name, size=size, tag_prefix=dataset, **kw)


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
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--members", type=int, default=1)
    ap.add_argument("--select-metric", default="f1_macro")
    a = ap.parse_args()
    run_cv(a.dataset, a.model, size=a.size, folds=a.folds, repeats=a.repeats,
           epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, patience=a.patience,
           device=a.device, seed=a.seed, out_dir=a.out_dir, save_models=a.save_models,
           dropout=a.dropout, members=a.members, select_metric=a.select_metric)


if __name__ == "__main__":
    main()

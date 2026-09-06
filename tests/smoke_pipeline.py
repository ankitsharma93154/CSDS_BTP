"""Fast end-to-end pipeline check (CPU, tiny subset). Not a scientific run.

Trains a small model on a 200-sample slice of VesselMNIST3D for 2 epochs, then
exercises MCD / DE / EMCD inference and the full uncertainty-metrics stack.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.medmnist3d import MedMNIST3DArray, load_pooled
from src.eval_utils import classification_metrics
from src.models.factory import build_model
from src.train import set_seed, train_model
from src.uncertainty.metrics import (minmax_normalize, predictive_entropy,
                                     predictive_std, referral_metrics,
                                     sweep_thresholds, uncertainty_confusion,
                                     uncertainty_ratio, mean_probs_from_members)
from src.uncertainty.predict import de_predict, emcd_predict, mcd_predict

t0 = time.time()
set_seed(0)
imgs, labels, _ = load_pooled("vessel", size=32)
rng = np.random.default_rng(0)
idx = rng.permutation(len(labels))[:240]
tr, va, te = idx[:160], idx[160:200], idx[200:]
mk = lambda i, train: MedMNIST3DArray(imgs[i], labels[i], size=32, train=train, seed=0)
train_ds, val_ds, test_ds = mk(tr, True), mk(va, False), mk(te, False)
print(f"data ready ({time.time()-t0:.1f}s)  train/val/test = {len(tr)}/{len(va)}/{len(te)}")

members = []
for mi in range(2):                       # 2-model ensemble to keep it fast
    set_seed(mi)
    m = build_model("resnet18", 1, 2, dropout=0.5, input_size=32)
    m, _ = train_model(m, train_ds, val_ds, epochs=2, lr=1e-3, batch_size=8,
                       patience=5, tag=f"m{mi}", log_every=1)
    members.append(m)
print(f"trained 2 models ({time.time()-t0:.1f}s)")

mcd_mem, y = mcd_predict(members[0], test_ds, passes=8)
de_mem, _ = de_predict(members, test_ds)
emcd_mem, _ = emcd_predict(members, test_ds, passes=8)
print("member_probs shapes:", mcd_mem.shape, de_mem.shape, emcd_mem.shape)

for name, mem in [("MCD", mcd_mem), ("DE", de_mem), ("EMCD", emcd_mem)]:
    mp = mean_probs_from_members(mem)
    preds = mp.argmax(1)
    correct = preds == y
    ent = predictive_entropy(mp)
    std = predictive_std(mem, mp)
    cm = classification_metrics(y, mp)
    ur = uncertainty_ratio(ent, correct)
    ent_n = minmax_normalize(ent)
    ucm = uncertainty_confusion(correct, ent_n, 0.5)
    ref = referral_metrics(y, preds, ent_n, 0.5)
    sweep = sweep_thresholds(correct, ent_n, y, preds)
    assert mp.shape == (len(y), 2)
    assert len(sweep) == 99
    assert set(ucm) >= {"TC", "FC", "TU", "FU", "U_Accuracy", "U_Precision",
                        "U_Recall", "U_Specificity"}
    print(f"{name:5s} acc={cm['accuracy']:.3f} f1={cm['f1']:.3f} auc={cm['auc']:.3f} "
          f"UncRatio={ur:.2f} U_Acc={ucm['U_Accuracy']:.3f} "
          f"refAcc={ref['accuracy']:.3f} retained={ref['pct_retained']:.0f}%")

print(f"\nALL PIPELINE STAGES OK  ({time.time()-t0:.1f}s total)")

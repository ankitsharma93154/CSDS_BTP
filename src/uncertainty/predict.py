"""Stochastic inference for the three UQ techniques (Section 3.3).

MCD  : one model, dropout active, M forward passes.
DE   : K independently trained models, 1 pass each.
EMCD : K models, each with dropout active for M passes  -> K*M realisations.

Each function returns ``member_probs`` of shape (S, N, C) plus the ground-truth
labels, which feed src/uncertainty/metrics.py.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Put model in eval mode but re-activate Dropout layers (MC dropout)."""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def _forward_once(model, loader, device) -> np.ndarray:
    probs = []
    for x, _ in loader:
        out = model(x.to(device))
        probs.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def collect_labels(loader) -> np.ndarray:
    return np.concatenate([y.numpy().reshape(-1) for _, y in loader], axis=0)


def mcd_predict(model, dataset, passes: int = 50, batch_size: int = 8,
                device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    enable_mc_dropout(model)
    members = np.stack([_forward_once(model, loader, device) for _ in range(passes)], axis=0)
    return members, collect_labels(loader)


def de_predict(models, dataset, batch_size: int = 8,
               device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mem = []
    for model in models:
        model.to(device)
        model.eval()
        mem.append(_forward_once(model, loader, device))
    return np.stack(mem, axis=0), collect_labels(loader)


def emcd_predict(models, dataset, passes: int = 50, batch_size: int = 8,
                 device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mem = []
    for model in models:
        model.to(device)
        enable_mc_dropout(model)
        for _ in range(passes):
            mem.append(_forward_once(model, loader, device))
    return np.stack(mem, axis=0), collect_labels(loader)

"""
Stage 1 — MedMNIST3D replication (paper Table 4) on Kaggle GPU.

HOW TO RUN ON KAGGLE
--------------------
1. kaggle.com -> Create -> New Notebook.
2. Settings -> Accelerator -> GPU T4 x2 (or P100). Internet: ON.
3. Paste this file into a cell (or upload as a script) and edit GITHUB_URL below.
4. Run all.  Expect ~1.5-3 h for the full grid (2 datasets x model-selection +
   UQ).  Use RUN_MODE="quick" first to sanity-check (~15 min).

Outputs land in /kaggle/working/results and are saved as a downloadable zip.
"""
import os, subprocess, sys, zipfile, time
from pathlib import Path

# --------------------------------------------------------------------------- #
GITHUB_URL = "https://github.com/<your-username>/<your-repo>.git"   # <-- EDIT
RUN_MODE   = "full"        # "quick" (smoke) or "full" (paper protocol)
DATASETS   = ["vessel", "synapse"]
SELECT_MODELS = ["resnet18", "densenet121", "inceptionv3", "inceptionresnetv2"]
# paper evaluates all 8; trim here to fit the weekly GPU budget. Add the rest
# (resnet34/50/101, densenet169) once timings are known.
# --------------------------------------------------------------------------- #

CFG = {
    "quick": dict(folds=2, repeats=1, epochs=5,  patience=3,  mc_passes=10),
    "full":  dict(folds=5, repeats=3, epochs=200, patience=25, mc_passes=50),
}[RUN_MODE]

ROOT = Path("/kaggle/working/repo")
if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", GITHUB_URL, str(ROOT)], check=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "monai>=1.3", "medmnist>=3.0"], check=True)

import numpy as np, torch, json
from src.train import run_cv
from src.run_uq import run as run_uq

Path("data").mkdir(exist_ok=True)
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
t_start = time.time()

# ----------------------------------------------------------------- Phase 1 --- #
selection = {}
for ds in DATASETS:
    selection[ds] = {}
    for model in SELECT_MODELS:
        print(f"\n{'='*70}\n[{ds}] model selection: {model}\n{'='*70}", flush=True)
        summ = run_cv(ds, model, folds=CFG["folds"], repeats=CFG["repeats"],
                      epochs=CFG["epochs"], lr=1e-4, batch_size=16, size=32,
                      patience=CFG["patience"], device=device, seed=0,
                      out_dir="results/stage1_selection", select_metric="f1_macro")
        selection[ds][model] = summ["f1_macro"][0]
    best = max(selection[ds], key=selection[ds].get)
    print(f"\n[{ds}] BEST base model = {best}  (macro-F1 {selection[ds][best]:.4f})")
    selection[ds]["_best"] = best

# ----------------------------------------------------------------- Phase 2 --- #
# retrain best model with dropout=0.5 and 3 ensemble members, then run UQ
for ds in DATASETS:
    best = selection[ds]["_best"]
    print(f"\n{'='*70}\n[{ds}] UQ base training: {best} (dropout .5, 3 members)\n{'='*70}", flush=True)
    run_cv(ds, best, folds=CFG["folds"], repeats=1, epochs=CFG["epochs"], lr=1e-4,
           batch_size=16, size=32, patience=CFG["patience"], device=device, seed=0,
           out_dir="results/stage1_uq_ckpts", save_models=True, dropout=0.5, members=3,
           select_metric="f1_macro")
    run_uq(ds, best, f"results/stage1_uq_ckpts/{ds}_{best}",
           folds=CFG["folds"], repeats=1, size=32, mc_passes=CFG["mc_passes"],
           n_members=3, dropout=0.5, seed=0, thresholds=(0.5, 0.75),
           out_dir="results/stage1_uq", device=device)

# ----------------------------------------------------------------- Package --- #
(Path("results") / "selection_summary.json").write_text(json.dumps(selection, indent=2))
zpath = "/kaggle/working/stage1_results.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for p in Path("results").rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to("results"))
print(f"\nDONE in {(time.time()-t_start)/60:.1f} min -> {zpath}")

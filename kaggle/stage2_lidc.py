"""
Stage 2 — LIDC-IDRI replication on Kaggle.

Two sub-steps, ideally two notebook runs:

  PHASE = "preprocess"   CPU only (~1-3 h). Builds data/lidc_nodules.npz from the
                         DICOM archive. Save Version -> the npz becomes a
                         downloadable output; publish it as a private Kaggle
                         Dataset so training runs skip this step.

  PHASE = "train"        GPU. Loads lidc_nodules.npz (from the dataset you made,
                         mounted under /kaggle/input), runs model selection
                         (Table 1) + UQ (Table 3) + referral sweep (Table 5).

SETUP
-----
* Add data:  "LIDC-IDRI" (TCIA) for preprocess  OR  your preprocessed-npz dataset
  for train.
* Accelerator: None for preprocess, GPU for train.  Internet: On.
* Edit GITHUB_URL and the paths below.
"""
import os, subprocess, sys, time, zipfile, json
from pathlib import Path

# --------------------------------------------------------------------------- #
GITHUB_URL   = "https://github.com/ankitsharma93154/CSDS_BTP.git"
PHASE        = "preprocess"          # "preprocess" | "train"
RUN_MODE     = "quick"               # "quick" first; then "full"

# preprocess: point at the mounted LIDC-IDRI DICOM root (contains LIDC-IDRI-* dirs)
LIDC_DICOM_DIR = "/kaggle/input/lidc-idri/LIDC-IDRI"
# train: point at the mounted npz produced by the preprocess phase
LIDC_NPZ       = "/kaggle/input/lidc-nodules-npz/lidc_nodules.npz"

SELECT_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101",
                 "densenet121", "densenet169", "inceptionv3", "inceptionresnetv2"]
# --------------------------------------------------------------------------- #

CFG = {
    "quick": dict(folds=2, repeats=1, epochs=5,   patience=3,  mc_passes=10),
    "full":  dict(folds=5, repeats=3, epochs=300, patience=30, mc_passes=50),
}[RUN_MODE]

ROOT = Path("/kaggle/working/repo")
if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", GITHUB_URL, str(ROOT)], check=True)
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
Path("data").mkdir(exist_ok=True)


# =============================================================== PREPROCESS === #
if PHASE == "preprocess":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pylidc", "pydicom", "scikit-image"], check=True)
    # pylidc needs a config file pointing at the DICOM root
    Path.home().joinpath(".pylidcrc").write_text(
        f"[dicom]\npath = {LIDC_DICOM_DIR}\nwarn = True\n")
    from src.data.lidc import build
    t0 = time.time()
    build("data/lidc_nodules.npz",
          limit=(40 if RUN_MODE == "quick" else None))
    # expose as a top-level output for easy "New Dataset from output"
    subprocess.run(["cp", "data/lidc_nodules.npz", "/kaggle/working/"], check=True)
    print(f"preprocess done in {(time.time()-t0)/60:.1f} min")
    sys.exit(0)


# ==================================================================== TRAIN === #
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "monai>=1.3", "medmnist>=3.0"], check=True)
import numpy as np, torch
from src.data.lidc_dataset import load_lidc_npz
from src.train import run_cv_arrays
from src.run_uq import run_arrays

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA:", torch.cuda.is_available())
images, labels = load_lidc_npz(LIDC_NPZ)
print("LIDC nodules:", images.shape, "labels:", np.bincount(labels).tolist())
t_start = time.time()

# -- Phase 1: model selection (paper Table 1) ------------------------------- #
sel = {}
for model in SELECT_MODELS:
    print(f"\n{'='*70}\nmodel selection: {model}\n{'='*70}", flush=True)
    s = run_cv_arrays(images, labels, model, size=75, folds=CFG["folds"],
                      repeats=CFG["repeats"], epochs=CFG["epochs"], lr=1e-5, batch_size=8,
                      patience=CFG["patience"], device=device, seed=0,
                      out_dir="results/stage2_selection", select_metric="f1",
                      tag_prefix="lidc")
    sel[model] = s["f1"][0]
best = max(sel, key=sel.get)
print("\nSELECTION (mean F1):", json.dumps({k: round(v, 4) for k, v in sel.items()}, indent=2))
print("BEST:", best)

# -- Phase 2: UQ base + MCD/DE/EMCD (Tables 3 & 5) ------------------------- #
run_cv_arrays(images, labels, best, size=75, folds=CFG["folds"], repeats=1,
              epochs=CFG["epochs"], lr=1e-5, batch_size=8, patience=CFG["patience"],
              device=device, seed=0, out_dir="results/stage2_uq_ckpts", save_models=True,
              dropout=0.5, members=3, select_metric="f1", tag_prefix="lidc")
run_arrays(images, labels, best, f"results/stage2_uq_ckpts/lidc_{best}",
           size=75, folds=CFG["folds"], repeats=1, mc_passes=CFG["mc_passes"],
           n_members=3, dropout=0.5, seed=0, thresholds=(0.5, 0.75),
           out_dir="results/stage2_uq", device=device, tag_prefix="lidc")

Path("results/selection_summary.json").write_text(json.dumps(sel, indent=2))
zp = "/kaggle/working/stage2_results.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in Path("results").rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to("results"))
print(f"\nDONE in {(time.time()-t_start)/60:.1f} min -> {zp}")

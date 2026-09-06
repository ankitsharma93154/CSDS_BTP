# BTP — Replicating "Uncertainty-aware image classification on 3D CT lung"

Base paper: R. Zahari, J. Cox, B. Obara, *Computers in Biology and Medicine* 172 (2024) 108324.
[doi:10.1016/j.compbiomed.2024.108324](https://doi.org/10.1016/j.compbiomed.2024.108324)

## What the paper does

Binary benign/malignant lung-nodule classification on 3D CT (LIDC-IDRI), with
three uncertainty-quantification (UQ) techniques and threshold-based referral of
uncertain cases to experts.

| Phase | Content | Key result |
|---|---|---|
| 1. Model selection | 8 3D CNNs, 5-fold CV × 3 repeats | InceptionResNetV2 best, F1 = 0.845 |
| 2. UQ | MCD / DE / EMCD on the best model | EMCD F1 = 0.866; uncertainty ratio ≈ 2.1 |
| 3. Referral | Threshold on normalised entropy | Accuracy → 0.959 (th 0.5), 0.933 (th 0.75) |
| Validation | SynapseMNIST3D, VesselMNIST3D | DE best on both |

## This repo

PyTorch reimplementation (paper uses TensorFlow). Structure:

```
src/
  models/            8-model zoo
    factory.py                 build_model(name, ...) — MONAI ResNet/DenseNet + local Inception nets
    inception_resnet_v2_3d.py  3D port of Inception-ResNet-v2 (paper's best model)
    inception_v3_3d.py         3D port of Inception-v3
  data/
    medmnist3d.py      VesselMNIST3D / SynapseMNIST3D loaders (pipeline validation)
    lidc.py            LIDC-IDRI preprocessing  (TODO — runs on Kaggle)
  uncertainty/
    predict.py        MCD / DE / EMCD stochastic inference
    metrics.py        entropy, std, uncertainty ratio, U-confusion matrix, referral sweep
  train.py            k-fold CV training  (phase 1 & phase 2 base models)
  run_uq.py           phase 2 + 3: UQ inference → Table 3 / Table 5 style reports
  eval_utils.py       accuracy / AUC / precision / recall / F1
```

### Param-count check vs paper Table 2

| Model | Paper | Ours |
|---|---|---|
| ResNet18/34/50/101 | 33.2 / 63.5 / 46.2 / 85.2 M | 33.2 / 63.5 / 46.2 / 85.2 M ✓ |
| DenseNet121/169 | 11.2 / 18.5 M | 11.2 / 18.5 M ✓ |
| InceptionV3 | 34.1 M | 34.0 M ✓ |
| InceptionResNetV2 | 67.5 M | 47.4 M (3D port differs — no standard reference) |

## Running

Local (CPU) — pipeline validation only:

```bash
pip install -r requirements.txt
# phase 1: pick best model on a small dataset
python -m src.train --dataset vessel --model inceptionresnetv2 --folds 5 --epochs 60
# phase 2: retrain best model with dropout + 3 ensemble members
python -m src.train --dataset vessel --model inceptionresnetv2 --folds 5 \
    --dropout 0.5 --members 3 --save-models --out-dir results/uq_ckpts
# phase 2/3: UQ + referral
python -m src.run_uq --dataset vessel --model inceptionresnetv2 \
    --ckpt-dir results/uq_ckpts/vessel_inceptionresnetv2
```

Kaggle GPU — full MedMNIST3D replication and all LIDC-IDRI work (see `notebooks/`).

## Compute plan

- **Local machine**: no CUDA GPU, ~9 GB free disk → pipeline dev + small MedMNIST3D runs only.
- **Kaggle** (30 GPU-hrs/week free): MedMNIST3D full replication, then LIDC-IDRI
  (dataset attachable, no local download of the 125 GB DICOM archive).

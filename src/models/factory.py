"""Model zoo for the replication.

The paper evaluates eight 3D CNNs: ResNet18/34/50/101, DenseNet121/169,
InceptionResNetV2, InceptionV3.  ResNet/DenseNet come from MONAI; the two
Inception nets are local 3D ports.

Every model exposes a single dropout layer immediately before the final
classification layer (``dropout`` arg) -- this is the layer the paper enables
at test time for Monte Carlo Dropout.
"""
from __future__ import annotations

import torch.nn as nn
from monai.networks.nets import DenseNet, ResNet
from monai.networks.nets.resnet import ResNetBlock, ResNetBottleneck

from .inception_resnet_v2_3d import InceptionResNetV2_3D
from .inception_v3_3d import InceptionV3_3D

_RESNET_CFG = {
    "resnet18":  (ResNetBlock,      [2, 2, 2, 2],  [64, 128, 256, 512]),
    "resnet34":  (ResNetBlock,      [3, 4, 6, 3],  [64, 128, 256, 512]),
    "resnet50":  (ResNetBottleneck, [3, 4, 6, 3],  [64, 128, 256, 512]),
    "resnet101": (ResNetBottleneck, [3, 4, 23, 3], [64, 128, 256, 512]),
}


def _build_resnet(name: str, in_channels: int, num_classes: int, dropout: float):
    block, layers, planes = _RESNET_CFG[name]
    net = ResNet(
        block=block, layers=layers, block_inplanes=planes,
        spatial_dims=3, n_input_channels=in_channels,
        conv1_t_size=7, conv1_t_stride=2, no_max_pool=False,
        shortcut_type="B", num_classes=num_classes,
    )
    # splice a Dropout in front of the final linear layer
    in_features = net.fc.in_features
    net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return net


def _build_densenet(name: str, in_channels: int, num_classes: int, dropout: float):
    cfg = {"densenet121": (6, 12, 24, 16), "densenet169": (6, 12, 32, 32)}[name]
    net = DenseNet(
        spatial_dims=3, in_channels=in_channels, out_channels=num_classes,
        init_features=64, growth_rate=32, block_config=cfg,
        dropout_prob=0.0,   # keep DenseNet's internal dropout off; use head dropout
    )
    # MONAI DenseNet head: class_layers = [relu, pool, flatten, linear]
    linear = net.class_layers.out
    net.class_layers.out = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(linear.in_features, num_classes))
    return net


def build_model(name: str, in_channels: int = 1, num_classes: int = 2,
                dropout: float = 0.0, input_size: int = 32) -> nn.Module:
    name = name.lower()
    stem = "standard" if input_size >= 64 else "small"
    if name in _RESNET_CFG:
        return _build_resnet(name, in_channels, num_classes, dropout)
    if name in ("densenet121", "densenet169"):
        return _build_densenet(name, in_channels, num_classes, dropout)
    if name in ("inceptionresnetv2", "irv2"):
        return InceptionResNetV2_3D(in_channels, num_classes, dropout, stem=stem)
    if name in ("inceptionv3", "iv3"):
        return InceptionV3_3D(in_channels, num_classes, dropout, stem=stem)
    raise ValueError(f"unknown model '{name}'")


MODEL_NAMES = [
    "resnet18", "resnet34", "resnet50", "resnet101",
    "densenet121", "densenet169", "inceptionresnetv2", "inceptionv3",
]

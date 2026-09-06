"""3D Inception-ResNet-v2  (Szegedy et al., 2016), ported to volumetric input.

The paper's best base model.  torchvision / MONAI do not ship it, so this is a
faithful 3D port: same block structure (stem, 5x Inception-ResNet-A,
Reduction-A, 10x Inception-ResNet-B, Reduction-B, 5x Inception-ResNet-C), with
2D ops replaced by their 3D counterparts.

``stem`` can be "standard" (aggressive downsampling, for >=75^3 inputs) or
"small" (gentler, for MedMNIST3D-sized 28-32^3 inputs).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Conv3dBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_ch, eps=1e-3, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Mixed5b(nn.Module):
    """Stem output -> 320 channels (Inception-A input)."""
    def __init__(self, in_ch):
        super().__init__()
        self.branch0 = Conv3dBN(in_ch, 96, 1)
        self.branch1 = nn.Sequential(Conv3dBN(in_ch, 48, 1), Conv3dBN(48, 64, 5, padding=2))
        self.branch2 = nn.Sequential(Conv3dBN(in_ch, 64, 1), Conv3dBN(64, 96, 3, padding=1),
                                     Conv3dBN(96, 96, 3, padding=1))
        self.branch3 = nn.Sequential(nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
                                     Conv3dBN(in_ch, 64, 1))

    def forward(self, x):
        return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)], 1)


class InceptionResNetA(nn.Module):
    def __init__(self, scale=0.17):
        super().__init__()
        self.scale = scale
        self.branch0 = Conv3dBN(320, 32, 1)
        self.branch1 = nn.Sequential(Conv3dBN(320, 32, 1), Conv3dBN(32, 32, 3, padding=1))
        self.branch2 = nn.Sequential(Conv3dBN(320, 32, 1), Conv3dBN(32, 48, 3, padding=1),
                                     Conv3dBN(48, 64, 3, padding=1))
        self.conv = nn.Conv3d(128, 320, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mix = torch.cat([self.branch0(x), self.branch1(x), self.branch2(x)], 1)
        return self.relu(x + self.scale * self.conv(mix))


class ReductionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = Conv3dBN(320, 384, 3, stride=2, padding=1)
        self.branch1 = nn.Sequential(Conv3dBN(320, 256, 1), Conv3dBN(256, 256, 3, padding=1),
                                     Conv3dBN(256, 384, 3, stride=2, padding=1))
        self.branch2 = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.branch0(x), self.branch1(x), self.branch2(x)], 1)  # -> 1088


class InceptionResNetB(nn.Module):
    def __init__(self, scale=0.10):
        super().__init__()
        self.scale = scale
        self.branch0 = Conv3dBN(1088, 192, 1)
        self.branch1 = nn.Sequential(
            Conv3dBN(1088, 128, 1),
            Conv3dBN(128, 160, (1, 1, 7), padding=(0, 0, 3)),
            Conv3dBN(160, 192, (1, 7, 1), padding=(0, 3, 0)),
        )
        self.conv = nn.Conv3d(384, 1088, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mix = torch.cat([self.branch0(x), self.branch1(x)], 1)
        return self.relu(x + self.scale * self.conv(mix))


class ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(Conv3dBN(1088, 256, 1),
                                     Conv3dBN(256, 384, 3, stride=2, padding=1))
        self.branch1 = nn.Sequential(Conv3dBN(1088, 256, 1),
                                     Conv3dBN(256, 288, 3, stride=2, padding=1))
        self.branch2 = nn.Sequential(Conv3dBN(1088, 256, 1), Conv3dBN(256, 288, 3, padding=1),
                                     Conv3dBN(288, 320, 3, stride=2, padding=1))
        self.branch3 = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.branch0(x), self.branch1(x),
                          self.branch2(x), self.branch3(x)], 1)  # -> 2080


class InceptionResNetC(nn.Module):
    def __init__(self, scale=0.20, activation=True):
        super().__init__()
        self.scale = scale
        self.activation = activation
        self.branch0 = Conv3dBN(2080, 192, 1)
        self.branch1 = nn.Sequential(
            Conv3dBN(2080, 192, 1),
            Conv3dBN(192, 224, (1, 1, 3), padding=(0, 0, 1)),
            Conv3dBN(224, 256, (1, 3, 1), padding=(0, 1, 0)),
        )
        self.conv = nn.Conv3d(448, 2080, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        mix = torch.cat([self.branch0(x), self.branch1(x)], 1)
        out = x + self.scale * self.conv(mix)
        return self.relu(out) if self.activation else out


def _stem_standard(in_ch):
    return nn.Sequential(
        Conv3dBN(in_ch, 32, 3, stride=2, padding=1), Conv3dBN(32, 32, 3, padding=1),
        Conv3dBN(32, 64, 3, padding=1),
        nn.MaxPool3d(3, stride=2, padding=1),
        Conv3dBN(64, 80, 1), Conv3dBN(80, 192, 3, padding=1),
    ), 192


def _stem_small(in_ch):
    return nn.Sequential(
        Conv3dBN(in_ch, 32, 3, padding=1), Conv3dBN(32, 64, 3, padding=1),
        nn.MaxPool3d(2), Conv3dBN(64, 128, 3, padding=1), Conv3dBN(128, 192, 3, padding=1),
    ), 192


class InceptionResNetV2_3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 dropout: float = 0.0, stem: str = "standard",
                 a_blocks: int = 5, b_blocks: int = 10, c_blocks: int = 5):
        super().__init__()
        self.stem, stem_out = (_stem_standard if stem == "standard" else _stem_small)(in_channels)
        self.mixed_5b = Mixed5b(stem_out)
        self.repeat_a = nn.Sequential(*[InceptionResNetA(0.17) for _ in range(a_blocks)])
        self.reduction_a = ReductionA()
        self.repeat_b = nn.Sequential(*[InceptionResNetB(0.10) for _ in range(b_blocks)])
        self.reduction_b = ReductionB()
        self.repeat_c = nn.Sequential(*[InceptionResNetC(0.20) for _ in range(c_blocks)])
        self.block8 = InceptionResNetC(1.0, activation=False)
        self.conv2d_7b = Conv3dBN(2080, 1536, 1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)          # the "dropout layer just before the classification layer"
        self.fc = nn.Linear(1536, num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.mixed_5b(x)
        x = self.repeat_a(x)
        x = self.reduction_a(x)
        x = self.repeat_b(x)
        x = self.reduction_b(x)
        x = self.repeat_c(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x = self.global_pool(x).flatten(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.fc(x)

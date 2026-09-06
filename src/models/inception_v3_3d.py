"""3D Inception-v3 (Szegedy et al., 2015), compact volumetric port.

Structure: stem -> 3x InceptionA -> ReductionA -> 4x InceptionB -> ReductionB
-> 2x InceptionC -> global pool -> dropout -> fc.  The auxiliary classifier is
omitted (not used by the paper's pipeline).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .inception_resnet_v2_3d import Conv3dBN


class InceptionA(nn.Module):
    def __init__(self, in_ch, pool_features):
        super().__init__()
        self.b1x1 = Conv3dBN(in_ch, 64, 1)
        self.b5x5 = nn.Sequential(Conv3dBN(in_ch, 48, 1), Conv3dBN(48, 64, 5, padding=2))
        self.b3x3 = nn.Sequential(Conv3dBN(in_ch, 64, 1), Conv3dBN(64, 96, 3, padding=1),
                                  Conv3dBN(96, 96, 3, padding=1))
        self.bpool = nn.Sequential(nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
                                   Conv3dBN(in_ch, pool_features, 1))

    def forward(self, x):
        return torch.cat([self.b1x1(x), self.b5x5(x), self.b3x3(x), self.bpool(x)], 1)


class ReductionA(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.b3x3 = Conv3dBN(in_ch, 384, 3, stride=2, padding=1)
        self.b3x3dbl = nn.Sequential(Conv3dBN(in_ch, 64, 1), Conv3dBN(64, 96, 3, padding=1),
                                     Conv3dBN(96, 96, 3, stride=2, padding=1))
        self.bpool = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.b3x3(x), self.b3x3dbl(x), self.bpool(x)], 1)


class InceptionB(nn.Module):
    def __init__(self, in_ch, c7):
        super().__init__()
        self.b1x1 = Conv3dBN(in_ch, 192, 1)
        self.b7 = nn.Sequential(
            Conv3dBN(in_ch, c7, 1),
            Conv3dBN(c7, c7, (1, 1, 7), padding=(0, 0, 3)),
            Conv3dBN(c7, 192, (1, 7, 1), padding=(0, 3, 0)),
        )
        self.b7dbl = nn.Sequential(
            Conv3dBN(in_ch, c7, 1),
            Conv3dBN(c7, c7, (1, 7, 1), padding=(0, 3, 0)),
            Conv3dBN(c7, c7, (1, 1, 7), padding=(0, 0, 3)),
            Conv3dBN(c7, c7, (1, 7, 1), padding=(0, 3, 0)),
            Conv3dBN(c7, 192, (1, 1, 7), padding=(0, 0, 3)),
        )
        self.bpool = nn.Sequential(nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
                                   Conv3dBN(in_ch, 192, 1))

    def forward(self, x):
        return torch.cat([self.b1x1(x), self.b7(x), self.b7dbl(x), self.bpool(x)], 1)


class ReductionB(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.b3x3 = nn.Sequential(Conv3dBN(in_ch, 192, 1),
                                  Conv3dBN(192, 320, 3, stride=2, padding=1))
        self.b7x3 = nn.Sequential(
            Conv3dBN(in_ch, 192, 1),
            Conv3dBN(192, 192, (1, 1, 7), padding=(0, 0, 3)),
            Conv3dBN(192, 192, (1, 7, 1), padding=(0, 3, 0)),
            Conv3dBN(192, 192, 3, stride=2, padding=1),
        )
        self.bpool = nn.MaxPool3d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.b3x3(x), self.b7x3(x), self.bpool(x)], 1)


class InceptionC(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.b1x1 = Conv3dBN(in_ch, 320, 1)
        self.b3_1 = Conv3dBN(in_ch, 384, 1)
        self.b3_2a = Conv3dBN(384, 384, (1, 1, 3), padding=(0, 0, 1))
        self.b3_2b = Conv3dBN(384, 384, (1, 3, 1), padding=(0, 1, 0))
        self.b3dbl_1 = nn.Sequential(Conv3dBN(in_ch, 448, 1), Conv3dBN(448, 384, 3, padding=1))
        self.b3dbl_2a = Conv3dBN(384, 384, (1, 1, 3), padding=(0, 0, 1))
        self.b3dbl_2b = Conv3dBN(384, 384, (1, 3, 1), padding=(0, 1, 0))
        self.bpool = nn.Sequential(nn.AvgPool3d(3, stride=1, padding=1, count_include_pad=False),
                                   Conv3dBN(in_ch, 192, 1))

    def forward(self, x):
        b3 = self.b3_1(x)
        b3 = torch.cat([self.b3_2a(b3), self.b3_2b(b3)], 1)
        bd = self.b3dbl_1(x)
        bd = torch.cat([self.b3dbl_2a(bd), self.b3dbl_2b(bd)], 1)
        return torch.cat([self.b1x1(x), b3, bd, self.bpool(x)], 1)


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
        nn.MaxPool3d(2), Conv3dBN(64, 192, 3, padding=1),
    ), 192


class InceptionV3_3D(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 dropout: float = 0.0, stem: str = "standard"):
        super().__init__()
        self.stem, c = (_stem_standard if stem == "standard" else _stem_small)(in_channels)
        self.a = nn.Sequential(InceptionA(c, 32), InceptionA(256, 64), InceptionA(288, 64))
        self.reduction_a = ReductionA(288)                       # -> 768
        self.b = nn.Sequential(InceptionB(768, 128), InceptionB(768, 160),
                               InceptionB(768, 160), InceptionB(768, 192))
        self.reduction_b = ReductionB(768)                       # -> 1280
        self.c = nn.Sequential(InceptionC(1280), InceptionC(2048))
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.a(x)
        x = self.reduction_a(x)
        x = self.b(x)
        x = self.reduction_b(x)
        x = self.c(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

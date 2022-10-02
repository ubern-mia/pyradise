from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A convolution building block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: Tuple[int, int] = (1, 1)
                 ) -> None:
        super().__init__()

        self.convBlk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.1),
            nn.ReLU(inplace=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.convBlk(inputs)
        return x


class InConv(nn.Module):
    """An in-convolution block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: Tuple[int, int] = (1, 1)
                 ) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels,
                              kernel_size, padding)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.conv(inputs)
        return x


class DownsamplingBlock(nn.Module):
    """A downsampling block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: Tuple[int, int] = (1, 1)
                 ) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, kernel_size, padding)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.mpconv(inputs)
        return x


class UpsamplingBlock(nn.Module):
    """An upsampling block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 padding: Tuple[int, int] = (1, 1),
                 bilinear: bool = False
                 ) -> None:
        super().__init__()

        self.bilinear = bilinear

        if not bilinear:
            self.up = nn.ConvTranspose2d(in_channels // 2,
                                         in_channels // 2,
                                         kernel_size=(2, 2),
                                         stride=(2, 2))
        self.conv = ConvBlock(in_channels,
                              out_channels,
                              kernel_size,
                              padding)

    # noinspection DuplicatedCode
    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1,
                               scale_factor=2,
                               mode='bilinear',
                               align_corners=True)
        else:
            x1 = self.up(x1)
        dx = x2.size()[2] - x1.size()[2]
        dy = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1,
                   [dy // 2, x2.size()[3] - (x1.size()[3] + dy // 2),
                    dx // 2, x2.size()[2] - (x1.size()[2] + dx // 2)])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class OutConv(nn.Module):
    """An out-convolution block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """A U-Net model with minor optimizations.

    Reference:
        Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image
        Segmentation. In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) Medical Image Computing and
        Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science,
        vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28
    """

    def __init__(self,
                 num_channels: int,
                 num_classes: int,
                 ) -> None:
        super().__init__()

        self.in1 = InConv(num_channels, 64)
        self.down1 = DownsamplingBlock(64, 128)
        self.down2 = DownsamplingBlock(128, 256)
        self.down3 = DownsamplingBlock(256, 512)
        self.down4 = DownsamplingBlock(512, 512)
        self.up1 = UpsamplingBlock(1024, 256)
        self.up2 = UpsamplingBlock(512, 128)
        self.up3 = UpsamplingBlock(256, 64)
        self.up4 = UpsamplingBlock(128, 64)
        self.out1 = OutConv(64, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.in1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xd1 = self.up1(x5, x4)
        xd2 = self.up2(xd1, x3)
        xd3 = self.up3(xd2, x2)
        xd4 = self.up4(xd3, x1)
        xd5 = self.out1(xd4)

        return xd5

"""PyTorch SVS-Net model.

The network consumes tensors shaped B x 1 x 4 x 512 x 512 and predicts a
single B x 1 x 512 x 512 vessel mask.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _same_padding(size: int, kernel: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    out_size = math.ceil(size / stride)
    pad_needed = max((out_size - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)
    before = pad_needed // 2
    after = pad_needed - before
    return before, after


class SamePadConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, depth, height, width = x.shape
        pd = _same_padding(depth, self.kernel_size[0], self.stride[0])
        ph = _same_padding(height, self.kernel_size[1], self.stride[1])
        pw = _same_padding(width, self.kernel_size[2], self.stride[2])
        x = F.pad(x, (pw[0], pw[1], ph[0], ph[1], pd[0], pd[1]))
        return self.conv(x)


class SamePadConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        ph = _same_padding(height, self.kernel_size[0], self.stride[0])
        pw = _same_padding(width, self.kernel_size[1], self.stride[1])
        x = F.pad(x, (pw[0], pw[1], ph[0], ph[1]))
        return self.conv(x)


class ConvBnRelu3d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
    ) -> None:
        super().__init__(
            SamePadConv3d(in_channels, out_channels, kernel_size, stride=stride, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )


class ConvBnRelu2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__(
            SamePadConv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBlock3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = SamePadConv3d(channels, channels, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = SamePadConv3d(channels, channels, 3, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class ResidualBlock2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = SamePadConv2d(channels, channels, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = SamePadConv2d(channels, channels, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class TemporalProject(nn.Module):
    def __init__(self, channels: int, frame_count: int) -> None:
        super().__init__()
        self.project = nn.Conv3d(channels, channels, (frame_count, 1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        return x.squeeze(2)


class GlobalChannelAttention3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv3d(channels, channels, 1)
        self.fc2 = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.amax(x, dim=(2, 3, 4), keepdim=True)
        weights = F.relu(self.fc1(weights), inplace=True)
        weights = torch.sigmoid(self.fc2(weights))
        return x * weights


class SaliencyAttention3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 1)
        self.conv2 = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.relu(self.conv1(x), inplace=True)
        weights = torch.sigmoid(self.conv2(weights))
        return x + x * weights


class ChannelAttentionSkip2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels * 2, channels, 1)
        self.fc2 = nn.Conv2d(channels, channels, 1)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        weights = torch.cat([low, high], dim=1).mean(dim=(2, 3), keepdim=True)
        weights = F.relu(self.fc1(weights), inplace=True)
        weights = torch.sigmoid(self.fc2(weights))
        return low * weights + high


class SVSNet(nn.Module):
    """Sequential vessel segmentation network implemented in PyTorch."""

    def __init__(self, in_channels: int = 1, frame_count: int = 4) -> None:
        super().__init__()
        self.frame_count = frame_count

        self.conv0 = SamePadConv3d(in_channels, 8, 1)
        self.enc1 = ResidualBlock3d(8)
        self.skip1 = TemporalProject(8, frame_count)
        self.down1 = ConvBnRelu3d(8, 16, (2, 2, 2), stride=(1, 2, 2))

        self.enc2 = ResidualBlock3d(16)
        self.skip2 = TemporalProject(16, frame_count)
        self.down2 = ConvBnRelu3d(16, 32, (2, 2, 2), stride=(1, 2, 2))

        self.enc3 = ResidualBlock3d(32)
        self.skip3 = TemporalProject(32, frame_count)
        self.down3 = ConvBnRelu3d(32, 64, (2, 2, 2), stride=(1, 2, 2))

        self.enc4 = ResidualBlock3d(64)
        self.skip4 = TemporalProject(64, frame_count)
        self.down4 = ConvBnRelu3d(64, 128, (2, 2, 2), stride=(1, 2, 2))

        self.enc5 = ResidualBlock3d(128)
        self.skip5 = TemporalProject(128, frame_count)
        self.drop5 = nn.Dropout3d(0.5)
        self.down5 = ConvBnRelu3d(128, 256, (2, 2, 2), stride=(1, 2, 2))

        self.enc6 = ResidualBlock3d(256)
        self.att6 = GlobalChannelAttention3d(256)
        self.skip6 = TemporalProject(256, frame_count)
        self.drop6 = nn.Dropout3d(0.5)
        self.down6 = ConvBnRelu3d(256, 512, (2, 2, 2), stride=(1, 2, 2))
        self.bottleneck = TemporalProject(512, frame_count)

        self.up1_conv = ConvBnRelu2d(512, 256, 2)
        self.up1_att = ChannelAttentionSkip2d(256)
        self.up1_block = ResidualBlock2d(256)

        self.up2_conv = ConvBnRelu2d(256, 128, 2)
        self.up2_att = ChannelAttentionSkip2d(128)
        self.up2_block = ResidualBlock2d(128)

        self.up3_conv = ConvBnRelu2d(128, 64, 2)
        self.up3_att = ChannelAttentionSkip2d(64)
        self.up3_block = ResidualBlock2d(64)

        self.up4_conv = ConvBnRelu2d(64, 32, 2)
        self.up4_att = ChannelAttentionSkip2d(32)
        self.up4_block = ResidualBlock2d(32)

        self.up5_conv = ConvBnRelu2d(32, 16, 2)
        self.up5_att = ChannelAttentionSkip2d(16)
        self.up5_block = ResidualBlock2d(16)

        self.up6_conv = ConvBnRelu2d(16, 8, 2)
        self.up6_att = ChannelAttentionSkip2d(8)
        self.up6_block = ResidualBlock2d(8)

        self.out = nn.Conv2d(8, 1, 1)

    @staticmethod
    def _upsample(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected B,C,T,H,W input, got shape {tuple(x.shape)}")
        if x.shape[2] != self.frame_count:
            raise ValueError(f"Expected {self.frame_count} frames, got {x.shape[2]}")

        x = self.conv0(x)
        x1 = self.enc1(x)
        s1 = self.skip1(x1)
        x = self.down1(x1)

        x2 = self.enc2(x)
        s2 = self.skip2(x2)
        x = self.down2(x2)

        x3 = self.enc3(x)
        s3 = self.skip3(x3)
        x = self.down3(x3)

        x4 = self.enc4(x)
        s4 = self.skip4(x4)
        x = self.down4(x4)

        x5 = self.enc5(x)
        s5 = self.skip5(x5)
        x = self.down5(self.drop5(x5))

        x6 = self.enc6(x)
        s6 = self.skip6(self.att6(x6))
        x = self.down6(self.drop6(x6))

        x = self.bottleneck(x)

        x = self.up1_conv(self._upsample(x))
        x = self.up1_block(self.up1_att(s6, x))

        x = self.up2_conv(self._upsample(x))
        x = self.up2_block(self.up2_att(s5, x))

        x = self.up3_conv(self._upsample(x))
        x = self.up3_block(self.up3_att(s4, x))

        x = self.up4_conv(self._upsample(x))
        x = self.up4_block(self.up4_att(s3, x))

        x = self.up5_conv(self._upsample(x))
        x = self.up5_block(self.up5_att(s2, x))

        x = self.up6_conv(self._upsample(x))
        x = self.up6_block(self.up6_att(s1, x))

        return torch.sigmoid(self.out(x))

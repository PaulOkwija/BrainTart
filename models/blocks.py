"""Building blocks for the Attention U-Net 3D."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResBlock3D(nn.Module):
    """Two Conv3d + GroupNorm + SiLU with residual connection.

    Strided version used for downsampling (stride=2) instead of MaxPool.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(groups, out_ch), out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.block(x) + self.skip(x))


class AttentionGate3D(nn.Module):
    """Attention gate on skip connections (Oktay et al. 2018).

    g  = gating signal from decoder (coarser)
    x  = skip connection from encoder (finer)
    Returns x gated by a soft spatial attention map.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=False),
            nn.GroupNorm(min(8, F_int), F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=False),
            nn.GroupNorm(min(8, F_int), F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        if g.shape[-3:] != x.shape[-3:]:
            g = F.interpolate(g, size=x.shape[-3:], mode="trilinear", align_corners=False)
        attn = self.psi(self.act(self.W_g(g) + self.W_x(x)))
        return x * attn


class SelfAttention3D(nn.Module):
    """Lightweight self-attention at the bottleneck.

    At depth=3 on a 96^3 input the spatial resolution is 12^3 = 1728 tokens,
    small enough for full attention without windowing.
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(
            channels, heads, batch_first=True, dropout=0.0
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        h = h.flatten(2).permute(0, 2, 1)           # (B, D*H*W, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).reshape(B, C, D, H, W)
        return x + h


class DeepSupHead(nn.Module):
    """1x1x1 conv to produce auxiliary output for deep supervision."""

    def __init__(self, in_ch: int, out_ch: int = 1, target_size: Optional[Tuple] = None):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.act = nn.Tanh()
        self.target_size = target_size

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(self.conv(x))
        if self.target_size is not None:
            out = F.interpolate(out, size=self.target_size, mode="trilinear", align_corners=False)
        return out

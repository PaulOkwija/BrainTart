"""Attention U-Net 3D for BraTS inpainting.

Architecture:
  - Strided-conv downsampling (nnU-Net style)
  - ResBlock3D encoder and bottleneck blocks
  - Asymmetric decoder: single ConvBlock3D per stage  (MonoUNet, Table 5)
  - MonoBlock3D + MonoGate3D: local phase features injected into the first
    k high-resolution encoder stages                  (MonoUNet, §2.2–2.3)
  - Attention gates on all skip connections
  - Self-attention bottleneck
  - Deep supervision at two decoder levels
  - GroupNorm throughout (safe for small batch sizes)

References
----------
- Kimbowa et al., "MonoUNet", arXiv:2604.07780
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .blocks import (
    ResBlock3D, AttentionGate3D, SelfAttention3D, DeepSupHead,
    ConvBlock3D, MonoBlock3D, MonoGate3D,
)


class AttentionUNet3D(nn.Module):
    """
    Forward input  : (B, 2, D, H, W)  — [voided | mask]
    Forward output : (B, 1, D, H, W)  — tanh predicted reconstruction
                     + list of deep-supervision tensors (same spatial size)

    Parameters
    ----------
    in_channels  : number of input channels
    out_channels : number of output channels
    base_ch      : channel count at the first encoder stage
    depth        : number of encoder downsampling stages
    mono_stages  : k — encoder stages to inject local phase into; 0 = disabled
    mono_scales  : M — log-Gabor scales per filter
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_ch: int = 32,
        depth: int = 3,
        mono_stages: int = 3,
        mono_scales: int = 3,
    ):
        super().__init__()
        self.depth = depth

        # Clamp: at most (init_conv + depth enc_blocks) stages exist
        self.n_mono = min(mono_stages, depth + 1)

        enc_chs = [base_ch * (2 ** i) for i in range(depth + 1)]

        # ── MonoUNet: local phase extractor + gated encoder injection ─────────
        if self.n_mono > 0:
            self.mono_block = MonoBlock3D(
                in_channels, k=self.n_mono, M=mono_scales
            )
            self.mono_gates = nn.ModuleList([
                MonoGate3D(phase_ch=self.n_mono, enc_ch=enc_chs[i])
                for i in range(self.n_mono)
            ])
        else:
            self.mono_block = None
            self.mono_gates = nn.ModuleList()

        # ── Initial projection ───────────────────────────────────────────────
        self.init_conv = ResBlock3D(in_channels, enc_chs[0])

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList()
        for i in range(depth):
            self.enc_blocks.append(ResBlock3D(enc_chs[i], enc_chs[i + 1], stride=2))

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ResBlock3D(enc_chs[depth], enc_chs[depth]),
            SelfAttention3D(enc_chs[depth], heads=4),
            ResBlock3D(enc_chs[depth], enc_chs[depth]),
        )

        # ── Decoder ── MonoUNet asymmetric: one ConvBlock3D per stage ─────────
        # Original uses two ResBlock3D (2 conv + residual each).
        # MonoUNet Table 5: halving decoder blocks → no accuracy loss, ~40%
        # parameter reduction in the decoder.
        self.up_convs = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        for i in range(depth - 1, -1, -1):
            F_g   = enc_chs[i + 1]
            F_l   = enc_chs[i]
            F_int = max(F_l // 2, 8)

            self.up_convs.append(
                nn.ConvTranspose3d(F_g, F_g, kernel_size=2, stride=2)
            )
            self.att_gates.append(AttentionGate3D(F_g=F_g, F_l=F_l, F_int=F_int))
            # Single ConvBlock3D replaces ResBlock3D (no residual in decoder)
            self.dec_blocks.append(ConvBlock3D(F_g + F_l, F_l))

        # ── Main output head ─────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(min(8, enc_chs[0]), enc_chs[0])
        self.out_conv = nn.Conv3d(enc_chs[0], out_channels, 1)

        # ── Deep supervision heads ───────────────────────────────────────────
        self.ds_heads = nn.ModuleList()
        for k in range(min(2, depth)):
            ch = enc_chs[depth - 1 - k]
            self.ds_heads.append(DeepSupHead(ch, out_channels))

    def forward(self, x: Tensor):
        target_size = x.shape[-3:]

        # ── Local phase features (computed once; reused across all gates) ─────
        # MonoBlock3D runs inside autocast-safe float32 internally.
        phase = self.mono_block(x) if self.mono_block is not None else None

        # ── Encoder ──────────────────────────────────────────────────────────
        x0 = self.init_conv(x)
        if phase is not None and 0 < self.n_mono:
            x0 = self.mono_gates[0](phase, x0)          # inject into stage 0
        skips = [x0]
        xi = x0

        for i, enc in enumerate(self.enc_blocks):
            xi = enc(xi)
            gate_idx = i + 1
            if phase is not None and gate_idx < self.n_mono:
                xi = self.mono_gates[gate_idx](phase, xi)  # inject into stages 1..k-1
            skips.append(xi)

        # ── Bottleneck ───────────────────────────────────────────────────────
        xi = self.bottleneck(skips[-1])

        # ── Decoder ──────────────────────────────────────────────────────────
        ds_outputs = []
        for k, (up, ag, dec) in enumerate(
            zip(self.up_convs, self.att_gates, self.dec_blocks)
        ):
            skip_idx = self.depth - 1 - k
            xi = up(xi)
            sk = ag(g=xi, x=skips[skip_idx])
            xi = dec(torch.cat([xi, sk], dim=1))

            if k < len(self.ds_heads):
                ds_out = self.ds_heads[k](xi)
                ds_out = F.interpolate(
                    ds_out, size=target_size, mode="trilinear", align_corners=False
                )
                ds_outputs.append(ds_out)

        out = torch.tanh(self.out_conv(F.silu(self.out_norm(xi))))
        return out, ds_outputs

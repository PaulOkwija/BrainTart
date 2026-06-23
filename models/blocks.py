"""Building blocks for the Attention U-Net 3D.

Includes MonoUNet-inspired blocks (Kimbowa et al., arXiv:2604.07780):
  - ConvBlock3D      : asymmetric decoder block (single conv, no residual)
  - MonoBlock3D      : 3-D log-Gabor + Riesz local phase extractor
  - MonoGate3D       : gated injection of phase features into encoder stages
"""

import math
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


# ── MonoUNet-inspired blocks ─────────────────────────────────────────────────
# Reference: Kimbowa et al., "MonoUNet", arXiv:2604.07780


class ConvBlock3D(nn.Module):
    """Single-conv decoder block — MonoUNet asymmetric decoder design.

    Replaces ``ResBlock3D`` in the decoder with one Conv3d + GroupNorm + SiLU
    (no residual path).  MonoUNet's ablation (Table 5) shows this halves
    decoder parameters with **no accuracy loss**.
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MonoBlock3D(nn.Module):
    """Trainable 3-D monogenic filter bank for local phase feature extraction.

    Adapts the Mono block from MonoUNet (Kimbowa et al., arXiv:2604.07780)
    to 3-D volumes using the 3-D monogenic signal:

        1. For each of *k* injection stages, a log-Gabor bandpass filter (LGF)
           is applied at *M* geometric scales in the frequency domain.
        2. The 3-D Riesz transform decomposes each filtered response into one
           even part and three odd parts (depth, height, width directions).
        3. Local phase  θ = atan2(||odd||, even)  is computed per scale.
        4. A 1×1×1 conv combines all k·M·C phase maps into k output feature
           maps — one per encoder injection stage.

    Local phase is structurally invariant to intensity rescaling, making it
    robust to inter-scanner and inter-subject MRI intensity variations.

    Parameters
    ----------
    in_channels : C  — input image channels
    k           : number of encoder stages to inject phase into
    M           : log-Gabor scales per filter  (paper default: 3)
    """

    def __init__(self, in_channels: int, k: int = 3, M: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.k = k
        self.M = M

        # Learnable LGF parameters — one set per injection stage.
        # Initialised to standard log-Gabor defaults from the literature:
        #   omega0 ≈ 0.4 (mid-band), sigma_r ≈ 0.65, r ≈ 2.0
        self.log_omega0  = nn.Parameter(torch.full((k,), math.log(0.4)))
        self.log_sigma_r = nn.Parameter(torch.full((k,), math.log(0.65)))
        self.log_r       = nn.Parameter(torch.full((k,), math.log(2.0)))

        # 1×1×1 conv: k·M phase maps per input channel → k output features
        self.combine = nn.Conv3d(k * M * in_channels, k, 1, bias=True)
        nn.init.kaiming_normal_(self.combine.weight, nonlinearity='linear')
        nn.init.zeros_(self.combine.bias)

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _freq_grid(D: int, H: int, W: int, device):
        """3-D frequency grid compatible with rfftn output shape.

        Returns
        -------
        freq_mag : (D, H, W//2+1)  — isotropic frequency magnitude
        r_d, r_h, r_w             — unit-vector components (Riesz filters)
        """
        fd = torch.fft.fftfreq(D, device=device)   # (-0.5 … +0.5)
        fh = torch.fft.fftfreq(H, device=device)
        fw = torch.fft.rfftfreq(W, device=device)  # (0 … +0.5), length W//2+1
        gd, gh, gw = torch.meshgrid(fd, fh, fw, indexing='ij')
        freq_mag = (gd**2 + gh**2 + gw**2).sqrt().clamp(min=1e-8)
        return freq_mag, gd / freq_mag, gh / freq_mag, gw / freq_mag

    def _lgf(self, freq_mag: Tensor, omega0_m: Tensor, sigma_r: Tensor) -> Tensor:
        """Isotropic 3-D log-Gabor filter; DC component forced to zero."""
        H = torch.exp(
            -(torch.log(freq_mag / omega0_m.clamp(min=1e-8)) ** 2)
            / (2.0 * sigma_r ** 2)
        )
        H[0, 0, 0] = 0.0
        return H

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, C, D, H, W)  — input image (any dtype)

        Returns
        -------
        (B, k, D, H, W)  — local phase features at full input resolution
        """
        B, C, D, H, W = x.shape
        dev = x.device

        freq_mag, r_d, r_h, r_w = self._freq_grid(D, H, W, dev)

        # FFT requires float32 — cast explicitly so AMP (float16) is safe.
        x32 = x.float()
        Xf = torch.fft.rfftn(x32, dim=(-3, -2, -1))   # (B, C, D, H, W//2+1)

        phase_maps: list = []

        for ki in range(self.k):
            omega0  = self.log_omega0[ki].exp()
            sigma_r = self.log_sigma_r[ki].exp().clamp(min=0.1)
            r_scale = self.log_r[ki].exp().clamp(min=1.01)      # must be > 1

            for m in range(self.M):
                omega0_m = omega0 * r_scale.pow(-m)
                lgf = self._lgf(freq_mag, omega0_m, sigma_r)    # (D, H, W//2+1)

                # LGF-filtered spectrum (even part of monogenic signal)
                Xe = Xf * lgf[None, None]                       # (B, C, D, H, W//2+1)
                fe = torch.fft.irfftn(Xe, s=(D, H, W), dim=(-3, -2, -1))

                # 3-D Riesz transform: i·(ωk / |ω|) preserves Hermitian symmetry
                # because ωk is antisymmetric and |ω| is symmetric, making the
                # product i·ωk/|ω| anti-Hermitian, so the IFFT is real-valued.
                fo_d = torch.fft.irfftn(Xe * (1j * r_d), s=(D, H, W), dim=(-3, -2, -1))
                fo_h = torch.fft.irfftn(Xe * (1j * r_h), s=(D, H, W), dim=(-3, -2, -1))
                fo_w = torch.fft.irfftn(Xe * (1j * r_w), s=(D, H, W), dim=(-3, -2, -1))

                # Local phase: angle between structural (even) and
                # directional (odd) components of the monogenic signal
                odd_mag = (fo_d**2 + fo_h**2 + fo_w**2).sqrt()
                phase   = torch.atan2(odd_mag, fe)              # (B, C, D, H, W)
                phase_maps.append(phase)

        # All k·M phase maps per channel: (B, k·M·C, D, H, W)
        all_feats = torch.cat(phase_maps, dim=1)

        # Cast back to input dtype for mixed-precision compatibility
        return self.combine(all_feats).to(x.dtype)              # (B, k, D, H, W)


class MonoGate3D(nn.Module):
    """Gated injection of local phase features into an encoder stage.

    Implements the Mono gate from MonoUNet (Kimbowa et al., arXiv:2604.07780)
    extended to 3-D.  Phase features are spatially downsampled (if needed),
    projected to the encoder's channel dimension, and added via a learnable
    per-channel gate weight α:

        out = enc_feat + sigmoid(α) · proj(phase)

    α is zero-initialised so the gate starts as a pure identity — training
    is numerically identical to the no-Mono baseline at step 0.

    Parameters
    ----------
    phase_ch : number of phase feature channels from MonoBlock3D  (= k)
    enc_ch   : channel count of the target encoder stage
    """

    def __init__(self, phase_ch: int, enc_ch: int):
        super().__init__()
        self.proj  = nn.Conv3d(phase_ch, enc_ch, 1, bias=False)
        # Zero-init: gate starts fully closed, opens gradually during training
        self.alpha = nn.Parameter(torch.zeros(1, enc_ch, 1, 1, 1))

    def forward(self, phase: Tensor, enc_feat: Tensor) -> Tensor:
        """
        Parameters
        ----------
        phase    : (B, phase_ch, D_full, H_full, W_full)
        enc_feat : (B, enc_ch,  D,      H,      W)
        """
        if phase.shape[-3:] != enc_feat.shape[-3:]:
            phase = F.adaptive_avg_pool3d(phase, enc_feat.shape[-3:])
        return enc_feat + torch.sigmoid(self.alpha) * self.proj(phase)

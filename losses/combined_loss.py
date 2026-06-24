"""Loss functions for BraTS inpainting: masked L1 + 3-D SSIM + edge + frequency + deep supervision.

v2 additions:
  - edge_loss_3d   : 3-D Sobel gradient matching at tumor boundaries
  - freq_loss_3d   : L1 on FFT magnitude for global coherence
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn


def ssim_loss_3d(pred: Tensor, target: Tensor, data_range: float = 2.0) -> Tensor:
    """1 - SSIM computed on 3-D volumes in [-1, 1] range."""
    return 1.0 - ssim_fn(
        pred + 1.0, target + 1.0,  # shift to [0, 2]
        data_range=data_range,
    )


# -- 3-D Sobel filters -------------------------------------------------------

def _sobel_kernels_3d() -> Tuple[Tensor, Tensor, Tensor]:
    """Construct 3×3×3 Sobel gradient kernels for D, H, W axes.

    These are the natural 3-D extensions of the 2-D Sobel operator:
    smooth in two axes, differentiate in the third.
    """
    s = torch.tensor([1., 2., 1.])
    d = torch.tensor([-1., 0., 1.])

    # Outer products: smooth ⊗ smooth ⊗ diff  (for each axis)
    # D-gradient
    kd = torch.einsum("i,j,k->ijk", d, s, s)
    # H-gradient
    kh = torch.einsum("i,j,k->ijk", s, d, s)
    # W-gradient
    kw = torch.einsum("i,j,k->ijk", s, s, d)

    # Shape for Conv3d: (out_ch, in_ch/groups, D, H, W)
    return (
        kd.unsqueeze(0).unsqueeze(0),
        kh.unsqueeze(0).unsqueeze(0),
        kw.unsqueeze(0).unsqueeze(0),
    )


def edge_loss_3d(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """L1 loss on 3-D Sobel gradients, computed only inside the mask region.

    Encourages the model to produce sharp, accurate tumor boundaries rather
    than blurry transitions at the inpainting border.

    Parameters
    ----------
    pred, target : (B, 1, D, H, W) in [-1, 1]
    mask         : (B, 1, D, H, W) boolean or float
    """
    kd, kh, kw = _sobel_kernels_3d()
    dev, dtype = pred.device, pred.dtype
    kd, kh, kw = kd.to(dev, dtype), kh.to(dev, dtype), kw.to(dev, dtype)

    # Compute gradients via depthwise conv (groups=1 since C=1)
    grad_pred = torch.cat([
        F.conv3d(pred, kd, padding=1),
        F.conv3d(pred, kh, padding=1),
        F.conv3d(pred, kw, padding=1),
    ], dim=1)  # (B, 3, D, H, W)

    grad_target = torch.cat([
        F.conv3d(target, kd, padding=1),
        F.conv3d(target, kh, padding=1),
        F.conv3d(target, kw, padding=1),
    ], dim=1)

    # Expand mask to 3 gradient channels
    m = mask.bool().expand_as(grad_pred)
    if m.any():
        return F.l1_loss(grad_pred[m], grad_target[m])
    return F.l1_loss(grad_pred, grad_target)


def freq_loss_3d(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss on FFT magnitude spectra for global coherence.

    Cheap frequency-domain consistency term - ensures the prediction
    preserves the global texture/structure of the target volume.

    Parameters
    ----------
    pred, target : (B, 1, D, H, W) in [-1, 1]
    """
    # rfftn is computed in float32 for numerical stability under AMP
    pred_fft = torch.fft.rfftn(pred.float(), dim=(-3, -2, -1))
    target_fft = torch.fft.rfftn(target.float(), dim=(-3, -2, -1))

    pred_mag = pred_fft.abs()
    target_mag = target_fft.abs()

    # Normalise by number of elements to keep the loss scale-invariant
    # with respect to spatial resolution
    loss = F.l1_loss(pred_mag, target_mag)
    return loss.to(pred.dtype)


def combined_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    ds_preds: List[Tensor],
    lam_l1: float = 1.0,
    lam_ssim: float = 1.0,
    lam_edge: float = 0.0,
    lam_freq: float = 0.0,
    lam_ds: Tuple[float, float] = (0.5, 0.25),
) -> Tuple[Tensor, dict]:
    """
    L = lam_l1   * L1_masked
      + lam_ssim * (1 - SSIM_3D)
      + lam_edge * EdgeLoss_masked      [v2]
      + lam_freq * FreqLoss             [v2]
      + lam_ds[0] * L_ds0
      + lam_ds[1] * L_ds1

    L1 and DS losses are computed inside the mask, with a small 0.1x penalty
    outside the mask to enforce seamless boundary blending.
    SSIM is computed over the full patch (standard practice).
    Edge loss uses 3-D Sobel gradients inside the mask.
    Frequency loss compares FFT magnitude spectra globally.
    """
    m = mask.bool()
    if m.any():
        l1_hole = F.l1_loss(pred[m], target[m])
        l1_bg = F.l1_loss(pred[~m], target[~m]) if (~m).any() else 0.0
        l1 = l1_hole + 0.1 * l1_bg
    else:
        l1 = F.l1_loss(pred, target)

    ls = ssim_loss_3d(pred, target)
    total = lam_l1 * l1 + lam_ssim * ls

    log = {"l1": l1.item(), "ssim_loss": ls.item()}

    # -- v2: edge loss --------------------------------------------------------
    if lam_edge > 0.0:
        le = edge_loss_3d(pred, target, mask)
        total = total + lam_edge * le
        log["edge_loss"] = le.item()

    # -- v2: frequency loss ---------------------------------------------------
    if lam_freq > 0.0:
        lf = freq_loss_3d(pred, target)
        total = total + lam_freq * lf
        log["freq_loss"] = lf.item()

    for i, (ds_pred, w) in enumerate(zip(ds_preds, lam_ds)):
        if m.any():
            ds_l_hole = F.l1_loss(ds_pred[m], target[m])
            ds_l_bg = F.l1_loss(ds_pred[~m], target[~m]) if (~m).any() else 0.0
            ds_l = ds_l_hole + 0.1 * ds_l_bg
        else:
            ds_l = F.l1_loss(ds_pred, target)
        total = total + w * ds_l
        log[f"ds{i}"] = ds_l.item()

    log["total"] = total.item()
    return total, log

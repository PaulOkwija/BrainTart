"""Loss functions for BraTS inpainting: masked L1 + 3-D SSIM + deep supervision."""

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


def combined_loss(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    ds_preds: List[Tensor],
    lam_l1: float = 1.0,
    lam_ssim: float = 1.0,
    lam_ds: Tuple[float, float] = (0.5, 0.25),
) -> Tuple[Tensor, dict]:
    """
    L = lam_l1  * L1_masked
      + lam_ssim * (1 - SSIM_3D)
      + lam_ds[0] * L_ds0
      + lam_ds[1] * L_ds1

    L1 and DS losses are computed only inside the mask region.
    SSIM is computed over the full patch (standard practice).
    """
    m = mask.bool()
    if m.any():
        l1 = F.l1_loss(pred[m], target[m])
    else:
        l1 = F.l1_loss(pred, target)

    ls = ssim_loss_3d(pred, target)
    total = lam_l1 * l1 + lam_ssim * ls

    log = {"l1": l1.item(), "ssim_loss": ls.item()}

    for i, (ds_pred, w) in enumerate(zip(ds_preds, lam_ds)):
        if m.any():
            ds_l = F.l1_loss(ds_pred[m], target[m])
        else:
            ds_l = F.l1_loss(ds_pred, target)
        total = total + w * ds_l
        log[f"ds{i}"] = ds_l.item()

    log["total"] = total.item()
    return total, log

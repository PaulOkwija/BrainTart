"""Spatial preprocessing utilities for BraTS 3-D volumes.

Mirrors the official baseline_utils.py from the 2026 challenge repo.
"""

import random
from math import floor, ceil
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def compute_bbox(array: np.ndarray, minimum: float = 0) -> Tuple[slice, slice, slice]:
    """Tight bounding box around voxels > minimum."""
    msk = array > minimum
    if not msk.any():
        raise ValueError("Array would reduce to zero size.")
    idx = np.where(msk)
    mins = [ax.min() for ax in idx]
    maxs = [ax.max() + 1 for ax in idx]
    return tuple(slice(mn, mx) for mn, mx in zip(mins, maxs))


def pad3d(
    target: Tuple[int, int, int],
    image: np.ndarray,
    bbox: Optional[Tuple] = None,
) -> Tuple[Tensor, Optional[Tuple]]:
    """Zero-pad image to at least *target* shape; update bbox if provided."""
    t = torch.tensor(image)
    d, h, w = t.shape[-3], t.shape[-2], t.shape[-1]
    pd = max((target[0] - d) / 2, 0)
    ph = max((target[1] - h) / 2, 0)
    pw = max((target[2] - w) / 2, 0)
    padding = (
        int(floor(pw)), int(ceil(pw)),
        int(floor(ph)), int(ceil(ph)),
        int(floor(pd)), int(ceil(pd)),
    )
    t = F.pad(t, padding, value=0)
    if bbox is not None:
        bbox = list(bbox)
        for i, s in enumerate(bbox, 1):
            ps = padding[2 * -i]
            pe = padding[2 * -i + 1]
            bbox[i - 1] = slice(s.start - ps, s.stop + pe)
        bbox = tuple(bbox)
    return t, bbox


def random_crop(
    target: Tuple[int, int, int], *arrs: Tensor
) -> Tuple[Tensor, ...]:
    """Random spatial crop to *target* shape; applied identically to all arrays."""
    sli = [slice(None)] * 3
    for i in range(3):
        excess = max(0, arrs[0].shape[-3 + i] - target[i])
        if excess:
            r = random.randint(0, excess)
            sli[i] = slice(r, r + target[i])
    return tuple(a[..., sli[0], sli[1], sli[2]] for a in arrs)


def normalize(t: Tensor) -> Tensor:
    """Map [0, 1] -> [-1, 1]."""
    return t * 2.0 - 1.0


def denormalize(t: Tensor) -> Tensor:
    """Map [-1, 1] -> [0, 1]."""
    return (t + 1.0) / 2.0


def center_bbox_on_mask(
    mask: np.ndarray, crop_shape: Tuple[int, int, int]
) -> Tuple[slice, slice, slice]:
    """Return a bbox centered on the mask centroid, clamped to valid indices."""
    shape = mask.shape
    min_bb = compute_bbox(mask)
    out = []
    for i, s in enumerate(min_bb):
        d = crop_shape[i] - (s.stop - s.start)
        s_new = slice(s.start - d // 2, s.stop + ceil(d / 2))
        if s_new.start < 0:
            s_new = slice(0, crop_shape[i])
        if s_new.stop > shape[i]:
            s_new = slice(shape[i] - crop_shape[i], shape[i])
        out.append(s_new)
    return tuple(out)

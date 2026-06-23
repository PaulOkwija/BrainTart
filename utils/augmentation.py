"""Data augmentation for 3-D medical volumes (nnU-Net style)."""

import random
from typing import Tuple

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter


def elastic_deform_3d(
    volume: np.ndarray,
    mask: np.ndarray,
    alpha: float = 6.0,
    sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Elastic deformation applied consistently to volume and mask."""
    shape = volume.shape
    dx = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dy = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    dz = gaussian_filter(np.random.randn(*shape), sigma) * alpha
    x, y, z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
        indexing="ij",
    )
    coords = [x + dx, y + dy, z + dz]
    vol_d = map_coordinates(volume, coords, order=1, mode="reflect")
    mask_d = map_coordinates(mask, coords, order=0, mode="reflect")
    return vol_d, mask_d


def gamma_augment(
    volume: np.ndarray, lo: float = 0.7, hi: float = 1.5
) -> np.ndarray:
    """Random gamma intensity augmentation on [0,1] normalised volume."""
    gamma = np.random.uniform(lo, hi)
    return np.power(volume.clip(0.0, 1.0), gamma)


def random_flip_3d(
    volume: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Random mirroring along each axis independently (p=0.5 per axis)."""
    for ax in range(3):
        if random.random() > 0.5:
            volume = np.flip(volume, axis=ax)
            mask = np.flip(mask, axis=ax)
    return volume.copy(), mask.copy()

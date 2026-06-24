"""3-D Datasets for BraTS 2026 inpainting challenge.

Training : t1n.nii.gz  + mask-healthy.nii.gz  - model trains on healthy inpainting.
Inference: t1n-voided.nii.gz + mask.nii.gz     - combined mask, no GT.

Normalisation constant: max(t1n_voided) - matches Synapse evaluation script.

Patch caching
-------------
The deterministic preprocessing steps (NIfTI load → normalise → bbox crop →
pad3d) are cached once per sample as a small .npz file (≈0.3 MB each vs the
≈54 MB raw volume).  Subsequent epochs skip decompression entirely and only
run the fast stochastic steps (random_crop + augmentation) on the cached patch.

Set ``cache_dir=None`` to disable caching (useful for quick experiments or
when disk space is constrained).
"""

import hashlib
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from utils.preprocessing import (
    compute_bbox, pad3d, random_crop, normalize, center_bbox_on_mask,
)
from utils.augmentation import elastic_deform_3d, gamma_augment, random_flip_3d


class BraTSTrainDataset(Dataset):
    """Training dataset following official Dataset_Training conventions."""

    REFERENCE_SHAPE = (240, 240, 155)

    def __init__(
        self,
        root_dir: Path,
        crop_shape: Tuple[int, int, int] = (96, 96, 96),
        center_on_mask: bool = True,
        augment: bool = True,
        cache_dir: Optional[Path] = Path("/kaggle/working/.patch_cache"),
    ):
        self.root_dir = Path(root_dir)
        self.crop_shape = crop_shape
        self.center_on_mask = center_on_mask
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.t1n_paths = sorted(self.root_dir.rglob("**/BraTS-GLI-*-*-t1n.nii.gz"))
        self.mask_h_paths = sorted(self.root_dir.rglob("**/BraTS-GLI-*-*-mask-healthy.nii.gz"))
        assert len(self.t1n_paths) == len(self.mask_h_paths), (
            f"t1n count {len(self.t1n_paths)} != mask-healthy count {len(self.mask_h_paths)}"
        )
        cache_status = str(self.cache_dir) if self.cache_dir else "disabled"
        print(f"[TrainDataset] {len(self.t1n_paths)} cases | augment={augment} | cache={cache_status}")

    def __len__(self) -> int:
        return len(self.t1n_paths)

    # -- Cache helpers --------------------------------------------------------

    def _cache_key(self, idx: int) -> str:
        """Stable hash that encodes sample identity + crop settings."""
        raw = f"{self.t1n_paths[idx]}|{self.mask_h_paths[idx]}|{self.crop_shape}|{self.center_on_mask}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, idx: int) -> Path:
        return self.cache_dir / f"{self._cache_key(idx)}.npz"

    def _load_or_build_patch(self, idx: int):
        """Return the deterministic (pre-random_crop) patch, using disk cache when available.

        Cached fields
        -------------
        t1n_padded   : float32 array ≥ crop_shape  (the normalised full t1n)
        mask_padded  : float32 array ≥ crop_shape
        t1n_max      : scalar - used for result reconstruction
        max_v        : scalar - used for result reconstruction
        crop_bbox_*  : the three slice start/stop ints (slices aren't numpy-serialisable)
        """
        if self.cache_dir is not None:
            cp = self._cache_path(idx)
            if cp.exists():
                d = np.load(cp)
                t1n_padded = torch.from_numpy(d["t1n_padded"])
                mask_padded = torch.from_numpy(d["mask_padded"])
                t1n_max = float(d["t1n_max"])
                max_v = float(d["max_v"])
                crop_bbox = tuple(
                    slice(int(d[f"bb{i}s"]), int(d[f"bb{i}e"]))
                    for i in range(3)
                )
                return t1n_padded, mask_padded, t1n_max, max_v, crop_bbox

        # -- Build from raw NIfTI ---------------------------------------------
        t1n = nib.load(self.t1n_paths[idx]).get_fdata().astype(np.float32)
        mask_h = nib.load(self.mask_h_paths[idx]).get_fdata().astype(np.float32)

        if t1n.shape != self.REFERENCE_SHAPE or mask_h.shape != self.REFERENCE_SHAPE:
            raise UserWarning(
                f"Expected {self.REFERENCE_SHAPE}, got t1n={t1n.shape}, mask={mask_h.shape}"
            )

        t1n[t1n < 0] = 0.0
        t1n_max = t1n.max()
        if t1n_max > 0:
            t1n /= t1n_max

        t1n_voided_arr = t1n * (1.0 - mask_h)
        max_v = t1n_voided_arr.max()
        if max_v == 0:
            max_v = t1n_max if t1n_max > 0 else 1.0

        mask_h = mask_h.astype(np.float32)

        if self.center_on_mask:
            bbox = center_bbox_on_mask(mask_h, self.crop_shape)
        else:
            bbox = compute_bbox(t1n)

        t1n_crop = t1n[bbox]
        mask_h_crop = mask_h[bbox]

        # pad3d returns Tensors; cache them as numpy to stay pickle-free
        t1n_padded, crop_bbox = pad3d(self.crop_shape, t1n_crop, bbox)
        mask_padded, _ = pad3d(self.crop_shape, mask_h_crop)

        if self.cache_dir is not None:
            save_kwargs = {
                "t1n_padded": t1n_padded.numpy(),
                "mask_padded": mask_padded.numpy(),
                "t1n_max": np.float32(t1n_max),
                "max_v": np.float32(max_v),
            }
            # Serialise the three slices as individual start/stop ints
            for i, s in enumerate(crop_bbox):
                save_kwargs[f"bb{i}s"] = np.int32(s.start)
                save_kwargs[f"bb{i}e"] = np.int32(s.stop)
            np.savez_compressed(self._cache_path(idx), **save_kwargs)

        return t1n_padded, mask_padded, t1n_max, max_v, crop_bbox

    # -- Preprocessing --------------------------------------------------------

    def _apply_stochastic(self, t1n_padded, mask_padded, augment: bool):
        """random_crop + augmentation on the already-padded patch tensors."""
        t1n_crop, mask_h_crop = random_crop(self.crop_shape, t1n_padded, mask_padded)

        if augment:
            t1n_crop_np = t1n_crop.numpy() if isinstance(t1n_crop, torch.Tensor) else t1n_crop
            mask_h_crop_np = mask_h_crop.numpy() if isinstance(mask_h_crop, torch.Tensor) else mask_h_crop

            if random.random() > 0.5:
                t1n_crop_np, mask_h_crop_np = elastic_deform_3d(t1n_crop_np, mask_h_crop_np)
            t1n_crop_np = gamma_augment(t1n_crop_np)
            t1n_crop_np, mask_h_crop_np = random_flip_3d(t1n_crop_np, mask_h_crop_np)

            t1n_crop = torch.tensor(t1n_crop_np.copy())
            mask_h_crop = torch.tensor(mask_h_crop_np.copy())

        voided_crop = t1n_crop * (1.0 - mask_h_crop)

        t1n_crop = normalize(t1n_crop).unsqueeze(0)
        voided_crop = normalize(voided_crop).unsqueeze(0)
        mask_h_crop = mask_h_crop.unsqueeze(0).bool()
        return voided_crop, t1n_crop, mask_h_crop

    def _preprocess(self, t1n: np.ndarray, mask_h: np.ndarray, augment: bool):
        """Legacy path (used when called with pre-loaded arrays; not used during training)."""
        if t1n.shape != self.REFERENCE_SHAPE or mask_h.shape != self.REFERENCE_SHAPE:
            raise UserWarning(
                f"Expected {self.REFERENCE_SHAPE}, got t1n={t1n.shape}, mask={mask_h.shape}"
            )

        t1n = t1n.copy().astype(np.float32)
        t1n[t1n < 0] = 0.0
        t1n_max = t1n.max()
        if t1n_max > 0:
            t1n /= t1n_max

        t1n_voided_arr = t1n * (1.0 - mask_h)
        max_v = t1n_voided_arr.max()
        if max_v == 0:
            max_v = t1n_max if t1n_max > 0 else 1.0

        mask_h = mask_h.astype(np.float32)

        if self.center_on_mask:
            bbox = center_bbox_on_mask(mask_h, self.crop_shape)
        else:
            bbox = compute_bbox(t1n)

        t1n_crop = t1n[bbox]
        mask_h_crop = mask_h[bbox]

        t1n_padded, crop_bbox = pad3d(self.crop_shape, t1n_crop, bbox)
        mask_padded, _ = pad3d(self.crop_shape, mask_h_crop)

        voided_crop, t1n_crop, mask_h_crop = self._apply_stochastic(t1n_padded, mask_padded, augment)
        return voided_crop, t1n_max, max_v, crop_bbox, t1n_crop, mask_h_crop

    def __getitem__(self, idx: int) -> dict:
        mask_path = self.mask_h_paths[idx]

        # Fast path: load tiny cached patch; skip NIfTI decompression
        t1n_padded, mask_padded, t1n_max, max_v, crop_bbox = self._load_or_build_patch(idx)

        # Stochastic steps run every call (random_crop + optional augmentation)
        voided_crop, t1n_crop, mask_h_crop = self._apply_stochastic(
            t1n_padded, mask_padded, self.augment
        )

        return {
            "gt_image": t1n_crop,
            "voided_healthy_image": voided_crop,
            "healthy_mask": mask_h_crop,
            "cropped_bbox": str(crop_bbox),
            "max_v": float(max_v),
            "t1n_max": float(t1n_max),
            "name": mask_path.name[:19],
            "t1n_path": str(self.t1n_paths[idx]),
            "healthy_mask_path": str(mask_path),
        }

    @staticmethod
    def get_result_image(prediction: np.ndarray, sample: dict):
        """Reverse normalisation and crop; paste prediction into full volume."""
        import scipy.ndimage as ndimage
        
        t1n_img = nib.load(sample["t1n_path"])
        affine = t1n_img.affine
        t1n = t1n_img.get_fdata().astype(np.float32)
        mask_h = nib.load(sample["healthy_mask_path"]).get_fdata().astype(np.float32)
        voided_full = t1n * (1.0 - mask_h)

        pred = prediction[0]
        pred = (pred + 1.0) / 2.0
        pred = pred * sample["max_v"]

        mask_crop = sample["healthy_mask"][0]
        if isinstance(mask_crop, torch.Tensor):
            mask_crop = mask_crop.numpy()
            
        # Create a feathered weight mask for smooth boundary blending
        W = mask_crop.astype(np.float32)
        W_blurred = ndimage.gaussian_filter(W, sigma=3.0)
        W_blend = np.maximum(W, W_blurred)

        bb = eval(sample["cropped_bbox"])
        voided_crop = voided_full[bb]
        
        # Blend the prediction with the healthy tissue to hide sharp square edges
        pred_blended = W_blend * pred + (1.0 - W_blend) * voided_crop

        result = voided_full.copy()
        result[bb] = pred_blended
        
        img = nib.Nifti1Image(result, affine=affine, header=t1n_img.header)
        return result, img


class BraTSInferDataset(Dataset):
    """Inference dataset - no GT, uses t1n-voided + mask."""

    REFERENCE_SHAPE = (240, 240, 155)

    def __init__(
        self,
        root_dir: Path,
        crop_shape: Tuple[int, int, int] = (96, 96, 96),
        center_on_mask: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.crop_shape = crop_shape
        self.center_on_mask = center_on_mask

        self.voided_paths = sorted(self.root_dir.rglob("**/BraTS-GLI-*-*-t1n-voided.nii.gz"))
        
        self.mask_paths = []
        for v in self.voided_paths:
            case_id = v.name.replace("-t1n-voided.nii.gz", "")
            mask_matches = list(self.root_dir.rglob(f"**/{case_id}-mask.nii.gz"))
            mask_matches = [p for p in mask_matches if "healthy" not in p.name and "unhealthy" not in p.name]
            if not mask_matches:
                raise ValueError(f"Missing mask for {v}")
            self.mask_paths.append(mask_matches[0])
            
        print(f"[InferDataset] {len(self.voided_paths)} cases")

    def __len__(self) -> int:
        return len(self.voided_paths)

    def __getitem__(self, idx: int) -> dict:
        voided_path = self.voided_paths[idx]
        mask_path = self.mask_paths[idx]

        voided_img = nib.load(voided_path)
        voided = voided_img.get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.float32)

        if voided.shape != self.REFERENCE_SHAPE:
            raise UserWarning(f"Unexpected shape {voided.shape} for {voided_path}")

        voided[voided < 0] = 0.0
        max_v = voided.max()
        if max_v > 0:
            voided /= max_v

        if self.center_on_mask:
            bbox = center_bbox_on_mask(mask, self.crop_shape)
        else:
            bbox = compute_bbox(voided)

        voided_crop = voided[bbox]
        mask_crop = mask[bbox]

        voided_crop, crop_bbox = pad3d(self.crop_shape, voided_crop, bbox)
        mask_crop, _ = pad3d(self.crop_shape, mask_crop)

        # Deterministic trim for inference (no random_crop)
        if any(voided_crop.shape[-3 + i] > self.crop_shape[i] for i in range(3)):
            voided_crop = voided_crop[..., :self.crop_shape[0], :self.crop_shape[1], :self.crop_shape[2]]
            mask_crop = mask_crop[..., :self.crop_shape[0], :self.crop_shape[1], :self.crop_shape[2]]

        # Refine mask to the exact missing region (in case the provided mask is a bounding box)
        exact_mask_crop = (voided_crop == 0.0) & (mask_crop > 0.5)

        voided_crop = normalize(voided_crop).unsqueeze(0)
        mask_crop = exact_mask_crop.unsqueeze(0).bool()

        return {
            "voided_image": voided_crop,
            "mask": mask_crop,
            "cropped_bbox": str(crop_bbox),
            "max_v": float(max_v),
            "name": mask_path.name[:19],
            "t1n_voided_path": str(voided_path),
        }

    @staticmethod
    def get_result_image(prediction: np.ndarray, sample: dict):
        """Paste prediction into full-resolution voided volume with boundary blending."""
        import scipy.ndimage as ndimage
        
        voided_img = nib.load(sample["t1n_voided_path"])
        affine = voided_img.affine
        voided = voided_img.get_fdata().astype(np.float32)

        pred = prediction[0]
        pred = (pred + 1.0) / 2.0
        pred = pred * sample["max_v"]

        mask_crop = sample["mask"][0]
        if isinstance(mask_crop, torch.Tensor):
            mask_crop = mask_crop.numpy()

        # Create a feathered weight mask for smooth boundary blending
        W = mask_crop.astype(np.float32)
        W_blurred = ndimage.gaussian_filter(W, sigma=3.0)
        W_blend = np.maximum(W, W_blurred)

        bb = eval(sample["cropped_bbox"])
        voided_crop = voided[bb]

        # Blend the prediction with the healthy tissue to hide sharp square edges
        pred_blended = W_blend * pred + (1.0 - W_blend) * voided_crop

        result = voided.copy()
        result[bb] = pred_blended
        
        img = nib.Nifti1Image(result, affine=affine, header=voided_img.header)
        return result, img

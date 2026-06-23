"""3-D Datasets for BraTS 2026 inpainting challenge.

Training : t1n.nii.gz  + mask-healthy.nii.gz  — model trains on healthy inpainting.
Inference: t1n-voided.nii.gz + mask.nii.gz     — combined mask, no GT.

Normalisation constant: max(t1n_voided) — matches Synapse evaluation script.
"""

import random
from pathlib import Path
from typing import Tuple

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
    ):
        self.root_dir = Path(root_dir)
        self.crop_shape = crop_shape
        self.center_on_mask = center_on_mask
        self.augment = augment

        self.t1n_paths = sorted(self.root_dir.rglob("**/BraTS-GLI-*-*-t1n.nii.gz"))
        self.mask_h_paths = sorted(self.root_dir.rglob("**/BraTS-GLI-*-*-mask-healthy.nii.gz"))
        assert len(self.t1n_paths) == len(self.mask_h_paths), (
            f"t1n count {len(self.t1n_paths)} != mask-healthy count {len(self.mask_h_paths)}"
        )
        print(f"[TrainDataset] {len(self.t1n_paths)} cases | augment={augment}")

    def __len__(self) -> int:
        return len(self.t1n_paths)

    def _preprocess(self, t1n: np.ndarray, mask_h: np.ndarray, augment: bool):
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

        t1n_crop, crop_bbox = pad3d(self.crop_shape, t1n_crop, bbox)
        mask_h_crop, _ = pad3d(self.crop_shape, mask_h_crop)

        t1n_crop, mask_h_crop = random_crop(self.crop_shape, t1n_crop, mask_h_crop)

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

        return voided_crop, t1n_max, max_v, crop_bbox, t1n_crop, mask_h_crop

    def __getitem__(self, idx: int) -> dict:
        t1n_path = self.t1n_paths[idx]
        mask_path = self.mask_h_paths[idx]

        t1n = nib.load(t1n_path).get_fdata().astype(np.float32)
        mask_h = nib.load(mask_path).get_fdata().astype(np.float32)

        voided_crop, t1n_max, max_v, crop_bbox, t1n_crop, mask_h_crop = (
            self._preprocess(t1n, mask_h, self.augment)
        )

        return {
            "gt_image": t1n_crop,
            "voided_healthy_image": voided_crop,
            "healthy_mask": mask_h_crop,
            "cropped_bbox": str(crop_bbox),
            "max_v": float(max_v),
            "t1n_max": float(t1n_max),
            "name": mask_path.name[:19],
            "t1n_path": str(t1n_path),
            "healthy_mask_path": str(mask_path),
        }

    @staticmethod
    def get_result_image(prediction: np.ndarray, sample: dict):
        """Reverse normalisation and crop; paste prediction into full volume."""
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
        pred_minimal = np.zeros_like(pred)
        pred_minimal[mask_crop] = pred[mask_crop]

        bb = eval(sample["cropped_bbox"])
        pred_full = np.zeros_like(voided_full)
        pred_full[bb] = pred_minimal

        result = voided_full + pred_full
        img = nib.Nifti1Image(result, affine=affine, header=t1n_img.header)
        return result, img


class BraTSInferDataset(Dataset):
    """Inference dataset — no GT, uses t1n-voided + mask."""

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
        self.mask_paths = sorted(self.root_dir.rglob("**/BraTS-GLI-*-*-mask.nii.gz"))
        self.mask_paths = [
            p for p in self.mask_paths
            if "healthy" not in p.name and "unhealthy" not in p.name
        ]
        assert len(self.voided_paths) == len(self.mask_paths), (
            f"voided={len(self.voided_paths)}, mask={len(self.mask_paths)}"
        )
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

        voided_crop = normalize(voided_crop).unsqueeze(0)
        mask_crop = mask_crop.unsqueeze(0).bool()

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
        """Paste prediction into full-resolution voided volume."""
        voided_img = nib.load(sample["t1n_voided_path"])
        affine = voided_img.affine
        voided = voided_img.get_fdata().astype(np.float32)

        pred = prediction[0]
        pred = (pred + 1.0) / 2.0
        pred = pred * sample["max_v"]

        mask_crop = sample["mask"][0]
        if isinstance(mask_crop, torch.Tensor):
            mask_crop = mask_crop.numpy()

        pred_minimal = np.zeros_like(pred)
        pred_minimal[mask_crop] = pred[mask_crop]

        bb = eval(sample["cropped_bbox"])
        pred_full = np.zeros_like(voided)
        pred_full[bb] = pred_minimal

        result = voided + pred_full
        img = nib.Nifti1Image(result, affine=affine, header=voided_img.header)
        return result, img

"""Visualisation helpers for training diagnostics."""

import random
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocessing import denormalize


@torch.no_grad()
def viz_sample(sample: dict, save_path: Optional[Path] = None):
    """Plot GT / Voided / Mask for a single training sample."""
    gt = denormalize(sample["gt_image"][0]).numpy()
    voided = denormalize(sample["voided_healthy_image"][0]).numpy()
    mask = sample["healthy_mask"][0].float().numpy()

    D = gt.shape[-1]
    slices = [D // 4, D // 2, 3 * D // 4]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    rows = [("GT", gt), ("Voided", voided), ("Mask", mask)]
    for row_idx, (label, vol) in enumerate(rows):
        for col_idx, s in enumerate(slices):
            ax = axes[row_idx][col_idx]
            ax.imshow(vol[:, :, s].T, cmap="gray", origin="lower")
            ax.set_title(f"{label} z={s}", fontsize=9)
            ax.axis("off")
    plt.suptitle(f"Sample: {sample['name']}", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def visualize_epoch(model, val_ds, epoch, n_cases=3, save_dir=None):
    """Plot voided | GT | prediction side-by-side for *n_cases* val samples."""
    model.eval()
    dev = next(model.parameters()).device

    indices = random.sample(range(len(val_ds)), min(n_cases, len(val_ds)))

    fig, axes = plt.subplots(n_cases, 3, figsize=(12, 4 * n_cases), squeeze=False)

    for row, idx in enumerate(indices):
        sample = val_ds[idx]
        voided = sample["voided_healthy_image"].unsqueeze(0).to(dev)
        mask = sample["healthy_mask"].unsqueeze(0).float().to(dev)
        gt = sample["gt_image"]

        m = model.module if hasattr(model, "module") else model
        pred, _ = m(torch.cat([voided, mask], dim=1))
        pred = pred.squeeze(0).cpu()

        mk_np = sample["healthy_mask"][0].numpy()
        best_z = int(mk_np.sum(axis=(0, 1)).argmax())

        gt_sl = denormalize(gt[0, :, :, best_z]).numpy()
        vd_sl = denormalize(voided[0, 0, :, :, best_z].cpu()).numpy()
        pr_sl = denormalize(pred[0, :, :, best_z]).numpy()

        for col, (img, title) in enumerate([
            (vd_sl, "Voided (input)"),
            (gt_sl, "GT"),
            (pr_sl, "Prediction"),
        ]):
            ax = axes[row][col]
            ax.imshow(img.T, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.set_title(f"{title}\n{sample['name']} z={best_z}", fontsize=8)
            ax.axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    if save_dir:
        save_path = Path(save_dir) / f"viz_epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_loss_curve(history: dict, save_path: Path):
    """Plot train/val loss curves."""
    if not history.get("epoch"):
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train", color="steelblue")
    ax.plot(history["epoch"], history["val_loss"], label="val", color="coral")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (L1 + SSIM + DS)")
    ax.set_title("Attention U-Net 3D — Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

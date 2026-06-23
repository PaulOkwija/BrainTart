#!/usr/usr/bin/env python3
"""Visualize BrainTart inference results.

Matches inference results with their original voided inputs and masks,
and saves a side-by-side comparison image slice.

Usage:
    python visualize_inference.py --dataset /path/to/data --results-dir /path/to/results --output-dir /path/to/save/plots
"""

import argparse
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_inference_sample(voided_path: Path, mask_path: Path, pred_path: Path, out_path: Path):
    """Plot Voided | Mask | Prediction side-by-side for the slice with the largest mask area."""
    # Load NIfTI volumes
    voided = nib.load(voided_path).get_fdata().astype(np.float32)
    mask = nib.load(mask_path).get_fdata().astype(np.float32)
    pred = nib.load(pred_path).get_fdata().astype(np.float32)

    # Find the axial slice (Z-axis) with the most mask pixels
    mask_sum_z = mask.sum(axis=(0, 1))
    best_z = int(np.argmax(mask_sum_z))

    if mask_sum_z[best_z] == 0:
        # If mask is entirely empty, just pick the middle slice
        best_z = voided.shape[2] // 2

    vd_sl = voided[:, :, best_z]
    mk_sl = mask[:, :, best_z]
    pr_sl = pred[:, :, best_z]

    # Normalize images for display (0 to 1) based on the 99th percentile to handle outliers
    vmax = np.percentile(voided, 99)
    if vmax <= 0: vmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Voided Image
    axes[0].imshow(vd_sl.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    axes[0].set_title(f"Input (Voided) - z={best_z}")
    axes[0].axis("off")

    # 2. Mask overlay (Voided + Red Mask)
    axes[1].imshow(vd_sl.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    # Create a red overlay for the mask
    mask_overlay = np.zeros((*mk_sl.shape, 4))
    mask_overlay[..., 0] = 1.0  # Red channel
    mask_overlay[..., 3] = mk_sl.T * 0.5  # Alpha channel
    axes[1].imshow(mask_overlay, origin="lower")
    axes[1].set_title(f"Mask Region")
    axes[1].axis("off")

    # 3. Prediction
    axes[2].imshow(pr_sl.T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
    axes[2].set_title(f"Reconstruction (Prediction)")
    axes[2].axis("off")

    case_name = pred_path.name.replace("-t1n-inference.nii.gz", "")
    plt.suptitle(f"Case: {case_name}", fontsize=14)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize BrainTart Inference Results")
    parser.add_argument("--dataset", type=str, required=True, help="Path to original dataset (with voided and mask files)")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to directory containing inference .nii.gz files")
    parser.add_argument("--output-dir", type=str, default="inference_viz", help="Directory to save the visualization PNGs")
    parser.add_argument("--n-cases", type=int, default=10, help="Number of random cases to visualize (0 for all)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all inference results
    pred_files = list(results_dir.glob("*-t1n-inference.nii.gz"))
    if not pred_files:
        print(f"No inference files found in {results_dir}")
        return

    print(f"Found {len(pred_files)} inference files.")

    # Optionally sample a subset
    if args.n_cases > 0 and args.n_cases < len(pred_files):
        pred_files = random.sample(pred_files, args.n_cases)
        print(f"Randomly selected {args.n_cases} cases for visualization.")

    missing = 0
    for pred_path in tqdm(pred_files, desc="Generating Visualizations"):
        case_name = pred_path.name.replace("-t1n-inference.nii.gz", "")
        
        # Locate corresponding voided and mask files in the dataset
        # They usually look like BraTS-GLI-XXXXX-YYY-t1n-voided.nii.gz
        voided_matches = list(dataset_dir.rglob(f"{case_name}-t1n-voided.nii.gz"))
        mask_matches = list(dataset_dir.rglob(f"{case_name}-mask.nii.gz"))

        # Filter out 'unhealthy' masks if they exist in the same dir, we want the standard healthy/voided mask
        mask_matches = [p for p in mask_matches if "healthy" not in p.name and "unhealthy" not in p.name]

        if not voided_matches or not mask_matches:
            missing += 1
            continue

        voided_path = voided_matches[0]
        mask_path = mask_matches[0]

        out_img_path = out_dir / f"{case_name}_viz.png"
        plot_inference_sample(voided_path, mask_path, pred_path, out_img_path)

    if missing > 0:
        print(f"Warning: Could not find original dataset files for {missing} inference results.")
        
    print(f"\nDone! Visualizations saved to: {out_dir.absolute()}")


if __name__ == "__main__":
    main()

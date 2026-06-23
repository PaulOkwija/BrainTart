#!/usr/bin/env python3
"""BrainTart — Local evaluation mirroring the Synapse server script.

Metrics: SSIM, PSNR, MSE  (via inpainting.challenge_metrics_2023.generate_metrics)

Usage:
    python evaluate.py --dataset /path/to/data --results /path/to/results
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inpainting.challenge_metrics_2023 import generate_metrics


def local_evaluate(
    dataset_path: Path,
    results_dir: Path,
    max_cases: Optional[int] = None,
) -> dict:
    """Reproduce Synapse server evaluation using mask-healthy + max(t1n_voided)."""
    result_files = sorted(results_dir.glob("BraTS-GLI-*-*-t1n-inference.nii.gz"))
    if max_cases:
        result_files = result_files[:max_cases]

    perf = {"ssim": [], "psnr": [], "mse": []}
    skipped = 0

    for rf in tqdm(result_files, desc="Evaluating"):
        case_id = rf.name[:19]
        case_dir = dataset_path / case_id

        t1n_path = case_dir / f"{case_id}-t1n.nii.gz"
        mask_healthy_path = case_dir / f"{case_id}-mask-healthy.nii.gz"
        t1n_voided_path = case_dir / f"{case_id}-t1n-voided.nii.gz"

        if not t1n_path.exists():
            skipped += 1
            continue

        pred = torch.tensor(nib.load(rf).get_fdata().astype(np.float32)).unsqueeze(0)
        gt = torch.tensor(nib.load(t1n_path).get_fdata().astype(np.float32)).unsqueeze(0)
        mask_h = torch.tensor(nib.load(mask_healthy_path).get_fdata().astype(np.float32)).bool().unsqueeze(0)
        t1n_voided = torch.tensor(nib.load(t1n_voided_path).get_fdata().astype(np.float32)).unsqueeze(0)

        metrics = generate_metrics(
            prediction=pred, target=gt, mask=mask_h,
            normalization_tensor=t1n_voided,
        )
        for k in perf:
            perf[k].append(metrics[k])

    n = len(perf["ssim"])
    print(f"\nEvaluated {n} cases (skipped {skipped} without GT)")
    if n > 0:
        print(f"  SSIM  median={np.median(perf['ssim']):.4f}  "
              f"[IQR {np.percentile(perf['ssim'], 25):.4f} - "
              f"{np.percentile(perf['ssim'], 75):.4f}]")
        print(f"  PSNR  median={np.median(perf['psnr']):.2f} dB")
        print(f"  MSE   median={np.median(perf['mse']):.6f}")
    return perf


def submission_checklist(results_dir: Path):
    """Run pre-submission validation checks."""
    print("=" * 55)
    print("SUBMISSION CHECKLIST")
    print("=" * 55)

    result_files = sorted(results_dir.glob("BraTS-GLI-*-*-t1n-inference.nii.gz"))
    print(f"1. Output files found      : {len(result_files)}")

    shape_ok = True
    name_ok = True
    for rf in result_files:
        img = nib.load(rf)
        if img.shape != (240, 240, 155):
            print(f"   SHAPE ERROR: {rf.name} -> {img.shape}")
            shape_ok = False
        if not rf.name.endswith("-t1n-inference.nii.gz"):
            print(f"   NAME ERROR : {rf.name}")
            name_ok = False

    print(f"2. Shape (240,240,155) OK  : {shape_ok}")
    print(f"3. Naming convention OK    : {name_ok}")
    if result_files:
        print(f"4. Sample filename         : {result_files[0].name}")
    print(f"5. Results folder          : {results_dir}")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description="BrainTart local evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--checklist", action="store_true", help="Run submission checklist only")
    args = parser.parse_args()

    results_dir = Path(args.results)

    if args.checklist:
        submission_checklist(results_dir)
    else:
        local_evaluate(Path(args.dataset), results_dir, args.max_cases)
        submission_checklist(results_dir)


if __name__ == "__main__":
    main()

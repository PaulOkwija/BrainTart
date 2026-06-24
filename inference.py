#!/usr/bin/env python3
"""BrainTart - Inference & submission generation.

v2 additions:
  - MC-Dropout ensemble inference (--mc-samples N)
  - Test-time flip augmentation   (--tta)
  - Uncertainty map export        (--save-uncertainty)

Output format: BraTS-GLI-XXXXX-YYY-t1n-inference.nii.gz  (240x240x155)

Usage:
    # Standard inference (v1-compatible)
    python inference.py --dataset /path/to/data --checkpoint checkpoints/best_model.pt

    # MC-Dropout ensemble + TTA
    python inference.py --dataset /path/to/data --checkpoint checkpoints/best_model.pt \
        --mc-samples 8 --tta --save-uncertainty
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import Config
from models import AttentionUNet3D
from data import BraTSInferDataset


def load_model(checkpoint_path: Path, cfg: Config, device) -> AttentionUNet3D:
    """Load a trained AttentionUNet3D from checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    
    use_trilinear = cfg.USE_TRILINEAR_UPSAMPLE
    if isinstance(state, dict) and "config" in state:
        use_trilinear = state["config"].get("USE_TRILINEAR_UPSAMPLE", False)
    elif isinstance(state, dict) and "model" in state:
        # Older model without config key saved
        use_trilinear = False

    model = AttentionUNet3D(
        in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS,
        base_ch=cfg.BASE_CHANNELS, depth=cfg.DEPTH,
        mono_stages=cfg.MONO_STAGES, mono_scales=cfg.MONO_SCALES,
        dropout_rate=cfg.DROPOUT_RATE,
        use_trilinear_upsample=use_trilinear,
    ).to(device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {checkpoint_path} (Trilinear Upsample: {use_trilinear})")
    return model


@torch.no_grad()
def run_inference(
    model: AttentionUNet3D,
    dataset_path: Path,
    results_dir: Path,
    device,
    crop_shape,
    mc_samples: int = 0,
    use_tta: bool = False,
    save_uncertainty: bool = False,
):
    """Generate one NIfTI per case into results_dir.

    Parameters
    ----------
    mc_samples      : if > 0, run MC-dropout ensemble with this many samples
    use_tta         : if True, include test-time flip augmentation
    save_uncertainty: if True, save per-voxel std maps alongside predictions
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    if save_uncertainty:
        unc_dir = results_dir / "uncertainty"
        unc_dir.mkdir(parents=True, exist_ok=True)

    infer_ds = BraTSInferDataset(dataset_path, crop_shape=crop_shape, center_on_mask=True)
    loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    use_mc = mc_samples > 0 and model.dropout_rate > 0.0
    mode_str = f"MC-Dropout (n={mc_samples}, TTA={use_tta})" if use_mc else "Standard"
    print(f"Running inference on {len(infer_ds)} cases - Mode: {mode_str}")

    for batch in tqdm(loader, desc="Inference"):
        voided = batch["voided_image"].to(device)
        mask = batch["mask"].float().to(device)

        model_in = torch.cat([voided, mask], dim=1)

        if use_mc:
            pred, std_map = model.mc_inference(
                model_in,
                n_samples=mc_samples,
                use_tta_flips=use_tta,
            )
            pred_np = pred.cpu().numpy()[0]
            if save_uncertainty:
                std_np = std_map.cpu().numpy()[0]
        else:
            pred, _ = model(model_in)
            pred_np = pred.cpu().numpy()[0]

        sample_meta = {
            "t1n_voided_path": batch["t1n_voided_path"][0],
            "cropped_bbox": batch["cropped_bbox"][0],
            "max_v": batch["max_v"].item(),
            "mask": batch["mask"].squeeze(0).cpu(),
        }
        result, img = BraTSInferDataset.get_result_image(pred_np, sample_meta)

        assert result.shape == (240, 240, 155), f"Shape mismatch: {result.shape}"

        out_path = results_dir / f"{batch['name'][0]}-t1n-inference.nii.gz"
        nib.save(img, out_path)

        # Save uncertainty map if requested
        if use_mc and save_uncertainty:
            unc_full = np.zeros_like(result)
            bb = eval(sample_meta["cropped_bbox"])
            mask_crop = sample_meta["mask"][0].numpy()
            std_slice = std_np[0]  # (D, H, W)
            unc_region = np.zeros_like(std_slice)
            unc_region[mask_crop] = std_slice[mask_crop]
            # Pad to full size using bounding box
            pad_shape = tuple(s.stop - s.start for s in bb)
            if unc_region.shape == pad_shape:
                unc_full[bb] = unc_region
            else:
                # Trim to match bounding box if needed
                slices = tuple(slice(0, s.stop - s.start) for s in bb)
                unc_full[bb] = unc_region[slices]

            unc_img = nib.Nifti1Image(unc_full, affine=img.affine, header=img.header)
            unc_path = unc_dir / f"{batch['name'][0]}-uncertainty.nii.gz"
            nib.save(unc_img, unc_path)

    saved = list(results_dir.glob("*-t1n-inference.nii.gz"))
    print(f"Saved {len(saved)} inference files to {results_dir}")
    if save_uncertainty and use_mc:
        unc_saved = list(unc_dir.glob("*-uncertainty.nii.gz"))
        print(f"Saved {len(unc_saved)} uncertainty maps to {unc_dir}")


def main():
    parser = argparse.ArgumentParser(description="BrainTart inference")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/kaggle/working/results")
    parser.add_argument("--crop", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    # v2 arguments
    parser.add_argument("--dropout", type=float, default=0.15, help="Bottleneck dropout rate (must match training)")
    parser.add_argument("--mc-samples", type=int, default=0, help="MC-dropout samples (0 = standard inference)")
    parser.add_argument("--tta", action="store_true", help="Enable test-time flip augmentation")
    parser.add_argument("--save-uncertainty", action="store_true", help="Save voxel-wise uncertainty maps")
    parser.add_argument("--mono-stages", type=int, default=3, help="MonoUNet: encoder stages for local phase")
    parser.add_argument("--mono-scales", type=int, default=3, help="MonoUNet: log-Gabor scales per filter")
    args = parser.parse_args()

    cfg = Config()
    cfg.CROP_SHAPE = tuple(args.crop)
    cfg.BASE_CHANNELS = args.base_ch
    cfg.DEPTH = args.depth
    cfg.RESULTS_DIR = Path(args.results_dir)
    cfg.DROPOUT_RATE = args.dropout
    cfg.MONO_STAGES = args.mono_stages
    cfg.MONO_SCALES = args.mono_scales

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), cfg, device)
    run_inference(
        model, Path(args.dataset), cfg.RESULTS_DIR, device, cfg.CROP_SHAPE,
        mc_samples=args.mc_samples,
        use_tta=args.tta,
        save_uncertainty=args.save_uncertainty,
    )


if __name__ == "__main__":
    main()

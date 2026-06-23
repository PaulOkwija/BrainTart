#!/usr/bin/env python3
"""BrainTart — Inference & submission generation.

Output format: BraTS-GLI-XXXXX-YYY-t1n-inference.nii.gz  (240x240x155)

Usage:
    python inference.py --dataset /path/to/data --checkpoint checkpoints/best_model.pt
"""

import sys
import argparse
from pathlib import Path

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
    model = AttentionUNet3D(
        in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS,
        base_ch=cfg.BASE_CHANNELS, depth=cfg.DEPTH,
    ).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded: {checkpoint_path}")
    return model


@torch.no_grad()
def run_inference(model, dataset_path: Path, results_dir: Path, device, crop_shape):
    """Generate one NIfTI per case into results_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)

    infer_ds = BraTSInferDataset(dataset_path, crop_shape=crop_shape, center_on_mask=True)
    loader = DataLoader(infer_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Running inference on {len(infer_ds)} cases...")

    for batch in tqdm(loader, desc="Inference"):
        voided = batch["voided_image"].to(device)
        mask = batch["mask"].float().to(device)

        model_in = torch.cat([voided, mask], dim=1)
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

    saved = list(results_dir.glob("*-t1n-inference.nii.gz"))
    print(f"Saved {len(saved)} inference files to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="BrainTart inference")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="/kaggle/working/results")
    parser.add_argument("--crop", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    cfg = Config()
    cfg.CROP_SHAPE = tuple(args.crop)
    cfg.BASE_CHANNELS = args.base_ch
    cfg.DEPTH = args.depth
    cfg.RESULTS_DIR = Path(args.results_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), cfg, device)
    run_inference(model, Path(args.dataset), cfg.RESULTS_DIR, device, cfg.CROP_SHAPE)


if __name__ == "__main__":
    main()

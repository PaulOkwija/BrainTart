"""BrainTart default configuration.

All hyperparameters in one place.  Import and override as needed:

    from configs import Config
    cfg = Config()
    cfg.NUM_EPOCHS = 100
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Config:
    # ── Paths ────────────────────────────────────────────────────────────────
    DATASET_ROOT: Path = Path("/kaggle/working/brats_data")
    RESULTS_DIR: Path = Path("/kaggle/working/results")
    CHECKPOINT_DIR: Path = Path("/kaggle/working/checkpoints")
    OUTPUT_DIR: Path = Path("/kaggle/working/outputs")

    # ── Reproducibility ──────────────────────────────────────────────────────
    SEED: int = 2023

    # ── Dataset split ────────────────────────────────────────────────────────
    TRAIN_SPLIT: float = 0.85

    # ── 3-D patch geometry ───────────────────────────────────────────────────
    CROP_SHAPE: Tuple[int, int, int] = (96, 96, 96)
    CENTER_ON_MASK: bool = True

    # ── Model ────────────────────────────────────────────────────────────────
    BASE_CHANNELS: int = 32
    DEPTH: int = 3
    IN_CHANNELS: int = 2   # voided_image + mask
    OUT_CHANNELS: int = 1

    # ── Training ─────────────────────────────────────────────────────────────
    NUM_EPOCHS: int = 60
    BATCH_PER_GPU: int = 2       # 2 GPUs x 2 = effective batch 4
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-5
    GRAD_CLIP: float = 1.0
    SAVE_EVERY: int = 5
    NUM_WORKERS: int = 4

    # ── Loss weights ─────────────────────────────────────────────────────────
    LAMBDA_L1: float = 1.0
    LAMBDA_SSIM: float = 1.0
    LAMBDA_DS1: float = 0.5      # deep supervision — coarser decoder level
    LAMBDA_DS2: float = 0.25     # deep supervision — coarsest decoder level

    # ── Inference ────────────────────────────────────────────────────────────
    INFER_BATCH: int = 1

    def makedirs(self):
        """Create output directories."""
        for d in [self.RESULTS_DIR, self.CHECKPOINT_DIR, self.OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)

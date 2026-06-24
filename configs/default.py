"""BrainTart default configuration.

All hyperparameters in one place.  Import and override as needed:

    from configs import Config
    cfg = Config()
    cfg.NUM_EPOCHS = 100
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # -- Paths ----------------------------------------------------------------
    DATASET_ROOT: Path = Path("/kaggle/working/brats_data")
    RESULTS_DIR: Path = Path("/kaggle/working/results")
    CHECKPOINT_DIR: Path = Path("/kaggle/working/checkpoints")
    OUTPUT_DIR: Path = Path("/kaggle/working/outputs")
    CACHE_DIR: Optional[Path] = Path("/kaggle/working/.patch_cache")  # set to None to disable

    # -- Reproducibility ------------------------------------------------------
    SEED: int = 2023

    # -- Dataset split --------------------------------------------------------
    TRAIN_SPLIT: float = 0.85

    # -- 3-D patch geometry ---------------------------------------------------
    CROP_SHAPE: Tuple[int, int, int] = (96, 96, 96)
    CENTER_ON_MASK: bool = True

    # -- Model ----------------------------------------------------------------
    BASE_CHANNELS: int = 32
    DEPTH: int = 3
    IN_CHANNELS: int = 2   # voided_image + mask
    OUT_CHANNELS: int = 1
    # MonoUNet-inspired options (Kimbowa et al., arXiv:2604.07780)
    MONO_STAGES: int = 3   # k: high-res encoder stages to inject local phase into; 0 = off
    MONO_SCALES: int = 3   # M: log-Gabor scales per filter
    # v2: MC-Dropout
    DROPOUT_RATE: float = 0.15  # spatial dropout probability in the bottleneck

    # -- Training -------------------------------------------------------------
    NUM_EPOCHS: int = 60
    BATCH_PER_GPU: int = 2       # 2 GPUs x 2 = effective batch 4
    LR: float = 2e-4
    WEIGHT_DECAY: float = 1e-5
    GRAD_CLIP: float = 1.0
    SAVE_EVERY: int = 5
    NUM_WORKERS: int = 4
    # v2: warmup + gradient accumulation
    WARMUP_EPOCHS: int = 3       # linear warmup from LR*0.01 → LR
    GRAD_ACCUM_STEPS: int = 2    # effective batch = BATCH_PER_GPU * GRAD_ACCUM_STEPS * n_gpus

    # -- Loss weights ---------------------------------------------------------
    LAMBDA_L1: float = 1.0
    LAMBDA_SSIM: float = 1.0
    LAMBDA_DS1: float = 0.5      # deep supervision - coarser decoder level
    LAMBDA_DS2: float = 0.25     # deep supervision - coarsest decoder level
    # v2: edge + frequency losses
    LAMBDA_EDGE: float = 0.25    # 3-D Sobel edge loss weight
    LAMBDA_FREQ: float = 0.1     # FFT magnitude frequency loss weight

    # -- Inference ------------------------------------------------------------
    INFER_BATCH: int = 1
    # v2: MC-Dropout ensemble
    MC_SAMPLES: int = 8          # number of stochastic forward passes
    USE_TTA_FLIPS: bool = True   # test-time axis flips (x, y, z)

    def makedirs(self):
        """Create output directories."""
        for d in [self.RESULTS_DIR, self.CHECKPOINT_DIR, self.OUTPUT_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        if self.CACHE_DIR is not None:
            self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

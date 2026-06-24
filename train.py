#!/usr/bin/env python3
"""BrainTart - DDP training script for Attention U-Net 3D.

v2 additions:
  - Linear warmup scheduler (WARMUP_EPOCHS) before cosine decay
  - Gradient accumulation (GRAD_ACCUM_STEPS)
  - Edge loss + frequency loss terms
  - MC-Dropout via dropout_rate in the bottleneck

Usage (Kaggle / 2x T4):
    python train.py --dataset /kaggle/working/brats_data --epochs 60

Usage (single GPU):
    python train.py --dataset /path/to/data --epochs 60 --batch 2
"""

import os
import sys
import random
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import Config
from models import AttentionUNet3D
from losses import combined_loss
from data import BraTSTrainDataset
from utils.visualization import visualize_epoch, plot_loss_curve


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_scheduler(optimizer, cfg: Config):
    """Build warmup + cosine decay scheduler.

    v2: Linear warmup from LR*0.01 → LR over WARMUP_EPOCHS, followed by
    cosine annealing over the remaining epochs.  If WARMUP_EPOCHS == 0,
    falls back to pure CosineAnnealingLR (identical to v1).
    """
    if cfg.WARMUP_EPOCHS > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=cfg.WARMUP_EPOCHS,
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.NUM_EPOCHS - cfg.WARMUP_EPOCHS,
            eta_min=cfg.LR * 0.01,
        )
        return optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.WARMUP_EPOCHS],
        )
    else:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.NUM_EPOCHS, eta_min=cfg.LR * 0.01
        )


def train_ddp(rank, world, train_ds, val_ds, cfg: Config, history):
    """DDP training worker. rank=0 handles checkpointing and visualisation."""
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world, rank=rank
    )
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, seed=cfg.SEED
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_PER_GPU, sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_PER_GPU, sampler=val_sampler,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    model = AttentionUNet3D(
        in_channels=cfg.IN_CHANNELS, out_channels=cfg.OUT_CHANNELS,
        base_ch=cfg.BASE_CHANNELS, depth=cfg.DEPTH,
        mono_stages=cfg.MONO_STAGES, mono_scales=cfg.MONO_SCALES,
        dropout_rate=cfg.DROPOUT_RATE,
    ).to(dev)

    if rank == 0:
        print(f"Model parameters: {count_params(model):,}")
        print(f"Dropout rate: {cfg.DROPOUT_RATE}")
        print(f"Warmup epochs: {cfg.WARMUP_EPOCHS}")
        print(f"Gradient accumulation steps: {cfg.GRAD_ACCUM_STEPS}")
        print(f"Effective batch size: {cfg.BATCH_PER_GPU * cfg.GRAD_ACCUM_STEPS * world}")
        print(f"Loss weights - L1:{cfg.LAMBDA_L1} SSIM:{cfg.LAMBDA_SSIM} "
              f"Edge:{cfg.LAMBDA_EDGE} Freq:{cfg.LAMBDA_FREQ}")

    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = build_scheduler(optimizer, cfg)

    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    best_val = float("inf")

    ckpt_files = sorted(cfg.CHECKPOINT_DIR.glob("ckpt_epoch_*.pt"))
    if ckpt_files:
        ckpt = torch.load(ckpt_files[-1], map_location=dev)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        history.update(ckpt["history"])
        start_epoch = ckpt["epoch"] + 1
        best_val = min(history["val_loss"]) if history["val_loss"] else float("inf")
        if rank == 0:
            print(f"Resumed from {ckpt_files[-1].name} (epoch {ckpt['epoch']})")

    accum = cfg.GRAD_ACCUM_STEPS

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss = {"total": 0.0, "l1": 0.0, "ssim_loss": 0.0}
        n_batches = 0

        pbar = tqdm(
            train_loader, desc=f"[GPU{rank}] Ep {epoch}/{cfg.NUM_EPOCHS}",
            leave=False, disable=(rank != 0),
        )

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            voided = batch["voided_healthy_image"].to(dev, non_blocking=True)
            gt = batch["gt_image"].to(dev, non_blocking=True)
            mask = batch["healthy_mask"].float().to(dev, non_blocking=True)

            model_in = torch.cat([voided, mask], dim=1)

            with torch.amp.autocast('cuda'):
                pred, ds_preds = model(model_in)

                loss, logs = combined_loss(
                    pred, gt, mask, ds_preds,
                    lam_l1=cfg.LAMBDA_L1, lam_ssim=cfg.LAMBDA_SSIM,
                    lam_edge=cfg.LAMBDA_EDGE, lam_freq=cfg.LAMBDA_FREQ,
                    lam_ds=(cfg.LAMBDA_DS1, cfg.LAMBDA_DS2),
                )
                # Scale loss by accumulation steps so gradient magnitudes
                # are equivalent to a larger effective batch
                loss = loss / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k in epoch_loss:
                epoch_loss[k] += logs.get(k, 0.0)
            n_batches += 1

            if rank == 0:
                pbar.set_postfix(loss=f"{logs['total']:.4f}", l1=f"{logs['l1']:.4f}")

        scheduler.step()

        # Validation (rank-0)
        val_loss = 0.0
        val_batches = 0
        if rank == 0:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    voided = batch["voided_healthy_image"].to(dev)
                    gt = batch["gt_image"].to(dev)
                    mask = batch["healthy_mask"].float().to(dev)
                    model_in = torch.cat([voided, mask], dim=1)

                    with torch.amp.autocast('cuda'):
                        pred, ds_preds = model(model_in)
                        _, logs = combined_loss(
                            pred, gt, mask, ds_preds,
                            lam_l1=cfg.LAMBDA_L1, lam_ssim=cfg.LAMBDA_SSIM,
                            lam_edge=cfg.LAMBDA_EDGE, lam_freq=cfg.LAMBDA_FREQ,
                            lam_ds=(cfg.LAMBDA_DS1, cfg.LAMBDA_DS2),
                        )
                    val_loss += logs["total"]
                    val_batches += 1

            val_loss /= max(val_batches, 1)
            tr_loss = epoch_loss["total"] / max(n_batches, 1)
            lr_now = optimizer.param_groups[0]["lr"]

            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch:>3}/{cfg.NUM_EPOCHS}  "
                f"train={tr_loss:.5f}  val={val_loss:.5f}  lr={lr_now:.2e}"
            )

            if epoch % cfg.SAVE_EVERY == 0 or epoch == 1:
                ckpt_path = cfg.CHECKPOINT_DIR / f"ckpt_epoch_{epoch:04d}.pt"
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": dict(history),
                    "config": {
                        "BASE_CHANNELS": cfg.BASE_CHANNELS,
                        "DEPTH": cfg.DEPTH,
                        "CROP_SHAPE": cfg.CROP_SHAPE,
                        "IN_CHANNELS": cfg.IN_CHANNELS,
                        "DROPOUT_RATE": cfg.DROPOUT_RATE,
                        "MONO_STAGES": cfg.MONO_STAGES,
                        "MONO_SCALES": cfg.MONO_SCALES,
                    },
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path.name}")
                visualize_epoch(model, val_ds, epoch, save_dir=cfg.OUTPUT_DIR)
                plot_loss_curve(history, cfg.OUTPUT_DIR / "training_curve.png")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.module.state_dict(), cfg.CHECKPOINT_DIR / "best_model.pt")
                print(f"  New best val={best_val:.6f} -> best_model.pt")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="BrainTart training")
    parser.add_argument("--dataset", type=str, required=True, help="Path to BraTS dataset root")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--crop", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--checkpoint-dir", type=str, default="/kaggle/working/checkpoints")
    parser.add_argument("--results-dir", type=str, default="/kaggle/working/results")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/outputs")
    parser.add_argument(
        "--cache-dir", type=str, default="/kaggle/working/.patch_cache",
        help="Directory for pre-processed patch cache. Pass 'none' to disable.",
    )
    parser.add_argument(
        "--mono-stages", type=int, default=3,
        help="MonoUNet: encoder stages to inject local phase into (0 = disabled).",
    )
    parser.add_argument(
        "--mono-scales", type=int, default=3,
        help="MonoUNet: log-Gabor scales per filter.",
    )
    # v2 arguments
    parser.add_argument("--dropout", type=float, default=0.15, help="Bottleneck dropout rate (0 = disabled)")
    parser.add_argument("--warmup", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lam-edge", type=float, default=0.25, help="Edge loss weight")
    parser.add_argument("--lam-freq", type=float, default=0.1, help="Frequency loss weight")
    args = parser.parse_args()

    cfg = Config()
    cfg.DATASET_ROOT = Path(args.dataset)
    cfg.NUM_EPOCHS = args.epochs
    cfg.BATCH_PER_GPU = args.batch
    cfg.LR = args.lr
    cfg.SEED = args.seed
    cfg.CROP_SHAPE = tuple(args.crop)
    cfg.BASE_CHANNELS = args.base_ch
    cfg.DEPTH = args.depth
    cfg.CHECKPOINT_DIR = Path(args.checkpoint_dir)
    cfg.RESULTS_DIR = Path(args.results_dir)
    cfg.OUTPUT_DIR = Path(args.output_dir)
    cfg.CACHE_DIR = None if args.cache_dir.lower() == "none" else Path(args.cache_dir)
    cfg.MONO_STAGES = args.mono_stages
    cfg.MONO_SCALES = args.mono_scales
    # v2
    cfg.DROPOUT_RATE = args.dropout
    cfg.WARMUP_EPOCHS = args.warmup
    cfg.GRAD_ACCUM_STEPS = args.accum
    cfg.LAMBDA_EDGE = args.lam_edge
    cfg.LAMBDA_FREQ = args.lam_freq
    cfg.makedirs()

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True

    full_train_ds = BraTSTrainDataset(
        cfg.DATASET_ROOT, crop_shape=cfg.CROP_SHAPE,
        center_on_mask=cfg.CENTER_ON_MASK, augment=True,
        cache_dir=cfg.CACHE_DIR,
    )
    n_train = int(len(full_train_ds) * cfg.TRAIN_SPLIT)
    n_val = len(full_train_ds) - n_train
    train_ds, val_ds = random_split(
        full_train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    val_ds.dataset = deepcopy(full_train_ds)
    val_ds.dataset.augment = False

    print(f"Train cases: {n_train} | Val cases: {n_val}")

    n_gpus = torch.cuda.device_count()
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    if n_gpus >= 2:
        print(f"Launching DDP training on {n_gpus} GPUs...")
        mp.spawn(train_ddp, args=(n_gpus, train_ds, val_ds, cfg, history), nprocs=n_gpus, join=True)
    elif n_gpus == 1:
        print("Single GPU - running without DDP wrapper...")
        train_ddp(0, 1, train_ds, val_ds, cfg, history)
    else:
        raise RuntimeError("No CUDA GPUs found.")

    # Reload history from checkpoint
    ckpt_files = sorted(cfg.CHECKPOINT_DIR.glob("ckpt_epoch_*.pt"))
    if ckpt_files:
        _ckpt = torch.load(ckpt_files[-1], map_location="cpu")
        history = _ckpt["history"]
    plot_loss_curve(history, cfg.OUTPUT_DIR / "training_curve_final.png")
    print(f"Training complete. Best model at: {cfg.CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()

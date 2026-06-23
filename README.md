# BrainTart — Attention U-Net 3D for BraTS 2026 Inpainting

3D U-Net with attention gates, self-attention bottleneck, and deep supervision for the [BraTS 2026 Local Synthesis (Inpainting) Challenge](https://challenges.synapse.org/Challenges/DetailsPage/Task4?id=syn74274097).

## Architecture

- Strided-conv downsampling (nnU-Net style, no MaxPool)
- ResBlock3D encoder/decoder with GroupNorm + SiLU
- Attention gates on all skip connections (Oktay et al. 2018)
- Self-attention at bottleneck (12^3 = 1728 tokens at depth=3)
- Deep supervision at two decoder levels
- Loss: masked L1 + 3D SSIM + deep supervision L1

**Hardware target:** 2x T4 (16 GB each), ~6h training, PyTorch DDP.

## Project Structure

```
BrainTart/
├── configs/
│   └── default.py          # All hyperparameters
├── data/
│   └── dataset3d.py        # Training + Inference datasets
├── models/
│   ├── blocks.py           # ResBlock3D, AttentionGate3D, SelfAttention3D
│   └── attention_unet3d.py # Main model
├── losses/
│   └── combined_loss.py    # Masked L1 + SSIM + deep supervision
├── utils/
│   ├── preprocessing.py    # bbox, pad, crop, normalize
│   ├── augmentation.py     # elastic deform, gamma, flips
│   └── visualization.py    # training diagnostics
├── train.py                # DDP training entry point
├── inference.py            # Submission NIfTI generation
├── evaluate.py             # Local evaluation (mirrors Synapse)
├── notebooks/
│   └── run_braintart.ipynb # One-click Kaggle notebook
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py --dataset /path/to/brats_data --epochs 60
```

### 3. Inference

```bash
python inference.py \
    --dataset /path/to/brats_data \
    --checkpoint checkpoints/best_model.pt
```

### 4. Evaluate

```bash
python evaluate.py \
    --dataset /path/to/brats_data \
    --results /path/to/results
```

### Kaggle Notebook

Open `notebooks/run_braintart.ipynb` — it clones this repo, downloads data via Synapse, and runs the full pipeline.

## Submission Format

- Filename: `BraTS-GLI-XXXXX-YYY-t1n-inference.nii.gz`
- Shape: `(240, 240, 155)`
- Normalisation: `max(t1n_voided)` matches Synapse evaluation script

## References

- [BraTS 2026 Challenge Manuscript](https://arxiv.org/abs/2305.08992)
- [Challenge Baseline Repository](https://github.com/BraTS-inpainting/2026_challenge)
- Oktay et al., "Attention U-Net", 2018

# Supplementary Code: Topology-Aware Coronary Artery Segmentation

Code accompanying the MICCAI 2026 submission:
*"When Dice Deceives: A Topology-Aware Evaluation of Coronary Artery Segmentation Using Bifurcation Connectedness"*

## Overview

This repository contains the full implementation for:
1. **Bifurcation Connectivity Score (BCS)** — a topology-sensitive metric for coronary artery segmentation
2. **Soft BCS Loss** — a differentiable surrogate for end-to-end training
3. **Multi-task CT-FM** — SegResNet foundation model with segmentation + vesselness decoders
4. **Training, inference, and analysis scripts** for all experiments in Table 1

## Directory Structure

```
code_release/
├── metrics/
│   ├── bifurcation_connectivity.py   # BCS metric (Algorithm 1, §3.1)
│   └── __init__.py
├── training/
│   ├── multitask_ctfm.py             # Multi-task SegResNet model (§3.2)
│   ├── ctfm_l0_baseline.py           # L₀: DiceCE baseline
│   ├── ctfm_l1_vesselness.py         # L₁: DiceCE + vesselness
│   ├── ctfm_l2_cldice.py             # L₂: DiceCE + clDice
│   ├── ctfm_l4_soft_bcs.py           # L₄: DiceCE + Soft BCS (standalone)
│   ├── ctfm_l4_finetune_from_l1.py   # L_s: Fine-tune L₁ with Soft BCS (§3.3, Table 1)
│   ├── nnunet_custom_trainers.py     # nnU-Net L₀/L₁/L₂ trainers
│   └── precompute_stub_labels.py     # Precompute bifurcation stub labels
├── inference/
│   └── run_inference.py              # Sliding window inference + metrics
├── analysis/
│   ├── bcs_perturbation_experiment.py # Controlled perturbation study (§4.1)
│   ├── bootstrap_l4_vs_l1.py         # Paired bootstrap tests (§4.2)
│   ├── 01_generate_table1.py         # Generate Table 1
│   ├── 02_statistical_tests.py       # Statistical significance
│   ├── 03_figure_dice_vs_bcs.py      # Figure 3: Dice vs BCS scatter
│   └── generate_metric_correlation_heatmap.py  # Metric correlations
└── data/
    └── split_1000.csv                # ImageCAS train/val/test split
```

## Reproducing Paper Results

### Prerequisites

```bash
pip install torch monai scikit-image scipy nibabel lighter-zoo tqdm matplotlib
```

- **Dataset**: [ImageCAS](https://github.com/XiaoweiXu/ImageCAS) (1000 CTCA volumes)
- **CT-FM weights**: Auto-downloaded from HuggingFace (`project-lighter/ct_fm_segresnet`)

### Step 1: Data Preparation

```bash
# Precompute bifurcation stub labels for Soft BCS training
python training/precompute_stub_labels.py \
    --data_root /path/to/ImageCAS \
    --output_dir /path/to/ImageCAS/stub_labels
```

### Step 2: Train Baseline (L₁)

```bash
python training/ctfm_l1_vesselness.py
```

This trains the multi-task CT-FM with DiceCE + vesselness loss on 638 ImageCAS cases.
Expected: val Dice ~0.80, test Dice ~0.790, test BCS ~0.690.

### Step 3: Train L_s (Fine-tune L₁ with Soft BCS — Main Result)

```bash
python training/ctfm_l4_finetune_from_l1.py
```

Fine-tunes the L₁ decoder with Soft BCS loss (w=0.03), encoder frozen.
Expected: test Dice = 0.792, test BCS = 0.716 (Table 1, last row).

### Step 4: Inference + Metrics

```bash
python inference/run_inference.py \
    --checkpoint /path/to/best_model.pth \
    --test_dir /path/to/ImageCAS/test \
    --output_dir ./predictions \
    --overlap 0.75
```

### Step 5: Statistical Tests

```bash
python analysis/bootstrap_l4_vs_l1.py
```

## Key Hyperparameters (Table 1 Experiments)

| Parameter | Value |
|-----------|-------|
| Patch size | 96 × 96 × 96 |
| Voxel spacing | 1.0 mm isotropic |
| HU window | [-200, 600] |
| Batch size | 1 |
| Optimizer | Adam |
| BCS: tolerance (τ) | 3 voxels |
| BCS: stub length (l) | 8 voxels |

### CT-FM Models
| Model | LR | Epochs | Soft BCS weight |
|-------|------|--------|----------------|
| L₀ (baseline) | 1e-4 | 100 | — |
| L₁ (vesselness) | 1e-4 | 100 | — |
| L₂ (clDice) | 1e-4 | 100 | — |
| L_s (Soft BCS) | 1e-5 | 30 | 0.03 |

### nnU-Net Models
All trained with `nnUNetv2` using `3d_fullres` configuration.
Custom trainers in `training/nnunet_custom_trainers.py`.

## BCS Metric

The Bifurcation Connectivity Score (Algorithm 1) measures structural preservation at vessel branch points:

```python
from metrics.bifurcation_connectivity import BifurcationConnectivityScore

bcs = BifurcationConnectivityScore(tolerance=3, stub_length=8)
result = bcs.compute_score(pred_mask, gt_mask)
print(f"BCS: {result['bcs']:.3f}")
print(f"Preserved: {result['n_preserved']}/{result['n_expected']}")
```

## Soft BCS Loss

The differentiable surrogate (§3.3) is implemented in `training/ctfm_l4_soft_bcs.py`:

```python
# Inside training loop
soft_bcs_loss = SoftBCSLoss(min_stubs=2, softmin_temp=0.2)
bcs_loss = soft_bcs_loss(seg_logits, stub_label_map)
total_loss = dice_ce_loss + w * bcs_loss  # w = 0.03
```

## License

This code is provided for review purposes. Full release upon acceptance.

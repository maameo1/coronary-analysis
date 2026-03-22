# coronary-analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

Topology-aware deep learning for coronary artery segmentation from cardiac CT.

## Why This Matters

Standard overlap metrics like Dice can score 0.79 on coronary artery segmentation while silently missing 30% of vessel bifurcations — structural failures invisible to voxel counting but critical for clinical planning. A cardiologist tracing a vessel tree needs every junction intact; one broken bifurcation loses the entire downstream territory from the 3D model.

This repository introduces the **Bifurcation Connectedness Score (BCS)**, a topology-sensitive evaluation metric that directly counts preserved branch points, and **Soft BCS Loss**, its differentiable surrogate for end-to-end training. Validated on 1,000 cardiac CT volumes from ImageCAS with integration for MONAI SegResNet and nnU-Net.

## Key Results -TBD



## Quick Start

### BCS Metric

```python
from papers.miccai2026.code_release.metrics.bifurcation_connectivity import BifurcationConnectivityScore

bcs = BifurcationConnectivityScore(tolerance=3, stub_length=8)
result = bcs.compute_score(pred_mask, gt_mask)
print(f"BCS: {result['bcs']:.3f}")
print(f"Preserved: {result['n_preserved']}/{result['n_expected']}")
```

### Soft BCS Loss

```python
from papers.miccai2026.code_release.training.ctfm_l4_finetune_from_l1 import SoftBCSLoss

soft_bcs_loss = SoftBCSLoss(min_stubs=2, softmin_temp=0.2)
bcs_loss = soft_bcs_loss(seg_logits, stub_label_map)
total_loss = dice_ce_loss + 0.03 * bcs_loss
```

## Installation

```bash
git clone https://github.com/maameo1/coronary-analysis.git
cd coronary-analysis
pip install torch monai scikit-image scipy nibabel lighter-zoo tqdm matplotlib
```

**Dataset**: Download [ImageCAS](https://github.com/XiaoweiXu/ImageCAS) (1,000 CTCA volumes).
**CT-FM weights**: Auto-downloaded from HuggingFace (`project-lighter/ct_fm_segresnet`).

## Repository Structure

```
coronary-analysis/
├── papers/
│   ├── miccai2026/code_release/       # MICCAI 2026 — BCS metric & Soft BCS training
│   │   ├── metrics/                   # BCS implementation (Algorithm 1)
│   │   ├── training/                  # CT-FM and nnU-Net trainers
│   │   ├── inference/                 # Sliding window inference
│   │   └── analysis/                  # Statistical tests, figures
│   └── miua2026/                      # MIUA 2026 — FM adaptation ablation
├── segmentation/                      # Core segmentation pipeline (in development)
├── shared/                            # Shared utilities and metrics
├── stenosis/                          # Stenosis detection (planned)
└── acs_prediction/                    # ACS risk prediction (planned)
```

## Reproducing Paper Results

Full reproduction instructions are in [`papers/miccai2026/code_release/README.md`](papers/miccai2026/code_release/README.md). Summary:

```bash
# 1. Precompute bifurcation stub labels
python papers/miccai2026/code_release/training/precompute_stub_labels.py \
    --data_root /path/to/ImageCAS --output_dir /path/to/ImageCAS/stub_labels

# 2. Train baseline (L₁: DiceCE + vesselness)
python papers/miccai2026/code_release/training/ctfm_l1_vesselness.py

# 3. Fine-tune with Soft BCS loss (Lₛ — main result)
python papers/miccai2026/code_release/training/ctfm_l4_finetune_from_l1.py

# 4. Run inference
python papers/miccai2026/code_release/inference/run_inference.py \
    --checkpoint /path/to/best_model.pth --test_dir /path/to/ImageCAS/test

# 5. Statistical significance tests
python papers/miccai2026/code_release/analysis/bootstrap_l4_vs_l1.py
```

<details>
<summary><strong>Hyperparameters</strong></summary>

| Parameter | Value |
|-----------|-------|
| Patch size | 96 x 96 x 96 |
| Voxel spacing | 1.0 mm isotropic |
| HU window | [-200, 600] |
| Batch size | 1 |
| Optimizer | Adam |
| BCS tolerance (τ) | 3 voxels |
| BCS stub length (l) | 8 voxels |

| Model | LR | Epochs | Soft BCS weight |
|-------|------|--------|----------------|
| L₀ (baseline) | 1e-4 | 100 | — |
| L₁ (vesselness) | 1e-4 | 100 | — |
| L₂ (clDice) | 1e-4 | 100 | — |
| Lₛ (Soft BCS) | 1e-5 | 30 | 0.03 |

</details>

## Publications

**MICCAI 2026** — *When Dice Deceives: A Topology-Aware Evaluation of Coronary Artery Segmentation Using Bifurcation Connectedness*

**MIUA 2026** — *Topology-aware Low-shot Adaptation of CT Foundation Models for Coronary Artery Segmentation*

**MedIA** — *BCS-F1: Topology-aware Metric and Differentiable Loss for Tubular Structure Segmentation* (in preparation)

## Citation

If you use BCS or Soft BCS Loss in your research, please cite:

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{owusuansah2026dice,
  title     = {When Dice Deceives: A Topology-Aware Evaluation of Coronary
               Artery Segmentation Using Bifurcation Connectedness},
  author    = {Owusu-Ansah, Maame and others},
  booktitle = {Medical Image Computing and Computer Assisted Intervention
               (MICCAI)},
  year      = {2026}
}

@inproceedings{owusuansah2026topology,
  title     = {Topology-aware Low-shot Adaptation of CT Foundation Models
               for Coronary Artery Segmentation},
  author    = {Owusu-Ansah, Maame and others},
  booktitle = {Medical Image Understanding and Analysis (MIUA)},
  year      = {2026}
}
```

</details>

## License

[MIT](LICENSE) — Copyright 2026 Maame Owusu-Ansah

## Acknowledgements

University of Lincoln | [ImageCAS](https://github.com/XiaoweiXu/ImageCAS) | [CT-FM](https://huggingface.co/project-lighter/ct_fm_segresnet)

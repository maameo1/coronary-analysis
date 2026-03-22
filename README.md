# coronary-analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)

Topology-aware deep learning for coronary artery segmentation from cardiac CT.

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
│   │   ├── metrics/                   # --- implementation (Algorithm 1)
│   │   ├── training/                  # CT-FM and nnU-Net trainers
│   │   ├── inference/                 # Sliding window inference
│   │   └── analysis/                  # Statistical tests, figures
│   └── miua2026/                      # MIUA 2026 — FM adaptation ablation
├── segmentation/                      # Core segmentation pipeline (in development)
├── shared/                            # Shared utilities and metrics
├── stenosis/                          # Stenosis detection (planned)
└── acs_prediction/                    # ACS risk prediction (planned)
```

## License

[MIT](LICENSE) — Copyright 2026 Maame Owusu-Ansah

## Acknowledgements

University of Lincoln | [ImageCAS](https://github.com/XiaoweiXu/ImageCAS) | [CT-FM](https://huggingface.co/project-lighter/ct_fm_segresnet)

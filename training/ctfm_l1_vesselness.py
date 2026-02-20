#!/usr/bin/env python3
"""
================================================================================
EXP-028: Progressive Ablation Study - Foundation Model vs U-Net
================================================================================

RESEARCH QUESTION: Does foundation model pretraining improve segmentation
quality, and how do additional components (vesselness, clDice, deep supervision)
contribute to performance?

ABLATION DESIGN (Progressive):
    Level 0: Baseline       - Dice + CE only
    Level 1: +Vesselness    - Dice + CE + Vesselness auxiliary
    Level 2: +clDice        - Dice + CE + Vesselness + clDice
    Level 3: +Deep Sup      - Dice + CE + Vesselness + clDice + Deep Supervision

Both U-Net and CT-FM use IDENTICAL configurations at each level.
This isolates the effect of foundation model pretraining.

EXPERIMENT MATRIX:
    4 ablation levels × 2 models × 5 data regimes = 40 experiments

Data regimes: 50, 100, 200, 350, 638 training cases
Metrics: Dice, clDice, Betti₀ Error

HYPOTHESIS:
- CT-FM should outperform U-Net at all levels, especially low-data regimes
- Each component should provide incremental improvement
- The gap between CT-FM and U-Net should be largest at Level 0 (pretraining matters most when other aids are absent)

Author: Anonymous
Date: December 2025
Experiment: EXP-028

================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label

from monai.networks.nets import UNet
from lighter_zoo import SegResNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import pad_list_data_collate

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    SpatialPadd, RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, EnsureTyped, Activations, AsDiscrete,
    MapTransform
)

from monai.data import Dataset, DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter
from imagecas_data_loader import ImageCASDataLoader

from skimage.filters import frangi

# Import BCS metric
from metrics.branch_connectivity import BranchConnectivityScore

print("="*70)
print("EXP-028: Progressive Ablation Study")
print("="*70)


# ============================================================================
# ABLATION LEVELS
# ============================================================================

ABLATION_LEVELS = {
    0: {'name': 'baseline', 'vesselness': False, 'cldice': False, 'deep_sup': False},
    1: {'name': 'vesselness', 'vesselness': True, 'cldice': False, 'deep_sup': False},
    2: {'name': 'cldice', 'vesselness': True, 'cldice': True, 'deep_sup': False},
    3: {'name': 'full', 'vesselness': True, 'cldice': True, 'deep_sup': True},
}


# ============================================================================
# SOFT SKELETONIZATION (for clDice)
# ============================================================================

def soft_erode(img):
    if len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    elif len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)


def soft_dilate(img):
    if len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
    elif len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_=50):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftclDiceLoss(nn.Module):
    def __init__(self, iter_=50, smooth=1e-5):
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth

    def forward(self, pred, target):
        pred_fg = pred[:, 1:2, ...]
        target_fg = target[:, 1:2, ...].float()
        skel_pred = soft_skel(pred_fg, self.iter_)
        skel_target = soft_skel(target_fg, self.iter_)
        tprec = ((skel_pred * target_fg).sum() + self.smooth) / (skel_pred.sum() + self.smooth)
        tsens = ((skel_target * pred_fg).sum() + self.smooth) / (skel_target.sum() + self.smooth)
        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice


class HardBettiError(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, pred, target):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        pred = np.squeeze(pred)
        target = np.squeeze(target)
        pred_binary = (pred > 0.5).astype(np.uint8)
        target_binary = (target > 0.5).astype(np.uint8)
        _, pred_n = label(pred_binary, return_num=True)
        _, target_n = label(target_binary, return_num=True)
        return abs(pred_n - target_n)


# ============================================================================
# VESSELNESS TRANSFORM
# ============================================================================

class AddVesselnessd(MapTransform):
    """Add vesselness target during data loading."""

    def __init__(self, keys, frangi_sigmas, frangi_alpha, frangi_beta, frangi_gamma):
        super().__init__(keys)
        self.frangi_sigmas = list(frangi_sigmas)
        self.frangi_alpha = frangi_alpha
        self.frangi_beta = frangi_beta
        self.frangi_gamma = frangi_gamma

    def __call__(self, data):
        d = dict(data)
        image = d["image"]

        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
        else:
            image_np = np.array(image)

        if image_np.ndim == 4:
            image_np = image_np[0]

        vessel_response = frangi(
            image_np,
            sigmas=self.frangi_sigmas,
            alpha=self.frangi_alpha,
            beta=self.frangi_beta,
            gamma=self.frangi_gamma,
            black_ridges=False
        )

        if vessel_response.max() > 0:
            vessel_response = vessel_response / (vessel_response.max() + 1e-8)

        d["vesselness"] = torch.from_numpy(vessel_response[np.newaxis].astype(np.float32))
        return d


# ============================================================================
# MODELS
# ============================================================================

class VanillaUNetWithAux(nn.Module):
    """
    Standard MONAI U-Net with optional vesselness decoder and deep supervision.
    Mirrors the CT-FM model architecture for fair comparison.
    """

    def __init__(self, in_channels=1, out_channels=2, use_vesselness=False, use_deep_sup=False):
        super().__init__()
        self.use_vesselness = use_vesselness
        self.use_deep_sup = use_deep_sup

        # Main U-Net
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm='instance'
        )

        # Vesselness decoder (from bottleneck)
        if use_vesselness:
            self.vessel_decoder = nn.Sequential(
                nn.ConvTranspose3d(512, 128, kernel_size=2, stride=2),
                nn.InstanceNorm3d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
                nn.InstanceNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        # Deep supervision heads
        if use_deep_sup:
            # Level 2: 128 channels -> upsample 4x
            self.ds_head_level2 = nn.Sequential(
                nn.Conv3d(128, out_channels, kernel_size=1),
                nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
            )
            # Level 3: 64 channels -> upsample 2x
            self.ds_head_level3 = nn.Sequential(
                nn.Conv3d(64, out_channels, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            )

        self._print_params()

    def _print_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"U-Net parameters: {total_params:,}")
        if self.use_vesselness:
            vessel_params = sum(p.numel() for p in self.vessel_decoder.parameters())
            print(f"  Vesselness decoder: {vessel_params:,}")
        if self.use_deep_sup:
            ds_params = sum(p.numel() for p in self.ds_head_level2.parameters()) + \
                        sum(p.numel() for p in self.ds_head_level3.parameters())
            print(f"  Deep supervision heads: {ds_params:,}")

    def forward(self, x):
        """
        Returns dict with:
            'seg': main segmentation output
            'vessel': vesselness prediction (if use_vesselness)
            'ds_level2', 'ds_level3': deep supervision outputs (if use_deep_sup)
        """
        outputs = {}

        # We need to manually run through encoder/decoder to get intermediate features
        # For simplicity with MONAI UNet, we'll extract features via hooks
        features = {}

        def get_activation(name):
            def hook(model, input, output):
                features[name] = output
            return hook

        # Register hooks for intermediate features
        hooks = []
        if self.use_vesselness or self.use_deep_sup:
            # Hook into encoder stages - MONAI UNet structure:
            # model.0 = first ResidualUnit (32ch)
            # model.1.submodule.0 = second encoder (64ch)
            # model.1.submodule.1.submodule.0 = third encoder (128ch)
            # model.1.submodule.1.submodule.1.submodule.0 = fourth encoder (256ch)
            # model.1.submodule.1.submodule.1.submodule.1.submodule = bottleneck (512ch)
            hooks.append(self.unet.model[1].submodule[0].register_forward_hook(get_activation('enc1')))  # 64ch
            hooks.append(self.unet.model[1].submodule[1].submodule[0].register_forward_hook(get_activation('enc2')))  # 128ch
            hooks.append(self.unet.model[1].submodule[1].submodule[1].submodule[1].submodule.register_forward_hook(get_activation('bottleneck')))  # 512ch

        # Forward pass
        seg_out = self.unet(x)
        outputs['seg'] = seg_out

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Vesselness from bottleneck
        if self.use_vesselness and 'bottleneck' in features:
            outputs['vessel'] = self.vessel_decoder(features['bottleneck'])

        # Deep supervision
        if self.use_deep_sup:
            if 'enc2' in features:
                outputs['ds_level2'] = self.ds_head_level2(features['enc2'])
            if 'enc1' in features:
                outputs['ds_level3'] = self.ds_head_level3(features['enc1'])

        return outputs


class CTFMWithAux(nn.Module):
    """
    CT Foundation Model with optional vesselness decoder and deep supervision.
    """

    def __init__(self, ctfm_model_name="project-lighter/ct_fm_segresnet",
                 use_vesselness=False, use_deep_sup=False):
        super().__init__()
        self.use_vesselness = use_vesselness
        self.use_deep_sup = use_deep_sup

        print(f"Loading CT-FM from {ctfm_model_name}...")
        self.ctfm = SegResNet.from_pretrained(ctfm_model_name)

        bottleneck_channels = 512

        # Vesselness decoder
        if use_vesselness:
            self.vessel_decoder = nn.Sequential(
                nn.ConvTranspose3d(bottleneck_channels, 128, kernel_size=2, stride=2),
                nn.InstanceNorm3d(128),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
                nn.InstanceNorm3d(64),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
                nn.InstanceNorm3d(32),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
                nn.InstanceNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 1, kernel_size=1),
                nn.Sigmoid(),
            )

        # Deep supervision heads
        if use_deep_sup:
            # Level 2: 64 channels at 48³ -> upsample 2x to 96³
            self.ds_head_level2 = nn.Sequential(
                nn.Conv3d(64, 2, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            )
            # Level 3: 128 channels at 24³ -> upsample 4x to 96³
            self.ds_head_level3 = nn.Sequential(
                nn.Conv3d(128, 2, kernel_size=1),
                nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
            )

        self._print_params()

    def _print_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"CT-FM parameters: {total_params:,}")
        if self.use_vesselness:
            vessel_params = sum(p.numel() for p in self.vessel_decoder.parameters())
            print(f"  Vesselness decoder: {vessel_params:,}")
        if self.use_deep_sup:
            ds_params = sum(p.numel() for p in self.ds_head_level2.parameters()) + \
                        sum(p.numel() for p in self.ds_head_level3.parameters())
            print(f"  Deep supervision heads: {ds_params:,}")

    def _run_decoder_with_features(self, enc_out):
        """Run decoder and capture intermediate features."""
        x = enc_out[-1]
        intermediate = {}

        for i, up_layer in enumerate(self.ctfm.up_layers):
            x = up_layer['upsample'](x)
            skip_idx = len(enc_out) - 2 - i
            if skip_idx >= 0:
                x = x + enc_out[skip_idx]
            x = up_layer['blocks'](x)
            x = up_layer['head'](x)

            # Capture for deep supervision
            if i == 1:  # 128 channels at 24³
                intermediate['level3'] = x.clone()
            elif i == 2:  # 64 channels at 48³
                intermediate['level2'] = x.clone()

        return x, intermediate

    def forward(self, x):
        """
        Returns dict with:
            'seg': main segmentation output
            'vessel': vesselness prediction (if use_vesselness)
            'ds_level2', 'ds_level3': deep supervision outputs (if use_deep_sup)
        """
        outputs = {}

        # Encoder
        enc_out = self.ctfm.encoder(x)

        # Decoder with feature capture
        seg_out, intermediate = self._run_decoder_with_features(enc_out)
        outputs['seg'] = seg_out

        # Vesselness from bottleneck
        if self.use_vesselness:
            outputs['vessel'] = self.vessel_decoder(enc_out[-1])

        # Deep supervision
        if self.use_deep_sup:
            if 'level2' in intermediate:
                outputs['ds_level2'] = self.ds_head_level2(intermediate['level2'])
            if 'level3' in intermediate:
                outputs['ds_level3'] = self.ds_head_level3(intermediate['level3'])

        return outputs


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for ablation experiments."""

    exp_name = "progressive_ablation"
    exp_id = "EXP-028"

    # Auto-detect HPC vs cloud environment
    _on_hpc = Path('/home/users').exists() or 'novel' in os.uname().nodename.lower()

    if _on_hpc:
        _home = Path(os.environ.get('HOME', '/path/to/home'))
        data_root = _home / 'cardiac-imaging' / 'data' / 'ImageCAS'
        split_file = _home / 'cardiac-imaging' / 'data' / 'ImageCAS' / 'split_1000.csv'
        results_file = _home / 'cardiac-imaging' / 'experiments' / 'ablation_results.json'
    else:
        data_root = Path("/path/to/datasets/ImageCAS")
        split_file = Path("/path/to/datasets/ImageCAS/split_1000.csv")
        results_file = Path("/path/to/project/ablation_results.json")

    # Data configuration
    test_fold = 1
    val_fraction = 0.15
    voxel_spacing = (1.0, 1.0, 1.0)
    patch_size = (96, 96, 96)

    hu_min = -200
    hu_max = 600

    # Frangi settings (for vesselness)
    frangi_sigmas = range(1, 4)
    frangi_alpha = 0.5
    frangi_beta = 0.5
    frangi_gamma = 15

    # Training
    batch_size = 1
    num_workers = 4
    learning_rate = 1e-4
    weight_decay = 1e-5
    max_epochs = 100
    val_interval = 1

    # Loss weights
    dice_weight = 0.35
    ce_weight = 0.15
    cldice_weight = 0.15
    vessel_weight = 0.35

    # Deep supervision weights
    ds_weight_level2 = 0.3
    ds_weight_level3 = 0.3

    cldice_iterations = 50

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True
    random_seed = 2025

    def __init__(self, model_type='unet', ablation_level=0, n_train_cases=100):
        self.model_type = model_type
        self.ablation_level = ablation_level
        self.n_train_cases = n_train_cases
        self.date = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Get ablation settings
        self.ablation_config = ABLATION_LEVELS[ablation_level]
        self.use_vesselness = self.ablation_config['vesselness']
        self.use_cldice = self.ablation_config['cldice']
        self.use_deep_sup = self.ablation_config['deep_sup']

        level_name = self.ablation_config['name']
        # Auto-detect HPC vs cloud
        if self._on_hpc:
            base_output = self._home / 'cardiac-imaging' / 'experiments'
        else:
            base_output = Path("/path/to/project/experiments")
        self.output_root = base_output / f"{self.exp_id}_L{ablation_level}_{level_name}_{model_type}_{n_train_cases}cases_{self.date}"
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "checkpoints").mkdir(exist_ok=True)
        (self.output_root / "logs").mkdir(exist_ok=True)

        self.save_config()

    def save_config(self):
        config_dict = {
            'exp_id': self.exp_id,
            'model_type': self.model_type,
            'ablation_level': self.ablation_level,
            'ablation_name': self.ablation_config['name'],
            'n_train_cases': self.n_train_cases,
            'use_vesselness': self.use_vesselness,
            'use_cldice': self.use_cldice,
            'use_deep_sup': self.use_deep_sup,
            'dice_weight': self.dice_weight,
            'ce_weight': self.ce_weight,
            'cldice_weight': self.cldice_weight if self.use_cldice else 0,
            'vessel_weight': self.vessel_weight if self.use_vesselness else 0,
            'ds_weight_level2': self.ds_weight_level2 if self.use_deep_sup else 0,
            'ds_weight_level3': self.ds_weight_level3 if self.use_deep_sup else 0,
            'max_epochs': self.max_epochs,
        }
        with open(self.output_root / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


# ============================================================================
# RESULTS TRACKING
# ============================================================================

def save_ablation_results(config, metrics_dict):
    """Save results for cross-experiment comparison."""
    results_file = config.results_file

    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    key = f"L{config.ablation_level}_{config.model_type}_{config.n_train_cases}"
    all_results[key] = {
        'model': config.model_type,
        'ablation_level': config.ablation_level,
        'ablation_name': config.ablation_config['name'],
        'n_train': config.n_train_cases,
        'dice': metrics_dict['dice'],
        'cldice': metrics_dict['cldice'],
        'betti_error': metrics_dict['betti_error'],
        'bcs': metrics_dict['bcs'],
        'use_vesselness': config.use_vesselness,
        'use_cldice': config.use_cldice,
        'use_deep_sup': config.use_deep_sup,
        'timestamp': datetime.now().isoformat()
    }

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {results_file}")


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(config, mode='train'):
    """Create transform pipeline based on ablation config."""

    # Keys depend on whether we need vesselness
    if config.use_vesselness:
        all_keys = ["image", "label", "vesselness"]
    else:
        all_keys = ["image", "label"]

    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config.voxel_spacing,
                mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                            a_min=config.hu_min, a_max=config.hu_max,
                            b_min=0.0, b_max=1.0, clip=True),
    ]

    # Add vesselness computation if needed
    if config.use_vesselness:
        common_transforms.append(
            AddVesselnessd(
                keys=["image"],
                frangi_sigmas=config.frangi_sigmas,
                frangi_alpha=config.frangi_alpha,
                frangi_beta=config.frangi_beta,
                frangi_gamma=config.frangi_gamma
            )
        )

    common_transforms.extend([
        CropForegroundd(keys=all_keys, source_key="image"),
        SpatialPadd(keys=all_keys, spatial_size=config.patch_size, mode="constant"),
        RandCropByPosNegLabeld(
            keys=all_keys, label_key="label",
            spatial_size=config.patch_size,
            pos=1, neg=1, num_samples=1,
            image_key="image", image_threshold=0, allow_smaller=False
        ),
        EnsureTyped(keys=all_keys),
    ])

    if mode == 'train':
        train_transforms = common_transforms + [
            RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=all_keys, prob=0.5, spatial_axes=(0, 1)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
        return Compose(train_transforms)
    else:
        return Compose(common_transforms)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fns, config, scaler=None):
    """Training epoch with configurable loss components."""
    model.train()

    dice_loss_fn, ce_loss_fn, cldice_loss_fn, vessel_loss_fn = loss_fns

    epoch_losses = {'total': 0, 'dice': 0, 'ce': 0}
    if config.use_cldice:
        epoch_losses['cldice'] = 0
    if config.use_vesselness:
        epoch_losses['vessel'] = 0
    if config.use_deep_sup:
        epoch_losses['ds_level2'] = 0
        epoch_losses['ds_level3'] = 0

    for batch_data in train_loader:
        inputs = batch_data["image"].to(config.device)
        labels = batch_data["label"].to(config.device)
        if config.use_vesselness:
            vesselness = batch_data["vesselness"].to(config.device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                seg_pred = outputs['seg']

                # Base losses: Dice + CE
                dice_loss = dice_loss_fn(seg_pred, labels)
                ce_loss = ce_loss_fn(seg_pred, labels.squeeze(1).long())

                total_loss = config.dice_weight * dice_loss + config.ce_weight * ce_loss

                # clDice (if enabled)
                if config.use_cldice:
                    seg_softmax = F.softmax(seg_pred, dim=1)
                    seg_onehot = torch.zeros_like(seg_softmax)
                    seg_onehot.scatter_(1, labels.long(), 1)
                    cldice_loss = cldice_loss_fn(seg_softmax, seg_onehot)
                    total_loss = total_loss + config.cldice_weight * cldice_loss

                # Vesselness (if enabled)
                if config.use_vesselness and 'vessel' in outputs:
                    vessel_loss = vessel_loss_fn(outputs['vessel'], vesselness)
                    total_loss = total_loss + config.vessel_weight * vessel_loss

                # Deep supervision (if enabled)
                if config.use_deep_sup:
                    ds_loss = 0.0
                    if 'ds_level2' in outputs:
                        ds2_dice = dice_loss_fn(outputs['ds_level2'], labels)
                        ds2_ce = ce_loss_fn(outputs['ds_level2'], labels.squeeze(1).long())
                        ds_level2_loss = ds2_dice + 0.5 * ds2_ce
                        ds_loss = ds_loss + config.ds_weight_level2 * ds_level2_loss
                    if 'ds_level3' in outputs:
                        ds3_dice = dice_loss_fn(outputs['ds_level3'], labels)
                        ds3_ce = ce_loss_fn(outputs['ds_level3'], labels.squeeze(1).long())
                        ds_level3_loss = ds3_dice + 0.5 * ds3_ce
                        ds_loss = ds_loss + config.ds_weight_level3 * ds_level3_loss
                    total_loss = total_loss + ds_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            seg_pred = outputs['seg']

            dice_loss = dice_loss_fn(seg_pred, labels)
            ce_loss = ce_loss_fn(seg_pred, labels.squeeze(1).long())
            total_loss = config.dice_weight * dice_loss + config.ce_weight * ce_loss

            if config.use_cldice:
                seg_softmax = F.softmax(seg_pred, dim=1)
                seg_onehot = torch.zeros_like(seg_softmax)
                seg_onehot.scatter_(1, labels.long(), 1)
                cldice_loss = cldice_loss_fn(seg_softmax, seg_onehot)
                total_loss = total_loss + config.cldice_weight * cldice_loss

            if config.use_vesselness and 'vessel' in outputs:
                vessel_loss = vessel_loss_fn(outputs['vessel'], vesselness)
                total_loss = total_loss + config.vessel_weight * vessel_loss

            if config.use_deep_sup:
                ds_loss = 0.0
                if 'ds_level2' in outputs:
                    ds2_dice = dice_loss_fn(outputs['ds_level2'], labels)
                    ds2_ce = ce_loss_fn(outputs['ds_level2'], labels.squeeze(1).long())
                    ds_level2_loss = ds2_dice + 0.5 * ds2_ce
                    ds_loss = ds_loss + config.ds_weight_level2 * ds_level2_loss
                if 'ds_level3' in outputs:
                    ds3_dice = dice_loss_fn(outputs['ds_level3'], labels)
                    ds3_ce = ce_loss_fn(outputs['ds_level3'], labels.squeeze(1).long())
                    ds_level3_loss = ds3_dice + 0.5 * ds3_ce
                    ds_loss = ds_loss + config.ds_weight_level3 * ds_level3_loss
                total_loss = total_loss + ds_loss

            total_loss.backward()
            optimizer.step()

        # Track losses
        epoch_losses['total'] += total_loss.item()
        epoch_losses['dice'] += dice_loss.item()
        epoch_losses['ce'] += ce_loss.item()
        if config.use_cldice:
            epoch_losses['cldice'] += cldice_loss.item()
        if config.use_vesselness:
            epoch_losses['vessel'] += vessel_loss.item()
        if config.use_deep_sup:
            epoch_losses['ds_level2'] += ds_level2_loss.item() if 'ds_level2' in outputs else 0
            epoch_losses['ds_level3'] += ds_level3_loss.item() if 'ds_level3' in outputs else 0

    n_batches = len(train_loader)
    for k in epoch_losses:
        epoch_losses[k] /= n_batches

    return epoch_losses


@torch.no_grad()
def validate(model, val_loader, config):
    """Validation with Dice, clDice, Betti error, and BCS metrics."""
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    betti_error_fn = HardBettiError()
    bcs_metric = BranchConnectivityScore(min_branch_length=10)

    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    all_cldice = []
    all_betti_error = []
    all_bcs = []

    for batch_data in val_loader:
        inputs = batch_data["image"].to(config.device)
        labels = batch_data["label"].to(config.device)

        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            seg_pred = outputs['seg']

        seg_pred_post = [post_pred(i) for i in decollate_batch(seg_pred)]
        labels_post = [post_label(i) for i in decollate_batch(labels)]

        dice_metric(seg_pred_post, labels_post)

        for pred, label in zip(seg_pred_post, labels_post):
            pred_fg = pred[1:2].float()
            label_fg = label[1:2].float()

            if label_fg.sum() > 0:
                # clDice
                skel_pred = soft_skel(pred_fg.unsqueeze(0), iter_=50).squeeze(0)
                skel_label = soft_skel(label_fg.unsqueeze(0), iter_=50).squeeze(0)

                tprec = (skel_pred * label_fg).sum() / (skel_pred.sum() + 1e-8)
                tsens = (skel_label * pred_fg).sum() / (skel_label.sum() + 1e-8)
                cldice = 2 * tprec * tsens / (tprec + tsens + 1e-8)
                all_cldice.append(cldice.item())

                # Betti error
                betti_err = betti_error_fn(pred_fg, label_fg)
                all_betti_error.append(betti_err)

                # Branch Connectivity Score
                pred_mask = pred_fg.squeeze().cpu().numpy() > 0.5
                label_mask = label_fg.squeeze().cpu().numpy() > 0.5
                bcs_scores = bcs_metric.compute_score(pred_mask, label_mask)
                all_bcs.append(bcs_scores['bcs'])

    mean_dice = dice_metric.aggregate().item()
    mean_cldice = np.mean(all_cldice) if all_cldice else 0.0
    mean_betti_error = np.mean(all_betti_error) if all_betti_error else 0.0
    mean_bcs = np.mean(all_bcs) if all_bcs else 0.0

    dice_metric.reset()

    return {
        'dice': mean_dice,
        'cldice': mean_cldice,
        'betti_error': mean_betti_error,
        'bcs': mean_bcs
    }


def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, config, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_dice': best_dice,
    }
    torch.save(checkpoint, config.output_root / "checkpoints" / "latest_checkpoint.pth")
    if is_best:
        torch.save(checkpoint, config.output_root / "checkpoints" / "best_model.pth")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(model_type='unet', ablation_level=0, n_train_cases=100):
    """Train a single model configuration."""

    config = Config(model_type=model_type, ablation_level=ablation_level, n_train_cases=n_train_cases)

    print(f"\n{'='*70}")
    print(f"ABLATION LEVEL {ablation_level}: {config.ablation_config['name'].upper()}")
    print(f"Model: {model_type.upper()} | Training cases: {n_train_cases}")
    print(f"{'='*70}")
    print(f"  Vesselness: {config.use_vesselness}")
    print(f"  clDice:     {config.use_cldice}")
    print(f"  Deep Sup:   {config.use_deep_sup}")
    print(f"Output: {config.output_root}")

    # Data
    data_loader = ImageCASDataLoader(
        data_root=config.data_root,
        split_file=config.split_file,
        test_fold=config.test_fold,
        val_fraction=config.val_fraction,
        random_seed=config.random_seed
    )

    splits = data_loader.get_splits(
        n_train_cases=config.n_train_cases,
        regime_seed=config.random_seed
    )

    print(f"Training: {len(splits['train'])} cases")
    print(f"Validation: {len(splits['val'])} cases")

    train_transforms = get_transforms(config, mode='train')
    val_transforms = get_transforms(config, mode='val')

    train_ds = Dataset(data=splits['train'], transform=train_transforms)
    val_ds = Dataset(data=splits['val'], transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=pad_list_data_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=config.num_workers, collate_fn=pad_list_data_collate
    )

    # Model
    if model_type == 'unet':
        model = VanillaUNetWithAux(
            in_channels=1, out_channels=2,
            use_vesselness=config.use_vesselness,
            use_deep_sup=config.use_deep_sup
        ).to(config.device)
    elif model_type == 'ctfm':
        model = CTFMWithAux(
            use_vesselness=config.use_vesselness,
            use_deep_sup=config.use_deep_sup
        ).to(config.device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Loss functions
    dice_loss_fn = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
    ce_loss_fn = nn.CrossEntropyLoss()
    cldice_loss_fn = SoftclDiceLoss(iter_=config.cldice_iterations) if config.use_cldice else None
    vessel_loss_fn = nn.MSELoss() if config.use_vesselness else None

    loss_fns = (dice_loss_fn, ce_loss_fn, cldice_loss_fn, vessel_loss_fn)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    scaler = torch.amp.GradScaler('cuda') if config.amp else None

    # TensorBoard
    writer = SummaryWriter(log_dir=config.output_root / "logs")

    # Check for existing checkpoint to resume from
    start_epoch = 0
    best_dice = 0.0
    best_metrics = None

    # Look for latest checkpoint in any matching experiment folder
    exp_pattern = f"EXP-028_L{ablation_level}_*_{model_type}_{n_train_cases}cases_*"
    # Use correct path based on HPC vs cloud
    if config._on_hpc:
        experiments_dir = config._home / 'cardiac-imaging' / 'experiments'
    else:
        experiments_dir = Path("/path/to/project/experiments")
    existing_exps = sorted(experiments_dir.glob(exp_pattern))

    for exp_dir in reversed(existing_exps):  # Check newest first
        ckpt_path = exp_dir / "checkpoints" / "latest_checkpoint.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=config.device, weights_only=False)
            if ckpt.get('epoch', 0) >= config.max_epochs - 1:
                # Already completed
                print(f"[SKIP] Found completed checkpoint at epoch {ckpt['epoch']+1}")
                return None
            # Resume from this checkpoint
            print(f"[RESUME] Loading checkpoint from epoch {ckpt['epoch']+1}")
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_dice = ckpt.get('best_dice', 0.0)
            break

    # Training loop
    for epoch in range(start_epoch, config.max_epochs):
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")

        train_losses = train_epoch(model, train_loader, optimizer, loss_fns, config, scaler)

        loss_parts = [f"Total: {train_losses['total']:.4f}",
                      f"Dice: {train_losses['dice']:.4f}",
                      f"CE: {train_losses['ce']:.4f}"]
        if config.use_cldice:
            loss_parts.append(f"clDice: {train_losses['cldice']:.4f}")
        if config.use_vesselness:
            loss_parts.append(f"Vessel: {train_losses['vessel']:.4f}")
        if config.use_deep_sup:
            loss_parts.append(f"DS2: {train_losses['ds_level2']:.4f}")
            loss_parts.append(f"DS3: {train_losses['ds_level3']:.4f}")
        print(f"Loss - {', '.join(loss_parts)}")

        for k, v in train_losses.items():
            writer.add_scalar(f'train/loss_{k}', v, epoch)

        if (epoch + 1) % config.val_interval == 0:
            metrics = validate(model, val_loader, config)
            print(f"Val Dice: {metrics['dice']:.4f} | clDice: {metrics['cldice']:.4f} | "
                  f"BCS: {metrics['bcs']:.4f} | Betti₀ Error: {metrics['betti_error']:.2f}")

            writer.add_scalar('val/dice', metrics['dice'], epoch)
            writer.add_scalar('val/cldice', metrics['cldice'], epoch)
            writer.add_scalar('val/bcs', metrics['bcs'], epoch)
            writer.add_scalar('val/betti_error', metrics['betti_error'], epoch)

            scheduler.step(metrics['dice'])

            is_best = metrics['dice'] > best_dice
            if is_best:
                best_dice = metrics['dice']
                best_metrics = metrics.copy()
                print(f"  -> New best Dice: {best_dice:.4f}")

            save_checkpoint(model, optimizer, scheduler, epoch, best_dice, config, is_best)

    writer.close()

    # Save final results
    if best_metrics:
        save_ablation_results(config, best_metrics)

    print(f"\n{'='*70}")
    print(f"COMPLETE: L{ablation_level} {model_type.upper()} @ {n_train_cases} cases")
    print(f"Best Dice: {best_dice:.4f}")
    print(f"{'='*70}")

    return best_metrics


# ============================================================================
# PLOTTING
# ============================================================================

def plot_ablation_results():
    """Generate comparison plots from ablation results."""

    results_file = Path("/path/to/project/ablation_results.json")

    if not results_file.exists():
        print("No results file found. Run experiments first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    # Organize results
    # Structure: {level: {model: {n_train: metrics}}}
    organized = {0: {'unet': {}, 'ctfm': {}},
                 1: {'unet': {}, 'ctfm': {}},
                 2: {'unet': {}, 'ctfm': {}},
                 3: {'unet': {}, 'ctfm': {}}}

    for key, r in results.items():
        level = r['ablation_level']
        model = r['model']
        n = r['n_train']
        organized[level][model][n] = r

    # Create figure: 4 rows (levels) x 4 cols (metrics)
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))

    level_names = ['L0: Baseline (Dice+CE)',
                   'L1: +Vesselness',
                   'L2: +clDice',
                   'L3: +Deep Supervision']

    colors = {'unet': 'blue', 'ctfm': 'red'}
    markers = {'unet': 'o', 'ctfm': 's'}

    for level in range(4):
        for model in ['unet', 'ctfm']:
            data = organized[level][model]
            if not data:
                continue

            n_vals = sorted(data.keys())
            dice_vals = [data[n]['dice'] for n in n_vals]
            cldice_vals = [data[n]['cldice'] for n in n_vals]
            bcs_vals = [data[n].get('bcs', 0) for n in n_vals]
            betti_vals = [data[n]['betti_error'] for n in n_vals]

            label = 'U-Net' if model == 'unet' else 'CT-FM'

            axes[level, 0].plot(n_vals, dice_vals, f'{markers[model]}-',
                               label=label, color=colors[model], linewidth=2, markersize=8)
            axes[level, 1].plot(n_vals, cldice_vals, f'{markers[model]}-',
                               label=label, color=colors[model], linewidth=2, markersize=8)
            axes[level, 2].plot(n_vals, bcs_vals, f'{markers[model]}-',
                               label=label, color=colors[model], linewidth=2, markersize=8)
            axes[level, 3].plot(n_vals, betti_vals, f'{markers[model]}-',
                               label=label, color=colors[model], linewidth=2, markersize=8)

        # Formatting
        axes[level, 0].set_ylabel(level_names[level], fontsize=10)
        axes[level, 0].set_ylim([0.5, 0.9])
        axes[level, 0].grid(True, alpha=0.3)
        axes[level, 0].legend(fontsize=9)

        axes[level, 1].set_ylim([0.5, 0.95])
        axes[level, 1].axhline(y=0.80, color='gray', linestyle='--', alpha=0.5)
        axes[level, 1].grid(True, alpha=0.3)

        axes[level, 2].set_ylim([0.0, 1.0])
        axes[level, 2].axhline(y=0.80, color='gray', linestyle='--', alpha=0.5)
        axes[level, 2].grid(True, alpha=0.3)

        axes[level, 3].set_ylim([0, 15])
        axes[level, 3].axhline(y=3.0, color='gray', linestyle='--', alpha=0.5)
        axes[level, 3].grid(True, alpha=0.3)

    # Column titles
    axes[0, 0].set_title('Dice Score', fontsize=12)
    axes[0, 1].set_title('clDice Score', fontsize=12)
    axes[0, 2].set_title('BCS (Branch Connectivity)', fontsize=12)
    axes[0, 3].set_title('Betti₀ Error', fontsize=12)

    # X-axis labels (bottom row only)
    for col in range(4):
        axes[3, col].set_xlabel('Training Cases', fontsize=11)

    plt.tight_layout()

    output_path = Path("/path/to/project/ablation_study_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    plt.close()


def print_ablation_table():
    """Print results as a formatted table."""

    results_file = Path("/path/to/project/ablation_results.json")

    if not results_file.exists():
        print("No results file found.")
        return

    with open(results_file) as f:
        results = json.load(f)

    print("\n" + "="*100)
    print("PROGRESSIVE ABLATION STUDY RESULTS")
    print("="*100)
    print(f"{'Level':<8} {'Config':<12} {'Model':<8} {'N':<6} {'Dice':<8} {'clDice':<8} {'BCS':<8} {'Betti₀':<8}")
    print("-"*100)

    for key in sorted(results.keys()):
        r = results[key]
        bcs = r.get('bcs', 0.0)
        print(f"L{r['ablation_level']:<7} {r['ablation_name']:<12} {r['model']:<8} "
              f"{r['n_train']:<6} {r['dice']:.4f}   {r['cldice']:.4f}   {bcs:.4f}   {r['betti_error']:.2f}")

    print("="*100)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="EXP-028: Progressive Ablation Study")
    parser.add_argument("--model", type=str, default='unet', choices=['unet', 'ctfm'],
                       help="Model type: unet or ctfm")
    parser.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3],
                       help="Ablation level: 0=baseline, 1=+vessel, 2=+cldice, 3=+DS")
    parser.add_argument("--n_train", type=int, default=100,
                       help="Number of training cases")
    parser.add_argument("--plot", action="store_true",
                       help="Generate comparison plot from existing results")
    parser.add_argument("--table", action="store_true",
                       help="Print results table")
    parser.add_argument("--run_all", action="store_true",
                       help="Run all 40 experiments")
    parser.add_argument("--ctfm_only", action="store_true",
                       help="Run only CT-FM experiments (skip U-Net)")

    args = parser.parse_args()

    if args.plot:
        plot_ablation_results()
    elif args.table:
        print_ablation_table()
    elif args.run_all:
        # Run all 40 experiments: 4 levels × 2 models × 5 data regimes
        data_regimes = [50, 100, 200, 350, 638]

        # Load existing results to skip completed experiments
        config = Config()
        existing_results = {}
        if config.results_file.exists():
            with open(config.results_file) as f:
                existing_results = json.load(f)
        print(f"Checking for existing results in: {config.results_file}")

        # Choose models based on flag
        models = ['ctfm'] if args.ctfm_only else ['unet', 'ctfm']

        for level in range(4):
            for model in models:
                for n in data_regimes:
                    key = f"L{level}_{model}_{n}"
                    if key in existing_results:
                        print(f"\n[SKIP] {key} already completed - skipping")
                        continue
                    print(f"\n{'#'*70}")
                    print(f"# Running: Level {level} | {model.upper()} | {n} cases")
                    print(f"{'#'*70}")
                    train_model(model_type=model, ablation_level=level, n_train_cases=n)

        # Generate plots and table
        plot_ablation_results()
        print_ablation_table()
    else:
        # Single experiment
        train_model(model_type=args.model, ablation_level=args.level, n_train_cases=args.n_train)


if __name__ == "__main__":
    main()

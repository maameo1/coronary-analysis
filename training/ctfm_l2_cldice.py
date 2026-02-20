#!/usr/bin/env python3
"""
================================================================================
EXP-022f: CT-FM + VESSELNESS + clDice (Pure Topology Loss)
================================================================================

Building on EXP-017b (CT-FM + Vesselness), adding ONLY clDice loss.
No auxiliary centerline task - just direct topology enforcement.

This is the cleanest test of whether clDice improves vessel connectivity.

Key difference from EXP-022b/c/d:
- NO centerline auxiliary decoder
- NO centerline prediction task
- ONLY clDice as additional loss term

Architecture (same as EXP-017b):
    CT-FM Encoder (pretrained, fine-tuned)
              ↓
        Shared Features
          ↓         ↓
    Seg Decoder  Vessel Decoder
          ↓           ↓
    Dice+CE+clDice  MSE Loss

Loss:
    L_total = λ_dice × Dice + λ_ce × CE + λ_cldice × clDice
            + λ_vessel × Vesselness_MSE

Hypothesis:
- clDice directly constrains segmentation topology
- No gradient interference from auxiliary tasks
- Should improve connectivity without hurting overall Dice

Target metrics:
- Overall Dice: 0.749 → 0.77+
- Better vessel connectivity (qualitative)

Author: Anonymous
Date: December 2025
Experiment: EXP-022f

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

print("="*70)
print("EXP-022f: CT-FM + Vesselness + clDice (Pure Topology)")
print("="*70)


# ============================================================================
# SOFT CLDICE LOSS (Topology-Preserving)
# ============================================================================

def soft_erode(img):
    """Soft morphological erosion for differentiable skeletonization."""
    if len(img.shape) == 5:  # 3D: [B, C, D, H, W]
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)
    elif len(img.shape) == 4:  # 2D: [B, C, H, W]
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)


def soft_dilate(img):
    """Soft morphological dilation."""
    if len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))
    elif len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img):
    """Soft morphological opening."""
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_=50):
    """
    Soft skeletonization via iterative erosion.
    
    This extracts the "centerline" of a shape by iteratively eroding
    and keeping track of what gets removed at each step.
    
    Reference: https://github.com/jocpae/clDice
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    
    return skel


class SoftclDiceLoss(nn.Module):
    """
    Soft centerline Dice loss for topology preservation.
    
    clDice measures how well the predicted skeleton matches the ground truth
    skeleton, ensuring vessel connectivity is preserved.
    
    clDice = 2 * (|S_pred ∩ V_true| + |S_true ∩ V_pred|) / (|S_pred| + |S_true|)
    
    where S = skeleton, V = volume
    
    Reference: https://arxiv.org/abs/2003.07311
    """
    
    def __init__(self, iter_=50, smooth=1e-5):
        """
        Args:
            iter_: Number of iterations for soft skeletonization
            smooth: Smoothing factor to prevent division by zero
        """
        super().__init__()
        self.iter_ = iter_
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: predicted probabilities [B, C, D, H, W] (after softmax)
            target: ground truth [B, C, D, H, W] (one-hot)
        
        Returns:
            1 - clDice (loss to minimize)
        """
        # Get foreground channel if multi-channel
        if pred.shape[1] > 1:
            pred = pred[:, 1:2]  # Vessel channel
            target = target[:, 1:2]
        
        # Compute soft skeletons
        skel_pred = soft_skel(pred, self.iter_)
        skel_target = soft_skel(target, self.iter_)
        
        # Topology precision: skeleton of pred inside target volume
        tprec = ((skel_pred * target).sum() + self.smooth) / (skel_pred.sum() + self.smooth)
        
        # Topology sensitivity: skeleton of target inside pred volume  
        tsens = ((skel_target * pred).sum() + self.smooth) / (skel_target.sum() + self.smooth)
        
        # clDice
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        
        return 1.0 - cl_dice


# ============================================================================
# VESSELNESS TARGET GENERATION (same as EXP-017b)
# ============================================================================

class AddVesselnessd(MapTransform):
    """Generate Frangi vesselness target from image."""
    
    def __init__(self, keys, frangi_sigmas=range(1, 4), frangi_alpha=0.5, 
                 frangi_beta=0.5, frangi_gamma=15):
        super().__init__(keys)
        self.frangi_sigmas = frangi_sigmas
        self.frangi_alpha = frangi_alpha
        self.frangi_beta = frangi_beta
        self.frangi_gamma = frangi_gamma
    
    def __call__(self, data):
        d = dict(data)
        
        img = d["image"]
        img_np = img[0].cpu().numpy() if torch.is_tensor(img) else img[0]
        
        # Frangi vesselness filter
        vesselness = frangi(
            img_np,
            sigmas=self.frangi_sigmas,
            alpha=self.frangi_alpha,
            beta=self.frangi_beta,
            gamma=self.frangi_gamma,
            black_ridges=False
        )
        
        # Normalize to [0, 1]
        if vesselness.max() > vesselness.min():
            vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min())
        else:
            vesselness = np.zeros_like(vesselness)
        
        d["vesselness"] = torch.from_numpy(vesselness).unsqueeze(0).float()
        
        return d


# ============================================================================
# MULTI-TASK MODEL (Same as EXP-017b - NO centerline decoder)
# ============================================================================

class MultiTaskCTFM(nn.Module):
    """
    CT-FM with vesselness auxiliary task only.
    
    Architecture:
        CT-FM Encoder (pretrained)
              ↓
        Bottleneck [512, 6, 6, 6]
              ↓
        ┌─────┼─────┐
        ↓           ↓
    Seg Decoder  Vessel Decoder
        ↓           ↓
    [2, 96³]    [1, 96³]
    """
    
    def __init__(self, pretrained_model_name="project-lighter/ct_fm_segresnet",
                 init_filters=32, aux_decoder_filters=16):
        super().__init__()
        
        print(f"Loading CT-FM from {pretrained_model_name}...")
        
        # Load pretrained CT-FM
        self.ctfm = SegResNet.from_pretrained(pretrained_model_name)
        
        self.init_filters = init_filters
        self.bottleneck_channels = 512
        
        # Vesselness decoder only (same as EXP-017b)
        self.vessel_decoder = self._create_aux_decoder(
            in_channels=self.bottleneck_channels,
            out_channels=1,
            filters=aux_decoder_filters
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        vessel_params = sum(p.numel() for p in self.vessel_decoder.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Vesselness decoder: {vessel_params:,}")
    
    def _create_aux_decoder(self, in_channels, out_channels, filters):
        """Create lightweight decoder for auxiliary task."""
        return nn.Sequential(
            # 6 -> 12
            nn.ConvTranspose3d(in_channels, filters * 8, kernel_size=2, stride=2),
            nn.InstanceNorm3d(filters * 8),
            nn.ReLU(inplace=True),
            # 12 -> 24
            nn.ConvTranspose3d(filters * 8, filters * 4, kernel_size=2, stride=2),
            nn.InstanceNorm3d(filters * 4),
            nn.ReLU(inplace=True),
            # 24 -> 48
            nn.ConvTranspose3d(filters * 4, filters * 2, kernel_size=2, stride=2),
            nn.InstanceNorm3d(filters * 2),
            nn.ReLU(inplace=True),
            # 48 -> 96
            nn.ConvTranspose3d(filters * 2, filters, kernel_size=2, stride=2),
            nn.InstanceNorm3d(filters),
            nn.ReLU(inplace=True),
            # Final conv
            nn.Conv3d(filters, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _run_decoder(self, enc_out):
        """Run the CT-FM segmentation decoder."""
        x = enc_out[-1]
        
        for i, up_layer in enumerate(self.ctfm.up_layers):
            x = up_layer['upsample'](x)
            skip_idx = len(enc_out) - 2 - i
            if skip_idx >= 0:
                x = x + enc_out[skip_idx]
            x = up_layer['blocks'](x)
            x = up_layer['head'](x)
        
        return x
    
    def forward(self, x):
        """
        Forward pass with two outputs (same as EXP-017b).
        
        Returns:
            seg_out: [B, 2, D, H, W] - segmentation logits
            vessel_out: [B, 1, D, H, W] - vesselness prediction
        """
        # Encoder
        enc_out = self.ctfm.encoder(x)
        
        # Segmentation decoder
        seg_out = self._run_decoder(enc_out)
        
        # Vesselness decoder from bottleneck
        bottleneck = enc_out[-1]
        vessel_out = self.vessel_decoder(bottleneck)
        
        return seg_out, vessel_out


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for EXP-022f"""

    exp_name = "ctfm_vesselness_cldice"
    exp_id = "EXP-022f"
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    data_root = Path("/path/to/datasets/ImageCAS")
    split_file = Path("/path/to/datasets/ImageCAS/split_1000.csv")

    # Resume from checkpoint
    resume_from = None
    output_root = None

    # Data configuration
    n_train_cases = 100
    test_fold = 1
    val_fraction = 0.15
    voxel_spacing = (1.0, 1.0, 1.0)
    patch_size = (96, 96, 96)

    # Intensity normalization
    hu_min = -200
    hu_max = 600

    # Frangi settings
    frangi_sigmas = range(1, 4)
    frangi_alpha = 0.5
    frangi_beta = 0.5
    frangi_gamma = 15

    # Model
    model_name = "project-lighter/ct_fm_segresnet"
    init_filters = 32
    aux_decoder_filters = 16

    # Training hyperparameters
    batch_size = 1
    num_workers = 4
    learning_rate = 1e-4
    weight_decay = 1e-5
    max_epochs = 100
    val_interval = 1

    # ==========================================================================
    # LOSS WEIGHTS - Key change: adding clDice to EXP-017b
    # ==========================================================================
    dice_weight = 0.4          # Standard Dice (same as 017b)
    ce_weight = 0.2            # Cross-entropy (same as 017b)
    cldice_weight = 0.2        # NEW: clDice for topology
    vessel_weight = 0.2        # Vesselness auxiliary (same as 017b)
    
    # clDice settings
    cldice_iterations = 50     # Soft skeletonization iterations

    # Optimization
    early_stopping_patience = 20
    lr_scheduler_patience = 10
    lr_scheduler_factor = 0.5

    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True
    random_seed = 2025

    def __init__(self, resume_from=None):
        if resume_from is not None:
            self.resume_from = Path(resume_from)
            self.output_root = self.resume_from.parent.parent
            print(f"Resuming from checkpoint: {self.resume_from}")
        else:
            self.resume_from = None
            self.output_root = Path(f"/path/to/project/experiments/{self.exp_id}_{self.exp_name}_{self.date}")

        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "checkpoints").mkdir(exist_ok=True)
        (self.output_root / "logs").mkdir(exist_ok=True)
        (self.output_root / "visualizations").mkdir(exist_ok=True)
        
        if not resume_from:
            self.save_config()

    def save_config(self):
        config_dict = {}
        for k, v in vars(self).items():
            if not k.startswith('_'):
                if isinstance(v, Path):
                    config_dict[k] = str(v)
                elif isinstance(v, torch.device):
                    config_dict[k] = str(v)
                elif isinstance(v, range):
                    config_dict[k] = list(v)
                else:
                    config_dict[k] = v

        with open(self.output_root / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(config, mode='train'):
    """Create transform pipeline with vesselness target."""
    
    all_keys = ["image", "label", "vesselness"]

    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config.voxel_spacing,
                mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                            a_min=config.hu_min, a_max=config.hu_max,
                            b_min=0.0, b_max=1.0, clip=True),
        
        # Generate vesselness target
        AddVesselnessd(
            keys=["image"],
            frangi_sigmas=config.frangi_sigmas,
            frangi_alpha=config.frangi_alpha,
            frangi_beta=config.frangi_beta,
            frangi_gamma=config.frangi_gamma
        ),
        
        CropForegroundd(keys=all_keys, source_key="image"),
        SpatialPadd(keys=all_keys, spatial_size=config.patch_size, mode="constant"),
        RandCropByPosNegLabeld(
            keys=all_keys,
            label_key="label",
            spatial_size=config.patch_size,
            pos=1, neg=1, num_samples=1,
            image_key="image",
            image_threshold=0,
            allow_smaller=False
        ),
        EnsureTyped(keys=all_keys),
    ]

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
    """
    Train for one epoch.
    
    Loss = dice_weight × Dice + ce_weight × CE + cldice_weight × clDice
         + vessel_weight × Vesselness_MSE
    """
    model.train()
    
    dice_loss_fn, ce_loss_fn, cldice_loss_fn, vessel_loss_fn = loss_fns
    
    epoch_losses = {
        'total': 0, 'dice': 0, 'ce': 0, 'cldice': 0, 'vessel': 0
    }
    
    for batch_data in train_loader:
        inputs = batch_data["image"].to(config.device)
        seg_labels = batch_data["label"].to(config.device)
        vessel_labels = batch_data["vesselness"].to(config.device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                seg_pred, vessel_pred = model(inputs)

                # Dice loss
                dice_loss = dice_loss_fn(seg_pred, seg_labels)
                
                # CE loss
                ce_loss = ce_loss_fn(seg_pred, seg_labels.squeeze(1).long())
                
                # clDice loss (needs softmax pred and one-hot target)
                seg_softmax = F.softmax(seg_pred, dim=1)
                seg_onehot = torch.zeros_like(seg_softmax)
                seg_onehot.scatter_(1, seg_labels.long(), 1)
                cldice_loss = cldice_loss_fn(seg_softmax, seg_onehot)
                
                # Vesselness loss
                vessel_loss = vessel_loss_fn(vessel_pred, vessel_labels)
                
                # Combined loss
                total_loss = (config.dice_weight * dice_loss +
                            config.ce_weight * ce_loss +
                            config.cldice_weight * cldice_loss +
                            config.vessel_weight * vessel_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_pred, vessel_pred = model(inputs)

            dice_loss = dice_loss_fn(seg_pred, seg_labels)
            ce_loss = ce_loss_fn(seg_pred, seg_labels.squeeze(1).long())
            
            seg_softmax = F.softmax(seg_pred, dim=1)
            seg_onehot = torch.zeros_like(seg_softmax)
            seg_onehot.scatter_(1, seg_labels.long(), 1)
            cldice_loss = cldice_loss_fn(seg_softmax, seg_onehot)
            
            vessel_loss = vessel_loss_fn(vessel_pred, vessel_labels)
            
            total_loss = (config.dice_weight * dice_loss +
                        config.ce_weight * ce_loss +
                        config.cldice_weight * cldice_loss +
                        config.vessel_weight * vessel_loss)

            total_loss.backward()
            optimizer.step()

        # Track losses
        epoch_losses['total'] += total_loss.item()
        epoch_losses['dice'] += dice_loss.item()
        epoch_losses['ce'] += ce_loss.item()
        epoch_losses['cldice'] += cldice_loss.item()
        epoch_losses['vessel'] += vessel_loss.item()

    # Average
    n = len(train_loader)
    for k in epoch_losses:
        epoch_losses[k] /= n

    return epoch_losses


# ============================================================================
# VALIDATION
# ============================================================================

def validate(model, val_loader, metric, config, post_pred, post_label):
    """Validate model."""
    model.eval()
    metric.reset()

    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data["image"].to(config.device)
            labels = batch_data["label"].to(config.device)

            seg_out, _ = model(inputs)

            outputs = [post_pred(i) for i in decollate_batch(seg_out)]
            labels_dec = [post_label(i) for i in decollate_batch(labels)]
            metric(y_pred=outputs, y=labels_dec)

    return metric.aggregate().item()


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_visualization(model, batch_data, epoch, config, dice_score):
    """Save visualization of predictions."""
    model.eval()

    inputs = batch_data["image"].to(config.device)
    labels = batch_data["label"]

    with torch.no_grad():
        seg_out, vessel_out = model(inputs)
        seg_pred = torch.argmax(seg_out, dim=1)
        
        # Compute soft skeleton for visualization
        seg_softmax = F.softmax(seg_out, dim=1)
        soft_skel_pred = soft_skel(seg_softmax[:, 1:2], iter_=50)

    # Convert to numpy
    input_np = inputs[0, 0].cpu().numpy()
    label_np = labels[0, 0].cpu().numpy()
    pred_np = seg_pred[0].cpu().numpy()
    vessel_np = vessel_out[0, 0].cpu().numpy()
    skel_np = soft_skel_pred[0, 0].cpu().numpy()

    # Middle slice
    mid = input_np.shape[0] // 2

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Row 1: Segmentation
    axes[0, 0].imshow(input_np[mid], cmap='gray')
    axes[0, 0].set_title('CT Input')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(label_np[mid], cmap='Reds')
    axes[0, 1].set_title('GT Segmentation')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_np[mid], cmap='Blues')
    axes[0, 2].set_title('Pred Segmentation')
    axes[0, 2].axis('off')

    # Error map
    error = np.zeros((*label_np[mid].shape, 3))
    error[:, :, 0] = (pred_np[mid] > 0) & (label_np[mid] == 0)  # FP = red
    error[:, :, 2] = (pred_np[mid] == 0) & (label_np[mid] > 0)  # FN = blue
    error[:, :, 1] = (pred_np[mid] > 0) & (label_np[mid] > 0)   # TP = green
    axes[0, 3].imshow(error)
    axes[0, 3].set_title('FP(red) FN(blue) TP(green)')
    axes[0, 3].axis('off')

    # Row 2: Vesselness and Skeleton
    axes[1, 0].imshow(vessel_np[mid], cmap='hot')
    axes[1, 0].set_title('Pred Vesselness')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(skel_np[mid], cmap='hot')
    axes[1, 1].set_title('Soft Skeleton (from pred)')
    axes[1, 1].axis('off')

    # Overlay: CT + skeleton
    axes[1, 2].imshow(input_np[mid], cmap='gray')
    axes[1, 2].imshow(skel_np[mid], cmap='hot', alpha=0.7)
    axes[1, 2].set_title('CT + Soft Skeleton')
    axes[1, 2].axis('off')

    # Overlay: Segmentation + skeleton
    axes[1, 3].imshow(pred_np[mid], cmap='Blues', alpha=0.5)
    axes[1, 3].imshow(skel_np[mid], cmap='Reds', alpha=0.8)
    axes[1, 3].set_title('Pred Seg + Skeleton')
    axes[1, 3].axis('off')

    plt.suptitle(f'Epoch {epoch} - Dice: {dice_score:.4f} (EXP-022f: Vesselness + clDice)', fontsize=14)
    plt.tight_layout()

    save_path = config.output_root / "visualizations" / f"epoch_{epoch:03d}_dice_{dice_score:.3f}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  Saved visualization: {save_path}")


def plot_training_curves(history, config):
    """Plot training/validation loss curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = history['epoch']

    # Total loss
    axes[0, 0].plot(epochs, history['train_loss'], linewidth=2, color='#1f77b4')
    axes[0, 0].set_title('Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Dice loss
    axes[0, 1].plot(epochs, history['dice_loss'], linewidth=2, color='#ff7f0e')
    axes[0, 1].set_title('Dice Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # clDice loss
    axes[0, 2].plot(epochs, history['cldice_loss'], linewidth=2, color='#2ca02c')
    axes[0, 2].set_title('clDice Loss (Topology)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].grid(True, alpha=0.3)

    # CE loss
    axes[1, 0].plot(epochs, history['ce_loss'], linewidth=2, color='#d62728')
    axes[1, 0].set_title('CE Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    # Vessel loss
    axes[1, 1].plot(epochs, history['vessel_loss'], linewidth=2, color='#9467bd')
    axes[1, 1].set_title('Vesselness Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
    axes[1, 1].grid(True, alpha=0.3)

    # Validation Dice
    if history['val_dice']:
        axes[1, 2].plot(epochs[:len(history['val_dice'])], history['val_dice'],
                       linewidth=2, color='#17becf', marker='o', markersize=4)
        axes[1, 2].set_title('Validation Dice Score', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Dice Score')
        axes[1, 2].grid(True, alpha=0.3)

        best_dice = max(history['val_dice'])
        axes[1, 2].axhline(y=best_dice, color='red', linestyle='--',
                          alpha=0.5, label=f'Best: {best_dice:.4f}')
        axes[1, 2].axhline(y=0.749, color='green', linestyle='--',
                          alpha=0.5, label='EXP-017b: 0.749')
        axes[1, 2].legend()

    plt.suptitle('EXP-022f: CT-FM + Vesselness + clDice',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    save_path = config.output_root / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved training curves: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def find_latest_checkpoint(experiments_dir):
    """Find the most recent experiment's latest checkpoint."""
    experiments_dir = Path(experiments_dir)
    exp_dirs = sorted(experiments_dir.glob("EXP-022f_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    for exp_dir in exp_dirs:
        checkpoint_path = exp_dir / "checkpoints" / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            return checkpoint_path
    return None


def main(resume_from=None):
    """Main training function."""
    
    # Handle auto-resume
    if resume_from == "auto":
        resume_from = find_latest_checkpoint("/path/to/project/experiments")
        if resume_from:
            print(f"Auto-detected checkpoint: {resume_from}")
        else:
            print("No existing checkpoint found. Starting fresh training.")
            resume_from = None

    config = Config(resume_from=resume_from)

    print(f"\nExperiment: {config.exp_id} - {config.exp_name}")
    print(f"Output: {config.output_root}")
    print(f"Device: {config.device}")

    print(f"\n What's different from EXP-017b:")
    print(f"   + clDice loss (weight={config.cldice_weight})")
    print(f"   = Same vesselness auxiliary task")
    print(f"   = Same architecture (no centerline decoder)")

    print(f"\n Loss weights:")
    print(f"   Dice:    {config.dice_weight}")
    print(f"   CE:      {config.ce_weight}")
    print(f"   clDice:  {config.cldice_weight} ← NEW")
    print(f"   Vessel:  {config.vessel_weight}")

    print(f"\n Target: Dice 0.749 → 0.77+")

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Data
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)

    data_loader = ImageCASDataLoader(
        data_root=config.data_root,
        split_file=config.split_file,
        test_fold=config.test_fold,
        val_fraction=config.val_fraction,
        random_seed=42
    )

    splits = data_loader.get_splits(n_train_cases=config.n_train_cases, regime_seed=2025)
    train_data = splits['train']
    val_data = splits['val']

    print(f"\nTraining: {len(train_data)} cases")
    print(f"Validation: {len(val_data)} cases")

    # Save splits
    split_info = {
        'train_case_ids': [d['case_id'] for d in train_data],
        'val_case_ids': [d['case_id'] for d in val_data],
        'test_case_ids': [d['case_id'] for d in splits['test']],
    }
    with open(config.output_root / "data_splits.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    # Transforms
    train_transforms = get_transforms(config, mode='train')
    val_transforms = get_transforms(config, mode='val')

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True,
                              collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True,
                            collate_fn=pad_list_data_collate)

    # Model
    print("\n" + "="*70)
    print("MODEL CREATION")
    print("="*70)

    model = MultiTaskCTFM(
        pretrained_model_name=config.model_name,
        init_filters=config.init_filters,
        aux_decoder_filters=config.aux_decoder_filters
    ).to(config.device)

    # Verify
    print("\nVerifying model...")
    with torch.no_grad():
        test_input = torch.randn(1, 1, 96, 96, 96).to(config.device)
        seg, vessel = model(test_input)
        print(f"   Seg output: {seg.shape}")
        print(f"   Vessel output: {vessel.shape}")
    del test_input, seg, vessel
    torch.cuda.empty_cache()

    # Loss functions
    dice_loss_fn = DiceLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    ce_loss_fn = nn.CrossEntropyLoss()
    cldice_loss_fn = SoftclDiceLoss(iter_=config.cldice_iterations)
    vessel_loss_fn = nn.MSELoss()

    loss_fns = (dice_loss_fn, ce_loss_fn, cldice_loss_fn, vessel_loss_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience)

    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    scaler = torch.amp.GradScaler('cuda') if config.amp else None
    writer = SummaryWriter(log_dir=config.output_root / "logs")

    # Training state
    start_epoch = 0
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'val_dice': [],
               'dice_loss': [], 'ce_loss': [], 'cldice_loss': [], 'vessel_loss': []}

    # Load checkpoint if resuming
    if config.resume_from is not None and config.resume_from.exists():
        print("\n" + "="*70)
        print("LOADING CHECKPOINT")
        print("="*70)
        checkpoint = torch.load(config.resume_from, map_location=config.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        patience_counter = checkpoint.get('patience_counter', 0)

        if 'history' in checkpoint:
            history = checkpoint['history']

        print(f"Resumed from epoch {start_epoch}")
        print(f"Best Dice so far: {best_dice:.4f} (epoch {best_epoch})")

    # Training
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    for epoch in range(start_epoch, config.max_epochs):
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        print("-" * 50)

        epoch_losses = train_epoch(model, train_loader, optimizer, loss_fns, config, scaler)

        print(f"Loss - Total: {epoch_losses['total']:.4f}, "
              f"Dice: {epoch_losses['dice']:.4f}, "
              f"CE: {epoch_losses['ce']:.4f}, "
              f"clDice: {epoch_losses['cldice']:.4f}, "
              f"Vessel: {epoch_losses['vessel']:.4f}")

        # Log to tensorboard
        for k, v in epoch_losses.items():
            writer.add_scalar(f"Loss/{k}", v, epoch)

        # Track history
        history['epoch'].append(epoch)
        history['train_loss'].append(epoch_losses['total'])
        history['dice_loss'].append(epoch_losses['dice'])
        history['ce_loss'].append(epoch_losses['ce'])
        history['cldice_loss'].append(epoch_losses['cldice'])
        history['vessel_loss'].append(epoch_losses['vessel'])

        # Validation
        if (epoch + 1) % config.val_interval == 0:
            val_dice = validate(model, val_loader, dice_metric, config, post_pred, post_label)
            print(f"Val Dice: {val_dice:.4f}")

            writer.add_scalar("Dice/val", val_dice, epoch)
            history['val_dice'].append(val_dice)

            scheduler.step(val_dice)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

            # Save visualization every 10 epochs
            if (epoch + 1) % 10 == 0:
                for batch in val_loader:
                    save_visualization(model, batch, epoch+1, config, val_dice)
                    break

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': val_dice,
                    'config': {
                        'cldice_weight': config.cldice_weight,
                        'n_train_cases': config.n_train_cases,
                    }
                }, config.output_root / "checkpoints" / "best_model.pth")

                print(f"⭐ New best! Dice: {best_dice:.4f}")

                # Save visualization for best model
                for batch in val_loader:
                    save_visualization(model, batch, epoch+1, config, val_dice)
                    break
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                print(f"\n⏹ Early stopping at epoch {epoch+1}")
                break

        # Save latest checkpoint
        latest_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'best_dice': best_dice,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'history': history,
        }
        torch.save(latest_checkpoint, config.output_root / "checkpoints" / "latest_checkpoint.pth")

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, config.output_root / "checkpoints" / "final_model.pth")

    writer.close()

    # Save history
    with open(config.output_root / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(history, config)

    # Results
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Dice: {best_dice:.4f} (epoch {best_epoch})")

    print(f"\n📊 Comparison:")
    print(f"   EXP-017b (CT-FM + Vesselness):      0.749")
    print(f"   EXP-022f (+ clDice):                {best_dice:.4f} ({(best_dice-0.749)/0.749*100:+.1f}%)")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    if best_dice >= 0.77:
        print("✅ SUCCESS! clDice improved performance")
        print("   → Topology-preserving loss works")
        print("   → Proceed with data efficiency experiments")
    elif best_dice >= 0.755:
        print("🔶 PARTIAL: Small improvement")
        print("   → Try adjusting cldice_weight (0.1 or 0.3)")
    elif best_dice >= 0.749:
        print("⚪ NEUTRAL: No change from baseline")
        print("   → clDice neither helps nor hurts")
    else:
        print("❌ REGRESSION: clDice hurt performance")
        print("   → May need lower weight or different formulation")

    # Save results
    results = {
        'exp_id': config.exp_id,
        'best_dice': best_dice,
        'best_epoch': best_epoch,
        'n_train_cases': config.n_train_cases,
        'loss_weights': {
            'dice': config.dice_weight,
            'ce': config.ce_weight,
            'cldice': config.cldice_weight,
            'vessel': config.vessel_weight,
        },
        'comparison': {
            'exp017b_baseline': 0.749,
            'exp022f_result': best_dice,
            'improvement': best_dice - 0.749
        }
    }
    with open(config.output_root / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {config.output_root}")

    return best_dice


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EXP-022f: CT-FM + Vesselness + clDice")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from. Use 'auto' to find latest checkpoint automatically."
    )
    args = parser.parse_args()
    main(resume_from=args.resume)
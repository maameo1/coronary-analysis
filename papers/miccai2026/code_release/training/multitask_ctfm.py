#!/usr/bin/env python3
"""
================================================================================
MULTI-TASK CT-FM: FOUNDATION MODEL + VESSELNESS AUXILIARY TASK
================================================================================

Combines:
1. CT-FM pretrained encoder (148K CT scans)
2. Multi-task learning (segmentation + vesselness prediction)

Hypothesis: Are CT-FM (+7.1%) and Multi-Task (+1.3%) complementary?
- If additive: expect +8-9% improvement
- If redundant: ~7% (CT-FM already captures vessel features)
- If negative: <7% (auxiliary task interferes)

Architecture:
    CT-FM Encoder (pretrained, fine-tuned)
              ↓
        Shared Features
          ↓         ↓
    Seg Decoder  Vesselness Decoder
          ↓             ↓
      Dice Loss    MSE Loss (Frangi target)

    Combined Loss = Dice + λ × Vesselness

Features:
- True feature sharing from CT-FM encoder
- GradCAM visualization for interpretability
- Comparison across λ values (0.1, 0.5, 1.0)

Author: Anonymous
Date: November 2025
Experiment: EXP-017 (Multi-Task CT-FM)

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
import nibabel as nib

from lighter_zoo import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import pad_list_data_collate
from monai.networks.blocks import Convolution, ResidualUnit

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

# Frangi filter for generating ground truth vesselness maps
try:
    from skimage.filters import frangi
    FRANGI_AVAILABLE = True
    print("✓ Frangi filter available")
except ImportError:
    print("✗ ERROR: scikit-image required")
    print("  Install with: pip install scikit-image")
    FRANGI_AVAILABLE = False


# ============================================================================
# GRADCAM FOR 3D MEDICAL IMAGES
# ============================================================================

class GradCAM3D:
    """
    Gradient-weighted Class Activation Mapping for 3D volumes
    
    Helps visualize what regions the model focuses on for predictions.
    Useful for understanding if CT-FM learns vessel-relevant features.
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
    """
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The neural network
            target_layer: Layer to compute CAM from (usually last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layer"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=1):
        """
        Generate GradCAM heatmap
        
        Args:
            input_tensor: Input image [B, C, H, W, D]
            target_class: Class to generate CAM for (1 = vessel)
        
        Returns:
            cam: Heatmap [H, W, D] normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Handle multi-task output
        if isinstance(output, tuple):
            seg_output = output[0]  # [B, 2, H, W, D]
        else:
            seg_output = output
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        # Sum over spatial dimensions to get scalar
        target = seg_output[:, target_class, :, :, :].sum()
        target.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, h, w, d]
        activations = self.activations  # [B, C, h, w, d]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)  # [B, C, 1, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, h, w, d]
        
        # ReLU and normalize
        cam = F.relu(cam)
        
        # Upsample to input size
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        # Normalize to [0, 1]
        cam = cam.squeeze()  # [H, W, D]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.cpu().numpy()


class GradCAMPlusPlus3D(GradCAM3D):
    """
    Grad-CAM++ for better localization
    
    Uses second-order gradients for improved visualization.
    """
    
    def generate_cam(self, input_tensor, target_class=1):
        """Generate Grad-CAM++ heatmap"""
        self.model.eval()
        
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            seg_output = output[0]
        else:
            seg_output = output
        
        self.model.zero_grad()
        target = seg_output[:, target_class, :, :, :].sum()
        target.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        # Grad-CAM++ weights
        grad_2 = gradients ** 2
        grad_3 = grad_2 * gradients
        
        sum_activations = activations.sum(dim=(2, 3, 4), keepdim=True)
        alpha_numer = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_numer / alpha_denom
        
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3, 4), keepdim=True)
        
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        cam = cam.squeeze()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam.cpu().numpy()


# ============================================================================
# CUSTOM TRANSFORM FOR VESSELNESS GROUND TRUTH
# ============================================================================

class AddVesselnessGroundTruthd(MapTransform):
    """
    Generate Frangi vesselness map as auxiliary ground truth
    
    Takes: {"image": [1, H, W, D], "label": [1, H, W, D]}
    Returns: {"image": ..., "label": ..., "vesselness": [1, H, W, D]}
    """
    
    def __init__(self, keys, sigmas=range(1, 4), alpha=0.5, beta=0.5, gamma=15):
        super().__init__(keys)
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        if not FRANGI_AVAILABLE:
            raise ImportError("scikit-image required")
    
    def __call__(self, data):
        d = dict(data)
        
        img = d["image"]  # [1, H, W, D]
        img_np = img[0].cpu().numpy() if torch.is_tensor(img) else img[0]
        
        vesselness = frangi(
            img_np,
            sigmas=self.sigmas,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            black_ridges=False
        )
        
        # Normalize to [0, 1]
        if vesselness.max() > vesselness.min():
            vesselness = (vesselness - vesselness.min()) / \
                        (vesselness.max() - vesselness.min())
        else:
            vesselness = np.zeros_like(vesselness)
        
        vesselness_tensor = torch.from_numpy(vesselness).unsqueeze(0).float()
        d["vesselness"] = vesselness_tensor
        
        return d


# ============================================================================
# MULTI-TASK CT-FM MODEL WITH TRUE FEATURE SHARING
# ============================================================================

class MultiTaskCTFM(nn.Module):
    """
    CT-FM with dual output heads sharing encoder features
    
    Architecture (based on CT-FM SegResNet inspection):
        Input [1, 96, 96, 96]
              ↓
        CT-FM Encoder (5 levels, pretrained)
              ↓
        Returns list of skip connections:
          - Level 0: [B, 32, 96, 96, 96]
          - Level 1: [B, 64, 48, 48, 48]
          - Level 2: [B, 128, 24, 24, 24]
          - Level 3: [B, 256, 12, 12, 12]
          - Level 4: [B, 512, 6, 6, 6]  ← Bottleneck
              ↓
        ┌─────┴─────┐
        ↓           ↓
    Seg Decoder  Vessel Decoder (from bottleneck)
        ↓           ↓
    [2, 96³]    [1, 96³]
    
    The vesselness decoder is lighter than segmentation decoder
    since it's an auxiliary task.
    """
    
    def __init__(self, pretrained_model_name="project-lighter/ct_fm_segresnet",
                 init_filters=32, vessel_decoder_filters=16):
        super().__init__()
        
        print(f"Loading CT-FM from {pretrained_model_name}...")
        
        # Load pretrained CT-FM
        self.ctfm = SegResNet.from_pretrained(pretrained_model_name)
        
        # Store architecture info
        self.init_filters = init_filters
        
        # Encoder output is a LIST of 5 skip connections
        # Bottleneck (index 4) is [B, 512, 6, 6, 6]
        self.bottleneck_channels = 512
        self.bottleneck_size = 6
        
        # Create vesselness decoder (lighter than main decoder)
        # Input: bottleneck [512, 6, 6, 6] → Output: [1, 96, 96, 96]
        self.vessel_decoder = self._create_vessel_decoder(
            in_channels=self.bottleneck_channels,
            out_channels=1,
            filters=vessel_decoder_filters
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        vessel_params = sum(p.numel() for p in self.vessel_decoder.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Vesselness decoder parameters: {vessel_params:,}")
    
    def _create_vessel_decoder(self, in_channels, out_channels, filters):
        """
        Create lightweight decoder for vesselness prediction
        
        Upsamples from 6³ → 96³ (4 stages of 2x upsampling = 16x total)
        Input: [B, 512, 6, 6, 6]
        Output: [B, 1, 96, 96, 96]
        """
        return nn.Sequential(
            # Upsample 6 -> 12
            nn.ConvTranspose3d(in_channels, filters * 8, kernel_size=2, stride=2),
            nn.InstanceNorm3d(filters * 8),
            nn.ReLU(inplace=True),
            
            # Upsample 12 -> 24
            nn.ConvTranspose3d(filters * 8, filters * 4, kernel_size=2, stride=2),
            nn.InstanceNorm3d(filters * 4),
            nn.ReLU(inplace=True),
            
            # Upsample 24 -> 48
            nn.ConvTranspose3d(filters * 4, filters * 2, kernel_size=2, stride=2),
            nn.InstanceNorm3d(filters * 2),
            nn.ReLU(inplace=True),
            
            # Upsample 48 -> 96
            nn.ConvTranspose3d(filters * 2, filters, kernel_size=2, stride=2),
            nn.InstanceNorm3d(filters),
            nn.ReLU(inplace=True),
            
            # Final conv
            nn.Conv3d(filters, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _run_decoder(self, enc_out):
        """
        Manually run the CT-FM decoder (up_layers)
        
        This replicates the forward pass of the decoder part of SegResNet
        """
        x = enc_out[-1]  # Start from bottleneck
        
        # up_layers expect skip connections in reverse order
        for i, up_layer in enumerate(self.ctfm.up_layers):
            # Upsample
            x = up_layer['upsample'](x)
            
            # Add skip connection (from earlier encoder level)
            skip_idx = len(enc_out) - 2 - i  # -2 because we started from -1
            if skip_idx >= 0:
                x = x + enc_out[skip_idx]
            
            # Apply blocks
            x = up_layer['blocks'](x)
            
            # Apply head (only last layer has actual conv, others are Identity)
            x = up_layer['head'](x)
        
        return x
    
    def forward(self, x):
        """
        Forward pass with dual outputs
        
        Returns:
            seg_out: [B, 2, H, W, D] - segmentation logits
            vessel_out: [B, 1, H, W, D] - vesselness prediction [0, 1]
        """
        # Run encoder directly to get skip connections
        enc_out = self.ctfm.encoder(x)  # List of 5 tensors
        
        # Run segmentation decoder
        seg_out = self._run_decoder(enc_out)
        
        # Run vesselness decoder from bottleneck
        bottleneck = enc_out[-1]  # [B, 512, 6, 6, 6]
        vessel_out = self.vessel_decoder(bottleneck)
        
        return seg_out, vessel_out
    
    def _fallback_vesselness(self, seg_out):
        """Fallback if encoder hook doesn't work"""
        # Use softmax of vessel class as proxy
        probs = F.softmax(seg_out, dim=1)
        vessel_prob = probs[:, 1:2, :, :, :]  # [B, 1, H, W, D]
        return vessel_prob
    
    def get_encoder_features(self, x):
        """Get encoder features for visualization"""
        return self.ctfm.encoder(x)
    
    def get_bottleneck(self, x):
        """Get just the bottleneck features"""
        enc_out = self.ctfm.encoder(x)
        return enc_out[-1] if enc_out else None


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for Multi-Task CT-FM experiment"""
    
    exp_name = "multitask_ctfm"
    exp_id = "EXP-017"
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    data_root = Path("/path/to/datasets/ImageCAS")
    split_file = Path("/path/to/datasets/ImageCAS/split_1000.csv")
    output_root = Path(f"/path/to/project/experiments/{exp_id}_{exp_name}_{date}")
    
    # Data configuration
    n_train_cases = 100  # Start with 100 to compare with other experiments
    test_fold = 1
    val_fraction = 0.15
    voxel_spacing = (1.0, 1.0, 1.0)
    patch_size = (96, 96, 96)
    
    # Intensity normalization
    hu_min = -200
    hu_max = 600
    
    # Frangi settings for vesselness ground truth
    frangi_sigmas = range(1, 4)
    frangi_alpha = 0.5
    frangi_beta = 0.5
    frangi_gamma = 15
    
    # Model
    model_name = "project-lighter/ct_fm_segresnet"
    init_filters = 32  # CT-FM default
    vessel_decoder_filters = 16  # Lighter decoder for auxiliary task
    
    # Training hyperparameters
    batch_size = 1
    num_workers = 4
    learning_rate = 1e-4
    weight_decay = 1e-5
    max_epochs = 100
    val_interval = 1
    
    # Multi-task loss weights
    # Loss = seg_weight × DiceCE + vessel_weight × MSE
    seg_loss_weight = 1.0
    vessel_loss_weight = 0.5  # λ - try 0.1, 0.5, 1.0
    
    # Segmentation loss components
    dice_weight = 0.5
    ce_weight = 0.5
    
    # Optimization
    early_stopping_patience = 20
    lr_scheduler_patience = 10
    lr_scheduler_factor = 0.5
    
    # Visualization
    save_gradcam = True
    gradcam_interval = 10  # Save GradCAM every N epochs
    num_vis_samples = 3  # Number of validation samples to visualize
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True
    random_seed = 2025
    
    def __init__(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "checkpoints").mkdir(exist_ok=True)
        (self.output_root / "logs").mkdir(exist_ok=True)
        (self.output_root / "visualizations").mkdir(exist_ok=True)
        (self.output_root / "gradcam").mkdir(exist_ok=True)
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
    """Create transform pipeline with vesselness ground truth"""
    
    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config.voxel_spacing, 
                mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], 
                            a_min=config.hu_min, a_max=config.hu_max, 
                            b_min=0.0, b_max=1.0, clip=True),
        
        # Generate Frangi vesselness as auxiliary target
        AddVesselnessGroundTruthd(
            keys=["image"],
            sigmas=config.frangi_sigmas,
            alpha=config.frangi_alpha,
            beta=config.frangi_beta,
            gamma=config.frangi_gamma
        ),
        
        CropForegroundd(keys=["image", "label", "vesselness"], source_key="image"),
        SpatialPadd(keys=["image", "label", "vesselness"], 
                   spatial_size=config.patch_size, mode="constant"),
        RandCropByPosNegLabeld(
            keys=["image", "label", "vesselness"],
            label_key="label",
            spatial_size=config.patch_size,
            pos=1, neg=1, num_samples=1,
            image_key="image",
            image_threshold=0,
            allow_smaller=False
        ),
        EnsureTyped(keys=["image", "label", "vesselness"]),
    ]
    
    if mode == 'train':
        train_transforms = common_transforms + [
            RandFlipd(keys=["image", "label", "vesselness"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label", "vesselness"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label", "vesselness"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label", "vesselness"], prob=0.5, spatial_axes=(0, 1)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
        return Compose(train_transforms)
    else:
        return Compose(common_transforms)


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_gradcam_visualization(model, data_batch, epoch, config, sample_idx=0):
    """
    Generate and save GradCAM visualization
    
    Creates a figure showing:
    - Input CT
    - Ground truth segmentation
    - Predicted segmentation
    - GradCAM heatmap
    - Vesselness prediction vs ground truth
    """
    model.eval()
    
    inputs = data_batch["image"].to(config.device)
    labels = data_batch["label"]
    vesselness_gt = data_batch["vesselness"]
    
    # Get predictions
    with torch.no_grad():
        seg_out, vessel_out = model(inputs)
        seg_pred = torch.argmax(seg_out, dim=1)  # [B, H, W, D]
    
    # Generate GradCAM
    try:
        # Find target layer for GradCAM
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv3d):
                target_layer = module  # Will get the last Conv3d
        
        if target_layer is not None:
            gradcam = GradCAM3D(model, target_layer)
            cam = gradcam.generate_cam(inputs, target_class=1)
        else:
            cam = np.zeros(config.patch_size)
    except Exception as e:
        print(f"  GradCAM failed: {e}")
        cam = np.zeros(config.patch_size)
    
    # Convert to numpy for visualization
    input_np = inputs[0, 0].cpu().numpy()
    label_np = labels[0, 0].cpu().numpy()
    pred_np = seg_pred[0].cpu().numpy()
    vessel_pred_np = vessel_out[0, 0].cpu().numpy()
    vessel_gt_np = vesselness_gt[0, 0].cpu().numpy()
    
    # Get middle slices
    mid_z = input_np.shape[2] // 2
    mid_y = input_np.shape[1] // 2
    mid_x = input_np.shape[0] // 2
    
    # Create visualization figure
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Row 1: Axial view (z-slice)
    axes[0, 0].imshow(input_np[:, :, mid_z], cmap='gray')
    axes[0, 0].set_title('CT Input (Axial)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(label_np[:, :, mid_z], cmap='Reds')
    axes[0, 1].set_title('GT Segmentation')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_np[:, :, mid_z], cmap='Blues')
    axes[0, 2].set_title('Pred Segmentation')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(input_np[:, :, mid_z], cmap='gray')
    axes[0, 3].imshow(cam[:, :, mid_z], cmap='jet', alpha=0.5)
    axes[0, 3].set_title('GradCAM Overlay')
    axes[0, 3].axis('off')
    
    axes[0, 4].imshow(cam[:, :, mid_z], cmap='hot')
    axes[0, 4].set_title('GradCAM Heatmap')
    axes[0, 4].axis('off')
    
    # Row 2: Coronal view (y-slice)
    axes[1, 0].imshow(input_np[:, mid_y, :], cmap='gray')
    axes[1, 0].set_title('CT Input (Coronal)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(vessel_gt_np[:, mid_y, :], cmap='hot')
    axes[1, 1].set_title('Vesselness GT (Frangi)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(vessel_pred_np[:, mid_y, :], cmap='hot')
    axes[1, 2].set_title('Vesselness Pred')
    axes[1, 2].axis('off')
    
    # Vesselness difference
    vessel_diff = np.abs(vessel_pred_np - vessel_gt_np)
    axes[1, 3].imshow(vessel_diff[:, mid_y, :], cmap='coolwarm')
    axes[1, 3].set_title('Vesselness |Error|')
    axes[1, 3].axis('off')
    
    axes[1, 4].imshow(cam[:, mid_y, :], cmap='hot')
    axes[1, 4].set_title('GradCAM (Coronal)')
    axes[1, 4].axis('off')
    
    # Row 3: Sagittal view (x-slice) + overlay
    axes[2, 0].imshow(input_np[mid_x, :, :], cmap='gray')
    axes[2, 0].set_title('CT Input (Sagittal)')
    axes[2, 0].axis('off')
    
    # Overlay: GT in red, pred in blue, overlap in purple
    overlay = np.zeros((*label_np[mid_x, :, :].shape, 3))
    overlay[:, :, 0] = label_np[mid_x, :, :]  # Red = GT
    overlay[:, :, 2] = pred_np[mid_x, :, :]   # Blue = Pred
    axes[2, 1].imshow(overlay)
    axes[2, 1].set_title('GT (red) vs Pred (blue)')
    axes[2, 1].axis('off')
    
    # Segmentation + vesselness overlay
    axes[2, 2].imshow(input_np[mid_x, :, :], cmap='gray')
    axes[2, 2].imshow(vessel_pred_np[mid_x, :, :], cmap='hot', alpha=0.5)
    axes[2, 2].set_title('CT + Vesselness Pred')
    axes[2, 2].axis('off')
    
    # GradCAM + segmentation
    axes[2, 3].imshow(cam[mid_x, :, :], cmap='hot')
    axes[2, 3].contour(label_np[mid_x, :, :], colors='cyan', linewidths=1)
    axes[2, 3].set_title('GradCAM + GT Contour')
    axes[2, 3].axis('off')
    
    # Histogram of GradCAM values inside vs outside vessels
    cam_in_vessel = cam[label_np > 0.5]
    cam_out_vessel = cam[label_np < 0.5]
    
    if len(cam_in_vessel) > 0 and len(cam_out_vessel) > 0:
        axes[2, 4].hist(cam_out_vessel.flatten(), bins=50, alpha=0.5, 
                        label='Background', density=True, color='gray')
        axes[2, 4].hist(cam_in_vessel.flatten(), bins=50, alpha=0.5, 
                        label='Vessel', density=True, color='red')
        axes[2, 4].set_xlabel('GradCAM Value')
        axes[2, 4].set_ylabel('Density')
        axes[2, 4].set_title('GradCAM Distribution')
        axes[2, 4].legend()
    else:
        axes[2, 4].axis('off')
    
    plt.suptitle(f'Epoch {epoch} - Multi-Task CT-FM Visualization', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    save_path = config.output_root / "gradcam" / f"epoch_{epoch:03d}_sample_{sample_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved GradCAM visualization: {save_path}")
    
    return cam


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, seg_loss_fn, vessel_loss_fn, 
                config, scaler=None):
    """Train for one epoch with multi-task loss"""
    model.train()
    epoch_seg_loss = 0
    epoch_vessel_loss = 0
    epoch_total_loss = 0
    
    for batch_data in train_loader:
        inputs = batch_data["image"].to(config.device)
        seg_labels = batch_data["label"].to(config.device)
        vessel_labels = batch_data["vesselness"].to(config.device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                seg_pred, vessel_pred = model(inputs)
                
                seg_loss = seg_loss_fn(seg_pred, seg_labels)
                vessel_loss = vessel_loss_fn(vessel_pred, vessel_labels)
                
                total_loss = (config.seg_loss_weight * seg_loss + 
                            config.vessel_loss_weight * vessel_loss)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_pred, vessel_pred = model(inputs)
            seg_loss = seg_loss_fn(seg_pred, seg_labels)
            vessel_loss = vessel_loss_fn(vessel_pred, vessel_labels)
            total_loss = (config.seg_loss_weight * seg_loss + 
                        config.vessel_loss_weight * vessel_loss)
            total_loss.backward()
            optimizer.step()
        
        epoch_seg_loss += seg_loss.item()
        epoch_vessel_loss += vessel_loss.item()
        epoch_total_loss += total_loss.item()
    
    n_batches = len(train_loader)
    return (epoch_total_loss / n_batches, 
            epoch_seg_loss / n_batches,
            epoch_vessel_loss / n_batches)


# ============================================================================
# VALIDATION
# ============================================================================

def validate(model, val_loader, metric, config, post_pred, post_label, 
             epoch=0, save_vis=False):
    """Validate and optionally save visualizations"""
    model.eval()
    metric.reset()
    
    vis_batches = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            inputs = batch_data["image"].to(config.device)
            labels = batch_data["label"].to(config.device)
            
            seg_out, _ = model(inputs)
            
            outputs = [post_pred(i) for i in decollate_batch(seg_out)]
            labels_dec = [post_label(i) for i in decollate_batch(labels)]
            metric(y_pred=outputs, y=labels_dec)
            
            # Collect samples for visualization
            if save_vis and len(vis_batches) < config.num_vis_samples:
                vis_batches.append(batch_data)
    
    # Save GradCAM visualizations
    if save_vis and config.save_gradcam:
        for idx, batch in enumerate(vis_batches):
            save_gradcam_visualization(model, batch, epoch, config, sample_idx=idx)
    
    return metric.aggregate().item()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training function"""
    
    config = Config()
    
    print("="*80)
    print("MULTI-TASK CT-FM EXPERIMENT")
    print("="*80)
    print(f"Experiment: {config.exp_id} - {config.exp_name}")
    print(f"Output: {config.output_root}")
    print(f"Device: {config.device}")
    print(f"\n🔬 Hypothesis: Are CT-FM and Multi-Task complementary?")
    print(f"   - CT-FM alone: +7.1% over baseline")
    print(f"   - Multi-Task alone: +1.3% over baseline")
    print(f"   - Combined: ???")
    print(f"\n⚖️  Loss weights:")
    print(f"   - Segmentation: {config.seg_loss_weight}")
    print(f"   - Vesselness (λ): {config.vessel_loss_weight}")
    
    if not FRANGI_AVAILABLE:
        print("\n ERROR: scikit-image not available!")
        return
    
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Data preparation
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
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
    
    print(f"\nUsing {len(train_data)} training cases")
    print(f" Using {len(val_data)} validation cases")
    print(f" Generating Frangi vesselness as auxiliary target")
    
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
    print("\n" + "="*80)
    print("MODEL CREATION - MULTI-TASK CT-FM")
    print("="*80)
    
    model = MultiTaskCTFM(
        pretrained_model_name=config.model_name,
        init_filters=config.init_filters,
        vessel_decoder_filters=config.vessel_decoder_filters
    ).to(config.device)
    
    # Verify model works before training
    print("\n🔍 Verifying model architecture...")
    with torch.no_grad():
        test_input = torch.randn(1, 1, 96, 96, 96).to(config.device)
        test_seg, test_vessel = model(test_input)
        print(f"   ✓ Seg output: {test_seg.shape}")
        print(f"   ✓ Vessel output: {test_vessel.shape}")
        enc_out = model.get_encoder_features(test_input)
        print(f"   ✓ Encoder outputs {len(enc_out)} skip connections")
        print(f"   ✓ Bottleneck shape: {enc_out[-1].shape}")
    del test_input, test_seg, test_vessel, enc_out
    torch.cuda.empty_cache()
    
    # Loss functions
    seg_loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True,
        lambda_dice=config.dice_weight, lambda_ce=config.ce_weight
    )
    vessel_loss_fn = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience)
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    
    scaler = torch.amp.GradScaler('cuda') if config.amp and torch.cuda.is_available() else None
    writer = SummaryWriter(log_dir=config.output_root / "logs")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_dice': [], 'seg_loss': [], 'vessel_loss': []}
    
    for epoch in range(config.max_epochs):
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        print("-" * 40)
        
        total_loss, seg_loss, vessel_loss = train_epoch(
            model, train_loader, optimizer, seg_loss_fn, vessel_loss_fn, config, scaler
        )
        
        print(f"Train - Total: {total_loss:.4f}, Seg: {seg_loss:.4f}, Vessel: {vessel_loss:.4f}")
        writer.add_scalar("Loss/train_total", total_loss, epoch)
        writer.add_scalar("Loss/train_seg", seg_loss, epoch)
        writer.add_scalar("Loss/train_vessel", vessel_loss, epoch)
        
        history['train_loss'].append(total_loss)
        history['seg_loss'].append(seg_loss)
        history['vessel_loss'].append(vessel_loss)
        
        if (epoch + 1) % config.val_interval == 0:
            # Determine if we should save visualizations this epoch
            save_vis = config.save_gradcam and ((epoch + 1) % config.gradcam_interval == 0)
            
            val_dice = validate(model, val_loader, dice_metric, config,
                               post_pred, post_label, epoch=epoch+1, save_vis=save_vis)
            print(f"Val Dice: {val_dice:.4f}")
            writer.add_scalar("Dice/val", val_dice, epoch)
            
            history['val_dice'].append(val_dice)
            
            scheduler.step(val_dice)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("Learning_rate", current_lr, epoch)
            
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
                        'vessel_loss_weight': config.vessel_loss_weight,
                        'n_train_cases': config.n_train_cases,
                    }
                }, config.output_root / "checkpoints" / "best_model.pth")
                
                print(f"⭐ New best model! Dice: {best_dice:.4f}")
                
                # Save visualization for best model
                if config.save_gradcam:
                    for batch_data in val_loader:
                        save_gradcam_visualization(model, batch_data, epoch+1, config, 
                                                  sample_idx=999)  # 999 = best model
                        break
            else:
                patience_counter += 1
            
            if patience_counter >= config.early_stopping_patience:
                print(f"\n⏹  Early stopping at epoch {epoch+1}")
                break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, config.output_root / "checkpoints" / "final_model.pth")
    
    writer.close()
    
    # Save training history
    with open(config.output_root / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Results
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation Dice: {best_dice:.4f} (epoch {best_epoch})")
    print(f"\n Comparison to baselines:")
    print(f"   - Baseline U-Net (100 cases):     0.688")
    print(f"   - Multi-Task U-Net (100 cases):   0.697 (+1.3%)")
    print(f"   - CT-FM (100 cases):              0.737 (+7.1%)")
    print(f"   - Multi-Task CT-FM (100 cases):   {best_dice:.3f} ({(best_dice-0.688)/0.688*100:+.1f}%)")
    print(f"\n Results saved to: {config.output_root}")
    
    # Interpret results
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if best_dice > 0.76:
        print(" ADDITIVE! Multi-Task + CT-FM exceeds either approach alone")
        print("   → Strong evidence for complementary mechanisms")
        print("   → Proceed to Phase 2 (extended auxiliary tasks)")
    elif best_dice > 0.73:
        print("〰️  REDUNDANT: Similar to CT-FM alone (~7%)")
        print("   → CT-FM may already capture vessel-relevant features")
        print("   → Still interesting finding - document in paper")
    else:
        print(" NEGATIVE: Auxiliary task appears to interfere")
        print("   → Try reducing λ (vessel_loss_weight)")
        print("   → Consider different auxiliary task formulation")
    
    # Save results
    results = {
        'best_dice': best_dice,
        'best_epoch': best_epoch,
        'final_epoch': epoch + 1,
        'n_train_cases': config.n_train_cases,
        'vessel_loss_weight': config.vessel_loss_weight,
        'comparison': {
            'baseline': 0.688,
            'multitask': 0.697,
            'ctfm': 0.737,
            'multitask_ctfm': best_dice
        },
        'interpretation': 'additive' if best_dice > 0.76 else ('redundant' if best_dice > 0.73 else 'negative')
    }
    with open(config.output_root / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_dice


if __name__ == "__main__":
    main()
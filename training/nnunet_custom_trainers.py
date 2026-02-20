#!/usr/bin/env python3
"""
Custom NN-UNet trainers for EXP-031 topology-aware ablation study.

These trainers implement TRUE auxiliary task learning with separate decoder heads,
matching the CT-FM architecture from EXP-028 for fair comparison.

Ablation levels (matching EXP-028):
    L0: Baseline NN-UNet (standard nnUNetTrainer)
    L1: +Vesselness auxiliary decoder (separate head predicting Frangi vesselness)
    L2: +clDice loss (vesselness decoder + clDice topology loss)
    L3: +Deep Supervision (vesselness + clDice + enhanced DS at multiple scales)

Architecture:
    nnU-Net Encoder
          ↓
    Bottleneck Features
      ↓           ↓
    Seg Decoder  Vessel Decoder (auxiliary)
      ↓              ↓
    DiceCE Loss   MSE Loss (vs Frangi target)

Author: Anonymous
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple
import numpy as np


# =============================================================================
# Topology Loss Functions
# =============================================================================

def soft_erode(img):
    """Soft morphological erosion for differentiable skeletonization."""
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    """Soft morphological dilation."""
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    """Soft morphological opening."""
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_=3):
    """Differentiable skeletonization."""
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred, target, iter_=3, smooth=1e-5):
    """
    Soft clDice loss for topology-aware training.

    Args:
        pred: Predicted segmentation (after softmax), shape [B, C, D, H, W]
        target: Ground truth one-hot, shape [B, C, D, H, W]
        iter_: Skeletonization iterations
        smooth: Smoothing factor
    """
    # Ensure we have at least 2 channels
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Get foreground channel
    pred_fg = pred[:, 1:2, ...]
    target_fg = target[:, 1:2, ...].float()

    # Skip if no foreground in target (avoids empty tensor issues)
    if target_fg.sum() < 1:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Compute soft skeletons
    skel_pred = soft_skel(pred_fg, iter_)
    skel_target = soft_skel(target_fg, iter_)

    # Topology precision and sensitivity
    tprec = ((skel_pred * target_fg).sum() + smooth) / (skel_pred.sum() + smooth)
    tsens = ((skel_target * pred_fg).sum() + smooth) / (skel_target.sum() + smooth)

    # clDice
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + smooth)

    return 1.0 - cl_dice


# =============================================================================
# Trainer Code Templates (to be installed into nnUNet)
# =============================================================================

# Common code shared by all trainers
COMMON_IMPORTS = '''
"""
NN-UNet Trainer with topology-aware training for EXP-031.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:
    from skimage.filters import frangi
    FRANGI_AVAILABLE = True
except ImportError:
    FRANGI_AVAILABLE = False
    print("Warning: skimage.filters.frangi not available, using skeleton-based vesselness")


def soft_erode(img):
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iter_=3):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred, target, iter_=3, smooth=1e-5):
    """
    Soft clDice loss.

    Args:
        pred: Predicted probabilities after softmax, shape [B, C, D, H, W]
        target: One-hot encoded target, shape [B, C, D, H, W]

    Note: For binary segmentation (2 classes), foreground is channel 1.
    We must ensure the target has at least 2 channels.
    """
    # Ensure we have at least 2 channels
    if pred.shape[1] < 2 or target.shape[1] < 2:
        # Return 0 loss if channels are missing
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_fg = pred[:, 1:2, ...]
    target_fg = target[:, 1:2, ...].float()

    # Skip if no foreground in target (avoids empty tensor issues)
    if target_fg.sum() < 1:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    skel_pred = soft_skel(pred_fg, iter_)
    skel_target = soft_skel(target_fg, iter_)
    tprec = ((skel_pred * target_fg).sum() + smooth) / (skel_pred.sum() + smooth)
    tsens = ((skel_target * pred_fg).sum() + smooth) / (skel_target.sum() + smooth)
    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + smooth)
    return 1.0 - cl_dice


def compute_frangi_vesselness(image_np, sigmas=range(1, 4), alpha=0.5, beta=0.5, gamma=15):
    """Compute Frangi vesselness filter response (matching CT-FM)."""
    if FRANGI_AVAILABLE:
        vessel_response = frangi(
            image_np,
            sigmas=list(sigmas),
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            black_ridges=False
        )
        if vessel_response.max() > 0:
            vessel_response = vessel_response / (vessel_response.max() + 1e-8)
        return vessel_response
    else:
        # Fallback: use soft skeleton of the mask
        return None


class VesselnessDecoder(nn.Module):
    """
    Auxiliary decoder for vesselness prediction.
    Takes bottleneck features and predicts vesselness map.
    Architecture matches CT-FM's vessel_decoder.
    """

    def __init__(self, in_channels, target_size):
        super().__init__()
        self.target_size = target_size

        # Decoder: upsample from bottleneck to full resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 128, kernel_size=2, stride=2),
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

    def forward(self, x):
        out = self.decoder(x)
        # Resize to match target if needed
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='trilinear', align_corners=False)
        return out


class NetworkWithVesselness(nn.Module):
    """
    Wrapper that adds vesselness auxiliary decoder to nnU-Net.

    Architecture:
        Input -> Encoder -> Bottleneck
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              Seg Decoder        Vessel Decoder
                    ↓                   ↓
              Segmentation         Vesselness
    """

    def __init__(self, base_network, patch_size=(96, 96, 96)):
        super().__init__()
        self.base_network = base_network
        self.patch_size = patch_size

        # Get bottleneck channels from the base network
        # nnU-Net's PlainConvUNet has encoder attribute
        bottleneck_channels = self._get_bottleneck_channels()

        self.vessel_decoder = VesselnessDecoder(
            in_channels=bottleneck_channels,
            target_size=patch_size
        )

        # Hook to capture bottleneck features
        self.bottleneck_features = None
        self._register_hook()

    @property
    def decoder(self):
        """Expose base network's decoder for nnU-Net compatibility."""
        return self.base_network.decoder

    @property
    def encoder(self):
        """Expose base network's encoder for nnU-Net compatibility."""
        return self.base_network.encoder

    def _get_bottleneck_channels(self):
        """Get number of channels at bottleneck."""
        # nnU-Net's PlainConvUNet structure
        if hasattr(self.base_network, 'encoder'):
            # Get the last stage channels
            stages = self.base_network.encoder.stages
            last_stage = stages[-1]
            # Get output channels from last conv
            for module in last_stage.modules():
                if isinstance(module, nn.Conv3d):
                    return module.out_channels
        # Default fallback
        return 320

    def _register_hook(self):
        """Register forward hook to capture bottleneck features."""
        def hook(module, input, output):
            self.bottleneck_features = output

        if hasattr(self.base_network, 'encoder'):
            # Hook the last encoder stage
            self.base_network.encoder.stages[-1].register_forward_hook(hook)

    def forward(self, x):
        # Run base network (captures bottleneck via hook)
        seg_output = self.base_network(x)

        # During inference (not training), return only segmentation tensor
        # This is required for nnUNet's predictor which expects tensor output
        if not self.training:
            return seg_output

        # During training, return dict with auxiliary outputs
        # Get vesselness from bottleneck
        # DETACH: stop vesselness gradients from corrupting shared encoder
        # The vessel decoder learns from bottleneck features but doesn't
        # pull the encoder away from segmentation-optimal representations
        if self.bottleneck_features is not None:
            vessel_output = self.vessel_decoder(self.bottleneck_features.detach())
        else:
            # Fallback: return zeros
            vessel_output = torch.zeros(x.shape[0], 1, *self.patch_size, device=x.device)

        return {'seg': seg_output, 'vessel': vessel_output}

'''

TRAINER_L1_VESSELNESS = COMMON_IMPORTS + '''

class nnUNetTrainerVesselness(nnUNetTrainer):
    """
    NN-UNet trainer with vesselness auxiliary decoder (L1).

    Adds a separate decoder head that predicts vesselness maps,
    matching the CT-FM architecture from EXP-028.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.vesselness_weight = 0.2
        self.print_to_log_file(f"L1: Vesselness auxiliary decoder with weight {self.vesselness_weight}")

        # Frangi filter parameters (matching CT-FM)
        self.frangi_sigmas = range(1, 4)
        self.frangi_alpha = 0.5
        self.frangi_beta = 0.5
        self.frangi_gamma = 15

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        """Build network with vesselness decoder."""
        # Build base network using parent static method
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )

        # Wrap with vesselness decoder (default patch size; actual size set during training)
        wrapped_network = NetworkWithVesselness(base_network, patch_size=(96, 96, 96))

        return wrapped_network

    def _compute_vesselness_target(self, image, mask):
        """Compute vesselness target from input image."""
        # Move to CPU for Frangi filter
        image_np = image.cpu().numpy()

        batch_vesselness = []
        for b in range(image_np.shape[0]):
            img = image_np[b, 0]  # [D, H, W]

            vessel = compute_frangi_vesselness(
                img,
                sigmas=self.frangi_sigmas,
                alpha=self.frangi_alpha,
                beta=self.frangi_beta,
                gamma=self.frangi_gamma
            )

            if vessel is None:
                # Fallback: use soft skeleton
                mask_tensor = mask[b:b+1].float()
                if mask_tensor.ndim == 4:
                    mask_tensor = mask_tensor.unsqueeze(0)
                vessel = soft_skel(mask_tensor, iter_=5).squeeze().cpu().numpy()

            batch_vesselness.append(vessel)

        vesselness = np.stack(batch_vesselness, axis=0)[:, np.newaxis]
        return torch.from_numpy(vesselness.astype(np.float32)).to(image.device)

    def train_step(self, batch):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)

            # Handle dict output from wrapped network
            if isinstance(output, dict):
                seg_output = output['seg']
                vessel_pred = output['vessel']
            else:
                seg_output = output
                vessel_pred = None

            # Base segmentation loss
            base_loss = self.loss(seg_output, target)

            # Vesselness auxiliary loss
            if vessel_pred is not None:
                # Get target mask for vesselness computation
                if isinstance(target, list):
                    tgt = target[0]
                else:
                    tgt = target

                # Compute Frangi vesselness target
                with torch.no_grad():
                    vessel_target = self._compute_vesselness_target(data, tgt)

                # Resize vessel_pred to match target (nnU-Net patch may differ)
                if vessel_pred.shape[2:] != vessel_target.shape[2:]:
                    vessel_pred = F.interpolate(
                        vessel_pred, size=vessel_target.shape[2:],
                        mode='trilinear', align_corners=False
                    )
                # MSE loss on vesselness prediction
                vessel_loss = F.mse_loss(vessel_pred, vessel_target)

                total_loss = base_loss + self.vesselness_weight * vessel_loss
            else:
                total_loss = base_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    def validation_step(self, batch):
        """Override to handle dict output and compute validation metrics."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Set network to eval mode temporarily to get tensor output
        self.network.eval()

        with torch.autocast(self.device.type, enabled=True):
            with torch.no_grad():
                output = self.network(data)

                # Handle dict output
                if isinstance(output, dict):
                    seg_output = output['seg']
                else:
                    seg_output = output

                # Handle list output (from deep supervision) - take full resolution
                if isinstance(seg_output, list):
                    seg_output = seg_output[0]

                # Compute loss - use unwrapped loss to handle tensor inputs
                # self.loss may be DeepSupervisionWrapper which expects lists
                base_loss_fn = self.loss.loss if hasattr(self.loss, 'loss') else self.loss
                target_for_loss = target[0] if isinstance(target, list) else target
                l = base_loss_fn(seg_output, target_for_loss)

                # Compute validation metrics (tp, fp, fn)
                # Get predicted segmentation
                if self.label_manager.has_regions:
                    predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
                else:
                    predicted_segmentation_onehot = seg_output.argmax(1)[:, None]
                    predicted_segmentation_onehot = torch.zeros(
                        (predicted_segmentation_onehot.shape[0], self.label_manager.num_segmentation_heads,
                         *predicted_segmentation_onehot.shape[2:]),
                        device=predicted_segmentation_onehot.device, dtype=torch.float32
                    ).scatter_(1, predicted_segmentation_onehot, 1)

                # Get ground truth
                if self.label_manager.has_regions:
                    if isinstance(target, list):
                        target_onehot = target[0]
                    else:
                        target_onehot = target
                else:
                    if isinstance(target, list):
                        target_for_metrics = target[0][:, 0:1]
                    else:
                        target_for_metrics = target[:, 0:1]
                    target_onehot = torch.zeros_like(predicted_segmentation_onehot)
                    target_onehot.scatter_(1, target_for_metrics.long(), 1)

                # Compute tp, fp, fn
                tp = (predicted_segmentation_onehot * target_onehot).sum(dim=(2, 3, 4))
                fp = (predicted_segmentation_onehot * (1 - target_onehot)).sum(dim=(2, 3, 4))
                fn = ((1 - predicted_segmentation_onehot) * target_onehot).sum(dim=(2, 3, 4))

        self.network.train()

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }

    def predict_step(self, batch, return_probabilities=False):
        """Override to handle dict output during inference."""
        data = batch['data']
        data = data.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)

                # Handle dict output (from custom wrappers)
                if isinstance(output, dict):
                    seg_output = output['seg']
                else:
                    seg_output = output

                # Handle list output (from deep supervision) - take full resolution
                if isinstance(seg_output, list):
                    seg_output = seg_output[0]

        return seg_output
'''

TRAINER_L2_CLDICE = COMMON_IMPORTS + '''

class nnUNetTrainerClDice(nnUNetTrainer):
    """
    NN-UNet trainer with vesselness auxiliary decoder + clDice loss (L2).

    Combines:
    - Vesselness auxiliary decoder (from L1)
    - clDice topology-aware loss
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.vesselness_weight = 0.2
        self.cldice_weight = 0.3
        self.print_to_log_file(f"L2: Vesselness ({self.vesselness_weight}) + clDice ({self.cldice_weight})")

        # Frangi filter parameters
        self.frangi_sigmas = range(1, 4)
        self.frangi_alpha = 0.5
        self.frangi_beta = 0.5
        self.frangi_gamma = 15

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        """Build network with vesselness decoder."""
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )

        wrapped_network = NetworkWithVesselness(base_network, patch_size=(96, 96, 96))

        return wrapped_network

    def _compute_vesselness_target(self, image, mask):
        """Compute vesselness target from input image."""
        image_np = image.cpu().numpy()

        batch_vesselness = []
        for b in range(image_np.shape[0]):
            img = image_np[b, 0]

            vessel = compute_frangi_vesselness(
                img,
                sigmas=self.frangi_sigmas,
                alpha=self.frangi_alpha,
                beta=self.frangi_beta,
                gamma=self.frangi_gamma
            )

            if vessel is None:
                mask_tensor = mask[b:b+1].float()
                if mask_tensor.ndim == 4:
                    mask_tensor = mask_tensor.unsqueeze(0)
                vessel = soft_skel(mask_tensor, iter_=5).squeeze().cpu().numpy()

            batch_vesselness.append(vessel)

        vesselness = np.stack(batch_vesselness, axis=0)[:, np.newaxis]
        return torch.from_numpy(vesselness.astype(np.float32)).to(image.device)

    def train_step(self, batch):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)

            if isinstance(output, dict):
                seg_output = output['seg']
                vessel_pred = output['vessel']
            else:
                seg_output = output
                vessel_pred = None

            # Base segmentation loss
            base_loss = self.loss(seg_output, target)

            # Get prediction for clDice
            if isinstance(seg_output, (list, tuple)):
                pred = seg_output[0]
            else:
                pred = seg_output

            pred_softmax = torch.softmax(pred, dim=1)

            # Get target for losses
            if isinstance(target, list):
                tgt = target[0]
            else:
                tgt = target

            # One-hot encode target for clDice
            if tgt.ndim == 4:
                tgt_onehot = torch.zeros_like(pred_softmax)
                tgt_onehot.scatter_(1, tgt.unsqueeze(1).long(), 1)
            else:
                tgt_onehot = tgt

            # clDice loss
            cldice_loss = soft_cldice_loss(pred_softmax, tgt_onehot)

            # Vesselness auxiliary loss
            if vessel_pred is not None:
                with torch.no_grad():
                    vessel_target = self._compute_vesselness_target(data, tgt)
                # Resize vessel_pred to match target (nnU-Net patch may differ)
                if vessel_pred.shape[2:] != vessel_target.shape[2:]:
                    vessel_pred = F.interpolate(
                        vessel_pred, size=vessel_target.shape[2:],
                        mode='trilinear', align_corners=False
                    )
                vessel_loss = F.mse_loss(vessel_pred, vessel_target)
            else:
                vessel_loss = 0.0

            total_loss = base_loss + self.vesselness_weight * vessel_loss + self.cldice_weight * cldice_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    def validation_step(self, batch):
        """Override to handle dict output and compute validation metrics."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Set network to eval mode temporarily to get tensor output
        self.network.eval()

        with torch.autocast(self.device.type, enabled=True):
            with torch.no_grad():
                output = self.network(data)

                if isinstance(output, dict):
                    seg_output = output['seg']
                else:
                    seg_output = output

                # Handle list output (from deep supervision) - take full resolution
                if isinstance(seg_output, list):
                    seg_output = seg_output[0]

                # Compute loss - use unwrapped loss to handle tensor inputs
                base_loss_fn = self.loss.loss if hasattr(self.loss, 'loss') else self.loss
                target_for_loss = target[0] if isinstance(target, list) else target
                l = base_loss_fn(seg_output, target_for_loss)

                # Compute validation metrics (tp, fp, fn)
                if self.label_manager.has_regions:
                    predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
                else:
                    predicted_segmentation_onehot = seg_output.argmax(1)[:, None]
                    predicted_segmentation_onehot = torch.zeros(
                        (predicted_segmentation_onehot.shape[0], self.label_manager.num_segmentation_heads,
                         *predicted_segmentation_onehot.shape[2:]),
                        device=predicted_segmentation_onehot.device, dtype=torch.float32
                    ).scatter_(1, predicted_segmentation_onehot, 1)

                if self.label_manager.has_regions:
                    if isinstance(target, list):
                        target_onehot = target[0]
                    else:
                        target_onehot = target
                else:
                    if isinstance(target, list):
                        target_for_metrics = target[0][:, 0:1]
                    else:
                        target_for_metrics = target[:, 0:1]
                    target_onehot = torch.zeros_like(predicted_segmentation_onehot)
                    target_onehot.scatter_(1, target_for_metrics.long(), 1)

                tp = (predicted_segmentation_onehot * target_onehot).sum(dim=(2, 3, 4))
                fp = (predicted_segmentation_onehot * (1 - target_onehot)).sum(dim=(2, 3, 4))
                fn = ((1 - predicted_segmentation_onehot) * target_onehot).sum(dim=(2, 3, 4))

        self.network.train()

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }

    def predict_step(self, batch, return_probabilities=False):
        """Override to handle dict output during inference."""
        data = batch['data']
        data = data.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)

                # Handle dict output (from custom wrappers)
                if isinstance(output, dict):
                    seg_output = output['seg']
                else:
                    seg_output = output

                # Handle list output (from deep supervision) - take full resolution
                if isinstance(seg_output, list):
                    seg_output = seg_output[0]

        return seg_output
'''

TRAINER_L3_FULL = COMMON_IMPORTS + '''

class NetworkWithVesselnessAndDS(nn.Module):
    """
    nnU-Net wrapper with vesselness decoder AND deep supervision heads.
    Matches CT-FM architecture from EXP-028.

    Architecture:
        Input -> Encoder -> Bottleneck -> Decoder
                    ↓           ↓            ↓
              (features)   Vessel Dec   DS Heads (from decoder)
                               ↓             ↓
                          Vesselness    Multi-scale seg

    Deep supervision is applied at decoder intermediate levels (like CT-FM),
    NOT encoder levels.
    """

    def __init__(self, base_network, patch_size=(96, 96, 96), num_classes=2):
        super().__init__()
        self.base_network = base_network
        self.patch_size = patch_size
        self.num_classes = num_classes

        bottleneck_channels = self._get_bottleneck_channels()

        self.vessel_decoder = VesselnessDecoder(
            in_channels=bottleneck_channels,
            target_size=patch_size
        )

        # Deep supervision heads - will be created dynamically on first forward pass
        # since we don't know decoder channel counts until then
        self.ds_heads = nn.ModuleDict()
        self.ds_heads_initialized = False

        # Hooks to capture features
        self.bottleneck_features = None
        self.decoder_features = {}
        self._register_hooks()

    @property
    def decoder(self):
        """Expose base network's decoder for nnU-Net compatibility."""
        return self.base_network.decoder

    @property
    def encoder(self):
        """Expose base network's encoder for nnU-Net compatibility."""
        return self.base_network.encoder

    def _get_bottleneck_channels(self):
        if hasattr(self.base_network, 'encoder'):
            stages = self.base_network.encoder.stages
            last_stage = stages[-1]
            for module in last_stage.modules():
                if isinstance(module, nn.Conv3d):
                    return module.out_channels
        return 320

    def _register_hooks(self):
        """Register hooks to capture bottleneck and decoder features."""
        # Hook bottleneck (last encoder stage)
        if hasattr(self.base_network, 'encoder'):
            def bottleneck_hook(module, input, output):
                self.bottleneck_features = output
            self.base_network.encoder.stages[-1].register_forward_hook(bottleneck_hook)

        # Hook decoder stages to capture intermediate features
        if hasattr(self.base_network, 'decoder'):
            for i, stage in enumerate(self.base_network.decoder.stages):
                def make_hook(idx):
                    def hook(module, input, output):
                        self.decoder_features[idx] = output
                    return hook
                stage.register_forward_hook(make_hook(i))

    def _init_ds_heads(self, device):
        """Initialize deep supervision heads based on actual decoder feature channels."""
        if self.ds_heads_initialized:
            return

        # Get channel counts from captured decoder features
        for idx, feat in self.decoder_features.items():
            in_channels = feat.shape[1]
            head_name = f'ds_level{idx}'
            if head_name not in self.ds_heads:
                self.ds_heads[head_name] = nn.Conv3d(in_channels, self.num_classes, kernel_size=1).to(device)
                print(f"Created DS head {head_name} with {in_channels} input channels")

        self.ds_heads_initialized = True

    def forward(self, x):
        self.decoder_features = {}

        # Run base network
        seg_output = self.base_network(x)

        # During inference (not training), return only segmentation tensor
        # This is required for nnUNet's predictor which expects tensor output
        if not self.training:
            return seg_output

        # During training, return dict with auxiliary outputs
        # Get vesselness from bottleneck
        # DETACH: stop vesselness gradients from corrupting shared encoder
        if self.bottleneck_features is not None:
            vessel_output = self.vessel_decoder(self.bottleneck_features.detach())
        else:
            vessel_output = torch.zeros(x.shape[0], 1, *self.patch_size, device=x.device)

        # Initialize DS heads on first forward pass (now we know channel counts)
        if not self.ds_heads_initialized and len(self.decoder_features) > 0:
            self._init_ds_heads(x.device)

        # Deep supervision from decoder intermediate features
        ds_outputs = {}

        # Use first two decoder stages for deep supervision
        # (matching CT-FM which uses 2 levels)
        ds_levels_to_use = [0, 1]  # First two decoder stages

        for idx in ds_levels_to_use:
            if idx in self.decoder_features:
                feat = self.decoder_features[idx]
                head_name = f'ds_level{idx}'

                if head_name in self.ds_heads:
                    ds_out = self.ds_heads[head_name](feat)
                    ds_out = F.interpolate(ds_out, size=self.patch_size, mode='trilinear', align_corners=False)
                    ds_outputs[head_name] = ds_out

        return {
            'seg': seg_output,
            'vessel': vessel_output,
            'ds_level0': ds_outputs.get('ds_level0'),
            'ds_level1': ds_outputs.get('ds_level1'),
        }


class nnUNetTrainerTopologyFull(nnUNetTrainer):
    """
    NN-UNet trainer with full topology-aware training (L3).
    Matches CT-FM L3 configuration from EXP-028.

    Combines:
    - Vesselness auxiliary decoder (predicts Frangi vesselness)
    - clDice topology loss
    - Deep supervision at 2 decoder scales (DiceCE loss, like CT-FM)

    Loss = DiceCE + 0.2*Vesselness + 0.3*clDice + 0.3*DS_level2 + 0.3*DS_level3
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.vesselness_weight = 0.2
        self.cldice_weight = 0.3
        self.ds_weight_level2 = 0.3  # Match CT-FM
        self.ds_weight_level3 = 0.3  # Match CT-FM
        self.print_to_log_file(f"L3: Full topology training (matching CT-FM)")
        self.print_to_log_file(f"  Vesselness: {self.vesselness_weight}")
        self.print_to_log_file(f"  clDice: {self.cldice_weight}")
        self.print_to_log_file(f"  DS weights: level2={self.ds_weight_level2}, level3={self.ds_weight_level3}")

        self.frangi_sigmas = range(1, 4)
        self.frangi_alpha = 0.5
        self.frangi_beta = 0.5
        self.frangi_gamma = 15

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision):
        """Build network with vesselness decoder and deep supervision."""
        base_network = super().build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )

        patch_size = self.configuration_manager.patch_size
        wrapped_network = NetworkWithVesselnessAndDS(
            base_network,
            patch_size=tuple(patch_size),
            num_classes=num_output_channels
        )

        self.print_to_log_file(f"  Added vesselness decoder + deep supervision")

        return wrapped_network

    def _compute_vesselness_target(self, image, mask):
        """Compute vesselness target from input image."""
        image_np = image.cpu().numpy()

        batch_vesselness = []
        for b in range(image_np.shape[0]):
            img = image_np[b, 0]

            vessel = compute_frangi_vesselness(
                img,
                sigmas=self.frangi_sigmas,
                alpha=self.frangi_alpha,
                beta=self.frangi_beta,
                gamma=self.frangi_gamma
            )

            if vessel is None:
                mask_tensor = mask[b:b+1].float()
                if mask_tensor.ndim == 4:
                    mask_tensor = mask_tensor.unsqueeze(0)
                vessel = soft_skel(mask_tensor, iter_=5).squeeze().cpu().numpy()

            batch_vesselness.append(vessel)

        vesselness = np.stack(batch_vesselness, axis=0)[:, np.newaxis]
        return torch.from_numpy(vesselness.astype(np.float32)).to(image.device)

    def train_step(self, batch):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)

            if isinstance(output, dict):
                seg_output = output['seg']
                vessel_pred = output['vessel']
                ds_level0 = output.get('ds_level0')
                ds_level1 = output.get('ds_level1')
            else:
                seg_output = output
                vessel_pred = None
                ds_level0 = None
                ds_level1 = None

            # Base segmentation loss (DiceCE)
            base_loss = self.loss(seg_output, target)

            # Get prediction for clDice
            if isinstance(seg_output, (list, tuple)):
                pred = seg_output[0]
            else:
                pred = seg_output

            pred_softmax = torch.softmax(pred, dim=1)

            # Get target
            if isinstance(target, list):
                tgt = target[0]
            else:
                tgt = target

            # One-hot encode target for clDice
            if tgt.ndim == 4:
                tgt_onehot = torch.zeros_like(pred_softmax)
                tgt_onehot.scatter_(1, tgt.unsqueeze(1).long(), 1)
            else:
                tgt_onehot = tgt

            # clDice loss
            cldice_loss = soft_cldice_loss(pred_softmax, tgt_onehot)

            # Vesselness auxiliary loss
            if vessel_pred is not None:
                with torch.no_grad():
                    vessel_target = self._compute_vesselness_target(data, tgt)
                # Resize vessel_pred to match target (nnU-Net patch may differ)
                if vessel_pred.shape[2:] != vessel_target.shape[2:]:
                    vessel_pred = F.interpolate(
                        vessel_pred, size=vessel_target.shape[2:],
                        mode='trilinear', align_corners=False
                    )
                vessel_loss = F.mse_loss(vessel_pred, vessel_target)
            else:
                vessel_loss = 0.0

            # Deep supervision losses (DiceCE like CT-FM, NOT clDice)
            # Use self.loss.loss to get the unwrapped loss (not DeepSupervisionWrapper)
            ds_loss = 0.0
            base_loss_fn = self.loss.loss if hasattr(self.loss, 'loss') else self.loss

            # Deep supervision from decoder level 0
            if ds_level0 is not None:
                # For DS outputs, use the full-resolution target (first element if list)
                ds_target = target[0] if isinstance(target, list) else target
                ds0_loss = base_loss_fn(ds_level0, ds_target)
                ds_loss = ds_loss + self.ds_weight_level2 * ds0_loss

            # Deep supervision from decoder level 1
            if ds_level1 is not None:
                ds_target = target[0] if isinstance(target, list) else target
                ds1_loss = base_loss_fn(ds_level1, ds_target)
                ds_loss = ds_loss + self.ds_weight_level3 * ds1_loss

            # Combine all losses (matching CT-FM L3)
            total_loss = (
                base_loss +
                self.vesselness_weight * vessel_loss +
                self.cldice_weight * cldice_loss +
                ds_loss
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    def validation_step(self, batch):
        """Override to handle dict output and compute validation metrics."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Set network to eval mode temporarily to get tensor output
        self.network.eval()

        with torch.autocast(self.device.type, enabled=True):
            with torch.no_grad():
                output = self.network(data)

                if isinstance(output, dict):
                    seg_output = output['seg']
                else:
                    seg_output = output

                # Handle list output (from deep supervision) - take full resolution
                if isinstance(seg_output, list):
                    seg_output = seg_output[0]

                # Compute loss - use unwrapped loss to handle tensor inputs
                base_loss_fn = self.loss.loss if hasattr(self.loss, 'loss') else self.loss
                target_for_loss = target[0] if isinstance(target, list) else target
                l = base_loss_fn(seg_output, target_for_loss)

                # Compute validation metrics (tp, fp, fn)
                if self.label_manager.has_regions:
                    predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
                else:
                    predicted_segmentation_onehot = seg_output.argmax(1)[:, None]
                    predicted_segmentation_onehot = torch.zeros(
                        (predicted_segmentation_onehot.shape[0], self.label_manager.num_segmentation_heads,
                         *predicted_segmentation_onehot.shape[2:]),
                        device=predicted_segmentation_onehot.device, dtype=torch.float32
                    ).scatter_(1, predicted_segmentation_onehot, 1)

                if self.label_manager.has_regions:
                    if isinstance(target, list):
                        target_onehot = target[0]
                    else:
                        target_onehot = target
                else:
                    if isinstance(target, list):
                        target_for_metrics = target[0][:, 0:1]
                    else:
                        target_for_metrics = target[:, 0:1]
                    target_onehot = torch.zeros_like(predicted_segmentation_onehot)
                    target_onehot.scatter_(1, target_for_metrics.long(), 1)

                tp = (predicted_segmentation_onehot * target_onehot).sum(dim=(2, 3, 4))
                fp = (predicted_segmentation_onehot * (1 - target_onehot)).sum(dim=(2, 3, 4))
                fn = ((1 - predicted_segmentation_onehot) * target_onehot).sum(dim=(2, 3, 4))

        self.network.train()

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }

    def predict_step(self, batch, return_probabilities=False):
        """Override to handle dict output during inference."""
        data = batch['data']
        data = data.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)

                # Handle dict output (from custom wrappers)
                if isinstance(output, dict):
                    seg_output = output['seg']
                else:
                    seg_output = output

                # Handle list output (from deep supervision) - take full resolution
                if isinstance(seg_output, list):
                    seg_output = seg_output[0]

        return seg_output
'''


TRAINER_L3_BCS_GUIDED = COMMON_IMPORTS + '''
from scipy import ndimage as scipy_ndimage
from skimage.morphology import skeletonize as ski_skeletonize


def compute_bifurcation_weight_map_patch(label_patch, sigma=5.0, alpha=3.0):
    """
    Compute bifurcation weight map from a GT label patch (on-the-fly).

    Uses GT bifurcation locations (always valid) rather than noisy model
    predictions. This avoids failure mode #4 (noisy intermediate predictions).

    Args:
        label_patch: Binary GT patch, shape [D, H, W]
        sigma: Gaussian sigma in voxels
        alpha: Peak weight multiplier (weight = 1 + alpha at bifurcation center)

    Returns:
        weight_map: Shape [D, H, W], values >= 1.0 (vessel) or 0.5 (background)
    """
    gt = label_patch > 0.5

    if gt.sum() < 100:
        return np.ones_like(gt, dtype=np.float32)

    # Skeletonize
    skel = ski_skeletonize(gt)

    # Find bifurcation points (>= 3 neighbors in 26-connected skeleton)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = scipy_ndimage.convolve(
        skel.astype(np.uint8), kernel, mode='constant', cval=0
    )
    bif_points = (skel > 0) & (neighbor_count >= 3)
    n_bif = bif_points.sum()

    if n_bif == 0:
        # No bifurcations in this patch — uniform weight
        weight_map = np.ones_like(gt, dtype=np.float32)
        weight_map[~gt] = 0.5
        return weight_map

    # Distance transform from bifurcation points
    bif_distance = scipy_ndimage.distance_transform_edt(~bif_points)

    # Gaussian weighting: higher near bifurcations
    bif_weight = alpha * np.exp(-0.5 * (bif_distance / sigma) ** 2)

    # Weight map: base 1.0 for vessel + bif_weight, 0.5 for background
    weight_map = np.ones_like(gt, dtype=np.float32)
    weight_map += bif_weight.astype(np.float32)
    weight_map[~gt] = 0.5

    return weight_map


class nnUNetTrainerBCSGuided(nnUNetTrainer):
    """
    NN-UNet with vesselness decoder + BCS-guided bifurcation weighting.

    Builds on L1 (vesselness auxiliary decoder) by upweighting Dice/CE loss
    near GT bifurcation junctions. Weight maps computed on-the-fly from GT
    patches (avoids noisy online prediction issue).

    Mitigations incorporated:
    - GT bifurcation locations used (always valid, no noisy predictions)
    - Warmup phase: first 30 epochs use standard DiceCE
    - Modest weight magnitude: peak 4x (alpha=3.0)

    Loss (after warmup):
        WeightedDiceCE(seg, gt, w_bif)
        + 0.2 * MSE(vessel_pred, frangi_target)
        + 0.3 * BifFocalDice(seg, gt, w_bif)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.vesselness_weight = 0.2
        self.bif_focal_weight = 0.3
        self.bif_sigma = 5.0
        self.bif_alpha = 3.0
        self.bif_weight_threshold = 2.0
        self.warmup_epochs = 30

        # Early stopping
        self.es_patience = 50
        self.es_min_epochs = 100

        self.print_to_log_file(
            f"L3-BCS: Vesselness ({self.vesselness_weight}) + "
            f"BCS-guided (focal={self.bif_focal_weight}, sigma={self.bif_sigma}, "
            f"alpha={self.bif_alpha}, warmup={self.warmup_epochs})"
        )
        self.print_to_log_file(
            f"  Early stopping: patience={self.es_patience}, min_epochs={self.es_min_epochs}"
        )

        self.frangi_sigmas = range(1, 4)
        self.frangi_alpha = 0.5
        self.frangi_beta = 0.5
        self.frangi_gamma = 15

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        """Build network with vesselness decoder (same as L1)."""
        base_network = nnUNetTrainer.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision
        )
        wrapped_network = NetworkWithVesselness(base_network, patch_size=(96, 96, 96))
        return wrapped_network

    def _compute_vesselness_target(self, image, mask):
        """Compute Frangi vesselness target from input image."""
        image_np = image.cpu().numpy()

        batch_vesselness = []
        for b in range(image_np.shape[0]):
            img = image_np[b, 0]
            vessel = compute_frangi_vesselness(
                img, sigmas=self.frangi_sigmas, alpha=self.frangi_alpha,
                beta=self.frangi_beta, gamma=self.frangi_gamma
            )
            if vessel is None:
                mask_tensor = mask[b:b+1].float()
                if mask_tensor.ndim == 4:
                    mask_tensor = mask_tensor.unsqueeze(0)
                vessel = soft_skel(mask_tensor, iter_=5).squeeze().cpu().numpy()
            batch_vesselness.append(vessel)

        vesselness = np.stack(batch_vesselness, axis=0)[:, np.newaxis]
        return torch.from_numpy(vesselness.astype(np.float32)).to(image.device)

    def _compute_bif_weights(self, target):
        """Compute bifurcation weight maps on-the-fly from GT patches."""
        if isinstance(target, list):
            tgt = target[0]
        else:
            tgt = target

        tgt_np = tgt.cpu().numpy()
        batch_weights = []

        for b in range(tgt_np.shape[0]):
            if tgt_np.ndim == 5:
                label_patch = tgt_np[b, 0]
            else:
                label_patch = tgt_np[b]

            w = compute_bifurcation_weight_map_patch(
                label_patch, sigma=self.bif_sigma, alpha=self.bif_alpha
            )
            batch_weights.append(w)

        weights = np.stack(batch_weights, axis=0)[:, np.newaxis]  # [B, 1, D, H, W]
        return torch.from_numpy(weights).to(tgt.device)

    def _weighted_dice_ce(self, pred, target, weight_map):
        """Bifurcation-weighted Dice + CE loss."""
        pred_soft = torch.softmax(pred, dim=1)

        if isinstance(target, list):
            tgt = target[0]
        else:
            tgt = target

        # One-hot encode
        if tgt.ndim == 4:
            tgt_long = tgt.unsqueeze(1).long()
        else:
            tgt_long = tgt.long()

        tgt_onehot = torch.zeros_like(pred_soft)
        tgt_onehot.scatter_(1, tgt_long, 1)

        # Weighted Dice (foreground channel)
        w = weight_map.expand_as(pred_soft)
        numerator = 2.0 * (w * pred_soft * tgt_onehot).sum(dim=(2, 3, 4))
        denominator = (w * (pred_soft ** 2 + tgt_onehot)).sum(dim=(2, 3, 4))
        dice_per_class = 1.0 - (numerator + 1e-5) / (denominator + 1e-5)
        weighted_dice = dice_per_class[:, 1].mean()

        # Weighted CE
        if tgt.ndim == 5:
            ce_target = tgt[:, 0].long()
        else:
            ce_target = tgt.long()
        ce_per_voxel = F.cross_entropy(pred, ce_target, reduction='none')
        weighted_ce = (weight_map[:, 0] * ce_per_voxel).mean()

        return 0.5 * weighted_dice + 0.5 * weighted_ce

    def _bif_focal_dice(self, pred, target, weight_map):
        """Dice loss computed ONLY within bifurcation-adjacent regions."""
        pred_soft = torch.softmax(pred, dim=1)

        if isinstance(target, list):
            tgt = target[0]
        else:
            tgt = target

        if tgt.ndim == 4:
            tgt_long = tgt.unsqueeze(1).long()
        else:
            tgt_long = tgt.long()

        tgt_onehot = torch.zeros_like(pred_soft)
        tgt_onehot.scatter_(1, tgt_long, 1)

        bif_mask = (weight_map > self.bif_weight_threshold).float()
        if bif_mask.sum() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        bif_mask_exp = bif_mask.expand_as(pred_soft)
        numerator = 2.0 * (bif_mask_exp * pred_soft * tgt_onehot).sum(dim=(2, 3, 4))
        denominator = (bif_mask_exp * (pred_soft ** 2 + tgt_onehot)).sum(dim=(2, 3, 4))
        dice_per_class = 1.0 - (numerator + 1e-5) / (denominator + 1e-5)

        return dice_per_class[:, 1].mean()

    def train_step(self, batch):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Determine if we're past warmup
        use_bcs = self.current_epoch >= self.warmup_epochs

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)

            if isinstance(output, dict):
                seg_output = output['seg']
                vessel_pred = output['vessel']
            else:
                seg_output = output
                vessel_pred = None

            # Get full-res prediction
            if isinstance(seg_output, (list, tuple)):
                pred = seg_output[0]
            else:
                pred = seg_output

            if use_bcs:
                # BCS-guided phase: weighted DiceCE + bifurcation focal
                with torch.no_grad():
                    bif_weights = self._compute_bif_weights(target)
                seg_loss = self._weighted_dice_ce(pred, target, bif_weights)
                bif_focal = self._bif_focal_dice(pred, target, bif_weights)
            else:
                # Warmup phase: standard nnU-Net DiceCE loss
                seg_loss = self.loss(seg_output, target)
                bif_focal = torch.tensor(0.0, device=self.device)

            # Vesselness auxiliary loss (always active)
            if vessel_pred is not None:
                tgt = target[0] if isinstance(target, list) else target
                with torch.no_grad():
                    vessel_target = self._compute_vesselness_target(data, tgt)
                # Resize vessel_pred to match target (nnU-Net patch may differ from decoder output)
                if vessel_pred.shape[2:] != vessel_target.shape[2:]:
                    vessel_pred = F.interpolate(
                        vessel_pred, size=vessel_target.shape[2:],
                        mode='trilinear', align_corners=False
                    )
                vessel_loss = F.mse_loss(vessel_pred, vessel_target)
            else:
                vessel_loss = torch.tensor(0.0, device=self.device)

            total_loss = (
                seg_loss +
                self.vesselness_weight * vessel_loss +
                self.bif_focal_weight * bif_focal
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    def validation_step(self, batch):
        """Override to handle dict output and compute validation metrics."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.network.eval()

        with torch.autocast(self.device.type, enabled=True):
            with torch.no_grad():
                output = self.network(data)

                if isinstance(output, dict):
                    seg_output = output['seg']
                else:
                    seg_output = output

                if isinstance(seg_output, list):
                    seg_output = seg_output[0]

                base_loss_fn = self.loss.loss if hasattr(self.loss, 'loss') else self.loss
                target_for_loss = target[0] if isinstance(target, list) else target
                l = base_loss_fn(seg_output, target_for_loss)

                if self.label_manager.has_regions:
                    predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
                else:
                    predicted_segmentation_onehot = seg_output.argmax(1)[:, None]
                    predicted_segmentation_onehot = torch.zeros(
                        (predicted_segmentation_onehot.shape[0], self.label_manager.num_segmentation_heads,
                         *predicted_segmentation_onehot.shape[2:]),
                        device=predicted_segmentation_onehot.device, dtype=torch.float32
                    ).scatter_(1, predicted_segmentation_onehot, 1)

                if self.label_manager.has_regions:
                    if isinstance(target, list):
                        target_onehot = target[0]
                    else:
                        target_onehot = target
                else:
                    if isinstance(target, list):
                        target_for_metrics = target[0][:, 0:1]
                    else:
                        target_for_metrics = target[:, 0:1]
                    target_onehot = torch.zeros_like(predicted_segmentation_onehot)
                    target_onehot.scatter_(1, target_for_metrics.long(), 1)

                tp = (predicted_segmentation_onehot * target_onehot).sum(dim=(2, 3, 4))
                fp = (predicted_segmentation_onehot * (1 - target_onehot)).sum(dim=(2, 3, 4))
                fn = ((1 - predicted_segmentation_onehot) * target_onehot).sum(dim=(2, 3, 4))

        self.network.train()

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }

    def predict_step(self, batch, return_probabilities=False):
        """Override to handle dict output during inference."""
        data = batch['data']
        data = data.to(self.device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True):
                output = self.network(data)

                if isinstance(output, dict):
                    seg_output = output['seg']
                else:
                    seg_output = output

                if isinstance(seg_output, list):
                    seg_output = seg_output[0]

        return seg_output

    def run_training(self):
        """Override training loop to add early stopping (matching L0 config)."""
        self.on_train_start()

        best_ema = -np.inf
        patience_counter = 0

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            # Early stopping check based on EMA foreground Dice
            current_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            if current_ema > best_ema:
                best_ema = current_ema
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.es_patience and self.current_epoch >= self.es_min_epochs:
                self.print_to_log_file(
                    f"Early stopping at epoch {self.current_epoch}: "
                    f"no improvement for {self.es_patience} epochs "
                    f"(best EMA Dice: {best_ema:.4f})"
                )
                break

        self.on_train_end()
'''


# =============================================================================
# Knowledge Distillation Trainer (CT-FM L1 teacher → nnU-Net student)
# =============================================================================

TRAINER_DISTILLATION = '''
"""
NN-UNet Trainer with Knowledge Distillation from CT-FM teacher.

Teacher predictions are stored as channel 1 in _seg.b2nd files
(quantized to int16 with ×10000 scale). Channel 0 remains the GT label.

Loss: DiceCE(pred, GT) + alpha * DistillLoss(softmax(pred), teacher_soft)

The distillation loss encourages the student (nnU-Net) to match the
teacher's (CT-FM) soft probability predictions, transferring boundary
precision while the standard DiceCE loss preserves GT-aligned training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDistillation(nnUNetTrainer):
    """
    NN-UNet trainer with knowledge distillation from CT-FM teacher.

    Expected seg format: (2, D, H, W)
        Channel 0: GT label (int, standard nnU-Net)
        Channel 1: Teacher probability × 10000 (int16, from CT-FM L1)

    Hyperparameters:
        distill_alpha: Weight for distillation loss (default 0.5)
        distill_temp: Temperature for soft targets (default 1.0, not used for prob matching)

    Loss = DiceCE(pred, GT) + alpha * SoftDiceLoss(softmax(pred), teacher)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.distill_alpha = 0.5
        self.print_to_log_file(
            f"Knowledge Distillation: alpha={self.distill_alpha}"
        )
        self.print_to_log_file("Teacher: CT-FM L1 (vesselness), stored in seg channel 1")

        # Early stopping
        self.es_patience = 25
        self.es_min_epochs = 50

    def _extract_teacher_and_gt(self, target):
        """
        Extract GT labels and teacher predictions from multi-channel target.

        Args:
            target: list of tensors (deep supervision) or single tensor
                    Each tensor has shape [B, 2, D, H, W] where:
                    - channel 0: GT label (integer)
                    - channel 1: teacher probability × 10000

        Returns:
            gt_target: same structure as target but with only GT channel
            teacher: teacher probability at full resolution [B, 1, D, H, W], float in [0, 1]
        """
        if isinstance(target, list):
            # Deep supervision: list of tensors at different resolutions
            # Extract teacher from full-resolution (first element)
            full_res = target[0]  # [B, 2, D, H, W]

            if full_res.shape[1] >= 2:
                teacher = full_res[:, 1:2].float() / 10000.0
                # Rebuild target list with only GT channel
                gt_target = [t[:, 0:1] for t in target]
            else:
                # No teacher channel (fallback for non-injected data)
                teacher = None
                gt_target = target
        else:
            if target.shape[1] >= 2:
                teacher = target[:, 1:2].float() / 10000.0
                gt_target = target[:, 0:1]
            else:
                teacher = None
                gt_target = target

        return gt_target, teacher

    def _soft_dice_distill_loss(self, pred_logits, teacher_prob):
        """
        Compute soft Dice distillation loss between student and teacher.

        Args:
            pred_logits: Student logits [B, C, D, H, W] (before softmax)
            teacher_prob: Teacher foreground probability [B, 1, D, H, W] in [0, 1]

        Returns:
            loss: 1 - SoftDice(student_fg_prob, teacher_prob)
        """
        pred_soft = torch.softmax(pred_logits, dim=1)
        pred_fg = pred_soft[:, 1:2]  # foreground probability

        # Resize if needed (deep supervision may cause size mismatch)
        if pred_fg.shape[2:] != teacher_prob.shape[2:]:
            teacher_prob = F.interpolate(
                teacher_prob, size=pred_fg.shape[2:],
                mode='trilinear', align_corners=False
            )

        # Soft Dice between student and teacher
        smooth = 1e-5
        intersection = (pred_fg * teacher_prob).sum(dim=(2, 3, 4))
        union = (pred_fg ** 2).sum(dim=(2, 3, 4)) + (teacher_prob ** 2).sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + smooth) / (union + smooth)

        return 1.0 - dice.mean()

    def train_step(self, batch):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Extract teacher predictions and clean GT target
        gt_target, teacher = self._extract_teacher_and_gt(target)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)

            # Standard nnU-Net DiceCE loss on GT
            base_loss = self.loss(output, gt_target)

            # Distillation loss (on full-resolution output only)
            if teacher is not None:
                # Get full-res prediction
                if isinstance(output, (list, tuple)):
                    pred_full = output[0]
                else:
                    pred_full = output

                distill_loss = self._soft_dice_distill_loss(pred_full, teacher)
                total_loss = base_loss + self.distill_alpha * distill_loss
            else:
                total_loss = base_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    def validation_step(self, batch):
        """Validation uses only GT (no distillation loss)."""
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Extract only GT for validation
        gt_target, _ = self._extract_teacher_and_gt(target)

        self.network.eval()

        with torch.autocast(self.device.type, enabled=True):
            with torch.no_grad():
                output = self.network(data)

                # Get full-res output
                if isinstance(output, (list, tuple)):
                    seg_output = output[0]
                else:
                    seg_output = output

                # Compute loss on GT only
                base_loss_fn = self.loss.loss if hasattr(self.loss, 'loss') else self.loss
                gt_for_loss = gt_target[0] if isinstance(gt_target, list) else gt_target
                l = base_loss_fn(seg_output, gt_for_loss)

                # Compute tp, fp, fn for Dice metric
                if self.label_manager.has_regions:
                    predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).long()
                else:
                    predicted_segmentation_onehot = seg_output.argmax(1)[:, None]
                    predicted_segmentation_onehot = torch.zeros(
                        (predicted_segmentation_onehot.shape[0], self.label_manager.num_segmentation_heads,
                         *predicted_segmentation_onehot.shape[2:]),
                        device=predicted_segmentation_onehot.device, dtype=torch.float32
                    ).scatter_(1, predicted_segmentation_onehot, 1)

                if self.label_manager.has_regions:
                    if isinstance(gt_target, list):
                        target_onehot = gt_target[0]
                    else:
                        target_onehot = gt_target
                else:
                    if isinstance(gt_target, list):
                        target_for_metrics = gt_target[0][:, 0:1]
                    else:
                        target_for_metrics = gt_target[:, 0:1]
                    target_onehot = torch.zeros_like(predicted_segmentation_onehot)
                    target_onehot.scatter_(1, target_for_metrics.long(), 1)

                tp = (predicted_segmentation_onehot * target_onehot).sum(dim=(2, 3, 4))
                fp = (predicted_segmentation_onehot * (1 - target_onehot)).sum(dim=(2, 3, 4))
                fn = ((1 - predicted_segmentation_onehot) * target_onehot).sum(dim=(2, 3, 4))

        self.network.train()

        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp.detach().cpu().numpy(),
            'fp_hard': fp.detach().cpu().numpy(),
            'fn_hard': fn.detach().cpu().numpy()
        }

    def run_training(self):
        """Training loop with early stopping."""
        self.on_train_start()

        best_ema = -np.inf
        patience_counter = 0

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            # Early stopping check based on EMA foreground Dice
            current_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            if current_ema > best_ema:
                best_ema = current_ema
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.es_patience and self.current_epoch >= self.es_min_epochs:
                self.print_to_log_file(
                    f"Early stopping at epoch {self.current_epoch}: "
                    f"no improvement for {self.es_patience} epochs "
                    f"(best EMA Dice: {best_ema:.4f})"
                )
                break

        self.on_train_end()
'''


# =============================================================================
# Online Knowledge Distillation (teacher model on GPU, no seg channel bloat)
# =============================================================================

TRAINER_DISTILLATION_ONLINE = '''
"""
NN-UNet Trainer with Online Knowledge Distillation from CT-FM teacher.

Instead of pre-computed teacher predictions stored in seg files (which doubles
data loading time), the teacher model runs on GPU alongside the student.

Teacher: CT-FM L1 (vesselness) — loaded from TEACHER_CHECKPOINT env var.
Loss: DiceCE(pred, GT) + alpha * KL_div(student_logits/T, teacher_logits/T) * T^2

Fixes from failed run (job 7274, alpha=0.5):
    1. alpha 0.5 → 0.1 (too aggressive caused train loss stuck at -0.22)
    2. 20-epoch warmup (pure DiceCE before adding distillation)
    3. KL divergence with temperature T=2 (instead of soft Dice)

Input normalization bridge:
    nnU-Net uses CTNormalization: (HU - mean) / std
    CT-FM uses ScaleIntensityRange: clip((HU + 200) / 800, 0, 1)
    Combined: ctfm_input = clip(x * (std/800) + (mean+200)/800, 0, 1)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDistillationOnline(nnUNetTrainer):
    """
    NN-UNet with online knowledge distillation from CT-FM teacher.

    The teacher model runs on GPU alongside the student. Seg files stay
    at normal 1-channel size, so data loading is at baseline speed.

    Environment variables:
        TEACHER_CHECKPOINT: Path to CT-FM L1 checkpoint (required)

    Hyperparameters:
        distill_alpha: Weight for distillation loss (default 0.1)
        distill_temp: Temperature for KL divergence (default 2.0)
        warmup_epochs: Pure DiceCE epochs before adding KD (default 20)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.distill_alpha = 0.1
        self.distill_temp = 2.0
        self.warmup_epochs = 20

        # Load teacher model
        self._load_teacher()

        self.print_to_log_file(f"Online Knowledge Distillation (v2):")
        self.print_to_log_file(f"  alpha={self.distill_alpha}, temp={self.distill_temp}, warmup={self.warmup_epochs}")
        self.print_to_log_file(f"  Teacher: CT-FM L1 (vesselness), running on GPU")
        self.print_to_log_file(f"  Loss: KL divergence (replaces soft Dice from v1)")

        # CTNormalization parameters for ImageCAS (Dataset105)
        # From foreground_intensity_properties: mean=148.17, std=179.68
        # CT-FM expects: clip((HU + 200) / 800, 0, 1)
        # Reverse nnU-Net norm then apply CT-FM:
        #   HU = x * std + mean
        #   ctfm = clip((HU + 200) / 800, 0, 1) = clip(x * std/800 + (mean+200)/800, 0, 1)
        self.norm_scale = 179.68 / 800.0   # 0.2246
        self.norm_offset = (148.17 + 200.0) / 800.0  # 0.4352

        # Early stopping
        self.es_patience = 25
        self.es_min_epochs = 50

    def _load_teacher(self):
        """Load CT-FM L1 teacher model from checkpoint."""
        checkpoint_path = os.environ.get('TEACHER_CHECKPOINT')
        if not checkpoint_path:
            raise RuntimeError(
                "TEACHER_CHECKPOINT environment variable not set. "
                "Set it to the path of the CT-FM L1 checkpoint."
            )

        self.print_to_log_file(f"Loading teacher from: {checkpoint_path}")

        from exp028_topology_comparison import CTFMWithAux

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']

        self.teacher = CTFMWithAux(use_vesselness=True, use_deep_sup=False)
        self.teacher.load_state_dict(state_dict)
        self.teacher.to(self.device)
        self.teacher.eval()

        # Freeze teacher — no gradients needed
        for param in self.teacher.parameters():
            param.requires_grad = False

        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        self.print_to_log_file(f"Teacher loaded: {teacher_params:,} parameters (frozen)")

    def _nnunet_to_ctfm_input(self, x):
        """Convert nnU-Net CTNormalization input to CT-FM [0, 1] range."""
        return torch.clamp(x * self.norm_scale + self.norm_offset, 0.0, 1.0)

    def _get_teacher_logits(self, data):
        """Run teacher on input data, return raw logits [B, 2, D, H, W]."""
        with torch.no_grad():
            teacher_input = self._nnunet_to_ctfm_input(data)
            teacher_out = self.teacher(teacher_input)
            teacher_logits = teacher_out['seg']   # [B, 2, D, H, W]
        return teacher_logits

    def _kl_distill_loss(self, student_logits, teacher_logits):
        """
        KL divergence distillation loss with temperature scaling.

        KL(teacher_soft || student_soft) * T^2
        The T^2 factor compensates for the gradient magnitude reduction
        from dividing logits by T.
        """
        T = self.distill_temp

        # Resize if needed (deep supervision may cause size mismatch)
        if student_logits.shape[2:] != teacher_logits.shape[2:]:
            teacher_logits = F.interpolate(
                teacher_logits, size=student_logits.shape[2:],
                mode='trilinear', align_corners=False
            )

        # Softened distributions
        student_log_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL divergence: sum over classes, mean over batch and spatial dims
        kl = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')

        return kl * (T * T)

    def train_step(self, batch):
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # Check if past warmup
        use_kd = self.current_epoch >= self.warmup_epochs

        # Teacher prediction (no grad, outside autocast for stable softmax)
        if use_kd:
            teacher_logits = self._get_teacher_logits(data)
        else:
            teacher_logits = None

        with torch.autocast(self.device.type, enabled=True):
            output = self.network(data)

            # Standard nnU-Net DiceCE loss on GT
            base_loss = self.loss(output, target)

            # Distillation loss on full-resolution output (after warmup)
            if use_kd and teacher_logits is not None:
                if isinstance(output, (list, tuple)):
                    pred_full = output[0]
                else:
                    pred_full = output

                distill_loss = self._kl_distill_loss(pred_full, teacher_logits)
                total_loss = base_loss + self.distill_alpha * distill_loss
            else:
                total_loss = base_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': total_loss.detach().cpu().numpy()}

    def run_training(self):
        """Training loop with early stopping."""
        self.on_train_start()

        best_ema = -np.inf
        patience_counter = 0

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

            # Early stopping
            current_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            if current_ema > best_ema:
                best_ema = current_ema
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.es_patience and self.current_epoch >= self.es_min_epochs:
                self.print_to_log_file(
                    f"Early stopping at epoch {self.current_epoch}: "
                    f"no improvement for {self.es_patience} epochs "
                    f"(best EMA Dice: {best_ema:.4f})"
                )
                break

        self.on_train_end()
'''


def install_trainers(nnunet_path=None):
    """
    Install custom trainers into nnUNet installation.

    Args:
        nnunet_path: Path to nnUNet installation. If None, auto-detect.
    """
    from pathlib import Path

    if nnunet_path is None:
        import nnunetv2
        nnunet_path = Path(nnunetv2.__file__).parent
    else:
        nnunet_path = Path(nnunet_path)

    trainer_dir = nnunet_path / "training" / "nnUNetTrainer"

    # Minimal EarlyStopping trainer - just inherits base trainer (sufficient for inference)
    TRAINER_EARLY_STOPPING = '''
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerEarlyStopping(nnUNetTrainer):
    """nnUNet trainer with early stopping. For inference, behaves identically to base trainer."""
    pass
'''

    trainers = {
        "nnUNetTrainerEarlyStopping.py": TRAINER_EARLY_STOPPING,
        "nnUNetTrainerVesselness.py": TRAINER_L1_VESSELNESS,
        "nnUNetTrainerClDice.py": TRAINER_L2_CLDICE,
        "nnUNetTrainerTopologyFull.py": TRAINER_L3_FULL,
        "nnUNetTrainerBCSGuided.py": TRAINER_L3_BCS_GUIDED,
        "nnUNetTrainerDistillation.py": TRAINER_DISTILLATION,
        "nnUNetTrainerDistillationOnline.py": TRAINER_DISTILLATION_ONLINE,
    }

    installed = []
    for filename, code in trainers.items():
        filepath = trainer_dir / filename
        with open(filepath, 'w') as f:
            f.write(code)
        installed.append(filepath)
        print(f"Installed: {filepath}")

    return installed


if __name__ == "__main__":
    from pathlib import Path

    print("Installing custom NN-UNet trainers for EXP-031...")
    print("These trainers use TRUE auxiliary decoder architecture (like CT-FM)")
    print()
    try:
        installed = install_trainers()
        print(f"\nSuccessfully installed {len(installed)} trainers:")
        for p in installed:
            print(f"  - {p.name}")
        print("\nTrainers available:")
        print("  L0: nnUNetTrainer (baseline)")
        print("  L1: nnUNetTrainerVesselness (+vesselness auxiliary decoder)")
        print("  L2: nnUNetTrainerClDice (+vesselness +clDice)")
        print("  L3: nnUNetTrainerTopologyFull (+vesselness +clDice +deep supervision)")
        print("  L3-BCS: nnUNetTrainerBCSGuided (+vesselness +BCS-guided bifurcation weighting)")
        print("  KD: nnUNetTrainerDistillation (CT-FM teacher → nnU-Net student, offline)")
        print("  KD-Online: nnUNetTrainerDistillationOnline (CT-FM teacher on GPU, fast)")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo install manually, run this script in an environment with nnUNet installed.")

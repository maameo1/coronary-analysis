#!/usr/bin/env python3
"""
================================================================================
CT-FM L4: SOFT BCS LOSS — CONNECTIVITY-AWARE TRAINING
================================================================================

Unlike bifocal loss (L3), which upweights voxels near bifurcations (voxel-wise),
the Soft BCS Loss directly optimizes for branch connectivity:

  For each GT bifurcation:
    1. Extract 3 branch stubs (precomputed as stub_label_map)
    2. Sample soft prediction probability along each stub
    3. Compute soft minimum over branch-wise mean probabilities
    4. Loss = 1 - mean(bif_scores) across all bifurcations in batch

The soft minimum is key: if ANY branch is missing from the prediction,
the entire bifurcation scores poorly. This is connectivity-aware, not
voxel-wise. Gradients flow primarily to the weakest branch, pushing the
model to preserve all 3 branches at every junction.

Author: Anonymous
Date: February 2026
Experiment: EXP-040 CT-FM L4 Soft BCS
================================================================================
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import pad_list_data_collate

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd,
    SpatialPadd, CenterSpatialCropd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, EnsureTyped, Activations, AsDiscrete,
    MapTransform
)

from monai.data import Dataset, DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter
from imagecas_data_loader import ImageCASDataLoader
from multitask_ctfm import MultiTaskCTFM, AddVesselnessGroundTruthd


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    exp_name = "ctfm_l4_soft_bcs"
    exp_id = "EXP-040"
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    data_root = Path("/path/to/project/data/ImageCAS")
    split_file = Path("/path/to/project/data/ImageCAS/split_1000.csv")
    stub_label_dir = Path("/path/to/project/data/ImageCAS/stub_labels")
    output_root = None

    # Resume settings
    resume = False
    resume_checkpoint = None
    start_epoch = 0

    n_train_cases = 638
    test_fold = 1
    val_fraction = 0.15

    voxel_spacing = (1.0, 1.0, 1.0)
    patch_size = (96, 96, 96)
    hu_min = -200
    hu_max = 600

    # Frangi vesselness params (same as L1)
    frangi_sigmas = range(1, 4)
    frangi_alpha = 0.5
    frangi_beta = 0.5
    frangi_gamma = 15

    # Model
    model_name = "project-lighter/ct_fm_segresnet"
    init_filters = 32
    vessel_decoder_filters = 16

    # Training
    batch_size = 1
    num_workers = 4
    learning_rate = 1e-4
    weight_decay = 1e-5
    max_epochs = 100
    val_interval = 1

    # Loss weights
    seg_loss_weight = 1.0        # Standard DiceCE
    vessel_loss_weight = 0.5     # Vesselness MSE (same as L1)
    soft_bcs_weight = 0.3        # Soft BCS loss weight
    dice_weight = 0.5
    ce_weight = 0.5

    # Warmup: train standard DiceCE first, then ramp in soft BCS
    warmup_epochs = 10

    # Bifurcation-biased sampling: probability that a patch is centered
    # on a bifurcation stub voxel instead of a random vessel voxel.
    # Ensures ~50% of patches contain complete bifurcations for BCS signal.
    bif_crop_prob = 0.5

    # Scheduling
    early_stopping_patience = 30   # More patience — connectivity loss may need time
    lr_scheduler_patience = 10
    lr_scheduler_factor = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True
    random_seed = 2025

    def __init__(self, resume_dir=None):
        if resume_dir:
            self.output_root = Path(resume_dir)
            self.resume = True
        else:
            self.output_root = Path(f"/path/to/project/experiments/{self.exp_id}_{self.exp_name}_{self.date}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "checkpoints").mkdir(exist_ok=True)
        (self.output_root / "logs").mkdir(exist_ok=True)
        if not self.resume:
            self.save_config()

    def save_config(self):
        config_dict = {}
        for k in dir(type(self)):
            if k.startswith('_') or callable(getattr(type(self), k)):
                continue
            v = getattr(self, k)
            config_dict[k] = str(v) if isinstance(v, (Path, torch.device)) else v
        with open(self.output_root / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)


# ============================================================================
# SOFT BCS LOSS — THE KEY INNOVATION
# ============================================================================

class SoftBCSLoss(nn.Module):
    """
    Differentiable approximation of the Bifurcation Connectivity Score.

    For each GT bifurcation in the patch:
      1. Decode stub voxel locations from the stub_label_map
      2. Sample soft prediction probability at each stub's voxels
      3. Compute mean probability per stub
      4. Take min across stubs — weakest branch determines score
      5. Loss = 1 - mean(min_scores)

    The min() operation is what makes this connectivity-aware:
    - If all 3 branches are predicted: min ≈ high → loss low
    - If 1 branch is missing: min ≈ 0 → loss high
    - Gradient flows to the weakest branch → model learns to fill gaps

    This is fundamentally different from bifocal loss, which just
    weights voxels near bifurcations more heavily (still voxel-wise).
    """

    def __init__(self, min_stubs=2, min_voxels_per_stub=3, softmin_temp=None):
        """
        Args:
            min_stubs: Minimum number of stubs required for a valid bifurcation.
            min_voxels_per_stub: Minimum voxels a stub must have in the patch
                to be counted. Prevents noisy gradients from stubs that are
                barely inside the patch boundary (e.g., 1-2 voxels).
            softmin_temp: If set, use softmin with this temperature instead of
                         hard min. Hard min has sparse gradients (only flows to
                         the argmin). Softmin smooths gradients across all stubs.
                         Recommended: 0.1-0.5 for smooth gradients.
        """
        super().__init__()
        self.min_stubs = min_stubs
        self.min_voxels_per_stub = min_voxels_per_stub
        self.softmin_temp = softmin_temp

    def forward(self, pred, stub_label_map):
        """
        Args:
            pred: [B, 2, D, H, W] logits (pre-softmax)
            stub_label_map: [B, 1, D, H, W] integer labels
                Encoding: value = bif_id * 4 + stub_idx (1-3), 0 = background

        Returns:
            Scalar loss (1 - mean bifurcation connectivity score)
        """
        # Softmax to get foreground probability
        pred_soft = F.softmax(pred, dim=1)
        fg_prob = pred_soft[:, 1]  # [B, D, H, W] — foreground channel

        batch_size = pred.shape[0]
        total_loss = torch.tensor(0.0, device=pred.device)
        n_bifs = 0

        for b in range(batch_size):
            # Round before long cast: MONAI loads NIfTI as float32, and
            # nearest-interp should preserve exact integers, but rounding
            # guards against any float drift (e.g., 4.9999998 → 4)
            stub_map = stub_label_map[b, 0].round().long()  # [D, H, W]
            fg = fg_prob[b]  # [D, H, W]

            # Find all unique values > 0 (encoded stubs)
            unique_vals = torch.unique(stub_map)
            unique_vals = unique_vals[unique_vals > 0]

            if len(unique_vals) == 0:
                continue

            # Decode bifurcation IDs from encoded values.
            # Integer division recovers bif_id since stub_idx ∈ {1,2,3}
            # e.g., value=9 → bif_id=2, value=11 → bif_id=2
            bif_ids = torch.unique(unique_vals // 4)

            for bif_id in bif_ids:
                bif_id_int = bif_id.item()
                if bif_id_int == 0:
                    continue

                stub_probs = []
                for stub_idx in range(1, 4):  # stubs 1, 2, 3
                    val = bif_id_int * 4 + stub_idx
                    mask = (stub_map == val)
                    n_voxels = mask.sum().item()
                    if n_voxels >= self.min_voxels_per_stub:
                        # Average foreground probability along this stub
                        avg_prob = fg[mask].mean()
                        stub_probs.append(avg_prob)

                if len(stub_probs) < self.min_stubs:
                    continue

                # Stack stub probabilities
                stacked = torch.stack(stub_probs)

                if self.softmin_temp is not None:
                    # Softmin: smooth approximation, gradients flow to all stubs
                    # softmin(x) = sum(x_i * softmax(-x_i / T))
                    weights = F.softmax(-stacked / self.softmin_temp, dim=0)
                    bif_score = (stacked * weights).sum()
                else:
                    # Hard min: gradient only flows to weakest branch
                    bif_score = stacked.min()

                total_loss = total_loss + (1.0 - bif_score)
                n_bifs += 1

        if n_bifs == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return total_loss / n_bifs


# ============================================================================
# TRANSFORMS
# ============================================================================

class AddStubLabeld(MapTransform):
    """Resolve stub label map path for each case."""

    def __init__(self, keys, stub_label_dir):
        super().__init__(keys)
        self.stub_label_dir = Path(stub_label_dir)

    def __call__(self, data):
        d = dict(data)
        label_path = Path(d.get("label", ""))
        case_name = label_path.stem.replace('.label', '')
        stub_path = self.stub_label_dir / f"{case_name}.stub_labels.nii.gz"
        if stub_path.exists():
            d["stub_labels"] = str(stub_path)
        return d


class DefaultStubLabeld(MapTransform):
    """Create zero stub label map if precomputed stubs are missing."""

    def __init__(self, keys, ref_key="label"):
        super().__init__(keys)
        self.ref_key = ref_key

    def __call__(self, data):
        d = dict(data)
        if d.get("stub_labels") is None:
            d["stub_labels"] = torch.zeros_like(d[self.ref_key], dtype=torch.float32)
        elif not isinstance(d["stub_labels"], torch.Tensor):
            d["stub_labels"] = torch.as_tensor(d["stub_labels"], dtype=torch.float32)
        return d


class BifurcationBiasedCropd(MapTransform):
    """Patch cropping biased toward bifurcation regions.

    Replaces RandCropByPosNegLabeld for L4 training. For each patch:
    - With prob bif_prob: center on a stub_labels > 0 voxel (bifurcation)
    - Otherwise: standard pos/neg sampling from vessel label

    This ensures ~50% of patches contain complete bifurcations,
    increasing the effective supervision for Soft BCS loss.
    Without this, many patches contain no bifurcations and the
    BCS loss fires on <20% of batches — too sparse to matter.
    """

    def __init__(self, keys, label_key="label", stub_key="stub_labels",
                 spatial_size=(96, 96, 96), num_samples=4, bif_prob=0.5,
                 pos=1, neg=1, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.label_key = label_key
        self.stub_key = stub_key
        self.spatial_size = list(spatial_size)
        self.num_samples = num_samples
        self.bif_prob = bif_prob
        self.pos_ratio = pos / (pos + neg)

    def _select_center(self, label_np, stub_np, spatial_shape):
        """Select a crop center, biased toward bifurcations."""
        # Try bifurcation-biased sampling
        if stub_np is not None and np.random.random() < self.bif_prob:
            bif_coords = np.argwhere(stub_np > 0)
            if len(bif_coords) > 0:
                return bif_coords[np.random.randint(len(bif_coords))]

        # Standard pos/neg sampling
        if np.random.random() < self.pos_ratio:
            fg_coords = np.argwhere(label_np > 0)
            if len(fg_coords) > 0:
                return fg_coords[np.random.randint(len(fg_coords))]

        # Negative (background) sample
        bg_coords = np.argwhere(label_np == 0)
        if len(bg_coords) > 0:
            return bg_coords[np.random.randint(len(bg_coords))]

        return np.array([s // 2 for s in spatial_shape])

    def _compute_slices(self, center, spatial_shape):
        """Compute crop slices, clamped to volume bounds."""
        slices = []
        for dim in range(3):
            half = self.spatial_size[dim] // 2
            start = max(0, int(center[dim]) - half)
            end = start + self.spatial_size[dim]
            if end > spatial_shape[dim]:
                end = spatial_shape[dim]
                start = max(0, end - self.spatial_size[dim])
            slices.append(slice(start, end))
        return slices

    def __call__(self, data):
        d = dict(data)

        # Extract label as numpy for index computation
        label = d[self.label_key]
        if isinstance(label, torch.Tensor):
            label_np = label[0].cpu().numpy()
        else:
            label_np = np.asarray(label[0])
        spatial_shape = label_np.shape

        # Extract stub data if available
        stub_np = None
        if self.stub_key in d and d[self.stub_key] is not None:
            stub = d[self.stub_key]
            if isinstance(stub, torch.Tensor):
                stub_np = stub[0].cpu().numpy()
            elif isinstance(stub, np.ndarray):
                stub_np = stub[0] if stub.ndim == 4 else stub

        results = []
        for _ in range(self.num_samples):
            center = self._select_center(label_np, stub_np, spatial_shape)
            crop_slices = self._compute_slices(center, spatial_shape)

            sample = {}
            # Copy non-transform keys as-is
            for key in d:
                if key not in self.keys:
                    sample[key] = d[key]
            # Crop the transform keys
            for key in self.key_iterator(d):
                val = d[key]
                if isinstance(val, (torch.Tensor, np.ndarray)) and val.ndim >= 3:
                    if val.ndim >= 4:  # [C, D, H, W]
                        sample[key] = val[:, crop_slices[0], crop_slices[1], crop_slices[2]]
                    else:  # [D, H, W]
                        sample[key] = val[crop_slices[0], crop_slices[1], crop_slices[2]]
                else:
                    sample[key] = val
            results.append(sample)

        return results


def get_transforms(config, mode='train'):
    all_spatial = ["image", "label", "vesselness", "stub_labels"]

    common_transforms = [
        AddStubLabeld(keys=["label"], stub_label_dir=config.stub_label_dir),
        LoadImaged(keys=["image", "label", "stub_labels"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label", "stub_labels"], allow_missing_keys=True),
        # stub_labels uses nearest interpolation to preserve integer encoding
        Spacingd(keys=["image", "label", "stub_labels"], pixdim=config.voxel_spacing,
                mode=("bilinear", "nearest", "nearest"), allow_missing_keys=True),
        ScaleIntensityRanged(keys=["image"],
                            a_min=config.hu_min, a_max=config.hu_max,
                            b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label", "stub_labels"], source_key="label",
                       allow_missing_keys=True),
        AddVesselnessGroundTruthd(
            keys=["image"],
            sigmas=config.frangi_sigmas,
            alpha=config.frangi_alpha,
            beta=config.frangi_beta,
            gamma=config.frangi_gamma
        ),
    ]

    if mode == 'train':
        return Compose(common_transforms + [
            SpatialPadd(keys=all_spatial, spatial_size=config.patch_size, allow_missing_keys=True),
            BifurcationBiasedCropd(
                keys=all_spatial, label_key="label", stub_key="stub_labels",
                spatial_size=config.patch_size, num_samples=4,
                bif_prob=config.bif_crop_prob, pos=1, neg=1,
                allow_missing_keys=True
            ),
            RandFlipd(keys=all_spatial, prob=0.2, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=all_spatial, prob=0.2, spatial_axis=1, allow_missing_keys=True),
            RandFlipd(keys=all_spatial, prob=0.2, spatial_axis=2, allow_missing_keys=True),
            RandRotate90d(keys=all_spatial, prob=0.2, max_k=3, allow_missing_keys=True),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
            DefaultStubLabeld(keys=["stub_labels"], ref_key="label"),
            EnsureTyped(keys=all_spatial, allow_missing_keys=True),
        ])
    else:
        # Use RandCropByPosNegLabeld (same as baseline) for fair val Dice comparison
        # Previously used CenterSpatialCropd which inflated val Dice by ~0.07
        return Compose(common_transforms + [
            SpatialPadd(keys=all_spatial, spatial_size=config.patch_size, allow_missing_keys=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.patch_size,
                pos=1, neg=1, num_samples=1,
                image_key="image",
                image_threshold=0,
                allow_smaller=False,
            ),
            DefaultStubLabeld(keys=["stub_labels"], ref_key="label"),
            EnsureTyped(keys=all_spatial, allow_missing_keys=True),
        ])


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, standard_loss_fn, vessel_loss_fn,
                soft_bcs_loss_fn, config, epoch, scaler=None):
    model.train()
    epoch_total, epoch_seg, epoch_vessel, epoch_bcs = 0.0, 0.0, 0.0, 0.0
    n_batches = 0
    n_bcs_active = 0  # Track how many batches had bifurcations
    n_bifs_epoch = 0   # Total bifurcations seen this epoch

    # Ramp soft BCS weight over 10 epochs instead of hard switch
    if epoch < config.warmup_epochs:
        bcs_weight_eff = 0.0
    else:
        ramp = min(1.0, (epoch - config.warmup_epochs) / 10.0)
        bcs_weight_eff = ramp * config.soft_bcs_weight

    for batch_data in train_loader:
        inputs = batch_data["image"].to(config.device)
        seg_labels = batch_data["label"].to(config.device)
        vessel_labels = batch_data["vesselness"].to(config.device)

        stub_labels = batch_data.get("stub_labels")
        if stub_labels is not None:
            stub_labels = stub_labels.to(config.device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                seg_pred, vessel_pred = model(inputs)

                seg_loss = standard_loss_fn(seg_pred, seg_labels)
                vessel_loss = vessel_loss_fn(vessel_pred, vessel_labels)

                if bcs_weight_eff > 0 and stub_labels is not None:
                    bcs_loss = soft_bcs_loss_fn(seg_pred, stub_labels)
                else:
                    bcs_loss = torch.tensor(0.0, device=config.device)

                total_loss = (config.seg_loss_weight * seg_loss +
                             config.vessel_loss_weight * vessel_loss +
                             bcs_weight_eff * bcs_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_pred, vessel_pred = model(inputs)
            seg_loss = standard_loss_fn(seg_pred, seg_labels)
            vessel_loss = vessel_loss_fn(vessel_pred, vessel_labels)

            if bcs_weight_eff > 0 and stub_labels is not None:
                bcs_loss = soft_bcs_loss_fn(seg_pred, stub_labels)
            else:
                bcs_loss = torch.tensor(0.0, device=config.device)

            total_loss = (config.seg_loss_weight * seg_loss +
                         config.vessel_loss_weight * vessel_loss +
                         bcs_weight_eff * bcs_loss)
            total_loss.backward()
            optimizer.step()

        epoch_total += total_loss.item()
        epoch_seg += seg_loss.item()
        epoch_vessel += vessel_loss.item()
        bcs_val = bcs_loss.item() if isinstance(bcs_loss, torch.Tensor) else bcs_loss
        epoch_bcs += bcs_val
        if bcs_val > 0:
            n_bcs_active += 1
        n_batches += 1

    n = max(n_batches, 1)
    return epoch_total / n, epoch_seg / n, epoch_vessel / n, epoch_bcs / n, n_bcs_active, bcs_weight_eff


def validate(model, val_loader, metric, config, post_pred, post_label):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data["image"].to(config.device)
            labels = batch_data["label"].to(config.device)
            seg_pred, _ = model(inputs)
            outputs = [post_pred(i) for i in decollate_batch(seg_pred)]
            labels = [post_label(i) for i in decollate_batch(labels)]
            metric(y_pred=outputs, y=labels)
    return metric.aggregate().item()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to experiment directory to resume from")
    args = parser.parse_args()

    config = Config(resume_dir=args.resume)
    print(f"{'='*80}")
    print(f"CT-FM L4: Soft BCS Loss — Connectivity-Aware Training")
    print(f"{'='*80}")
    print(f"Experiment: {config.exp_id} - {config.exp_name}")
    print(f"Output: {config.output_root}")
    if config.resume:
        print(f"*** RESUMING from checkpoint ***")
    print(f"Loss weights: seg={config.seg_loss_weight}, vessel={config.vessel_loss_weight}, "
          f"soft_bcs={config.soft_bcs_weight}")
    print(f"Warmup: {config.warmup_epochs} epochs standard, then + Soft BCS")
    print(f"Bifurcation-biased sampling: {config.bif_crop_prob:.0%} of patches centered on bifurcations")
    print(f"Patience: {config.early_stopping_patience} epochs")

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Data
    print(f"\n{'='*80}\nDATA PREPARATION\n{'='*80}")
    data_loader = ImageCASDataLoader(
        data_root=config.data_root, split_file=config.split_file,
        test_fold=config.test_fold, val_fraction=config.val_fraction, random_seed=42
    )
    splits = data_loader.get_splits(n_train_cases=config.n_train_cases, regime_seed=2025)
    train_data, val_data = splits['train'], splits['val']
    print(f"Training: {len(train_data)}, Validation: {len(val_data)}")

    n_with_stubs = sum(
        1 for d in train_data
        if (config.stub_label_dir / f"{Path(d['label']).stem.replace('.label', '')}.stub_labels.nii.gz").exists()
    )
    print(f"Stub label maps: {n_with_stubs}/{len(train_data)}")
    if n_with_stubs == 0:
        print("WARNING: No stub label maps found! Run precompute_stub_labels.py first.")
        print("Training will proceed but Soft BCS loss will be 0.")

    with open(config.output_root / "data_splits.json", 'w') as f:
        json.dump({k: [d['case_id'] for d in v] for k, v in
                   [('train', train_data), ('val', val_data), ('test', splits['test'])]}, f, indent=2)

    train_ds = Dataset(data=train_data, transform=get_transforms(config, 'train'))
    val_ds = Dataset(data=val_data, transform=get_transforms(config, 'val'))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True,
                              collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True,
                            collate_fn=pad_list_data_collate)

    # Model
    print(f"\n{'='*80}\nMODEL: MultiTaskCTFM (L4 — Soft BCS)\n{'='*80}")
    model = MultiTaskCTFM(
        pretrained_model_name=config.model_name,
        init_filters=config.init_filters,
        vessel_decoder_filters=config.vessel_decoder_filters
    ).to(config.device)

    # Losses
    standard_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True,
                                   lambda_dice=config.dice_weight, lambda_ce=config.ce_weight)
    vessel_loss_fn = nn.MSELoss()
    # min_stubs=2: lenient to keep bifurcations near patch boundaries usable
    # min_voxels_per_stub=1: stubs shrink from ~6 voxels to ~2 after
    # resampling from 0.32mm to 1mm spacing. Original threshold of 3
    # filtered out ALL stubs, causing BCS loss to never fire.
    soft_bcs_loss_fn = SoftBCSLoss(min_stubs=2, min_voxels_per_stub=1, softmin_temp=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    writer = SummaryWriter(log_dir=config.output_root / "logs")

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0.0
    best_epoch = 0
    if config.resume:
        ckpt_path = config.output_root / "checkpoints" / "best_model.pth"
        if ckpt_path.exists():
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_dice = checkpoint.get('dice', 0.0)
            best_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}, best_dice={best_dice:.4f}")
        else:
            print(f"WARNING: No checkpoint found at {ckpt_path}, starting fresh")

    # Train
    print(f"\n{'='*80}\nTRAINING\n{'='*80}")
    patience_counter = 0

    for epoch in range(start_epoch, config.max_epochs):
        total_loss, seg_loss, vessel_loss, bcs_loss, n_bcs_active, bcs_wt = train_epoch(
            model, train_loader, optimizer,
            standard_loss_fn, vessel_loss_fn, soft_bcs_loss_fn,
            config, epoch, scaler
        )
        phase = f"Soft-BCS(w={bcs_wt:.2f})" if epoch >= config.warmup_epochs else "warmup"
        print(f"Epoch {epoch+1} [{phase}]: total={total_loss:.4f} seg={seg_loss:.4f} "
              f"vessel={vessel_loss:.4f} bcs={bcs_loss:.4f} (active in {n_bcs_active} batches)")

        writer.add_scalar("Loss/total", total_loss, epoch)
        writer.add_scalar("Loss/seg", seg_loss, epoch)
        writer.add_scalar("Loss/vessel", vessel_loss, epoch)
        writer.add_scalar("Loss/soft_bcs", bcs_loss, epoch)
        writer.add_scalar("BCS/n_batches_active", n_bcs_active, epoch)
        writer.add_scalar("BCS/weight_effective", bcs_wt, epoch)

        if (epoch + 1) % config.val_interval == 0:
            val_dice = validate(model, val_loader, dice_metric, config,
                               post_pred, post_label)
            print(f"  Val Dice: {val_dice:.4f}")
            writer.add_scalar("Dice/val", val_dice, epoch)
            scheduler.step(val_dice)

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice': val_dice,
                }, config.output_root / "checkpoints" / "best_model.pth")
                print(f"  *** New best: {best_dice:.4f} ***")
            else:
                patience_counter += 1

            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
               config.output_root / "checkpoints" / "final_model.pth")
    writer.close()

    print(f"\nCOMPLETE: Best Dice = {best_dice:.4f} (epoch {best_epoch})")
    with open(config.output_root / "results.json", 'w') as f:
        json.dump({'best_dice': best_dice, 'best_epoch': best_epoch,
                   'warmup_epochs': config.warmup_epochs,
                   'soft_bcs_weight': config.soft_bcs_weight}, f)


if __name__ == "__main__":
    main()

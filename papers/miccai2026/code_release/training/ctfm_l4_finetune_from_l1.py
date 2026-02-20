#!/usr/bin/env python3
"""
================================================================================
CT-FM L4 OPTION 1: FINE-TUNE L1 WITH SOFT BCS (w=0.03)
================================================================================

Strategy: Take the trained L1 checkpoint (Dice=0.790, BCS=0.690), freeze the
encoder, and fine-tune only the decoder layers with a very low soft BCS weight
for ~20 epochs. This should nudge topology (BCS) while preserving L1's Dice.

Key differences from EXP-043:
  - Starts from L1 checkpoint (not random init)
  - Encoder frozen (only decoder + vessel decoder train)
  - Lower soft BCS weight: 0.03 (vs 0.10)
  - Lower LR: 1e-5 (vs 1e-4) — fine-tuning regime
  - Short run: 30 epochs max (vs 100)
  - Standard RandCropByPosNeg (no BifurcationBiasedCropd)

Author: Anonymous
Date: February 2026
Experiment: EXP-044 CT-FM L4 Fine-tune from L1
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
    SpatialPadd, RandCropByPosNegLabeld,
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
    exp_name = "ctfm_l4_finetune_from_l1"
    exp_id = "EXP-044"
    date = datetime.now().strftime('%Y%m%d_%H%M%S')

    data_root = Path("/path/to/project/data/ImageCAS")
    split_file = Path("/path/to/project/data/ImageCAS/split_1000.csv")
    stub_label_dir = Path("/path/to/project/data/ImageCAS/stub_labels")
    output_root = None

    # L1 checkpoint to fine-tune from
    l1_checkpoint = Path("/path/to/project/experiments/EXP-028_L1_vesselness_ctfm_638cases_20260109_043910/checkpoints/best_model.pth")

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

    # Fine-tuning regime
    batch_size = 1
    num_workers = 4
    learning_rate = 1e-5       # 10x lower than full training
    weight_decay = 1e-5
    max_epochs = 30            # Short run
    val_interval = 1

    # Loss weights — conservative soft BCS
    seg_loss_weight = 1.0
    vessel_loss_weight = 0.5
    soft_bcs_weight = 0.03     # Very conservative
    dice_weight = 0.5
    ce_weight = 0.5

    # No warmup needed — model already converged, just nudging
    warmup_epochs = 0

    # Freeze encoder — only decoder trains
    freeze_encoder = True

    early_stopping_patience = 15
    lr_scheduler_patience = 5
    lr_scheduler_factor = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True
    random_seed = 2025

    def __init__(self, resume_dir=None):
        if resume_dir:
            self.output_root = Path(resume_dir)
        else:
            self.output_root = Path(f"/path/to/project/experiments/{self.exp_id}_{self.exp_name}_{self.date}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "checkpoints").mkdir(exist_ok=True)
        (self.output_root / "logs").mkdir(exist_ok=True)
        if not resume_dir:
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
# SOFT BCS LOSS (same as EXP-043)
# ============================================================================

class SoftBCSLoss(nn.Module):
    def __init__(self, min_stubs=2, min_voxels_per_stub=1, softmin_temp=0.2):
        super().__init__()
        self.min_stubs = min_stubs
        self.min_voxels_per_stub = min_voxels_per_stub
        self.softmin_temp = softmin_temp

    def forward(self, pred, stub_label_map):
        pred_soft = F.softmax(pred, dim=1)
        fg_prob = pred_soft[:, 1]

        batch_size = pred.shape[0]
        total_loss = torch.tensor(0.0, device=pred.device)
        n_bifs = 0

        for b in range(batch_size):
            stub_map = stub_label_map[b, 0].round().long()
            fg = fg_prob[b]

            unique_vals = torch.unique(stub_map)
            unique_vals = unique_vals[unique_vals > 0]

            if len(unique_vals) == 0:
                continue

            bif_ids = torch.unique(unique_vals // 4)

            for bif_id in bif_ids:
                bif_id_int = bif_id.item()
                if bif_id_int == 0:
                    continue

                stub_probs = []
                for stub_idx in range(1, 4):
                    val = bif_id_int * 4 + stub_idx
                    mask = (stub_map == val)
                    n_voxels = mask.sum().item()
                    if n_voxels >= self.min_voxels_per_stub:
                        avg_prob = fg[mask].mean()
                        stub_probs.append(avg_prob)

                if len(stub_probs) < self.min_stubs:
                    continue

                stacked = torch.stack(stub_probs)
                if self.softmin_temp is not None:
                    weights = F.softmax(-stacked / self.softmin_temp, dim=0)
                    bif_score = (stacked * weights).sum()
                else:
                    bif_score = stacked.min()

                total_loss = total_loss + (1.0 - bif_score)
                n_bifs += 1

        if n_bifs == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return total_loss / n_bifs


# ============================================================================
# TRANSFORMS — Standard cropping (no BifurcationBiasedCropd)
# ============================================================================

class AddStubLabeld(MapTransform):
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


def get_transforms(config, mode='train'):
    all_spatial = ["image", "label", "vesselness", "stub_labels"]

    common_transforms = [
        AddStubLabeld(keys=["label"], stub_label_dir=config.stub_label_dir),
        LoadImaged(keys=["image", "label", "stub_labels"], allow_missing_keys=True),
        EnsureChannelFirstd(keys=["image", "label", "stub_labels"], allow_missing_keys=True),
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
            # Standard cropping — no bifurcation bias
            RandCropByPosNegLabeld(
                keys=["image", "label", "vesselness", "stub_labels"],
                label_key="label",
                spatial_size=config.patch_size,
                pos=1, neg=1, num_samples=4,
                image_key="image",
                image_threshold=0,
                allow_smaller=False,
                allow_missing_keys=True,
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
    n_bcs_active = 0

    bcs_weight_eff = config.soft_bcs_weight  # No warmup — start with BCS immediately

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

                if stub_labels is not None:
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

            if stub_labels is not None:
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
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = Config(resume_dir=args.resume)
    print(f"{'='*80}")
    print(f"CT-FM L4 Option 1: Fine-tune L1 with Soft BCS (w={config.soft_bcs_weight})")
    print(f"{'='*80}")
    print(f"Experiment: {config.exp_id} - {config.exp_name}")
    print(f"Output: {config.output_root}")
    print(f"L1 checkpoint: {config.l1_checkpoint}")
    print(f"Freeze encoder: {config.freeze_encoder}")
    print(f"LR: {config.learning_rate}, max epochs: {config.max_epochs}")
    print(f"Loss weights: seg={config.seg_loss_weight}, vessel={config.vessel_loss_weight}, "
          f"soft_bcs={config.soft_bcs_weight}")

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Data
    data_loader = ImageCASDataLoader(
        data_root=config.data_root, split_file=config.split_file,
        test_fold=config.test_fold, val_fraction=config.val_fraction, random_seed=42
    )
    splits = data_loader.get_splits(n_train_cases=config.n_train_cases, regime_seed=2025)
    train_data, val_data = splits['train'], splits['val']
    print(f"Training: {len(train_data)}, Validation: {len(val_data)}")

    train_ds = Dataset(data=train_data, transform=get_transforms(config, 'train'))
    val_ds = Dataset(data=val_data, transform=get_transforms(config, 'val'))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, pin_memory=True,
                              collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=config.num_workers, pin_memory=True,
                            collate_fn=pad_list_data_collate)

    # Model — load from L1 checkpoint
    print(f"\nLoading L1 checkpoint: {config.l1_checkpoint}")
    model = MultiTaskCTFM(
        pretrained_model_name=config.model_name,
        init_filters=config.init_filters,
        vessel_decoder_filters=config.vessel_decoder_filters
    ).to(config.device)

    checkpoint = torch.load(config.l1_checkpoint, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded L1 weights (epoch {checkpoint.get('epoch', '?')}, dice={checkpoint.get('dice', '?')})")

    # Freeze encoder
    if config.freeze_encoder:
        frozen_params = 0
        for name, param in model.named_parameters():
            if 'encoder' in name or 'backbone' in name:
                param.requires_grad = False
                frozen_params += param.numel()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Frozen: {frozen_params:,} params, Trainable: {trainable:,} / {total:,}")

    # Validate L1 baseline before fine-tuning
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    print(f"\nValidating L1 baseline before fine-tuning...")
    baseline_dice = validate(model, val_loader, dice_metric, config, post_pred, post_label)
    print(f"  L1 baseline val Dice: {baseline_dice:.4f}")

    # Losses
    standard_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True,
                                   lambda_dice=config.dice_weight, lambda_ce=config.ce_weight)
    vessel_loss_fn = nn.MSELoss()
    soft_bcs_loss_fn = SoftBCSLoss(min_stubs=2, min_voxels_per_stub=1, softmin_temp=0.2)

    # Only optimize trainable params
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience)

    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    writer = SummaryWriter(log_dir=config.output_root / "logs")

    # Train
    print(f"\n{'='*80}\nFINE-TUNING (decoder only, soft BCS w={config.soft_bcs_weight})\n{'='*80}")
    best_dice = baseline_dice
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.max_epochs):
        total_loss, seg_loss, vessel_loss, bcs_loss, n_bcs_active, bcs_wt = train_epoch(
            model, train_loader, optimizer,
            standard_loss_fn, vessel_loss_fn, soft_bcs_loss_fn,
            config, epoch, scaler
        )
        print(f"Epoch {epoch+1}: total={total_loss:.4f} seg={seg_loss:.4f} "
              f"vessel={vessel_loss:.4f} bcs={bcs_loss:.4f} (active in {n_bcs_active} batches)")

        writer.add_scalar("Loss/total", total_loss, epoch)
        writer.add_scalar("Loss/seg", seg_loss, epoch)
        writer.add_scalar("Loss/vessel", vessel_loss, epoch)
        writer.add_scalar("Loss/soft_bcs", bcs_loss, epoch)

        if (epoch + 1) % config.val_interval == 0:
            val_dice = validate(model, val_loader, dice_metric, config,
                               post_pred, post_label)
            delta = val_dice - baseline_dice
            print(f"  Val Dice: {val_dice:.4f} (delta from L1: {delta:+.4f})")
            writer.add_scalar("Dice/val", val_dice, epoch)
            writer.add_scalar("Dice/delta_from_l1", delta, epoch)
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
                    'baseline_dice': baseline_dice,
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
    print(f"  L1 baseline Dice: {baseline_dice:.4f}")
    print(f"  Delta: {best_dice - baseline_dice:+.4f}")


if __name__ == "__main__":
    main()
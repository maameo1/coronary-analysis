#!/usr/bin/env python3
"""
================================================================================
CT-FM FOUNDATION MODEL FOR CORONARY ARTERY SEGMENTATION
================================================================================

Test CT-FM (Vision Foundation Model for CT) pretrained on 148K CT scans
for coronary artery segmentation.

Hypothesis: CT-specific pretraining may transfer better than MedicalNet
(which showed negative transfer).

Architecture:
    - Encoder: SegResEncoder pretrained on 148K CT scans
    - Decoder: SegResNet decoder (also pretrained)
    - Fine-tune entire network on ImageCAS

Key Differences from Baseline:
    - Different architecture (SegResNet vs U-Net)
    - ~87M parameters vs 4.8M
    - CT-specific pretraining via contrastive learning

Author: Anonymous
Date: November 2025
Experiment: EXP-014

================================================================================
"""

import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from lighter_zoo import SegResNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import pad_list_data_collate

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    SpatialPadd, RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, EnsureTyped, Activations, AsDiscrete
)

from monai.data import Dataset, DataLoader, decollate_batch
from torch.utils.tensorboard import SummaryWriter
from imagecas_data_loader import ImageCASDataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for CT-FM experiment"""
    
    exp_name = "ctfm_segresnet"
    exp_id = "EXP-016"
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    data_root = Path("/path/to/datasets/ImageCAS")
    split_file = Path("/path/to/datasets/ImageCAS/split_1000.csv")
    output_root = Path(f"/path/to/project/experiments/{exp_id}_{exp_name}_{date}")
    
    # Data configuration
    n_train_cases = 638  # Start with 100 to compare with baseline
    test_fold = 1
    val_fraction = 0.15
    
    # CT-FM uses different preprocessing than our baseline
    # Their paper: spacing [3, 1, 1], HU [-1024, 2048]
    # We'll use our standard for fair comparison, but note this
    voxel_spacing = (1.0, 1.0, 1.0)  # Our standard
    patch_size = (96, 96, 96)
    
    # Intensity normalization
    # CT-FM used [-1024, 2048], we use [-200, 600] for cardiac
    hu_min = -200
    hu_max = 600
    
    # Model - loaded from HuggingFace
    model_name = "project-lighter/ct_fm_segresnet"
    
    # Training hyperparameters
    batch_size = 1  # May need to reduce due to larger model
    num_workers = 4
    learning_rate = 1e-4  # Start with same as baseline
    weight_decay = 1e-5
    max_epochs = 100
    val_interval = 1
    
    # Loss weights
    dice_weight = 0.5
    ce_weight = 0.5
    
    # Optimization
    early_stopping_patience = 20
    lr_scheduler_patience = 10
    lr_scheduler_factor = 0.5
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = True
    random_seed = 2025
    
    def __init__(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "checkpoints").mkdir(exist_ok=True)
        (self.output_root / "logs").mkdir(exist_ok=True)
        (self.output_root / "visualizations").mkdir(exist_ok=True)
        self.save_config()
    
    def save_config(self):
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        with open(self.output_root / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(config, mode='train'):
    """Create transform pipeline"""
    
    common_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=config.voxel_spacing, 
                mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], 
                            a_min=config.hu_min, a_max=config.hu_max, 
                            b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=config.patch_size, mode="constant"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config.patch_size,
            pos=1, neg=1, num_samples=1,
            image_key="image",
            image_threshold=0,
            allow_smaller=False
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
    
    if mode == 'train':
        train_transforms = common_transforms + [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
        return Compose(train_transforms)
    else:
        return Compose(common_transforms)


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model(config):
    """Load CT-FM pretrained model"""
    
    print(f"Loading CT-FM from {config.model_name}...")
    
    # Load pretrained CT-FM
    model = SegResNet.from_pretrained(config.model_name)
    
    # The model outputs 2 classes by default (background + foreground)
    # This matches our binary segmentation task
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    return model.to(config.device)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, loss_fn, config, scaler=None):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch_data in train_loader:
        inputs = batch_data["image"].to(config.device)
        labels = batch_data["label"].to(config.device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)


# ============================================================================
# VALIDATION
# ============================================================================

def validate(model, val_loader, metric, config, post_pred, post_label):
    """Validate model"""
    model.eval()
    metric.reset()
    
    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data["image"].to(config.device)
            labels = batch_data["label"].to(config.device)
            
            outputs = model(inputs)
            
            outputs = [post_pred(i) for i in decollate_batch(outputs)]
            labels = [post_label(i) for i in decollate_batch(labels)]
            metric(y_pred=outputs, y=labels)
    
    return metric.aggregate().item()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training function"""
    
    config = Config()
    print(f"Experiment: {config.exp_id} - {config.exp_name}")
    print(f"Output: {config.output_root}")
    print(f"Device: {config.device}")
    
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
    print(f"Using {len(val_data)} validation cases")
    
    # Save splits
    split_info = {
        'train_case_ids': [d['case_id'] for d in train_data],
        'val_case_ids': [d['case_id'] for d in val_data],
        'test_case_ids': [d['case_id'] for d in splits['test']],
    }
    with open(config.output_root / "data_splits.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Create transforms
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
    
    # Model creation
    print("\n" + "="*80)
    print("MODEL CREATION - CT-FM FOUNDATION MODEL")
    print("="*80)
    
    model = create_model(config)
    
    # Loss function
    loss_fn = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True,
        lambda_dice=config.dice_weight, lambda_ce=config.ce_weight
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience)
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    
    scaler = torch.cuda.amp.GradScaler() if config.amp else None
    writer = SummaryWriter(log_dir=config.output_root / "logs")
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    best_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(config.max_epochs):
        print(f"\nEpoch {epoch+1}/{config.max_epochs}")
        print("-" * 40)
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, config, scaler)
        print(f"Train Loss: {train_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        
        if (epoch + 1) % config.val_interval == 0:
            val_dice = validate(model, val_loader, dice_metric, config,
                               post_pred, post_label)
            print(f"Val Dice: {val_dice:.4f}")
            writer.add_scalar("Dice/val", val_dice, epoch)
            
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
                }, config.output_root / "checkpoints" / "best_model.pth")
                
                print(f"New best model! Dice: {best_dice:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best Dice: {best_dice:.4f} at epoch {best_epoch}")
                break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, config.output_root / "checkpoints" / "final_model.pth")
    
    writer.close()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation Dice: {best_dice:.4f} (epoch {best_epoch})")
    print(f"Results saved to: {config.output_root}")
    
    # Save final results
    results = {
        'best_dice': best_dice,
        'best_epoch': best_epoch,
        'final_epoch': epoch + 1,
        'n_train_cases': config.n_train_cases,
    }
    with open(config.output_root / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_dice


if __name__ == "__main__":
    main()
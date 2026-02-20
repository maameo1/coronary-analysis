#!/usr/bin/env python3
"""
CT-FM L4 (Soft BCS) Test Set Inference
=======================================
Runs sliding window inference on 250 test cases using the best checkpoint
from EXP-040. Uses the same MultiTaskCTFM architecture as L1/L2 but
with the L4 soft BCS checkpoint.

Predictions are saved in original space via Invertd, then BCS + standard
metrics are computed.

Author: Anonymous
Date: February 2026
"""

import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / '..' / 'src'))
sys.path.insert(0, '/path/to/project/src')

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped,
    Invertd, AsDiscreted, SaveImaged
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference

from multitask_ctfm import MultiTaskCTFM


def load_l4_model(checkpoint_path, device='cuda'):
    """Load CT-FM L4 model (MultiTaskCTFM architecture)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    model = MultiTaskCTFM(
        pretrained_model_name="project-lighter/ct_fm_segresnet",
        init_filters=32,
        vessel_decoder_filters=16,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    best_dice = checkpoint.get('best_dice', checkpoint.get('best_val_dice', '?'))
    print(f"Loaded L4 checkpoint: epoch={epoch}, best_dice={best_dice}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def run_inference(model, test_dir, output_dir, device='cuda', overlap=0.5):
    """Run sliding window inference on test set with proper inverse transforms."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_images = sorted(Path(test_dir).glob("imagesTs/*.nii.gz"))
    test_files = [{"image": str(img)} for img in test_images]
    print(f"Found {len(test_files)} test cases")

    # Same preprocessing as L4 training: 1.0mm spacing, HU [-200, 600]
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-200, a_max=600, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
    ])

    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=test_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
        SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir=str(output_dir),
            output_postfix="",
            resample=False,
            separate_folder=False,
        ),
    ])

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4)

    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="L4 Inference"):
            inputs = batch_data["image"].to(device)

            # MultiTaskCTFM returns (seg_out, vessel_out) — take seg only
            def predictor_fn(x):
                seg_out, _ = model(x)
                return seg_out

            outputs = sliding_window_inference(
                inputs=inputs,
                roi_size=(96, 96, 96),
                sw_batch_size=4,
                predictor=predictor_fn,
                overlap=overlap,
            )

            batch_data["pred"] = outputs
            batch_data = [post_transforms(i) for i in decollate_batch(batch_data)]

    n_preds = len(list(output_dir.glob("*.nii.gz")))
    print(f"Generated {n_preds} predictions in {output_dir}")
    return n_preds


def compute_metrics(pred_dir, test_dir, output_json):
    """Compute Dice + BCS + topology metrics on predictions.
    Supports resuming: loads existing partial results and skips done cases.
    """
    from skimage.morphology import skeletonize
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from metrics.bifurcation_connectivity import BifurcationConnectivityScore

    bcs_scorer = BifurcationConnectivityScore(tolerance=3, stub_length=8)

    label_dir = Path(test_dir) / "labelsTs"
    pred_files = sorted(Path(pred_dir).glob("*.nii.gz"))

    # Resume support: load existing partial results
    results = []
    done_cases = set()
    if Path(output_json).exists():
        try:
            with open(output_json) as f:
                existing = json.load(f)
            results = existing.get('per_case', [])
            done_cases = {r['case_id'] for r in results}
            print(f"\nResuming: {len(done_cases)} cases already computed, skipping them")
        except Exception:
            pass

    print(f"\nComputing metrics on {len(pred_files)} predictions ({len(pred_files) - len(done_cases)} remaining)...")

    for pred_path in tqdm(pred_files, desc="Metrics"):
        case_name = pred_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")

        if case_name in done_cases:
            continue

        label_path = label_dir / f"{case_name}.nii.gz"
        if not label_path.exists():
            continue

        pred_nii = nib.load(pred_path)
        gt_nii = nib.load(label_path)

        pred = (pred_nii.get_fdata() > 0.5).astype(np.uint8)
        gt = (gt_nii.get_fdata() > 0.5).astype(np.uint8)

        if pred.shape != gt.shape:
            print(f"  Shape mismatch {case_name}: pred={pred.shape}, gt={gt.shape}")
            continue

        spacing = np.array(gt_nii.header.get_zooms()[:3])

        # Dice
        intersection = np.logical_and(pred, gt).sum()
        dice = float(2 * intersection / (pred.sum() + gt.sum() + 1e-8))

        # clDice
        try:
            pred_skel = skeletonize(pred.astype(bool))
            gt_skel = skeletonize(gt.astype(bool))
            tprec = (pred_skel * gt).sum() / (pred_skel.sum() + 1e-8)
            tsens = (gt_skel * pred).sum() / (gt_skel.sum() + 1e-8)
            cldice = float(2 * tprec * tsens / (tprec + tsens + 1e-8))
        except Exception:
            cldice = 0.0

        # BCS
        try:
            bcs = bcs_scorer.compute_score(pred, gt)['bcs']
        except Exception:
            bcs = 0.0

        # HD95
        try:
            pred_surface = pred ^ ndimage.binary_erosion(pred)
            gt_surface = gt ^ ndimage.binary_erosion(gt)
            pred_pts = np.array(np.where(pred_surface)).T * spacing
            gt_pts = np.array(np.where(gt_surface)).T * spacing
            if len(pred_pts) > 0 and len(gt_pts) > 0:
                d1, _ = cKDTree(gt_pts).query(pred_pts)
                d2, _ = cKDTree(pred_pts).query(gt_pts)
                hd95 = float(np.percentile(np.concatenate([d1, d2]), 95))
            else:
                hd95 = 999.0
        except Exception:
            hd95 = 999.0

        # Betti error
        struct = np.ones((3, 3, 3))
        _, b0_pred = ndimage.label(pred, structure=struct)
        _, b0_gt = ndimage.label(gt, structure=struct)
        betti_err = abs(b0_pred - b0_gt)

        results.append({
            'case_id': case_name,
            'dice': dice, 'cldice': cldice, 'bcs': bcs,
            'hd95': hd95, 'betti_error': int(betti_err),
        })
        n = len(results)
        print(f"  [{n:3d}/250] {case_name}  Dice={dice:.4f}  clDice={cldice:.4f}  BCS={bcs:.4f}  HD95={hd95:.2f}  Betti={betti_err}", flush=True)

        # Checkpoint every 10 cases
        if n % 10 == 0:
            _save_results(results, output_json)

    _save_results(results, output_json)
    return results


def _save_results(results, output_json):
    """Save current results with summary stats."""
    if not results:
        return
    dices = [r['dice'] for r in results]
    bcss = [r['bcs'] for r in results]
    cldices = [r['cldice'] for r in results]
    hd95s = [r['hd95'] for r in results]

    summary = {
        'n_cases': len(results),
        'dice_mean': float(np.mean(dices)), 'dice_std': float(np.std(dices)),
        'cldice_mean': float(np.mean(cldices)), 'cldice_std': float(np.std(cldices)),
        'bcs_mean': float(np.mean(bcss)), 'bcs_std': float(np.std(bcss)),
        'hd95_mean': float(np.mean(hd95s)), 'hd95_std': float(np.std(hd95s)),
        'betti_error_mean': float(np.mean([r['betti_error'] for r in results])),
    }

    print(f"\n{'='*60}")
    print(f"CT-FM L4 TEST RESULTS (n={len(results)})")
    print(f"{'='*60}")
    print(f"  Dice:    {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    print(f"  clDice:  {summary['cldice_mean']:.4f} ± {summary['cldice_std']:.4f}")
    print(f"  BCS:     {summary['bcs_mean']:.4f} ± {summary['bcs_std']:.4f}")
    print(f"  HD95:    {summary['hd95_mean']:.2f} ± {summary['hd95_std']:.2f}")
    print(f"  Betti:   {summary['betti_error_mean']:.2f}")

    with open(output_json, 'w') as f:
        json.dump({'summary': summary, 'per_case': results}, f, indent=2)
    print(f"Saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="CT-FM L4 Test Inference + Metrics")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap (default 0.5)")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Skip inference, only compute metrics")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pred_dir = Path(args.output_dir)

    if not args.metrics_only:
        model = load_l4_model(args.checkpoint, device)
        run_inference(model, args.test_dir, args.output_dir, device, overlap=args.overlap)

    # Compute metrics
    metrics_json = pred_dir / "l4_test_metrics.json"
    compute_metrics(pred_dir, args.test_dir, str(metrics_json))


if __name__ == "__main__":
    main()

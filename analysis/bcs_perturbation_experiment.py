"""
BCS Sensitivity to Simulated Pathology
=======================================

Validates that BCS detects clinically meaningful topological defects
that Dice overlooks. Three controlled perturbations applied to real
ImageCAS test predictions:

1. Branch Severing: Complete vessel transection at bifurcation (total occlusion)
2. Focal Stenosis: Cylindrical narrowing of vessel lumen over short segment
   (reduces radius to ~25% while maintaining tubular structure)
3. Distal Pruning: Remove terminal branches furthest from ostium (root)
   (simulates microvascular disease affecting distal vessels)

For each perturbation, measure Dice change vs BCS change.
Expected: Dice changes minimally, BCS drops for connectivity-affecting perturbations.
"""

import numpy as np
import nibabel as nib
import json
import sys
from pathlib import Path
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from metrics.bifurcation_connectivity import BifurcationConnectivityScore


TEST_DIR = Path.home() / "cardiac-imaging/nnUNet/nnUNet_raw/Dataset105_ImageCAS_638cases_testset"
PRED_DIR = Path.home() / "cardiac-imaging/test_results"
OUTPUT_DIR = Path.home() / "cardiac-imaging/miccai2026/results"

# Use a well-performing model for the perturbation experiment
MODEL = "ctfm_l0_638"
N_CASES = 10  # Enough cases for robust stats, fast enough to run on login node


def dice(a, b):
    inter = np.sum(a & b)
    total = np.sum(a) + np.sum(b)
    if total == 0:
        return 1.0
    return 2 * inter / total


def load_case(case_id):
    """Load GT and prediction for a case."""
    gt_path = TEST_DIR / "labelsTs" / f"case{case_id:05d}.nii.gz"
    pred_path = PRED_DIR / MODEL / f"case{case_id:05d}_0000.nii.gz"

    if not gt_path.exists() or not pred_path.exists():
        return None, None

    gt = nib.load(str(gt_path)).get_fdata() > 0.5
    pred = nib.load(str(pred_path)).get_fdata() > 0.5
    return gt, pred


def find_good_cases():
    """Find cases where original prediction has high Dice and >=5 bifurcations."""
    bcs_metric = BifurcationConnectivityScore(tolerance=3, stub_length=8)
    good_cases = []

    pred_dir = PRED_DIR / MODEL
    pred_files = sorted(pred_dir.glob("case*_0000.nii.gz"))

    print(f"Scanning {len(pred_files)} predictions for good cases...")

    for pf in pred_files:
        # case00751_0000.nii.gz -> 751
        case_id = int(pf.stem.split("_")[0].replace("case", ""))
        gt, pred = load_case(case_id)
        if gt is None:
            continue

        d = dice(pred, gt)
        if d < 0.75:
            continue

        # Quick bifurcation count from GT skeleton
        gt_skel = skeletonize(gt)
        bif_clusters = bcs_metric.get_bifurcation_clusters(gt_skel)
        n_bif = len(bif_clusters)

        if n_bif >= 5:
            scores = bcs_metric.compute_score(pred, gt)
            if scores['bcs'] >= 0.6:
                good_cases.append({
                    'case_id': case_id,
                    'dice': d,
                    'bcs': scores['bcs'],
                    'n_bif': n_bif,
                    'n_preserved': scores['n_preserved'],
                })
                print(f"  Case {case_id}: Dice={d:.3f}, BCS={scores['bcs']:.3f}, "
                      f"bif={n_bif}, preserved={scores['n_preserved']}")

        if len(good_cases) >= N_CASES:
            break

    return good_cases


def perturbation_branch_sever(pred, gt, bcs_metric):
    """
    Sever a branch at a randomly selected bifurcation.
    Remove ~15 voxels of vessel on one child branch.
    """
    gt_skel = skeletonize(gt)
    bif_clusters = bcs_metric.get_bifurcation_clusters(gt_skel)

    if len(bif_clusters) == 0:
        return pred, {}

    # Pick a random bifurcation
    idx = np.random.randint(len(bif_clusters))
    bif = bif_clusters[idx]
    stubs = bcs_metric.get_branch_stubs(gt_skel, bif)

    if len(stubs) < 2:
        return pred, {}

    # Find the stub that overlaps with prediction, pick one to sever
    # Dilate the stub to get the actual vessel region to remove
    stub_to_sever = stubs[-1]  # Last stub (usually a child branch)

    # Dilate stub to get the vessel volume around it
    sever_region = ndimage.binary_dilation(stub_to_sever, iterations=5)

    # Also extend further along the branch direction
    # Get the branch segment beyond the stub
    all_bif = bcs_metric.find_bifurcation_points(gt_skel)
    skel_no_bif = gt_skel.copy()
    skel_no_bif[all_bif] = False

    struct = ndimage.generate_binary_structure(3, 3)
    labeled_branches, _ = ndimage.label(skel_no_bif, structure=struct)

    # Find which branch the stub belongs to
    stub_labels = set(np.unique(labeled_branches[stub_to_sever])) - {0}
    if stub_labels:
        branch_label = list(stub_labels)[0]
        branch_mask = (labeled_branches == branch_label)
        # Dilate the full branch to get vessel volume
        branch_vessel = ndimage.binary_dilation(branch_mask, iterations=4)
        sever_region = sever_region | branch_vessel

    # Apply severing
    pred_severed = pred.copy()
    pred_severed[sever_region] = False

    voxels_removed = np.sum(pred & sever_region)

    return pred_severed, {
        'type': 'branch_sever',
        'voxels_removed': int(voxels_removed),
        'pct_removed': float(voxels_removed / max(pred.sum(), 1) * 100),
        'bifurcation_idx': idx,
    }


def perturbation_focal_stenosis(pred, gt, bcs_metric):
    """
    Simulate focal stenosis: cylindrical narrowing along vessel axis.

    A real stenosis narrows the vessel lumen over a short segment (5-15mm)
    but maintains the tubular structure. We reduce the vessel radius to ~30-40%
    of original along a segment of the skeleton.
    """
    gt_skel = skeletonize(gt)

    # Find a mid-vessel segment (not at bifurcations, not at endpoints)
    all_bif = bcs_metric.find_bifurcation_points(gt_skel)

    # Find endpoints
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = ndimage.convolve(gt_skel.astype(np.uint8), kernel,
                                       mode='constant', cval=0)
    endpoints = (gt_skel > 0) & (neighbor_count == 1)

    # Get skeleton points that are not bifurcations or endpoints (mid-vessel)
    mid_vessel = gt_skel & ~all_bif & ~endpoints
    mid_coords = np.array(np.where(mid_vessel)).T

    if len(mid_coords) < 20:
        return pred, {}

    # Pick a random mid-vessel point as stenosis center
    center_idx = np.random.randint(len(mid_coords))
    center = mid_coords[center_idx]

    # Get nearby skeleton points to define the stenosis segment (~10 voxels along vessel)
    skel_coords = np.array(np.where(gt_skel)).T
    dists_to_center = np.linalg.norm(skel_coords - center, axis=1)
    stenosis_segment_mask = dists_to_center <= 8  # ~8 voxel segment length
    stenosis_skel_coords = skel_coords[stenosis_segment_mask]

    if len(stenosis_skel_coords) < 5:
        return pred, {}

    # Create stenosis by eroding the vessel in this region
    # First, identify the vessel region around the stenosis segment
    stenosis_skel = np.zeros_like(pred, dtype=bool)
    for coord in stenosis_skel_coords:
        stenosis_skel[coord[0], coord[1], coord[2]] = True

    # Dilate skeleton to approximate original vessel, then create narrowed version
    # Original vessel ~ 3-4 voxel radius, stenosis ~ 1-2 voxel radius
    original_region = ndimage.binary_dilation(stenosis_skel, iterations=4)
    narrowed_region = ndimage.binary_dilation(stenosis_skel, iterations=1)

    # The stenosis effect: remove the outer shell of the vessel in this region
    erosion_mask = original_region & ~narrowed_region

    pred_stenosed = pred.copy()
    pred_stenosed[erosion_mask] = False

    voxels_removed = np.sum(pred & erosion_mask)

    return pred_stenosed, {
        'type': 'focal_stenosis',
        'voxels_removed': int(voxels_removed),
        'pct_removed': float(voxels_removed / max(pred.sum(), 1) * 100),
        'stenosis_center': center.tolist(),
        'segment_length_voxels': len(stenosis_skel_coords),
    }


def perturbation_distal_pruning(pred, gt, bcs_metric):
    """
    Remove distal tips of terminal branches (furthest from root/ostium).

    Distal pruning simulates microvascular disease affecting the terminal
    tips of vessels - NOT entire branches. We remove only the last ~15-20
    voxels of skeleton near each distal endpoint, not the whole branch.
    """
    gt_skel = skeletonize(gt)

    # Find endpoints (skeleton voxels with exactly 1 neighbor)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbor_count = ndimage.convolve(gt_skel.astype(np.uint8), kernel,
                                       mode='constant', cval=0)
    endpoints = (gt_skel > 0) & (neighbor_count == 1)
    endpoint_coords = np.array(np.where(endpoints)).T

    if len(endpoint_coords) < 4:
        return pred, {}

    # Identify the root (ostium) - typically the endpoint with largest z-coordinate
    z_coords = endpoint_coords[:, 2]
    root_idx = np.argmax(z_coords)
    root_coord = endpoint_coords[root_idx]

    # Compute distance from root for each endpoint
    distances_from_root = np.linalg.norm(endpoint_coords - root_coord, axis=1)

    # Get distal endpoints (excluding root), sorted by distance from root
    distal_endpoints = []
    for i, ep in enumerate(endpoint_coords):
        if i == root_idx:
            continue
        distal_endpoints.append({
            'coord': ep,
            'distance_from_root': distances_from_root[i],
        })

    if len(distal_endpoints) < 3:
        return pred, {}

    # Sort by distance from root (descending) - most distal first
    distal_endpoints.sort(key=lambda x: -x['distance_from_root'])

    # Remove just the TIPS of the 3 most distal branches
    # Only remove skeleton voxels within ~15 voxels of the endpoint
    TIP_LENGTH = 15  # voxels of skeleton to remove from each tip
    n_tips_to_remove = min(3, len(distal_endpoints))

    skel_coords = np.array(np.where(gt_skel)).T
    prune_mask = np.zeros_like(pred, dtype=bool)

    for tip_info in distal_endpoints[:n_tips_to_remove]:
        ep = tip_info['coord']
        # Find skeleton voxels near this endpoint (the tip)
        dists_to_ep = np.linalg.norm(skel_coords - ep, axis=1)
        tip_mask = dists_to_ep <= TIP_LENGTH
        tip_skel_coords = skel_coords[tip_mask]

        # Create tip skeleton and dilate to get vessel volume
        tip_skel = np.zeros_like(pred, dtype=bool)
        for coord in tip_skel_coords:
            tip_skel[coord[0], coord[1], coord[2]] = True

        tip_vessel = ndimage.binary_dilation(tip_skel, iterations=4)
        prune_mask |= tip_vessel

    pred_pruned = pred.copy()
    pred_pruned[prune_mask] = False

    voxels_removed = np.sum(pred & prune_mask)

    return pred_pruned, {
        'type': 'distal_pruning',
        'tips_removed': n_tips_to_remove,
        'tip_length_voxels': TIP_LENGTH,
        'voxels_removed': int(voxels_removed),
        'pct_removed': float(voxels_removed / max(pred.sum(), 1) * 100),
        'avg_distance_from_root': float(np.mean([t['distance_from_root'] for t in distal_endpoints[:n_tips_to_remove]])),
    }


def main():
    np.random.seed(42)
    bcs_metric = BifurcationConnectivityScore(tolerance=3, stub_length=8)

    print("=" * 70)
    print("BCS Sensitivity to Simulated Pathology")
    print("=" * 70)

    # Step 1: Find good cases
    print("\n--- Finding suitable test cases ---")
    good_cases = find_good_cases()

    if len(good_cases) < 3:
        print(f"Only found {len(good_cases)} good cases. Need at least 3.")
        return

    print(f"\nUsing {len(good_cases)} cases for perturbation experiment")

    # Step 2: Apply perturbations
    perturbations = {
        'branch_sever': perturbation_branch_sever,
        'focal_stenosis': perturbation_focal_stenosis,
        'distal_pruning': perturbation_distal_pruning,
    }

    all_results = {
        'model': MODEL,
        'n_cases': len(good_cases),
        'cases': [],
    }

    for case_info in good_cases:
        case_id = case_info['case_id']
        gt, pred = load_case(case_id)
        if gt is None:
            continue

        print(f"\n{'='*60}")
        print(f"Case {case_id}: original Dice={case_info['dice']:.3f}, "
              f"BCS={case_info['bcs']:.3f}")
        print(f"{'='*60}")

        case_results = {
            'case_id': case_id,
            'original': {
                'dice': case_info['dice'],
                'bcs': case_info['bcs'],
                'n_bif': case_info['n_bif'],
            },
            'perturbations': {},
        }

        for pert_name, pert_fn in perturbations.items():
            pred_perturbed, pert_info = pert_fn(pred, gt, bcs_metric)

            if not pert_info:
                print(f"  {pert_name}: skipped (no suitable target)")
                continue

            # Compute metrics on perturbed prediction
            d_pert = dice(pred_perturbed, gt)
            bcs_pert = bcs_metric.compute_score(pred_perturbed, gt)

            delta_dice = d_pert - case_info['dice']
            delta_bcs = bcs_pert['bcs'] - case_info['bcs']

            print(f"  {pert_name}:")
            print(f"    Voxels removed: {pert_info['voxels_removed']} "
                  f"({pert_info['pct_removed']:.1f}%)")
            print(f"    Dice: {case_info['dice']:.3f} → {d_pert:.3f} "
                  f"(Δ={delta_dice:+.3f})")
            print(f"    BCS:  {case_info['bcs']:.3f} → {bcs_pert['bcs']:.3f} "
                  f"(Δ={delta_bcs:+.3f})")
            print(f"    Bif preserved: {bcs_pert['n_preserved']}/{bcs_pert['n_expected']}")

            case_results['perturbations'][pert_name] = {
                'dice_after': float(d_pert),
                'bcs_after': float(bcs_pert['bcs']),
                'delta_dice': float(delta_dice),
                'delta_bcs': float(delta_bcs),
                'n_preserved': int(bcs_pert['n_preserved']),
                'n_expected': int(bcs_pert['n_expected']),
                **{k: v for k, v in pert_info.items()
                   if k not in ['stenosis_center']},
            }

        all_results['cases'].append(case_results)

    # Step 3: Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for pert_name in perturbations:
        deltas_dice = []
        deltas_bcs = []
        for cr in all_results['cases']:
            if pert_name in cr['perturbations']:
                deltas_dice.append(cr['perturbations'][pert_name]['delta_dice'])
                deltas_bcs.append(cr['perturbations'][pert_name]['delta_bcs'])

        if deltas_dice:
            print(f"\n{pert_name} (n={len(deltas_dice)}):")
            print(f"  ΔDice: {np.mean(deltas_dice):+.4f} ± {np.std(deltas_dice):.4f}")
            print(f"  ΔBCS:  {np.mean(deltas_bcs):+.4f} ± {np.std(deltas_bcs):.4f}")
            print(f"  Ratio |ΔBCS/ΔDice|: {abs(np.mean(deltas_bcs))/max(abs(np.mean(deltas_dice)), 1e-6):.1f}x")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "bcs_perturbation_results.json"

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

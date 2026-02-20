#!/usr/bin/env python3
"""
Pre-compute bifurcation stub label maps for Soft BCS Loss training.

For each training case, computes:
1. GT skeleton
2. Bifurcation clusters (voxels with >=3 neighbors)
3. Branch stubs for each bifurcation (typically 3: 1 parent + 2 children)
4. Encodes into integer label map: value = bif_id * 4 + stub_idx

The stub label map is saved as NIfTI (int16) and goes through the same
MONAI spatial transforms as images/labels during training (nearest interp).

Unlike the Gaussian weight maps used by bifocal loss, these maps preserve
per-bifurcation, per-stub identity — enabling the Soft BCS Loss to compute
min(mean_prob_per_stub) for each bifurcation.

Usage:
    python precompute_stub_labels.py

Output: One NIfTI per case in data/ImageCAS/stub_labels/

Author: Anonymous
Date: February 2026
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage.morphology import skeletonize
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics.bifurcation_connectivity import BifurcationConnectivityScore

# Paths
DATA_DIR = Path.home() / "cardiac-imaging/data/ImageCAS"
OUTPUT_DIR = Path.home() / "cardiac-imaging/data/ImageCAS/stub_labels"

# BCS parameters (must match evaluation)
TOLERANCE = 3
STUB_LENGTH = 8


def compute_stub_label_map(label_path):
    """Compute a stub label map for a single case.

    Returns a 3D int16 array where:
    - 0 = not a stub
    - bif_id * 4 + stub_idx = stub belonging to bifurcation bif_id (1-based),
      stub_idx in {1, 2, 3}

    Encoding: bif_id = value // 4, stub_idx = value % 4
    With int16, supports up to ~8000 bifurcations per case.
    """
    nii = nib.load(str(label_path))
    gt = (nii.get_fdata() > 0.5).astype(np.uint8)

    if gt.sum() < 100:
        return np.zeros(gt.shape, dtype=np.int16), nii.affine, nii.header, 0, 0

    # Use BCS scorer's methods for consistent bifurcation detection
    bcs = BifurcationConnectivityScore(tolerance=TOLERANCE, stub_length=STUB_LENGTH)

    skel = bcs.extract_skeleton(gt)
    bif_clusters = bcs.get_bifurcation_clusters(skel)

    if len(bif_clusters) == 0:
        return np.zeros(gt.shape, dtype=np.int16), nii.affine, nii.header, 0, 0

    stub_map = np.zeros(gt.shape, dtype=np.int16)
    n_valid_bifs = 0
    total_stubs = 0

    for bif_idx, bif_cluster in enumerate(bif_clusters):
        bif_id = bif_idx + 1  # 1-based
        stubs = bcs.get_branch_stubs(skel, bif_cluster)

        if len(stubs) < 2:
            continue  # Need at least 2 stubs for meaningful bifurcation

        n_valid_bifs += 1

        for stub_idx, stub_mask in enumerate(stubs[:3]):  # Cap at 3 stubs
            value = bif_id * 4 + (stub_idx + 1)  # stub_idx 1-based
            # Don't overwrite existing labels (in case stubs overlap)
            new_voxels = stub_mask & (stub_map == 0)
            stub_map[new_voxels] = value
            total_stubs += 1

    return stub_map, nii.affine, nii.header, n_valid_bifs, total_stubs


def process_case(label_path, output_path):
    """Process a single case."""
    case_name = label_path.stem.replace('.label', '')

    try:
        stub_map, affine, header, n_bifs, n_stubs = compute_stub_label_map(label_path)

        # Save as int16 NIfTI
        nii_out = nib.Nifti1Image(stub_map, affine, header)
        nib.save(nii_out, str(output_path))

        return case_name, n_bifs, n_stubs, True
    except Exception as e:
        print(f"  ERROR {case_name}: {e}")
        return case_name, 0, 0, False


def main():
    print("=" * 70)
    print("Pre-computing Bifurcation Stub Label Maps")
    print(f"Parameters: tolerance={TOLERANCE}, stub_length={STUB_LENGTH}")
    print(f"Encoding: value = bif_id * 4 + stub_idx (1-3)")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all label files
    subdirs = ["1-200", "201-400", "401-600", "601-800", "801-1000"]
    label_files = []
    for subdir in subdirs:
        subdir_path = DATA_DIR / subdir
        if subdir_path.exists():
            label_files.extend(sorted(subdir_path.glob("*.label.nii.gz")))

    print(f"Found {len(label_files)} label files")

    # Process in parallel
    n_workers = min(16, multiprocessing.cpu_count())
    print(f"Processing with {n_workers} workers...")

    tasks = []
    for lf in label_files:
        case_name = lf.stem.replace('.label', '')
        out_path = OUTPUT_DIR / f"{case_name}.stub_labels.nii.gz"
        tasks.append((lf, out_path))

    processed = 0
    succeeded = 0
    total_bifs = 0
    total_stubs = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_case, lf, op): lf
            for lf, op in tasks
        }

        for future in as_completed(futures):
            processed += 1
            case_name, n_bifs, n_stubs, success = future.result()
            if success:
                succeeded += 1
                total_bifs += n_bifs
                total_stubs += n_stubs
                if processed % 50 == 0:
                    print(f"  [{processed}/{len(tasks)}] {case_name}: "
                          f"{n_bifs} bifurcations, {n_stubs} stubs")
            else:
                print(f"  [{processed}/{len(tasks)}] {case_name}: FAILED")

    print(f"\nDone: {succeeded}/{len(tasks)} cases processed")
    print(f"Total bifurcations: {total_bifs}")
    print(f"Total stubs: {total_stubs}")
    print(f"Mean bifurcations per case: {total_bifs / max(succeeded, 1):.1f}")
    print(f"Mean stubs per bifurcation: {total_stubs / max(total_bifs, 1):.1f}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

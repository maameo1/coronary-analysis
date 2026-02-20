#!/usr/bin/env python3
"""
Bifurcation Connectivity Score (BCS) - Correct Implementation
==============================================================

Measures the percentage of bifurcations preserved in coronary artery segmentation.

Algorithm:
1. Skeletonize the ground truth segmentation.
2. Find all bifurcation points (skeleton voxels with >=3 neighbors).
3. For each bifurcation, identify the 3 branch stubs meeting there
   (1 parent toward aortic root, 2 children downstream).
4. In the prediction's (dilated) skeleton, check whether all 3 branch
   stubs are mutually reachable — i.e., connected through the prediction.
5. A bifurcation is "preserved" only if ALL 3 branches are connected.
   If any connection is broken, the bifurcation counts as "lost."

Final score: preserved bifurcations / total bifurcations.

Clinically interpretable: "23 of 27 bifurcations preserved" tells a
cardiologist exactly how many junctions failed and which ones.

Author: Anonymous
Date: January 2026
"""

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Dict, List
import torch


class BifurcationConnectivityScore:
    """
    Bifurcation Connectivity Score for coronary artery segmentation.

    Algorithm:
    1. Skeletonize the ground truth segmentation.
    2. Find all bifurcation points (skeleton voxels with >=3 neighbors).
    3. For each bifurcation, identify the 3 branch stubs meeting there
       (1 parent toward aortic root, 2 children downstream).
    4. In the prediction's (dilated) skeleton, check whether all 3 branch
       stubs are mutually reachable — i.e., connected through the prediction.
    5. A bifurcation is "preserved" only if ALL 3 branches are connected.
       If any connection is broken, the bifurcation counts as "lost."

    Final score: preserved bifurcations / total bifurcations.

    Clinically interpretable: "23 of 27 bifurcations preserved" tells a
    cardiologist exactly how many junctions failed and which ones.
    """

    def __init__(self, tolerance: int = 3, stub_length: int = 8):
        """
        Args:
            tolerance: Dilation radius (voxels) for prediction skeleton
                       to handle small alignment differences.
            stub_length: Length (voxels) of branch stubs to extract around
                         each bifurcation for the connectivity check.
        """
        self.tolerance = tolerance
        self.stub_length = stub_length

    def extract_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        """Extract 3D skeleton from binary segmentation."""
        return skeletonize(binary_mask.astype(bool))

    def find_bifurcation_points(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Find bifurcation points in skeleton (voxels with >=3 neighbors).
        Returns binary mask.
        """
        kernel = np.ones((3, 3, 3), dtype=np.uint8)
        kernel[1, 1, 1] = 0

        neighbor_count = ndimage.convolve(
            skeleton.astype(np.uint8),
            kernel,
            mode='constant',
            cval=0
        )

        return (skeleton > 0) & (neighbor_count > 2)

    def get_bifurcation_clusters(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """
        Cluster adjacent bifurcation voxels into individual bifurcation regions.

        Returns list of binary masks, one per bifurcation cluster.
        """
        bif_mask = self.find_bifurcation_points(skeleton)

        if not np.any(bif_mask):
            return []

        struct = ndimage.generate_binary_structure(3, 3)
        labeled, n_clusters = ndimage.label(bif_mask, structure=struct)

        clusters = []
        for i in range(1, n_clusters + 1):
            clusters.append(labeled == i)

        return clusters

    def get_branch_stubs(
        self,
        skeleton: np.ndarray,
        bif_cluster: np.ndarray
    ) -> List[np.ndarray]:
        """
        For a single bifurcation cluster, find the branch stubs meeting there.

        Removes the bifurcation voxels from the skeleton, then finds the
        connected components adjacent to the cluster. Each component is a
        branch stub (parent or child).

        Returns list of binary masks for each stub (typically 3: 1 parent + 2 children).
        """
        struct = ndimage.generate_binary_structure(3, 3)

        # Remove ALL bifurcation points (not just this cluster) to cleanly
        # separate branches
        all_bif = self.find_bifurcation_points(skeleton)
        skel_no_bif = skeleton.copy().astype(bool)
        skel_no_bif[all_bif] = False

        # Label remaining branch segments
        labeled_branches, n_branches = ndimage.label(skel_no_bif, structure=struct)

        # Find which branch segments are adjacent to this bifurcation cluster
        # Dilate the cluster by 1 voxel to catch neighbors
        dilated_cluster = ndimage.binary_dilation(bif_cluster, structure=struct,
                                                  iterations=1)

        # Find branch labels that overlap with the dilated cluster
        adjacent_labels = set(np.unique(labeled_branches[dilated_cluster])) - {0}

        stubs = []
        for lbl in adjacent_labels:
            branch_mask = (labeled_branches == lbl)
            # Only keep the stub portion close to the bifurcation
            # (within stub_length voxels)
            stub = self._extract_stub(branch_mask, bif_cluster)
            if np.any(stub):
                stubs.append(stub)

        return stubs

    def _extract_stub(
        self,
        branch_mask: np.ndarray,
        bif_cluster: np.ndarray
    ) -> np.ndarray:
        """
        Extract the portion of a branch within stub_length voxels
        of the bifurcation cluster. This is the "endpoint" we check
        for reachability in the prediction.
        """
        # Get branch voxel coordinates
        branch_coords = np.array(np.where(branch_mask)).T
        if len(branch_coords) == 0:
            return np.zeros_like(branch_mask, dtype=bool)

        # Get bifurcation cluster centroid
        bif_coords = np.array(np.where(bif_cluster)).T
        bif_centroid = bif_coords.mean(axis=0)

        # Distance of each branch voxel from the bifurcation
        dists = np.linalg.norm(branch_coords - bif_centroid, axis=1)

        # Keep voxels within stub_length
        stub_mask = np.zeros_like(branch_mask, dtype=bool)
        close_voxels = branch_coords[dists <= self.stub_length]
        if len(close_voxels) > 0:
            stub_mask[tuple(close_voxels.T)] = True

        return stub_mask

    def is_bifurcation_preserved(
        self,
        pred_skel_dilated: np.ndarray,
        stubs: List[np.ndarray]
    ) -> bool:
        """
        Check if a bifurcation is preserved in the prediction.

        A bifurcation is preserved if ALL branch stubs meeting at it
        are mutually reachable through the prediction's skeleton.

        This means: for every pair of stubs, there exists a connected
        path through pred_skel_dilated.

        Args:
            pred_skel_dilated: Prediction skeleton dilated by tolerance.
            stubs: List of branch stub masks (typically 3).

        Returns:
            True if all stubs are mutually connected through prediction.
        """
        if len(stubs) < 2:
            # Need at least 2 branches to form a meaningful bifurcation
            return len(stubs) > 0

        # Label connected components in prediction skeleton
        struct = ndimage.generate_binary_structure(3, 3)
        labeled_pred, _ = ndimage.label(pred_skel_dilated, structure=struct)

        # For each stub, find which prediction component(s) it overlaps with
        stub_components = []
        for stub in stubs:
            overlap = labeled_pred[stub]
            component_labels = set(np.unique(overlap)) - {0}
            if not component_labels:
                # This stub has no prediction skeleton nearby → not preserved
                return False
            stub_components.append(component_labels)

        # All stubs must share at least one common component
        # (i.e., they are all mutually reachable)
        common = stub_components[0]
        for sc in stub_components[1:]:
            common = common & sc
            if not common:
                return False

        return True

    def compute_score(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Bifurcation Connectivity Score.

        Args:
            pred_mask: Binary prediction (H, W, D)
            gt_mask: Binary ground truth (H, W, D)

        Returns:
            Dict with:
                - bcs: Bifurcation Connectivity Score (0-1)
                - n_preserved: Number of bifurcations preserved
                - n_expected: Total expected bifurcations (from GT)
                - preservation_rate: bcs (alias for clarity)
        """
        gt_skel = self.extract_skeleton(gt_mask)
        pred_skel = self.extract_skeleton(pred_mask)

        # Dilate prediction skeleton slightly for tolerance
        pred_skel_dilated = ndimage.binary_dilation(
            pred_skel,
            iterations=self.tolerance
        )

        # Get bifurcation clusters from GT
        bif_clusters = self.get_bifurcation_clusters(gt_skel)
        n_expected = len(bif_clusters)

        if n_expected == 0:
            return {
                'bcs': 1.0,
                'n_preserved': 0,
                'n_expected': 0,
                'preservation_rate': 1.0
            }

        n_preserved = 0

        for bif_cluster in bif_clusters:
            # Get the 3 branch stubs meeting at this bifurcation
            stubs = self.get_branch_stubs(gt_skel, bif_cluster)

            # Check if all stubs are mutually reachable in prediction
            if self.is_bifurcation_preserved(pred_skel_dilated, stubs):
                n_preserved += 1

        bcs = n_preserved / n_expected

        return {
            'bcs': bcs,
            'n_preserved': n_preserved,
            'n_expected': n_expected,
            'preservation_rate': bcs
        }

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, float]:
        """Compute BCS for PyTorch tensors."""
        if pred.dim() == 5:
            pred = pred[0]
            target = target[0]

        if pred.shape[0] > 1:
            pred_mask = pred[1].cpu().numpy() > 0.5
            target_mask = target[1].cpu().numpy() > 0.5
        else:
            pred_mask = pred[0].cpu().numpy() > 0.5
            target_mask = target[0].cpu().numpy() > 0.5

        return self.compute_score(pred_mask, target_mask)


if __name__ == "__main__":
    # Test with synthetic bifurcation
    print("Testing Bifurcation Connectivity Score...")

    # Create Y-shaped bifurcation: parent splits into 2 children
    gt = np.zeros((64, 64, 64), dtype=np.uint8)

    # Parent branch (vertical, from bottom)
    gt[20:40, 30:32, 30:32] = 1

    # Bifurcation region
    gt[38:42, 28:34, 28:34] = 1

    # Child branch 1 (goes left)
    gt[40:55, 25:30, 30:32] = 1

    # Child branch 2 (goes right)
    gt[40:55, 34:39, 30:32] = 1

    # Perfect prediction
    pred_perfect = gt.copy()

    # Broken prediction - disconnect one child
    pred_broken = gt.copy()
    pred_broken[40:45, 34:39, 30:32] = 0  # Break child 2

    bcs = BifurcationConnectivityScore(tolerance=3, stub_length=8)

    # Test perfect
    scores_perfect = bcs.compute_score(pred_perfect, gt)
    print(f"\nPerfect prediction:")
    print(f"  BCS: {scores_perfect['bcs']:.2%}")
    print(f"  Bifurcations: {scores_perfect['n_expected']}")
    print(f"  Preserved: {scores_perfect['n_preserved']}")

    # Test broken
    scores_broken = bcs.compute_score(pred_broken, gt)
    print(f"\nBroken prediction (one child disconnected):")
    print(f"  BCS: {scores_broken['bcs']:.2%}")
    print(f"  Bifurcations: {scores_broken['n_expected']}")
    print(f"  Preserved: {scores_broken['n_preserved']}")

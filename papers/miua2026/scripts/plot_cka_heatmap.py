#!/usr/bin/env python3
"""
Figure 2: CKA heatmap — encoder representation drift from baseline.

Rows = models (frozen, partial, full), Columns = encoder levels (0-4).
Color intensity = CKA similarity to L1 (1.0 = identical, 0.0 = completely different).
Annotations show exact CKA values in each cell.

Key finding: Frozen/partial stay >0.84 everywhere; full drops to 0.37 at bottleneck.

Input:  miua2026/results/cka/cka_results.json
Output: miua2026/figures/fig1_cka_heatmap.pdf
        miua2026/figures/fig1_cka_heatmap.png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "miua2026" / "results" / "cka" / "cka_results.json"
OUTPUT_DIR = PROJECT_ROOT / "miua2026" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "mathtext.fontset": "cm",
})

# Model display order and labels
MODELS = [
    ("EXP-044 (frozen, w=0.03)",  "Frozen ($w$=0.03)"),
    ("EXP-048 (frozen, w=0.05)",  "Frozen ($w$=0.05)"),
    ("EXP-047 (partial, w=0.05)", "Partial ($w$=0.05)"),
    ("Full+BCS matched (w=0.10, LR=1e-5)",    "Full+BCS (LR=$10^{-5}$)"),
    ("Full+clDice (w=0.10, LR=1e-5)",         "Full+clDice (LR=$10^{-5}$)"),
    ("EXP-045 (full, w=0.10)",    "Full (LR=$10^{-4}$)$^\\dagger$"),
]

LEVELS = ["level_0", "level_1", "level_2", "level_3", "level_4"]
LEVEL_LABELS = ["L0\n32ch, 96$^3$", "L1\n64ch, 48$^3$", "L2\n128ch, 24$^3$",
                "L3\n256ch, 12$^3$", "L4\n512ch, 6$^3$"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    results = data["results"]

    # Build matrix
    n_models = len(MODELS)
    n_levels = len(LEVELS)
    matrix = np.zeros((n_models, n_levels))

    for i, (key, _) in enumerate(MODELS):
        if key not in results:
            print(f"WARNING: {key} not found in CKA results")
            continue
        for j, level in enumerate(LEVELS):
            matrix[i, j] = results[key][level]["cka"]

    fig, ax = plt.subplots(figsize=(10, 4.8))

    # Heatmap: colorblind-safe diverging map (blue=low, white=mid, orange=high)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.15, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(n_models):
        for j in range(n_levels):
            val = matrix[i, j]
            color = "white" if val < 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    # Labels
    ax.set_xticks(range(n_levels))
    ax.set_xticklabels(LEVEL_LABELS, ha="center")
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([label for _, label in MODELS])
    ax.set_xlabel("Encoder Level")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("CKA Similarity to Baseline", fontsize=14)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"fig1_cka_heatmap.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    # Print summary
    print("\nCKA Summary:")
    for i, (key, label) in enumerate(MODELS):
        vals = [f"{matrix[i,j]:.3f}" for j in range(n_levels)]
        print(f"  {label:25s}  {' → '.join(vals)}")


if __name__ == "__main__":
    main()

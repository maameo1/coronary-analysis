#!/usr/bin/env python3
"""
Figure 3 (Money Figure): Dice vs BCS scatter plot.

Shows that high Dice does not guarantee good topology (BCS).
Highlights discordant cases (Dice>0.8 but BCS<0.7).

Input:  test_results/lcc_results_<model>.json  (one per model)
Output: miccai2026/figures/fig3_dice_vs_bcs.pdf
        miccai2026/figures/fig3_dice_vs_bcs.png
        miccai2026/results/discordant_cases.json
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "test_results"
OUTPUT_DIR = PROJECT_ROOT / "miccai2026"

# Two-colour scheme: architecture → colour, loss level → marker shape
NNUNET_COLOR = "#1f77b4"   # blue
CTFM_COLOR   = "#d62728"   # red

MODEL_STYLES = {
    "nnunet_baseline":      {"label": r"nnU-Net $\mathcal{C}_b$", "color": NNUNET_COLOR, "marker": "o"},
    "nnunet_l1_vesselness": {"label": r"nnU-Net $\mathcal{C}_v$", "color": NNUNET_COLOR, "marker": "s"},
    "nnunet_l2_cldice":     {"label": r"nnU-Net $\mathcal{C}_c$", "color": NNUNET_COLOR, "marker": "^"},
    "ctfm_l0_638":          {"label": r"CT-FM $\mathcal{C}_b$",   "color": CTFM_COLOR,   "marker": "o"},
    "ctfm_l1_638":          {"label": r"CT-FM $\mathcal{C}_v$",   "color": CTFM_COLOR,   "marker": "s"},
    "ctfm_l2_638":          {"label": r"CT-FM $\mathcal{C}_c$",   "color": CTFM_COLOR,   "marker": "^"},
}


def main():
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)

    # Load all LCC result files
    data = {}
    for model_key in MODEL_STYLES:
        path = RESULTS_DIR / f"lcc_results_{model_key}.json"
        if not path.exists():
            print(f"WARNING: {path} not found")
            continue
        with open(path) as f:
            raw = json.load(f)
        data[model_key] = raw[model_key]["raw"]  # use raw results

    # Standardise font: serif, larger labels
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "mathtext.fontset": "cm",
    })

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))  # roughly square

    all_dice, all_bcs = [], []
    discordant_cases = []

    for model_key, style in MODEL_STYLES.items():
        if model_key not in data:
            continue
        cases = data[model_key]
        dice_vals = [c["dice"] for c in cases]
        bcs_vals = [c["bcs"] for c in cases]

        ax.scatter(dice_vals, bcs_vals, label=style["label"],
                   color=style["color"], marker=style["marker"],
                   alpha=0.5, s=35, edgecolors="none")

        all_dice.extend(dice_vals)
        all_bcs.extend(bcs_vals)

        # Identify discordant cases
        for c in cases:
            if c["dice"] > 0.8 and c["bcs"] < 0.7:
                discordant_cases.append({
                    "model": model_key,
                    "case_id": c["case_id"],
                    "dice": c["dice"],
                    "bcs": c["bcs"],
                    "cldice": c.get("cldice", None),
                })

    # Pearson correlation
    r, p = pearsonr(all_dice, all_bcs)

    # Diagonal reference
    ax.plot([0.2, 1], [0.2, 1], "k--", alpha=0.3, linewidth=0.8)

    # Discordant region shading (data coordinates)
    ax.fill_between([0.8, 1.02], 0.2, 0.7, alpha=0.08, color="red")
    ax.text(0.91, 0.35, "Discordant\nregion", ha="center", va="center",
            fontsize=11, color="red", alpha=0.7)

    ax.set_xlabel("Dice Score")
    ax.set_ylabel("BCS (Bifurcation Connectivity Score)")
    ax.set_title(f"Dice vs Topology Preservation ($r$={r:.3f})")
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.2, 1.02)
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / "figures" / f"fig3_dice_vs_bcs.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    # Save discordant cases
    disc_path = OUTPUT_DIR / "results" / "discordant_cases.json"
    with open(disc_path, "w") as f:
        json.dump({
            "pearson_r": float(r),
            "pearson_p": float(p),
            "n_discordant": len(discordant_cases),
            "n_total_points": len(all_dice),
            "discordant_cases": sorted(discordant_cases, key=lambda x: x["bcs"]),
        }, f, indent=2)
    print(f"Saved: {disc_path}")
    print(f"\nPearson r(Dice, BCS) = {r:.4f} (p={p:.2e})")
    print(f"Discordant cases (Dice>0.8, BCS<0.7): {len(discordant_cases)}/{len(all_dice)}")


if __name__ == "__main__":
    main()

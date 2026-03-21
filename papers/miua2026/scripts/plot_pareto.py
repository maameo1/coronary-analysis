#!/usr/bin/env python3
"""
Figure 1: Pareto plot — Dice vs BCS for all adaptation strategies.

Shows the Dice-BCS trade-off across encoder adaptation levels and BCS weights.
Each adaptation strategy is a different color; BCS weight is encoded by marker shape.
Arrows connect the w=0.03 → w=0.05 → w=0.10 trajectory for frozen and partial rows.
L1 baseline shown as a star reference point.

Input:  miua2026/results/ablation/ablation_summary.json
Output: miua2026/figures/fig1_pareto.pdf
        miua2026/figures/fig1_pareto.png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "miua2026" / "results" / "ablation" / "ablation_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "miua2026" / "figures"

# Consistent style for LNCS
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "mathtext.fontset": "cm",
})

# Colorblind-safe palette (Okabe-Ito / Wong)
from colors import STRATEGY_COLORS as COLORS

# Marker per BCS weight
WEIGHT_MARKERS = {
    0.0:  "*",   # baseline
    0.03: "o",
    0.05: "s",
    0.10: "D",
}

# Experiment definitions: (key_in_json, strategy, weight, label)
EXPERIMENTS = [
    ("L1_baseline",          "baseline", 0.0,  r"$\mathcal{L}_v$ (L1 baseline)"),
    ("EXP-044_frozen_w003",  "frozen",   0.03, "Frozen, $w$=0.03"),
    ("EXP-048_frozen_w005",  "frozen",   0.05, "Frozen, $w$=0.05"),
    ("EXP-046_frozen_w010",  "frozen",   0.10, "Frozen, $w$=0.10"),
    ("EXP-049_partial_w003", "partial",  0.03, "Partial, $w$=0.03"),
    ("EXP-047_partial_w005", "partial",  0.05, "Partial, $w$=0.05"),
    ("EXP-050_partial_w010", "partial",  0.10, "Partial, $w$=0.10"),
    ("EXP-045_full_w010",    "full",     0.10, "Full, $w$=0.10"),
    ("EXP-052_bitfit_w005",  "bitfit",   0.05, "BitFit, $w$=0.05"),
    ("EXP-053_adapter_w005", "adapter",  0.05, "Adapter, $w$=0.05"),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_PATH) as f:
        data = json.load(f)
    experiments = data["experiments"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Plot each experiment
    plotted = {}  # strategy -> list of (dice, bcs, weight) for trajectory arrows
    for key, strategy, weight, label in EXPERIMENTS:
        exp = experiments.get(key)
        if exp is None or exp.get("metrics") is None:
            print(f"  Skipping {key} (no metrics)")
            continue

        m = exp["metrics"]
        dice = m["dice"]["mean"]
        bcs = m["bcs"]["mean"]

        color = COLORS[strategy]
        marker = WEIGHT_MARKERS.get(weight, "o")
        size = 120 if strategy == "baseline" else 80

        ax.scatter(dice, bcs, color=color, marker=marker, s=size,
                   edgecolors="white", linewidths=0.8, zorder=5, label=label)

        # Store for trajectory arrows
        if strategy not in plotted:
            plotted[strategy] = []
        plotted[strategy].append((dice, bcs, weight))

    # Draw trajectory arrows for frozen and partial (connecting w values)
    for strategy in ["frozen", "partial"]:
        if strategy not in plotted:
            continue
        pts = sorted(plotted[strategy], key=lambda x: x[2])  # sort by weight
        if len(pts) >= 2:
            color = COLORS[strategy]
            for i in range(len(pts) - 1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                ax.annotate("", xy=(pts[i+1][0], pts[i+1][1]),
                            xytext=(pts[i][0], pts[i][1]),
                            arrowprops=dict(arrowstyle="->", color=color,
                                            alpha=0.4, lw=1.5,
                                            connectionstyle="arc3,rad=0.05"))

    # Arrow annotation for increasing w direction
    ax.annotate(r"increasing $w$", xy=(0.787, 0.728), fontsize=8,
                color="grey", style="italic", ha="center")

    ax.set_xlabel("Dice Score")
    ax.set_ylabel("BCS")
    ax.set_xlim(0.74, 0.80)
    ax.set_ylim(0.68, 0.75)

    # Legend: group by strategy (one entry per strategy + weight markers)
    handles, labels = ax.get_legend_handles_labels()
    # Reorder: baseline first, then frozen, partial, full, bitfit, adapter
    order_map = {"baseline": 0, "frozen": 1, "partial": 2, "full": 3, "bitfit": 4, "adapter": 5}
    indexed = [(order_map.get(EXPERIMENTS[i][1], 99), i, h, l)
               for i, (h, l) in enumerate(zip(handles, labels))]
    indexed.sort(key=lambda x: x[0])
    ax.legend([x[2] for x in indexed], [x[3] for x in indexed],
              loc="lower left", framealpha=0.9, ncol=1)

    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"fig1_pareto.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    # Print summary
    print("\nPlotted experiments:")
    for key, strategy, weight, label in EXPERIMENTS:
        exp = experiments.get(key)
        if exp and exp.get("metrics"):
            m = exp["metrics"]
            print(f"  {label:30s}  Dice={m['dice']['mean']:.3f}  BCS={m['bcs']['mean']:.3f}")
        else:
            print(f"  {label:30s}  (pending)")


if __name__ == "__main__":
    main()

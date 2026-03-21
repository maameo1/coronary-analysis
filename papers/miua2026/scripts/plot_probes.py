#!/usr/bin/env python3
"""
Figure 3: Linear probe AUC — bifurcation detection across encoder and decoder levels.

Two subplots side by side:
  (a) Encoder probe: AUC at 5 encoder levels (level_0 to level_4)
  (b) Decoder probe: AUC at 4 decoder levels (dec_level_0 to dec_level_3)

Key findings:
  - Encoder: All models peak at level_1 (64ch, 48^3). Full (EXP-045) shifts peak to level_2.
  - Decoder: Full (EXP-045) shows elevated mid-decoder AUC — bifurcation info reorganised.
  - Frozen models nearly identical to L1 (encoder unchanged, as expected).

Input:  miua2026/results/linear_probe/linear_probe_results.json
        miua2026/results/decoder_probe/decoder_probe_results.json
Output: miua2026/figures/fig3_probes.pdf
        miua2026/figures/fig3_probes.png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENC_PATH = PROJECT_ROOT / "miua2026" / "results" / "linear_probe" / "linear_probe_results.json"
DEC_PATH = PROJECT_ROOT / "miua2026" / "results" / "decoder_probe" / "decoder_probe_results.json"
OUTPUT_DIR = PROJECT_ROOT / "miua2026" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "cm",
})

# Colorblind-safe palette (Okabe-Ito / Wong)
from colors import STRATEGY_COLORS

# Models to plot (key in JSON, display label, color, linestyle, marker)
MODELS = [
    ("L1 (vesselness)",           r"$\mathcal{L}_v$ (L1)",    STRATEGY_COLORS["baseline"], "--", "o"),
    ("EXP-044 (frozen, w=0.03)",  "Frozen ($w$=0.03)", STRATEGY_COLORS["frozen"],   "-",  "s"),
    ("EXP-048 (frozen, w=0.05)",  "Frozen ($w$=0.05)", "#005A8C",                   "-",  "D"),  # darker blue variant
    ("EXP-047 (partial, w=0.05)", "Partial ($w$=0.05)", STRATEGY_COLORS["partial"],  "-",  "^"),
    ("EXP-045 (full, w=0.10)",    "Full ($w$=0.10)",   STRATEGY_COLORS["full"],     "-",  "v"),
]

ENC_LEVELS = ["level_0", "level_1", "level_2", "level_3", "level_4"]
ENC_LABELS = ["L0\n32ch", "L1\n64ch", "L2\n128ch", "L3\n256ch", "L4\n512ch"]

DEC_LEVELS = ["dec_level_0", "dec_level_1", "dec_level_2", "dec_level_3"]
DEC_LABELS = ["D0\n256ch", "D1\n128ch", "D2\n64ch", "D3\n32ch"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ENC_PATH) as f:
        enc_data = json.load(f)
    with open(DEC_PATH) as f:
        dec_data = json.load(f)["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    # --- Encoder probe (left) ---
    for key, label, color, ls, marker in MODELS:
        if key not in enc_data:
            print(f"  Skipping encoder {key}")
            continue
        aucs = [enc_data[key][level]["auc"] for level in ENC_LEVELS]
        ax1.plot(range(len(ENC_LEVELS)), aucs, color=color, linestyle=ls,
                 marker=marker, markersize=6, linewidth=1.5, label=label)

    ax1.set_xticks(range(len(ENC_LEVELS)))
    ax1.set_xticklabels(ENC_LABELS)
    ax1.set_xlabel("Encoder Level")
    ax1.set_ylabel("Bifurcation Detection AUC")
    ax1.set_title("(a) Encoder Probe")
    ax1.set_ylim(0.58, 0.84)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right")

    # --- Decoder probe (right) ---
    for key, label, color, ls, marker in MODELS:
        if key not in dec_data:
            print(f"  Skipping decoder {key}")
            continue
        aucs = [dec_data[key][level]["auc"] for level in DEC_LEVELS]
        ax2.plot(range(len(DEC_LEVELS)), aucs, color=color, linestyle=ls,
                 marker=marker, markersize=6, linewidth=1.5, label=label)

    ax2.set_xticks(range(len(DEC_LEVELS)))
    ax2.set_xticklabels(DEC_LABELS)
    ax2.set_xlabel("Decoder Level")
    ax2.set_title("(b) Decoder Probe")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = OUTPUT_DIR / f"fig3_probes.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()

    # Print summary table
    print("\nEncoder Probe AUC:")
    for key, label, *_ in MODELS:
        if key in enc_data:
            vals = [f"{enc_data[key][l]['auc']:.3f}" for l in ENC_LEVELS]
            print(f"  {label:25s}  {' | '.join(vals)}")

    print("\nDecoder Probe AUC:")
    for key, label, *_ in MODELS:
        if key in dec_data:
            vals = [f"{dec_data[key][l]['auc']:.3f}" for l in DEC_LEVELS]
            print(f"  {label:25s}  {' | '.join(vals)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Generate a publication-quality Pearson correlation heatmap for five
segmentation-evaluation metrics pooled across 6 models x 250 test cases
(N = 1500 predictions).

Outputs
-------
- miccai2026/figures/metric_correlation_heatmap.pdf
- miccai2026/figures/metric_correlation_heatmap.png  (300 DPI)
- miccai2026/results/metric_correlations.json
"""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────
BASE = pathlib.Path("/path/to/project")
RESULTS_DIR = BASE / "test_results"
OUT_FIG = BASE / "miccai2026" / "figures"
OUT_RES = BASE / "miccai2026" / "results"

MODEL_KEYS = [
    "nnunet_baseline",
    "nnunet_l1_vesselness",
    "nnunet_l2_cldice",
    "ctfm_l0_638",
    "ctfm_l1_638",
    "ctfm_l2_638",
]

METRIC_FIELDS = ["dice", "cldice", "bcs", "betti_error", "hd95"]
METRIC_LABELS = ["Dice", "clDice", "BCS", r"Betti$_0$ Error", "HD95"]

# ── 1. Load & pool per-case metrics ───────────────────────────────────
rows = {m: [] for m in METRIC_FIELDS}

for mk in MODEL_KEYS:
    fpath = RESULTS_DIR / f"bcs_results_{mk}.json"
    with open(fpath) as f:
        data = json.load(f)
    per_case = data[mk]["per_case"]           # list of 250 dicts
    for case in per_case:
        for m in METRIC_FIELDS:
            rows[m].append(case[m])

# Convert to arrays
arrays = {m: np.array(rows[m], dtype=np.float64) for m in METRIC_FIELDS}
n_total = len(arrays["dice"])
print(f"Pooled N = {n_total}  ({len(MODEL_KEYS)} models x "
      f"{n_total // len(MODEL_KEYS)} cases)")

# ── 2. Pearson correlation matrix + p-values ──────────────────────────
n_metrics = len(METRIC_FIELDS)
corr_matrix = np.zeros((n_metrics, n_metrics))
pval_matrix = np.zeros((n_metrics, n_metrics))

for i in range(n_metrics):
    for j in range(n_metrics):
        r, p = stats.pearsonr(arrays[METRIC_FIELDS[i]],
                              arrays[METRIC_FIELDS[j]])
        corr_matrix[i, j] = r
        pval_matrix[i, j] = p

# ── 3. Save correlation data as JSON ─────────────────────────────────
corr_json = {
    "description": ("Pearson correlation matrix for 5 segmentation metrics, "
                     "pooled across 6 models x 250 test cases (N=1500)."),
    "metrics": METRIC_FIELDS,
    "metric_labels": ["Dice", "clDice", "BCS", "Betti0 Error", "HD95"],
    "n_predictions": int(n_total),
    "n_models": len(MODEL_KEYS),
    "model_keys": MODEL_KEYS,
    "correlation_matrix": corr_matrix.tolist(),
    "p_value_matrix": pval_matrix.tolist(),
}
json_path = OUT_RES / "metric_correlations.json"
with open(json_path, "w") as f:
    json.dump(corr_json, f, indent=2)
print(f"Saved  {json_path}")

# ── 4. Plot ───────────────────────────────────────────────────────────
# LNCS single-column width ~ 3.5 in; keep aspect square-ish
fig_w, fig_h = 4.2, 3.8

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "mathtext.fontset": "cm",
})

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# Mask upper triangle (keep diagonal + lower)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Draw heatmap
hm = sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.6,
    linecolor="white",
    xticklabels=METRIC_LABELS,
    yticklabels=METRIC_LABELS,
    cbar_kws={
        "label": "Pearson $r$",
        "shrink": 0.78,
        "aspect": 18,
        "pad": 0.04,
    },
    annot_kws={"size": 10.5, "weight": "bold"},
    ax=ax,
)

# ── 5. Highlight the BCS row & column ────────────────────────────────
bcs_idx = METRIC_FIELDS.index("bcs")  # = 2

# Highlight the BCS cells that are visible (lower triangle + diagonal)
highlight_cells = set()
for j in range(bcs_idx + 1):            # row - left of diagonal incl.
    highlight_cells.add((bcs_idx, j))
for i in range(bcs_idx, n_metrics):      # col - below diagonal incl.
    highlight_cells.add((i, bcs_idx))

for (r, c) in highlight_cells:
    if mask[r, c]:
        continue  # skip masked upper-triangle cells
    ax.add_patch(mpatches.Rectangle(
        (c, r), 1, 1,
        fill=False,
        edgecolor="#222222",
        linewidth=1.6,
        clip_on=False,
    ))

# Rotate x-tick labels for readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Bold the BCS tick labels
xtick_labels = ax.get_xticklabels()
ytick_labels = ax.get_yticklabels()
for lbl in list(xtick_labels) + list(ytick_labels):
    if "BCS" in lbl.get_text():
        lbl.set_weight("bold")

plt.tight_layout(pad=0.4)

# ── 6. Save ──────────────────────────────────────────────────────────
pdf_path = OUT_FIG / "metric_correlation_heatmap.pdf"
png_path = OUT_FIG / "metric_correlation_heatmap.png"
fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=0.05)
fig.savefig(str(png_path), bbox_inches="tight", pad_inches=0.05, dpi=300)
plt.close(fig)

print(f"Saved  {pdf_path}")
print(f"Saved  {png_path}")

# ── 7. Print summary ─────────────────────────────────────────────────
print("\n-- Correlation matrix (lower triangle) --")
header = f"{'':>16s}" + "".join(f"{l:>16s}" for l in METRIC_LABELS)
print(header)
for i in range(n_metrics):
    row_str = f"{METRIC_LABELS[i]:>16s}"
    for j in range(n_metrics):
        if j > i:
            row_str += "                "
        else:
            row_str += f"{corr_matrix[i, j]:16.3f}"
    print(row_str)

print("\nDone.")

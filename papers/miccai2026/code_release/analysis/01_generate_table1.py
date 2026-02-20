#!/usr/bin/env python3
"""
Generate Table 1: Main results table for MICCAI 2026 paper.
10 models x 5 metrics (Dice, clDice, BCS, Betti0 Error, HD95) with mean±std.

Models: nnU-Net L0-L3, CT-FM L0-L2, CT-FM L3a/L3b/L3c

Input:  test_results/lcc_results_<model>.json  (one per model, from apply_lcc_and_recompute.py)
Output: miccai2026/tables/table1_main_results.csv
        miccai2026/tables/table1_main_results.tex
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "test_results"
OUTPUT_DIR = PROJECT_ROOT / "miccai2026" / "tables"

# Display names and ordering
MODEL_ORDER = [
    ("nnunet_baseline", "nnU-Net L0"),
    ("nnunet_l1_vesselness", "nnU-Net L1"),
    ("nnunet_l2_cldice", "nnU-Net L2"),
    ("nnunet_l3_bcs", "nnU-Net L3"),
    ("ctfm_l0_638", "CT-FM L0"),
    ("ctfm_l1_638", "CT-FM L1"),
    ("ctfm_l2_638", "CT-FM L2"),
    ("ctfm_l3a", "CT-FM L3a"),
    ("ctfm_l3b", "CT-FM L3b"),
    ("ctfm_l3c", "CT-FM L3c"),
]

METRICS = [
    ("dice", "Dice", True),       # (key, display, higher_is_better)
    ("cldice", "clDice", True),
    ("bcs", "BCS", True),
    ("betti_error", "Betti$_0$ Err", False),
    ("hd95", "HD95 (mm)", False),
]


def load_lcc_results():
    """Load all LCC result files into unified dict with per_case arrays."""
    data = {}
    for model_key, _ in MODEL_ORDER:
        path = RESULTS_DIR / f"lcc_results_{model_key}.json"
        if not path.exists():
            print(f"WARNING: {path} not found")
            continue
        with open(path) as f:
            raw = json.load(f)
        # File structure: {model_key: {"raw": [...], "lcc": [...]}}
        model_data = raw[model_key]
        data[model_key] = model_data["raw"]  # use raw results (LCC destroys signal)
    return data


def format_mean_std(mean, std, metric_key, best=False):
    if metric_key in ("hd95", "betti_error"):
        s = f"{mean:.2f}±{std:.2f}"
    else:
        s = f"{mean:.3f}±{std:.3f}"
    if best:
        s = f"\\textbf{{{s}}}"
    return s


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_lcc_results()

    # Build table rows
    rows = []
    for dir_name, display_name in MODEL_ORDER:
        if dir_name not in data:
            continue
        cases = data[dir_name]
        row = {"Model": display_name, "N": len(cases)}
        for metric_key, _, _ in METRICS:
            vals = [c[metric_key] for c in cases if metric_key in c]
            row[f"{metric_key}_mean"] = float(np.mean(vals))
            row[f"{metric_key}_std"] = float(np.std(vals))
        rows.append(row)

    # Find best per metric
    best_vals = {}
    for metric_key, _, higher in METRICS:
        vals = [r[f"{metric_key}_mean"] for r in rows]
        if higher:
            best_vals[metric_key] = max(vals)
        else:
            best_vals[metric_key] = min(vals)

    # CSV output
    csv_rows = []
    for row in rows:
        csv_row = {"Model": row["Model"], "N": row["N"]}
        for metric_key, display, _ in METRICS:
            csv_row[display] = f"{row[f'{metric_key}_mean']:.4f}±{row[f'{metric_key}_std']:.4f}"
        csv_rows.append(csv_row)

    df = pd.DataFrame(csv_rows)
    csv_path = OUTPUT_DIR / "table1_main_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX output
    tex_lines = []
    tex_lines.append("\\begin{table}[t]")
    tex_lines.append("\\centering")
    tex_lines.append("\\caption{Quantitative results on ImageCAS test set (N=250). "
                      "Best results per metric in \\textbf{bold}.}")
    tex_lines.append("\\label{tab:main_results}")
    tex_lines.append("\\begin{tabular}{l" + "c" * len(METRICS) + "}")
    tex_lines.append("\\toprule")

    header = "Model & " + " & ".join(d for _, d, _ in METRICS) + " \\\\"
    tex_lines.append(header)
    tex_lines.append("\\midrule")

    for i, row in enumerate(rows):
        if i == 3:  # separator between nnU-Net and CT-FM
            tex_lines.append("\\midrule")
        cells = [row["Model"]]
        for metric_key, _, _ in METRICS:
            is_best = abs(row[f"{metric_key}_mean"] - best_vals[metric_key]) < 1e-6
            cells.append(format_mean_std(
                row[f"{metric_key}_mean"], row[f"{metric_key}_std"],
                metric_key, best=is_best
            ))
        tex_lines.append(" & ".join(cells) + " \\\\")

    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")

    tex_path = OUTPUT_DIR / "table1_main_results.tex"
    tex_path.write_text("\n".join(tex_lines))
    print(f"Saved: {tex_path}")

    # Print summary to console
    print(f"\n{'='*70}")
    print("TABLE 1: Main Results (ImageCAS Test Set)")
    print(f"{'='*70}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

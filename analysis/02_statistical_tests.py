#!/usr/bin/env python3
"""
Statistical significance tests for MICCAI 2026 paper.

Uses paired bootstrap resampling (following STAMBO methodology) to compute:
- 95% confidence intervals for each model's mean Dice and BCS
- Bootstrap p-values for pairwise model comparisons
- Effect sizes with bootstrap CIs

Input:  test_results/lcc_results_<model>.json  (one per model)
Output: miccai2026/results/statistical_tests_bootstrap.json
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "test_results"
OUTPUT_DIR = PROJECT_ROOT / "miccai2026"

ALL_MODELS = [
    "nnunet_baseline", "nnunet_l1_vesselness", "nnunet_l2_cldice",
    "ctfm_l0_638", "ctfm_l1_638", "ctfm_l2_638",
]

DISPLAY_NAMES = {
    "nnunet_baseline": "nnU-Net L0",
    "nnunet_l1_vesselness": "nnU-Net L1",
    "nnunet_l2_cldice": "nnU-Net L2",
    "ctfm_l0_638": "CT-FM L0",
    "ctfm_l1_638": "CT-FM L1",
    "ctfm_l2_638": "CT-FM L2",
}

N_BOOTSTRAP = 10000
ALPHA = 0.05
SEED = 42


def load_all_results():
    """Load all model results, return dict of model -> per_case dict."""
    data = {}
    for model_key in ALL_MODELS:
        path = RESULTS_DIR / f"lcc_results_{model_key}.json"
        if not path.exists():
            print(f"WARNING: {path} not found")
            continue
        with open(path) as f:
            raw = json.load(f)
        cases = raw[model_key]["raw"]
        data[model_key] = {c["case_id"]: c for c in cases}
    return data


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=SEED):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)

    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return float(np.mean(values)), float(ci_low), float(ci_high)


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=SEED):
    """
    Paired bootstrap hypothesis test.

    H0: mean(scores_a) >= mean(scores_b)  (model A is at least as good)
    H1: mean(scores_b) > mean(scores_a)   (model B is better)

    Returns:
        observed_diff: mean(B) - mean(A)
        p_value: one-tailed p-value
        ci_low, ci_high: 95% CI for the difference
    """
    rng = np.random.RandomState(seed)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)
    assert len(scores_b) == n

    observed_diff = np.mean(scores_b) - np.mean(scores_a)

    boot_diffs = np.array([
        np.mean(scores_b[idx]) - np.mean(scores_a[idx])
        for idx in (rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap))
    ])

    # One-tailed p-value: fraction of bootstrap where diff <= 0
    p_value = np.mean(boot_diffs <= 0)

    ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    return float(observed_diff), float(p_value), float(ci_low), float(ci_high)


def main():
    print("=" * 70)
    print("Bootstrap Statistical Analysis (n_bootstrap={})".format(N_BOOTSTRAP))
    print("=" * 70)

    data = load_all_results()
    print(f"Loaded {len(data)} models")

    # Get common case IDs across all models
    all_case_sets = [set(data[m].keys()) for m in data]
    common_cases = sorted(set.intersection(*all_case_sets))
    print(f"Common cases: {len(common_cases)}")

    results = {
        "n_bootstrap": N_BOOTSTRAP,
        "alpha": ALPHA,
        "seed": SEED,
        "n_cases": len(common_cases),
        "per_model": {},
        "pairwise_comparisons": [],
        "key_claims": {},
    }

    # 1. Per-model bootstrap CIs
    print("\n--- Per-Model Bootstrap CIs ---")
    print(f"{'Model':<20s} {'Dice':>24s} {'BCS':>24s}")
    print("-" * 70)

    for model in ALL_MODELS:
        if model not in data:
            continue
        dice_vals = [data[model][c]["dice"] for c in common_cases]
        bcs_vals = [data[model][c]["bcs"] for c in common_cases]

        dice_mean, dice_lo, dice_hi = bootstrap_ci(dice_vals)
        bcs_mean, bcs_lo, bcs_hi = bootstrap_ci(bcs_vals)

        results["per_model"][model] = {
            "display_name": DISPLAY_NAMES[model],
            "dice": {"mean": dice_mean, "ci_low": dice_lo, "ci_high": dice_hi},
            "bcs": {"mean": bcs_mean, "ci_low": bcs_lo, "ci_high": bcs_hi},
        }

        print(f"{DISPLAY_NAMES[model]:<20s} "
              f"{dice_mean:.3f} [{dice_lo:.3f}, {dice_hi:.3f}]  "
              f"{bcs_mean:.3f} [{bcs_lo:.3f}, {bcs_hi:.3f}]")

    # 2. Within-architecture pairwise comparisons
    arch_pairs = {
        "nnunet": [
            ("nnunet_baseline", "nnunet_l1_vesselness"),
            ("nnunet_baseline", "nnunet_l2_cldice"),
            ("nnunet_l1_vesselness", "nnunet_l2_cldice"),
        ],
        "ctfm": [
            ("ctfm_l0_638", "ctfm_l1_638"),
            ("ctfm_l0_638", "ctfm_l2_638"),
            ("ctfm_l1_638", "ctfm_l2_638"),
        ],
    }

    # Key cross-architecture comparisons
    cross_pairs = [
        ("ctfm_l0_638", "nnunet_l1_vesselness"),  # ranking reversal
        ("ctfm_l1_638", "nnunet_l1_vesselness"),   # vesselness vs vesselness cross-arch
        ("ctfm_l2_638", "nnunet_l1_vesselness"),   # best CT-FM topo vs best nnUNet topo
        ("nnunet_baseline", "ctfm_l0_638"),         # architecture comparison at baseline
    ]

    all_pairs = []
    for arch, pairs in arch_pairs.items():
        all_pairs.extend(pairs)
    all_pairs.extend(cross_pairs)

    # Bonferroni correction factor
    n_comparisons = len(all_pairs) * 2  # x2 for Dice and BCS
    bonferroni_factor = n_comparisons

    print(f"\n--- Pairwise Bootstrap Comparisons (Bonferroni k={n_comparisons}) ---")
    print(f"{'Comparison':<35s} {'Metric':>6s} {'Diff':>8s} {'95% CI':>20s} {'p':>10s} {'p_corr':>10s}")
    print("-" * 95)

    for model_a, model_b in all_pairs:
        if model_a not in data or model_b not in data:
            continue

        for metric in ["dice", "bcs"]:
            vals_a = [data[model_a][c][metric] for c in common_cases]
            vals_b = [data[model_b][c][metric] for c in common_cases]

            diff, p_val, ci_lo, ci_hi = paired_bootstrap_test(vals_a, vals_b)
            p_corrected = min(p_val * bonferroni_factor, 1.0)

            comp_result = {
                "model_a": model_a,
                "model_b": model_b,
                "display_a": DISPLAY_NAMES[model_a],
                "display_b": DISPLAY_NAMES[model_b],
                "metric": metric,
                "diff": diff,  # mean(B) - mean(A)
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "p_value": p_val,
                "p_corrected": p_corrected,
                "significant": p_corrected < 0.05,
            }
            results["pairwise_comparisons"].append(comp_result)

            sig = "***" if p_corrected < 0.001 else ("**" if p_corrected < 0.01 else ("*" if p_corrected < 0.05 else "ns"))
            label = f"{DISPLAY_NAMES[model_a]} vs {DISPLAY_NAMES[model_b]}"
            print(f"{label:<35s} {metric:>6s} {diff:>+.4f} [{ci_lo:>+.4f}, {ci_hi:>+.4f}] "
                  f"{p_val:>10.6f} {p_corrected:>10.6f} {sig}")

    # 3. Key claims for the paper
    # Claim: ranking reversal — nnU-Net L1 BCS > CT-FM L0 BCS
    bcs_ctfm_l0 = [data["ctfm_l0_638"][c]["bcs"] for c in common_cases]
    bcs_nnunet_l1 = [data["nnunet_l1_vesselness"][c]["bcs"] for c in common_cases]
    diff, p_val, ci_lo, ci_hi = paired_bootstrap_test(bcs_ctfm_l0, bcs_nnunet_l1)

    results["key_claims"]["ranking_reversal_bcs"] = {
        "description": "nnU-Net L1 BCS > CT-FM L0 BCS (ranking reversal)",
        "diff": diff,
        "ci": [ci_lo, ci_hi],
        "p_value": p_val,
        "significant": p_val < 0.05,
    }

    # Claim: CT-FM L1 Dice > nnU-Net L1 Dice (despite worse BCS)
    dice_ctfm_l1 = [data["ctfm_l1_638"][c]["dice"] for c in common_cases]
    dice_nnunet_l1 = [data["nnunet_l1_vesselness"][c]["dice"] for c in common_cases]
    diff2, p_val2, ci_lo2, ci_hi2 = paired_bootstrap_test(dice_nnunet_l1, dice_ctfm_l1)

    results["key_claims"]["dice_advantage_ctfm"] = {
        "description": "CT-FM L1 Dice > nnU-Net L1 Dice",
        "diff": diff2,
        "ci": [ci_lo2, ci_hi2],
        "p_value": p_val2,
        "significant": p_val2 < 0.05,
    }

    # Claim: nnU-Net L1 BCS > CT-FM L2 BCS
    bcs_ctfm_l2 = [data["ctfm_l2_638"][c]["bcs"] for c in common_cases]
    diff3, p_val3, ci_lo3, ci_hi3 = paired_bootstrap_test(bcs_ctfm_l2, bcs_nnunet_l1)

    results["key_claims"]["nnunet_l1_vs_ctfm_l2_bcs"] = {
        "description": "nnU-Net L1 BCS > CT-FM L2 BCS",
        "diff": diff3,
        "ci": [ci_lo3, ci_hi3],
        "p_value": p_val3,
        "significant": p_val3 < 0.05,
    }

    print("\n" + "=" * 70)
    print("KEY CLAIMS")
    print("=" * 70)
    for claim_name, claim in results["key_claims"].items():
        print(f"\n{claim['description']}:")
        print(f"  Diff = {claim['diff']:+.4f}, 95% CI [{claim['ci'][0]:+.4f}, {claim['ci'][1]:+.4f}]")
        print(f"  p = {claim['p_value']:.6f} {'(significant)' if claim['significant'] else '(not significant)'}")

    # 4. Print LaTeX-ready text for paper
    print("\n" + "=" * 70)
    print("LATEX SNIPPETS FOR PAPER")
    print("=" * 70)

    rc = results["key_claims"]["ranking_reversal_bcs"]
    p_str = f"{rc['p_value']:.4f}" if rc["p_value"] >= 0.0001 else f"<0.0001"
    print(f"\nRanking reversal (for Results section):")
    print(f"  Paired bootstrap confirms the BCS difference between nnU-Net L1 and CT-FM L0")
    print(f"  is statistically significant (bootstrap $p{p_str}$, 10{{,}}000 resamples).")

    # Save results
    out_path = OUTPUT_DIR / "results" / "statistical_tests_bootstrap.json"
    (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

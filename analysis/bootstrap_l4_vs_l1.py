#!/usr/bin/env python3
"""
Paired bootstrap test: CT-FM L4 (EXP-044) vs CT-FM L1.

Uses the same methodology as miccai2026/scripts/02_statistical_tests.py:
- Paired bootstrap resampling (10,000 resamples)
- 95% confidence intervals for mean differences
- One-tailed p-values

Usage:
    python scripts/bootstrap_l4_vs_l1.py [--l4_results PATH]

If --l4_results is not given, defaults to EXP-044 test metrics.
"""

import argparse
import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default paths
L1_PATH = PROJECT_ROOT / "test_results" / "lcc_results_ctfm_l1_638.json"
L4_DEFAULT = (
    PROJECT_ROOT
    / "experiments"
    / "EXP-044_ctfm_l4_finetune_from_l1_20260214_124914"
    / "test_predictions_configB"
    / "l4_test_metrics.json"
)

N_BOOTSTRAP = 10000
ALPHA = 0.05
SEED = 42


def load_l1_results(path):
    """Load L1 results from lcc_results format."""
    with open(path) as f:
        raw = json.load(f)
    cases = raw["ctfm_l1_638"]["raw"]
    return {c["case_id"]: c for c in cases}


def load_l4_results(path):
    """Load L4 results from EXP-044 metrics format."""
    with open(path) as f:
        raw = json.load(f)
    cases = raw["per_case"]
    # Normalize case_id: "case00751" -> 751
    out = {}
    for c in cases:
        cid = c["case_id"]
        if isinstance(cid, str):
            cid = int(cid.replace("case", ""))
        out[cid] = c
    return out


def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Paired bootstrap hypothesis test.
    H0: mean(B) <= mean(A)
    H1: mean(B) > mean(A)
    Returns: observed_diff, p_value, ci_low, ci_high
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    n = len(a)
    assert len(b) == n

    observed_diff = np.mean(b) - np.mean(a)

    boot_diffs = np.array([
        np.mean(b[idx]) - np.mean(a[idx])
        for idx in (rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap))
    ])

    # One-tailed p-value: fraction where diff <= 0
    p_value = np.mean(boot_diffs <= 0)

    ci_low = np.percentile(boot_diffs, 100 * ALPHA / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - ALPHA / 2))

    return float(observed_diff), float(p_value), float(ci_low), float(ci_high)


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Bootstrap 95% CI for the mean."""
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)
    boot_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    return (
        float(np.mean(values)),
        float(np.percentile(boot_means, 100 * ALPHA / 2)),
        float(np.percentile(boot_means, 100 * (1 - ALPHA / 2))),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l4_results", type=str, default=str(L4_DEFAULT))
    args = parser.parse_args()

    print("=" * 70)
    print(f"Paired Bootstrap: CT-FM L4 vs L1 (n_bootstrap={N_BOOTSTRAP})")
    print("=" * 70)

    # Load data
    l1_data = load_l1_results(L1_PATH)
    l4_data = load_l4_results(args.l4_results)

    # Find common cases
    common = sorted(set(l1_data.keys()) & set(l4_data.keys()))
    print(f"\nL1 cases: {len(l1_data)}, L4 cases: {len(l4_data)}, Common: {len(common)}")

    if len(common) == 0:
        print("ERROR: No common cases found!")
        return

    # All metrics to test: (key, display_name, higher_is_better)
    METRICS = [
        ("dice", "Dice", True),
        ("cldice", "clDice", True),
        ("bcs", "BCS", True),
        ("hd95", "HD95", False),
        ("betti_error", "Betti Err", False),
    ]

    # Extract paired values for all metrics
    paired = {}
    for key, _, _ in METRICS:
        paired[key] = (
            [l1_data[c][key] for c in common],
            [l4_data[c][key] for c in common],
        )

    # Per-model summaries
    header = f"{'Model':<12s}"
    for _, name, _ in METRICS:
        header += f" {name:>24s}"
    print(f"\n{header}")
    print("-" * (12 + 25 * len(METRICS)))
    for label, src in [("CT-FM L1", 0), ("CT-FM L4", 1)]:
        row = f"{label:<12s}"
        for key, _, _ in METRICS:
            vals = paired[key][src]
            m, lo, hi = bootstrap_ci(vals)
            row += f" {m:.4f} [{lo:.4f}, {hi:.4f}]"
        print(row)

    # Paired bootstrap tests
    print(f"\n{'='*70}")
    print("PAIRED BOOTSTRAP TESTS")
    print(f"{'='*70}")

    results = {}
    for key, display_name, higher_better in METRICS:
        vals_l1, vals_l4 = paired[key]
        if higher_better:
            # H1: L4 > L1 (improvement = positive diff)
            diff, p_val, ci_lo, ci_hi = paired_bootstrap_test(vals_l1, vals_l4)
        else:
            # H1: L4 < L1 (improvement = negative diff, test reversed)
            diff, p_val, ci_lo, ci_hi = paired_bootstrap_test(vals_l4, vals_l1)
            # Report as L4 - L1 (negative = improvement)
            diff = -diff
            ci_lo, ci_hi = -ci_hi, -ci_lo

        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))

        results[key] = {
            "diff": diff,
            "p_value": p_val,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "significant": p_val < 0.05,
            "higher_is_better": higher_better,
        }

        direction = "↑" if higher_better else "↓"
        print(f"\n  {display_name} ({direction}): L4 - L1 = {diff:+.4f}")
        print(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        p_str = f"{p_val:.6f}" if p_val > 0 else "<0.0001"
        print(f"  p = {p_str} {sig}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<12s} {'Diff':>10s} {'95% CI':>24s} {'p-value':>12s} {'Sig':>5s}")
    print(f"  {'-'*65}")
    for key, display_name, higher_better in METRICS:
        r = results[key]
        direction = "↑" if higher_better else "↓"
        p_str = f"{r['p_value']:.4f}" if r['p_value'] > 0 else "<0.0001"
        sig = "***" if r['p_value'] < 0.001 else ("**" if r['p_value'] < 0.01 else ("*" if r['p_value'] < 0.05 else "ns"))
        print(f"  {display_name+' ('+direction+')':<12s} {r['diff']:>+10.4f} [{r['ci_low']:>+.4f}, {r['ci_high']:>+.4f}] {p_str:>12s} {sig:>5s}")

    # Save results
    l1_summary = {}
    l4_summary = {}
    for key, _, _ in METRICS:
        vals_l1, vals_l4 = paired[key]
        l1_summary[f"{key}_mean"] = float(np.mean(vals_l1))
        l1_summary[f"{key}_std"] = float(np.std(vals_l1))
        l4_summary[f"{key}_mean"] = float(np.mean(vals_l4))
        l4_summary[f"{key}_std"] = float(np.std(vals_l4))

    out = {
        "n_bootstrap": N_BOOTSTRAP,
        "n_cases": len(common),
        "l1_path": str(L1_PATH),
        "l4_path": args.l4_results,
        "l1_summary": l1_summary,
        "l4_summary": l4_summary,
        "tests": results,
    }

    out_path = PROJECT_ROOT / "miccai2026" / "results" / "bootstrap_l4_vs_l1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

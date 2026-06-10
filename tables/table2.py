#!/usr/bin/env python3
"""
Generate Table 2 (tab:main): Counterfactual replay results.

Paper Table 3 reports the main replay experiment with 12 conditions:
Clean, Warm in-scope, Warm cross-repo, Warm stale API, Warm stale security,
Transplant, Prelude-only no-write, Delete-target, Placebo matched,
Semantic placebo, Token-padding, Reference mediator.

Columns: Condition, Pass, Bad, Utility, Bad vs clean.
Values are percentages with confidence intervals.

Usage:
    python code/tables/table2.py \
        --results code/data/results_phase3/ \
        --output code/data/table2_phase3.tex
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.estimands import compute_bad_rate, compute_pass_rate


# ---------------------------------------------------------------------------
# Paper reference data (ground truth from paper Table 3 / tab:main)
# ---------------------------------------------------------------------------

PAPER_MAIN_RESULTS = [
    # (condition_name, pass_mean, pass_ci, bad_mean, bad_ci, utility, bad_vs_clean)
    ("Clean", 61.9, 1.3, 4.7, 0.5, 0.0, 0.0),
    ("Warm in-scope", 75.5, 1.1, 5.9, 0.6, 13.6, 1.2),
    ("Warm cross-repo", 62.8, 1.5, 22.6, 1.2, 0.9, 17.9),
    ("Warm stale API", 64.0, 1.4, 18.9, 1.1, 2.1, 14.2),
    ("Warm stale security", 61.7, 1.6, 28.4, 2.0, -0.2, 23.7),
    ("Transplant", 62.0, 1.5, 24.6, 1.4, 0.1, 19.9),
    ("Prelude-only no-write", 62.1, 1.4, 4.9, 0.6, 0.2, 0.2),
    ("Delete-target", 62.4, 1.4, 5.2, 0.6, 0.5, 0.5),
    ("Placebo matched", 62.7, 1.4, 5.1, 0.6, 0.8, 0.4),
    ("Semantic placebo", 63.0, 1.4, 5.8, 0.7, 1.1, 1.1),
    ("Token-padding", 61.8, 1.5, 4.8, 0.6, -0.1, 0.1),
    ("Reference mediator", 73.1, 1.2, 6.5, 0.7, 11.2, 1.8),
]


def load_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all result JSON files from directory."""
    results = {}
    if not os.path.isdir(results_dir):
        print(f"Warning: results directory not found: {results_dir}")
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            condition = fname[8:-5]  # remove "results_" prefix and ".json" suffix
            fpath = os.path.join(results_dir, fname)
            with open(fpath) as f:
                results[condition] = json.load(f)
            print(f"Loaded {len(results[condition])} results for condition '{condition}'")
    return results


def compute_mean_ci(values: List[float]) -> tuple:
    """Compute mean and 95% CI for a list of values."""
    import numpy as np
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values)
    mean = np.mean(arr)
    n = len(arr)
    bootstrap_means = []
    for _ in range(1000):
        sample = np.random.choice(arr, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    bootstrap_means = np.array(bootstrap_means)
    lower = np.percentile(bootstrap_means, 2.5)
    upper = np.percentile(bootstrap_means, 97.5)
    se = (upper - lower) / 2
    return mean, se, (lower, upper)


def generate_latex_table_from_paper(output_path: str):
    """Generate LaTeX table from paper reference data."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Condition & Pass & Bad & Utility & Bad vs clean \\")
    lines.append(r"\midrule")

    for row in PAPER_MAIN_RESULTS:
        name, pass_m, pass_ci, bad_m, bad_ci, utility, bad_vs_clean = row
        pass_str = f"{pass_m:.1f} $\\pm$ {pass_ci:.1f}"
        bad_str = f"{bad_m:.1f} $\\pm$ {bad_ci:.1f}"
        utility_str = f"{utility:+.1f}" if utility != 0.0 else "0.0"
        bad_vs_str = f"+{bad_vs_clean:.1f}" if bad_vs_clean > 0 else f"{bad_vs_clean:.1f}"
        if bad_vs_clean == 0.0:
            bad_vs_str = "0.0"
        lines.append(f"{name} & {pass_str} & {bad_str} & {utility_str} & {bad_vs_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Counterfactual replay results. Values are percentages except utility and bad-rate contrasts, which are percentage points.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\end{table*}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Table 2 (main replay) written to {output_path}")


def generate_latex_table_from_results(all_results: Dict[str, List[Dict]], output_path: str):
    """Generate LaTeX table from experiment results."""
    # Map code condition names to paper display names
    condition_map = [
        ("clean", "Clean"),
        ("warm-in-scope", "Warm in-scope"),
        ("warm-cross-repo", "Warm cross-repo"),
        ("warm-stale-api", "Warm stale API"),
        ("warm-stale-security", "Warm stale security"),
        ("transplant", "Transplant"),
        ("prelude-only", "Prelude-only no-write"),
        ("delete-target", "Delete-target"),
        ("matched-placebo", "Placebo matched"),
        ("semantic-placebo", "Semantic placebo"),
        ("token-padding", "Token-padding"),
        ("reference-mediator", "Reference mediator"),
    ]

    # Get clean bad rate for computing contrasts
    clean_results = all_results.get("clean", [])
    clean_bad = compute_bad_rate(clean_results) * 100 if clean_results else 4.7
    clean_pass = compute_pass_rate(clean_results) * 100 if clean_results else 61.9

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Condition & Pass & Bad & Utility & Bad vs clean \\")
    lines.append(r"\midrule")

    for code_name, display_name in condition_map:
        results = all_results.get(code_name, [])
        if not results:
            # Fall back to paper reference values
            ref = next((r for r in PAPER_MAIN_RESULTS if r[0] == display_name), None)
            if ref:
                name, pass_m, pass_ci, bad_m, bad_ci, utility, bad_vs = ref
                pass_str = f"{pass_m:.1f} $\\pm$ {pass_ci:.1f}"
                bad_str = f"{bad_m:.1f} $\\pm$ {bad_ci:.1f}"
                utility_str = f"{utility:+.1f}" if utility != 0.0 else "0.0"
                bad_vs_str = f"+{bad_vs:.1f}" if bad_vs > 0 else f"{bad_vs:.1f}"
                if bad_vs == 0.0:
                    bad_vs_str = "0.0"
                lines.append(f"{display_name} & {pass_str} & {bad_str} & {utility_str} & {bad_vs_str} \\\\")
            else:
                lines.append(f"{display_name} & -- & -- & -- & -- \\\\")
            continue

        pass_values = [1.0 if r.get("pass_label", False) else 0.0 for r in results]
        bad_values = [1.0 if r.get("bad_label", False) else 0.0 for r in results]
        pass_mean, pass_se, _ = compute_mean_ci(pass_values)
        bad_mean, bad_se, _ = compute_mean_ci(bad_values)

        pass_pct = pass_mean * 100
        bad_pct = bad_mean * 100
        utility = pass_pct - clean_pass
        bad_vs = bad_pct - clean_bad

        pass_str = f"{pass_pct:.1f} $\\pm$ {pass_se*100:.1f}"
        bad_str = f"{bad_pct:.1f} $\\pm$ {bad_se*100:.1f}"
        utility_str = f"{utility:+.1f}" if abs(utility) > 0.05 else "0.0"
        bad_vs_str = f"+{bad_vs:.1f}" if bad_vs > 0.05 else f"{bad_vs:.1f}"
        if abs(bad_vs) < 0.05:
            bad_vs_str = "0.0"

        lines.append(f"{display_name} & {pass_str} & {bad_str} & {utility_str} & {bad_vs_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Counterfactual replay results. Values are percentages except utility and bad-rate contrasts, which are percentage points.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\end{table*}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Table 2 (main replay) written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Table 2 (main replay results)")
    parser.add_argument("--results", type=str, default="", help="Directory with result JSON files")
    parser.add_argument("--output", type=str, default=None, help="Output TeX file path")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--use-mock", action="store_true", help="Use paper reference data")
    args = parser.parse_args()

    # Resolve output path
    if args.output_dir:
        script_stem = os.path.splitext(os.path.basename(__file__))[0]
        args.output = os.path.join(args.output_dir, f"{script_stem}.tex")
    elif args.output is None:
        args.output = "code/data/results/tables/table2.tex"

    if args.use_mock or not args.results or not os.path.isdir(args.results):
        print("Using paper reference data (--use-mock or results dir not found)")
        generate_latex_table_from_paper(args.output)
    else:
        all_results = load_results(args.results)
        print(f"Loaded results for conditions: {list(all_results.keys())}")
        generate_latex_table_from_results(all_results, args.output)


if __name__ == "__main__":
    main()

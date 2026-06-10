#!/usr/bin/env python3
"""
Generate Figure 6: Difficulty-residual anatomy.
Shows quadrant chart of difficulty vs residual for different sequence types.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.estimands import compute_bad_rate


def generate_figure6(all_results: dict, output_path: str):
    """Generate Figure 6: Difficulty-residual anatomy."""
    # Use warm results for analysis
    results_warm = all_results.get("warm", [])
    if not results_warm:
        print("No warm results found, skipping Figure 6")
        return

    # Group by memory_type
    groups = {}
    for r in results_warm:
        mt = r.get("memory_type", "unknown")
        if mt not in groups:
            groups[mt] = []
        groups[mt].append(r)

    # Compute difficulty (bad rate) and residual (some metric) for each group
    labels = []
    difficulties = []
    residuals = []

    for mt, results in groups.items():
        if not results:
            continue
        difficulty = compute_bad_rate(results) * 100
        # Residual: simplified as (pass_rate - avg_pass_rate)
        pass_rate = sum(1 for r in results if r.get("pass_label", False)) / len(results) * 100
        residual = pass_rate - 50  # simplified
        labels.append(mt)
        difficulties.append(difficulty)
        residuals.append(residual)

    # Create quadrant chart
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot scatter
    ax.scatter(difficulties, residuals, s=100, alpha=0.7)
    # Annotate points
    for i, label in enumerate(labels):
        ax.annotate(label, (difficulties[i], residuals[i]), fontsize=8)

    # Add quadrant lines (median-based)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=np.median(difficulties), color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Difficulty (bad rate %)")
    ax.set_ylabel("Residual (pass rate - 50%)")
    ax.set_title("Difficulty-Residual Anatomy")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Figure 6 saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Load results
    all_results = {}
    for fname in sorted(os.listdir(args.results)):
        if fname.startswith("results_") and fname.endswith(".json"):
            condition = fname[8:-5]
            fpath = os.path.join(args.results, fname)
            with open(fpath) as f:
                all_results[condition] = json.load(f)
            print(f"Loaded {len(all_results[condition])} results for condition '{condition}'")

    generate_figure6(all_results, args.output)


if __name__ == "__main__":
    main()

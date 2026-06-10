#!/usr/bin/env python3
"""
Generate Figure 4: Utility-harm frontier.
Shows trade-off between utility (pass rate) and harm (bad rate) for different conditions.
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
from core.estimands import compute_bad_rate, compute_pass_rate


def generate_figure4(all_results: dict, output_path: str):
    """Generate Figure 4: Utility-harm frontier."""
    conditions = ["clean", "warm", "delete-target", "matched-placebo", "reference-mediator"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Collect data points
    x_vals = []  # harm (bad rate)
    y_vals = []  # utility (pass rate)
    labels = []

    for c in conditions:
        results = all_results.get(c, [])
        if not results:
            continue
        harm = compute_bad_rate(results) * 100
        utility = compute_pass_rate(results) * 100
        x_vals.append(harm)
        y_vals.append(utility)
        labels.append(c)

    # Plot scatter
    ax.scatter(x_vals, y_vals, s=100, alpha=0.7)
    # Annotate points
    for i, label in enumerate(labels):
        ax.annotate(label, (x_vals[i], y_vals[i]), fontsize=8)

    ax.set_xlabel("Harm (bad rate %)")
    ax.set_ylabel("Utility (pass rate %)")
    ax.set_title("Utility-Harm Frontier")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Figure 4 saved to {output_path}")


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

    generate_figure4(all_results, args.output)


if __name__ == "__main__":
    main()

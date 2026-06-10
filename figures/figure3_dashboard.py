#!/usr/bin/env python3
"""
Generate Figure 3: Main replay evidence dashboard.
Shows utility, harm, controls, dose response, diagnostic slices.
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


def generate_figure3(all_results: dict, output_path: str):
    """Generate Figure 3 dashboard."""
    # Create 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Main Replay Evidence Dashboard", fontsize=14)

    # Subplot 1: Utility (pass rate by condition)
    ax = axes[0, 0]
    conditions = ["clean", "warm", "delete-target", "matched-placebo", "reference-mediator"]
    pass_rates = [compute_pass_rate(all_results.get(c, [])) * 100 for c in conditions]
    ax.bar(conditions, pass_rates)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Utility")
    ax.tick_params(axis="x", rotation=45)

    # Subplot 2: Harm (bad rate by condition)
    ax = axes[0, 1]
    bad_rates = [compute_bad_rate(all_results.get(c, [])) * 100 for c in conditions]
    ax.bar(conditions, bad_rates, color="red")
    ax.set_ylabel("Bad rate (%)")
    ax.set_title("Harm")
    ax.tick_params(axis="x", rotation=45)

    # Subplot 3: Controls (token-padding, prelude-only, semantic-placebo)
    ax = axes[0, 2]
    control_conditions = ["clean", "token-padding", "prelude-only", "semantic-placebo"]
    control_bad = [compute_bad_rate(all_results.get(c, [])) * 100 for c in control_conditions]
    ax.bar(control_conditions, control_bad, color="orange")
    ax.set_ylabel("Bad rate (%)")
    ax.set_title("Controls")
    ax.tick_params(axis="x", rotation=45)

    # Subplot 4: Dose response
    ax = axes[1, 0]
    doses = [0, 1, 2, 4]
    dose_bad = []
    for d in doses:
        key = f"dose-response-{d}" if d > 0 else "clean"
        dose_bad.append(compute_bad_rate(all_results.get(key, [])) * 100)
    ax.plot(doses, dose_bad, marker="o")
    ax.set_ylabel("Bad rate (%)")
    ax.set_xlabel("Number of invalid memories")
    ax.set_title("Dose Response")

    # Subplot 5: Diagnostic slices (by memory_type)
    ax = axes[1, 1]
    slices = ["cross-repo", "stale-api", "stale-security", "hidden-channel"]
    slice_bad = []
    for s in slices:
        results = [r for r in all_results.get("warm", []) if r.get("memory_type") == s]
        slice_bad.append(compute_bad_rate(results) * 100 if results else 0)
    ax.bar(slices, slice_bad, color="purple")
    ax.set_ylabel("Bad rate (%)")
    ax.set_title("Diagnostic Slices")
    ax.tick_params(axis="x", rotation=45)

    # Subplot 6: Rank/Position shuffle
    ax = axes[1, 2]
    shuffle_conditions = ["warm", "rank-shuffle", "position-shuffle"]
    shuffle_bad = [compute_bad_rate(all_results.get(c, [])) * 100 for c in shuffle_conditions]
    ax.bar(shuffle_conditions, shuffle_bad, color="brown")
    ax.set_ylabel("Bad rate (%)")
    ax.set_title("Shuffle Tests")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Figure 3 saved to {output_path}")


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

    generate_figure3(all_results, args.output)


if __name__ == "__main__":
    main()

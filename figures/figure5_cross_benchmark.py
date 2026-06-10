#!/usr/bin/env python3
"""
Generate Figure 5: Cross-benchmark replication.

Shows bad rates across different benchmarks (GitHub, SWE-bench, ReAct)
under clean and warm conditions, with 95% confidence intervals.

This is Phase 3 Part 4 of the "Memory Is a Hidden Dependency" replication project.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.estimands import compute_bad_rate
from core.bootstrap import bootstrap_ci


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARKS = ["GitHub", "SWE-bench", "ReAct"]
CONDITIONS = ["clean", "warm"]
COLORS = {"clean": "#4C78A8", "warm": "#E15759"}
FIG_SIZE = (8, 5)
FONT_SIZE = 11


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results_from_dir(results_dir: str) -> dict:
    """
    Load all result JSON files from a directory.

    Expected file naming: results_{condition}_{benchmark}.json
    e.g., results_clean_github.json, results_warm_swe-bench.json

    Returns:
        dict with keys like ("clean", "GitHub") mapping to list of result dicts.
    """
    assert os.path.isdir(results_dir), f"Results directory does not exist: {results_dir}"

    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not (fname.startswith("results_") and fname.endswith(".json")):
            continue
        fpath = os.path.join(results_dir, fname)

        # Parse condition and benchmark from filename
        # Expected: results_{condition}_{benchmark}.json
        name = fname[8:-5]  # strip "results_" prefix and ".json" suffix
        parts = name.split("_", 1)
        assert len(parts) == 2, (
            f"Cannot parse condition and benchmark from filename: {fname!r}. "
            f"Expected format: results_{{condition}}_{{benchmark}}.json"
        )
        condition, benchmark = parts

        assert condition in CONDITIONS, (
            f"Unknown condition {condition!r} in file {fname!r}. Valid: {CONDITIONS}"
        )

        with open(fpath) as f:
            data = json.load(f)

        assert isinstance(data, list), f"Expected list in {fname!r}, got {type(data)}"
        assert len(data) > 0, f"Empty results in {fname!r}"

        results[(condition, benchmark)] = data
        print(f"  Loaded {len(data)} results from {fname} (condition={condition}, benchmark={benchmark})")

    assert len(results) > 0, f"No result files found in {results_dir}"
    return results


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------

def generate_mock_data(output_dir: str):
    """
    Generate mock cross-benchmark results for testing.

    Creates results_{condition}_{benchmark}.json files with simulated bad rates:
    - GitHub: clean ~2%, warm ~18%
    - SWE-bench: clean ~3%, warm ~22%
    - ReAct: clean ~1%, warm ~15%
    """
    os.makedirs(output_dir, exist_ok=True)

    # Mock parameters: (clean_bad_rate, warm_bad_rate) for each benchmark
    mock_params = {
        "GitHub": (0.02, 0.18),
        "SWE-bench": (0.03, 0.22),
        "ReAct": (0.01, 0.15),
    }

    np.random.seed(42)  # Reproducible mock data

    for benchmark, (clean_rate, warm_rate) in mock_params.items():
        for condition, true_rate in [("clean", clean_rate), ("warm", warm_rate)]:
            # Generate 200 mock samples
            n_samples = 200
            # Binomial samples with some noise
            labels = np.random.binomial(1, true_rate, size=n_samples).tolist()

            results = []
            for i, bad_label in enumerate(labels):
                results.append({
                    "sequence_id": f"{benchmark}_{condition}_seq_{i}",
                    "bad_label": bool(bad_label),
                    "pass_label": np.random.random() > 0.5,
                    "benchmark": benchmark,
                    "condition": condition,
                })

            fname = f"results_{condition}_{benchmark}.json"
            fpath = os.path.join(output_dir, fname)
            with open(fpath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Mock: wrote {fpath} ({len(results)} samples, true_rate={true_rate:.2%})")

    print(f"\nMock data written to {output_dir}")


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_bad_rate_with_ci(results: list, confidence: float = 0.95) -> tuple:
    """
    Compute bad rate and confidence interval using bootstrap.

    Args:
        results: List of result dicts with 'bad_label' field.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        (bad_rate, ci_lower, ci_upper)
    """
    assert len(results) > 0, "Cannot compute bad rate from empty results"
    assert all("bad_label" in r for r in results), "Results missing 'bad_label' field"

    scores = [float(r["bad_label"]) for r in results]
    bad_rate = sum(scores) / len(scores)

    # Bootstrap CI
    alpha = 1 - confidence
    _, ci_lower, ci_upper = bootstrap_ci(
        scores,
        n_resamples=10000,
        alpha=alpha,
    )

    return bad_rate, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_figure5(results: dict, output_path: str):
    """
    Generate Figure 5: Cross-benchmark replication.

    Creates a grouped bar chart showing bad rates for each benchmark
    under clean and warm conditions, with 95% confidence intervals.

    Args:
        results: Dict mapping (condition, benchmark) to list of result dicts.
        output_path: Path to save the PDF file.
    """
    assert len(results) > 0, "No results provided to generate_figure5"

    # Prepare data for plotting
    benchmarks = BENCHMARKS
    x = np.arange(len(benchmarks))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Collect data for each condition
    clean_rates = []
    clean_cis = []
    warm_rates = []
    warm_cis = []

    for benchmark in benchmarks:
        # Clean condition
        clean_key = ("clean", benchmark)
        assert clean_key in results, f"Missing clean results for benchmark {benchmark!r}"
        clean_data = results[clean_key]
        clean_rate, clean_lower, clean_upper = compute_bad_rate_with_ci(clean_data)
        clean_rates.append(clean_rate * 100)
        clean_cis.append((clean_rate * 100 - clean_lower * 100, clean_upper * 100 - clean_rate * 100))

        # Warm condition
        warm_key = ("warm", benchmark)
        assert warm_key in results, f"Missing warm results for benchmark {benchmark!r}"
        warm_data = results[warm_key]
        warm_rate, warm_lower, warm_upper = compute_bad_rate_with_ci(warm_data)
        warm_rates.append(warm_rate * 100)
        warm_cis.append((warm_rate * 100 - warm_lower * 100, warm_upper * 100 - warm_rate * 100))

    # Plot bars
    bars1 = ax.bar(
        x - width/2,
        clean_rates,
        width,
        label="Clean",
        color=COLORS["clean"],
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width/2,
        warm_rates,
        width,
        label="Warm",
        color=COLORS["warm"],
        edgecolor="black",
        linewidth=0.5,
    )

    # Add error bars (95% CI)
    ax.errorbar(
        x - width/2,
        clean_rates,
        yerr=np.array(clean_cis).T,
        fmt="none",
        color="black",
        capsize=5,
        capthick=1,
        elinewidth=1,
    )
    ax.errorbar(
        x + width/2,
        warm_rates,
        yerr=np.array(warm_cis).T,
        fmt="none",
        color="black",
        capsize=5,
        capthick=1,
        elinewidth=1,
    )

    # Annotate bars with values
    for bars, rates in [(bars1, clean_rates), (bars2, warm_rates)]:
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.annotate(
                f"{rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE - 1,
            )

    # Formatting
    ax.set_xlabel("Benchmark", fontsize=FONT_SIZE)
    ax.set_ylabel("Bad rate (%)", fontsize=FONT_SIZE)
    ax.set_title("Cross-benchmark Replication: Bad Rate by Benchmark and Condition", fontsize=FONT_SIZE + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=FONT_SIZE - 1)
    ax.legend(fontsize=FONT_SIZE - 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)

    # Tight layout and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure 5 saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Figure 5: Cross-benchmark replication")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results/",
        help="Directory containing result JSON files (default: data/results/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/figures/figure5.pdf",
        help="Output PDF file path (default: data/results/figures/figure5.pdf)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Generate and use mock data instead of reading from disk",
    )
    args = parser.parse_args()

    # Generate mock data if requested
    if args.use_mock:
        print("Generating mock data...")
        mock_dir = os.path.join(args.results_dir, "_mock")
        generate_mock_data(mock_dir)
        results_dir = mock_dir
    else:
        results_dir = args.results_dir

    # Load results
    print(f"Loading results from {results_dir}...")
    results = load_results_from_dir(results_dir)

    # Verify we have data for all benchmark-condition combinations
    for condition in CONDITIONS:
        for benchmark in BENCHMARKS:
            key = (condition, benchmark)
            assert key in results, (
                f"Missing results for condition={condition!r}, benchmark={benchmark!r}. "
                f"Expected file: results_{condition}_{benchmark}.json"
            )

    # Generate figure
    print("Generating Figure 5...")
    generate_figure5(results, args.output)


if __name__ == "__main__":
    main()

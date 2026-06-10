#!/usr/bin/env python3
"""
Generate minimized Table 2 in LaTeX format.
Follows paper Table 2 format strictly.
"""

import sys
import os

# Add parent directory to path so 'core' can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from typing import Dict, List, Any


def load_results(results_dir: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load results from JSON files, grouped by condition and memory_type.

    Returns:
        Nested dict: results[condition][memory_type] = [list of result dicts]
    """
    assert os.path.isdir(results_dir), f"Results directory not found: {results_dir}"

    results = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("results_") and fname.endswith(".json"):
            condition = fname.replace("results_", "").replace(".json", "")
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)

            # Group by memory_type
            results[condition] = {}
            for r in data:
                mt = r.get("memory_type", "unknown")
                if mt not in results[condition]:
                    results[condition][mt] = []
                results[condition][mt].append(r)

            # Print summary
            for mt, mdata in results[condition].items():
                print(f"Loaded {len(mdata)} records for condition={condition}, memory_type={mt}")

    return results


def compute_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute pass rate, bad rate, and 95% CI."""
    from core.estimands import compute_pass_rate, compute_bad_rate
    from core.bootstrap import bootstrap_ci

    pass_rate = compute_pass_rate(results)
    bad_rate = compute_bad_rate(results)

    # Bootstrap CI for bad rate
    bad_scores = [float(r.get("bad_label", 0)) for r in results]
    _, ci_lower, ci_upper = bootstrap_ci(bad_scores)

    # Compute utility (pass and not bad)
    utility = sum(
        1 for r in results if r.get("pass_label", False) and not r.get("bad_label", False)
    ) / len(results)

    return {
        "pass_rate": pass_rate,
        "bad_rate": bad_rate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "utility": utility,
    }


def format_value(mean: float, lower: float, upper: float) -> str:
    """Format value as mean +- CI."""
    ci_width = (upper - lower) / 2
    return f"{mean * 100:.1f} $\\pm$ {ci_width * 100:.1f}"


def generate_latex(results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> str:
    """Generate LaTeX table with warm condition grouped by memory_type."""
    # Compute stats for each condition + memory_type combination
    stats = {}
    for condition, mem_groups in results.items():
        stats[condition] = {}
        for memory_type, data in mem_groups.items():
            stats[condition][memory_type] = compute_stats(data)

    # Build table rows
    rows = []

    # Clean
    if "clean" in stats:
        s = stats["clean"]
        # Clean has no memory_type grouping (or has a single group)
        for mt, s_mt in s.items():
            rows.append(
                f"Clean & {s_mt['pass_rate']*100:.1f} & {s_mt['bad_rate']*100:.1f} & {s_mt['utility']*100:.1f} & -- \\\\"
            )
            break  # Only show first group for clean

    # Warm: display each memory_type as separate row
    if "warm" in stats:
        for memory_type, s in stats["warm"].items():
            rows.append(
                f"Warm ({memory_type}) & {s['pass_rate']*100:.1f} & {s['bad_rate']*100:.1f} & {s['utility']*100:.1f} & $\\Delta$ \\\\"
            )

    # Delete-target
    if "delete-target" in stats:
        s = stats["delete-target"]
        for mt, s_mt in s.items():
            rows.append(
                f"Delete-target & {s_mt['pass_rate']*100:.1f} & {s_mt['bad_rate']*100:.1f} & {s_mt['utility']*100:.1f} & Recovery \\\\"
            )
            break  # Only show first group for delete-target

    # Build full table
    latex = """\\begin{table}[t]
\\centering\\small\\setlength{\\tabcolsep}{4pt}
\\begin{tabular}{lcccc}
\\toprule
Condition & Pass & Bad & Utility & Bad vs clean \\\\
\\cmidrule(lr){2-5}
"""
    latex += "\n".join(rows)
    latex += """
\\bottomrule
\\end{tabular}
\\caption{MemTrace-Bench v5 Phase 1 results. Bad rate is the percentage of runs with bad outcomes. Utility is the percentage of runs that pass and are not bad.}
\\label{tab:table2_minimal}
\\end{table}
"""
    return latex


def generate_mock_results() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Generate mock results for testing table generation.

    Returns mock data with realistic structure matching real experiment results.
    """
    import random
    random.seed(42)
    conditions = ["clean", "warm", "delete-target"]
    memory_types = ["short_term", "long_term", "episodic"]

    mock_results = {}
    for condition in conditions:
        mock_results[condition] = {}
        for mt in memory_types:
            results = []
            for _ in range(50):
                results.append({
                    "pass_label": random.random() < 0.8,
                    "bad_label": random.random() < 0.1,
                    "memory_type": mt,
                })
            mock_results[condition][mt] = results
    return mock_results


def main():
    parser = argparse.ArgumentParser(description="Generate Table 2 (minimal)")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output LaTeX file path (alternative to --output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files (alternative to --output)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of reading from files",
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output_dir:
        script_stem = os.path.splitext(os.path.basename(__file__))[0]
        args.output = os.path.join(args.output_dir, f"{script_stem}.tex")
    elif args.output is None:
        raise AssertionError("Must specify either --output or --output-dir")

    # Load results or use mock
    if args.use_mock:
        print("Using mock data (--use-mock set)")
        results = generate_mock_results()
    else:
        # Validate
        assert os.path.isdir(args.results), f"Directory not found: {args.results}"
        # Load results
        results = load_results(args.results)

    # Generate LaTeX
    latex = generate_latex(results)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX table to {args.output}")

    # Print to stdout
    print("\n=== LaTeX Output ===")
    print(latex)


if __name__ == "__main__":
    main()

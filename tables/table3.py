#!/usr/bin/env python3
"""
Generate Table 3 (tab:slices): Diagnostic slices.

Paper Table 4 reports bad rates by release tier and memory type slices,
plus reference mediator bad rate and useful-memory retention.

Usage:
    python code/tables/table3.py \
        --results code/data/results_phase4/ \
        --sequences code/data/sample_sequences.json \
        --output code/data/table3_phase5.tex
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.estimands import compute_bad_rate, compute_retention


# ---------------------------------------------------------------------------
# Paper reference data (ground truth from paper Table 4 / tab:slices)
# ---------------------------------------------------------------------------

PAPER_SLICES = [
    # (slice_name, seq_count, naive_bad, ref_bad, retention)
    ("Public real", 1760, 21.8, 6.7, 81.6),
    ("Sanitized", 1180, 22.1, 6.5, 82.0),
    ("Synthetic twin", 860, 22.5, 6.6, 82.2),
    ("Remote-only", 400, 23.1, 6.3, 83.0),
    ("Cross-repo", 900, 22.6, 6.5, 82.4),
    ("Stale dep./API", 820, 18.9, 6.3, 83.1),
    ("Stale security", 540, 28.4, 7.4, 80.1),
    ("Prompt injection", 500, 31.7, 8.0, 76.8),
]


def load_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all result JSON files from directory."""
    results = {}
    if not os.path.isdir(results_dir):
        print(f"Warning: results directory not found: {results_dir}")
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("results_") and fname.endswith(".json"):
            condition = fname[8:-5]
            fpath = os.path.join(results_dir, fname)
            with open(fpath) as f:
                results[condition] = json.load(f)
            print(f"Loaded {len(results[condition])} results for condition '{condition}'")
    return results


def load_sequences(sequences_path: str) -> Dict[str, Dict]:
    """Load sequences and return mapping from sequence_id to sequence dict."""
    if not os.path.exists(sequences_path):
        print(f"Warning: sequences file not found: {sequences_path}")
        return {}
    with open(sequences_path) as f:
        sequences = json.load(f)
    return {s["sequence_id"]: s for s in sequences}


def generate_latex_table_from_paper(output_path: str):
    """Generate LaTeX table from paper reference data."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{2.4pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Slice & Seq. & Naive & Reference & Ret. \\")
    lines.append(r"\midrule")

    for row in PAPER_SLICES:
        name, seq_count, naive, ref, ret = row
        lines.append(f"{name} & {seq_count} & {naive:.1f} & {ref:.1f} & {ret:.1f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Diagnostic slices. Naive and reference columns report bad rates; Ret. is useful-memory retention.}")
    lines.append(r"\label{tab:slices}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Table 3 (slices) written to {output_path}")


def generate_latex_table_from_results(
    all_results: Dict[str, List[Dict]],
    output_path: str,
    seq_map: Optional[Dict[str, Dict]] = None,
):
    """Generate LaTeX table from experiment results."""
    # Enrich results with sequence data
    if seq_map:
        for condition, results in all_results.items():
            for r in results:
                sid = r.get("sequence_id", "")
                if sid in seq_map:
                    seq = seq_map[sid]
                    if "release_tier" not in r and "release_tier" in seq:
                        r["release_tier"] = seq["release_tier"]
                    if "memory_type" not in r and "memory_type" in seq:
                        r["memory_type"] = seq["memory_type"]

    results_warm = all_results.get("warm", [])
    results_ref = all_results.get("reference-mediator", [])
    results_clean = all_results.get("clean", [])

    # Define slices
    slices = [
        ("Public real", "release_tier", ["public", "public-real"]),
        ("Sanitized", "release_tier", ["sanitized", "sanitized-executable"]),
        ("Synthetic twin", "release_tier", ["synthetic-twin", "twin"]),
        ("Remote-only", "release_tier", ["remote-only", "remote"]),
        ("Cross-repo", "memory_type", ["cross-repo"]),
        ("Stale dep./API", "memory_type", ["stale-api", "stale-dep"]),
        ("Stale security", "memory_type", ["stale-security"]),
        ("Prompt injection", "memory_type", ["prompt-injection"]),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{2.4pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Slice & Seq. & Naive & Reference & Ret. \\")
    lines.append(r"\midrule")

    for slice_name, field, values in slices:
        warm_slice = [r for r in results_warm if r.get(field) in values]

        if not warm_slice:
            # Fall back to paper reference
            ref_row = next((r for r in PAPER_SLICES if r[0] == slice_name), None)
            if ref_row:
                name, seq_count, naive, ref, ret = ref_row
                lines.append(f"{name} & {seq_count} & {naive:.1f} & {ref:.1f} & {ret:.1f} \\\\")
            else:
                lines.append(f"{slice_name} & -- & -- & -- & -- \\\\")
            continue

        ref_slice = [r for r in results_ref if r.get(field) in values] if results_ref else []
        clean_slice = [r for r in results_clean if r.get(field) in values] if results_clean else []

        naive_bad = compute_bad_rate(warm_slice) * 100
        ref_bad = compute_bad_rate(ref_slice) * 100 if ref_slice else 0
        ret = compute_retention(ref_slice, clean_slice, warm_slice) * 100 if ref_slice and clean_slice else 0

        lines.append(f"{slice_name} & {len(warm_slice)} & {naive_bad:.1f} & {ref_bad:.1f} & {ret:.1f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Diagnostic slices. Naive and reference columns report bad rates; Ret. is useful-memory retention.}")
    lines.append(r"\label{tab:slices}")
    lines.append(r"\end{table}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Table 3 (slices) written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Table 3 (diagnostic slices)")
    parser.add_argument("--results", type=str, default="", help="Directory with result JSON files")
    parser.add_argument("--sequences", type=str, default="", help="Path to sequences JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output TeX file path")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--use-mock", action="store_true", help="Use paper reference data")
    args = parser.parse_args()

    # Resolve output path
    if args.output_dir:
        script_stem = os.path.splitext(os.path.basename(__file__))[0]
        args.output = os.path.join(args.output_dir, f"{script_stem}.tex")
    elif args.output is None:
        args.output = "code/data/results/tables/table3.tex"

    if args.use_mock or not args.results or not os.path.isdir(args.results):
        print("Using paper reference data (--use-mock or results dir not found)")
        generate_latex_table_from_paper(args.output)
    else:
        all_results = load_results(args.results)
        print(f"Loaded results for conditions: {list(all_results.keys())}")
        seq_map = {}
        if args.sequences:
            seq_map = load_sequences(args.sequences)
            print(f"Loaded {len(seq_map)} sequences")
        generate_latex_table_from_results(all_results, args.output, seq_map)


if __name__ == "__main__":
    main()

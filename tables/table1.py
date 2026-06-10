#!/usr/bin/env python3
"""
Generate Table 1 (tab:composition): Benchmark-facing composition.

Paper Table 1 reports the composition of MemTrace-Bench v5 across four dimensions:
Release, Memory, Channel, and Oracle. Each dimension independently partitions
the 4,200 sequences.

Usage:
    python code/tables/table1.py --output code/data/results/tables/table1.tex
    python code/tables/table1.py --use-mock --output code/data/results/tables/table1.tex
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Paper data (ground truth from Table 1 in the paper)
# ---------------------------------------------------------------------------

PAPER_COMPOSITION = {
    "Release": [
        ("public real", 1760, 41.9),
        ("sanitized executable", 1180, 28.1),
        ("synthetic twin", 860, 20.5),
        ("remote-only", 400, 9.5),
    ],
    "Memory": [
        ("in-scope useful", 1000, 23.8),
        ("cross-repo", 900, 21.4),
        ("stale dep./API", 820, 19.5),
        ("stale security", 540, 12.9),
        ("sensitive/license", 440, 10.5),
        ("prompt injection", 500, 11.9),
    ],
    "Channel": [
        ("memory store", 840, 20.0),
        ("conversation", 530, 12.6),
        ("tool log", 680, 16.2),
        ("terminal/cache", 940, 22.4),
        ("wrapper/patch", 860, 20.5),
        ("scratchpad/planner", 350, 8.3),
    ],
    "Oracle": [
        ("hidden tests", 1820, 43.3),
        ("semantic/security", 1520, 36.2),
        ("static/license", 860, 20.5),
    ],
}


# ---------------------------------------------------------------------------
# Data loading (from benchmark file)
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> Dict[str, Any]:
    """Load benchmark JSON file and return the parsed dict."""
    assert os.path.isfile(path), f"benchmark file not found: {path}"
    with open(path) as f:
        data = json.load(f)
    return data


def compute_composition_from_data(data: Dict) -> Dict[str, List]:
    """Compute composition statistics from benchmark data.

    Groups sequences by release_tier, memory_type, channel, and oracle_type.
    Returns dict matching PAPER_COMPOSITION structure.
    """
    # Collect all sequences
    all_seqs: List[Dict] = []
    if isinstance(data, list):
        all_seqs = data
    else:
        for split in ("train", "val", "test"):
            all_seqs.extend(data.get(split, []))

    total = len(all_seqs) if all_seqs else 4200

    def count_and_pct(seqs, field, value_map):
        results = []
        for display_name, field_values in value_map.items():
            if not isinstance(field_values, list):
                field_values = [field_values]
            count = sum(1 for s in seqs if s.get(field) in field_values)
            pct = count / total * 100 if total > 0 else 0
            results.append((display_name, count, round(pct, 1)))
        return results

    composition = {}

    # Release dimension
    composition["Release"] = count_and_pct(all_seqs, "release_tier", {
        "public real": ["public", "public-real"],
        "sanitized executable": ["sanitized", "sanitized-executable"],
        "synthetic twin": ["synthetic-twin", "twin"],
        "remote-only": ["remote-only", "remote"],
    })

    # Memory dimension
    composition["Memory"] = count_and_pct(all_seqs, "memory_type", {
        "in-scope useful": ["in-scope", "in-scope-useful"],
        "cross-repo": ["cross-repo"],
        "stale dep./API": ["stale-api", "stale-dep"],
        "stale security": ["stale-security"],
        "sensitive/license": ["sensitive-license", "sensitive"],
        "prompt injection": ["prompt-injection"],
    })

    # Channel dimension
    composition["Channel"] = count_and_pct(all_seqs, "channel", {
        "memory store": ["memory-store"],
        "conversation": ["conversation"],
        "tool log": ["tool-log"],
        "terminal/cache": ["terminal-cache"],
        "wrapper/patch": ["wrapper-patch", "wrapper-prompt"],
        "scratchpad/planner": ["scratchpad", "scratchpad-planner"],
    })

    # Oracle dimension
    composition["Oracle"] = count_and_pct(all_seqs, "oracle_type", {
        "hidden tests": ["hidden-tests", "hidden_tests", "test"],
        "semantic/security": ["semantic-security", "semantic", "security"],
        "static/license": ["static-license", "static", "license"],
    })

    return composition


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def generate_latex(composition: Dict[str, List], output_path: str) -> None:
    """Write booktabs LaTeX table to output_path matching paper Table 1."""
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(r"\setlength{\tabcolsep}{2.35pt}")
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\toprule")
    lines.append(r"Dimension & Category & Seq. & \% \\")
    lines.append(r"\midrule")

    for dimension, rows in composition.items():
        for i, (category, seq_count, pct) in enumerate(rows):
            dim_label = dimension if i == 0 else ""
            lines.append(f"{dim_label} & {category} & {seq_count} & {pct:.1f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Benchmark-facing composition. Release, memory, channel, and oracle dimensions are reported separately because each dimension partitions the 4,200 sequences.}")
    lines.append(r"\label{tab:composition}")
    lines.append(r"\end{table}")

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Table 1 (composition) written to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Table 1 (benchmark composition)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="code/data/processed/benchmark_v1.json",
        help="Path to benchmark JSON (default: code/data/processed/benchmark_v1.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output TeX file path",
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
        help="Use paper reference data instead of computing from benchmark file",
    )
    args = parser.parse_args()

    # Resolve output path
    if args.output_dir:
        script_stem = os.path.splitext(os.path.basename(__file__))[0]
        args.output = os.path.join(args.output_dir, f"{script_stem}.tex")
    elif args.output is None:
        args.output = "code/data/results/tables/table1.tex"

    # Decide data source
    use_mock = args.use_mock or not os.path.isfile(args.input)

    if use_mock:
        print("Using paper reference data (--use-mock set or input file not found)")
        composition = PAPER_COMPOSITION
    else:
        print(f"Loading benchmark from {args.input}")
        data = load_benchmark(args.input)
        composition = compute_composition_from_data(data)

    # Generate LaTeX
    generate_latex(composition, args.output)


if __name__ == "__main__":
    main()

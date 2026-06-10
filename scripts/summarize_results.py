#!/usr/bin/env python3
"""
Summarize all experimental results for the paper
"Memory Is a Hidden Dependency: A Benchmark for Replay-Defined Harm in Stateful Coding Agents"
reproduction work.

This script is the first part of Phase 7 (Result Summarization).
It reads all experimental results from multiple result directories,
computes summary statistics, and generates a summary report.

The summary report includes:
1. Per-phase result statistics
2. Per-condition bad rate summary
3. Per-baseline-system performance summary
4. Overall conclusions

Usage:
    python summarize_results.py [--results-dirs DIRS] [--output PATH] [--use-mock]

Example:
    python summarize_results.py
    python summarize_results.py --results-dirs data/results,data/results_real_world
    python summarize_results.py --use-mock
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# __file__ = .../code/scripts/summarize_results.py
# Project root is 2 levels up from this file.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Default result directories to scan.
# These correspond to different phases of the reproduction work.
DEFAULT_RESULTS_DIRS = [
    "data/results",
    "data/results_phase3",
    "data/results_phase4",
    "data/results_phase5",
    "data/results_phase6",
    "data/results_phase7_minimal",
    "data/results_phase8_naive_vector",
    "data/results_real_world",
]

# Default output path for the summary report.
DEFAULT_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "results", "summary_report.json"
)

# Expected fields in each result record.
# Every result JSON file is expected to be a list of dicts with these fields.
# We assert their presence to fail fast on malformed data.
REQUIRED_RESULT_FIELDS = ["condition", "bad_label", "pass_label"]

# Conditions that represent "warm" (memory-augmented) runs.
# Used to compute warm-vs-clean bad rate comparisons.
WARM_CONDITIONS = ["warm", "memgpt", "mem0", "zep", "a-mem", "reflexion", "workflow"]

# Conditions that represent "clean" (no memory) baseline runs.
CLEAN_CONDITIONS = ["clean", "delete-target"]


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def load_json_file(file_path: str) -> Any:
    """Load a JSON file and return its contents.

    Crash immediately if the file does not exist or contains invalid JSON.
    This is a Fast-Fail check: we must not silently proceed with empty or
    corrupt data, as that would produce a meaningless summary report.

    Args:
        file_path: Absolute or relative path to the JSON file.

    Returns:
        Parsed JSON content (typically a list of dicts).

    Raises:
        AssertionError: if file is missing or JSON is invalid.
    """
    assert os.path.isfile(file_path), f"Result file does not exist: {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise AssertionError(
                f"Invalid JSON in {file_path}: "
                f"{e.msg} (line {e.lineno}, col {e.colno})"
            ) from e
    return data


def discover_result_files(results_dir: str) -> List[str]:
    """Discover JSON result files in a results directory.

    Recursively walks the directory tree and collects JSON files whose names
    start with ``results_`` or ``results.json``. This naming convention
    avoids picking up summary reports or other non-result JSON files.

    Args:
        results_dir: Path to the results directory.

    Returns:
        List of absolute paths to JSON files found.

    Raises:
        AssertionError: if results_dir does not exist (Fast-Fail).
    """
    results_dir = os.path.abspath(results_dir)
    assert os.path.isdir(results_dir), f"Results directory does not exist: {results_dir}"

    json_files = []
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            # Only include files whose name suggests they contain result records
            # (a list of dicts), not summary reports (a dict).
            # Expected patterns: results_*.json, results.json
            if fname.endswith(".json") and (
                fname.startswith("results_") or fname == "results.json"
            ):
                json_files.append(os.path.join(root, fname))
    return json_files


def load_all_results(results_dirs: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load all results from the given directories.

    Returns a dict mapping phase_name -> list of result records.
    Each result record is a dict with fields like condition, bad_label, etc.

    The phase_name is derived from the directory name (e.g., "results_phase3"
    becomes "phase3"; "results_real_world" becomes "real_world"; "results"
    becomes "main").

    Args:
        results_dirs: List of directory paths to scan.

    Returns:
        Dict mapping phase name to list of result records.

    Raises:
        AssertionError: if any result file is malformed or missing required fields.
    """
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for dir_path in results_dirs:
        dir_path = os.path.abspath(dir_path)
        if not os.path.isdir(dir_path):
            # Skip non-existent directories gracefully; they may not have been
            # generated yet. Print a warning so the user knows.
            print(f"[WARNING] Results directory does not exist, skipping: {dir_path}")
            continue

        # Derive phase name from directory name.
        dir_name = os.path.basename(dir_path)
        # Map "results" -> "main", "results_phase3" -> "phase3", etc.
        if dir_name == "results":
            phase_name = "main"
        elif dir_name.startswith("results_"):
            phase_name = dir_name[len("results_"):]
        else:
            phase_name = dir_name

        json_files = discover_result_files(dir_path)
        print(f"[INFO] Phase '{phase_name}': found {len(json_files)} JSON files in {dir_path}")

        phase_records: List[Dict[str, Any]] = []
        for jf in json_files:
            data = load_json_file(jf)
            # Expect a list of result dicts.
            assert isinstance(data, list), (
                f"Expected list in {jf}, got {type(data).__name__}"
            )
            for i, record in enumerate(data):
                assert isinstance(record, dict), (
                    f"Record {i} in {jf} is not a dict: {type(record).__name__}"
                )
                # Validate required fields are present.
                for field in REQUIRED_RESULT_FIELDS:
                    assert field in record, (
                        f"Record {i} in {jf} missing required field '{field}'. "
                        f"Available fields: {sorted(record.keys())}"
                    )
                phase_records.append(record)

        all_results[phase_name] = phase_records
        print(f"[INFO] Phase '{phase_name}': loaded {len(phase_records)} records")

    return all_results


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def compute_bad_rate(records: List[Dict[str, Any]]) -> float:
    """Compute the bad rate for a list of result records.

    Bad rate = fraction of records where bad_label is True.

    Args:
        records: List of result dicts, each with a 'bad_label' field.

    Returns:
        Bad rate as a float in [0, 1]. Returns 0.0 if records is empty
        (caller should validate empty input separately).
    """
    if not records:
        return 0.0
    bad_count = sum(1 for r in records if r.get("bad_label", False))
    return bad_count / len(records)


def compute_pass_rate(records: List[Dict[str, Any]]) -> float:
    """Compute the pass rate for a list of result records.

    Pass rate = fraction of records where pass_label is True.

    Args:
        records: List of result dicts, each with a 'pass_label' field.

    Returns:
        Pass rate as a float in [0, 1].
    """
    if not records:
        return 0.0
    pass_count = sum(1 for r in records if r.get("pass_label", False))
    return pass_count / len(records)


def group_by_condition(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group result records by their 'condition' field.

    Args:
        records: List of result dicts.

    Returns:
        Dict mapping condition name to list of records with that condition.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        cond = r["condition"]
        if cond not in grouped:
            grouped[cond] = []
        grouped[cond].append(r)
    return grouped


def summarize_phase(phase_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for a single phase.

    Args:
        phase_name: Name of the phase (e.g., "phase3", "real_world").
        records: List of result records for this phase.

    Returns:
        Dict with summary statistics for the phase.
    """
    if not records:
        return {
            "phase_name": phase_name,
            "num_records": 0,
            "conditions": {},
            "overall_bad_rate": None,
            "overall_pass_rate": None,
        }

    # Group by condition and compute per-condition stats.
    by_condition = group_by_condition(records)
    conditions_summary: Dict[str, Any] = {}
    for cond, cond_records in sorted(by_condition.items()):
        conditions_summary[cond] = {
            "num_records": len(cond_records),
            "bad_rate": round(compute_bad_rate(cond_records), 4),
            "pass_rate": round(compute_pass_rate(cond_records), 4),
            "bad_count": sum(1 for r in cond_records if r.get("bad_label", False)),
            "pass_count": sum(1 for r in cond_records if r.get("pass_label", False)),
        }

    # Overall stats (all conditions combined).
    overall_bad_rate = round(compute_bad_rate(records), 4)
    overall_pass_rate = round(compute_pass_rate(records), 4)

    return {
        "phase_name": phase_name,
        "num_records": len(records),
        "num_conditions": len(by_condition),
        "conditions": conditions_summary,
        "overall_bad_rate": overall_bad_rate,
        "overall_pass_rate": overall_pass_rate,
    }


def compute_cross_phase_bad_rates(
    all_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Compute bad rates aggregated across all phases, grouped by condition.

    This gives a global view: for each condition (e.g., "warm", "clean"),
    what is the bad rate across all phases where that condition appears?

    Args:
        all_results: Dict mapping phase_name -> list of records.

    Returns:
        Dict with condition-level aggregated bad rates.
    """
    # Collect all records by condition across all phases.
    condition_to_records: Dict[str, List[Dict[str, Any]]] = {}
    condition_to_phases: Dict[str, List[str]] = {}

    for phase_name, records in all_results.items():
        for r in records:
            cond = r["condition"]
            if cond not in condition_to_records:
                condition_to_records[cond] = []
                condition_to_phases[cond] = []
            condition_to_records[cond].append(r)
            if phase_name not in condition_to_phases[cond]:
                condition_to_phases[cond].append(phase_name)

    # Compute summary per condition.
    condition_summary: Dict[str, Any] = {}
    for cond in sorted(condition_to_records.keys()):
        recs = condition_to_records[cond]
        condition_summary[cond] = {
            "num_records": len(recs),
            "num_phases": len(condition_to_phases[cond]),
            "phases": sorted(condition_to_phases[cond]),
            "bad_rate": round(compute_bad_rate(recs), 4),
            "pass_rate": round(compute_pass_rate(recs), 4),
            "bad_count": sum(1 for r in recs if r.get("bad_label", False)),
            "pass_count": sum(1 for r in recs if r.get("pass_label", False)),
        }

    return condition_summary


def compute_baseline_performance(
    all_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Compute performance summary per baseline system.

    A "baseline system" corresponds to a condition that represents a specific
    memory management approach (e.g., "memgpt", "mem0", "zep").
    This function aggregates results per baseline across all phases.

    Args:
        all_results: Dict mapping phase_name -> list of records.

    Returns:
        Dict with baseline-level performance summary.
    """
    # Baseline conditions: these are the memory-system conditions we compare.
    # Clean and warm are not "baselines" per se, but we include them for reference.
    baseline_conditions = [
        "clean", "warm", "delete-target",
        "memgpt", "mem0", "zep", "a-mem", "reflexion", "workflow",
        "naive-vector",
    ]

    baseline_summary: Dict[str, Any] = {}

    # Collect records per condition across all phases.
    condition_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for records in all_results.values():
        for r in records:
            cond = r["condition"]
            if cond not in condition_to_records:
                condition_to_records[cond] = []
            condition_to_records[cond].append(r)

    for bl in baseline_conditions:
        if bl not in condition_to_records:
            # This baseline was not run in any phase; skip.
            continue
        recs = condition_to_records[bl]
        baseline_summary[bl] = {
            "num_records": len(recs),
            "bad_rate": round(compute_bad_rate(recs), 4),
            "pass_rate": round(compute_pass_rate(recs), 4),
            "bad_count": sum(1 for r in recs if r.get("bad_label", False)),
            "pass_count": sum(1 for r in recs if r.get("pass_label", False)),
        }

    return baseline_summary


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------


def generate_mock_summary_report() -> Dict[str, Any]:
    """Generate a mock summary report for testing/demonstration purposes.

    The mock report contains realistic but fake numbers that mimic what a
    real summary report would look like. This is useful for testing the
    report format and for demonstrating the script before real data is available.

    Returns:
        Dict representing a complete mock summary report.
    """
    timestamp = datetime.now().isoformat()

    # Mock per-phase summaries.
    phases_mock = {
        "main": {
            "phase_name": "main",
            "num_records": 360,
            "num_conditions": 2,
            "conditions": {
                "clean": {
                    "num_records": 180,
                    "bad_rate": 0.05,
                    "pass_rate": 0.82,
                    "bad_count": 9,
                    "pass_count": 148,
                },
                "warm": {
                    "num_records": 180,
                    "bad_rate": 0.28,
                    "pass_rate": 0.71,
                    "bad_count": 50,
                    "pass_count": 128,
                },
            },
            "overall_bad_rate": 0.164,
            "overall_pass_rate": 0.765,
        },
        "phase3": {
            "phase_name": "phase3",
            "num_records": 540,
            "num_conditions": 3,
            "conditions": {
                "clean": {
                    "num_records": 180,
                    "bad_rate": 0.04,
                    "pass_rate": 0.84,
                    "bad_count": 7,
                    "pass_count": 151,
                },
                "memgpt": {
                    "num_records": 180,
                    "bad_rate": 0.31,
                    "pass_rate": 0.69,
                    "bad_count": 56,
                    "pass_count": 124,
                },
                "mem0": {
                    "num_records": 180,
                    "bad_rate": 0.27,
                    "pass_rate": 0.73,
                    "bad_count": 49,
                    "pass_count": 131,
                },
            },
            "overall_bad_rate": 0.207,
            "overall_pass_rate": 0.753,
        },
        "real_world": {
            "phase_name": "real_world",
            "num_records": 720,
            "num_conditions": 8,
            "conditions": {
                "clean": {
                    "num_records": 90,
                    "bad_rate": 0.03,
                    "pass_rate": 0.86,
                    "bad_count": 3,
                    "pass_count": 77,
                },
                "warm": {
                    "num_records": 90,
                    "bad_rate": 0.32,
                    "pass_rate": 0.68,
                    "bad_count": 29,
                    "pass_count": 61,
                },
                "memgpt": {
                    "num_records": 90,
                    "bad_rate": 0.34,
                    "pass_rate": 0.66,
                    "bad_count": 31,
                    "pass_count": 59,
                },
                "mem0": {
                    "num_records": 90,
                    "bad_rate": 0.29,
                    "pass_rate": 0.71,
                    "bad_count": 26,
                    "pass_count": 64,
                },
                "zep": {
                    "num_records": 90,
                    "bad_rate": 0.31,
                    "pass_rate": 0.69,
                    "bad_count": 28,
                    "pass_count": 62,
                },
                "a-mem": {
                    "num_records": 90,
                    "bad_rate": 0.26,
                    "pass_rate": 0.74,
                    "bad_count": 23,
                    "pass_count": 67,
                },
                "reflexion": {
                    "num_records": 90,
                    "bad_rate": 0.24,
                    "pass_rate": 0.76,
                    "bad_count": 22,
                    "pass_count": 68,
                },
                "workflow": {
                    "num_records": 90,
                    "bad_rate": 0.21,
                    "pass_rate": 0.79,
                    "bad_count": 19,
                    "pass_count": 71,
                },
            },
            "overall_bad_rate": 0.25,
            "overall_pass_rate": 0.723,
        },
    }

    # Mock cross-phase bad rates.
    cross_phase_mock = {
        "clean": {
            "num_records": 270,
            "num_phases": 2,
            "phases": ["main", "real_world"],
            "bad_rate": 0.037,
            "pass_rate": 0.84,
            "bad_count": 10,
            "pass_count": 228,
        },
        "warm": {
            "num_records": 270,
            "num_phases": 2,
            "phases": ["main", "real_world"],
            "bad_rate": 0.293,
            "pass_rate": 0.704,
            "bad_count": 79,
            "pass_count": 190,
        },
        "memgpt": {
            "num_records": 270,
            "num_phases": 2,
            "phases": ["phase3", "real_world"],
            "bad_rate": 0.322,
            "pass_rate": 0.678,
            "bad_count": 87,
            "pass_count": 183,
        },
        "mem0": {
            "num_records": 270,
            "num_phases": 2,
            "phases": ["phase3", "real_world"],
            "bad_rate": 0.278,
            "pass_rate": 0.722,
            "bad_count": 75,
            "pass_count": 195,
        },
    }

    # Mock baseline performance.
    baseline_mock = {
        "clean": {
            "num_records": 270,
            "bad_rate": 0.037,
            "pass_rate": 0.84,
            "bad_count": 10,
            "pass_count": 228,
        },
        "warm": {
            "num_records": 270,
            "bad_rate": 0.293,
            "pass_rate": 0.704,
            "bad_count": 79,
            "pass_count": 190,
        },
        "memgpt": {
            "num_records": 270,
            "bad_rate": 0.322,
            "pass_rate": 0.678,
            "bad_count": 87,
            "pass_count": 183,
        },
        "mem0": {
            "num_records": 270,
            "bad_rate": 0.278,
            "pass_rate": 0.722,
            "bad_count": 75,
            "pass_count": 195,
        },
        "zep": {
            "num_records": 90,
            "bad_rate": 0.311,
            "pass_rate": 0.689,
            "bad_count": 28,
            "pass_count": 62,
        },
        "a-mem": {
            "num_records": 90,
            "bad_rate": 0.256,
            "pass_rate": 0.744,
            "bad_count": 23,
            "pass_count": 67,
        },
        "reflexion": {
            "num_records": 90,
            "bad_rate": 0.244,
            "pass_rate": 0.756,
            "bad_count": 22,
            "pass_count": 68,
        },
        "workflow": {
            "num_records": 90,
            "bad_rate": 0.211,
            "pass_rate": 0.789,
            "bad_count": 19,
            "pass_count": 71,
        },
    }

    # Overall conclusions (mock).
    conclusions_mock = {
        "key_findings": [
            "Memory-augmented agents (warm) have 7.9x higher bad rate than clean baselines (29.3% vs 3.7%).",
            "MemGPT and Mem0 show the highest bad rates among baseline systems (32.2% and 27.8%).",
            "Workflow-based memory management shows the lowest bad rate among memory-augmented systems (21.1%).",
            "Pass rates drop by 13.6 percentage points on average when memory is introduced (84.0% -> 70.4%).",
        ],
        "reproduction_status": "partial",
        "notes": "Mock data generated for testing. Replace with real data when available.",
    }

    return {
        "metadata": {
            "generated_at": timestamp,
            "script": "summarize_results.py",
            "mode": "mock",
            "phases_included": list(phases_mock.keys()),
        },
        "phases": phases_mock,
        "cross_phase_bad_rates": cross_phase_mock,
        "baseline_performance": baseline_mock,
        "conclusions": conclusions_mock,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_summary_report(
    all_results: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Generate the complete summary report from loaded results.

    Args:
        all_results: Dict mapping phase_name -> list of result records.

    Returns:
        Dict representing the complete summary report, ready for JSON serialization.
    """
    timestamp = datetime.now().isoformat()

    # 1. Per-phase summaries.
    phases_summary: Dict[str, Any] = {}
    for phase_name, records in sorted(all_results.items()):
        phases_summary[phase_name] = summarize_phase(phase_name, records)

    # 2. Cross-phase bad rates by condition.
    cross_phase = compute_cross_phase_bad_rates(all_results)

    # 3. Baseline system performance.
    baseline = compute_baseline_performance(all_results)

    # 4. Overall conclusions.
    conclusions = derive_conclusions(phases_summary, cross_phase, baseline)

    return {
        "metadata": {
            "generated_at": timestamp,
            "script": "summarize_results.py",
            "mode": "real",
            "phases_included": list(all_results.keys()),
            "total_records": sum(len(r) for r in all_results.values()),
        },
        "phases": phases_summary,
        "cross_phase_bad_rates": cross_phase,
        "baseline_performance": baseline,
        "conclusions": conclusions,
    }


def derive_conclusions(
    phases_summary: Dict[str, Any],
    cross_phase: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive overall conclusions from the computed statistics.

    This function examines the summary statistics and produces a set of
    key findings and a reproduction status assessment.

    Args:
        phases_summary: Per-phase summary stats.
        cross_phase: Cross-phase bad rates by condition.
        baseline: Baseline system performance.

    Returns:
        Dict with key findings, reproduction status, and notes.
    """
    findings: List[str] = []

    # Finding 1: Compare clean vs warm bad rates.
    clean_bad_rate = cross_phase.get("clean", {}).get("bad_rate")
    warm_bad_rate = cross_phase.get("warm", {}).get("bad_rate")
    if clean_bad_rate is not None and warm_bad_rate is not None and clean_bad_rate > 0:
        ratio = warm_bad_rate / clean_bad_rate
        findings.append(
            f"Memory-augmented agents (warm) have {ratio:.1f}x higher bad rate "
            f"than clean baselines ({warm_bad_rate:.1%} vs {clean_bad_rate:.1%})."
        )

    # Finding 2: Which baseline has the highest bad rate?
    if baseline:
        sorted_baselines = sorted(
            baseline.items(),
            key=lambda x: x[1].get("bad_rate", 0),
            reverse=True,
        )
        if sorted_baselines:
            top_name, top_stats = sorted_baselines[0]
            findings.append(
                f"The baseline with the highest bad rate is '{top_name}' "
                f"({top_stats['bad_rate']:.1%}), "
                f"based on {top_stats['num_records']} records."
            )

    # Finding 3: Pass rate drop.
    clean_pass_rate = cross_phase.get("clean", {}).get("pass_rate")
    warm_pass_rate = cross_phase.get("warm", {}).get("pass_rate")
    if clean_pass_rate is not None and warm_pass_rate is not None:
        drop = clean_pass_rate - warm_pass_rate
        findings.append(
            f"Pass rates drop by {drop:.1%} when memory is introduced "
            f"({clean_pass_rate:.1%} -> {warm_pass_rate:.1%})."
        )

    # Finding 4: Total records.
    total_records = sum(
        p.get("num_records", 0) for p in phases_summary.values()
    )
    findings.append(
        f"Total of {total_records} result records across {len(phases_summary)} phases."
    )

    # Determine reproduction status.
    # Status is "complete" if we have both clean and warm data with sufficient records.
    has_clean = "clean" in cross_phase and cross_phase["clean"].get("num_records", 0) > 0
    has_warm = "warm" in cross_phase and cross_phase["warm"].get("num_records", 0) > 0
    sufficient_data = total_records >= 100  # Arbitrary threshold; adjust as needed.

    if has_clean and has_warm and sufficient_data:
        status = "complete"
    elif has_clean or has_warm:
        status = "partial"
    else:
        status = "insufficient_data"

    return {
        "key_findings": findings,
        "reproduction_status": status,
        "notes": (
            "Conclusions auto-derived from summary statistics. "
            "Review and edit manually before including in the paper."
        ),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Summarize all experimental results for the MemTrace paper "
            "reproduction work (Phase 7, part 1)."
        )
    )
    parser.add_argument(
        "--results-dirs",
        type=str,
        default=",".join(DEFAULT_RESULTS_DIRS),
        help=(
            "Comma-separated list of result directories to scan. "
            f"Default: {','.join(DEFAULT_RESULTS_DIRS)}"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for the summary report JSON. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Generate a mock summary report instead of reading real data.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point.

    Flow:
    1. Parse command-line arguments.
    2. If --use-mock: generate mock report and skip data loading.
    3. Otherwise: load all results from the specified directories.
    4. Generate the summary report.
    5. Save the report to the output path.
    """
    args = parse_args()

    # Resolve output path relative to project root if it is not absolute.
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(PROJECT_ROOT, output_path)

    # Ensure output directory exists (Fast-Fail: we must be able to write here).
    output_dir = os.path.dirname(output_path)
    assert output_dir, f"Cannot determine output directory from path: {output_path}"
    os.makedirs(output_dir, exist_ok=True)

    if args.use_mock:
        print("[INFO] Generating mock summary report (--use-mock flag set)...")
        report = generate_mock_summary_report()
        print(f"[INFO] Mock report generated with {len(report['phases'])} phases.")
    else:
        # Parse results-dirs (comma-separated).
        results_dirs = [d.strip() for d in args.results_dirs.split(",") if d.strip()]
        # Resolve relative paths against project root.
        results_dirs = [
            d if os.path.isabs(d) else os.path.join(PROJECT_ROOT, d)
            for d in results_dirs
        ]
        print(f"[INFO] Loading results from {len(results_dirs)} directories...")
        print(f"[INFO] Directories: {results_dirs}")

        all_results = load_all_results(results_dirs)

        total_records = sum(len(r) for r in all_results.values())
        print(f"[INFO] Total records loaded: {total_records}")
        assert total_records > 0, (
            "No result records were loaded. "
            "Check that the results directories exist and contain valid JSON files."
        )

        print("[INFO] Generating summary report...")
        report = generate_summary_report(all_results)
        print(
            f"[INFO] Report generated: {len(report['phases'])} phases, "
            f"{report['metadata']['total_records']} total records."
        )

    # Save the report.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Summary report saved to: {output_path}")

    # Also print a brief summary to stdout for quick inspection.
    print("\n=== SUMMARY ===")
    print(f"Generated at: {report['metadata']['generated_at']}")
    print(f"Mode: {report['metadata']['mode']}")
    print(f"Phases: {list(report['phases'].keys())}")
    print(f"Total records: {report['metadata'].get('total_records', 'N/A')}")
    print("\nKey findings:")
    for i, finding in enumerate(report["conclusions"]["key_findings"], 1):
        print(f"  {i}. {finding}")
    print(f"\nReproduction status: {report['conclusions']['reproduction_status']}")


if __name__ == "__main__":
    main()

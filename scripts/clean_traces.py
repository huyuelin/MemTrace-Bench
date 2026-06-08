#!/usr/bin/env python3
"""
Clean merged agent traces for the paper "Memory Is a Hidden Dependency:
A Benchmark for Replay-Defined Harm in Stateful Coding Agents" reproduction work.

This script is the third part of Phase 1 (Data Collection & Processing).
It reads merged traces from data/processed/merged_traces.json, applies
cleaning rules to remove invalid or corrupted traces, and writes the
cleaned result to data/processed/cleaned_traces.json along with a
cleaning report.

Cleaning rules:
1. Remove traces with empty sequence_id
2. Remove traces with empty repo_url
3. Remove traces with empty files list
4. Fix missing hashes field (set to empty dict)
5. Fix missing seeds field (set to [42])
6. Standardize timestamp (parse string to float if needed)
7. Remove duplicate sequence_id (keep first occurrence)

Usage:
    python clean_traces.py [--input PATH] [--output PATH] [--report PATH]
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

# Default paths: the user-specified paths are relative to the project root
# (/Users/jackey/Desktop/idea_factory/ai_infra_icse2027), not code/.
# __file__ = .../code/scripts/clean_traces.py, so project root is 2 levels up.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "merged_traces.json")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_traces.json")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaning_report.json")

# Required fields that must be non-empty (after cleaning).
# Traces missing these after cleaning are removed.
REQUIRED_NON_EMPTY_FIELDS = ["sequence_id", "repo_url", "files"]


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------


def load_traces(input_path: str) -> List[Dict[str, Any]]:
    """Load traces from a JSON file.

    Crash immediately if the file does not exist or contains invalid JSON.
    This is a Fast-Fail check: we must not silently proceed with empty data.

    Args:
        input_path: Path to the input JSON file.

    Returns:
        List of trace dicts.

    Raises:
        AssertionError: if file is missing or JSON is invalid.
    """
    assert os.path.isfile(input_path), f"Input file does not exist: {input_path}"
    with open(input_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise AssertionError(
                f"Invalid JSON in {input_path}: {e.msg} (line {e.lineno}, col {e.colno})"
            ) from e

    # The file may contain a dict with a "traces" key, or a plain list.
    if isinstance(data, dict) and "traces" in data:
        traces = data["traces"]
    else:
        traces = data

    assert isinstance(traces, list), (
        f"Expected list of traces in {input_path}, got {type(traces).__name__}"
    )
    return traces


def clean_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Apply field-level cleaning rules to a single trace.

    This function does NOT decide whether to keep or discard the trace.
    It only normalizes fields so that downstream removal rules can be applied
    consistently. Removal happens in clean_traces() after all fields are fixed.

    Cleaning steps:
    1. Fix missing/non-dict hashes -> {}
    2. Fix missing/non-list seeds -> [42]
    3. Standardize timestamp to float

    Args:
        trace: Raw trace dict.

    Returns:
        Cleaned trace dict (same object, mutated in place).
    """
    # Rule 4: Fix missing or invalid hashes field.
    # hashes should be a dict mapping file paths to hash strings.
    if "hashes" not in trace or not isinstance(trace.get("hashes"), dict):
        trace["hashes"] = {}

    # Rule 5: Fix missing or invalid seeds field.
    # seeds should be a list of ints. Default [42] is the standard seed.
    seeds = trace.get("seeds")
    if seeds is None:
        trace["seeds"] = [42]
    elif not isinstance(seeds, list):
        trace["seeds"] = [42]
    else:
        # Ensure all seeds are ints (coerce floats, drop invalid).
        cleaned_seeds: List[int] = []
        for s in seeds:
            if isinstance(s, (int, float)) and not isinstance(s, bool):
                cleaned_seeds.append(int(s))
        trace["seeds"] = cleaned_seeds if cleaned_seeds else [42]

    # Rule 6: Standardize timestamp.
    # timestamp may be a string like "2024-01-15T10:30:00Z" or a Unix epoch float.
    ts = trace.get("timestamp")
    if ts is not None and isinstance(ts, str):
        try:
            # Try parsing ISO format string to Unix timestamp (float).
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            trace["timestamp"] = dt.timestamp()
        except (ValueError, TypeError):
            # If parsing fails, set to 0.0 (unknown time).
            trace["timestamp"] = 0.0
    elif ts is not None:
        # Already a number; ensure it is float.
        try:
            trace["timestamp"] = float(ts)
        except (ValueError, TypeError):
            trace["timestamp"] = 0.0
    # If timestamp is missing, leave it as-is (removal rules will catch it if needed).

    return trace


def should_remove_trace(trace: Dict[str, Any]) -> Optional[str]:
    """Check if a trace should be removed. Returns reason or None.

    Removal rules (applied after field-level cleaning):
    1. sequence_id is empty (None, "", or whitespace-only)
    2. repo_url is empty (None, "", or whitespace-only)
    3. files is empty (None, or empty list)

    Args:
        trace: Cleaned trace dict.

    Returns:
        Reason string if trace should be removed, None if trace is valid.
    """
    seq_id = trace.get("sequence_id")
    if not seq_id or (isinstance(seq_id, str) and not seq_id.strip()):
        return "empty_sequence_id"

    repo_url = trace.get("repo_url")
    if not repo_url or (isinstance(repo_url, str) and not repo_url.strip()):
        return "empty_repo_url"

    files = trace.get("files")
    if not files or not isinstance(files, list) or len(files) == 0:
        return "empty_files"

    return None


def clean_traces(traces: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply all cleaning rules to the trace list.

    Processing order:
    1. Field-level cleaning (normalize hashes, seeds, timestamp)
    2. Removal by rule (empty fields)
    3. Deduplication by sequence_id (keep first)

    Args:
        traces: Raw list of trace dicts.

    Returns:
        Tuple of (cleaned_traces, cleaning_report_dict).
    """
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "input_count": len(traces),
        "removed": {
            "empty_sequence_id": 0,
            "empty_repo_url": 0,
            "empty_files": 0,
            "duplicate_sequence_id": 0,
        },
        "kept": 0,
        "output_count": 0,
    }

    # Step 1: Field-level cleaning on all traces.
    cleaned: List[Dict[str, Any]] = []
    for i, trace in enumerate(traces):
        assert isinstance(trace, dict), (
            f"Trace at index {i} is not a dict: {type(trace).__name__}"
        )
        cleaned.append(clean_trace(trace))

    # Step 2: Remove invalid traces.
    valid: List[Dict[str, Any]] = []
    for trace in cleaned:
        reason = should_remove_trace(trace)
        if reason is not None:
            report["removed"][reason] = report["removed"].get(reason, 0) + 1
        else:
            valid.append(trace)

    # Step 3: Deduplicate by sequence_id (keep first occurrence).
    seen_ids: set = set()
    deduplicated: List[Dict[str, Any]] = []
    for trace in valid:
        sid = trace["sequence_id"]
        if sid in seen_ids:
            report["removed"]["duplicate_sequence_id"] += 1
        else:
            seen_ids.add(sid)
            deduplicated.append(trace)

    # Final counts.
    report["kept"] = len(deduplicated)
    report["output_count"] = len(deduplicated)

    # Compute additional statistics on cleaned data.
    report["stats"] = _compute_cleaned_stats(deduplicated)

    return deduplicated, report


def _compute_cleaned_stats(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics on the cleaned traces.

    These stats help verify data quality and are included in the report.

    Args:
        traces: Cleaned list of trace dicts.

    Returns:
        Dict with counts and distributions.
    """
    if not traces:
        return {"count": 0}

    # Count by memory_type.
    memory_type_counts: Dict[str, int] = {}
    for t in traces:
        mt = t.get("memory_type", "unknown")
        memory_type_counts[mt] = memory_type_counts.get(mt, 0) + 1

    # Count by task_type.
    task_type_counts: Dict[str, int] = {}
    for t in traces:
        tt = t.get("task_type", "unknown")
        task_type_counts[tt] = task_type_counts.get(tt, 0) + 1

    # Count by bad_label.
    bad_label_counts: Dict[str, int] = {}
    for t in traces:
        bl = t.get("bad_label", "unknown")
        bad_label_counts[bl] = bad_label_counts.get(bl, 0) + 1

    # Timestamp range.
    timestamps = [t.get("timestamp", 0.0) for t in traces if t.get("timestamp", 0.0) > 0]
    ts_min = min(timestamps) if timestamps else None
    ts_max = max(timestamps) if timestamps else None

    return {
        "count": len(traces),
        "memory_type_counts": memory_type_counts,
        "task_type_counts": task_type_counts,
        "bad_label_counts": bad_label_counts,
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "timestamp_count": len(timestamps),
    }


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def write_report(report: Dict[str, Any], report_path: str) -> None:
    """Write the cleaning report to a JSON file.

    Uses atomic write (write to tmp, then rename) to prevent corruption
    if the script crashes during write.

    Args:
        report: Report dict to write.
        report_path: Output file path.
    """
    report_dir = os.path.dirname(report_path)
    assert report_dir, f"Cannot determine report directory from {report_path}"
    os.makedirs(report_dir, exist_ok=True)

    tmp_path = report_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, report_path)


def write_traces(traces: List[Dict[str, Any]], output_path: str) -> None:
    """Write cleaned traces to a JSON file.

    Uses atomic write (write to tmp, then rename) to prevent corruption.

    Args:
        traces: Cleaned list of trace dicts.
        output_path: Output file path.
    """
    output_dir = os.path.dirname(output_path)
    assert output_dir, f"Cannot determine output directory from {output_path}"
    os.makedirs(output_dir, exist_ok=True)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean merged agent traces: remove invalid entries, deduplicate, and normalize fields."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSON file path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help=f"Cleaning report JSON file path (default: {DEFAULT_REPORT_PATH})",
    )
    args = parser.parse_args()

    # Resolve paths to absolute.
    input_path: str = os.path.abspath(args.input)
    output_path: str = os.path.abspath(args.output)
    report_path: str = os.path.abspath(args.report)

    # Validate input path exists (Fast-Fail: do not create dummy input).
    assert os.path.isfile(input_path), (
        f"Input file not found: {input_path}\n"
        f"Run the merge step first to create {input_path}"
    )

    # Ensure output and report directories exist.
    for p in [output_path, report_path]:
        d = os.path.dirname(p)
        assert d, f"Cannot determine directory from {p}"
        os.makedirs(d, exist_ok=True)

    print("=== Trace Cleaner ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print()

    # Load, clean, report.
    print("[1/4] Loading traces...")
    traces = load_traces(input_path)
    print(f"  Loaded {len(traces)} traces.")

    print("[2/4] Cleaning traces...")
    cleaned, report = clean_traces(traces)
    print(f"  Removed {report['input_count'] - report['output_count']} traces:")
    for reason, count in report["removed"].items():
        if count > 0:
            print(f"    - {reason}: {count}")
    print(f"  Kept {report['output_count']} traces.")

    print("[3/4] Writing cleaned traces...")
    write_traces(cleaned, output_path)
    print(f"  Wrote {len(cleaned)} traces to {output_path}")

    print("[4/4] Writing cleaning report...")
    write_report(report, report_path)
    print(f"  Wrote report to {report_path}")

    print()
    print("=== Cleaning Summary ===")
    print(f"  Input count:  {report['input_count']}")
    print(f"  Output count: {report['output_count']}")
    print(f"  Removed:      {report['input_count'] - report['output_count']}")
    if report.get("stats", {}).get("memory_type_counts"):
        print(f"  Memory types: {report['stats']['memory_type_counts']}")
    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()

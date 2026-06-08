#!/usr/bin/env python3
"""
Validate agent traces quality for the paper "Memory Is a Hidden Dependency:
A Benchmark for Replay-Defined Harm in Stateful Coding Agents" reproduction work.

This script is the fourth part of stage one. It validates that cleaned traces
conform to the SequenceCard schema defined in core/schemas.py, computes
distribution statistics, and generates a quality report.

Usage:
    python validate_traces.py [--input PATH] [--schema PATH] [--output PATH] [--strict]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Constants: valid enum values (must match core/schemas.py SequenceCard)
# ---------------------------------------------------------------------------

VALID_MEMORY_TYPES = [
    "cross-repo",
    "stale-api",
    "stale-security",
    "hidden-channel",
    "in-scope",
]

VALID_CHANNELS = [
    "memory-store",
    "conversation",
    "tool-log",
    "terminal-cache",
    "wrapper-patch",
    "scratchpad",
]

VALID_TASK_TYPES = ["bugfix", "feature", "refactor", "optimization"]

# SequenceCard fields that must be present and non-null in a valid trace.
# These are derived from the dataclass fields in core/schemas.py.
REQUIRED_FIELDS = [
    "sequence_id",
    "repo_url",
    "repo_commit",
    "repo_license",
    "task_type",
    "prompt_hash",
    "files",
    "memory_type",
    "channel",
    "evidence",
    "oracle_type",
    "tests",
    "rules",
    "policy",
    "conditions",
    "placebo_match",
    "scope_label",
    "staleness_label",
    "bad_label",
    "security_label",
    "docker_image",
    "hashes",
    "seeds",
]

# Fields with expected types (Python type name as string).
FIELD_TYPES = {
    "sequence_id": str,
    "repo_url": str,
    "repo_commit": str,
    "repo_license": str,
    "task_type": str,
    "prompt_hash": str,
    "files": list,
    "memory_type": str,
    "channel": str,
    "evidence": str,
    "oracle_type": str,
    "tests": list,
    "rules": str,
    "policy": str,
    "conditions": list,
    "placebo_match": str,
    "scope_label": str,
    "staleness_label": str,
    "bad_label": str,
    "security_label": str,
    "docker_image": str,
    "hashes": dict,
    "seeds": list,
}

# Default paths relative to this script's location.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR.parent / "data" / "processed" / "cleaned_traces.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR.parent / "data" / "processed" / "quality_report.json"


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def load_traces(input_path: str) -> List[Dict[str, Any]]:
    """Load traces from a JSON file. Crash on any error (Fast-Fail).

    Args:
        input_path: Path to the JSON file containing traces.

    Returns:
        List of trace dicts.

    Raises:
        AssertionError: if the file cannot be read or parsed.
    """
    assert os.path.isfile(input_path), f"Input file does not exist: {input_path}"
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Invalid JSON in {input_path}: {e}") from e
    except OSError as e:
        raise AssertionError(f"Cannot read {input_path}: {e}") from e

    assert isinstance(data, list), (
        f"Expected a JSON list at top level, got {type(data).__name__}"
    )
    assert len(data) > 0, f"Input file {input_path} contains an empty list"
    print(f"[validate] Loaded {len(data)} traces from {input_path}")
    return data


def validate_required_fields(
    trace: Dict[str, Any], index: int
) -> List[str]:
    """Check that all REQUIRED_FIELDS are present in the trace.

    Args:
        trace: A single trace dict.
        index: Position of this trace in the input list (for error messages).

    Returns:
        List of error strings (empty if all fields present).
    """
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in trace:
            errors.append(f"trace[{index}]: missing required field '{field}'")
    return errors


def validate_field_types(
    trace: Dict[str, Any], index: int
) -> List[str]:
    """Check that each field has the expected Python type.

    Args:
        trace: A single trace dict.
        index: Position of this trace in the input list.

    Returns:
        List of error strings.
    """
    errors: List[str] = []
    for field, expected_type in FIELD_TYPES.items():
        if field not in trace:
            continue  # Missing field caught by validate_required_fields.
        value = trace[field]
        if not isinstance(value, expected_type):
            errors.append(
                f"trace[{index}]: field '{field}' has type "
                f"{type(value).__name__}, expected {expected_type.__name__}"
            )
    return errors


def validate_sequence_id_uniqueness(
    traces: List[Dict[str, Any]]
) -> List[str]:
    """Check that all sequence_id values are unique.

    Args:
        traces: All loaded traces.

    Returns:
        List of error strings for duplicates.
    """
    errors: List[str] = []
    seen: Dict[str, int] = {}
    for i, trace in enumerate(traces):
        sid = trace.get("sequence_id")
        if not isinstance(sid, str):
            continue  # Type error caught elsewhere.
        if sid in seen:
            errors.append(
                f"Duplicate sequence_id '{sid}' at trace[{i}] "
                f"(first seen at trace[{seen[sid]}])"
            )
        else:
            seen[sid] = i
    return errors


def validate_repo_url(trace: Dict[str, Any], index: int) -> List[str]:
    """Check that repo_url starts with 'https://github.com/'.

    Args:
        trace: A single trace dict.
        index: Position of this trace in the input list.

    Returns:
        List of error strings.
    """
    errors: List[str] = []
    url = trace.get("repo_url")
    if not isinstance(url, str):
        return errors  # Type error caught elsewhere.
    if not url.startswith("https://github.com/"):
        errors.append(
            f"trace[{index}]: repo_url does not start with "
            f"'https://github.com/': {url!r}"
        )
    return errors


def validate_memory_type(trace: Dict[str, Any], index: int) -> List[str]:
    """Check that memory_type is one of VALID_MEMORY_TYPES.

    Args:
        trace: A single trace dict.
        index: Position of this trace in the input list.

    Returns:
        List of error strings.
    """
    errors: List[str] = []
    mt = trace.get("memory_type")
    if not isinstance(mt, str):
        return errors
    if mt not in VALID_MEMORY_TYPES:
        errors.append(
            f"trace[{index}]: memory_type '{mt}' is not valid. "
            f"Allowed: {VALID_MEMORY_TYPES}"
        )
    return errors


def validate_channel(trace: Dict[str, Any], index: int) -> List[str]:
    """Check that channel is one of VALID_CHANNELS.

    Args:
        trace: A single trace dict.
        index: Position of this trace in the input list.

    Returns:
        List of error strings.
    """
    errors: List[str] = []
    ch = trace.get("channel")
    if not isinstance(ch, str):
        return errors
    if ch not in VALID_CHANNELS:
        errors.append(
            f"trace[{index}]: channel '{ch}' is not valid. "
            f"Allowed: {VALID_CHANNELS}"
        )
    return errors


def validate_conditions_non_empty(
    trace: Dict[str, Any], index: int
) -> List[str]:
    """Check that conditions list is non-empty.

    An empty conditions list means the trace has no intervention/control
    labeling, which makes it unusable for the benchmark.

    Args:
        trace: A single trace dict.
        index: Position of this trace in the input list.

    Returns:
        List of error strings.
    """
    errors: List[str] = []
    conds = trace.get("conditions")
    if conds is None:
        return errors  # Missing field caught elsewhere.
    if isinstance(conds, list) and len(conds) == 0:
        errors.append(
            f"trace[{index}]: conditions list is empty (needs at least one condition)"
        )
    return errors


def validate_seeds_non_empty(
    trace: Dict[str, Any], index: int
) -> List[str]:
    """Check that seeds list is non-empty.

    Empty seeds means the experiment is not reproducible because there is
    no recorded random seed.

    Args:
        trace: A single trace dict.
        index: Position of this trace in the input list.

    Returns:
        List of error strings.
    """
    errors: List[str] = []
    seeds = trace.get("seeds")
    if seeds is None:
        return errors
    if isinstance(seeds, list) and len(seeds) == 0:
        errors.append(
            f"trace[{index}]: seeds list is empty (needs at least one seed for reproducibility)"
        )
    return errors


def validate_task_type(trace: Dict[str, Any], index: int) -> List[str]:
    """Check that task_type is one of VALID_TASK_TYPES.

    Args:
        trace: A single trace dict.
        index: Position of this trace in the input list.

    Returns:
        List of error strings.
    """
    errors: List[str] = []
    tt = trace.get("task_type")
    if not isinstance(tt, str):
        return errors
    if tt not in VALID_TASK_TYPES:
        errors.append(
            f"trace[{index}]: task_type '{tt}' is not valid. "
            f"Allowed: {VALID_TASK_TYPES}"
        )
    return errors


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------


def compute_source_stats(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count traces by source (inferred from sequence_id prefix).

    The sequence_id naming convention used by the collection scripts is:
      - 'github_...'  -> sourced from GitHub
      - 'swe_bench_...' -> sourced from SWE-bench
      - 'react_...'    -> sourced from ReAct traces

    Args:
        traces: All loaded traces.

    Returns:
        Dict mapping source label to count.
    """
    stats: Dict[str, int] = {"github": 0, "swe_bench": 0, "react": 0, "unknown": 0}
    for trace in traces:
        sid = trace.get("sequence_id", "")
        if isinstance(sid, str):
            if sid.startswith("github_"):
                stats["github"] += 1
            elif sid.startswith("swe-bench-"):
                stats["swe_bench"] += 1
            elif sid.startswith("react-"):
                stats["react"] += 1
            else:
                stats["unknown"] += 1
        else:
            stats["unknown"] += 1
    return stats


def compute_memory_type_stats(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count traces by memory_type.

    Args:
        traces: All loaded traces.

    Returns:
        Dict mapping memory_type value to count.
    """
    stats: Dict[str, int] = {mt: 0 for mt in VALID_MEMORY_TYPES}
    for trace in traces:
        mt = trace.get("memory_type", "")
        if isinstance(mt, str) and mt in stats:
            stats[mt] += 1
        else:
            stats[mt] = stats.get(mt, 0) + 1
    return stats


def compute_channel_stats(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count traces by channel.

    Args:
        traces: All loaded traces.

    Returns:
        Dict mapping channel value to count.
    """
    stats: Dict[str, int] = {ch: 0 for ch in VALID_CHANNELS}
    for trace in traces:
        ch = trace.get("channel", "")
        if isinstance(ch, str) and ch in stats:
            stats[ch] += 1
        else:
            stats[ch] = stats.get(ch, 0) + 1
    return stats


def compute_task_type_stats(traces: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count traces by task_type.

    Args:
        traces: All loaded traces.

    Returns:
        Dict mapping task_type value to count.
    """
    stats: Dict[str, int] = {tt: 0 for tt in VALID_TASK_TYPES}
    for trace in traces:
        tt = trace.get("task_type", "")
        if isinstance(tt, str) and tt in stats:
            stats[tt] += 1
        else:
            stats[tt] = stats.get(tt, 0) + 1
    return stats


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------


def validate_all(
    traces: List[Dict[str, Any]], strict: bool = False
) -> Tuple[List[str], bool]:
    """Run all validation checks on the traces list.

    Args:
        traces: All loaded traces.
        strict: If True, raise on first error instead of collecting all.

    Returns:
        Tuple of (error_list, all_passed). all_passed is True iff error_list is empty.
    """
    all_errors: List[str] = []

    # Per-trace validations: run on every trace regardless of strict mode,
    # because the user needs the full error list to fix data. In strict mode
    # we still collect all per-trace errors but raise after the loop.
    for i, trace in enumerate(traces):
        # Each validator returns a list of error strings (may be empty).
        validators = [
            validate_required_fields,
            validate_field_types,
            validate_repo_url,
            validate_memory_type,
            validate_channel,
            validate_conditions_non_empty,
            validate_seeds_non_empty,
            validate_task_type,
        ]
        for validator in validators:
            errors = validator(trace, i)
            all_errors.extend(errors)
            if strict and errors:
                # In strict mode, raise immediately on the first error.
                raise AssertionError(errors[0])

    # Global validations (cross-trace checks).
    global_validators = [
        validate_sequence_id_uniqueness,
    ]
    for gv in global_validators:
        errors = gv(traces)
        all_errors.extend(errors)
        if strict and errors:
            raise AssertionError(errors[0])

    all_passed = len(all_errors) == 0
    return all_errors, all_passed


def build_quality_report(
    traces: List[Dict[str, Any]], errors: List[str], all_passed: bool
) -> Dict[str, Any]:
    """Build the quality report dict.

    Args:
        traces: All loaded traces.
        errors: List of validation error strings.
        all_passed: True if there were no errors.

    Returns:
        Dict suitable for JSON serialization as the quality report.
    """
    report: Dict[str, Any] = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "script": "validate_traces.py",
            "total_traces": len(traces),
        },
        "validation": {
            "passed": all_passed,
            "error_count": len(errors),
            "errors": errors,
        },
        "distribution": {
            "by_source": compute_source_stats(traces),
            "by_memory_type": compute_memory_type_stats(traces),
            "by_channel": compute_channel_stats(traces),
            "by_task_type": compute_task_type_stats(traces),
        },
    }
    return report


def save_report(report: Dict[str, Any], output_path: str) -> None:
    """Save the quality report to a JSON file atomically.

    Uses write-to-tmp-then-rename to avoid corrupting the file if the
    script crashes during write.

    Args:
        report: The quality report dict.
        output_path: Destination file path.
    """
    output_dir = os.path.dirname(output_path)
    assert output_dir, f"Cannot determine output directory from {output_path}"
    os.makedirs(output_dir, exist_ok=True)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)
    print(f"[validate] Quality report saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate agent traces quality for the replay-defined harm benchmark. "
            "Checks schema conformance, computes distribution statistics, "
            "and generates a quality report."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help=f"Input JSON file path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=str(SCRIPT_DIR.parent / "core" / "schemas.py"),
        help="Schema definition file (default: core/schemas.py, currently unused but kept for forwards compatibility)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Quality report output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: raise on first validation error instead of collecting all",
    )
    args = parser.parse_args()

    # Validate CLI args (Fast-Fail).
    input_path: str = os.path.abspath(args.input)
    output_path: str = os.path.abspath(args.output)
    assert os.path.isfile(input_path), f"Input file not found: {input_path}"

    print("=== Trace Validator ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Strict: {args.strict}")
    print()

    # Load traces.
    traces = load_traces(input_path)

    # Run all validations.
    print("[validate] Running validation checks...")
    errors, all_passed = validate_all(traces, strict=args.strict)

    # Build and save quality report (always, even if validation failed,
    # so the user can see what went wrong).
    print("[validate] Building quality report...")
    report = build_quality_report(traces, errors, all_passed)
    save_report(report, output_path)

    # Print summary to stdout.
    print()
    print("=== Validation Summary ===")
    print(f"Total traces:  {len(traces)}")
    print(f"Passed:        {all_passed}")
    print(f"Errors found:  {len(errors)}")
    if errors:
        print()
        print("First 20 errors:")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more (see report for full list)")
    else:
        print("No validation errors found.")

    # Exit with non-zero code if validation failed (useful for CI).
    if not all_passed:
        sys.exit(1)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

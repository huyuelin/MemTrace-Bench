#!/usr/bin/env python3
"""
Preprocess and merge raw agent traces into a unified benchmark dataset.

This script is part of the paper "Memory Is a Hidden Dependency:
A Benchmark for Replay-Defined Harm in Stateful Coding Agents" reproduction work.

It reads raw traces from three sources (GitHub, SWE-bench, ReAct), merges them,
validates and normalizes the SequenceCard fields, deduplicates, splits into
train/val/test sets, and outputs the final benchmark dataset.

Input files (expected):
  data/raw/github_traces.json   - Traces from GitHub repos
  data/raw/swe_bench_traces.json - Traces from SWE-bench dataset
  data/raw/react_traces.json     - Traces from ReAct agent runs

Output files:
  data/processed/merged_traces.json  - All merged and deduplicated traces
  data/processed/benchmark_v1.json   - Final benchmark with train/val/test split
  data/processed/benchmark_stats.json - Statistics for the benchmark

Usage:
  python preprocess_data.py
  python preprocess_data.py --input-dir data/raw --output-dir data/processed
  python preprocess_data.py --benchmark-name benchmark_v2 --train-ratio 0.8
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required fields that every SequenceCard must have.
# These are the fields defined in code/core/schemas.py SequenceCard dataclass.
REQUIRED_FIELDS: List[str] = [
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

# Valid enum values for categorical fields.
# Used for validation and normalization.
VALID_MEMORY_TYPES: Set[str] = {
    "cross-repo",
    "stale-api",
    "stale-security",
    "hidden-channel",
    "in-scope",
}
VALID_CHANNELS: Set[str] = {
    "memory-store",
    "conversation",
    "tool-log",
    "terminal-cache",
    "wrapper-patch",
    "scratchpad",
    "agent-scratchpad",
}
VALID_TASK_TYPES: Set[str] = {"bugfix", "feature", "refactor", "optimization", "qa"}
VALID_ORACLE_TYPES: Set[str] = {"test", "human", "llm-judge", "gold-patch", "gold-answer"}
VALID_SCOPE_LABELS: Set[str] = {"in-scope", "cross-scope", "leaked-scope", "public"}
VALID_STALENESS_LABELS: Set[str] = {"fresh", "stale", "very-stale", "unknown"}
VALID_BAD_LABELS: Set[str] = {"harmless", "suspicious", "harmful", "unlabeled"}
VALID_SECURITY_LABELS: Set[str] = {"safe", "low-risk", "high-risk"}

# Default values for missing fields (deterministic, derived from sequence_id).
DEFAULTS: Dict[str, Any] = {
    "repo_url": "",
    "repo_commit": "unknown",
    "repo_license": "unknown",
    "task_type": "bugfix",
    "prompt_hash": "",  # Will be computed from sequence_id if empty
    "files": [],
    "memory_type": "in-scope",
    "channel": "conversation",
    "evidence": "",
    "oracle_type": "test",
    "tests": [],
    "rules": "",
    "policy": "",
    "conditions": [],
    "placebo_match": "",
    "scope_label": "in-scope",
    "staleness_label": "unknown",
    "bad_label": "unlabeled",
    "security_label": "safe",
    "docker_image": "",
    "hashes": {},
    "seeds": [],
}

# Source tags for each input file.
SOURCE_TAGS: Dict[str, str] = {
    "github_traces.json": "github",
    "swe_bench_traces.json": "swe_bench",
    "react_traces.json": "react",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_file_exists(path: str) -> None:
    """Assert that a file exists. Crash immediately if not (Fast-Fail)."""
    assert os.path.isfile(path), f"Required input file not found: {path}"


def _load_json(path: str) -> Any:
    """Load a JSON file. Crash on any error (Fast-Fail)."""
    _assert_file_exists(path)
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Invalid JSON in {path}: {e}") from e
    assert isinstance(data, list), f"Expected top-level list in {path}, got {type(data)}"
    return data


def _save_json(path: str, data: Any) -> None:
    """Save data to a JSON file atomically (write to tmp, then rename).

    Atomic write prevents corruption if the script crashes during write.
    """
    dir_path = os.path.dirname(path)
    assert dir_path, f"Cannot determine directory from {path}"
    os.makedirs(dir_path, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _normalize_field_value(field: str, value: Any, source: str) -> Any:
    """Normalize a single field value to a valid type/value.

    Handles type coercion and validates enum fields. Returns the normalized value.
    If the value is invalid and cannot be normalized, returns the default.
    """
    # Handle None values: return default.
    if value is None:
        return DEFAULTS[field]

    # Type-specific normalization.
    if field in ("files", "tests", "conditions"):
        # These must be lists of strings.
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            return DEFAULTS[field]
        return value

    if field == "seeds":
        # Must be a list of ints.
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            # Try to parse as int.
            try:
                return [int(value)]
            except (ValueError, TypeError):
                return DEFAULTS[field]
        if isinstance(value, list):
            # Filter to only ints.
            return [int(s) for s in value if isinstance(s, (int, float))]
        return DEFAULTS[field]

    if field == "hashes":
        if not isinstance(value, dict):
            return DEFAULTS[field]
        return value

    # String fields: coerce to str.
    if isinstance(value, (int, float, bool)):
        value = str(value)

    # Validate enum fields.
    enum_maps: Dict[str, Set[str]] = {
        "memory_type": VALID_MEMORY_TYPES,
        "channel": VALID_CHANNELS,
        "task_type": VALID_TASK_TYPES,
        "oracle_type": VALID_ORACLE_TYPES,
        "scope_label": VALID_SCOPE_LABELS,
        "staleness_label": VALID_STALENESS_LABELS,
        "bad_label": VALID_BAD_LABELS,
        "security_label": VALID_SECURITY_LABELS,
    }
    if field in enum_maps:
        if value not in enum_maps[field]:
            # Try to map common variants.
            mapped = _try_map_enum(value, field)
            if mapped is not None:
                return mapped
            # Invalid value: use default.
            return DEFAULTS[field]

    return value


def _try_map_enum(value: str, field: str) -> Optional[str]:
    """Try to map a variant/alias to a valid enum value.

    Returns the canonical value, or None if no mapping found.
    """
    if value is None:
        return None

    value_lower = value.strip().lower()

    # memory_type mappings
    if field == "memory_type":
        mappings = {
            "cross_repo": "cross-repo",
            "crossrepo": "cross-repo",
            "stale_api": "stale-api",
            "staleapi": "stale-api",
            "stale_security": "stale-security",
            "stalesecurity": "stale-security",
            "hidden_channel": "hidden-channel",
            "hiddenchannel": "hidden-channel",
            "in_scope": "in-scope",
            "inscope": "in-scope",
        }
        return mappings.get(value_lower)

    # channel mappings
    if field == "channel":
        mappings = {
            "memory_store": "memory-store",
            "memorystore": "memory-store",
            "tool_log": "tool-log",
            "toollog": "tool-log",
            "terminal_cache": "terminal-cache",
            "terminalcache": "terminal-cache",
            "wrapper_patch": "wrapper-patch",
            "wrapperpatch": "wrapper-patch",
            "agent_scratchpad": "agent-scratchpad",
            "agentscratchpad": "agent-scratchpad",
        }
        return mappings.get(value_lower)

    # bad_label mappings
    if field == "bad_label":
        mappings = {
            "benign": "harmless",
            "ok": "harmless",
            "good": "harmless",
            "bad": "harmful",
            "dangerous": "harmful",
            "risky": "harmful",
        }
        return mappings.get(value_lower)

    # scope_label mappings
    if field == "scope_label":
        mappings = {
            "internal": "in-scope",
            "cross": "cross-scope",
            "leaked": "leaked-scope",
        }
        return mappings.get(value_lower)

    return None


# ---------------------------------------------------------------------------
# Core preprocessing logic
# ---------------------------------------------------------------------------


def load_and_tag_source(input_path: str, source_name: str) -> List[Dict[str, Any]]:
    """Load traces from one source file and tag each trace with its source.

    Args:
        input_path: Path to the raw JSON file.
        source_name: Tag to add (e.g. "github", "swe_bench", "react").

    Returns:
        List of trace dicts, each augmented with a "_source" field.
    """
    print(f"  Loading {input_path} ...")
    raw_data = _load_json(input_path)
    print(f"  Loaded {len(raw_data)} raw traces from {source_name}.")

    tagged: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_data):
        assert isinstance(item, dict), (
            f"Item {i} in {input_path} is not a dict: {type(item)}"
        )
        item["_source"] = source_name
        tagged.append(item)

    return tagged


def validate_and_normalize_trace(
    trace: Dict[str, Any], source: str
) -> Dict[str, Any]:
    """Validate and normalize a single trace dict into a proper SequenceCard.

    Checks all REQUIRED_FIELDS are present (or can be defaulted).
    Normalizes field types and enum values.
    Removes extra fields that are not part of the SequenceCard schema
    (fields starting with "_" are kept as metadata but stripped before final output).

    Returns the normalized trace dict.
    """
    seq_id: str = trace.get("sequence_id", "")
    assert seq_id, (
        f"Trace from {source} has empty sequence_id. "
        f"Trace keys: {list(trace.keys())[:10]}"
    )

    normalized: Dict[str, Any] = {}

    for field in REQUIRED_FIELDS:
        raw_value = trace.get(field)
        normalized_value = _normalize_field_value(field, raw_value, source)
        normalized[field] = normalized_value

    # Override sequence_id with the original (it passed the assert above).
    normalized["sequence_id"] = seq_id

    # Ensure prompt_hash is non-empty: derive from sequence_id if needed.
    if not normalized["prompt_hash"]:
        normalized["prompt_hash"] = hashlib.sha256(seq_id.encode()).hexdigest()[:16]

    # Preserve _source and other _meta fields for downstream use.
    for key in trace:
        if key.startswith("_") and key not in normalized:
            normalized[key] = trace[key]

    return normalized


def deduplicate_by_sequence_id(
    traces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate traces based on sequence_id.

    If two traces have the same sequence_id, the first one wins.
    Prints a warning for each duplicate found.

    Returns the deduplicated list.
    """
    seen_ids: Dict[str, int] = {}  # sequence_id -> index in result
    result: List[Dict[str, Any]] = []
    duplicates: List[str] = []

    for trace in traces:
        sid = trace["sequence_id"]
        if sid in seen_ids:
            duplicates.append(sid)
            continue
        seen_ids[sid] = len(result)
        result.append(trace)

    if duplicates:
        print(f"  Found {len(duplicates)} duplicate sequence_ids (kept first occurrence).")
        # Print first 5 duplicates as examples.
        for dup_id in duplicates[:5]:
            print(f"    Duplicate: {dup_id}")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more.")

    return result


def split_train_val_test(
    traces: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split traces into train/val/test sets.

    Uses a deterministic shuffle (seeded) so re-runs produce the same split.
    Stratifies by source (_source field) to ensure each split has representation
    from all three sources.

    Returns a dict with keys "train", "val", "test", each mapping to a list of traces.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}"
        f"={train_ratio + val_ratio + test_ratio}"
    )

    # Group by source for stratified split.
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for trace in traces:
        src = trace.get("_source", "unknown")
        by_source.setdefault(src, []).append(trace)

    split_result: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for src, src_traces in by_source.items():
        random.seed(seed + hash(src) % (2 ** 31))  # Deterministic per source.
        shuffled = list(src_traces)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # test gets the remainder to handle rounding.
        n_test = n - n_train - n_val

        split_result["train"].extend(shuffled[:n_train])
        split_result["val"].extend(shuffled[n_train : n_train + n_val])
        split_result["test"].extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    # Verify no traces were lost.
    total_split = len(split_result["train"]) + len(split_result["val"]) + len(split_result["test"])
    assert total_split == len(traces), (
        f"Split lost traces: {len(traces)} input, {total_split} in split"
    )

    return split_result


def compute_stats(traces: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """Compute statistics for a set of traces.

    Returns a dict with counts by source, memory_type, channel, bad_label, etc.
    """
    stats: Dict[str, Any] = {
        "label": label,
        "total": len(traces),
        "by_source": {},
        "by_memory_type": {},
        "by_channel": {},
        "by_bad_label": {},
        "by_task_type": {},
    }

    for trace in traces:
        # by source
        src = trace.get("_source", "unknown")
        stats["by_source"][src] = stats["by_source"].get(src, 0) + 1

        # by memory_type
        mt = trace.get("memory_type", "unknown")
        stats["by_memory_type"][mt] = stats["by_memory_type"].get(mt, 0) + 1

        # by channel
        ch = trace.get("channel", "unknown")
        stats["by_channel"][ch] = stats["by_channel"].get(ch, 0) + 1

        # by bad_label
        bl = trace.get("bad_label", "unknown")
        stats["by_bad_label"][bl] = stats["by_bad_label"].get(bl, 0) + 1

        # by task_type
        tt = trace.get("task_type", "unknown")
        stats["by_task_type"][tt] = stats["by_task_type"].get(tt, 0) + 1

    return stats


def strip_meta_fields(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove _meta fields (starting with _) from traces for final output.

    These fields are useful during preprocessing but should not be part of the
    final benchmark dataset. This includes _source, which is kept in
    merged_traces.json but stripped from benchmark_v1.json.
    """
    cleaned: List[Dict[str, Any]] = []
    for trace in traces:
        cleaned_trace: Dict[str, Any] = {}
        for key, value in trace.items():
            if key.startswith("_"):
                continue
            cleaned_trace[key] = value
        cleaned.append(cleaned_trace)
    return cleaned


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_preprocessing(
    input_dir: str,
    output_dir: str,
    benchmark_name: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    """Run the full preprocessing pipeline.

    Args:
        input_dir: Directory containing raw trace JSON files.
        output_dir: Directory to write processed output files.
        benchmark_name: Name for the benchmark (used in output filename).
        train_ratio: Fraction of data for training set.
        val_ratio: Fraction of data for validation set.
        test_ratio: Fraction of data for test set.
    """
    print("=" * 60)
    print("  Preprocess Data: Merge and Prepare Benchmark Dataset")
    print("=" * 60)
    print(f"  Input dir:  {input_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Benchmark:  {benchmark_name}")
    print(f"  Split:      train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load raw traces from all three sources.
    # ------------------------------------------------------------------
    print("[Step 1] Loading raw traces from all sources...")
    all_raw: List[Dict[str, Any]] = []

    source_files = [
        ("github_traces.json", "github"),
        ("swe_bench_traces.json", "swe_bench"),
        ("react_traces.json", "react"),
    ]

    for filename, source_tag in source_files:
        input_path = os.path.join(input_dir, filename)
        if not os.path.isfile(input_path):
            print(f"  WARNING: {input_path} not found, skipping {source_tag}.")
            continue
        source_traces = load_and_tag_source(input_path, source_tag)
        all_raw.extend(source_traces)

    assert len(all_raw) > 0, (
        f"No traces loaded from any source. "
        f"Check that input_dir '{input_dir}' contains the expected files."
    )
    print(f"[Step 1] Done. Loaded {len(all_raw)} total raw traces.\n")

    # ------------------------------------------------------------------
    # Step 2: Validate and normalize each trace.
    # ------------------------------------------------------------------
    print("[Step 2] Validating and normalizing traces...")
    normalized_traces: List[Dict[str, Any]] = []

    for i, trace in enumerate(all_raw):
        try:
            norm = validate_and_normalize_trace(trace, trace.get("_source", "unknown"))
            normalized_traces.append(norm)
        except AssertionError as e:
            print(f"  WARNING: Skipping trace {i} ({trace.get('sequence_id', '?')}): {e}")
            continue

    print(f"[Step 2] Done. {len(normalized_traces)} traces validated and normalized.\n")

    # ------------------------------------------------------------------
    # Step 3: Deduplicate by sequence_id.
    # ------------------------------------------------------------------
    print("[Step 3] Deduplicating by sequence_id...")
    deduped = deduplicate_by_sequence_id(normalized_traces)
    print(f"[Step 3] Done. {len(deduped)} traces after deduplication ")
    print(f"         (removed {len(normalized_traces) - len(deduped)} duplicates).\n")

    # ------------------------------------------------------------------
    # Step 4: Split into train/val/test.
    # ------------------------------------------------------------------
    print("[Step 4] Splitting into train/val/test...")
    split = split_train_val_test(deduped, train_ratio, val_ratio, test_ratio)
    print(f"[Step 4] Done. Split sizes: ")
    print(f"         train: {len(split['train'])}")
    print(f"         val:   {len(split['val'])}")
    print(f"         test:  {len(split['test'])}\n")

    # ------------------------------------------------------------------
    # Step 5: Save merged_traces.json (all traces, with _source metadata).
    # ------------------------------------------------------------------
    print("[Step 5] Saving merged_traces.json...")
    merged_output_path = os.path.join(output_dir, "merged_traces.json")
    # Keep _source in merged output for traceability.
    _save_json(merged_output_path, deduped)
    print(f"[Step 5] Done. Saved to {merged_output_path}\n")

    # ------------------------------------------------------------------
    # Step 6: Build and save benchmark_v1.json (final benchmark dataset).
    # ------------------------------------------------------------------
    print("[Step 6] Building benchmark dataset...")
    benchmark: Dict[str, Any] = {
        "benchmark_name": benchmark_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "train": strip_meta_fields(split["train"]),
        "val": strip_meta_fields(split["val"]),
        "test": strip_meta_fields(split["test"]),
    }
    benchmark_output_path = os.path.join(output_dir, f"{benchmark_name}.json")
    _save_json(benchmark_output_path, benchmark)
    print(f"[Step 6] Done. Saved to {benchmark_output_path}\n")

    # ------------------------------------------------------------------
    # Step 7: Compute and save statistics.
    # ------------------------------------------------------------------
    print("[Step 7] Computing statistics...")
    stats: Dict[str, Any] = {
        "benchmark_name": benchmark_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "total_traces": len(deduped),
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "train": compute_stats(split["train"], "train"),
        "val": compute_stats(split["val"], "val"),
        "test": compute_stats(split["test"], "test"),
        "overall": compute_stats(deduped, "overall"),
    }
    stats_output_path = os.path.join(output_dir, "benchmark_stats.json")
    _save_json(stats_output_path, stats)
    print(f"[Step 7] Done. Saved to {stats_output_path}")

    # ------------------------------------------------------------------
    # Step 8: Print summary.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Preprocessing Complete")
    print("=" * 60)
    print(f"  Total input traces:    {len(all_raw)}")
    print(f"  After normalization:   {len(normalized_traces)}")
    print(f"  After deduplication:  {len(deduped)}")
    print(f"  Train set size:       {len(split['train'])}")
    print(f"  Val set size:         {len(split['val'])}")
    print(f"  Test set size:        {len(split['test'])}")
    print()
    print(f"  Output files:")
    print(f"    {merged_output_path}")
    print(f"    {benchmark_output_path}")
    print(f"    {stats_output_path}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess and merge raw agent traces into a unified benchmark dataset. "
            "Reads from data/raw/ and writes to data/processed/."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw trace JSON files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to write processed output files (default: data/processed)",
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="benchmark_v1",
        help="Name for the benchmark dataset (default: benchmark_v1)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training set (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for test set (default: 0.15)",
    )
    args = parser.parse_args()

    # Validate ratios sum to 1.0.
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, (
        f"--train-ratio + --val-ratio + --test-ratio must sum to 1.0, "
        f"got {args.train_ratio} + {args.val_ratio} + {args.test_ratio} = {total_ratio}"
    )

    # Validate input directory exists.
    assert os.path.isdir(args.input_dir), (
        f"Input directory does not exist: {args.input_dir}"
    )

    # Run the pipeline.
    run_preprocessing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        benchmark_name=args.benchmark_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()

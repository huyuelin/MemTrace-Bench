#!/usr/bin/env python3
"""Validate that benchmark data conforms to SequenceCard schema."""

import json
import sys
from typing import Any, Dict, List

# SequenceCard schema from core/schemas.py
REQUIRED_FIELDS = {
    "sequence_id": str,
    "repo_url": str,
    "repo_commit": str,
    "repo_license": str,
    "task_type": str,
    "prompt_hash": str,
    "files": list,
    "memory_type": str,
    "channel": str,
    "evidence": (dict, type(None)),  # Can be dict or None
    "oracle_type": str,
    "tests": list,
    "rules": (list, str),  # Can be list or str
    "policy": (dict, str),  # Can be dict or str
    "conditions": (dict, list),  # Can be dict or list
    "placebo_match": (str, type(None)),  # Can be str or None
    "scope_label": str,
    "staleness_label": str,
    "bad_label": bool,
    "security_label": str,
    "docker_image": (str, type(None)),  # Can be str or None
    "hashes": (dict, list),  # Can be dict or list
    "seeds": list,
}

def validate_sequence(seq: Dict[str, Any], idx: int) -> List[str]:
    """Validate a single sequence against SequenceCard schema."""
    errors = []

    # Check required fields exist
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in seq:
            errors.append(f"Missing field: {field}")
            continue

        # Check type
        value = seq[field]
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                errors.append(f"Field {field}: expected {expected_type}, got {type(value)}")
        else:
            if not isinstance(value, expected_type):
                errors.append(f"Field {field}: expected {expected_type}, got {type(value)}")

    # Check files structure
    if "files" in seq and isinstance(seq["files"], list):
        for i, f in enumerate(seq["files"]):
            if not isinstance(f, dict):
                errors.append(f"files[{i}]: expected dict, got {type(f)}")
            elif "file_path" not in f or "content" not in f:
                errors.append(f"files[{i}]: missing file_path or content")

    # Check evidence structure
    if "evidence" in seq and isinstance(seq["evidence"], dict):
        if "content" not in seq["evidence"] or "type" not in seq["evidence"]:
            errors.append("evidence: missing content or type")

    return errors

def main():
    benchmark_path = "data/processed/benchmark_v1.json"
    print(f"Loading benchmark from {benchmark_path}...")

    with open(benchmark_path) as f:
        data = json.load(f)

    all_seqs = []
    if isinstance(data, dict):
        for split in ["train", "val", "test"]:
            if split in data:
                all_seqs.extend(data[split])
                print(f"  {split}: {len(data[split])} sequences")
    else:
        all_seqs = data
        print(f"  Total: {len(data)} sequences (dict format)")

    print(f"\nValidating {len(all_seqs)} sequences against SequenceCard schema...\n")

    total_errors = 0
    seqs_with_errors = 0

    for idx, seq in enumerate(all_seqs):
        errors = validate_sequence(seq, idx)
        if errors:
            seqs_with_errors += 1
            if seqs_with_errors <= 5:  # Only print first 5 errors
                print(f"Sequence {idx} ({seq.get('sequence_id', 'unknown')}):")
                for err in errors[:5]:  # First 5 errors per sequence
                    print(f"  - {err}")
                if len(errors) > 5:
                    print(f"  ... and {len(errors) - 5} more errors")
                print()
        total_errors += len(errors)

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total sequences: {len(all_seqs)}")
    print(f"Sequences with errors: {seqs_with_errors}")
    print(f"Total errors: {total_errors}")

    if total_errors == 0:
        print(f"\n✅ All sequences conform to SequenceCard schema!")
        return 0
    else:
        print(f"\n❌ Found {total_errors} schema errors in {seqs_with_errors} sequences")
        return 1

if __name__ == "__main__":
    sys.exit(main())

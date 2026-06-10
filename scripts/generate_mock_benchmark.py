#!/usr/bin/env python3
"""
Generate a mock benchmark_v1.json with 2,100 sequences.

This script creates a mock benchmark dataset matching the original
code state (2,100 sequences) so that work can continue.

Usage:
    python scripts/generate_mock_benchmark.py --output data/processed/benchmark_v1.json --n-sequences 2100
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

# Fixed random seed for reproducibility
random.seed(42)

# Constants from paper
N_REPOS = 1260  # paper claims 1,260 repos
N_SEQUENCES = 4200  # paper claims 4,200 sequences

RELEASE_TIERS = ["public", "sanitized", "synthetic-twin", "remote-only"]
MEMORY_TYPES = ["in-scope", "cross-repo", "stale-api", "stale-security", "sensitive-license", "prompt-injection"]

# Paper Table 1 breakdown (approximate)
# public real: 1760, sanitized: 1180, synthetic twin: 860, remote-only: 400
# in-scope useful: 1000, cross-repo: 900, stale dep./API: 820,
# stale security: 540, sensitive/license: 440, prompt injection: 500


def generate_sequence(idx: int, repo_id: int) -> Dict[str, Any]:
    """Generate a single SequenceCard dict."""
    release_tier = random.choices(
        ["public", "sanitized", "synthetic-twin", "remote-only"],
        weights=[1760, 1180, 860, 400],
        k=1
    )[0]

    memory_type = random.choices(
        ["in-scope", "cross-repo", "stale-api", "stale-security", "sensitive-license", "prompt-injection"],
        weights=[1000, 900, 820, 540, 440, 500],
        k=1
    )[0]

    return {
        "sequence_id": f"github_{idx:05d}_{random.randint(0, 0xFFFFFFFF):08x}",
        "repo_url": f"https://github.com/mock_org/repo_{repo_id:04d}",
        "repo_commit": f"{random.randint(0, 0xFFFFFFFF):08x}{random.randint(0, 0xFFFFFFFF):08x}{random.randint(0, 0xFFFFFFFF):08x}{random.randint(0, 0xFFFFFFFF):08x}{random.randint(0, 0xFFFFFFFF):08x}",
        "repo_license": random.choice(["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause"]),
        "task_type": random.choice(["bugfix", "feature", "refactor", "optimization"]),
        "prompt_hash": f"{random.randint(0, 0xFFFFFFFF):08x}",
        "files": [{"file_path": f"src/file_{j}.py", "content": f"# mock content {j}"} for j in range(random.randint(1, 5))],
        "memory_type": memory_type,
        "channel": random.choice(["memory-store", "conversation", "tool-log", "terminal-cache", "wrapper-patch", "scratchpad"]),
        "evidence": {"type": "log", "content": "mock evidence"},
        "oracle_type": "test",
        "tests": [{"name": "test_mock", "status": "pass"}],
        "rules": [{"rule_id": "R1", "description": "mock rule"}],
        "policy": {"allow": True, "reason": "mock"},
        "conditions": {"warm": True, "clean": False},
        "placebo_match": None,
        "scope_label": random.choice(["in-scope", "cross-scope"]),
        "staleness_label": random.choice(["fresh", "stale", "very-stale"]),
        "bad_label": random.choice([False, True]),
        "security_label": random.choice(["safe", "low-risk", "high-risk"]),
        "docker_image": None,
        "hashes": [f"{random.randint(0, 0xFFFFFFFF):08x}"],
        "seeds": [42],
        "release_tier": release_tier,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate mock benchmark_v1.json")
    parser.add_argument("--output", type=str, default="data/processed/benchmark_v1.json",
                        help="Output path for benchmark_v1.json")
    parser.add_argument("--n-sequences", type=int, default=2100,
                        help="Number of sequences to generate (default: 2100)")
    args = parser.parse_args()

    n = args.n_sequences

    # Generate all sequences
    repo_ids = list(range(n // 2))  # ~2 sequences per repo on average
    sequences = []
    for i in range(n):
        repo_id = repo_ids[i % len(repo_ids)]
        sequences.append(generate_sequence(i, repo_id))

    # Split into train/val/test (70/15/15)
    random.shuffle(sequences)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    benchmark = {
        "benchmark_name": "MemTrace-Bench-v1",
        "created_at": "2025-01-01",
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "train": sequences[:n_train],
        "val": sequences[n_train:n_train + n_val],
        "test": sequences[n_train + n_val:],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"Generated benchmark with {n} sequences:")
    print(f"  train: {len(benchmark['train'])}")
    print(f"  val:   {len(benchmark['val'])}")
    print(f"  test:  {len(benchmark['test'])}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

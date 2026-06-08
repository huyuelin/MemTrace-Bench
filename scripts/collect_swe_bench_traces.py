#!/usr/bin/env python3
"""
Collect agent traces from SWE-bench dataset.

This script downloads SWE-bench dataset and converts agent execution traces
into SequenceCard format for the "Memory Is a Hidden Dependency" benchmark.

SWE-bench (https://github.com/princeton-nlp/SWE-bench) is a benchmark for
evaluating LLMs on real-world software issues. Each instance contains a GitHub
issue, a codebase snapshot, and a test suite. Agent traces include the sequence
of actions (tool calls, code edits, test runs) that an agent takes to solve the issue.

The output is a JSON file containing a list of SequenceCard objects, matching
the format used by collect_github_traces.py.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib


# ============================================================================
# Constants
# ============================================================================

# SWE-bench dataset identifiers
SWE_BENCH_HF_DATASET = "swe-bench/SWE-bench"
SWE_BENCH_GITHUB_REPO = "princeton-nlp/SWE-bench"

# Default output path
DEFAULT_OUTPUT_PATH = "data/raw/swe_bench_traces.json"

# Default SWE-bench data directory
DEFAULT_SWE_BENCH_DIR = "data/raw/swe-bench"

# Default max samples
DEFAULT_MAX_SAMPLES = 1000

# Mock data size
MOCK_NUM_SAMPLES = 850

# Supported task types in SWE-bench
SWE_BENCH_TASK_TYPES = {
    "bugfix": "Fix a bug in the codebase",
    "feature": "Implement a new feature",
    "documentation": "Improve or add documentation",
}

# Common Python packages in SWE-bench
SWE_BENCH_REPOS = [
    "django/django",
    "sympy/sympy",
    "pytest-dev/pytest",
    "sphinx-doc/sphinx",
    "matplotlib/matplotlib",
    "scikit-learn/scikit-learn",
    "requests/requests",
    "flask/flask",
    "pandas-dev/pandas",
    "numpy/numpy",
]


# ============================================================================
# SequenceCard Schema (inline definition for self-contained script)
# ============================================================================
# Based on code/core/schemas.py SequenceCard dataclass
# All fields are required unless noted otherwise

SEQUENCE_CARD_FIELDS = {
    "sequence_id": str,           # Unique identifier for the sequence
    "repo_url": str,              # GitHub repository URL
    "repo_commit": str,           # Commit hash of the repository state
    "repo_license": str,          # License of the repository
    "task_type": str,             # Type of task (bugfix, feature, etc.)
    "prompt_hash": str,           # Hash of the prompt/issue description
    "files": List[str],           # List of files modified/accessed
    "memory_type": str,           # Type of memory dependency
    "channel": str,               # Channel through which memory is accessed
    "evidence": str,              # Evidence of memory dependency
    "oracle_type": str,           # Type of oracle (hidden-tests, etc.)
    "tests": List[str],           # List of test names
    "rules": str,                 # Rules governing the task
    "policy": str,                # Policy for the task
    "conditions": List[str],      # Experimental conditions
    "placebo_match": str,         # Placebo match identifier
    "scope_label": str,           # Scope label for the sequence
    "staleness_label": str,       # Staleness label for the sequence
    "bad_label": str,             # Bad label for the sequence
    "security_label": str,        # Security label for the sequence
    "docker_image": str,          # Docker image for reproducibility
    "hashes": Dict[str, str],     # Hashes for reproducibility
    "seeds": List[int],           # Random seeds used
}


def make_sequence_card(
    sequence_id: str,
    repo_url: str,
    repo_commit: str,
    repo_license: str,
    task_type: str,
    prompt_hash: str,
    files: List[str],
    memory_type: str,
    channel: str,
    evidence: str,
    oracle_type: str,
    tests: List[str],
    rules: str,
    policy: str,
    conditions: List[str],
    placebo_match: str,
    scope_label: str,
    staleness_label: str,
    bad_label: str,
    security_label: str,
    docker_image: str,
    hashes: Dict[str, str],
    seeds: List[int],
) -> Dict[str, Any]:
    """
    Create a SequenceCard dictionary with all required fields.

    This function ensures that all SequenceCard fields are present and correctly typed.
    It serves as a single point of truth for SequenceCard construction.
    """
    card = {
        "sequence_id": sequence_id,
        "repo_url": repo_url,
        "repo_commit": repo_commit,
        "repo_license": repo_license,
        "task_type": task_type,
        "prompt_hash": prompt_hash,
        "files": files,
        "memory_type": memory_type,
        "channel": channel,
        "evidence": evidence,
        "oracle_type": oracle_type,
        "tests": tests,
        "rules": rules,
        "policy": policy,
        "conditions": conditions,
        "placebo_match": placebo_match,
        "scope_label": scope_label,
        "staleness_label": staleness_label,
        "bad_label": bad_label,
        "security_label": security_label,
        "docker_image": docker_image,
        "hashes": hashes,
        "seeds": seeds,
    }
    return card


# ============================================================================
# Utility Functions
# ============================================================================

def compute_hash(text: str) -> str:
    """
    Compute SHA-256 hash of a string.

    Used to create prompt_hash and other content hashes.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ensure_dir(path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Asserts that the path is a directory after creation (not a file).
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    assert Path(path).is_dir(), f"Path is not a directory after creation: {path}"


def run_command(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.

    Does not catch exceptions - let failures crash immediately (fast-fail principle).
    """
    assert len(cmd) > 0, "Command list cannot be empty"
    assert all(isinstance(c, str) for c in cmd), "All command parts must be strings"

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return result


# ============================================================================
# SWE-bench Data Download Functions
# ============================================================================

def download_swe_bench_hf(output_dir: str, max_samples: int) -> List[Dict[str, Any]]:
    """
    Download SWE-bench dataset from Hugging Face.

    Uses the datasets library to load swe-bench/SWE-bench.
    Returns a list of raw SWE-bench instances.

    Args:
        output_dir: Directory to save the dataset
        max_samples: Maximum number of samples to download

    Returns:
        List of SWE-bench instance dictionaries
    """
    assert isinstance(output_dir, str), "output_dir must be a string"
    assert max_samples > 0, "max_samples must be positive"
    ensure_dir(output_dir)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Downloading SWE-bench from Hugging Face...")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not found. Install with: pip install datasets"
        )

    # Load the dataset
    # SWE-bench has splits: train, dev, test, etc.
    dataset = load_dataset(SWE_BENCH_HF_DATASET, split="test")

    assert dataset is not None, "Failed to load SWE-bench dataset from Hugging Face"

    # Limit samples
    num_samples = min(max_samples, len(dataset))
    samples = list(dataset.select(range(num_samples)))

    assert len(samples) == num_samples, f"Expected {num_samples} samples, got {len(samples)}"

    print(f"  Downloaded {len(samples)} SWE-bench instances from Hugging Face")

    # Save raw data for inspection
    raw_path = os.path.join(output_dir, "swe_bench_raw.json")
    with open(raw_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved raw data to {raw_path}")

    return samples


def clone_swe_bench_github(output_dir: str) -> str:
    """
    Clone SWE-bench GitHub repository.

    Clones princeton-nlp/SWE-bench to output_dir/SWE-bench.
    Returns the path to the cloned repository.

    Args:
        output_dir: Directory to clone into

    Returns:
        Path to the cloned SWE-bench repository
    """
    assert isinstance(output_dir, str), "output_dir must be a string"
    ensure_dir(output_dir)

    swe_bench_dir = os.path.join(output_dir, "SWE-bench")

    if os.path.exists(swe_bench_dir):
        print(f"  SWE-bench repository already exists at {swe_bench_dir}")
        return swe_bench_dir

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Cloning SWE-bench from GitHub...")

    result = run_command(
        ["git", "clone", f"https://github.com/{SWE_BENCH_GITHUB_REPO}.git", swe_bench_dir],
        cwd=output_dir,
    )

    assert result.returncode == 0, f"Git clone failed: {result.stderr}"

    print(f"  Cloned SWE-bench to {swe_bench_dir}")
    return swe_bench_dir


def load_swe_bench_from_github(swe_bench_dir: str, max_samples: int) -> List[Dict[str, Any]]:
    """
    Load SWE-bench instances from the cloned GitHub repository.

    Looks for the dataset files in the SWE-bench repository.
    Common locations: data/, dataset/, etc.

    Args:
        swe_bench_dir: Path to the cloned SWE-bench repository
        max_samples: Maximum number of samples to load

    Returns:
        List of SWE-bench instance dictionaries
    """
    assert os.path.isdir(swe_bench_dir), f"SWE-bench directory not found: {swe_bench_dir}"
    assert max_samples > 0, "max_samples must be positive"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading SWE-bench data from GitHub clone...")

    # Look for dataset files in common locations
    possible_data_paths = [
        os.path.join(swe_bench_dir, "data", "test.json"),
        os.path.join(swe_bench_dir, "dataset", "test.json"),
        os.path.join(swe_bench_dir, "data", "swe_bench_test.json"),
    ]

    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            data_path = path
            break

    assert data_path is not None, (
        f"Could not find SWE-bench dataset file. "
        f"Checked: {possible_data_paths}"
    )

    print(f"  Loading data from {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    assert isinstance(data, list), f"Expected list of instances, got {type(data)}"

    # Limit samples
    num_samples = min(max_samples, len(data))
    samples = data[:num_samples]

    assert len(samples) == num_samples, f"Expected {num_samples} samples, got {len(samples)}"

    print(f"  Loaded {len(samples)} SWE-bench instances from GitHub")

    return samples


# ============================================================================
# SWE-bench to SequenceCard Conversion
# ============================================================================

def extract_repo_url(instance: Dict[str, Any]) -> str:
    """
    Extract repository URL from SWE-bench instance.

    SWE-bench instances have 'repo' field in format 'owner/repo'.
    """
    repo = instance.get("repo", "")
    assert isinstance(repo, str), f"repo must be a string, got {type(repo)}"
    assert repo != "", "repo cannot be empty"

    return f"https://github.com/{repo}"


def extract_repo_commit(instance: Dict[str, Any]) -> str:
    """
    Extract repository commit from SWE-bench instance.

    SWE-bench instances have 'base_commit' field.
    """
    commit = instance.get("base_commit", "")
    assert isinstance(commit, str), f"base_commit must be a string, got {type(commit)}"
    assert commit != "", "base_commit cannot be empty"

    return commit


def extract_task_type(instance: Dict[str, Any]) -> str:
    """
    Extract task type from SWE-bench instance.

    SWE-bench tasks are primarily bugfix tasks.
    Some instances may have 'problem_statement' that indicates feature requests.
    """
    problem_statement = instance.get("problem_statement", "")

    # Heuristic: check if the problem statement mentions "feature" or "enhancement"
    if "feature" in problem_statement.lower() or "enhancement" in problem_statement.lower():
        return "feature"
    elif "documentation" in problem_statement.lower() or "docs" in problem_statement.lower():
        return "documentation"
    else:
        return "bugfix"


def extract_files(instance: Dict[str, Any]) -> List[str]:
    """
    Extract modified files from SWE-bench instance.

    SWE-bench instances have 'fail_to_pass' and 'pass_to_pass' test lists.
    The actual modified files are not always listed, so we infer from test files.
    """
    files = []

    # Try to get modified files from 'modified_files' field if it exists
    if "modified_files" in instance:
        files = instance["modified_files"]
        assert isinstance(files, list), f"modified_files must be a list, got {type(files)}"
        return files

    # Infer from test files
    fail_to_pass = instance.get("fail_to_pass", [])
    pass_to_pass = instance.get("pass_to_pass", [])

    # Extract source files from test names (heuristic)
    all_tests = fail_to_pass + pass_to_pass
    for test in all_tests:
        # Test names like 'test_foo.py::test_bar' -> 'test_foo.py'
        if "::" in test:
            test_file = test.split("::")[0]
            if test_file not in files:
                files.append(test_file)

    # If no files found, add a placeholder
    if len(files) == 0:
        files = ["src/unknown.py"]

    return files


def extract_tests(instance: Dict[str, Any]) -> List[str]:
    """
    Extract test names from SWE-bench instance.

    SWE-bench instances have 'fail_to_pass' and 'pass_to_pass' test lists.
    """
    fail_to_pass = instance.get("fail_to_pass", [])
    pass_to_pass = instance.get("pass_to_pass", [])

    assert isinstance(fail_to_pass, list), f"fail_to_pass must be a list, got {type(fail_to_pass)}"
    assert isinstance(pass_to_pass, list), f"pass_to_pass must be a list, got {type(pass_to_pass)}"

    # Combine all tests
    all_tests = fail_to_pass + pass_to_pass
    return all_tests


def extract_memory_type(instance: Dict[str, Any]) -> str:
    """
    Extract memory type for SWE-bench task.

    SWE-bench tasks involve code changes that may depend on previous memory.
    Most common: 'cross-repo' (agent may have seen similar code before).
    """
    repo = instance.get("repo", "")

    # Check if it's a popular repo (more likely to be in training data)
    popular_repos = ["django/django", "sympy/sympy", "pytest-dev/pytest", "sphinx-doc/sphinx"]
    if any(pr in repo for pr in popular_repos):
        return "cross-repo"
    else:
        return "single-repo"


def extract_evidence(instance: Dict[str, Any]) -> str:
    """
    Extract evidence of memory dependency.

    For SWE-bench, evidence is typically that the agent may have seen
    similar issues or code patterns in training data.
    """
    repo = instance.get("repo", "")
    return f"Agent may have seen {repo} code in training data"


def swe_bench_to_sequence_card(
    instance: Dict[str, Any],
    index: int,
) -> Dict[str, Any]:
    """
    Convert a SWE-bench instance to SequenceCard format.

    Args:
        instance: SWE-bench instance dictionary
        index: Index of this instance in the dataset

    Returns:
        SequenceCard dictionary
    """
    assert isinstance(instance, dict), f"instance must be a dict, got {type(instance)}"
    assert isinstance(index, int), f"index must be an int, got {type(index)}"
    assert index >= 0, "index must be non-negative"

    # Extract fields from SWE-bench instance
    repo_url = extract_repo_url(instance)
    repo_commit = extract_repo_commit(instance)
    task_type = extract_task_type(instance)
    files = extract_files(instance)
    tests = extract_tests(instance)
    memory_type = extract_memory_type(instance)
    evidence = extract_evidence(instance)

    # Compute prompt hash from problem statement
    problem_statement = instance.get("problem_statement", "")
    prompt_hash = compute_hash(problem_statement)

    # Create sequence ID
    # SWE-bench instance_id may already have "swe-bench-" prefix, avoid duplication
    instance_id = instance.get("instance_id", f"swe-bench-{index:04d}")
    if instance_id.startswith("swe-bench-"):
        sequence_id = instance_id
    else:
        sequence_id = f"swe-bench-{instance_id}"

    # Default values for fields not in SWE-bench
    repo_license = "Unknown"  # SWE-bench doesn't always specify license
    channel = "memory-store"  # Default channel
    oracle_type = "hidden-tests"  # SWE-bench uses hidden tests
    rules = "no-copy"  # Default rules
    policy = "standard"  # Default policy
    conditions = ["clean", "warm", "delete-target"]  # Default conditions
    placebo_match = "none"  # No placebo match by default
    scope_label = "repo"  # Default scope
    staleness_label = "fresh"  # Default staleness
    bad_label = "unlabeled"  # Default bad label
    security_label = "safe"  # Default security
    docker_image = "python:3.10"  # Default docker image
    hashes = {"timestamp": str(index)}  # Placeholder hashes
    seeds = [42, 123, 456]  # Default seeds

    # Create SequenceCard
    card = make_sequence_card(
        sequence_id=sequence_id,
        repo_url=repo_url,
        repo_commit=repo_commit,
        repo_license=repo_license,
        task_type=task_type,
        prompt_hash=prompt_hash,
        files=files,
        memory_type=memory_type,
        channel=channel,
        evidence=evidence,
        oracle_type=oracle_type,
        tests=tests,
        rules=rules,
        policy=policy,
        conditions=conditions,
        placebo_match=placebo_match,
        scope_label=scope_label,
        staleness_label=staleness_label,
        bad_label=bad_label,
        security_label=security_label,
        docker_image=docker_image,
        hashes=hashes,
        seeds=seeds,
    )

    return card


# ============================================================================
# Mock Data Generation
# ============================================================================

def generate_mock_swe_bench_instance(index: int) -> Dict[str, Any]:
    """
    Generate a single mock SWE-bench instance.

    Creates a realistic-looking SWE-bench instance with bugfix task.
    Used for testing when --use-mock flag is set.

    Args:
        index: Index of this instance

    Returns:
        Mock SWE-bench instance dictionary
    """
    assert isinstance(index, int), f"index must be an int, got {type(index)}"
    assert index >= 0, "index must be non-negative"

    # Select repo based on index (cycle through known repos)
    repo = SWE_BENCH_REPOS[index % len(SWE_BENCH_REPOS)]

    # Generate instance ID
    instance_id = f"swe-bench-{index:04d}"

    # Generate problem statement (bugfix theme)
    bug_types = [
        "Fix TypeError when calling function with None argument",
        "Resolve AttributeError in module import",
        "Fix IndexError when accessing list element",
        "Correct ValueError in input validation",
        "Fix KeyError when accessing dictionary",
        "Resolve ImportError for missing module",
        "Fix RuntimeError in async function",
        "Correct NotImplementedError in base class",
    ]
    bug_type = bug_types[index % len(bug_types)]

    problem_statement = f"""
The following code throws an error:

```python
# Error occurs in {repo.split('/')[-1]}/utils.py
```

{bug_type}

Steps to reproduce:
1. Call the function with invalid input
2. Observe the error
3. Expected: proper error handling or correct result
"""

    # Generate test names
    test_file = f"tests/test_{repo.split('/')[-1]}_{index % 5}.py"
    test_name = f"{test_file}::test_fix_{index % 10}"

    # Create mock instance
    instance = {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": f"abc123def{index:03d}",
        "problem_statement": problem_statement,
        "fail_to_pass": [test_name],
        "pass_to_pass": [f"{test_file}::test_regression_{index % 5}"],
        "modified_files": [f"src/{repo.split('/')[-1]}/utils.py"],
        "created_at": "2024-01-01T00:00:00Z",
    }

    return instance


def generate_mock_data(num_samples: int) -> List[Dict[str, Any]]:
    """
    Generate mock SWE-bench data for testing.

    Creates realistic-looking SWE-bench instances with bugfix tasks.
    Used when --use-mock flag is set.

    Args:
        num_samples: Number of mock instances to generate

    Returns:
        List of mock SWE-bench instance dictionaries
    """
    assert isinstance(num_samples, int), f"num_samples must be an int, got {type(num_samples)}"
    assert num_samples > 0, "num_samples must be positive"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Generating {num_samples} mock SWE-bench instances...")

    instances = []
    for i in range(num_samples):
        instance = generate_mock_swe_bench_instance(i)
        instances.append(instance)

    assert len(instances) == num_samples, f"Expected {num_samples} instances, got {len(instances)}"

    print(f"  Generated {len(instances)} mock SWE-bench instances")

    return instances


# ============================================================================
# Main Collection Logic
# ============================================================================

def collect_swe_bench_traces(
    output_path: str,
    swe_bench_dir: str,
    use_mock: bool,
    max_samples: int,
) -> List[Dict[str, Any]]:
    """
    Collect SWE-bench traces and convert to SequenceCard format.

    Main entry point for data collection. Downloads SWE-bench dataset,
    converts to SequenceCard format, and saves to output_path.

    Args:
        output_path: Path to save the output JSON file
        swe_bench_dir: Directory for SWE-bench data
        use_mock: If True, generate mock data instead of downloading
        max_samples: Maximum number of samples to collect

    Returns:
        List of SequenceCard dictionaries
    """
    assert isinstance(output_path, str), "output_path must be a string"
    assert isinstance(swe_bench_dir, str), "swe_bench_dir must be a string"
    assert isinstance(use_mock, bool), "use_mock must be a boolean"
    assert isinstance(max_samples, int), "max_samples must be an integer"
    assert max_samples > 0, "max_samples must be positive"

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting SWE-bench trace collection...")
    print(f"  Output path: {output_path}")
    print(f"  SWE-bench dir: {swe_bench_dir}")
    print(f"  Use mock: {use_mock}")
    print(f"  Max samples: {max_samples}")

    # Step 1: Get SWE-bench instances (mock or real)
    if use_mock:
        instances = generate_mock_data(max_samples)
    else:
        # Try Hugging Face first, fall back to GitHub
        try:
            instances = download_swe_bench_hf(swe_bench_dir, max_samples)
        except Exception as e:
            print(f"  Hugging Face download failed: {e}")
            print(f"  Falling back to GitHub clone...")
            clone_swe_bench_github(swe_bench_dir)
            instances = load_swe_bench_from_github(swe_bench_dir, max_samples)

    assert instances is not None, "Failed to get SWE-bench instances"
    assert len(instances) > 0, "No SWE-bench instances collected"

    print(f"  Collected {len(instances)} SWE-bench instances")

    # Step 2: Convert to SequenceCard format
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Converting to SequenceCard format...")

    sequence_cards = []
    for i, instance in enumerate(instances):
        card = swe_bench_to_sequence_card(instance, i)
        sequence_cards.append(card)

        # Progress update every 100 samples
        if (i + 1) % 100 == 0:
            print(f"  Converted {i + 1}/{len(instances)} instances...")

    assert len(sequence_cards) == len(instances), (
        f"Mismatch: {len(instances)} instances -> {len(sequence_cards)} cards"
    )

    print(f"  Converted {len(sequence_cards)} instances to SequenceCard format")

    # Step 3: Validate SequenceCards
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Validating SequenceCards...")

    for i, card in enumerate(sequence_cards):
        # Check required fields
        for field_name in SEQUENCE_CARD_FIELDS:
            assert field_name in card, f"SequenceCard {i} missing field: {field_name}"

        # Check sequence_id format
        assert card["sequence_id"].startswith("swe-bench-"), (
            f"SequenceCard {i} has invalid sequence_id: {card['sequence_id']}"
        )

        # Check repo_url format
        assert card["repo_url"].startswith("https://github.com/"), (
            f"SequenceCard {i} has invalid repo_url: {card['repo_url']}"
        )

    print(f"  Validated {len(sequence_cards)} SequenceCards")

    # Step 4: Save to output file
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saving to {output_path}...")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        ensure_dir(output_dir)

    with open(output_path, "w") as f:
        json.dump(sequence_cards, f, indent=2)

    # Verify file was written
    assert os.path.exists(output_path), f"Output file not created: {output_path}"
    file_size = os.path.getsize(output_path)
    assert file_size > 0, f"Output file is empty: {output_path}"

    print(f"  Saved {len(sequence_cards)} SequenceCards to {output_path} ({file_size} bytes)")

    return sequence_cards


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Collect agent traces from SWE-bench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect real data from Hugging Face (default)
  python collect_swe_bench_traces.py

  # Use mock data for testing
  python collect_swe_bench_traces.py --use-mock

  # Collect with custom output path and max samples
  python collect_swe_bench_traces.py --output data/custom/swe_bench.json --max-samples 500

  # Specify SWE-bench data directory
  python collect_swe_bench_traces.py --swe-bench-dir data/swe-bench-v2
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output file path (default: {DEFAULT_OUTPUT_PATH})",
    )

    parser.add_argument(
        "--swe-bench-dir",
        type=str,
        default=DEFAULT_SWE_BENCH_DIR,
        help=f"SWE-bench data directory (default: {DEFAULT_SWE_BENCH_DIR})",
    )

    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of downloading from SWE-bench",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of samples to collect (default: {DEFAULT_MAX_SAMPLES})",
    )

    args = parser.parse_args()

    # Validate arguments
    assert args.max_samples > 0, f"--max-samples must be positive, got {args.max_samples}"
    assert isinstance(args.output, str), "--output must be a string"
    assert isinstance(args.swe_bench_dir, str), "--swe-bench-dir must be a string"

    return args


def main() -> None:
    """
    Main entry point.

    Parses command line arguments and runs the collection process.
    """
    args = parse_args()

    print("=" * 60)
    print("SWE-bench Trace Collection")
    print("=" * 60)

    # Run collection
    sequence_cards = collect_swe_bench_traces(
        output_path=args.output,
        swe_bench_dir=args.swe_bench_dir,
        use_mock=args.use_mock,
        max_samples=args.max_samples,
    )

    # Print summary
    print("=" * 60)
    print("Collection Complete")
    print("=" * 60)
    print(f"Total SequenceCards: {len(sequence_cards)}")

    # Count by task type
    task_type_counts = {}
    for card in sequence_cards:
        task_type = card["task_type"]
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    print("\nTask type distribution:")
    for task_type, count in sorted(task_type_counts.items()):
        print(f"  {task_type}: {count}")

    # Count by memory type
    memory_type_counts = {}
    for card in sequence_cards:
        memory_type = card["memory_type"]
        memory_type_counts[memory_type] = memory_type_counts.get(memory_type, 0) + 1

    print("\nMemory type distribution:")
    for memory_type, count in sorted(memory_type_counts.items()):
        print(f"  {memory_type}: {count}")

    print(f"\nOutput saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

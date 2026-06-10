#!/usr/bin/env python3
"""
Generate all paper tables by calling table-generation scripts in tables/ directory.

This script is part of the paper "Memory Is a Hidden Dependency:
A Benchmark for Replay-Defined Harm in Stateful Coding Agents" reproduction work,
Stage 7 Part 3.

It discovers and runs all table-generation scripts under the tables/ directory,
collects their outputs, and organizes final tables into data/results/tables/.

Table scripts expected (one per paper table):
  tables/table1.py        -> Table 1: Main results
  tables/table2.py        -> Table 2: Additional results (or table2_minimal.py)
  tables/table3.py        -> Table 3: Ablation / breakdown
  tables/table4.py        -> Table 4: Human evaluation
  tables/table5.py        -> Table 5: Cost / runtime analysis

Each table script is expected to accept:
  --output-dir DIR   : where to write its output files
  --use-mock         : use mock data instead of real results
  --skip-existing    : skip if output already exists

Usage:
  python generate_all_tables.py
  python generate_all_tables.py --tables-dir tables --output-dir data/results/tables
  python generate_all_tables.py --use-mock --skip-existing
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default directories relative to this script's location
# Tables scripts are in code/tables/ (sibling of code/scripts/)
_DEFAULT_SCRIPT_DIR = Path(__file__).parent  # code/scripts/
_DEFAULT_CODE_DIR = _DEFAULT_SCRIPT_DIR.parent  # code/
DEFAULT_TABLES_DIR = _DEFAULT_CODE_DIR / "tables"
DEFAULT_OUTPUT_DIR = _DEFAULT_CODE_DIR / "data" / "results" / "tables"

# Expected table script names (in paper order).
# If a script is missing the dispatcher will warn but continue.
EXPECTED_TABLE_SCRIPTS: List[str] = [
    "table1.py",
    "table2.py",
    "table2_minimal.py",   # alternative / fallback for Table 2
    "table3.py",
    "table4.py",
    "table5.py",
]

# Results directory for each table script (relative to code/ directory).
# Table scripts read experiment results from these directories.
# Default can be overridden via --results-dir-base command-line argument.
# Set to None for scripts that don't use --results argument.
TABLE_RESULTS_DIRS: Dict[str, Optional[str]] = {
    "table1.py": None,                        # table1 uses --input, not --results
    "table2.py": "data/results_phase3",       # Phase 3 replay results
    "table2_minimal.py": "data/results_phase3",
    "table3.py": "data/results_phase4",       # Phase 4 ablation results
    "table4.py": "data/results_phase4",       # Phase 4 baseline results
    "table5.py": "data/results_phase5",       # Phase 5 channel results
}

# Python executable to use when spawning sub-scripts.
# Falls back to sys.executable so the same interpreter is used.
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_table_scripts(tables_dir: Path) -> List[Path]:
    """
    Discover table-generation scripts inside *tables_dir*.

    Scans for files named table{N}.py (N = 1..5) and table{N}_minimal.py.
    Returns paths sorted by table number so output order matches paper order.
    """
    assert tables_dir.is_dir(), f"Tables directory does not exist: {tables_dir}"
    found: List[Path] = []
    for pattern in ["table1.py", "table2.py", "table2_minimal.py",
                    "table3.py", "table4.py", "table5.py"]:
        candidate = tables_dir / pattern
        if candidate.is_file():
            found.append(candidate)
    # Also pick up any extra table*.py not in the explicit list (future-proofing).
    for p in sorted(tables_dir.glob("table*.py")):
        if p not in found:
            found.append(p)
    assert found, f"No table scripts found in {tables_dir}"
    return found


def run_table_script(
    script: Path,
    output_dir: Path,
    results_dir: Path,
    use_mock: bool,
    skip_existing: bool,
) -> Tuple[bool, str]:
    """
    Run a single table-generation script via subprocess.

    Returns (success: bool, stdout/stderr output: str).
    Fast-fails by checking script existence and output directory upfront.
    """
    assert script.is_file(), f"Table script not found: {script}"
    assert script.suffix == ".py", f"Not a Python script: {script}"

    # Compute expected output path: {output_dir}/{script_stem}.tex
    # Table scripts follow the convention of writing to this path when --output-dir is used.
    expected_output = output_dir / f"{script.stem}.tex"

    # Skip if requested and output already exists
    if skip_existing and expected_output.is_file():
        print(f"  [SKIP] {script.name} (output exists: {expected_output})")
        return True, f"Skipped (output exists: {expected_output})"

    # Build command — use absolute path so cwd does not matter
    cmd = [PYTHON, str(script.resolve()), "--output-dir", str(output_dir)]
    # Pass --results to scripts that need it (not table1.py which uses --input)
    if results_dir is not None:
        cmd.extend(["--results", str(results_dir)])
    if use_mock:
        cmd.append("--use-mock")

    # Run — no cwd override; sub-script should use its own __file__ for relative paths
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    success = result.returncode == 0
    output = result.stdout + "\n" + result.stderr
    return success, output


def ensure_output_dir(output_dir: Path) -> None:
    """Create output directory if needed; fail fast if path is not writable."""
    assert not output_dir.exists() or output_dir.is_dir(), \
        f"Output path exists but is not a directory: {output_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Quick write test
    test_file = output_dir / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as exc:
        raise AssertionError(
            f"Cannot write to output directory {output_dir}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all paper tables by running table scripts."
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help=f"Directory containing table-generation scripts (default: {DEFAULT_TABLES_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where table outputs are written (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--results-dir-base",
        type=Path,
        default=None,
        help="Base directory for results (overrides TABLE_RESULTS_DIRS mapping).",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Pass --use-mock to sub-scripts (generate tables from mock data).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip table scripts whose output already exists.",
    )
    args = parser.parse_args()

    # Fast-fail preconditions
    assert args.tables_dir.is_dir(), \
        f"Tables directory does not exist: {args.tables_dir}"
    ensure_output_dir(args.output_dir)

    # Discover scripts
    scripts = find_table_scripts(args.tables_dir)
    print(f"[generate_all_tables] Found {len(scripts)} table script(s):")
    for s in scripts:
        print(f"  - {s.name}")

    # Run each script
    failures: List[Tuple[str, str]] = []
    for script in scripts:
        label = script.stem   # e.g. "table1"
        print(f"\n[generate_all_tables] Running {script.name} ...")

        # Determine results_dir for this script
        if args.results_dir_base is not None:
            # Use --results-dir-base as the results dir for all scripts
            results_dir = args.results_dir_base
        elif script.name in TABLE_RESULTS_DIRS and TABLE_RESULTS_DIRS[script.name] is not None:
            results_dir = Path(TABLE_RESULTS_DIRS[script.name])
        else:
            results_dir = None  # script may not need --results (e.g., table1.py)

        ok, output = run_table_script(
            script, args.output_dir, results_dir, args.use_mock, args.skip_existing
        )
        if ok:
            print(f"  [OK] {script.name} completed.")
            # Print last few lines of output for quick diagnosis
            tail = "\n".join(output.strip().splitlines()[-8:])
            if tail:
                print(f"  Output (tail):\n{tail}")
        else:
            print(f"  [FAIL] {script.name} exited with error.")
            print(f"  Output:\n{output[-2000:]}"  )  # last 2000 chars
            failures.append((script.name, output))

    # Summary
    print(f"\n{'=' * 60}")
    print(f"[generate_all_tables] Done. {len(scripts) - len(failures)}/{len(scripts)} succeeded.")
    if failures:
        print(f"Failures ({len(failures)}):")
        for name, _ in failures:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print("All table scripts completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()

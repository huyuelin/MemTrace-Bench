#!/usr/bin/env python3
"""
Generate all figures for the "Memory Is a Hidden Dependency" paper.

This script orchestrates the generation of all six figures by calling the
individual figure generation scripts in figures/ via subprocess. It supports
both mock mode (for development) and real data mode (for publication).

Usage:
    # Generate all figures with real data
    python scripts/generate_all_figures.py

    # Generate all figures with mock data (no real traces needed)
    python scripts/generate_all_figures.py --use-mock

    # Skip figures that already exist
    python scripts/generate_all_figures.py --skip-existing

    # Custom output directory
    python scripts/generate_all_figures.py --output-dir custom/figures/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Constants: figure script metadata
# ---------------------------------------------------------------------------
# Each entry: (script_name, output_filename, kind)
#   kind: "mock_ok"  -> script accepts --use-mock and --output
#         "results"   -> script requires --results <directory> and --output <pdf>
# The kind determines how the subprocess command is constructed.
FIGURE_ENTRIES = [
    # Figure 1: running example (mock_ok: --use-mock --output)
    ("figure1_running_example.py", "figure1.pdf", "mock_ok"),
    # Figure 2: harm definition (mock_ok: --use-mock --output; --use-mock is accepted but not required)
    ("figure2_harm_definition.py", "figure2.pdf", "mock_ok"),
    # Figure 3: dashboard (results: --results <dir> --output <pdf>)
    ("figure3_dashboard.py", "figure3.pdf", "results"),
    # Figure 4: frontier (results: --results <dir> --output <pdf>)
    ("figure4_frontier.py", "figure4.pdf", "results"),
    # Figure 5: cross-benchmark (mock_ok: --results-dir <dir> --output <pdf> --use-mock)
    ("figure5_cross_benchmark.py", "figure5.pdf", "mock_ok"),
    # Figure 6: anatomy (results: --results <dir> --output <pdf>)
    ("figure6_anatomy.py", "figure6.pdf", "results"),
]

# Default results directory used by "results" kind figures (Figures 3, 4, 6).
# These figures expect --results <directory> where individual results_*.json files live.
DEFAULT_RESULTS_DIR = "data/results"


def resolve_code_dir() -> Path:
    """Return the code/ directory (project root for all script paths).

    This script lives at code/scripts/generate_all_figures.py.
    code/ is one level up from scripts/.
    """
    script_dir = Path(__file__).resolve().parent          # code/scripts/
    code_dir = script_dir.parent                           # code/
    return code_dir


def build_command(
    figure_script: Path,
    output_path: Path,
    kind: str,
    use_mock: bool,
    results_dir: Optional[Path],
) -> List[str]:
    """Construct the subprocess command for a single figure script.

    Parameters
    ----------
    figure_script : Path
        Absolute path to the figure generation script.
    output_path : Path
        Absolute path where the figure PDF should be written.
    kind : str
        "mock_ok" or "results" - determines the CLI interface.
    use_mock : bool
        If True, pass --use-mock to scripts that accept it.
    results_dir : Optional[Path]
        Path to the results directory (used by "results" kind figures).

    Returns
    -------
    List[str]
        The command as a list of strings, ready for subprocess.run().
    """
    cmd = [sys.executable, str(figure_script)]

    if kind == "mock_ok":
        # Figures 1, 2, 5: accept --output and optionally --use-mock
        cmd.extend(["--output", str(output_path)])
        if use_mock:
            cmd.append("--use-mock")
        # Figure 5 also accepts --results-dir; point it at the results directory
        if "figure5" in figure_script.name:
            results_dir_path = figure_script.parent.parent / "data" / "results"
            cmd.extend(["--results-dir", str(results_dir_path)])
    elif kind == "results":
        # Figures 3, 4, 6: require --results <directory> and --output <pdf>
        # If results_dir is None or doesn't exist, still pass it and let the
        # subprocess fail with a clear error (Fast-Fail: no silent fallback).
        results_arg = str(results_dir) if results_dir is not None else "NONE"
        cmd.extend(["--results", results_arg, "--output", str(output_path)])
    else:
        raise ValueError(f"Unknown figure kind: {kind}")

    return cmd


def generate_figure(
    figure_script: Path,
    output_path: Path,
    kind: str,
    use_mock: bool,
    results_dir: Optional[Path],
    skip_existing: bool,
) -> bool:
    """Generate a single figure by calling its script via subprocess.

    Returns True on success, False on failure.

    Fast-Fail: if the subprocess exits with non-zero, print the stderr
    and return False. The caller decides whether to continue.
    """
    # Skip if output already exists and --skip-existing is set
    if skip_existing and output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"  [SKIP] {output_path.name} already exists ({size_kb:.0f} KB)")
        return True

    cmd = build_command(figure_script, output_path, kind, use_mock, results_dir)

    # Print the command for transparency (without flooding the log)
    cmd_display = " ".join(cmd[:3]) + " ... " + cmd[-1]
    print(f"  [RUN] {figure_script.name}")
    print(f"        {cmd_display}")

    # Run the subprocess; capture stdout/stderr for error reporting
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(figure_script.parent.parent),  # run from code/ directory
    )

    if result.returncode != 0:
        print(f"  [FAIL] {figure_script.name} exited with code {result.returncode}")
        # Print last 20 lines of stderr for diagnosability
        stderr_lines = result.stderr.strip().splitlines()
        for line in stderr_lines[-20:]:
            print(f"         {line}")
        return False

    # Verify the output file was created
    assert output_path.exists(), \
        f"Script {figure_script.name} exited 0 but output missing: {output_path}"
    size_kb = output_path.stat().st_size / 1024
    print(f"  [OK]   {output_path.name} ({size_kb:.0f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures for the 'Memory Is a Hidden Dependency' paper."
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help="Directory containing figure generation scripts (relative to code/). Default: figures",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/figures",
        help="Output directory for generated figure PDFs (relative to code/). Default: data/results/figures",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Path to the results directory for 'results' kind figures "
            "(Figures 3, 4, 6). Default: data/results/"
        ),
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock data instead of real traces (passed to scripts that accept --use-mock).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip figures whose output PDF already exists.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Resolve paths relative to code/ directory
    # -----------------------------------------------------------------------
    code_dir = resolve_code_dir()
    figures_dir = code_dir / args.figures_dir
    output_dir = code_dir / args.output_dir
    results_dir = code_dir / (args.results_dir or DEFAULT_RESULTS_DIR)

    # Fast-Fail: verify figures directory exists
    assert figures_dir.is_dir(), \
        f"Figures directory not found: {figures_dir}. " \
        f"Check --figures-dir (current: {args.figures_dir})."

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify results directory exists (needed for "results" kind figures in non-mock mode)
    # In mock mode, figures 3/4/6 may still need the directory; warn but don't fail here
    # because the figure scripts themselves should handle missing dirs in mock mode.
    if not results_dir.is_dir():
        if not args.use_mock:
            raise FileNotFoundError(
                f"Results directory not found: {results_dir}. "
                f"Run the data collection pipeline first, or use --use-mock, "
                f"or specify --results-dir."
            )
        else:
            print(
                f"[WARN] Results directory not found: {results_dir}. "
                f"Figures 3/4/6 may fail or use fallback data."
            )
            results_dir_valid = None   # pass None to generate_figure
    else:
        results_dir_valid = results_dir

    # -----------------------------------------------------------------------
    # Generate each figure in order
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Generating all figures")
    print(f"  Figures dir  : {figures_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Results dir  : {results_dir}")
    print(f"  Use mock     : {args.use_mock}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"{'=' * 60}\n")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for script_name, output_filename, kind in FIGURE_ENTRIES:
        figure_script = figures_dir / script_name
        output_path = output_dir / output_filename

        # Fast-Fail: verify the figure script exists
        assert figure_script.exists(), \
            f"Figure script not found: {figure_script}. " \
            f"Expected at: {figures_dir}/{script_name}"

        print(f"[Figure] {script_name} -> {output_filename} ({kind})")

        ok = generate_figure(
            figure_script=figure_script,
            output_path=output_path,
            kind=kind,
            use_mock=args.use_mock,
            results_dir=results_dir_valid,
            skip_existing=args.skip_existing,
        )

        if ok:
            if args.skip_existing and output_path.exists():
                skip_count += 1
            else:
                success_count += 1
        else:
            fail_count += 1
            # Continue with remaining figures; fail at the end with a summary
            print(f"         (continuing with remaining figures...)\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Summary: {success_count} succeeded, {fail_count} failed, {skip_count} skipped")
    print(f"{'=' * 60}")

    if fail_count > 0:
        print(f"\n[FAIL] {fail_count} figure(s) failed. Check the logs above.")
        sys.exit(1)
    else:
        print(f"\n[OK] All figures generated successfully -> {output_dir}/")
        sys.exit(0)


if __name__ == "__main__":
    main()

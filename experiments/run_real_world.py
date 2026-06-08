#!/usr/bin/env python3
"""
Phase 4: Real-world validation experiment runner.

Runs real-world validation experiments on three benchmarks:
- GitHub: Code repository Q&A tasks
- SWE-bench: Software engineering benchmark tasks
- ReAct: Reasoning and acting tasks

Uses the same framework as run_replay.py but with real-world agents and benchmarks.
Supports all experimental conditions and baseline memory systems.
"""

import sys
import os

# Add parent directory to path so 'core' can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from typing import List, Dict, Any, Optional


# Import functions from run_replay.py
from experiments.run_replay import (
    load_sequences,
    run_condition,
    compute_and_print_estimands,
    save_results,
)


# Benchmark configurations
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "..")

BENCHMARK_CONFIGS = {
    "github": {
        "default_sequences": os.path.join(_CODE_DIR, "data/processed/benchmark_v1.json"),
        "default_conditions": "clean,warm,delete-target",
        "default_baselines": "memgpt,mem0,zep,a-mem,reflexion,workflow",
        "sequence_filter": "github",
    },
    "swe-bench": {
        "default_sequences": os.path.join(_CODE_DIR, "data/processed/benchmark_v1.json"),
        "default_conditions": "clean,warm,delete-target",
        "default_baselines": "memgpt,mem0,zep,a-mem,reflexion,workflow",
        "sequence_filter": "swe-bench",
    },
    "react": {
        "default_sequences": os.path.join(_CODE_DIR, "data/processed/benchmark_v1.json"),
        "default_conditions": "clean,warm,delete-target",
        "default_baselines": "memgpt,mem0,zep,a-mem,reflexion,workflow",
        "sequence_filter": "react",
    },
}

# All valid conditions (same as run_replay.py)
VALID_CONDITIONS = [
    "clean", "warm", "delete-target",
    "transplant", "matched-placebo", "semantic-placebo",
    "token-padding", "prelude-only", "dose-response",
    "rank-shuffle", "position-shuffle", "reference-mediator",
    "memgpt", "mem0", "naive-vector", "zep", "a-mem", "reflexion", "workflow",
    "conversation", "memorybank", "langmem",
    "current-repo-rag", "time-aware-rag", "tool-verified-rag",
    "store-only", "full-instr",
]

# All valid baselines
VALID_BASELINES = [
    "memgpt", "mem0", "naive-vector", "zep", "a-mem", "reflexion", "workflow",
    "conversation", "memorybank", "langmem",
    "current-repo-rag", "time-aware-rag", "tool-verified-rag",
]


def filter_sequences_by_benchmark(
    sequences: List[Dict[str, Any]],
    benchmark: str,
) -> List[Dict[str, Any]]:
    """Filter sequences to only include those for a specific benchmark."""
    config = BENCHMARK_CONFIGS.get(benchmark)
    assert config is not None, f"Unknown benchmark: {benchmark}"
    
    filter_key = config["sequence_filter"]
    # Filter by sequence_id prefix (github_*, swe_bench_*, react_*)
    filtered = [s for s in sequences if s.get("sequence_id", "").startswith(filter_key)]
    
    assert len(filtered) > 0, f"No sequences found for benchmark '{benchmark}' (filter='{filter_key}')"
    print(f"Filtered {len(filtered)} sequences for benchmark '{benchmark}'")
    return filtered


def load_benchmark_sequences(
    benchmark: str,
    sequences_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load and filter sequences for a specific benchmark.

    Args:
        benchmark: Benchmark name (github, swe-bench, react)
        sequences_path: Optional path to sequences JSON file (overrides default)

    Returns:
        List of sequence cards for the benchmark
    """
    config = BENCHMARK_CONFIGS.get(benchmark)
    assert config is not None, f"Unknown benchmark: {benchmark}. Valid: {list(BENCHMARK_CONFIGS.keys())}"

    # Use provided path or default
    path = sequences_path if sequences_path else config["default_sequences"]
    assert os.path.exists(path), f"Sequences file not found: {path}"

    # Load all sequences (handle both list format and benchmark_v1 dict format)
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "train" in raw:
        # benchmark_v1.json format: {"train": [...], "val": [...], "test": [...]}
        all_sequences = []
        for split in ["train", "val", "test"]:
            if split in raw and isinstance(raw[split], list):
                all_sequences.extend(raw[split])
    else:
        # Plain list format
        assert isinstance(raw, list), f"Expected list or benchmark dict, got {type(raw)}"
        all_sequences = raw

    # Filter for this benchmark
    filtered = filter_sequences_by_benchmark(all_sequences, benchmark)

    return filtered


def create_agent(
    agent_type: str,
    model: str,
    use_mock: bool = False,
) -> Any:
    """Create an agent instance for real-world experiments.

    Args:
        agent_type: Type of agent (github, swe-bench, react, mock)
        model: Model identifier
        use_mock: If True, use mock agent for testing

    Returns:
        Agent instance
    """
    if use_mock or agent_type == "mock":
        # Inline MockAgent for testing (avoids dependency on agents.mock_agent)
        class MockAgent:
            def __init__(self, model, **kwargs):
                self.model = model
            def run(self, sequence, **kwargs):
                return {"output": f"mock output for {sequence.get('sequence_id', 'unknown')}"}
        return MockAgent(model=model)

    # Real mode: create agent with use_real_llm=True
    use_real_llm = not use_mock
    if agent_type == "github":
        from agents.github_agent import GitHubAgent
        return GitHubAgent(
            model=model,
            use_real_llm=use_real_llm,
            use_real_tools=use_real_llm,
        )
    elif agent_type == "swe-bench":
        from agents.swe_bench_agent import SWEBenchAgent
        return SWEBenchAgent(
            model=model,
            use_real_llm=use_real_llm,
            use_real_tools=use_real_llm,
        )
    elif agent_type == "react":
        from agents.react_agent import ReActAgent
        return ReActAgent(
            model=model,
            use_real_llm=use_real_llm,
            use_real_tools=use_real_llm,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Valid: github, swe-bench, react, mock")


def validate_conditions(conditions: List[str]) -> None:
    """Validate that all conditions are valid.

    Args:
        conditions: List of condition names to validate

    Raises:
        AssertionError: If any condition is invalid
    """
    for c in conditions:
        valid = c in VALID_CONDITIONS or (c.startswith("dose-response-") and c.split("-")[-1].isdigit())
        assert valid, f"Invalid condition: {c}. Valid conditions: {VALID_CONDITIONS}"


def validate_baselines(baselines: List[str]) -> None:
    """Validate that all baselines are valid.

    Args:
        baselines: List of baseline names to validate

    Raises:
        AssertionError: If any baseline is invalid
    """
    for b in baselines:
        assert b in VALID_BASELINES, f"Invalid baseline: {b}. Valid baselines: {VALID_BASELINES}"


def run_real_world_experiment(
    benchmark: str,
    sequences: List[Dict[str, Any]],
    conditions: List[str],
    baselines: List[str],
    agent_type: str,
    model: str,
    repeats: int,
    output_dir: str,
    use_mock: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run real-world validation experiment.

    Args:
        benchmark: Benchmark name (github, swe-bench, react)
        sequences: List of sequence cards
        conditions: List of conditions to run
        baselines: List of baseline systems to run
        agent_type: Type of agent to use
        model: Model identifier
        repeats: Number of repeats per condition
        output_dir: Directory to save results
        use_mock: If True, use mock mode for testing

    Returns:
        Dictionary mapping condition names to lists of results
    """
    # Validate inputs
    validate_conditions(conditions)
    validate_baselines(baselines)

    # Create agent
    agent = create_agent(agent_type, model, use_mock)
    print(f"Created agent: {agent_type} (mock={use_mock})")

    # Model config
    model_config = {
        "model": model,
        "temperature": 0.7,
        "top_k": 50,
        "seed": 42,
        "use_mock": use_mock,
    }

    # Combine conditions and baselines
    all_conditions = conditions + baselines
    print(f"Running {len(all_conditions)} conditions: {all_conditions}")
    print(f"On {len(sequences)} sequences with {repeats} repeats each")

    # Run experiments
    all_results = {c: [] for c in all_conditions}
    for seq_idx, seq in enumerate(sequences):
        seq_id = seq.get("sequence_id", f"seq_{seq_idx}")
        print(f"\n[{seq_idx + 1}/{len(sequences)}] Processing sequence: {seq_id}")

        for condition in all_conditions:
            print(f"  Running condition: {condition}")
            try:
                results = run_condition(
                    seq, condition, agent, model_config, repeats, use_real=not use_mock
                )
                all_results[condition].extend(results)
                print(f"    Completed: {len(results)} runs")
            except Exception as e:
                print(f"    ERROR in {condition}: {e}")
                raise

    # Compute estimands
    print("\n" + "=" * 50)
    print("Computing estimands...")
    compute_and_print_estimands(sequences, all_results)

    # Save results
    print("\n" + "=" * 50)
    print(f"Saving results to: {output_dir}")
    save_results(all_results, output_dir)

    return all_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run real-world validation experiments (Phase 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GitHub benchmark with default settings
  python run_real_world.py --benchmark github

  # Run SWE-bench with specific conditions
  python run_real_world.py --benchmark swe-bench --conditions clean,warm,delete-target

  # Run ReAct with specific baselines
  python run_real_world.py --benchmark react --baselines memgpt,mem0,zep

  # Use mock mode for quick testing
  python run_real_world.py --benchmark github --use-mock

  # Custom sequences and output
  python run_real_world.py --benchmark github --sequences data/custom.json --output data/results_custom/
        """
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["github", "swe-bench", "react"],
        help="Benchmark type (github, swe-bench, react)",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default=None,
        help="Path to sequences JSON file (overrides benchmark default)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated list of conditions to run (overrides benchmark default)",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default=None,
        help="Comma-separated list of baseline systems to run (overrides benchmark default)",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default=None,
        help="Agent type (github, swe-bench, react, mock). Defaults to benchmark name.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model type (default: gpt-4)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per condition (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/results_real_world/<benchmark>)",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock mode for quick testing",
    )

    args = parser.parse_args()

    # Apply benchmark defaults if not specified
    config = BENCHMARK_CONFIGS[args.benchmark]

    if args.conditions is None:
        args.conditions = config["default_conditions"]
    if args.baselines is None:
        args.baselines = config["default_baselines"]
    if args.agent_type is None:
        args.agent_type = "mock" if args.use_mock else args.benchmark
    if args.output is None:
        args.output = f"data/results_real_world/{args.benchmark}"

    # Parse comma-separated lists
    args.conditions = [c.strip() for c in args.conditions.split(",")]
    args.baselines = [b.strip() for b in args.baselines.split(",")]

    return args


def main() -> None:
    """Main entry point for real-world validation experiments."""
    args = parse_args()

    print("=" * 60)
    print("Real-World Validation Experiment Runner (Phase 4)")
    print("=" * 60)
    print(f"Benchmark: {args.benchmark}")
    print(f"Agent type: {args.agent_type}")
    print(f"Model: {args.model}")
    print(f"Repeats: {args.repeats}")
    print(f"Mock mode: {args.use_mock}")
    print(f"Conditions: {args.conditions}")
    print(f"Baselines: {args.baselines}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Load sequences
    sequences = load_benchmark_sequences(args.benchmark, args.sequences)
    print(f"Loaded {len(sequences)} sequences for benchmark '{args.benchmark}'")

    # Run experiments
    results = run_real_world_experiment(
        benchmark=args.benchmark,
        sequences=sequences,
        conditions=args.conditions,
        baselines=args.baselines,
        agent_type=args.agent_type,
        model=args.model,
        repeats=args.repeats,
        output_dir=args.output,
        use_mock=args.use_mock,
    )

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 3: Replay experiment runner.

Runs all 14+ experimental conditions with either:
  - Mock mode (--use-real False, default): uses mock LLM calls and test results
  - Real mode (--use-real True): uses real LLM API calls and test execution

Usage:
  # Mock mode (default, for testing)
  python experiments/run_replay.py --sequences data/processed/benchmark_v1.json --conditions clean,warm

  # Real mode (requires API keys and real repo data)
  python experiments/run_replay.py --sequences data/processed/benchmark_v1.json --conditions clean,warm --use-real --agent-type github
"""

import sys
import os

# Add parent directory to path so 'core' can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from typing import List, Dict, Any


def load_sequences(path: str) -> List[Dict[str, Any]]:
    """Load sequence cards from JSON file.

    Handles both formats:
      - List format: [seq1, seq2, ...]
      - Dict format (benchmark_v1.json): {"train": [...], "val": [...], "test": [...]}
    """
    assert os.path.exists(path), f"Sequences file not found: {path}"
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and any(k in raw for k in ["train", "val", "test"]):
        # benchmark_v1.json format: merge all splits
        sequences = []
        for split in ["train", "val", "test"]:
            if split in raw and isinstance(raw[split], list):
                sequences.extend(raw[split])
    else:
        assert isinstance(raw, list), f"Expected list or benchmark dict, got {type(raw)}"
        sequences = raw
    assert len(sequences) > 0, "No sequences loaded"
    return sequences


def create_agent(
    agent_type: str,
    use_real_llm: bool = False,
    use_real_tools: bool = False,
    work_dir: str = None,
) -> Any:
    """
    Create agent instance based on agent_type.

    Args:
        agent_type:     "mock", "github", or "swe_bench"
        use_real_llm:  If True, agent uses real LLM API calls
        use_real_tools: If True, agent executes real tools (bash, file I/O, tests)
        work_dir:      Working directory for tool execution (repo root)

    Returns:
        BaseAgent subclass instance.
    """
    if agent_type == "mock":
        from agents.base_agent import _TestAgent
        agent = _TestAgent(
            model="mock",
            temperature=0.0,
            max_tokens=4096,
            top_k=50,
            seed=42,
            use_real_llm=use_real_llm,
            use_real_tools=use_real_tools,
            work_dir=work_dir,
        )
        return agent
    elif agent_type == "github":
        from agents.github_agent import GitHubAgent
        agent = GitHubAgent(
            model="hunyuan-2.0-instruct-20251111" if use_real_llm else "mock",
            temperature=0.0,
            max_tokens=4096,
            top_k=50,
            seed=42,
            use_real_llm=use_real_llm,
            use_real_tools=use_real_tools,
            work_dir=work_dir,
        )
        return agent
    elif agent_type == "swe_bench":
        from agents.swe_bench_agent import SWEBenchAgent
        agent = SWEBenchAgent(
            model="hunyuan-2.0-instruct-20251111" if use_real_llm else "mock",
            temperature=0.0,
            max_tokens=4096,
            top_k=50,
            seed=42,
            use_real_llm=use_real_llm,
            use_real_tools=use_real_tools,
            work_dir=work_dir,
        )
        return agent
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}. Valid: mock, github, swe_bench")


def run_condition(
    seq: Dict[str, Any],
    condition: str,
    agent: Any,
    model_config: Dict[str, Any],
    repeat: int = 1,
    use_real: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run a single condition for a sequence, with repeats.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        condition:   Condition name string.
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        repeat:      Number of repeats per condition.
        use_real:    If True, use real LLM calls and test execution.

    Returns:
        List of result dicts (one per repeat).
    """
    from core.conditions import (
        run_clean, run_warm, run_delete_target,
        run_transplant, run_matched_placebo, run_semantic_placebo,
        run_token_padding, run_prelude_only, run_dose_response,
        run_rank_shuffle, run_position_shuffle, run_reference_mediator,
        run_store_only, run_full_instr,
    )

    results = []
    for i in range(repeat):
        config = {**model_config, "seed": model_config.get("seed", 42) + i}
        if condition == "clean":
            result = run_clean(seq, agent, config, use_real=use_real)
        elif condition == "warm":
            result = run_warm(seq, agent, config, use_real=use_real)
        elif condition == "delete-target":
            result = run_delete_target(seq, agent, config, use_real=use_real)
        elif condition == "transplant":
            result = run_transplant(seq, agent, config, use_real=use_real)
        elif condition == "matched-placebo":
            result = run_matched_placebo(seq, agent, config, use_real=use_real)
        elif condition == "semantic-placebo":
            result = run_semantic_placebo(seq, agent, config, use_real=use_real)
        elif condition == "token-padding":
            result = run_token_padding(seq, agent, config, use_real=use_real)
        elif condition == "prelude-only":
            result = run_prelude_only(seq, agent, config, use_real=use_real)
        elif condition.startswith("dose-response"):
            n = int(condition.split("-")[-1]) if condition.split("-")[-1].isdigit() else 1
            result = run_dose_response(seq, agent, config, n_invalid=n, use_real=use_real)
        elif condition == "rank-shuffle":
            result = run_rank_shuffle(seq, agent, config, use_real=use_real)
        elif condition == "position-shuffle":
            result = run_position_shuffle(seq, agent, config, use_real=use_real)
        elif condition == "reference-mediator":
            result = run_reference_mediator(seq, agent, config, use_real=use_real)
        elif condition == "store-only":
            result = run_store_only(seq, agent, config, use_real=use_real)
        elif condition == "full-instr":
            result = run_full_instr(seq, agent, config, use_real=use_real)
        elif condition == "memgpt":
            from baselines.memgpt import MemGPTMemorySystem
            baseline = MemGPTMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "mem0":
            from baselines.mem0 import Mem0MemorySystem
            baseline = Mem0MemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "naive-vector":
            from baselines.naive_vector import NaiveVectorMemorySystem
            baseline = NaiveVectorMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "zep":
            from baselines.zep import ZepMemorySystem
            baseline = ZepMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "a-mem":
            from baselines.a_mem import AMEMMemorySystem
            baseline = AMEMMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "reflexion":
            from baselines.reflexion import ReflexionMemorySystem
            baseline = ReflexionMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "workflow":
            from baselines.workflow import WorkflowMemorySystem
            baseline = WorkflowMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "conversation":
            from baselines.conversation import ConversationMemorySystem
            baseline = ConversationMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "memorybank":
            from baselines.memorybank import MemoryBankMemorySystem
            baseline = MemoryBankMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "langmem":
            from baselines.langmem import LangMemMemorySystem
            baseline = LangMemMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "current-repo-rag":
            from baselines.current_repo_rag import CurrentRepoRAGMemorySystem
            baseline = CurrentRepoRAGMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "time-aware-rag":
            from baselines.time_aware_rag import TimeAwareRAGMemorySystem
            baseline = TimeAwareRAGMemorySystem()
            result = baseline.run(seq, agent, model_config)
        elif condition == "tool-verified-rag":
            from baselines.tool_verified_rag import ToolVerifiedRAGMemorySystem
            baseline = ToolVerifiedRAGMemorySystem()
            result = baseline.run(seq, agent, model_config)
        else:
            raise ValueError(f"Unknown condition: {condition}")
        results.append(result)
    return results


def compute_and_print_estimands(
    sequences: List[Dict[str, Any]],
    all_results: Dict[str, List[Dict[str, Any]]],
):
    """Compute and print hat_b and Deltas."""
    from core.estimands import (
        compute_bad_rate,
        compute_pass_rate,
        delta_del,
        delta_pc,
        compute_C_pc,
    )

    print("\n=== Results ===")

    for condition, results in all_results.items():
        bad_rate = compute_bad_rate(results)
        pass_rate = compute_pass_rate(results)
        print(f"  {condition}: pass={pass_rate:.3f}, bad={bad_rate:.3f}")

    if "warm" in all_results and "delete-target" in all_results:
        dd = delta_del(all_results["warm"], all_results["delete-target"])
        print(f"  Delta_del (warm - delete-target): {dd:.3f}")

    if "warm" in all_results and "clean" in all_results:
        dp = delta_pc(all_results["warm"], all_results["clean"])
        print(f"  Delta_pc (warm - clean): {dp:.3f}")

    if "warm" in all_results and "matched-placebo" in all_results:
        c_pc = compute_C_pc(all_results["warm"], all_results["matched-placebo"], sequences)
        print(f"  C_pc (warm - matched-placebo): {c_pc:.3f}")


def save_results(
    results: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
):
    """Save RunManifest results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    for condition, result_list in results.items():
        output_path = os.path.join(output_dir, f"results_{condition}.json")
        with open(output_path, "w") as f:
            json.dump(result_list, f, indent=2)
        print(f"  Saved {len(result_list)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run replay experiments (Phase 3)")
    parser.add_argument(
        "--sequences",
        type=str,
        required=True,
        help="Path to sequences JSON file",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="clean,warm,delete-target",
        help="Comma-separated list of conditions to run",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="mock",
        choices=["mock", "github", "swe_bench"],
        help="Agent type to use",
    )
    parser.add_argument(
        "--use-real",
        action="store_true",
        help="Use real LLM calls and test execution (default: mock mode)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        help="Model name (used when --use-real is set)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeats per condition (use 1 for real mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Max number of sequences to process (for testing)",
    )
    args = parser.parse_args()

    # Validate inputs
    assert os.path.exists(args.sequences), f"File not found: {args.sequences}"
    conditions = [c.strip() for c in args.conditions.split(",")]
    valid_conditions = [
        "clean", "warm", "delete-target",
        "transplant", "matched-placebo", "semantic-placebo",
        "token-padding", "prelude-only", "dose-response",
        "rank-shuffle", "position-shuffle", "reference-mediator",
        "memgpt", "mem0", "naive-vector", "zep", "a-mem", "reflexion", "workflow",
        "conversation", "memorybank", "langmem",
        "current-repo-rag", "time-aware-rag", "tool-verified-rag",
        "store-only", "full-instr",
    ]
    for c in conditions:
        valid = c in valid_conditions or (c.startswith("dose-response-") and c.split("-")[-1].isdigit())
        assert valid, f"Invalid condition: {c}. Valid: {valid_conditions}"

    # Load sequences
    sequences = load_sequences(args.sequences)
    if args.max_sequences is not None:
        sequences = sequences[:args.max_sequences]
    print(f"Loaded {len(sequences)} sequences")

    # Model config
    model_config = {
        "model": args.model if args.use_real else "mock",
        "temperature": 0.0,
        "top_k": 50,
        "seed": 42,
    }

    # Create agent
    print(f"Creating agent: type={args.agent_type}, use_real={args.use_real}")
    agent = create_agent(
        agent_type=args.agent_type,
        use_real_llm=args.use_real,
        use_real_tools=args.use_real,
        work_dir=None,  # Will be set per-sequence in real mode
    )

    # Run experiments
    all_results = {c: [] for c in conditions}
    for seq in sequences:
        seq_id = seq.get("sequence_id", "unknown")
        print(f"\n=== Processing sequence: {seq_id} ===")

        # In real mode, set up work_dir for this sequence
        if args.use_real:
            repo_url = seq.get("repo_url", "")
            repo_commit = seq.get("repo_commit", "")
            if repo_url and repo_commit:
                # Clone repo and set work_dir
                work_dir = _setup_repo(repo_url, repo_commit)
                agent.work_dir = work_dir
                print(f"  work_dir set to: {work_dir}")
            else:
                print(f"  WARNING: no repo_url/commit for {seq_id}, skipping real mode")
                continue

        for condition in conditions:
            print(f"  Running condition: {condition}")
            try:
                results = run_condition(
                    seq, condition, agent, model_config, args.repeats, use_real=args.use_real
                )
                all_results[condition].extend(results)
            except Exception as e:
                print(f"  ERROR in {condition}: {e}")
                import traceback
                traceback.print_exc()

            # Reset agent state between conditions
            agent.reset()

    # Compute estimands
    compute_and_print_estimands(sequences, all_results)

    # Save results
    save_results(all_results, args.output)
    print("\n=== Done ===")


def _setup_repo(repo_url: str, repo_commit: str) -> str:
    """
    Clone a repo and checkout a specific commit.
    Returns the path to the cloned repo.

    Args:
        repo_url:    GitHub repo URL (e.g., "https://github.com/owner/repo")
        repo_commit: Commit hash to checkout

    Returns:
        Path to the cloned repo directory.
    """
    import tempfile
    import subprocess

    assert isinstance(repo_url, str) and repo_url.strip() != "", f"repo_url must be non-empty, got {repo_url!r}"
    assert isinstance(repo_commit, str) and repo_commit.strip() != "", f"repo_commit must be non-empty, got {repo_commit!r}"

    clone_dir = tempfile.mkdtemp(prefix="run_replay_repo_")
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    clone_path = os.path.join(clone_dir, repo_name)

    # Clone
    clone_cmd = f"git clone --depth 1 {repo_url} {repo_name}"
    result = subprocess.run(clone_cmd, shell=True, capture_output=True, text=True, cwd=clone_dir, timeout=120)
    assert result.returncode == 0, f"git clone failed: {result.stderr}"

    # Checkout commit
    checkout_cmd = f"git checkout {repo_commit}"
    result = subprocess.run(checkout_cmd, shell=True, capture_output=True, text=True, cwd=clone_path, timeout=30)
    assert result.returncode == 0, f"git checkout {repo_commit} failed: {result.stderr}"

    return clone_path


if __name__ == "__main__":
    main()

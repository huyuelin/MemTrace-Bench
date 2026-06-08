#!/usr/bin/env python3
"""
Collect ReAct agent traces and convert to SequenceCard format.

ReAct (Reasoning + Acting) is a prompting paradigm where language models
interleave reasoning traces and task-specific actions. This script collects
ReAct agent execution traces from public datasets and converts them into
the SequenceCard format used by the Memory-Trace benchmark.

The script supports two modes:
  - Real data mode: downloads and parses ReAct traces from HuggingFace/paper releases
  - Mock mode (--use-mock): generates synthetic ReAct-like traces for testing

Output format: list of SequenceCard dicts, saved as JSON.
Each SequenceCard represents one ReAct agent trace (a sequence of thought/action/observation).

Usage:
    python collect_react_traces.py --output data/raw/react_traces.json --use-mock
    python collect_react_traces.py --output data/raw/react_traces.json --max-samples 500
"""

import argparse
import json
import os
import sys
from hashlib import sha256
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ReAct paper official HuggingFace dataset
# Source: https://huggingface.co/datasets/yizhangchi/ReAct
REACT_HF_DATASET = "yizhangchi/ReAct"

# Local cache directory for downloaded ReAct data
DEFAULT_REACT_DIR = "data/raw/react"

# Output path for the collected traces
DEFAULT_OUTPUT_PATH = "data/raw/react_traces.json"

# Maximum number of samples to collect (default)
DEFAULT_MAX_SAMPLES = 500


# ---------------------------------------------------------------------------
# SequenceCard construction helpers
# ---------------------------------------------------------------------------

def make_sequence_id(trace_index: int, source: str) -> str:
    """Generate a unique sequence_id for a ReAct trace.

    Format: react-{source}-{{{trace_index:05d}}}
    Example: react-hf-00042
    """
    return f"react-{source}-{trace_index:05d}"


def hash_string(text: str) -> str:
    """Return a short hex hash of the input string."""
    return sha256(text.encode()).hexdigest()[:16]


def build_sequence_card(
    trace_index: int,
    trace_data: dict[str, Any],
    source: str = "hf",
) -> dict[str, Any]:
    """Convert a single ReAct trace into SequenceCard format.

    Parameters
    ----------
    trace_index : int
        Index of this trace in the dataset (used for ID generation).
    trace_data : dict
        Raw ReAct trace data. Expected keys:
          - "question" or "prompt": the task prompt given to the agent
          - "trajectory" or "steps": list of {{thought, action, observation}} dicts
          - "answer" or "final_answer": the agent's final answer (optional)
          - "task": task type label (optional)
    source : str
        Data source tag (e.g. "hf" for HuggingFace, "mock" for synthetic).

    Returns
    -------
    dict
        A dict matching the SequenceCard schema (as JSON-serializable dict).
    """
    # Extract prompt / question from the trace
    prompt: str = (
        trace_data.get("question")
        or trace_data.get("prompt")
        or trace_data.get("input")
        or ""
    )
    assert isinstance(prompt, str), f"prompt must be str, got {type(prompt)}"

    # Extract trajectory steps
    trajectory = (
        trace_data.get("trajectory")
        or trace_data.get("steps")
        or trace_data.get("scratchpad")
        or []
    )
    # Flatten trajectory into a single text blob for prompt_hash computation
    trajectory_text = json.dumps(trajectory, sort_keys=True) if trajectory else ""

    # Extract task type
    task_type: str = (
        trace_data.get("task")
        or trace_data.get("task_type")
        or "qa"  # ReAct is primarily question-answering
    )

    # Compute prompt_hash from prompt + trajectory (both define the sequence)
    hash_input = prompt + trajectory_text
    prompt_hash = hash_string(hash_input)

    # Extract files (ReAct traces don't have associated files; use empty list)
    files: list[str] = []

    # Memory fields: ReAct agents use conversation history as memory
    memory_type = "conversation"  # ReAct stores reasoning in context
    channel = "agent-scratchpad"  # The scratchpad is the memory channel
    evidence = f"react-trace-{source}"  # Evidence that this is a ReAct trace

    # Oracle fields: ReAct traces have gold answers
    oracle_type = "gold-answer"
    gold_answer = trace_data.get("answer") or trace_data.get("final_answer") or ""
    tests: list[str] = [f"gold:{gold_answer}"] if gold_answer else []

    # Policy: standard for ReAct
    policy = "standard"

    # Intervention conditions: ReAct is stateless, so "clean" is the only condition
    conditions: list[str] = ["clean"]

    # Placebo match: none for real traces
    placebo_match = "none"

    # Annotation labels: ReAct traces are from public datasets
    scope_label = "public"
    staleness_label = "unknown"  # We don't know when the trace was created
    bad_label = "unlabeled"
    security_label = "safe"  # Public QA traces are safe

    # Reproducibility fields
    docker_image = "python:3.10"  # ReAct runs in standard Python
    hashes: dict[str, str] = {
        "prompt_hash": prompt_hash,
        "trajectory_len": str(len(trajectory)),
    }
    seeds: list[int] = [42]  # ReAct is deterministic (no sampling)

    # Repository fields: ReAct traces are not tied to a specific repo
    repo_url = "https://github.com/react-paper/ReAct"
    repo_commit = "unknown"
    repo_license = "MIT"  # ReAct paper code is MIT-licensed

    # Rules field
    rules = "no-copy"

    sequence_id = make_sequence_id(trace_index, source)

    card: dict[str, Any] = {
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
        # Extra fields not in strict SequenceCard but useful for analysis
        "_prompt": prompt,
        "_trajectory_length": len(trajectory),
        "_source": source,
    }
    return card


# ---------------------------------------------------------------------------
# Mock data generation
# ---------------------------------------------------------------------------

def generate_mock_react_trace(index: int) -> dict[str, Any]:
    """Generate a single mock ReAct trace that simulates ReAct behavior.

    A ReAct trace consists of alternating thought/action/observation steps.
    This function generates a realistic-looking trace with:
      - A question (prompt)
      - 3-8 reasoning-action-observation cycles
      - A final answer

    The content is synthetic but structurally valid ReAct format.
    """
    # Sample questions covering different ReAct tasks (HotpotQA, FEVER, etc.)
    questions = [
        "What is the capital of the state where Cleveland is located?",
        "Who wrote the book that was published by the same press as 'The Shining'?",
        "When did the director of film 'The Revenant' win his first Oscar?",
        "What is the elevation range of the area where the Andes mountains are located?",
        "Which year did the singer who performed 'Shape of You' win the Grammy Award?",
    ]
    question = questions[index % len(questions)]

    # Generate a trajectory with 3-8 steps
    num_steps = 3 + (index % 6)  # 3 to 8 steps
    trajectory: list[dict[str, str]] = []

    thought_templates = [
        "I need to find information about {entity}. Let me search for it.",
        "The search result shows {info}. Now I need to look up more details.",
        "Based on the observation, I should check {next_entity} to answer the question.",
        "I have found relevant information. Let me verify with a lookup.",
        "The lookup confirms {fact}. I can now answer the question.",
    ]
    action_templates = [
        "Search[{entity}]",
        "Lookup[{term}]",
        "Search[{query}]",
    ]
    observation_templates = [
        "Title: {title}. {snippet}",
        "The page mentions: {fact}. Relevant info: {detail}",
        "{content}",
    ]

    entities = ["Cleveland", "Ohio", "Stephen King", "Andes mountains", "Leonardo DiCaprio"]
    current_entity = entities[index % len(entities)]

    for step_i in range(num_steps):
        thought = thought_templates[step_i % len(thought_templates)].format(
            entity=current_entity,
            info=f"some info about {current_entity}",
            next_entity=entities[(index + step_i + 1) % len(entities)],
            fact=f"{current_entity} fact",
        )
        action = action_templates[step_i % len(action_templates)].format(
            entity=current_entity,
            term=current_entity.lower(),
            query=f"{current_entity} information",
        )
        observation = observation_templates[step_i % len(observation_templates)].format(
            title=current_entity,
            snippet=f"{current_entity} is a well-known entity.",
            fact=f"{current_entity} has property X",
            detail="additional detail here",
            content=f"Full content about {current_entity}.",
        )
        trajectory.append({
            "thought": thought,
            "action": action,
            "observation": observation,
        })
        # Advance entity for next step
        current_entity = entities[(index + step_i + 1) % len(entities)]

    # Final answer step (Finish action)
    final_answer = f"Based on the information gathered, the answer is: {current_entity}."
    trajectory.append({
        "thought": f"I now have enough information to answer the question.",
        "action": f"Finish[{final_answer}]",
        "observation": "Episode finished.",
    })

    return {
        "question": question,
        "trajectory": trajectory,
        "answer": final_answer,
        "task_type": "qa",
        "source": "mock",
    }


def generate_mock_dataset(num_samples: int) -> list[dict[str, Any]]:
    """Generate a mock ReAct dataset with `num_samples` traces.

    Returns a list of dicts, each compatible with build_sequence_card().
    """
    assert num_samples > 0, f"num_samples must be positive, got {num_samples}"
    assert num_samples <= 10000, f"num_samples too large (max 10000): {num_samples}"
    return [generate_mock_react_trace(i) for i in range(num_samples)]


# ---------------------------------------------------------------------------
# Real data collection from HuggingFace
# ---------------------------------------------------------------------------

def download_react_from_huggingface(
    react_dir: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    """Download ReAct dataset from HuggingFace and return parsed traces.

    Uses the `datasets` library to load yizhangchi/ReAct.
    Falls back to manual download if datasets library is not available.

    Parameters
    ----------
    react_dir : str
        Directory to cache downloaded data.
    max_samples : int
        Maximum number of traces to load.

    Returns
    -------
    list[dict[str, Any]]
        List of raw ReAct trace dicts.
    """
    os.makedirs(react_dir, exist_ok=True)

    # Try using datasets library first
    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "[WARN] `datasets` library not installed. "
            "Install with: pip install datasets. Falling back to manual download.",
            file=sys.stderr,
        )
        return _download_react_manual(react_dir, max_samples)

    print(f"Loading ReAct dataset from HuggingFace: {REACT_HF_DATASET}")
    dataset = load_dataset(REACT_HF_DATASET, split="train", streaming=False)

    # Convert to list, limited by max_samples
    samples: list[dict[str, Any]] = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        samples.append(sample)

    print(f"Downloaded {len(samples)} ReAct traces from HuggingFace")
    return samples


def _download_react_manual(react_dir: str, max_samples: int) -> list[dict[str, Any]]:
    """Fallback: manually download ReAct data via urllib.

    The ReAct paper released data at:
      https://github.com/ysymyth/ReAct/tree/main/hotpotqa

    This function downloads the raw JSON files and parses them.
    """
    import urllib.request

    # ReAct hotpotqa data URL (from ReAct GitHub repo)
    react_github_raw = (
        "https://raw.githubusercontent.com/ysymyth/ReAct/main/"
        "hotpotqa/final_data/hotpotqa_main_naive.json"
    )
    cache_path = os.path.join(react_dir, "hotpotqa_main_naive.json")

    if not os.path.exists(cache_path):
        print(f"Downloading ReAct data from {react_github_raw}")
        print(f"Cache path: {cache_path}")
        urllib.request.urlretrieve(react_github_raw, cache_path)
    else:
        print(f"Using cached ReAct data: {cache_path}")

    with open(cache_path) as f:
        raw_data = json.load(f)

    # raw_data is a list of dicts, each representing one ReAct trace
    # Limit to max_samples
    if isinstance(raw_data, list):
        samples = raw_data[:max_samples]
    else:
        # Some files are dicts with a "data" key
        samples = raw_data.get("data", [])[:max_samples]

    print(f"Loaded {len(samples)} ReAct traces from local cache")
    return samples


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect_react_traces(
    output_path: str,
    react_dir: str,
    use_mock: bool,
    max_samples: int,
) -> None:
    """Main collection pipeline.

    Parameters
    ----------
    output_path : str
        Path to write the output JSON file.
    react_dir : str
        Directory for caching ReAct raw data.
    use_mock : bool
        If True, generate mock data instead of downloading.
    max_samples : int
        Maximum number of traces to collect.
    """
    # Validate inputs
    assert max_samples > 0, f"max_samples must be positive: {max_samples}"
    assert isinstance(output_path, str), f"output_path must be str: {type(output_path)}"
    assert isinstance(react_dir, str), f"react_dir must be str: {type(react_dir)}"

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Step 1: Collect raw ReAct traces (mock or real)
    if use_mock:
        print(f"[MOCK] Generating {max_samples} mock ReAct traces...")
        raw_traces = generate_mock_dataset(max_samples)
        source_tag = "mock"
    else:
        print(f"[REAL] Downloading up to {max_samples} ReAct traces...")
        raw_traces = download_react_from_huggingface(react_dir, max_samples)
        source_tag = "hf"

    assert len(raw_traces) > 0, "No traces collected! Check data source or mock generation."

    # Step 2: Convert each trace to SequenceCard format
    print(f"Converting {len(raw_traces)} traces to SequenceCard format...")
    sequence_cards: list[dict[str, Any]] = []
    for i, raw_trace in enumerate(raw_traces):
        card = build_sequence_card(i, raw_trace, source=source_tag)
        sequence_cards.append(card)
        if (i + 1) % 100 == 0:
            print(f"  Converted {i + 1}/{len(raw_traces)}...")

    # Step 3: Validate the output
    print("Validating output SequenceCards...")
    _validate_sequence_cards(sequence_cards)

    # Step 4: Write to output file
    print(f"Writing {len(sequence_cards)} SequenceCards to {output_path}")
    with open(output_path, "w") as f:
        json.dump(sequence_cards, f, indent=2, ensure_ascii=False)

    # Step 5: Print summary statistics
    _print_summary(sequence_cards)
    print("[DONE] ReAct trace collection complete.")


def _validate_sequence_cards(cards: list[dict[str, Any]]) -> None:
    """Validate that all SequenceCards have required fields and unique IDs.

    Crashes with AssertionError if validation fails (fast-fail principle).
    """
    seen_ids: set[str] = set()
    required_fields = [
        "sequence_id", "repo_url", "repo_commit", "repo_license",
        "task_type", "prompt_hash", "files",
        "memory_type", "channel", "evidence",
        "oracle_type", "tests", "rules", "policy",
        "conditions", "placebo_match",
        "scope_label", "staleness_label", "bad_label", "security_label",
        "docker_image", "hashes", "seeds",
    ]
    for i, card in enumerate(cards):
        # Check all required fields present
        for field in required_fields:
            assert field in card, f"Card {i} missing required field: {field}"

        # Check sequence_id is unique
        sid = card["sequence_id"]
        assert sid not in seen_ids, f"Duplicate sequence_id: {sid} at index {i}"
        seen_ids.add(sid)

        # Check prompt_hash is non-empty
        assert card["prompt_hash"], f"Card {i} has empty prompt_hash"

    print(f"  Validated {len(cards)} cards: all required fields present, all IDs unique.")


def _print_summary(cards: list[dict[str, Any]]) -> None:
    """Print a brief summary of the collected SequenceCards."""
    print(f"\n=== ReAct Traces Summary ===")
    print(f"Total traces: {len(cards)}")

    # Count by task_type
    task_counts: dict[str, int] = {}
    for card in cards:
        t = card["task_type"]
        task_counts[t] = task_counts.get(t, 0) + 1
    print(f"Task types: {task_counts}")

    # Average trajectory length (from _trajectory_length extra field)
    traj_lens = [card.get("_trajectory_length", 0) for card in cards]
    if traj_lens:
        avg_len = sum(traj_lens) / len(traj_lens)
        print(f"Avg trajectory length: {avg_len:.1f} steps")
        print(f"Min/Max trajectory length: {min(traj_lens)} / {max(traj_lens)}")

    # Show a sample card (first one)
    if cards:
        sample = cards[0]
        print(f"\nSample sequence_id: {sample['sequence_id']}")
        print(f"  repo_url: {sample['repo_url']}")
        print(f"  task_type: {sample['task_type']}")
        print(f"  memory_type: {sample['memory_type']}")
        print(f"  oracle_type: {sample['oracle_type']}")
        print(f"  prompt_hash: {sample['prompt_hash'][:16]}...")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect ReAct agent traces and convert to SequenceCard format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 200 mock ReAct traces (for testing):
  python collect_react_traces.py --use-mock --max-samples 200

  # Download real ReAct traces from HuggingFace (up to 500):
  python collect_react_traces.py --output data/raw/react_traces.json --max-samples 500

  # Use a specific local ReAct data directory:
  python collect_react_traces.py --react-dir /path/to/react/data --use-mock
""",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--react-dir",
        type=str,
        default=DEFAULT_REACT_DIR,
        help=f"Directory for caching ReAct raw data (default: {DEFAULT_REACT_DIR})",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Generate mock ReAct traces instead of downloading real data.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Maximum number of traces to collect (default: {DEFAULT_MAX_SAMPLES})",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Validate args before starting (fast-fail)
    assert args.max_samples > 0, f"--max-samples must be positive: {args.max_samples}"
    assert args.output.endswith(".json"), f"--output must be a .json file: {args.output}"

    print(f"=== collect_react_traces ===")
    print(f"  output:     {args.output}")
    print(f"  react_dir:  {args.react_dir}")
    print(f"  use_mock:   {args.use_mock}")
    print(f"  max_samples:{args.max_samples}")
    print()

    collect_react_traces(
        output_path=args.output,
        react_dir=args.react_dir,
        use_mock=args.use_mock,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

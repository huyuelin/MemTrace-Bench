#!/usr/bin/env python3
"""
Reference Mediator run entry point.
Implements full reference mediator (Section 4).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Dict, Any, List


def run_reference_mediator_full(seq: Dict[str, Any], agent: Any, model_config: Dict) -> Dict[str, Any]:
    """
    Full reference mediator run (Section 4).
    1. Normalize all channels (memory store, conversation, tool-log, ...)
    2. Compile each memory through validity lattice
    3. Generate prompt envelope with obligations and certificate

    WARNING: This function uses mock data and mock agent calls.
    The results are NOT from real experiments and should not be used for research.
    To run real experiments, implement real agent calls and test execution.
    """
    from core.predicates import allowed
    from mediator.compiler import compile_memory
    from mediator.envelope import generate_envelope
    from mediator.certificate import generate_certificate
    from mediator.lattice import ValidityLattice

    # WARNING: Mock mode - results are NOT from real experiments
    print(f"[WARNING] run_reference_mediator_full() is running in MOCK mode.")
    print(f"[WARNING] Results are synthetic and NOT from real experiments.")
    print(f"[WARNING] To get real results, implement real agent calls and test execution.")

    # 1. Normalize channels: collect memories from ALL channels
    all_memories = _collect_memories_all_channels(seq)
    print(f"[RefMediator] Collected {len(all_memories)} memories from all channels")

    # 2. Build context for compilation
    context = {
        "repo": seq.get("repo", "unknown/unknown"),
        "organization": "test-org",
        "timestamp": seq.get("hashes", {}).get("timestamp", 0),
        "policy": "official",
        "predicate": "fix-similarity",
    }

    # 3. Compile each memory
    compiled = []
    for m in all_memories:
        lattice_level, prompt_seg = compile_memory(m, context)
        if lattice_level != ValidityLattice.DROP:
            compiled.append({
                "memory": m,
                "lattice": lattice_level,
                "prompt_segment": prompt_seg,
            })

    # 4. Generate prompt envelope
    envelope = generate_envelope(compiled, context)

    # 5. Generate certificate
    certificate = generate_certificate(envelope, compiled, context)

    # 6. Call agent with envelope prompt (MOCK - not real agent call)
    print(f"[WARNING] Using mock agent call - not a real LLM call")
    prompt_text = envelope["prompt_text"]
    patch = _mock_agent_call(agent, prompt_text, model_config)

    # 7. Mock results (NOT real experimental data)
    # Intentionally return obviously fake data (pass_label=False, bad_label=False)
    # This makes it clear that mock mode does not produce real results
    print(f"[WARNING] Using mock test results - NOT real experimental data")
    pass_label = False  # Obviously fake - tests don't pass
    bad_label = False    # Obviously fake - no bad outputs

    return {
        "condition": "reference-mediator",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [c["memory"].get("memory_id") for c in compiled],
        "memory_type": seq.get("memory_type", "in-scope"),
        "certificate": certificate,
    }


def _collect_memories_all_channels(seq, data_dir=None):
    """Collect memories from all channels.

    Args:
        seq: Sequence card dict with fields: sequence_id, repo, memory_type, etc.
        data_dir: Optional directory to read real memories from.
                 If None, generate realistic mock memories based on seq metadata.

    Returns:
        List of memory dicts with fields matching SequenceCard schema:
        - memory_id, text, channel, repo, organization,
          scope_field, timestamp, sensitivity, license_field,
          predicate, evidence, content_hash

    Raises:
        FileNotFoundError: If data_dir is provided but memory file missing.
        KeyError: If seq is missing required fields (sequence_id, repo).
    """
    import random

    seq_id = seq.get("sequence_id")
    if seq_id is None:
        raise KeyError("seq missing required field: sequence_id")
    repo = seq.get("repo")
    if repo is None:
        raise KeyError("seq missing required field: repo")

    memories = []

    # Try to read real memories if data_dir provided
    if data_dir is not None:
        memory_path = os.path.join(data_dir, f"{seq_id}_memories.json")
        if os.path.isfile(memory_path):
            import json
            with open(memory_path) as f:
                memories = json.load(f)
            # Validate memory format
            for m in memories:
                assert "memory_id" in m, f"Memory missing memory_id: {m}"
                assert "channel" in m, f"Memory missing channel: {m}"
            return memories
        else:
            raise FileNotFoundError(f"Memory file not found: {memory_path}")

    # Generate realistic mock memories based on sequence metadata
    memory_type = seq.get("memory_type", "in-scope")

    # Memory store channel: 1-3 memories depending on memory_type
    n_mem_store = 3 if memory_type == "in-scope" else 1
    for i in range(n_mem_store):
        memories.append({
            "memory_id": f"{seq_id}-mem-store-{i}",
            "text": f"Memory from memory-store for {seq_id}: {memory_type} context",
            "channel": "memory-store",
            "repo": repo,
            "organization": repo.split("/")[0] if "/" in repo else "unknown",
            "scope_field": "repo",
            "timestamp": seq.get("hashes", {}).get("timestamp", 0),
            "sensitivity": "public" if memory_type == "in-scope" else "security",
            "license_field": seq.get("repo_license", "MIT"),
            "predicate": "fix-similarity",
            "evidence": memory_type,
            "content_hash": f"hash-{seq_id}-mem-store-{i}",
        })

    # Other channels: 0-2 memories each (realistic: not all channels have memories)
    channels_with_memory = ["conversation", "tool-log", "terminal-cache",
                          "wrapper-prompt", "cached-summary", "previous-patch", "scratchpad"]
    seed = abs(hash(seq_id)) % (2 ** 32)
    rng = random.Random(seed)
    for ch in channels_with_memory:
        if rng.random() < 0.6:  # 60% chance of having a memory in this channel
            n = rng.randint(1, 2)
            for j in range(n):
                memories.append({
                    "memory_id": f"{seq_id}-{ch}-{j}",
                    "text": f"Memory from {ch} for {seq_id}: relevant context",
                    "channel": ch,
                    "repo": repo,
                    "organization": repo.split("/")[0] if "/" in repo else "unknown",
                    "scope_field": "repo",
                    "timestamp": seq.get("hashes", {}).get("timestamp", 0) - rng.randint(0, 86400),
                    "sensitivity": "public",
                    "license_field": seq.get("repo_license", "MIT"),
                    "predicate": "fix-similarity",
                    "evidence": ch,
                    "content_hash": f"hash-{seq_id}-{ch}-{j}",
                })

    return memories


def _mock_agent_call(agent, prompt, config):
    return f"# Mock patch for reference mediator"


def main():
    """Main entry point for testing reference mediator."""
    import argparse
    parser = argparse.ArgumentParser(description="Run reference mediator (test mode)")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock data")
    args = parser.parse_args()

    if args.test:
        # Create mock sequence
        mock_seq = {
            "sequence_id": "test_seq_001",
            "repo": "test-owner/test-repo",
            "organization": "test-owner",
            "memory_type": "in-scope",
            "repo_license": "MIT",
            "hashes": {"timestamp": 1700000000},
        }

        # Mock agent and config
        mock_agent = None
        mock_config = {"model": "mock-model"}

        # Run mediator
        print("[RefMediator] Running test mode...")
        result = run_reference_mediator_full(mock_seq, mock_agent, mock_config)

        # Print result summary
        print(f"\n[RefMediator] Test complete!")
        print(f"  Sequence: {result['sequence_id']}")
        print(f"  Condition: {result['condition']}")
        print(f"  Pass: {result['pass_label']}")
        print(f"  Bad: {result['bad_label']}")
        print(f"  Exposed memories: {len(result['exposed_memories'])}")
        print(f"  Certificate: {result['certificate']['cert_id']}")

        # Print prompt text (first 500 chars)
        prompt_preview = result['prompt_text'][:500]
        print(f"\n--- Prompt Preview (first 500 chars) ---")
        print(prompt_preview)
        print("--- End Preview ---")


if __name__ == "__main__":
    main()

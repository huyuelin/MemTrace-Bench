#!/usr/bin/env python3
"""
Test script for Phase 2 GitHubAgent implementation.

Tests:
1. GitHubAgent can be instantiated with use_real_llm=False, use_real_tools=False (mock mode)
2. GitHubAgent.run() works with a dummy sequence
3. GitHubAgent can be instantiated with use_real_tools=True (real tools mode)
"""

import sys
import os

# Add code dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.github_agent import GitHubAgent


def test_mock_mode():
    """Test GitHubAgent in mock mode (use_real_llm=False, use_real_tools=False)."""
    print("=== Test 1: GitHubAgent mock mode ===")

    agent = GitHubAgent(
        model="gpt-4",
        temperature=0.0,
        max_tokens=1024,
        top_k=50,
        seed=42,
        use_real_llm=False,
        use_real_tools=False,
    )
    print(f"  Agent created: {agent.__class__.__name__}")
    print(f"  use_real_llm: {agent.use_real_llm}")
    print(f"  use_real_tools: {agent.use_real_tools}")
    assert agent.use_real_llm == False
    assert agent.use_real_tools == False

    # Run with dummy sequence
    dummy_sequence = {
        "sequence_id": "test_seq_001",
        "repo_url": "https://github.com/test/repo",
        "repo_commit": "abc123",
        "task_type": "bugfix",
        "files": ["foo.py", "bar.py"],
        "tests": ["test_foo.py"],
        "issue_text": "The foo function returns None instead of True",
        "issue_id": "123",
        "issue_title": "Fix foo function",
    }

    try:
        manifest = agent.run(dummy_sequence)
        print(f"  run() returned RunManifest: {manifest.sequence_id}")
        assert manifest.sequence_id == "test_seq_001"
        print("  [PASS] mock mode run() works")
    except Exception as e:
        print(f"  [FAIL] run() raised: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_real_tools_mode():
    """Test GitHubAgent with use_real_tools=True (requires git repo clone).

    Note: This test will try to clone a repo, which requires network and disk space.
    For now, just test that the agent can be instantiated.
    """
    print("\n=== Test 2: GitHubAgent real_tools mode (init only) ===")

    try:
        agent = GitHubAgent(
            model="gpt-4",
            temperature=0.0,
            max_tokens=1024,
            top_k=50,
            seed=42,
            use_real_llm=False,
            use_real_tools=True,  # Real tools, but mock LLM
        )
        print(f"  Agent created with use_real_tools=True")
        assert agent.use_real_tools == True
        print("  [PASS] real_tools mode init works")
    except Exception as e:
        print(f"  [FAIL] init raised: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("Running Phase 2 GitHubAgent tests...\n")

    results = []

    results.append(("Mock mode", test_mock_mode()))
    results.append(("Real tools init", test_real_tools_mode()))

    print("\n=== Test Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    sys.exit(0 if all_passed else 1)

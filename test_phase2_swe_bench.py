#!/usr/bin/env python3
"""
Test script for Phase 2 SWEBenchAgent implementation.

Tests:
1. SWEBenchAgent can be instantiated with use_real_llm=False, use_real_tools=False (mock mode)
2. SWEBenchAgent.run() works with a dummy sequence
"""

import sys
import os

# Add code dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.swe_bench_agent import SWEBenchAgent, SWEBenchTask


def test_mock_mode():
    """Test SWEBenchAgent in mock mode."""
    print("=== Test 1: SWEBenchAgent mock mode ===")

    agent = SWEBenchAgent(
        model="gpt-4",
        temperature=0.0,
        max_tokens=1024,
        top_k=50,
        seed=42,
        docker_enabled=False,
        max_iterations=3,
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
        "sequence_id": "swe_bench_test_001",
        "swe_bench_instance": {
            "instance_id": "django__django-12345",
            "repo": "django/django",
            "base_commit": "abc123def",
            "patch": "--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,4 @@\n+fix\n",
            "test_patch": "--- a/test_foo.py\n+++ b/test_foo.py\n@@ -1,3 +1,4 @@\n+test\n",
            "FAIL_TO_PASS": ["test_foo"],
            "PASS_TO_PASS": ["test_bar"],
            "problem_statement": "The foo function is broken",
        },
    }

    try:
        manifest = agent.run(dummy_sequence)
        print(f"  run() returned RunManifest: {manifest.sequence_id}")
        assert manifest.sequence_id == "swe_bench_test_001"
        print("  [PASS] mock mode run() works")
    except Exception as e:
        print(f"  [FAIL] run() raised: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    print("Running Phase 2 SWEBenchAgent tests...\n")

    results = []
    results.append(("Mock mode", test_mock_mode()))

    print("\n=== Test Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

    sys.exit(0 if all_passed else 1)

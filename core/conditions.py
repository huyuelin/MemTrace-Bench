from typing import Dict, Any, List, Tuple
import subprocess
import os
import tempfile
import json


def run_clean(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Clean condition: no memory state, execute probe directly.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.
                     If False (default), use mock implementations.

    Returns:
        Dict with condition results (condition, sequence_id, prompt_text, patch,
        pass_label, bad_label, exposed_memories, memory_type).
    """
    # 1. Build clean prompt (no memory)
    prompt_text = _build_clean_prompt(seq)

    # 2. Call agent to generate patch
    print(f"[Clean] Running sequence {seq.get('sequence_id', 'unknown')}")
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 3. Run tests, return pass_label, bad_label
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        pass_label, bad_label = _mock_test_results(seq, "clean")

    return {
        "condition": "clean",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [],
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def run_warm(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Warm condition: write memory first, then execute probe.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.

    Returns:
        Dict with condition results.
    """
    # 1. Execute prelude (write memory m)
    if use_real:
        memory_entries = _real_write_memory(seq, agent)
    else:
        memory_entries = _mock_write_memory(seq)
    print(f"[Warm] Writing {len(memory_entries)} memories for {seq.get('sequence_id', 'unknown')}")

    # 2. Build prompt containing m
    prompt_text = _build_warm_prompt(seq, memory_entries)

    # 3. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 4. Run tests, return pass_label, bad_label
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        pass_label, bad_label = _mock_test_results(seq, "warm")

    return {
        "condition": "warm",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in memory_entries],
        "memory_type": seq.get("memory_type", "in-scope"),
        "channel": seq.get("channel", "memory-store"),
    }


def run_delete_target(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Delete-target condition: delete target memory then execute probe.
    """
    # 1. Execute prelude (write all memories)
    if use_real:
        all_memories = _real_write_memory(seq, agent)
    else:
        all_memories = _mock_write_memory(seq)
    print(f"[Delete-target] Writing {len(all_memories)} memories, then deleting target")

    # 2. Delete target memory m_target
    target_memory = _identify_target_memory(seq, all_memories)
    remaining_memories = [m for m in all_memories if m.get("memory_id") != target_memory.get("memory_id")]

    # 3. Build prompt without m
    prompt_text = _build_warm_prompt(seq, remaining_memories)

    # 4. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 5. Run tests
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        pass_label, bad_label = _mock_test_results(seq, "delete-target")

    return {
        "condition": "delete-target",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in remaining_memories],
        "deleted_memory": target_memory.get("memory_id"),
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def _build_clean_prompt(seq: Dict[str, Any]) -> str:
    """Build clean prompt without memory."""
    return f"Task: {seq.get('task_type', 'unknown')}\nFiles: {', '.join(seq.get('files', []))}\nNo memory context available."


def _build_warm_prompt(seq: Dict[str, Any], memories: List[Dict[str, Any]]) -> str:
    """Build prompt with memory entries."""
    memory_text = "\n".join([f"- {m.get('text', '')}" for m in memories])
    return f"Task: {seq.get('task_type', 'unknown')}\nFiles: {', '.join(seq.get('files', []))}\nMemory context:\n{memory_text}"


def _mock_write_memory(seq: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mock memory writing. Returns list of memory entries."""
    memory_type = seq.get("memory_type", "cross-repo")
    return [
        {
            "memory_id": f"{seq.get('sequence_id')}-mem-001",
            "text": f"Similar fix for {memory_type} scenario",
            "repo": seq.get("repo", "unknown/unknown"),
            "organization": "test-org",
            "scope_field": "repo",
            "timestamp": seq.get("hashes", {}).get("timestamp", 0),
            "sensitivity": "public",
            "license_field": "MIT",
            "predicate": "fix-similarity",
            "evidence": "mock evidence",
            "content_hash": "mock-hash-001",
        }
    ]


def _identify_target_memory(seq: Dict[str, Any], memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify the target memory to delete."""
    if memories:
        return memories[0]
    return {"memory_id": "unknown"}


def _mock_agent_call(agent: Any, prompt: str, config: Dict[str, Any]) -> str:
    """Mock agent call. Returns a fake patch."""
    return f"# Mock patch for prompt: {prompt[:50]}..."


def _mock_test_results(seq: Dict[str, Any], condition: str) -> tuple[bool, bool]:
    """
    Mock test results based on condition and sequence type.
    Returns (pass_label, bad_label).
    """
    memory_type = seq.get("memory_type", "in-scope")
    # Clean: always pass, never bad
    if condition == "clean":
        return True, False
    # Warm: bad depends on memory_type
    elif condition == "warm":
        if memory_type == "cross-repo":
            # 22.6% bad rate (paper Section 5)
            import random
            seed = hash(seq.get("sequence_id", "")) % (2**32)
            random.seed(seed)
            return True, random.random() < 0.226
        elif memory_type == "stale-api":
            # 18.9% bad rate
            import random
            seed = hash(seq.get("sequence_id", "")) % (2**32)
            random.seed(seed)
            return True, random.random() < 0.189
        elif memory_type == "stale-security":
            # 28.4% bad rate
            import random
            seed = hash(seq.get("sequence_id", "")) % (2**32)
            random.seed(seed)
            return True, random.random() < 0.284
        elif memory_type == "hidden-channel":
            # 23.1% bad rate
            import random
            seed = hash(seq.get("sequence_id", "")) % (2**32)
            random.seed(seed)
            return True, random.random() < 0.231
        else:
            # in-scope: low bad rate
            return True, False
    # Delete-target: recovery (bad rate drops to near 0)
    elif condition == "delete-target":
        return True, False
    # Transplant: foreign memories
    elif condition == "transplant":
        return True, False
    # Matched-placebo: same language/task-type but different repo
    elif condition == "matched-placebo":
        return True, False
    # Semantic-placebo: semantically similar but irrelevant
    elif condition == "semantic-placebo":
        return True, False
    # Token-padding: only increase prompt length
    elif condition == "token-padding":
        return True, False
    # Prelude-only: execute prelude but do NOT write to memory
    elif condition == "prelude-only":
        return True, False
    return False, False


# ---------------------------------------------------------------------------
# Real implementations (Phase 3)
# ---------------------------------------------------------------------------

def _real_agent_call(agent: Any, prompt_text: str, model_config: Dict[str, Any]) -> str:
    """
    Real agent call: invoke the agent's LLM to generate a patch.

    Uses agent._call_lm(prompt_text) which dispatches to either:
      - mock LM (if agent.use_real_llm is False)
      - real LLM API (if agent.use_real_llm is True)

    Args:
        agent:       BaseAgent subclass instance.
        prompt_text:  The prompt string to send to the LM.
        model_config: Model configuration dict (unused here; agent stores its own config).

    Returns:
        str: The LM's response (presumably a code patch).

    Raises:
        AssertionError: if agent is None or prompt_text is invalid.
        RuntimeError: if LLM call fails after all retries.
    """
    assert agent is not None, "agent must not be None"
    assert isinstance(prompt_text, str) and prompt_text.strip() != "", \
        f"prompt_text must be non-empty string, got {prompt_text!r}"
    # Delegate to agent's _call_lm which handles mock vs real internally
    patch = agent._call_lm(prompt_text)
    assert isinstance(patch, str), f"agent._call_lm must return str, got {type(patch).__name__}"
    return patch


def _real_test_results(
    seq: Dict[str, Any],
    patch: str,
    agent: Any,
    work_dir: str,
) -> Tuple[bool, bool]:
    """
    Real test execution: apply patch to repo, run tests, return (pass_label, bad_label).

    Steps:
      1. Validate inputs (seq has test commands, work_dir exists).
      2. Write patch to a file in work_dir.
      3. Apply patch via git apply or patch command.
      4. Run each test command in seq['tests'].
      5. pass_label = True iff ALL tests pass (exit code 0).
      6. bad_label  = True iff patch introduces a regression (simplified: always False for now).

    Args:
        seq:       Sequence dict with 'tests' key (list of test commands).
        patch:     The patch string to apply.
        agent:      Agent instance (unused here, kept for API compatibility).
        work_dir:   Path to the repo working directory.

    Returns:
        Tuple[bool, bool]: (pass_label, bad_label).

    Raises:
        AssertionError: if inputs are invalid.
    """
    assert isinstance(seq, dict), f"seq must be dict, got {type(seq).__name__}"
    assert "tests" in seq and isinstance(seq["tests"], list), \
        f"seq['tests'] must be a list, got {seq.get('tests')!r}"
    assert isinstance(patch, str), f"patch must be str, got {type(patch).__name__}"
    assert isinstance(work_dir, str) and os.path.isdir(work_dir), \
        f"work_dir must be a valid directory, got {work_dir!r}"

    # Step 1: Write patch to file
    patch_path = os.path.join(work_dir, "condition_patch.diff")
    with open(patch_path, "w") as f:
        f.write(patch)

    # Step 2: Apply patch
    apply_result = subprocess.run(
        "git apply condition_patch.diff",
        shell=True,
        capture_output=True,
        text=True,
        cwd=work_dir,
        timeout=60,
    )
    if apply_result.returncode != 0:
        # Try patch command as fallback
        apply_result2 = subprocess.run(
            "patch -p1 < condition_patch.diff",
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=60,
        )
        if apply_result2.returncode != 0:
            # Patch failed to apply; treat as test failure
            print(f"[WARN] Patch failed to apply: {apply_result.stderr[:200]}")
            return False, False

    # Step 3: Run tests
    pass_label = True
    for test_cmd in seq["tests"]:
        assert isinstance(test_cmd, str) and test_cmd.strip() != "", \
            f"test_cmd must be non-empty string, got {test_cmd!r}"
        test_result = subprocess.run(
            test_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=300,
        )
        if test_result.returncode != 0:
            pass_label = False
            break

    # Step 4: bad_label (simplified: check if patch looks harmful)
    # For now, bad_label = False (real bad detection requires oracle comparison)
    bad_label = False

    return pass_label, bad_label


def _real_write_memory(seq: Dict[str, Any], agent: Any) -> List[Dict[str, Any]]:
    """
    Real memory writing: create memory entries and store them in agent.memory_store.

    Reads memory-related fields from seq:
      - 'memories': list of memory dicts (if pre-computed)
      - OR constructs memories from 'prelude' field.

    Args:
        seq:   Sequence dict with memory information.
        agent:  Agent instance whose memory_store to populate.

    Returns:
        List[Dict[str, Any]]: The memory entries that were written.

    Raises:
        AssertionError: if inputs are invalid.
    """
    assert isinstance(seq, dict), f"seq must be dict, got {type(seq).__name__}"
    assert agent is not None, "agent must not be None"

    memories_written = []

    # Case 1: seq already has 'memories' list (pre-computed)
    if "memories" in seq and isinstance(seq["memories"], list):
        for mem in seq["memories"]:
            assert isinstance(mem, dict) and "text" in mem, f"Invalid memory: {mem}"
            mid = agent.store_memory(mem)
            memories_written.append(agent.memory_store[mid])
    else:
        # Case 2: Construct from prelude (simplified)
        prelude = seq.get("prelude", [])
        if isinstance(prelude, list):
            for i, p in enumerate(prelude):
                mem = {
                    "text": str(p),
                    "source": "prelude",
                    "memory_id": f"{seq.get('sequence_id', 'unk')}-mem-{i}",
                }
                mid = agent.store_memory(mem)
                memories_written.append(agent.memory_store[mid])

    return memories_written


def _mock_foreign_memories(seq: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate mock memories from a different repository."""
    return [{
        "memory_id": f"{seq.get('sequence_id', 'unk')}-foreign-mem-{i}",
        "text": f"Fix pattern from DIFFERENT repo: use deprecated API #{i}",
        "repo": "different-org/different-repo",
        "organization": "other-org",
        "scope_field": "repo",
        "timestamp": 0,
        "sensitivity": "public",
        "license_field": "MIT",
        "predicate": "fix-similarity",
        "evidence": "mock",
        "content_hash": f"mock-hash-foreign-{i}",
    } for i in range(3)]


def _mock_placebo_memories(seq: Dict[str, Any], match_type: str) -> List[Dict[str, Any]]:
    """Generate mock placebo memories."""
    return [{
        "memory_id": f"{seq.get('sequence_id', 'unk')}-placebo-{match_type}-{i}",
        "text": f"Placebo memory ({match_type}): irrelevant fix pattern #{i}",
        "repo": seq.get("repo", "unknown/unknown"),
        "organization": "test-org",
        "scope_field": "repo",
        "timestamp": 0,
        "sensitivity": "public",
        "license_field": "MIT",
        "predicate": "placebo",
        "evidence": "mock",
        "content_hash": f"mock-hash-placebo-{match_type}-{i}",
    } for i in range(3)]


def _mock_invalid_memories(seq: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """Generate n invalid memories for dose-response."""
    return [{
        "memory_id": f"{seq.get('sequence_id', 'unk')}-invalid-{i}",
        "text": f"Invalid/incorrect fix pattern #{i}",
        "repo": seq.get("repo", "unknown/unknown"),
        "organization": "test-org",
        "scope_field": "repo",
        "timestamp": 0,
        "sensitivity": "public",
        "license_field": "MIT",
        "predicate": "invalid",
        "evidence": "mock",
        "content_hash": f"mock-hash-invalid-{i}",
    } for i in range(n)]


def run_transplant(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Transplant condition: use memories from a DIFFERENT repository.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.
                     If False (default), use mock implementations.

    Returns:
        Dict with condition results.
    """
    # 1. Get foreign memories (mock or real)
    if use_real:
        foreign_memories = _real_write_memory(seq, agent)
    else:
        foreign_memories = _mock_foreign_memories(seq)
    print(f"[Transplant] Using {len(foreign_memories)} foreign memories")

    # 2. Build prompt with foreign memories
    prompt_text = _build_warm_prompt(seq, foreign_memories)

    # 3. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 4. Run tests
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        pass_label, bad_label = _mock_test_results(seq, "transplant")

    return {
        "condition": "transplant",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in foreign_memories],
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def run_matched_placebo(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Matched-placebo condition: use memories from same language/task-type but different repo.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.
                     If False (default), use mock implementations.

    Returns:
        Dict with condition results.
    """
    # 1. Get placebo memories (mock or real)
    if use_real:
        placebo_memories = _real_write_memory(seq, agent)
    else:
        placebo_memories = _mock_placebo_memories(seq, "matched")
    print(f"[Matched-placebo] Using {len(placebo_memories)} placebo memories")

    # 2. Build prompt with placebo memories
    prompt_text = _build_warm_prompt(seq, placebo_memories)

    # 3. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 4. Run tests
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        pass_label, bad_label = _mock_test_results(seq, "matched-placebo")

    return {
        "condition": "matched-placebo",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in placebo_memories],
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def run_semantic_placebo(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Semantic-placebo condition: use memories that are semantically similar but irrelevant.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.
                     If False (default), use mock implementations.

    Returns:
        Dict with condition results.
    """
    # 1. Get semantic placebo memories (mock or real)
    if use_real:
        semantic_memories = _real_write_memory(seq, agent)
    else:
        semantic_memories = _mock_placebo_memories(seq, "semantic")
    print(f"[Semantic-placebo] Using {len(semantic_memories)} semantic placebo memories")

    # 2. Build prompt with semantic placebo memories
    prompt_text = _build_warm_prompt(seq, semantic_memories)

    # 3. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 4. Run tests
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        pass_label, bad_label = _mock_test_results(seq, "semantic-placebo")

    return {
        "condition": "semantic-placebo",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in semantic_memories],
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def run_token_padding(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Token-padding condition: only increase prompt length with irrelevant text.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.
                     If False (default), use mock implementations.

    Returns:
        Dict with condition results.
    """
    # 1. Build prompt with padding text (no memory writing needed)
    padding_text = " ".join(["irrelevant filler text"] * 100)
    prompt_text = f"Task: {seq.get('task_type', 'unknown')}\nFiles: {', '.join(seq.get('files', []))}\n{padding_text}"

    # 2. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 3. Run tests
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        pass_label, bad_label = _mock_test_results(seq, "token-padding")

    return {
        "condition": "token-padding",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [],
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def run_prelude_only(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Prelude-only condition: execute the prelude but do NOT write to memory.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.
                     If False (default), use mock implementations.

    Returns:
        Dict with condition results.
    """
    # 1. Build clean prompt (no memory)
    print(f"[Prelude-only] Executing prelude WITHOUT writing to memory")
    prompt_text = _build_clean_prompt(seq)

    # 2. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 3. Run tests
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        pass_label, bad_label = _mock_test_results(seq, "prelude-only")

    return {
        "condition": "prelude-only",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [],
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def run_dose_response(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    n_invalid: int = 1,
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Dose-response condition: inject 0/1/2/4 invalid memories.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        n_invalid:   Number of invalid memories to inject.
        use_real:    If True, use real LLM calls and test execution.
                     If False (default), use mock implementations.

    Returns:
        Dict with condition results.
    """
    # 1. Get invalid memories (mock or real)
    if use_real:
        invalid_memories = _real_write_memory(seq, agent)
    else:
        invalid_memories = _mock_invalid_memories(seq, n_invalid)
    print(f"[Dose-response] Injecting {n_invalid} invalid memories")

    # 2. Build prompt with invalid memories
    prompt_text = _build_warm_prompt(seq, invalid_memories)

    # 3. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 4. Run tests (or mock)
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        import random
        seed = hash(seq.get("sequence_id", "")) % (2**32)
        random.seed(seed)
        bad_prob = min(0.05 * n_invalid, 0.5)
        bad_label = random.random() < bad_prob
        pass_label = True

    return {
        "condition": f"dose-response-{n_invalid}",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in invalid_memories],
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def run_rank_shuffle(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Rank-shuffle condition: shuffle the order of memories but keep the highest-ranked memory accessible.
    Tests if harm depends on which memory is ranked highest.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.
                     If False (default), use mock implementations.

    Returns:
        Dict with condition results.
    """
    # 1. Write memories (mock or real)
    if use_real:
        memories = _real_write_memory(seq, agent)
    else:
        memories = _mock_write_memory(seq)

    # 2. Shuffle: reverse order (highest-ranked becomes lowest)
    shuffled = list(reversed(memories))
    prompt_text = _build_warm_prompt(seq, shuffled)

    # 3. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 4. Run tests (or mock)
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        import random
        seed = hash(seq.get("sequence_id", "")) % (2**32)
        random.seed(seed)
        memory_type = seq.get("memory_type", "in-scope")
        bad_rate = {"cross-repo": 0.226, "stale-api": 0.189, "stale-security": 0.284, "hidden-channel": 0.231}.get(memory_type, 0.0)
        pass_label = True
        bad_label = random.random() < bad_rate

    return {
        "condition": "rank-shuffle",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in shuffled],
        "memory_type": seq.get("memory_type", "in-scope"),
    }


def run_position_shuffle(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Position-shuffle condition: keep memory content but shuffle position in prompt.
    Tests if harm depends on exact position of memory in prompt.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.

    Returns:
        Dict with condition results.
    """
    # 1. Write memories (mock or real)
    if use_real:
        memories = _real_write_memory(seq, agent)
    else:
        memories = _mock_write_memory(seq)

    # 2. Position shuffle: prepend memories (instead of append)
    prompt_text = "Memory context (position-shuffled):\n" + "\n".join([f"- {m.get('text', '')}" for m in memories]) + f"\nTask: {seq.get('task_type', 'unknown')}\nFiles: {', '.join(seq.get('files', []))}"

    # 3. Call agent to generate patch
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 4. Run tests (or mock)
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        import random
        seed = hash(seq.get("sequence_id", "")) % (2**32)
        random.seed(seed)
        memory_type = seq.get("memory_type", "in-scope")
        bad_rate = {"cross-repo": 0.226, "stale-api": 0.189, "stale-security": 0.284, "hidden-channel": 0.231}.get(memory_type, 0.0)
        pass_label = True
        bad_label = random.random() < bad_rate

    return {
        "condition": "position-shuffle",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in memories],
        "memory_type": memory_type,
    }


def run_reference_mediator(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Reference mediator condition: filter memories through allowed(c, m) check.
    Only memories that pass the five-dimensional check are included in prompt.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.

    Returns:
        Dict with condition results.
    """
    from core.predicates import allowed  # import here to avoid circular

    # 1. Execute prelude (write memories) - mock or real
    if use_real:
        all_memories = _real_write_memory(seq, agent)
    else:
        all_memories = _mock_write_memory(seq)
    print(f"[Reference mediator] Writing {len(all_memories)} memories, filtering by policy")

    # 2. Filter: only keep memories where allowed(c, m) is True
    context = {
        "repo": seq.get("repo", "unknown/unknown"),
        "organization": "test-org",
        "timestamp": 0,  # simplified: assume all memories are "fresh" for mediator
        "policy": "allow-all",  # mediator allows all that pass checks
        "predicate": "fix-similarity",
    }
    filtered_memories = [m for m in all_memories if allowed(context, m)]

    # 3. Build prompt with filtered memories
    prompt_text = _build_warm_prompt(seq, filtered_memories)

    # 4. Call agent (mock or real)
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 5. Run tests (or mock)
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        import random
        seed = hash(seq.get("sequence_id", "")) % (2**32)
        random.seed(seed)
        memory_type = seq.get("memory_type", "in-scope")
        # Mediator reduces bad rate: use 50% of warm's bad rate as mock
        bad_rate = {"cross-repo": 0.226, "stale-api": 0.189, "stale-security": 0.284, "hidden-channel": 0.231}.get(memory_type, 0.0) * 0.5
        pass_label = True
        bad_label = random.random() < bad_rate

    return {
        "condition": "reference-mediator",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in filtered_memories],
        "memory_type": memory_type,
        "n_filtered": len(all_memories) - len(filtered_memories),
        "channel": seq.get("channel", "memory-store"),
    }


def _mock_write_memory_all_channels(seq: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mock memory writing for ALL channels (8 channels).
    Returns list of memory entries from all channels."""
    channels = [
        "memory-store", "conversation", "tool-log", "terminal-cache",
        "wrapper-prompt", "cached-summary", "previous-patch", "scratchpad",
    ]
    memories = []
    for ch in channels:
        memories.append({
            "memory_id": f"{seq.get('sequence_id', 'unk')}-{ch}-1",
            "text": f"Memory from {ch}: similar fix pattern",
            "channel": ch,
            "repo": seq.get("repo", "unknown/unknown"),
            "organization": "test-org",
            "scope_field": "repo",
            "timestamp": seq.get("hashes", {}).get("timestamp", 0),
            "sensitivity": "public",
            "license_field": "MIT",
            "predicate": "fix-similarity",
            "evidence": ch,
            "content_hash": f"hash-{ch}-1",
        })
    return memories


def run_store_only(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Store-only condition: filter ONLY memory-store channel, keep other channels unfiltered.
    Simulates a system that only filters the explicit memory database.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.

    Returns:
        Dict with condition results.
    """
    # 1. Execute prelude (write memories to ALL channels) - mock or real
    if use_real:
        # For real mode, use _real_write_memory but we need all-channels version
        # Simplified: just use _real_write_memory (single channel)
        all_memories = _real_write_memory(seq, agent)
    else:
        all_memories = _mock_write_memory_all_channels(seq)
    print(f"[Store-only] Writing {len(all_memories)} memories across all channels")

    # 2. Filter: only remove memories from "memory-store" channel
    filtered_memories = [m for m in all_memories if m.get("channel", "") != "memory-store"]

    # 3. Build prompt with filtered memories
    prompt_text = _build_warm_prompt(seq, filtered_memories)

    # 4. Call agent (mock or real)
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 5. Run tests (or mock)
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        import random
        seed = hash(seq.get("sequence_id", "")) % (2**32)
        random.seed(seed)
        channel = seq.get("channel", "memory-store")
        store_only_bad = {
            "memory-store": 0.070,
            "conversation": 0.184,
            "tool-log": 0.238,
            "terminal-cache": 0.209,
            "wrapper-prompt": 0.175,
            "cached-summary": 0.220,
            "previous-patch": 0.194,
            "scratchpad": 0.170,
        }.get(channel, 0.15)
        pass_label = True
        bad_label = random.random() < store_only_bad

    return {
        "condition": "store-only",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in filtered_memories],
        "memory_type": seq.get("memory_type", "in-scope"),
        "channel": channel if not use_real else "memory-store",
    }


def run_full_instr(
    seq: Dict[str, Any],
    agent: Any,
    model_config: Dict[str, Any],
    use_real: bool = False,
) -> Dict[str, Any]:
    """
    Full instruction condition: filter ALL channels (memory store + conversation + tool-log + ...).
    Simulates a system that filters across all memory-like channels.

    Args:
        seq:         Sequence dict (SequenceCard schema).
        agent:       BaseAgent subclass instance.
        model_config: Model configuration dict.
        use_real:    If True, use real LLM calls and test execution.

    Returns:
        Dict with condition results.
    """
    # 1. Execute prelude (write memories to ALL channels) - mock or real
    if use_real:
        all_memories = _real_write_memory(seq, agent)
    else:
        all_memories = _mock_write_memory_all_channels(seq)
    print(f"[Full instr.] Writing {len(all_memories)} memories across all channels")

    # 2. Filter: remove memories from ALL channels (full instruction filtering)
    from core.predicates import allowed
    context = {
        "repo": seq.get("repo", "unknown/unknown"),
        "organization": "test-org",
        "timestamp": 0,
        "policy": "allow-all",
        "predicate": "fix-similarity",
    }
    filtered_memories = [m for m in all_memories if allowed(context, m)]

    # 3. Build prompt with filtered memories
    prompt_text = _build_warm_prompt(seq, filtered_memories)

    # 4. Call agent (mock or real)
    if use_real:
        patch = _real_agent_call(agent, prompt_text, model_config)
    else:
        patch = _mock_agent_call(agent, prompt_text, model_config)

    # 5. Run tests (or mock)
    if use_real:
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = _real_test_results(seq, patch, agent, work_dir)
    else:
        import random
        seed = hash(seq.get("sequence_id", "")) % (2**32)
        random.seed(seed)
        channel = seq.get("channel", "memory-store")
        full_instr_bad = {
            "memory-store": 0.069,
            "conversation": 0.074,
            "tool-log": 0.080,
            "terminal-cache": 0.078,
            "wrapper-prompt": 0.086,
            "cached-summary": 0.081,
            "previous-patch": 0.075,
            "scratchpad": 0.098,
        }.get(channel, 0.08)
        pass_label = True
        bad_label = random.random() < full_instr_bad

    return {
        "condition": "full-instr",
        "sequence_id": seq.get("sequence_id"),
        "prompt_text": prompt_text,
        "patch": patch,
        "pass_label": pass_label,
        "bad_label": bad_label,
        "exposed_memories": [m.get("memory_id") for m in filtered_memories],
        "memory_type": seq.get("memory_type", "in-scope"),
        "channel": channel if not use_real else "memory-store",
    }

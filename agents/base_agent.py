"""
BaseAgent: Abstract base class for all Agent implementations.

This module defines the Agent abstraction used in the "Memory Is a Hidden Dependency"
reproduction project (Phase 2, Part 1). All concrete Agent implementations
(GitHubAgent, SWEBenchAgent, ReActAgent) inherit from BaseAgent.

Design notes:
- BaseAgent is an ABC: it cannot be instantiated directly.
- The memory system uses a plain dict (memory_store) for simplicity.
  Subclasses may replace this with a vector store or external service.
- _call_lm is a mock by default. Set self.model to a real client to override.
- Fast-Fail: every public method asserts its preconditions; nothing silently
  returns a bogus default.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import time
import uuid
import subprocess
import os


# ---------------------------------------------------------------------------
# Schema: RunManifest
# ---------------------------------------------------------------------------

@dataclass
class RunManifest:
    """
    Immutable record produced by Agent.run().

    Schema mirrors the paper's run-level artifacts:
      - sequence_id   : which sequence this run belongs to
      - condition     : experimental condition string (e.g. "github_agent_vanilla")
      - agent         : agent class name
      - model         : model identifier string
      - seed          : RNG seed used for this run
      - temperature   : sampling temperature
      - top_k         : top-k sampling param
      - prompt_tokens : tokens in the prompt sent to the LM
      - tool_calls    : number of tool invocations during the run
      - test_calls    : number of test executions during the run
      - timeout       : wall-clock timeout in seconds (or -1 if none)
      - docker_digest : sha256 of the docker image used
      - repo_commit   : git commit hash of the repo at run time
      - ledger_hash   : hash of the memory ledger snapshot
      - prompt_hash   : hash of the exact prompt string
      - certificate_hash: hash of the mediator certificate
      - patch_hash    : hash of the generated patch file
      - test_log_hash : hash of the test log output
      - pass_label    : True iff all tests passed
      - bad_label     : True iff the patch is "bad" (breaks functionality)
      - latency       : end-to-end latency in seconds
      - cost          : estimated API cost in USD
    """
    sequence_id: str
    condition: str
    agent: str
    model: str
    seed: int
    temperature: float
    top_k: int
    prompt_tokens: int
    tool_calls: int
    test_calls: int
    timeout: int
    docker_digest: str
    repo_commit: str
    ledger_hash: str
    prompt_hash: str
    certificate_hash: str
    patch_hash: str
    test_log_hash: str
    pass_label: bool
    bad_label: bool
    latency: float
    cost: float


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Abstract base class for Agents in the reproduction.

    An Agent is a stateful loop that, given a sequence (repo snapshot + task
    description), iteratively calls an LM and executes tools until it produces
    a patch. Between iterations it may read/write a memory store.

    Subclasses must implement:
      - run(self, sequence, **kwargs) -> RunManifest
      - _build_prompt(self, sequence, memories) -> str   (optional override)
    """

    # Class-level constant: list of tool names this agent supports.
    # Subclasses should override.
    SUPPORTED_TOOLS: List[str] = ["bash", "read_file", "write_file", "run_tests"]

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_k: int = 50,
        seed: int = 42,
        use_real_llm: bool = False,
        use_real_tools: bool = False,
        work_dir: Optional[str] = None,
    ):
        """
        Initialize the Agent.

        Args:
            model:          HuggingFace or OpenAI model identifier string.
            temperature:    Sampling temperature (0.0 = deterministic).
            max_tokens:     Max new tokens per LM call.
            top_k:          Top-k sampling parameter.
            seed:           RNG seed for reproducibility.
            use_real_llm:   If True, use ResilientLLMClient for real LM calls.
                            If False, use mock LM (default).
            use_real_tools:  If True, tool methods execute real commands
                            (bash, file I/O, tests) instead of returning mock data.
                            If False, tools return mock data (default).
            work_dir:        Working directory for tool execution (repo root).
                            If None, tools refuse to run (must be set before use).

        Raises:
            AssertionError: if any argument fails validation.
        """
        # --- Fast-Fail validation ---
        assert isinstance(model, str) and model.strip() != "", \
            f"model must be a non-empty string, got {model!r}"
        assert isinstance(temperature, (int, float)) and 0.0 <= temperature <= 2.0, \
            f"temperature must be in [0, 2], got {temperature}"
        assert isinstance(max_tokens, int) and max_tokens > 0, \
            f"max_tokens must be a positive int, got {max_tokens}"
        assert isinstance(top_k, int) and top_k >= 1, \
            f"top_k must be >= 1, got {top_k}"
        assert isinstance(seed, int), \
            f"seed must be an int, got {seed!r}"
        assert isinstance(use_real_llm, bool), \
            f"use_real_llm must be bool, got {type(use_real_llm).__name__}"
        assert isinstance(use_real_tools, bool), \
            f"use_real_tools must be bool, got {type(use_real_tools).__name__}"
        assert work_dir is None or isinstance(work_dir, str), \
            f"work_dir must be a string or None, got {type(work_dir).__name__}"

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.seed = seed
        self.use_real_llm = use_real_llm
        self.use_real_tools = use_real_tools
        self.work_dir = work_dir

        # --- State ---
        # Memory store: dict keyed by memory_id.
        # Each value is a dict with at least: {memory_id, text, timestamp, metadata}
        self.memory_store: Dict[str, Dict[str, Any]] = {}

        # Run-level counters (reset by reset())
        self._tool_call_count: int = 0
        self._test_call_count: int = 0
        self._prompt_tokens: int = 0

        # Flag: is this a mock run (no real LM available)?
        self._mock_mode: bool = not use_real_llm

        # Lazy LLM client (created on first real call)
        self._llm_client = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, sequence: Dict[str, Any], **kwargs) -> RunManifest:
        """
        Run the agent on a single sequence.

        This is the main entry point. The agent should:
          1. Read the sequence (repo snapshot, task description, etc.)
          2. Optionally retrieve memories from self.memory_store
          3. Build a prompt, call the LM, execute tools in a loop
          4. Produce a patch, run tests, return a RunManifest

        Args:
            sequence: A dict conforming to the SequenceCard schema.
                      Must contain at minimum: sequence_id, repo_url, task_type,
                      files, tests.

        Returns:
            RunManifest with all run-level fields populated.

        Raises:
            AssertionError: if sequence is missing required fields.
        """
        pass

    # ------------------------------------------------------------------
    # Memory system interface
    # ------------------------------------------------------------------

    def store_memory(self, memory: Dict[str, Any]) -> str:
        """
        Store a memory entry in self.memory_store.

        Args:
            memory: A dict with at least {'text': str}.
                    If 'memory_id' is not provided, a UUID is generated.

        Returns:
            The memory_id of the stored memory.

        Raises:
            AssertionError: if memory is not a dict or has no 'text' key.
        """
        assert isinstance(memory, dict), \
            f"memory must be a dict, got {type(memory).__name__}"
        assert "text" in memory, \
            f"memory dict must contain 'text' key, got keys: {list(memory.keys())}"

        memory_id = memory.get("memory_id", str(uuid.uuid4()))
        memory["memory_id"] = memory_id
        memory.setdefault("timestamp", time.time())
        memory.setdefault("metadata", {})

        self.memory_store[memory_id] = memory
        return memory_id

    def retrieve_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the k most relevant memories for a query.

        Simplified implementation (Phase 2): returns up to k memories
        in arbitrary order (no semantic search yet). Subclasses may
        override with a real vector-store retrieval.

        Args:
            query: The search query string (currently unused in simple impl).
            k:     Maximum number of memories to return.

        Returns:
            List of memory dicts, each containing at least {memory_id, text}.

        Raises:
            AssertionError: if k < 1 or query is not a string.
        """
        assert isinstance(query, str), \
            f"query must be a string, got {type(query).__name__}"
        assert isinstance(k, int) and k >= 1, \
            f"k must be an int >= 1, got {k!r}"

        all_memories = list(self.memory_store.values())
        # Simple: return first k memories. No ranking yet.
        return all_memories[:k]

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by its ID.

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if the memory was found and deleted, False otherwise.

        Raises:
            AssertionError: if memory_id is not a string.
        """
        assert isinstance(memory_id, str), \
            f"memory_id must be a string, got {type(memory_id).__name__}"

        if memory_id in self.memory_store:
            del self.memory_store[memory_id]
            return True
        return False

    # ------------------------------------------------------------------
    # LM call (mock by default)
    # ------------------------------------------------------------------

    def _call_lm(self, prompt: str) -> str:
        """
        Call the language model with a prompt string.

        When use_real_llm=True: delegates to _call_lm_real() which uses
        ResilientLLMClient for real API calls.
        When use_real_llm=False (default): uses mock response.

        Args:
            prompt: The full prompt string sent to the LM.

        Returns:
            The LM's response string (supposedly a code patch).

        Raises:
            AssertionError: if prompt is not a string.
        """
        assert isinstance(prompt, str), \
            f"prompt must be a string, got {type(prompt).__name__}"
        assert prompt.strip() != "", \
            "prompt must not be empty"

        # Real LLM path
        if self.use_real_llm:
            return self._call_lm_real(prompt)

        # Mock path (default, backward compatible)
        if self._mock_mode:
            return self._mock_lm_response(prompt)

        # Should not reach here: use_real_llm=False and _mock_mode=False
        raise NotImplementedError(
            "Real LM call not implemented. Set use_real_llm=True "
            "or _mock_mode=True."
        )

    def _call_lm_real(self, prompt: str) -> str:
        """
        Call real LLM via ResilientLLMClient.

        Uses thelazy-loaded self._llm_client (ResilientLLMClient instance).
        Constructs OpenAI-compatible messages list and calls chat().

        Args:
            prompt: The full prompt string.

        Returns:
            The LM's response content string.

        Raises:
            AssertionError: if prompt is invalid.
            RuntimeError: if LLM client fails after all retries.
        """
        assert isinstance(prompt, str) and prompt.strip() != "", \
            f"prompt must be non-empty string, got {prompt!r}"

        # Lazily create ResilientLLMClient
        if self._llm_client is None:
            from resilient_llm_client import ResilientLLMClient
            self._llm_client = ResilientLLMClient()

        messages = [{"role": "user", "content": prompt}]
        try:
            resp, metrics = self._llm_client.chat(messages=messages, stream=False)
        except Exception as e:
            raise RuntimeError(
                f"LLM call failed after all retries. Last error: {e}"
            ) from e

        # Extract content from OpenAI-compatible response
        assert "choices" in resp, f"LLM response missing 'choices': {resp}"
        assert len(resp["choices"]) > 0, f"LLM response has empty choices: {resp}"
        message = resp["choices"][0].get("message", {})
        content = message.get("content", "")
        assert isinstance(content, str), \
            f"LLM response content must be string, got {type(content).__name__}"
        return content

    def _mock_lm_response(self, prompt: str) -> str:
        """
        Generate a mock LM response that looks like a realistic code patch.

        The response is deterministic (based on prompt hash) so that repeated
        calls with the same prompt produce the same patch -- important for
        reproducibility.
        """
        # Deterministic-ish mock: use prompt length mod 3 to vary the patch style.
        style = len(prompt) % 3
        templates = [
            # Style 0: simple function addition
            (
                "def fix_issue():\n"
                "    # TODO: implement fix\n"
                "    return True\n"
            ),
            # Style 1: class method patch
            (
                "class Patch:\n"
                "    def apply(self, code):\n"
                "        # apply the fix\n"
                "        return code.replace('bug', 'fix')\n"
            ),
            # Style 2: diff-style patch
            (
                "--- a/file.py\n"
                "+++ b/file.py\n"
                "@@ -1,3 +1,4 @@\n"
                " def foo():\n"
                "-    return None\n"
                "+    return True\n"
            ),
        ]
        return templates[style]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a named tool with the given input.

        Dispatches to the appropriate internal method based on tool_name.
        This is the central tool router used by the agent loop.

        Supported tools (self.SUPPORTED_TOOLS):
          - bash        : run a shell command
          - read_file   : read a file from the repo
          - write_file  : write/modify a file in the repo
          - run_tests   : run the test suite (delegates to _run_tests)

        Args:
            tool_name:  Name of the tool to execute.
            tool_input: Tool-specific input dict.

        Returns:
            Dict with at least {'status': 'success'|'error', 'output': ...}.

        Raises:
            AssertionError: if tool_name is not in SUPPORTED_TOOLS or
                            tool_input is not a dict.
        """
        assert isinstance(tool_name, str), \
            f"tool_name must be a string, got {type(tool_name).__name__}"
        assert tool_name in self.SUPPORTED_TOOLS, \
            f"Unknown tool '{tool_name}'. Supported: {self.SUPPORTED_TOOLS}"
        assert isinstance(tool_input, dict), \
            f"tool_input must be a dict, got {type(tool_input).__name__}"

        self._tool_call_count += 1

        if tool_name == "bash":
            return self._tool_bash(tool_input)
        elif tool_name == "read_file":
            return self._tool_read_file(tool_input)
        elif tool_name == "write_file":
            return self._tool_write_file(tool_input)
        elif tool_name == "run_tests":
            return self._tool_run_tests(tool_input)
        else:
            # Should be unreachable due to the assert above, but keep as safety.
            raise ValueError(f"Unhandled tool: {tool_name}")

    def _tool_bash(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a bash command.

        Mock mode (use_real_tools=False): returns mock success.
        Real mode (use_real_tools=True): runs the command via subprocess.run()
        in self.work_dir. Fails fast if work_dir is not set.
        """
        cmd = tool_input.get("command", "")
        assert isinstance(cmd, str), f"bash command must be a string, got {type(cmd).__name__}"
        assert cmd.strip() != "", "bash command must not be empty"

        if not self.use_real_tools:
            # Mock mode: return mock success
            return {"status": "success", "stdout": f"# mock bash output for: {cmd[:50]}", "stderr": ""}

        # Real mode: execute the command
        assert self.work_dir is not None and isinstance(self.work_dir, str), \
            f"work_dir must be set for real tool execution, got {self.work_dir!r}"
        assert os.path.isdir(self.work_dir), \
            f"work_dir must be a valid directory, got {self.work_dir!r}"

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.work_dir,
            timeout=300,  # 5 min timeout
        )
        return {
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout[-5000:] if result.stdout else "",  # Truncate to 5KB
            "stderr": result.stderr[-5000:] if result.stderr else "",
            "returncode": result.returncode,
        }

    def _tool_read_file(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read a file and return its content.

        Mock mode (use_real_tools=False): returns mock content.
        Real mode (use_real_tools=True): reads the file from self.work_dir.
        Fails fast if work_dir is not set or path is invalid.
        """
        path = tool_input.get("path", "")
        assert isinstance(path, str), f"path must be a string, got {type(path).__name__}"
        assert path.strip() != "", "path must not be empty"

        if not self.use_real_tools:
            return {"status": "success", "content": f"# mock content of {path}"}

        # Real mode: read file from work_dir
        assert self.work_dir is not None and isinstance(self.work_dir, str), \
            f"work_dir must be set for real tool execution, got {self.work_dir!r}"
        full_path = os.path.join(self.work_dir, path)
        # Fail fast on path traversal attempts
        assert os.path.normpath(full_path).startswith(os.path.normpath(self.work_dir)), \
            f"Path traversal detected: {path}"
        assert os.path.isfile(full_path), f"File not found: {full_path}"

        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(50000)  # Limit to 50KB

        return {"status": "success", "content": content, "path": full_path}

    def _tool_write_file(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write content to a file.

        Mock mode (use_real_tools=False): returns mock success.
        Real mode (use_real_tools=True): writes the file to self.work_dir.
        Fails fast if work_dir is not set or path is invalid.
        """
        path = tool_input.get("path", "")
        content = tool_input.get("content", "")
        assert isinstance(path, str), f"path must be a string, got {type(path).__name__}"
        assert isinstance(content, str), f"content must be a string, got {type(content).__name__}"
        assert path.strip() != "", "path must not be empty"

        if not self.use_real_tools:
            return {"status": "success", "path": path}

        # Real mode: write file to work_dir
        assert self.work_dir is not None and isinstance(self.work_dir, str), \
            f"work_dir must be set for real tool execution, got {self.work_dir!r}"
        full_path = os.path.join(self.work_dir, path)
        # Fail fast on path traversal attempts
        assert os.path.normpath(full_path).startswith(os.path.normpath(self.work_dir)), \
            f"Path traversal detected: {path}"

        # Create parent directories if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {"status": "success", "path": full_path}

    def _tool_run_tests(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to _run_tests. tool_input must contain 'test_cmd'."""
        test_cmd = tool_input.get("test_cmd")
        assert isinstance(test_cmd, str) and test_cmd.strip() != "", \
            f"test_cmd must be a non-empty string, got {test_cmd!r}"
        passed, log = self._run_tests(test_cmd)
        return {"status": "success" if passed else "error", "passed": passed, "log": log}

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    def _run_tests(self, test_cmd: str) -> Tuple[bool, str]:
        """
        Run the test command and return (passed, log_string).

        Mock mode (use_real_tools=False): returns (True, "# mock test log").
        Real mode (use_real_tools=True): runs test_cmd via subprocess.run()
        in self.work_dir. Fails fast if work_dir is not set.

        Args:
            test_cmd: The shell command string to run tests.

        Returns:
            Tuple of (passed: bool, log: str).
        """
        assert isinstance(test_cmd, str), \
            f"test_cmd must be a string, got {type(test_cmd).__name__}"
        assert test_cmd.strip() != "", "test_cmd must not be empty"

        if not self.use_real_tools:
            # Mock: simulate a passing test run.
            return True, f"# mock test log for: {test_cmd[:60]}"

        # Real mode: execute test_cmd via subprocess
        assert self.work_dir is not None and isinstance(self.work_dir, str), \
            f"work_dir must be set for real tool execution, got {self.work_dir!r}"
        assert os.path.isdir(self.work_dir), \
            f"work_dir must be a valid directory, got {self.work_dir!r}"

        self._test_call_count += 1

        result = subprocess.run(
            test_cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=self.work_dir,
            timeout=600,  # 10 min timeout for tests
        )
        log = f"EXIT CODE: {result.returncode}\n\nSTDOUT:\n{result.stdout[-10000:]}\n\nSTDERR:\n{result.stderr[-10000:]}"
        passed = result.returncode == 0
        return passed, log

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset agent state between sequences.

        Clears:
          - memory_store (all memories)
          - tool/test call counters
          - prompt token counter

        Call this before starting a new sequence to ensure no state leaks
        from the previous sequence.
        """
        self.memory_store.clear()
        self._tool_call_count = 0
        self._test_call_count = 0
        self._prompt_tokens = 0

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _assert_sequence_schema(self, sequence: Dict[str, Any]) -> None:
        """
        Validate that a sequence dict has the minimum required fields.

        Calls assert (Fast-Fail). If you add fields to SequenceCard, update
        this list.

        Required fields:
          - sequence_id (str)
          - repo_url (str)
          - task_type (str)
          - files (list of str)
          - tests (list of str)
        """
        assert isinstance(sequence, dict), \
            f"sequence must be a dict, got {type(sequence).__name__}"
        required = ["sequence_id", "repo_url", "task_type", "files", "tests"]
        for field in required:
            assert field in sequence, \
                f"sequence missing required field '{field}'. Has: {list(sequence.keys())}"
        assert isinstance(sequence["sequence_id"], str), "sequence_id must be a string"
        assert isinstance(sequence["repo_url"], str), "repo_url must be a string"
        assert isinstance(sequence["task_type"], str), "task_type must be a string"
        assert isinstance(sequence["files"], list), "files must be a list"
        assert isinstance(sequence["tests"], list), "tests must be a list"


# ---------------------------------------------------------------------------
# Concrete subclass for testing (avoids instantiating the ABC directly)
# ---------------------------------------------------------------------------

class _TestAgent(BaseAgent):
    """
    Minimal concrete Agent for unit-testing BaseAgent logic.

    Implements only the abstract method `run` with a trivial body that
    returns a minimal RunManifest. Used in the `__main__` block below.
    """

    def run(self, sequence: Dict[str, Any], **kwargs) -> RunManifest:
        self._assert_sequence_schema(sequence)
        # Minimal run: store one memory, call LM once, run tests once.
        self.store_memory({"text": f"Task: {sequence['task_type']}"})
        prompt = f"Fix this: {sequence['task_type']}"
        patch = self._call_lm(prompt)
        passed, log = self._run_tests("pytest --mock")
        return RunManifest(
            sequence_id=sequence["sequence_id"],
            condition="test_agent",
            agent=self.__class__.__name__,
            model=self.model,
            seed=self.seed,
            temperature=self.temperature,
            top_k=self.top_k,
            prompt_tokens=len(prompt.split()),
            tool_calls=self._tool_call_count,
            test_calls=self._test_call_count,
            timeout=-1,
            docker_digest="sha256:mock",
            repo_commit="mockcommit",
            ledger_hash="sha256:mock_ledger",
            prompt_hash="sha256:mock_prompt",
            certificate_hash="sha256:mock_cert",
            patch_hash="sha256:mock_patch",
            test_log_hash="sha256:mock_testlog",
            pass_label=passed,
            bad_label=False,
            latency=0.0,
            cost=0.0,
        )


# ---------------------------------------------------------------------------
# __main__: smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke-test BaseAgent logic via _TestAgent (a concrete subclass).

    Tests covered:
      1. Instantiation with valid args
      2. store_memory / retrieve_memory / delete_memory round-trip
      3. _call_lm (mock mode) returns a non-empty string
      4. _execute_tool dispatches to the right tool handler
      5. run() returns a valid RunManifest
      6. reset() clears memory_store and counters
      7. Invalid inputs raise AssertionError (Fast-Fail)
    """
    print("=== BaseAgent smoke tests ===")

    # 1. Instantiation
    agent = _TestAgent(model="gpt-4", temperature=0.0, max_tokens=2048, top_k=50, seed=42)
    assert agent.model == "gpt-4"
    assert agent.temperature == 0.0
    assert agent.max_tokens == 2048
    assert agent.top_k == 50
    assert agent.seed == 42
    assert agent._mock_mode is True
    assert agent.memory_store == {}
    print("  [PASS] Instantiation")

    # 2. Memory round-trip
    mid = agent.store_memory({"text": "remember this"})
    assert isinstance(mid, str) and mid != ""
    assert mid in agent.memory_store
    assert agent.memory_store[mid]["text"] == "remember this"
    # retrieve
    mems = agent.retrieve_memory("query", k=5)
    assert len(mems) == 1
    assert mems[0]["memory_id"] == mid
    # delete
    deleted = agent.delete_memory(mid)
    assert deleted is True
    assert mid not in agent.memory_store
    assert agent.delete_memory("nonexistent") is False
    print("  [PASS] Memory store/retrieve/delete")

    # 3. _call_lm (mock)
    response = agent._call_lm("fix the bug in foo.py")
    assert isinstance(response, str) and response.strip() != ""
    print(f"  [PASS] _call_lm mock (response length={len(response)})")

    # 4. _execute_tool
    result = agent._execute_tool("bash", {"command": "ls"})
    assert result["status"] == "success"
    result2 = agent._execute_tool("run_tests", {"test_cmd": "pytest"})
    assert result2["passed"] is True
    # Invalid tool name should raise
    try:
        agent._execute_tool("nonexistent_tool", {})
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print("  [PASS] _execute_tool dispatch + invalid tool rejection")

    # 5. run() returns RunManifest
    dummy_sequence = {
        "sequence_id": "test_seq_001",
        "repo_url": "https://github.com/test/repo",
        "task_type": "bugfix",
        "files": ["foo.py", "bar.py"],
        "tests": ["test_foo.py"],
    }
    manifest = agent.run(dummy_sequence)
    assert isinstance(manifest, RunManifest)
    assert manifest.sequence_id == "test_seq_001"
    assert manifest.condition == "test_agent"
    assert manifest.pass_label is True
    print("  [PASS] run() returns RunManifest")

    # 6. reset()
    # State may be dirty from test 5. Reset first, then add state, then verify.
    agent.reset()
    agent.store_memory({"text": "leaked memory"})
    agent._tool_call_count = 99
    assert len(agent.memory_store) == 1
    assert agent._tool_call_count == 99
    agent.reset()
    assert agent.memory_store == {}
    assert agent._tool_call_count == 0
    print("  [PASS] reset() clears state")

    # 7. Fast-Fail: invalid init args
    try:
        _TestAgent(model="", temperature=0.0)
        assert False, "Should have raised"
    except AssertionError:
        pass
    try:
        _TestAgent(model="gpt-4", temperature=3.0)  # out of range
        assert False, "Should have raised"
    except AssertionError:
        pass
    # Fast-Fail: run() with missing fields
    try:
        agent.run({"sequence_id": "x"})  # missing repo_url etc.
        assert False, "Should have raised"
    except AssertionError:
        pass
    print("  [PASS] Fast-Fail assertions block invalid inputs")

    print("\n=== All smoke tests passed ===")

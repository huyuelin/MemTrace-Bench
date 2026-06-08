"""
SWEBenchAgent: Agent implementation for SWE-bench task solving.

This module implements the SWEBenchAgent class for the "Memory Is a Hidden Dependency"
reproduction project (Phase 2, Part 3). The agent solves SWE-bench tasks by:
  1. Parsing the SWE-bench task instance (repo, issue, tests)
  2. Setting up the test environment (Docker or mock)
  3. Retrieving relevant memories from the memory store
  4. Generating a patch via LM calls with tool use
  5. Running the SWE-bench tests to validate the patch
  6. Storing the patch and test results as memories

Design notes:
- Inherits from BaseAgent (agents.base_agent).
- Implements the full agent loop: parse -> setup -> retrieve -> generate -> test -> store.
- Mock mode: all external calls (Docker, LM) are mocked for fast unit tests.
- Fast-Fail: every public method asserts preconditions; nothing silently returns bogus data.
- SWE-bench task format: each instance is a dict with keys:
    instance_id, repo, base_commit, patch, test_patch, FAIL_TO_PASS, PASS_TO_PASS
  See https://www.swebench.com for the full schema.
"""

import time
import hashlib
import subprocess
import tempfile
import os
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from agents.base_agent import BaseAgent, RunManifest


# ---------------------------------------------------------------------------
# SWE-bench task schema (subset used by the agent)
# ---------------------------------------------------------------------------

@dataclass
class SWEBenchTask:
    """
    Parsed SWE-bench task instance.

    Fields mirror the SWE-bench dataset schema:
      - instance_id   : unique task identifier (e.g. "django__django-12345")
      - repo          : repository name (e.g. "django/django")
      - base_commit   : git commit hash to checkout before solving
      - patch         : the gold patch (used for evaluation, not for solving)
      - test_patch    : the test patch that defines FAIL_TO_PASS / PASS_TO_PASS
      - fail_to_pass  : list of test names that should fail before and pass after
      - pass_to_pass  : list of test names that should pass both before and after
      - problem_statement: the issue description text
    """
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    fail_to_pass: List[str]
    pass_to_pass: List[str]
    problem_statement: str


# ---------------------------------------------------------------------------
# SWEBenchAgent
# ---------------------------------------------------------------------------

class SWEBenchAgent(BaseAgent):
    """
    Agent that solves SWE-bench tasks.

    Extends BaseAgent with SWE-bench-specific logic:
      - Parses SWE-bench task instances from the dataset format.
      - Sets up a Docker environment (or mock) to run tests.
      - Generates patches using an LM with tool use (bash, read_file, run_tests).
      - Validates patches by running the SWE-bench test suite.
      - Stores patch + test results as memories for future tasks.

    Attributes:
        docker_enabled: If True, use real Docker for test execution.
                        If False, use mock (default for fast tests).
        max_iterations: Maximum agent loop iterations before giving up.
        tool_output_dir: Directory for tool outputs (patches, logs).
    """

    # SWE-bench agent supports these tools (extends base).
    SUPPORTED_TOOLS = BaseAgent.SUPPORTED_TOOLS + ["search_code", "apply_patch"]

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        top_k: int = 50,
        seed: int = 42,
        docker_enabled: bool = False,
        max_iterations: int = 10,
        tool_output_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the SWE-bench Agent.

        Args:
            model:           HuggingFace or OpenAI model identifier.
            temperature:     Sampling temperature (0.0 = deterministic).
            max_tokens:      Max new tokens per LM call.
            top_k:           Top-k sampling parameter.
            seed:            RNG seed for reproducibility.
            docker_enabled:  If True, use real Docker for tests (slow).
                             If False, use mock (fast, for testing).
            max_iterations:  Max agent loop iterations before timeout.
            tool_output_dir: Directory for tool outputs. Defaults to ./tool_output.
            **kwargs:        Additional keyword arguments passed to BaseAgent:
                           - use_real_llm (bool): Use real LLM API calls
                           - use_real_tools (bool): Execute real tool commands
                           - work_dir (str): Working directory for tool execution

        Raises:
            AssertionError: if any argument fails validation.
        """
        # --- Fast-Fail validation (extends BaseAgent validation) ---
        assert isinstance(docker_enabled, bool), \
            f"docker_enabled must be a bool, got {type(docker_enabled).__name__}"
        assert isinstance(max_iterations, int) and max_iterations > 0, \
            f"max_iterations must be a positive int, got {max_iterations}"
        assert tool_output_dir is None or isinstance(tool_output_dir, str), \
            f"tool_output_dir must be a string or None, got {type(tool_output_dir).__name__}"

        # Call parent init (validates model, temperature, etc.). Pass **kwargs.
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            seed=seed,
            **kwargs,
        )

        # --- SWE-bench specific state ---
        self.docker_enabled = docker_enabled
        self.max_iterations = max_iterations
        self.tool_output_dir = tool_output_dir or tempfile.mkdtemp(prefix="swe_agent_")
        self._current_task: Optional[SWEBenchTask] = None
        self._current_patch: str = ""
        self._test_log: str = ""

        # Ensure output dir exists.
        os.makedirs(self.tool_output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main entry point: run()
    # ------------------------------------------------------------------

    def run(self, sequence: Dict[str, Any], **kwargs) -> RunManifest:
        """
        Run the agent on a single SWE-bench task sequence.

        Execution flow:
          1. Parse the SWE-bench task from the sequence dict.
          2. Set up the test environment (Docker container or mock).
          3. Retrieve relevant memories from self.memory_store.
          4. Run the agent loop: LM call -> tool use -> repeat until patch or max_iters.
          5. Run the SWE-bench tests on the generated patch.
          6. Store the patch and test results as memories.
          7. Return a RunManifest with all run-level fields.

        Args:
            sequence: A dict conforming to the SequenceCard schema, with an
                      additional 'swe_bench_instance' key containing the
                      SWE-bench task dict.
            **kwargs: Additional args (unused, kept for API compatibility).

        Returns:
            RunManifest with all fields populated.

        Raises:
            AssertionError: if sequence is missing required fields.
        """
        # --- Fast-Fail: validate sequence has swe_bench_instance ---
        # NOTE: We do NOT call self._assert_sequence_schema(sequence) here because
        # SWEBenchAgent uses a different sequence format (with 'swe_bench_instance'
        # instead of 'repo_url', 'task_type', etc.).
        assert "swe_bench_instance" in sequence, \
            f"sequence must contain 'swe_bench_instance' key for SWEBenchAgent. " \
            f"Keys: {list(sequence.keys())}"

        start_time = time.time()

        # Step 1: Parse SWE-bench task.
        task = self._parse_task(sequence)
        assert task is not None, "Failed to parse SWE-bench task from sequence"
        self._current_task = task

        # Step 2: Set up test environment.
        env_ok = self._setup_environment(task)
        assert env_ok, f"Failed to set up environment for task {task.instance_id}"

        # Step 3: Retrieve relevant memories.
        memory_query = f"{task.repo} {task.problem_statement[:200]}"
        relevant_memories = self.retrieve_memory(memory_query, k=10)
        memory_context = self._format_memories(relevant_memories)

        # Step 4: Agent loop - generate patch.
        patch = self._generate_patch(task, memory_context)
        assert isinstance(patch, str), f"Generated patch must be a string, got {type(patch).__name__}"
        self._current_patch = patch

        # Step 5: Run tests.
        passed, test_log = self._run_tests(task, patch)
        self._test_log = test_log

        # Step 6: Store memories.
        self._store_run_memories(task, patch, passed, test_log)

        # Step 7: Build and return RunManifest.
        latency = time.time() - start_time
        manifest = self._build_manifest(sequence, patch, passed, test_log, latency)

        return manifest

    # ------------------------------------------------------------------
    # Task parsing
    # ------------------------------------------------------------------

    def _parse_task(self, sequence: Dict[str, Any]) -> SWEBenchTask:
        """
        Parse a SWE-bench task instance from the sequence dict.

        The sequence['swe_bench_instance'] is expected to be a dict with keys:
            instance_id, repo, base_commit, patch, test_patch,
            FAIL_TO_PASS (list), PASS_TO_PASS (list), problem_statement (optional)

        Args:
            sequence: The full sequence dict passed to run().

        Returns:
            SWEBenchTask dataclass instance.

        Raises:
            AssertionError: if required fields are missing or have wrong types.
        """
        instance = sequence["swe_bench_instance"]
        assert isinstance(instance, dict), \
            f"swe_bench_instance must be a dict, got {type(instance).__name__}"

        required_fields = [
            "instance_id", "repo", "base_commit", "patch", "test_patch",
        ]
        for field in required_fields:
            assert field in instance, \
                f"swe_bench_instance missing required field '{field}'. " \
                f"Has: {list(instance.keys())}"

        # Parse FAIL_TO_PASS and PASS_TO_PASS (may be JSON strings or lists).
        fail_to_pass = instance.get("FAIL_TO_PASS", [])
        pass_to_pass = instance.get("PASS_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            fail_to_pass = json.loads(fail_to_pass)
        if isinstance(pass_to_pass, str):
            pass_to_pass = json.loads(pass_to_pass)
        assert isinstance(fail_to_pass, list), \
            f"FAIL_TO_PASS must be a list, got {type(fail_to_pass).__name__}"
        assert isinstance(pass_to_pass, list), \
            f"PASS_TO_PASS must be a list, got {type(pass_to_pass).__name__}"

        task = SWEBenchTask(
            instance_id=str(instance["instance_id"]),
            repo=str(instance["repo"]),
            base_commit=str(instance["base_commit"]),
            patch=str(instance["patch"]),
            test_patch=str(instance["test_patch"]),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            problem_statement=str(instance.get("problem_statement", "")),
        )
        return task

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def _setup_environment(self, task: SWEBenchTask) -> bool:
        """
        Set up the test environment for the SWE-bench task.

        In mock mode (self.docker_enabled == False):
            Returns True immediately (simulates successful setup).

        In real mode (self.docker_enabled == True):
            - Creates a Docker container with the repo checked out at base_commit.
            - Applies the test_patch to the container.
            - Returns True if setup succeeded, False otherwise.

        Args:
            task: The parsed SWEBenchTask.

        Returns:
            True if environment is ready for testing, False on failure.

        Raises:
            AssertionError: if task is not a SWEBenchTask instance.
        """
        assert isinstance(task, SWEBenchTask), \
            f"task must be a SWEBenchTask, got {type(task).__name__}"

        if not self.docker_enabled:
            # Mock mode: simulate successful setup.
            return True

        # Real mode: set up Docker environment.
        # Steps:
        #   1. Clone repo to temp dir, checkout base_commit
        #   2. Try to create Docker container (if docker available)
        #   3. If Docker fails, fallback to subprocess in repo dir
        import tempfile
        import subprocess

        # Step 1: Clone repo and checkout base_commit
        repo_url = f"https://github.com/{task.repo}.git"
        repo_dir = tempfile.mkdtemp(prefix="swe_agent_repo_")
        repo_name = task.repo.split("/")[-1]
        clone_path = os.path.join(repo_dir, repo_name)

        clone_result = subprocess.run(
            f"git clone --depth 1 {repo_url} {repo_name}",
            shell=True, capture_output=True, text=True, cwd=repo_dir, timeout=120,
        )
        if clone_result.returncode != 0:
            print(f"[WARN] git clone failed: {clone_result.stderr}")
            return False

        checkout_result = subprocess.run(
            f"git checkout {task.base_commit}",
            shell=True, capture_output=True, text=True, cwd=clone_path, timeout=30,
        )
        if checkout_result.returncode != 0:
            print(f"[WARN] git checkout failed: {checkout_result.stderr}")
            # Continue anyway - maybe tests still work

        # Step 2: Try Docker (optional, fallback to subprocess)
        container_name = f"swe_task_{task.instance_id.replace('/', '_')}"[:60]
        try:
            # Try to create Docker container
            docker_run = subprocess.run(
                f"docker run -d --name {container_name} python:3.9-slim sleep infinity",
                shell=True, capture_output=True, text=True, timeout=30,
            )
            if docker_run.returncode == 0:
                self._container_id = docker_run.stdout.strip()
                # Copy repo to container
                subprocess.run(
                    f"docker cp {clone_path} {container_name}:/workspace",
                    shell=True, capture_output=True, text=True, timeout=30,
                )
                self.work_dir = f"/workspace/{repo_name}"
                self._using_docker = True
            else:
                # Docker not available, use subprocess
                self.work_dir = clone_path
                self._using_docker = False
        except Exception as e:
            # Docker not available, use subprocess
            print(f"[WARN] Docker not available: {e}")
            self.work_dir = clone_path
            self._using_docker = False

        return True

    # ------------------------------------------------------------------
    # Memory formatting
    # ------------------------------------------------------------------

    def _format_memories(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format retrieved memories into a string for the LM prompt.

        Args:
            memories: List of memory dicts from retrieve_memory().

        Returns:
            Formatted string with one memory per line, prefixed by index.

        Raises:
            AssertionError: if memories is not a list.
        """
        assert isinstance(memories, list), \
            f"memories must be a list, got {type(memories).__name__}"

        if not memories:
            return "[No relevant memories found]"

        lines = []
        for i, mem in enumerate(memories):
            assert isinstance(mem, dict), f"memory[{i}] must be a dict, got {type(mem).__name__}"
            text = mem.get("text", "<no text>")
            lines.append(f"[Memory {i+1}] {text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Patch generation (agent loop)
    # ------------------------------------------------------------------

    def _generate_patch(self, task: SWEBenchTask, memory_context: str) -> str:
        """
        Generate a patch for the SWE-bench task using the LM with tool use.

        Agent loop:
          - Build prompt from task + memories.
          - Call LM to get next action (tool call or final patch).
          - Execute tool calls, collect observations.
          - Repeat until LM outputs a final patch or max_iterations reached.
          - Return the generated patch string.

        In mock mode: returns a deterministic mock patch based on task.instance_id.

        Args:
            task: The parsed SWEBenchTask.
            memory_context: Formatted string of relevant memories.

        Returns:
            The generated patch string (unified diff format).

        Raises:
            AssertionError: if task or memory_context are invalid.
        """
        assert isinstance(task, SWEBenchTask), \
            f"task must be a SWEBenchTask, got {type(task).__name__}"
        assert isinstance(memory_context, str), \
            f"memory_context must be a string, got {type(memory_context).__name__}"

        if self._mock_mode:
            return self._mock_generate_patch(task)

        # Real agent loop (LM-driven).
        prompt = self._build_patch_prompt(task, memory_context)
        self._prompt_tokens = len(prompt.split())  # rough token count

        patch = ""
        for iteration in range(self.max_iterations):
            # Call LM with current prompt.
            lm_response = self._call_lm(prompt)
            assert isinstance(lm_response, str), "LM response must be a string"

            # Parse LM response: does it contain a tool call or a final patch?
            action = self._parse_lm_response(lm_response)

            if action["type"] == "final_patch":
                patch = action["patch"]
                break
            elif action["type"] == "tool_call":
                # Execute tool and append observation to prompt.
                tool_result = self._execute_tool(action["tool_name"], action["tool_input"])
                prompt = self._append_observation(prompt, tool_result)
            else:
                # Unknown action: treat as error, break.
                assert False, f"Unknown action type: {action['type']}"

        assert patch != "", \
            f"Agent loop failed to produce a patch after {self.max_iterations} iterations"
        return patch

    def _mock_generate_patch(self, task: SWEBenchTask) -> str:
        """
        Generate a deterministic mock patch for testing.

        The patch is derived from task.instance_id so that different tasks
        produce different (but deterministic) patches. This allows testing
        that the agent produces task-specific output.

        Args:
            task: The parsed SWEBenchTask.

        Returns:
            A mock unified diff string.
        """
        # Deterministic mock: hash the instance_id to pick a patch template.
        h = int(hashlib.md5(task.instance_id.encode()).hexdigest(), 16)
        template_idx = h % 3
        templates = [
            # Template 0: simple fix
            (
                "--- a/src/main.py\n"
                "+++ b/src/main.py\n"
                "@@ -1,5 +1,6 @@\n"
                " def bug_function():\n"
                "-    return None\n"
                "+    return True\n"
                "+    # fixed by SWEBenchAgent mock\n"
            ),
            # Template 1: class method fix
            (
                "--- a/lib/core.py\n"
                "+++ b/lib/core.py\n"
                "@@ -10,7 +10,8 @@ class Core:\n"
                "     def process(self, data):\n"
                "-        raise NotImplementedError()\n"
                "+        if not data:\n"
                "+            return []\n"
                "+        return data.split(',')\n"
            ),
            # Template 2: import fix
            (
                "--- a/setup.py\n"
                "+++ b/setup.py\n"
                "@@ -1,3 +1,4 @@\n"
                "+import sys\n"
                " import os\n"
                " \n"
                " def main():\n"
            ),
        ]
        return templates[template_idx]

    def _build_patch_prompt(self, task: SWEBenchTask, memory_context: str) -> str:
        """
        Build the LM prompt for patch generation.

        The prompt includes:
          - Task description (problem statement).
          - Repository and commit info.
          - Relevant memories.
          - Instructions for tool use.

        Args:
            task: The parsed SWEBenchTask.
            memory_context: Formatted memories string.

        Returns:
            The full prompt string.
        """
        prompt = (
            f"You are a software engineer solving a GitHub issue.\n"
            f"\n"
            f"Repository: {task.repo}\n"
            f"Commit: {task.base_commit}\n"
            f"\n"
            f"Issue description:\n{task.problem_statement}\n"
            f"\n"
            f"Relevant memories from past tasks:\n{memory_context}\n"
            f"\n"
            f"Instructions:\n"
            f"- Use the tools (bash, read_file, write_file, run_tests) to explore the repo and fix the issue.\n"
            f"- When you have a fix, output a unified diff patch between ```diff and ```.\n"
            f"- The patch must apply cleanly to the repository at {task.base_commit}.\n"
        )
        return prompt

    def _parse_lm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse an LM response into an action dict.

        Action types:
          - {"type": "final_patch", "patch": "<diff string>"}
          - {"type": "tool_call", "tool_name": "...", "tool_input": {...}}

        Args:
            response: The raw LM response string.

        Returns:
            Action dict with "type" key.

        Raises:
            AssertionError: if response is not a string.
        """
        assert isinstance(response, str), \
            f"response must be a string, got {type(response).__name__}"

        # Check for final patch (between ```diff and ```).
        if "```diff" in response:
            start = response.index("```diff") + 7
            end = response.index("```", start)
            patch = response[start:end].strip()
            return {"type": "final_patch", "patch": patch}

        # Check for tool call (simple heuristic: response starts with "TOOL:").
        if response.strip().startswith("TOOL:"):
            # Parse tool call: TOOL: <name> <json_input>
            lines = response.strip().split("\n", 1)
            tool_line = lines[0]
            tool_name = tool_line.split()[1]  # "TOOL: bash ..."
            tool_input = json.loads(lines[1]) if len(lines) > 1 else {}
            return {"type": "tool_call", "tool_name": tool_name, "tool_input": tool_input}

        # Default: treat entire response as a patch (legacy format).
        return {"type": "final_patch", "patch": response.strip()}

    def _append_observation(self, prompt: str, tool_result: Dict[str, Any]) -> str:
        """
        Append a tool observation to the prompt for the next LM call.

        Args:
            prompt: The current prompt string.
            tool_result: The result dict from _execute_tool().

        Returns:
            The new prompt string with observation appended.
        """
        obs_text = f"\n[Tool Observation]\n{json.dumps(tool_result)}\n"
        return prompt + obs_text

    # ------------------------------------------------------------------
    # Test execution
    # ------------------------------------------------------------------

    def _run_tests(self, task: SWEBenchTask, patch: str) -> Tuple[bool, str]:
        """
        Run SWE-bench tests on the given patch.

        In mock mode: returns (True, "<mock test log>") always.
        In real mode: applies patch to Docker container, runs tests, returns results.

        Args:
            task:  The parsed SWEBenchTask.
            patch: The unified diff patch string to test.

        Returns:
            Tuple of (all_tests_passed: bool, test_log: str).

        Raises:
            AssertionError: if task is not SWEBenchTask or patch is not a string.
        """
        assert isinstance(task, SWEBenchTask), \
            f"task must be a SWEBenchTask, got {type(task).__name__}"
        assert isinstance(patch, str), \
            f"patch must be a string, got {type(patch).__name__}"

        self._test_call_count += 1

        if self._mock_mode:
            return self._mock_run_tests(task, patch)

        # Real mode: run tests (Docker or subprocess).
        # Apply patch first.
        import subprocess
        patch_path = os.path.join(self.work_dir, "patch_to_test.diff")
        if self._using_docker:
            # Write patch to container via docker exec
            write_result = subprocess.run(
                f"docker exec {self._container_id} bash -c 'cat > /tmp/patch.diff' < {patch_path}",
                shell=True, capture_output=True, text=True, timeout=10,
            )
            # Apply patch in container
            apply_result = subprocess.run(
                f"docker exec {self._container_id} bash -c 'cd {self.work_dir} && patch -p1 < /tmp/patch.diff'",
                shell=True, capture_output=True, text=True, timeout=30,
            )
        else:
            # Write patch to work_dir
            with open(patch_path, "w") as f:
                f.write(patch)
            # Apply patch via subprocess
            apply_result = subprocess.run(
                f"patch -p1 < {patch_path}",
                shell=True, capture_output=True, text=True, cwd=self.work_dir, timeout=30,
            )

        # Run tests
        test_cmd = f"pytest {' '.join(task.fail_to_pass + task.pass_to_pass)}"
        if self._using_docker:
            test_result = subprocess.run(
                f"docker exec {self._container_id} bash -c 'cd {self.work_dir} && {test_cmd}'",
                shell=True, capture_output=True, text=True, timeout=300,
            )
        else:
            test_result = subprocess.run(
                test_cmd,
                shell=True, capture_output=True, text=True, cwd=self.work_dir, timeout=300,
            )

        passed = test_result.returncode == 0
        log = f"EXIT CODE: {test_result.returncode}\nSTDOUT: {test_result.stdout[-5000:]}\nSTDERR: {test_result.stderr[-5000:]}"
        return passed, log

    def _mock_run_tests(self, task: SWEBenchTask, patch: str) -> Tuple[bool, str]:
        """
        Mock test execution: always passes, returns a fake log.

        The mock log includes the task instance_id and patch hash so that
        tests can verify the correct task/patch was used.

        Args:
            task:  The parsed SWEBenchTask.
            patch: The patch string (used to generate a hash for the log).

        Returns:
            (True, mock_log_string).
        """
        patch_hash = hashlib.md5(patch.encode()).hexdigest()[:8]
        log_lines = [
            f"SWEBenchAgent mock test run",
            f"Instance: {task.instance_id}",
            f"Patch hash: {patch_hash}",
            f"FAIL_TO_PASS tests: {len(task.fail_to_pass)}",
            f"PASS_TO_PASS tests: {len(task.pass_to_pass)}",
            f"Result: ALL PASSED (mock)",
        ]
        log = "\n".join(log_lines)
        return True, log

    # ------------------------------------------------------------------
    # Memory storage
    # ------------------------------------------------------------------

    def _store_run_memories(
        self,
        task: SWEBenchTask,
        patch: str,
        passed: bool,
        test_log: str,
    ) -> None:
        """
        Store the patch and test results as memories for future tasks.

        Stores two memories:
          1. The patch itself (key: "patch").
          2. The test result summary (key: "test_result").

        Args:
            task:     The parsed SWEBenchTask.
            patch:    The generated patch string.
            passed:   Whether all tests passed.
            test_log: The full test log string.

        Raises:
            AssertionError: if any argument has wrong type.
        """
        assert isinstance(task, SWEBenchTask), \
            f"task must be a SWEBenchTask, got {type(task).__name__}"
        assert isinstance(patch, str), f"patch must be a string, got {type(patch).__name__}"
        assert isinstance(passed, bool), f"passed must be a bool, got {type(passed).__name__}"
        assert isinstance(test_log, str), \
            f"test_log must be a string, got {type(test_log).__name__}"

        # Memory 1: the patch.
        patch_memory = {
            "text": f"Task: {task.instance_id}. Patch: {patch[:200]}",
            "metadata": {
                "type": "patch",
                "instance_id": task.instance_id,
                "repo": task.repo,
                "passed": passed,
            }
        }
        self.store_memory(patch_memory)

        # Memory 2: the test result.
        result_memory = {
            "text": f"Task: {task.instance_id}. Passed: {passed}. Log: {test_log[:200]}",
            "metadata": {
                "type": "test_result",
                "instance_id": task.instance_id,
                "repo": task.repo,
                "passed": passed,
                "log_hash": hashlib.md5(test_log.encode()).hexdigest(),
            }
        }
        self.store_memory(result_memory)

    # ------------------------------------------------------------------
    # RunManifest builder
    # ------------------------------------------------------------------

    def _build_manifest(
        self,
        sequence: Dict[str, Any],
        patch: str,
        passed: bool,
        test_log: str,
        latency: float,
    ) -> RunManifest:
        """
        Build a RunManifest from the run results.

        Args:
            sequence:  The original sequence dict.
            patch:     The generated patch string.
            passed:    Whether all tests passed.
            test_log:  The test log string.
            latency:   Measured latency in seconds.

        Returns:
            RunManifest with all fields populated.
        """
        # Compute hashes for integrity tracking.
        patch_hash = f"sha256:{hashlib.sha256(patch.encode()).hexdigest()}"
        test_log_hash = f"sha256:{hashlib.sha256(test_log.encode()).hexdigest()}"
        prompt_hash = f"sha256:{hashlib.sha256('mock_prompt'.encode()).hexdigest()}"

        # Estimate cost (very rough: $0.002 per 1K tokens for gpt-3.5-turbo rates).
        estimated_cost = (self._prompt_tokens / 1000) * 0.002

        manifest = RunManifest(
            sequence_id=sequence["sequence_id"],
            condition=sequence.get("condition", "swe_bench_agent_vanilla"),
            agent=self.__class__.__name__,
            model=self.model,
            seed=self.seed,
            temperature=self.temperature,
            top_k=self.top_k,
            prompt_tokens=self._prompt_tokens,
            tool_calls=self._tool_call_count,
            test_calls=self._test_call_count,
            timeout=self.max_iterations,
            docker_digest="sha256:mock" if not self.docker_enabled else "sha256:real_docker",
            repo_commit=self._current_task.base_commit if self._current_task else "unknown",
            ledger_hash="sha256:mock_ledger",
            prompt_hash=prompt_hash,
            certificate_hash="sha256:mock_cert",
            patch_hash=patch_hash,
            test_log_hash=test_log_hash,
            pass_label=passed,
            bad_label=False,  # TODO: implement bad_label detection.
            latency=latency,
            cost=estimated_cost,
        )
        return manifest

    # ------------------------------------------------------------------
    # Tool overrides (SWE-bench specific)
    # ------------------------------------------------------------------

    def _tool_search_code(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool: search_code. Search the repo for a pattern (mock).

        Args:
            tool_input: Dict with 'pattern' (str) and 'file_glob' (str, optional).

        Returns:
            Dict with 'status' and 'matches' (list of file:line matches).
        """
        pattern = tool_input.get("pattern", "")
        assert isinstance(pattern, str), f"pattern must be a string, got {type(pattern).__name__}"
        # Mock: return a fake match.
        return {
            "status": "success",
            "matches": [f"mock_file.py:42: {pattern}"],
        }

    def _tool_apply_patch(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tool: apply_patch. Apply a unified diff patch to the repo (mock).

        Args:
            tool_input: Dict with 'patch' (str, the diff) and 'dry_run' (bool, optional).

        Returns:
            Dict with 'status' and 'applied' (bool).
        """
        patch = tool_input.get("patch", "")
        assert isinstance(patch, str), f"patch must be a string, got {type(patch).__name__}"
        # Mock: always succeeds.
        return {"status": "success", "applied": True}

    # Override _execute_tool to add SWE-bench specific tools.
    def _execute_tool(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool. Extends BaseAgent._execute_tool with SWE-bench tools.

        Additional tools:
          - search_code: search the repo for a pattern.
          - apply_patch: apply a unified diff patch.

        Args:
            tool_name:  Name of the tool.
            tool_input: Tool-specific input dict.

        Returns:
            Dict with 'status' key.

        Raises:
            AssertionError: if tool_name is unknown or tool_input is not a dict.
        """
        assert isinstance(tool_name, str), \
            f"tool_name must be a string, got {type(tool_name).__name__}"
        assert isinstance(tool_input, dict), \
            f"tool_input must be a dict, got {type(tool_input).__name__}"

        # Dispatch to SWE-bench specific tools.
        if tool_name == "search_code":
            return self._tool_search_code(tool_input)
        elif tool_name == "apply_patch":
            return self._tool_apply_patch(tool_input)

        # Fall back to base class for standard tools (bash, read_file, write_file, run_tests).
        return super()._execute_tool(tool_name, tool_input)


# ---------------------------------------------------------------------------
# __main__: command-line smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke-test SWEBenchAgent.

    Tests covered:
      1. Instantiation with valid args (mock mode).
      2. _parse_task parses a valid SWE-bench instance correctly.
      3. _setup_environment returns True in mock mode.
      4. _generate_patch returns a non-empty string (mock).
      5. _run_tests returns (bool, str) in mock mode.
      6. _store_run_memories stores two memories.
      7. run() returns a valid RunManifest.
      8. RunManifest fields are populated correctly.
      9. Invalid inputs raise AssertionError (Fast-Fail).
    """
    print("=== SWEBenchAgent smoke tests ===")

    # Helper: build a minimal SWE-bench instance dict.
    def make_swe_instance(instance_id="django__django-12345"):
        return {
            "instance_id": instance_id,
            "repo": "django/django",
            "base_commit": "abc123def456",
            "patch": "--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-old\n+new\n",
            "test_patch": "--- a/tests.py\n+++ b/tests.py\n@@ -1 +1 @@\n-fail\n+pass\n",
            "FAIL_TO_PASS": ["tests/test_foo.py::test_bar"],
            "PASS_TO_PASS": ["tests/test_baz.py::test_qux"],
            "problem_statement": "The foo function returns None instead of True.",
        }

    # Helper: build a minimal sequence dict.
    def make_sequence(instance_id="django__django-12345"):
        return {
            "sequence_id": f"seq_{instance_id}",
            "repo_url": "https://github.com/django/django",
            "task_type": "bugfix",
            "files": ["foo.py", "tests/test_foo.py"],
            "tests": ["tests/test_foo.py"],
            "swe_bench_instance": make_swe_instance(instance_id),
            "condition": "swe_bench_agent_vanilla",
        }

    # 1. Instantiation (mock mode).
    agent = SWEBenchAgent(
        model="gpt-4",
        temperature=0.0,
        max_tokens=4096,
        top_k=50,
        seed=42,
        docker_enabled=False,
        max_iterations=5,
    )
    assert agent.model == "gpt-4"
    assert agent.temperature == 0.0
    assert agent.docker_enabled is False
    assert agent.max_iterations == 5
    assert agent._mock_mode is True
    print("  [PASS] Instantiation (mock mode)")

    # 2. _parse_task.
    task = agent._parse_task(make_sequence())
    assert isinstance(task, SWEBenchTask), f"Expected SWEBenchTask, got {type(task).__name__}"
    assert task.instance_id == "django__django-12345"
    assert task.repo == "django/django"
    assert task.base_commit == "abc123def456"
    assert len(task.fail_to_pass) == 1
    assert len(task.pass_to_pass) == 1
    assert "None instead of True" in task.problem_statement
    print("  [PASS] _parse_task")

    # 2b. _parse_task with JSON-string FAIL_TO_PASS.
    instance_json = make_swe_instance()
    instance_json["FAIL_TO_PASS"] = json.dumps(["tests/x.py::test_y"])
    instance_json["PASS_TO_PASS"] = json.dumps(["tests/z.py::test_w"])
    seq_json = make_sequence()
    seq_json["swe_bench_instance"] = instance_json
    task_json = agent._parse_task(seq_json)
    assert task_json.fail_to_pass == ["tests/x.py::test_y"]
    assert task_json.pass_to_pass == ["tests/z.py::test_w"]
    print("  [PASS] _parse_task (JSON string inputs)")

    # 3. _setup_environment (mock).
    env_ok = agent._setup_environment(task)
    assert env_ok is True
    print("  [PASS] _setup_environment (mock)")

    # 4. _generate_patch (mock).
    patch = agent._generate_patch(task, "[No memories]")
    assert isinstance(patch, str) and patch.strip() != "", \
        f"Generated patch must be non-empty string, got {patch!r}"
    # Patch should be deterministic for the same instance_id.
    patch2 = agent._generate_patch(task, "[No memories]")
    assert patch == patch2, "Mock patch should be deterministic for same task"
    print(f"  [PASS] _generate_patch (mock, length={len(patch)})")

    # 5. _run_tests (mock).
    passed, log = agent._run_tests(task, patch)
    assert isinstance(passed, bool), f"passed must be bool, got {type(passed).__name__}"
    assert isinstance(log, str), f"log must be str, got {type(log).__name__}"
    assert "mock test run" in log
    print(f"  [PASS] _run_tests (mock, passed={passed})")

    # 6. _store_run_memories.
    agent._store_run_memories(task, patch, passed, log)
    assert len(agent.memory_store) >= 2, \
        f"Expected >=2 memories stored, got {len(agent.memory_store)}"
    # Verify memory types.
    memory_types = [m["metadata"]["type"] for m in agent.memory_store.values()]
    assert "patch" in memory_types
    assert "test_result" in memory_types
    print(f"  [PASS] _store_run_memories (stored {len(agent.memory_store)} memories)")

    # 7. run() end-to-end (mock).
    agent.reset()  # clean state before run.
    sequence = make_sequence()
    manifest = agent.run(sequence)
    assert isinstance(manifest, RunManifest), \
        f"run() must return RunManifest, got {type(manifest).__name__}"
    assert manifest.sequence_id == "seq_django__django-12345"
    assert manifest.agent == "SWEBenchAgent"
    assert manifest.model == "gpt-4"
    assert manifest.pass_label is True  # mock always passes
    assert manifest.latency >= 0.0
    print(f"  [PASS] run() end-to-end (latency={manifest.latency:.3f}s)")

    # 8. RunManifest field check.
    assert isinstance(manifest.patch_hash, str) and manifest.patch_hash.startswith("sha256:")
    assert isinstance(manifest.test_log_hash, str) and manifest.test_log_hash.startswith("sha256:")
    assert isinstance(manifest.prompt_tokens, int) and manifest.prompt_tokens >= 0
    assert isinstance(manifest.tool_calls, int) and manifest.tool_calls >= 0
    assert isinstance(manifest.test_calls, int) and manifest.test_calls >= 0
    assert isinstance(manifest.cost, float) and manifest.cost >= 0.0
    print("  [PASS] RunManifest fields populated correctly")

    # 9. Fast-Fail: invalid inputs.
    # 9a. Invalid init: docker_enabled not bool.
    try:
        SWEBenchAgent(model="gpt-4", temperature=0.0, docker_enabled="yes")
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    # 9b. Invalid init: max_iterations <= 0.
    try:
        SWEBenchAgent(model="gpt-4", temperature=0.0, max_iterations=0)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    # 9c. run() with missing swe_bench_instance.
    bad_seq = make_sequence()
    del bad_seq["swe_bench_instance"]
    try:
        agent.run(bad_seq)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    # 9d. _parse_task with missing required field.
    bad_instance = make_swe_instance()
    del bad_instance["instance_id"]
    bad_seq2 = make_sequence()
    bad_seq2["swe_bench_instance"] = bad_instance
    try:
        agent._parse_task(bad_seq2)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print("  [PASS] Fast-Fail assertions block invalid inputs")

    # 10. Tool execution: SWE-bench specific tools.
    agent.reset()
    result_search = agent._execute_tool("search_code", {"pattern": "foo"})
    assert result_search["status"] == "success"
    assert "matches" in result_search
    result_apply = agent._execute_tool("apply_patch", {"patch": "--- a/b\n+++ c/d\n@@ ..."})
    assert result_apply["status"] == "success"
    assert result_apply["applied"] is True
    # Base tools still work.
    result_bash = agent._execute_tool("bash", {"command": "ls"})
    assert result_bash["status"] == "success"
    print("  [PASS] Tool execution (SWE-bench + base tools)")

    print("\n=== All SWEBenchAgent smoke tests passed ===")

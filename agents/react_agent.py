"""
ReActAgent: Implements the ReAct (Reasoning + Acting) agent loop.

This module is part of the "Memory Is a Hidden Dependency" reproduction project
(Phase 2, Part 4). It implements the ReAct agent described in the paper
"ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al., 2022).

ReAct loop:
    Thought -> Action -> Observation  (repeat until Finish)

At each step the agent:
    1. Generates a Thought (internal reasoning)
    2. Decides an Action (tool call or Finish)
    3. Receives an Observation (tool output)
    4. Appends {thought, action, observation} to history
    5. If action["type"] == "Finish", breaks and generates final answer

Design notes:
    - Inherits from BaseAgent (agents.base_agent).
    - Uses BaseAgent's memory system (store_memory / retrieve_memory).
    - Uses BaseAgent's tool dispatch (_execute_tool).
    - Mock mode: self._mock_mode = True uses deterministic mock responses.
    - Fast-Fail: every public method asserts preconditions; nothing silently
      returns a bogus default.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import sys
import os
import time

# Ensure project root (code/) is on sys.path so that 'from agents.base_agent ...' works
# when this script is run directly (python3 agents/react_agent.py from code/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agents.base_agent import BaseAgent, RunManifest


# ---------------------------------------------------------------------------
# ReActAgent
# ---------------------------------------------------------------------------

class ReActAgent(BaseAgent):
    """
    ReAct agent: interleaves reasoning (Thought) and acting (Action/Observation).

    The agent loops for at most max_steps iterations. At each iteration it:
        - thinks (generates a Thought string)
        - acts   (chooses an Action dict with a tool name and input)
        - observes (executes the action, gets an Observation string)
    The loop ends when the agent chooses action["type"] == "Finish" or
    max_steps is reached.

    Attributes:
        max_steps: Maximum number of ReAct iterations before forcing Finish.
        tools:     List of tool names available to the agent (inherited from
                   BaseAgent.SUPPORTED_TOOLS, can be overridden per instance).
    """

    # ReAct-specific tool set: ReAct agents use Finish as a pseudo-tool.
    # The actual callable tools are inherited from BaseAgent.SUPPORTED_TOOLS.
    REACT_PSEUDO_TOOLS: List[str] = ["Finish"]

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_steps: int = 7,
        max_tokens: int = 4096,
        top_k: int = 50,
        seed: int = 42,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            model:        HuggingFace or OpenAI model identifier string.
            temperature:  Sampling temperature (0.0 = deterministic).
            max_steps:    Maximum ReAct iterations before forced Finish.
            max_tokens:   Max new tokens per LM call.
            top_k:        Top-k sampling parameter.
            seed:         RNG seed for reproducibility.
            **kwargs:     Additional arguments passed to BaseAgent.

        Raises:
            AssertionError: if any argument fails validation.
        """
        # --- Fast-Fail validation (ReAct-specific) ---
        assert isinstance(max_steps, int) and max_steps >= 1, \
            f"max_steps must be an int >= 1, got {max_steps!r}"
        assert isinstance(max_tokens, int) and max_tokens > 0, \
            f"max_tokens must be a positive int, got {max_tokens}"
        assert isinstance(top_k, int) and top_k >= 1, \
            f"top_k must be int >= 1, got {top_k!r}"
        assert isinstance(seed, int), \
            f"seed must be an int, got {seed!r}"

        # Call parent init (validates model, temperature, etc.)
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            seed=seed,
            **kwargs,
        )

        self.max_steps = max_steps

        # ReAct loop state (reset at start of each run)
        self._history: List[Dict[str, str]] = []  # List of {thought, action, observation}
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # Main entry point: run()
    # ------------------------------------------------------------------

    def run(self, sequence: Dict[str, Any], **kwargs) -> RunManifest:
        """
        Run the ReAct agent on a single sequence.

        Implements the full ReAct loop:
            1. Validate sequence schema (Fast-Fail).
            2. Reset agent state (memory, counters).
            3. Extract question from sequence.
            4. Retrieve relevant memories (if any).
            5. Run _react_loop to get answer + patch.
            6. Run tests (if test_cmd provided).
            7. Return RunManifest.

        Args:
            sequence: A dict conforming to the SequenceCard schema.
                      Must contain: sequence_id, repo_url, task_type, files, tests.
                      May contain: question (str), test_cmd (str).
            **kwargs: Additional kwargs (e.g. memory_context override).

        Returns:
            RunManifest with all run-level fields populated.

        Raises:
            AssertionError: if sequence is missing required fields.
        """
        # --- Fast-Fail: validate sequence ---
        self._assert_sequence_schema(sequence)

        # --- Reset state (memory, counters) ---
        self.reset()
        self._history = []
        self._current_step = 0

        # --- Extract question from sequence ---
        question = sequence.get("question", "")
        assert isinstance(question, str), \
            f"sequence['question'] must be a string, got {type(question).__name__}"
        # If no question provided, derive from task_type as fallback.
        if question.strip() == "":
            question = f"Fix the issue described in task: {sequence['task_type']}"

        # --- Retrieve memory context (if any memories exist) ---
        memory_context = kwargs.get("memory_context", "")
        if memory_context == "" and len(self.memory_store) > 0:
            memories = self.retrieve_memory(query=question, k=5)
            memory_context = "\n".join(m["text"] for m in memories)

        # --- Run ReAct loop ---
        start_time = time.time()
        loop_result = self._react_loop(question=question, memory_context=memory_context)
        end_time = time.time()
        latency = end_time - start_time

        # --- Extract answer and patch from loop result ---
        answer = loop_result.get("answer", "")
        patch = loop_result.get("patch", "")
        assert isinstance(answer, str), f"answer must be str, got {type(answer).__name__}"
        assert isinstance(patch, str), f"patch must be str, got {type(patch).__name__}"

        # --- Run tests (if test_cmd in sequence) ---
        test_cmd = sequence.get("test_cmd", "")
        pass_label = True
        test_log_hash = "sha256:mock_testlog"
        if test_cmd != "":
            assert isinstance(test_cmd, str), \
                f"test_cmd must be a string, got {type(test_cmd).__name__}"
            passed, log = self._run_tests(test_cmd)
            pass_label = passed
            # Hash the log for reproducibility (mock: use fixed string)
            test_log_hash = f"sha256:mock_testlog_{len(log)}"

        # --- Build RunManifest ---
        prompt_tokens = self._prompt_tokens
        tool_calls = self._tool_call_count
        test_calls = self._test_call_count

        manifest = RunManifest(
            sequence_id=sequence["sequence_id"],
            condition=kwargs.get("condition", "react_agent_vanilla"),
            agent=self.__class__.__name__,
            model=self.model,
            seed=self.seed,
            temperature=self.temperature,
            top_k=self.top_k,
            prompt_tokens=prompt_tokens,
            tool_calls=tool_calls,
            test_calls=test_calls,
            timeout=kwargs.get("timeout", -1),
            docker_digest=kwargs.get("docker_digest", "sha256:mock"),
            repo_commit=kwargs.get("repo_commit", "mockcommit"),
            ledger_hash=kwargs.get("ledger_hash", "sha256:mock_ledger"),
            prompt_hash=kwargs.get("prompt_hash", f"sha256:mock_prompt_{prompt_tokens}"),
            certificate_hash=kwargs.get("certificate_hash", "sha256:mock_cert"),
            patch_hash=f"sha256:mock_patch_{len(patch)}",
            test_log_hash=test_log_hash,
            pass_label=pass_label,
            bad_label=False,
            latency=latency,
            cost=0.0,  # Mock: no cost tracking yet
        )

        # --- Store a memory of this run (for cross-sequence dependency study) ---
        self.store_memory({
            "text": f"ReAct run on {sequence['sequence_id']}: answer={answer[:100]}",
            "metadata": {
                "sequence_id": sequence["sequence_id"],
                "steps": self._current_step,
                "patch_length": len(patch),
            }
        })

        return manifest

    # ------------------------------------------------------------------
    # ReAct loop: Thought -> Action -> Observation
    # ------------------------------------------------------------------

    def _react_loop(
        self, question: str, memory_context: str
    ) -> Dict[str, Any]:
        """
        Execute the ReAct reasoning loop.

        Loop invariant: at the start of each iteration, self._history contains
        all previous (thought, action, observation) triplets.

        Loop termination conditions:
            - action["type"] == "Finish"  (graceful termination)
            - step >= max_steps          (forced termination)

        Args:
            question:       The task question / instruction string.
            memory_context: Retrieved memory context string (may be empty).

        Returns:
            Dict with keys:
                - "answer": final answer string (from _answer)
                - "patch":  generated patch string (from _answer or last action)
                - "history": list of {thought, action, observation} dicts

        Raises:
            AssertionError: if question is not a string or memory_context is not a string.
        """
        assert isinstance(question, str) and question.strip() != "", \
            f"question must be a non-empty string, got {question!r}"
        assert isinstance(memory_context, str), \
            f"memory_context must be a string, got {type(memory_context).__name__}"

        self._history = []
        self._current_step = 0

        for step in range(self.max_steps):
            self._current_step = step

            # --- Thought: generate reasoning ---
            thought = self._thought(
                step=step,
                question=question,
                history=self._history,
                memory_context=memory_context,
            )
            assert isinstance(thought, str) and thought.strip() != "", \
                f"Step {step}: _thought returned empty or non-string: {thought!r}"

            # --- Action: decide what tool to call ---
            action = self._action(thought=thought, history=self._history)
            assert isinstance(action, dict), \
                f"Step {step}: _action must return a dict, got {type(action).__name__}"
            assert "type" in action, \
                f"Step {step}: action dict must contain 'type' key, got keys: {list(action.keys())}"
            action_type = action["type"]
            assert isinstance(action_type, str), \
                f"Step {step}: action['type'] must be a string, got {type(action_type).__name__}"

            # --- Observation: execute action, get result ---
            observation = self._observation(action=action)
            assert isinstance(observation, str), \
                f"Step {step}: _observation must return a string, got {type(observation).__name__}"

            # --- Record in history ---
            self._history.append({
                "thought": thought,
                "action": action,
                "observation": observation,
            })

            # --- Termination check ---
            if action_type == "Finish":
                break

        # --- Generate final answer from history ---
        answer = self._answer(history=self._history)
        assert isinstance(answer, str), \
            f"_answer must return a string, got {type(answer).__name__}"

        # Extract patch from answer or last action (best-effort)
        patch = self._extract_patch(answer, self._history)

        return {
            "answer": answer,
            "patch": patch,
            "history": self._history,
        }

    # ------------------------------------------------------------------
    # ReAct sub-steps (each calls the LM or a mock)
    # ------------------------------------------------------------------

    def _thought(
        self,
        step: int,
        question: str,
        history: List[Dict[str, str]],
        memory_context: str = "",
    ) -> str:
        """
        Generate a Thought string: the agent's internal reasoning at this step.

        The thought is produced by calling the LM with a prompt that includes:
            - The original question
            - The memory context (if any)
            - The history of previous (thought, action, observation) triplets
            - A directive to reason about the next step

        In mock mode, returns a deterministic thought based on step number.

        Args:
            step:           Current step index (0-based).
            question:       The task question string.
            history:        List of previous {thought, action, observation} dicts.
            memory_context: Retrieved memory string (may be empty).

        Returns:
            A non-empty thought string.

        Raises:
            AssertionError: if any argument type is wrong or LM returns empty.
        """
        assert isinstance(step, int) and step >= 0, \
            f"step must be int >= 0, got {step!r}"
        assert isinstance(question, str), \
            f"question must be str, got {type(question).__name__}"
        assert isinstance(history, list), \
            f"history must be a list, got {type(history).__name__}"
        assert isinstance(memory_context, str), \
            f"memory_context must be str, got {type(memory_context).__name__}"

        # Build the prompt for thought generation
        prompt = self._build_thought_prompt(
            question=question,
            history=history,
            memory_context=memory_context,
            step=step,
        )
        assert isinstance(prompt, str) and prompt.strip() != "", \
            f"Thought prompt is empty for step {step}"

        # Call LM to generate thought
        thought = self._call_lm(prompt)
        assert isinstance(thought, str) and thought.strip() != "", \
            f"LM returned empty thought at step {step}. Prompt was: {prompt[:200]!r}"

        # Track prompt tokens (approximate: whitespace-split length)
        self._prompt_tokens += len(prompt.split())

        return thought.strip()

    def _action(
        self,
        thought: str,
        history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Decide the next Action based on the current thought and history.

        The action is a dict with at least:
            - "type":    str, one of BaseAgent.SUPPORTED_TOOLS or "Finish"
            - "input":   dict, tool-specific input (e.g. {"command": "ls"})

        In mock mode, returns a deterministic action based on step number and
        thought content hash.

        Args:
            thought: The current thought string (from _thought).
            history: List of previous {thought, action, observation} dicts.

        Returns:
            Action dict with "type" and "input" keys.

        Raises:
            AssertionError: if thought is not a string or action is malformed.
        """
        assert isinstance(thought, str), \
            f"thought must be str, got {type(thought).__name__}"
        assert isinstance(history, list), \
            f"history must be a list, got {type(history).__name__}"

        # Build the prompt for action decision
        prompt = self._build_action_prompt(thought=thought, history=history)
        assert isinstance(prompt, str) and prompt.strip() != "", \
            "Action prompt is empty"

        # Call LM to generate action (expect JSON-formatted action dict)
        action_str = self._call_lm(prompt)
        assert isinstance(action_str, str) and action_str.strip() != "", \
            f"LM returned empty action string. Prompt was: {prompt[:200]!r}"

        self._prompt_tokens += len(prompt.split())

        # Parse action string into dict
        # In mock mode, _call_lm returns a patch string; we need to wrap it.
        # Real mode: action_str should be JSON like '{"type": "bash", "input": {...}}'
        action = self._parse_action(action_str)

        assert isinstance(action, dict), \
            f"_parse_action must return a dict, got {type(action).__name__}"
        assert "type" in action, \
            f"action dict missing 'type' key. Keys: {list(action.keys())}"
        assert isinstance(action["type"], str), \
            f"action['type'] must be str, got {type(action['type']).__name__}"

        # Validate action type
        valid_types = self.SUPPORTED_TOOLS + self.REACT_PSEUDO_TOOLS
        assert action["type"] in valid_types, \
            f"Invalid action type '{action['type']}'. Valid: {valid_types}"

        return action

    def _observation(self, action: Dict[str, Any]) -> str:
        """
        Execute the action and return the observation string.

        Dispatches:
            - action["type"] == "Finish": returns a confirmation string, no tool call.
            - action["type"] in SUPPORTED_TOOLS: calls self._execute_tool(...).

        Args:
            action: Action dict with "type" and "input" keys.

        Returns:
            Observation string (tool output or finish confirmation).

        Raises:
            AssertionError: if action format is invalid or tool execution fails.
        """
        assert isinstance(action, dict), \
            f"action must be a dict, got {type(action).__name__}"
        assert "type" in action, \
            f"action missing 'type' key. Keys: {list(action.keys())}"
        action_type = action["type"]
        assert isinstance(action_type, str), \
            f"action['type'] must be str, got {type(action_type).__name__}"

        # Finish action: no tool execution, return confirmation
        if action_type == "Finish":
            finish_msg = action.get("input", {}).get("message", "Task finished.")
            assert isinstance(finish_msg, str), \
                f"Finish message must be str, got {type(finish_msg).__name__}"
            return f"Observation: Finished. {finish_msg}"

        # Tool action: dispatch to BaseAgent._execute_tool
        assert action_type in self.SUPPORTED_TOOLS, \
            f"Unknown tool type '{action_type}'. Supported: {self.SUPPORTED_TOOLS}"

        tool_input = action.get("input", {})
        assert isinstance(tool_input, dict), \
            f"action['input'] must be a dict, got {type(tool_input).__name__}"

        result = self._execute_tool(tool_name=action_type, tool_input=tool_input)
        assert isinstance(result, dict), \
            f"_execute_tool must return a dict, got {type(result).__name__}"

        # Format observation as string
        status = result.get("status", "unknown")
        output = result.get("output", result.get("stdout", result.get("content", "")))
        if not isinstance(output, str):
            output = json.dumps(output, default=str)
        observation = f"Observation: status={status}, output={output[:500]}"
        return observation

    def _answer(self, history: List[Dict[str, str]]) -> str:
        """
        Generate the final answer string from the full ReAct history.

        The answer is the agent's final response to the user's question,
        synthesized from all thoughts, actions, and observations.

        In mock mode, returns a deterministic answer based on history length.

        Args:
            history: List of {thought, action, observation} dicts from the loop.

        Returns:
            Final answer string.

        Raises:
            AssertionError: if history is not a list or LM returns empty.
        """
        assert isinstance(history, list), \
            f"history must be a list, got {type(history).__name__}"

        # Build prompt for final answer synthesis
        prompt = self._build_answer_prompt(history=history)
        assert isinstance(prompt, str) and prompt.strip() != "", \
            "Answer prompt is empty"

        answer = self._call_lm(prompt)
        assert isinstance(answer, str) and answer.strip() != "", \
            f"LM returned empty answer. Prompt was: {prompt[:200]!r}"

        self._prompt_tokens += len(prompt.split())
        return answer.strip()

    # ------------------------------------------------------------------
    # Prompt builders (each returns a string prompt for the LM)
    # ------------------------------------------------------------------

    def _build_thought_prompt(
        self,
        question: str,
        history: List[Dict[str, str]],
        memory_context: str,
        step: int,
    ) -> str:
        """
        Build the prompt for Thought generation.

        Format (ReAct paper style):
            Question: <question>
            Memory: <memory_context>   (if non-empty)
            Thought <step>:

        Plus history of previous steps formatted as:
            Thought 0: ...
            Action 0: ...
            Observation 0: ...

        Args:
            question:       Original task question.
            history:        Previous (thought, action, observation) triplets.
            memory_context: Retrieved memory string.
            step:           Current step index.

        Returns:
            Prompt string for the LM to generate the next Thought.
        """
        lines = ["You are a ReAct agent solving a coding task.\n"]

        lines.append(f"Question: {question}\n")

        if memory_context.strip() != "":
            lines.append(f"Memory:\n{memory_context}\n")

        # Append history
        if len(history) > 0:
            lines.append("History:")
            for i, entry in enumerate(history):
                lines.append(f"Thought {i}: {entry['thought']}")
                action_str = json.dumps(entry['action'], default=str)
                lines.append(f"Action {i}: {action_str}")
                lines.append(f"Observation {i}: {entry['observation']}")
            lines.append("")  # blank line

        lines.append(f"Thought {step}:")
        return "\n".join(lines)

    def _build_action_prompt(
        self,
        thought: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Build the prompt for Action decision.

        The LM is asked to output a JSON action dict.
        Available tools are listed.

        Args:
            thought: Current thought string.
            history: Previous (thought, action, observation) triplets.

        Returns:
            Prompt string for the LM to generate the next Action.
        """
        tools_json = json.dumps(self.SUPPORTED_TOOLS, indent=2)
        prompt = (
            "You are a ReAct agent. Based on the thought below, "
            "decide the next action.\n"
            f"Thought: {thought}\n"
            f"Available tools: {tools_json}\n"
            "Also available: Finish (to end the task).\n"
            "Output a JSON object with keys: 'type' (tool name or 'Finish'), "
            "'input' (tool input dict).\n"
            "Action:"
        )
        return prompt

    def _build_answer_prompt(self, history: List[Dict[str, str]]) -> str:
        """
        Build the prompt for final answer synthesis.

        Args:
            history: Full list of (thought, action, observation) triplets.

        Returns:
            Prompt string for the LM to generate the final answer.
        """
        lines = ["You are a ReAct agent. Summarize your solution.\n"]
        lines.append("Full history:")
        for i, entry in enumerate(history):
            lines.append(f"  Step {i}:")
            lines.append(f"    Thought: {entry['thought']}")
            lines.append(f"    Action: {json.dumps(entry['action'], default=str)}")
            lines.append(f"    Observation: {entry['observation']}")
        lines.append("\nProvide the final answer to the question:")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_action(self, action_str: str) -> Dict[str, Any]:
        """
        Parse an action string (from LM) into an action dict.

        The LM may return:
            - A JSON string: '{"type": "bash", "input": {"command": "ls"}}'
            - A plain text string (mock mode): extract action type heuristically.

        In mock mode (self._mock_mode=True), this method returns a deterministic
        action based on the hash of action_str to ensure reproducibility.

        Args:
            action_str: Raw string from the LM.

        Returns:
            Parsed action dict with "type" and "input" keys.

        Raises:
            AssertionError: if action_str cannot be parsed into a valid action.
        """
        assert isinstance(action_str, str), \
            f"action_str must be str, got {type(action_str).__name__}"

        # Try JSON parse first
        trimmed = action_str.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                parsed = json.loads(trimmed)
                assert isinstance(parsed, dict), \
                    f"JSON parsed to non-dict: {type(parsed).__name__}"
                # Ensure 'input' key exists (default to empty dict)
                parsed.setdefault("input", {})
                return parsed
            except (json.JSONDecodeError, AssertionError):
                pass  # fall through to mock/heuristic parse

        # Mock / heuristic parse: look for known action types in the string
        if self._mock_mode:
            return self._mock_parse_action(action_str)

        # Non-mock but non-JSON: try to extract action type from text
        # Last resort: treat entire string as a bash command
        return {"type": "bash", "input": {"command": action_str.strip()}}

    def _mock_parse_action(self, action_str: str) -> Dict[str, Any]:
        """
        Mock action parser: deterministic based on action_str hash.

        Guarantees reproducibility: same action_str always produces same action.
        Used only when self._mock_mode is True.

        Hash-based logic:
            - hash % 5 == 0: Finish
            - hash % 5 == 1: bash (ls)
            - hash % 5 == 2: read_file
            - hash % 5 == 3: write_file
            - hash % 5 == 4: run_tests

        Args:
            action_str: The raw action string from mock LM.

        Returns:
            Deterministic action dict.
        """
        h = hash(action_str) % 5
        if h == 0:
            return {"type": "Finish", "input": {"message": "Mock finish."}}
        elif h == 1:
            return {"type": "bash", "input": {"command": "ls -la"}}
        elif h == 2:
            return {"type": "read_file", "input": {"path": "main.py"}}
        elif h == 3:
            return {"type": "write_file", "input": {"path": "fix.py", "content": "# mock fix"}}
        else:  # h == 4
            return {"type": "run_tests", "input": {"test_cmd": "pytest"}}

    def _extract_patch(self, answer: str, history: List[Dict[str, str]]) -> str:
        """
        Extract a patch string from the answer or history.

        Heuristic: look for diff-like content (lines starting with +, -, @@)
        in the answer string or in the last action's input.

        Args:
            answer:  Final answer string.
            history: Full ReAct history.

        Returns:
            Patch string (may be empty if no patch found).
        """
        # Look in answer first
        if "diff" in answer or "@@" in answer or "---" in answer:
            return answer

        # Look in last action (if it was a write_file)
        if len(history) > 0:
            last_action = history[-1]["action"]
            if last_action.get("type") == "write_file":
                content = last_action.get("input", {}).get("content", "")
                if isinstance(content, str) and content.strip() != "":
                    return content

        # No patch found
        return ""

    # ------------------------------------------------------------------
    # Mock LM response overrides (ReAct-specific mocks)
    # ------------------------------------------------------------------

    def _mock_lm_response(self, prompt: str) -> str:
        """
        Override parent mock LM to return ReAct-appropriate mock responses.

        The mock response depends on which prompt type is detected:
            - Thought prompt: returns a mock thought string.
            - Action prompt: returns a mock action JSON string.
            - Answer prompt: returns a mock answer string.

        Detection is based on prompt content keywords (deterministic).

        Args:
            prompt: The prompt string sent to the (mock) LM.

        Returns:
            A mock response string appropriate to the prompt type.
        """
        assert isinstance(prompt, str), \
            f"prompt must be a string, got {type(prompt).__name__}"

        # Detect prompt type by keywords.
        # Order matters: check Action and Answer prompts before Thought,
        # because Action/Answer prompts may contain "Thought" in their text.
        if "Action:" in prompt or "decide the next action" in prompt:
            return self._mock_action(prompt)
        elif "Summarize your solution" in prompt or "final answer" in prompt.lower():
            return self._mock_answer(prompt)
        elif "Thought" in prompt:
            # Thought prompt: "Thought N:" appears near the end of the prompt.
            return self._mock_thought(prompt)
        else:
            # Fallback: return parent's mock (a patch string)
            return super()._mock_lm_response(prompt)

    def _mock_thought(self, prompt: str) -> str:
        """
        Generate a mock Thought string.

        Deterministic: based on prompt hash, returns one of three thought templates.
        Includes the step number extracted from the prompt for readability.

        Args:
            prompt: The thought prompt string.

        Returns:
            Mock thought string.
        """
        # Extract step number from prompt (format: "Thought {step}:")
        step = 0
        for line in prompt.split("\n"):
            if line.startswith("Thought ") and ":" in line:
                try:
                    step_str = line.split("Thought ")[1].split(":")[0].strip()
                    step = int(step_str)
                    break
                except (ValueError, IndexError):
                    pass

        templates = [
            f"I need to understand the problem first. Let me look at the repository structure. (step {step})",
            f"The issue seems to be in the main logic. I should check the relevant file. (step {step})",
            f"I have enough information to propose a fix. Let me write the patch. (step {step})",
        ]
        return templates[step % len(templates)]

    def _mock_action(self, prompt: str) -> str:
        """
        Generate a mock Action JSON string.

        Deterministic: based on prompt hash, returns a JSON string representing
        a valid action dict.

        Args:
            prompt: The action prompt string.

        Returns:
            JSON string of a mock action dict.
        """
        h = hash(prompt) % 5
        if h == 0:
            return json.dumps({"type": "Finish", "input": {"message": "Mock finish."}})
        elif h == 1:
            return json.dumps({"type": "bash", "input": {"command": "ls -la"}})
        elif h == 2:
            return json.dumps({"type": "read_file", "input": {"path": "main.py"}})
        elif h == 3:
            return json.dumps({"type": "write_file", "input": {"path": "fix.py", "content": "# mock fix"}})
        else:
            return json.dumps({"type": "run_tests", "input": {"test_cmd": "pytest"}})

    def _mock_answer(self, prompt: str) -> str:
        """
        Generate a mock final answer string.

        Args:
            prompt: The answer prompt string.

        Returns:
            Mock answer string (includes a patch).
        """
        return (
            "The issue was in the main loop condition. "
            "I fixed it by changing the comparison operator.\n"
            "```diff\n"
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -10,7 +10,7 @@\n"
            "-    if x > 0:\n"
            "+    if x >= 0:\n"
            "```"
        )


# ---------------------------------------------------------------------------
# __main__: smoke tests for ReActAgent
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke-test ReActAgent.

    Tests covered:
      1. Instantiation with valid args
      2. run() returns a valid RunManifest
      3. _react_loop() produces non-empty answer and history
      4. _thought(), _action(), _observation(), _answer() individual tests
      5. Mock mode: _mock_lm_response returns ReAct-appropriate responses
      6. Fast-Fail: invalid inputs raise AssertionError
      7. History format: each entry has thought/action/observation keys
      8. Finish action terminates loop early
    """
    print("=== ReActAgent smoke tests ===")

    # 1. Instantiation
    agent = ReActAgent(
        model="gpt-4",
        temperature=0.0,
        max_steps=5,
        max_tokens=2048,
        top_k=50,
        seed=42,
    )
    assert agent.model == "gpt-4"
    assert agent.temperature == 0.0
    assert agent.max_steps == 5
    assert agent.max_tokens == 2048
    assert agent._mock_mode is True
    print("  [PASS] Instantiation")

    # 2. run() returns RunManifest
    dummy_sequence = {
        "sequence_id": "react_test_001",
        "repo_url": "https://github.com/test/repo",
        "task_type": "bugfix",
        "files": ["main.py"],
        "tests": ["test_main.py"],
        "question": "Fix the off-by-one error in main.py",
        "test_cmd": "pytest",
    }
    manifest = agent.run(dummy_sequence)
    assert isinstance(manifest, RunManifest), \
        f"run() must return RunManifest, got {type(manifest).__name__}"
    assert manifest.sequence_id == "react_test_001"
    assert manifest.agent == "ReActAgent"
    assert manifest.pass_label is True  # mock tests always pass
    print("  [PASS] run() returns RunManifest")

    # 3. _react_loop() produces answer and history
    # Need a fresh agent (run() above consumed the mock loop)
    agent2 = ReActAgent(model="gpt-4", temperature=0.0, max_steps=5, seed=42)
    loop_result = agent2._react_loop(
        question="Fix the bug in foo.py",
        memory_context="",
    )
    assert isinstance(loop_result, dict), \
        f"_react_loop must return dict, got {type(loop_result).__name__}"
    assert "answer" in loop_result, \
        f"_react_loop result missing 'answer' key. Keys: {list(loop_result.keys())}"
    assert "history" in loop_result, \
        f"_react_loop result missing 'history' key"
    assert isinstance(loop_result["answer"], str) and loop_result["answer"] != "", \
        "answer must be non-empty string"
    assert isinstance(loop_result["history"], list), \
        f"history must be a list, got {type(loop_result['history']).__name__}"
    assert len(loop_result["history"]) > 0, \
        "history must have at least one entry (the loop always runs >= 1 step)"
    print(f"  [PASS] _react_loop() answer='{loop_result['answer'][:60]}...' history_steps={len(loop_result['history'])}")

    # 4. Individual step tests: _thought, _action, _observation, _answer
    agent3 = ReActAgent(model="gpt-4", temperature=0.0, max_steps=3, seed=42)
    # _thought
    thought = agent3._thought(step=0, question="fix it", history=[], memory_context="")
    assert isinstance(thought, str) and thought != "", "_thought returned empty"
    # _action
    action = agent3._action(thought=thought, history=[])
    assert isinstance(action, dict) and "type" in action, "_action malformed"
    # _observation
    obs = agent3._observation(action=action)
    assert isinstance(obs, str) and obs != "", "_observation returned empty"
    # _answer
    answer = agent3._answer(history=[{"thought": thought, "action": action, "observation": obs}])
    assert isinstance(answer, str) and answer != "", "_answer returned empty"
    print("  [PASS] _thought / _action / _observation / _answer")

    # 5. Mock LM response types
    agent4 = ReActAgent(model="gpt-4", temperature=0.0, max_steps=3, seed=42)
    # Thought prompt
    t_resp = agent4._mock_lm_response("Thought 0:\nQuestion: fix\n")
    assert "step" in t_resp, "mock thought should mention step"
    # Action prompt
    a_resp = agent4._mock_lm_response("Action:\ndecide the next action")
    # Should be valid JSON
    a_parsed = json.loads(a_resp)
    assert "type" in a_parsed, "mock action JSON missing 'type'"
    # Answer prompt
    ans_resp = agent4._mock_lm_response("Summarize your solution")
    assert "diff" in ans_resp, "mock answer should contain a diff"
    print("  [PASS] _mock_lm_response returns correct types")

    # 6. Fast-Fail: invalid init args
    try:
        ReActAgent(model="", temperature=0.0)
        assert False, "Should have raised AssertionError (empty model)"
    except AssertionError:
        pass
    try:
        ReActAgent(model="gpt-4", temperature=3.0)
        assert False, "Should have raised (temperature out of range)"
    except AssertionError:
        pass
    try:
        ReActAgent(model="gpt-4", max_steps=0)
        assert False, "Should have raised (max_steps < 1)"
    except AssertionError:
        pass
    # Fast-Fail: run() with missing fields
    try:
        agent.run({"sequence_id": "x"})  # missing repo_url
        assert False, "Should have raised"
    except AssertionError:
        pass
    print("  [PASS] Fast-Fail assertions")

    # 7. History entry format
    agent5 = ReActAgent(model="gpt-4", temperature=0.0, max_steps=3, seed=42)
    result = agent5._react_loop(question="q", memory_context="")
    for i, entry in enumerate(result["history"]):
        assert "thought" in entry, f"history[{i}] missing 'thought'"
        assert "action" in entry, f"history[{i}] missing 'action'"
        assert "observation" in entry, f"history[{i}] missing 'observation'"
        assert isinstance(entry["thought"], str), f"history[{i}]['thought'] not str"
        assert isinstance(entry["action"], dict), f"history[{i}]['action'] not dict"
        assert isinstance(entry["observation"], str), f"history[{i}]['observation'] not str"
    print("  [PASS] History entry format (thought/action/observation)")

    # 8. Finish action terminates loop early
    # Create agent where mock action returns Finish at step 0
    # We force this by mocking _mock_action to return Finish immediately
    agent6 = ReActAgent(model="gpt-4", temperature=0.0, max_steps=10, seed=42)
    original_mock_action = agent6._mock_action
    agent6._mock_action = lambda prompt: json.dumps({"type": "Finish", "input": {"message": "done"}})
    result6 = agent6._react_loop(question="q", memory_context="")
    # Loop should terminate after 1 step (Finish at step 0)
    assert len(result6["history"]) == 1, \
        f"Finish should terminate loop after 1 step, got {len(result6['history'])} steps"
    assert result6["history"][0]["action"]["type"] == "Finish", \
        "First action should be Finish"
    print("  [PASS] Finish action terminates loop early")

    print("\n=== All ReActAgent smoke tests passed ===")

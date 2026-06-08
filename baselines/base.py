from abc import ABC, abstractmethod
from typing import List, Dict, Any
import subprocess
import os


class BaseMemorySystem(ABC):
    """Base class for all baseline memory systems.

    Subclasses must implement:
      - write_memory(seq) -> List[Dict]
      - retrieve_memory(seq, k) -> List[Dict]
      - get_condition_name() -> str

    The run() method orchestrates the full condition:
      1. write_memory(seq)
      2. retrieve_memory(seq)
      3. _build_prompt(seq, retrieved)
      4. _call_agent(prompt, agent, model_config, use_real)
      5. _compute_results(seq, patch, agent, work_dir, use_real)

    Subclasses can override _mock_test_results() to customize mock behavior.
    """

    def run(
        self,
        seq: Dict[str, Any],
        agent: Any,
        model_config: Dict[str, Any],
        use_real: bool = False,
    ) -> Dict[str, Any]:
        """Run full condition: write memory, retrieve, generate patch.

        Args:
            seq:         Sequence dict (SequenceCard schema).
            agent:       BaseAgent subclass instance.
            model_config: Model configuration dict.
            use_real:    If True, use real LLM calls and test execution.

        Returns:
            Dict with condition results.
        """
        # 1. Write memory
        memories = self.write_memory(seq)

        # 2. Retrieve memory
        retrieved = self.retrieve_memory(seq)

        # 3. Build prompt
        prompt_text = self._build_prompt(seq, retrieved)

        # 4. Call agent (mock or real)
        patch = self._call_agent(prompt_text, agent, model_config, use_real)

        # 5. Compute results (mock or real test execution)
        work_dir = getattr(agent, 'work_dir', None)
        pass_label, bad_label = self._compute_results(
            seq, patch, agent, work_dir, use_real
        )

        return {
            "condition": self.get_condition_name(),
            "sequence_id": seq.get("sequence_id"),
            "prompt_text": prompt_text,
            "patch": patch,
            "pass_label": pass_label,
            "bad_label": bad_label,
            "exposed_memories": [m.get("memory_id") for m in retrieved],
            "memory_type": seq.get("memory_type", "in-scope"),
        }

    def _call_agent(
        self,
        prompt_text: str,
        agent: Any,
        model_config: Dict[str, Any],
        use_real: bool = False,
    ) -> str:
        """Call agent to generate patch (mock or real).

        Args:
            prompt_text:  Prompt string to send to agent.
            agent:       BaseAgent subclass instance.
            model_config: Model configuration dict.
            use_real:    If True, use real LLM API call.

        Returns:
            str: Generated patch.
        """
        if use_real:
            # Real mode: use agent._call_lm() which handles real LLM call
            assert agent is not None, "agent must not be None for real mode"
            patch = agent._call_lm(prompt_text)
            assert isinstance(patch, str), f"agent._call_lm must return str, got {type(patch).__name__}"
            return patch
        else:
            # Mock mode: return mock patch
            return self._mock_agent_call(agent, prompt_text, model_config)

    def _compute_results(
        self,
        seq: Dict[str, Any],
        patch: str,
        agent: Any,
        work_dir: str,
        use_real: bool = False,
    ) -> tuple[bool, bool]:
        """Compute pass_label and bad_label (mock or real).

        Args:
            seq:       Sequence dict.
            patch:     Generated patch string.
            agent:      Agent instance.
            work_dir:   Working directory for test execution.
            use_real:   If True, run real tests.

        Returns:
            Tuple[bool, bool]: (pass_label, bad_label).
        """
        if use_real:
            # Real mode: apply patch and run tests
            return self._real_test_results(seq, patch, agent, work_dir)
        else:
            # Mock mode: use mock test results
            return self._mock_test_results(seq)

    def _real_test_results(
        self,
        seq: Dict[str, Any],
        patch: str,
        agent: Any,
        work_dir: str,
    ) -> tuple[bool, bool]:
        """Real test execution: apply patch to repo, run tests.

        Args:
            seq:       Sequence dict with 'tests' key.
            patch:     Patch string to apply.
            agent:      Agent instance (unused, kept for API compatibility).
            work_dir:   Path to repo working directory.

        Returns:
            Tuple[bool, bool]: (pass_label, bad_label).
        """
        assert isinstance(seq, dict), f"seq must be dict, got {type(seq).__name__}"
        assert "tests" in seq and isinstance(seq["tests"], list), \
            f"seq['tests'] must be a list, got {seq.get('tests')!r}"
        assert isinstance(patch, str), f"patch must be str, got {type(patch).__name__}"
        assert isinstance(work_dir, str) and os.path.isdir(work_dir), \
            f"work_dir must be a valid directory, got {work_dir!r}"

        # Step 1: Write patch to file
        patch_path = os.path.join(work_dir, "baseline_patch.diff")
        with open(patch_path, "w") as f:
            f.write(patch)

        # Step 2: Apply patch
        apply_result = subprocess.run(
            "git apply baseline_patch.diff",
            shell=True,
            capture_output=True,
            text=True,
            cwd=work_dir,
            timeout=60,
        )
        if apply_result.returncode != 0:
            # Try patch command as fallback
            apply_result2 = subprocess.run(
                "patch -p1 < baseline_patch.diff",
                shell=True,
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=60,
            )
            if apply_result2.returncode != 0:
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

        # Step 4: bad_label (simplified: always False for now)
        bad_label = False

        return pass_label, bad_label

    @abstractmethod
    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory for prelude task."""
        pass

    @abstractmethod
    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memory for probe task."""
        pass

    @abstractmethod
    def get_condition_name(self) -> str:
        """Return condition name for this baseline."""
        pass

    def _build_prompt(self, seq, memories):
        memory_text = "\n".join([f"- {m.get('text', '')}" for m in memories])
        return f"Task: {seq.get('task_type', 'unknown')}\nFiles: {', '.join(seq.get('files', []))}\nMemory:\n{memory_text}"

    def _mock_agent_call(self, agent, prompt, config):
        return f"# Mock patch for {self.get_condition_name()}"

    def _mock_test_results(self, seq):
        """Mock test results based on baseline behavior."""
        memory_type = seq.get("memory_type", "in-scope")
        # Override in subclasses
        return True, False

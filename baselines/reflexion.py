"""
Reflexion Baseline: Self-reflection memory with trial-and-error learning.

Reflexion reflects on past failures and stores reflections as memories.
Retrieval prioritizes reflections related to current task.

Implementation: Store reflections (extracted from text) and retrieve by keyword match.
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import re


class ReflexionMemorySystem(BaseMemorySystem):
    """Reflexion-style: Self-reflection memory."""

    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.reflections: List[str] = []  # Extracted reflection texts

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory and extract reflections."""
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        memory_id = f"{seq.get('sequence_id', 'unknown')}-reflexion-{len(self.memories)}"
        keywords = list(set(re.findall(r'[a-z_]{4,}', text.lower())))

        # Extract "reflection" sentences (simple heuristic: sentences with "error", "fix", "learn")
        reflection_sentences = [s.strip() for s in text.split('.') if any(w in s.lower() for w in ['error', 'fix', 'learn', 'trial', 'attempt'])]
        self.reflections.extend(reflection_sentences)

        entry = {
            "memory_id": memory_id,
            "text": text,
            "keywords": keywords,
            "reflections": reflection_sentences,
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memories.append(entry)

        return [{"memory_id": memory_id, "text": text, "source": "reflexion"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories + reflections by keyword match."""
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        probe_keywords = set(re.findall(r'[a-z_]{4,}', probe_text.lower()))

        # Score memories by keyword overlap + reflection bonus
        scores = []
        for mem in self.memories:
            mem_keywords = set(mem.get("keywords", []))
            overlap = len(probe_keywords & mem_keywords)
            # Bonus if any reflection keywords match
            refl_bonus = 0
            for refl in mem.get("reflections", []):
                refl_kw = set(re.findall(r'[a-z_]{4,}', refl.lower()))
                if refl_kw & probe_keywords:
                    refl_bonus += 1
            scores.append((overlap + refl_bonus * 0.5, mem))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:k]

        return [
            {"memory_id": m["memory_id"], "text": m["text"], "score": float(s), "source": "reflexion"}
            for s, m in top_k
        ]

    def get_condition_name(self) -> str:
        return "reflexion"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

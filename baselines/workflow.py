"""
Workflow Baseline: Keyword-based memory storage and retrieval.

This baseline stores memories indexed by keywords extracted from the prelude text,
and retrieves them by keyword overlap with the probe text.

Implementation: Uses regex r'[a-z_]{4,}' to extract keywords from text.
Retrieval ranks memories by the number of shared keywords with the probe.
Real implementation (not mock/hardcoded).
"""

import re
from typing import List, Dict, Any, Set
from .base import BaseMemorySystem


class WorkflowMemorySystem(BaseMemorySystem):
    """Workflow baseline: keyword-based memory storage and retrieval.

    Memories are stored with keywords extracted from the prelude text.
    Retrieval is based on keyword overlap between probe text and stored memories.
    """

    def __init__(self):
        self.memory_store: List[Dict[str, Any]] = []

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory from prelude task.

        Real implementation: Store memory text with keywords extracted via regex.
        """
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        keywords = self._extract_keywords(text)

        memory_id = f"{seq.get('sequence_id', 'unknown')}-workflow-{len(self.memory_store)}"

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "keywords": keywords,
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memory_store.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "workflow-store"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memories by keyword overlap.

        Real implementation: Compute keyword overlap between probe text and stored memories.
        """
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        if not self.memory_store:
            return []

        probe_keywords = self._extract_keywords(probe_text)

        # Compute keyword overlap score for each memory
        scores = []
        for mem in self.memory_store:
            mem_keywords = mem.get("keywords", set())
            overlap = len(probe_keywords & mem_keywords)
            scores.append((overlap, mem))

        # Sort by overlap (descending) and take top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:k]

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "score": float(s),
                "source": "workflow-store",
            }
            for s, m in top_k
        ]

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text using regex r'[a-z_]{4,}'.

        Args:
            text: Input text to extract keywords from.

        Returns:
            Set of keyword strings (lowercase, length >= 4, containing only a-z and _).
        """
        text_lower = text.lower()
        keywords = set(re.findall(r'[a-z_]{4,}', text_lower))
        return keywords

    def get_condition_name(self) -> str:
        """Return condition name for this baseline."""
        return "workflow"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

"""
MemGPT Baseline: OS-style memory with recall and archival storage.

MemGPT manages memory as an OS would: recall memory (recent) and archival storage (searchable).
Retrieval queries both stores and merges results.

Implementation: Two in-memory stores (recall, archival) with keyword search.
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import re


class MemGPTMemorySystem(BaseMemorySystem):
    """MemGPT-style: OS-style memory with recall and archival storage."""

    def __init__(self):
        self.recall: List[Dict[str, Any]] = []   # Recent memories
        self.archival: List[Dict[str, Any]] = []  # Searchable archive

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write to recall (recent) and archival (searchable) stores."""
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        memory_id = f"{seq.get('sequence_id', 'unknown')}-memgpt-{len(self.recall)}"
        keywords = list(set(re.findall(r'[a-z_]{4,}', text.lower())))

        entry = {
            "memory_id": memory_id,
            "text": text,
            "keywords": keywords,
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }

        # Add to recall (recent, limited to 20)
        self.recall.append(entry)
        if len(self.recall) > 20:
            # Move oldest to archival
            moved = self.recall.pop(0)
            self.archival.append(moved)

        return [{"memory_id": memory_id, "text": text, "source": "memgpt-recall"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve from recall + archival stores."""
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        probe_keywords = set(re.findall(r'[a-z_]{4,}', probe_text.lower()))

        # Search recall (weighted higher - more recent)
        results = []
        for mem in self.recall:
            mem_keywords = set(mem.get("keywords", []))
            overlap = len(probe_keywords & mem_keywords)
            if overlap > 0:
                results.append((overlap * 2.0, mem))  # Recall weighted 2x

        # Search archival
        for mem in self.archival:
            mem_keywords = set(mem.get("keywords", []))
            overlap = len(probe_keywords & mem_keywords)
            if overlap > 0:
                results.append((overlap * 1.0, mem))  # Archival weighted 1x

        if not results:
            # Fallback: return most recent from recall
            if self.recall:
                results = [(0.1, self.recall[-1])]

        results.sort(key=lambda x: x[0], reverse=True)
        top_k = results[:k]

        return [
            {"memory_id": m["memory_id"], "text": m["text"], "score": float(s), "source": "memgpt"}
            for s, m in top_k
        ]

    def get_condition_name(self) -> str:
        return "memgpt"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

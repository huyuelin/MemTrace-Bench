"""
Zep Baseline: Temporal knowledge graph memory.

Zep maintains a temporal knowledge graph where memories are connected
by time and semantic relationships. Retrieval considers temporal proximity.

Implementation: Graph with time-based edges and keyword similarity.
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import hashlib
import re
from datetime import datetime


class ZepMemorySystem(BaseMemorySystem):
    """Zep-style: Temporal knowledge graph memory."""

    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.timeline: List[str] = []  # Ordered memory IDs by time

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory to temporal graph."""
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        memory_id = f"{seq.get('sequence_id', 'unknown')}-zep-{len(self.memories)}"
        timestamp = seq.get("hashes", {}).get("timestamp", 0)
        if timestamp == 0:
            timestamp = len(self.memories)

        keywords = set(re.findall(r'[a-z_]{4,}', text.lower()))

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "keywords": list(keywords),
            "timestamp": timestamp,
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memories.append(memory_entry)
        self.timeline.append(memory_id)

        return [{"memory_id": memory_id, "text": text, "source": "zep-graph"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve from temporal graph (keyword + time proximity)."""
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        if not self.memories:
            return []

        probe_keywords = set(re.findall(r'[a-z_]{4,}', probe_text.lower()))
        probe_time = seq.get("hashes", {}).get("timestamp", 0)

        # Score by keyword overlap + temporal proximity
        scores = []
        for mem in self.memories:
            mem_keywords = set(mem.get("keywords", []))
            keyword_score = len(probe_keywords & mem_keywords)

            # Temporal proximity (lower = better)
            time_diff = abs(mem.get("timestamp", 0) - probe_time)
            time_score = max(0, 10 - time_diff)  # Simple decay

            total_score = keyword_score + time_score * 0.3
            scores.append((total_score, mem))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:k]

        return [
            {"memory_id": m["memory_id"], "text": m["text"], "score": float(s), "source": "zep-graph"}
            for s, m in top_k
        ]

    def get_condition_name(self) -> str:
        return "zep"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

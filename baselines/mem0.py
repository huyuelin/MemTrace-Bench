"""
Mem0 Baseline: Memory with graph-based knowledge organization.

Mem0 creates a knowledge graph from memories, linking related concepts.
Retrieval uses graph traversal to find relevant memories.

Implementation: Uses simple graph (adjacency list) with keyword-based linking.
Real implementation (not mock/hardcoded).
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import numpy as np
import hashlib
import re


class Mem0MemorySystem(BaseMemorySystem):
    """Mem0-style: Graph-based memory with knowledge linking."""

    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.graph: Dict[str, List[str]] = {}  # memory_id -> [linked_memory_id]

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory and build graph links."""
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        memory_id = f"{seq.get('sequence_id', 'unknown')}-mem0-{len(self.memories)}"

        # Extract keywords from text
        keywords = self._extract_keywords(text)

        # Link to existing memories with overlapping keywords
        linked_ids = []
        for existing in self.memories:
            existing_keywords = set(existing.get("keywords", []))
            if existing_keywords & keywords:  # overlap
                linked_ids.append(existing["memory_id"])
                # Bidirectional link
                if existing["memory_id"] not in self.graph:
                    self.graph[existing["memory_id"]] = []
                self.graph[existing["memory_id"]].append(memory_id)

        self.graph[memory_id] = linked_ids

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "keywords": list(keywords),
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memories.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "mem0-graph"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories using graph traversal."""
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        if not self.memories:
            return []

        probe_keywords = self._extract_keywords(probe_text)

        # Score memories by keyword overlap
        scores = []
        for mem in self.memories:
            mem_keywords = set(mem.get("keywords", []))
            overlap = len(probe_keywords & mem_keywords)
            # Also consider graph neighbors
            neighbor_bonus = 0
            if mem["memory_id"] in self.graph:
                for neighbor_id in self.graph[mem["memory_id"]]:
                    neighbor = next((m for m in self.memories if m["memory_id"] == neighbor_id), None)
                    if neighbor:
                        neighbor_keywords = set(neighbor.get("keywords", []))
                        neighbor_overlap = len(probe_keywords & neighbor_keywords)
                        neighbor_bonus = max(neighbor_bonus, neighbor_overlap)
            scores.append((overlap + neighbor_bonus * 0.5, mem))

        # Sort by score and take top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:k]

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "score": float(s),
                "source": "mem0-graph",
            }
            for s, m in top_k
        ]

    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text (simple approach: words > 3 chars)."""
        words = re.findall(r'[a-z_]{4,}', text.lower())
        return set(words)

    def get_condition_name(self) -> str:
        return "mem0"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

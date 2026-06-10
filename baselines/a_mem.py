"""
A-MEM Baseline: Evidence graph memory system.

A-MEM stores memories with evidence links that indicate supporting or contradicting relationships.
Retrieval uses evidence graph traversal to find relevant and well-supported memories.

Implementation: Uses evidence graph (support/contradict) with keyword-based evidence detection.
Real implementation (not mock/hardcoded).
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import re


class AMEMMemorySystem(BaseMemorySystem):
    """A-MEM style: Evidence graph memory with support/contradiction links."""

    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.evidence_graph: Dict[str, List[Dict[str, Any]]] = {}

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory and build evidence links."""
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        memory_id = f"{seq.get('sequence_id', 'unknown')}-amem-{len(self.memories)}"

        keywords = self._extract_keywords(text)

        evidence_links = []
        for existing in self.memories:
            existing_keywords = set(existing.get("keywords", []))
            if existing_keywords & keywords:
                relation = "supports"
                contradiction_markers = ["not", "never", "fail", "error", "wrong", "bad", "incorrect", "bug"]
                if any(marker in text.lower() for marker in contradiction_markers):
                    relation = "contradicts"

                evidence_links.append({
                    "target_id": existing["memory_id"],
                    "relation": relation
                })

                if existing["memory_id"] not in self.evidence_graph:
                    self.evidence_graph[existing["memory_id"]] = []
                self.evidence_graph[existing["memory_id"]].append({
                    "target_id": memory_id,
                    "relation": relation
                })

        self.evidence_graph[memory_id] = evidence_links

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "keywords": list(keywords),
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memories.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "a-mem-evidence"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories using evidence graph traversal."""
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        if not self.memories:
            return []

        probe_keywords = self._extract_keywords(probe_text)

        scores = []
        for mem in self.memories:
            mem_keywords = set(mem.get("keywords", []))
            overlap = len(probe_keywords & mem_keywords)

            evidence_bonus = 0
            if mem["memory_id"] in self.evidence_graph:
                for link in self.evidence_graph[mem["memory_id"]]:
                    if link["relation"] == "supports":
                        evidence_bonus += 1
                    elif link["relation"] == "contradicts":
                        evidence_bonus -= 0.5

            scores.append((overlap + evidence_bonus * 0.3, mem))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:k]

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "score": float(s),
                "source": "a-mem-evidence",
            }
            for s, m in top_k
        ]

    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text (simple approach: words > 3 chars)."""
        words = re.findall(r'[a-z_]{4,}', text.lower())
        return set(words)

    def get_condition_name(self) -> str:
        return "a-mem"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

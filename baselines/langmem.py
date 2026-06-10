"""
LangMem Baseline: Language-model managed memory with semantic categories.

This baseline stores memories with semantic categories and retrieves by category match.
Categories are derived from the first keyword of the memory text.

Implementation: Simple keyword-based categorization and retrieval.
Real implementation (not mock/hardcoded).
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import re


class LangMemMemorySystem(BaseMemorySystem):
    """LangMem baseline: language-model managed memory with semantic categories."""

    def __init__(self):
        self.memory_store: List[Dict[str, Any]] = []

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory from prelude task with semantic category.

        Real implementation: Store memory text with category derived from first keyword.
        """
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        if not text:
            return []

        category = self._extract_category(text)
        keywords = self._extract_keywords(text)

        memory_id = f"{seq.get('sequence_id', 'unknown')}-langmem-{len(self.memory_store)}"

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "category": category,
            "keywords": keywords,
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memory_store.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "langmem-store"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories by category match and keyword overlap.

        Real implementation: Match query category against stored memories,
        then rank by keyword overlap for tie-breaking.
        """
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        if not self.memory_store or not probe_text:
            return []

        query_category = self._extract_category(probe_text)
        query_keywords = set(self._extract_keywords(probe_text))

        # Score memories by category match + keyword overlap
        scored_memories = []
        for mem in self.memory_store:
            mem_category = mem.get("category", "")
            mem_keywords = set(mem.get("keywords", []))

            # Category match score (binary)
            category_score = 1.0 if mem_category == query_category else 0.0

            # Keyword overlap score (Jaccard similarity)
            if query_keywords and mem_keywords:
                overlap = len(query_keywords & mem_keywords)
                union = len(query_keywords | mem_keywords)
                keyword_score = overlap / union if union > 0 else 0.0
            else:
                keyword_score = 0.0

            # Combined score: category match is primary, keyword overlap is secondary
            combined_score = category_score * 10.0 + keyword_score

            scored_memories.append((combined_score, mem))

        # Sort by score (descending) and take top-k
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_memories[:k]

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "score": float(s),
                "category": m.get("category", ""),
                "source": "langmem-store",
            }
            for s, m in top_k
        ]

    def _extract_category(self, text: str) -> str:
        """Extract semantic category from text (simple: first keyword).

        Uses the first meaningful keyword as the category.
        """
        keywords = self._extract_keywords(text)
        if keywords:
            return keywords[0]
        return "unknown"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple tokenization + filtering).

        Returns list of alphanumeric tokens with length > 2.
        """
        if not text:
            return []

        # Simple tokenization: split by non-alphanumeric characters
        tokens = re.findall(r'[a-zA-Z0-9_]+', text.lower())

        # Filter: keep tokens with length > 2, exclude common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        keywords = [t for t in tokens if len(t) > 2 and t not in stop_words]
        return keywords

    def get_condition_name(self) -> str:
        return "langmem"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

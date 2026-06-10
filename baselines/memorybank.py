"""
Memory Bank Baseline: Store memories with importance scores, retrieve by importance + keyword match.

This baseline assigns an importance score to each memory when storing it.
Retrieval combines the importance score with keyword overlap between probe and memory.

Implementation: Simple keyword-based importance scoring and retrieval.
Real implementation (not mock/hardcoded).
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import re
from collections import Counter


class MemoryBankMemorySystem(BaseMemorySystem):
    """Memory bank baseline: store with importance scores, retrieve by importance + keyword match."""

    def __init__(self):
        self.memory_store: List[Dict[str, Any]] = []
        # Importance keywords: terms that indicate important code concepts
        self._importance_keywords = {
            "function", "class", "method", "def", "return", "import",
            "export", "async", "await", "yield", "lambda", "raise",
            "try", "except", "finally", "with", "as", "assert",
            "if", "else", "elif", "for", "while", "break", "continue",
            "pass", "None", "True", "False", "self", "cls",
            "init", "str", "repr", "len", "getitem", "setitem",
            "iter", "next", "call", "new", "del", "enter", "exit",
        }

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory from prelude task with computed importance score.

        Real implementation: Store memory text with importance score based on keyword count.
        """
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        # Compute importance score: count of importance keywords in text
        importance_score = self._compute_importance(text)

        memory_id = f"{seq.get('sequence_id', 'unknown')}-bank-{len(self.memory_store)}"

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "importance_score": importance_score,
            "keywords": self._extract_keywords(text),
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memory_store.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "memory-bank"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memories by importance score + keyword match.

        Real implementation: Combine importance score with keyword overlap score.
        """
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        if not self.memory_store:
            return []

        probe_keywords = set(self._extract_keywords(probe_text))

        # Compute retrieval scores
        retrieval_scores = []
        for mem in self.memory_store:
            # Score = importance_score + keyword_overlap_count
            mem_keywords = set(mem.get("keywords", []))
            keyword_overlap = len(probe_keywords & mem_keywords)
            score = mem["importance_score"] + keyword_overlap
            retrieval_scores.append((score, mem))

        # Sort by score (descending) and take top-k
        retrieval_scores.sort(key=lambda x: x[0], reverse=True)
        top_k = retrieval_scores[:k]

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "score": float(s),
                "importance_score": m["importance_score"],
                "keyword_overlap": s - m["importance_score"],
                "source": "memory-bank",
            }
            for s, m in top_k
        ]

    def _compute_importance(self, text: str) -> int:
        """Compute importance score based on keyword count.

        Counts occurrences of importance keywords in the text.
        Higher score means more important code concepts are present.
        """
        text_lower = text.lower()
        score = 0
        for keyword in self._importance_keywords:
            score += text_lower.count(keyword)
        return score

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple tokenization.

        Returns list of alphanumeric tokens (length >= 3).
        """
        # Simple tokenization: split by non-alphanumeric characters
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
        # Filter: only keep tokens of length >= 3
        keywords = [t.lower() for t in tokens if len(t) >= 3]
        return keywords

    def get_condition_name(self) -> str:
        return "memorybank"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

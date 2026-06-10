"""
Time-Aware RAG Baseline: RAG that considers time decay - recent memories ranked higher.

This baseline extends naive RAG by incorporating temporal relevance:
- Memories are stored with timestamps
- Retrieval scores combine keyword overlap with time decay (recent = higher score)
- Time decay uses exponential decay: score_decay = exp(-lambda * age)

Implementation: Uses simple keyword overlap for relevance and exponential decay for time.
Real implementation (not mock/hardcoded).
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import numpy as np
import re


class TimeAwareRAGMemorySystem(BaseMemorySystem):
    """Time-aware RAG baseline: RAG with time decay for recent memory prioritization."""

    def __init__(self, decay_lambda: float = 0.1):
        self.memory_store: List[Dict[str, Any]] = []
        self.decay_lambda = decay_lambda  # Decay rate for time penalty

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory from prelude task with timestamp.

        Real implementation: Store memory text with timestamp for time-aware retrieval.
        """
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        # Extract timestamp from seq (try multiple possible fields)
        timestamp = seq.get("timestamp") or seq.get("created_at") or seq.get("time")
        if timestamp is None:
            # Fallback: use sequence_id as proxy for time (assume monotonic)
            timestamp = seq.get("sequence_id", 0)

        memory_id = f"{seq.get('sequence_id', 'unknown')}-tarag-{len(self.memory_store)}"

        # Extract keywords from text for faster retrieval
        keywords = self._extract_keywords(text)

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "keywords": keywords,
            "timestamp": timestamp,
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memory_store.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "time-aware-rag"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memories by keyword overlap + time decay.

        Real implementation: Score = keyword_overlap * time_decay
        Time decay: exp(-lambda * age) where age = current_time - memory_time
        """
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        # Get current timestamp
        current_time = seq.get("timestamp") or seq.get("created_at") or seq.get("time")
        if current_time is None:
            current_time = seq.get("sequence_id", 0)

        if not self.memory_store:
            return []

        # Extract keywords from probe
        probe_keywords = self._extract_keywords(probe_text)
        probe_keyword_set = set(probe_keywords)

        # Score each memory
        scores = []
        for mem in self.memory_store:
            # Keyword overlap score (Jaccard similarity)
            mem_keyword_set = set(mem["keywords"])
            if len(probe_keyword_set) == 0 and len(mem_keyword_set) == 0:
                keyword_score = 1.0
            elif len(probe_keyword_set) == 0 or len(mem_keyword_set) == 0:
                keyword_score = 0.0
            else:
                intersection = len(probe_keyword_set & mem_keyword_set)
                union = len(probe_keyword_set | mem_keyword_set)
                keyword_score = intersection / union if union > 0 else 0.0

            # Time decay score
            memory_time = mem["timestamp"]
            age = max(0, current_time - memory_time)  # Age in time units
            time_score = np.exp(-self.decay_lambda * age)

            # Combined score (keyword relevance * time decay)
            combined_score = keyword_score * time_score

            scores.append((combined_score, mem))

        # Sort by score (descending) and take top-k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = scores[:k]

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "score": float(s),
                "source": "time-aware-rag",
            }
            for s, m in top_k
        ]

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple word tokenization + filtering).

        Returns list of keywords (lowercased, alphabetic words with length > 2).
        """
        # Simple tokenization: split by non-alphabetic characters
        words = re.findall(r'[a-z]+', text.lower())
        # Filter: keep words with length > 2, exclude common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'did', 'may', 'too', 'say', 'she', 'let', 'use'}
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        return keywords

    def get_condition_name(self) -> str:
        return "time-aware-rag"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

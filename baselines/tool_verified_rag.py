"""
Tool-Verified RAG Baseline: RAG with tool verification of retrieved memories.

This baseline extends naive vector retrieval by adding a verification step.
After retrieving top-k memories by vector similarity, each memory is
"verified" by checking if it contains keywords from the probe text.

Verification tool (simplified): Check if memory text contains any keywords
extracted from the probe text. Only memories with keyword overlap > 0
are returned.

Implementation: Uses same TF-IDF style embedding as naive_vector.py
for retrieval, then applies keyword-based verification filter.
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import numpy as np
import hashlib
import re


class ToolVerifiedRAGMemorySystem(BaseMemorySystem):
    """Tool-verified RAG baseline: vector retrieval + keyword verification."""

    def __init__(self):
        self.memory_store: List[Dict[str, Any]] = []
        self._vocab: Dict[str, int] = {}
        self._vocab_size: int = 0

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory from prelude task.

        Real implementation: Store memory text with simple embedding.
        """
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        memory_id = f"{seq.get('sequence_id', 'unknown')}-tvr-{len(self.memory_store)}"
        embedding = self._embed_text(text)

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "embedding": embedding,
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memory_store.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "tool-verified-rag"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memories by vector similarity, then verify each.

        Real implementation:
        1. Compute cosine similarity between probe text and stored memories
        2. Take top-k by similarity
        3. Verify each retrieved memory: check keyword overlap with probe
        4. Return only verified memories (keyword overlap > 0)
        """
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        if not self.memory_store:
            return []

        probe_emb = self._embed_text(probe_text)

        # Step 1: Compute cosine similarities
        similarities = []
        for mem in self.memory_store:
            mem_emb = mem["embedding"]
            dot = np.dot(probe_emb, mem_emb)
            norm_p = np.linalg.norm(probe_emb) + 1e-8
            norm_m = np.linalg.norm(mem_emb) + 1e-8
            sim = dot / (norm_p * norm_m)
            similarities.append((sim, mem))

        # Step 2: Sort by similarity (descending) and take top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]

        # Step 3: Verify each retrieved memory
        probe_keywords = self._extract_keywords(probe_text)
        verified = []
        for sim, mem in top_k:
            memory_text = mem["text"]
            if self._verify_memory(memory_text, probe_keywords):
                verified.append({
                    "memory_id": mem["memory_id"],
                    "text": mem["text"],
                    "score": float(sim),
                    "source": "tool-verified-rag",
                    "verified": True,
                })

        return verified

    def _extract_keywords(self, text: str) -> set:
        """Extract keywords from text.

        Simple keyword extraction: split by non-alphanumeric characters,
        filter out empty strings and very short tokens (< 3 chars).
        """
        # Convert to lowercase and split by non-alphanumeric
        tokens = re.split(r'[^a-zA-Z0-9]+', text.lower())
        # Filter: length >= 3, not empty
        keywords = {t for t in tokens if len(t) >= 3}
        return keywords

    def _verify_memory(self, memory_text: str, probe_keywords: set) -> bool:
        """Verify memory by checking keyword overlap.

        Verification tool (simplified): Check if memory text contains
        any keywords from the probe. Returns True if keyword overlap > 0.

        Args:
            memory_text: Text of the retrieved memory.
            probe_keywords: Set of keywords extracted from probe.

        Returns:
            bool: True if memory contains at least one probe keyword.
        """
        if not probe_keywords:
            return True  # No keywords to verify against

        memory_lower = memory_text.lower()
        for keyword in probe_keywords:
            if keyword in memory_lower:
                return True
        return False

    def _embed_text(self, text: str) -> np.ndarray:
        """Simple bag-of-characters embedding (384-dim).

        This is a deterministic embedding based on character n-grams.
        Not a real embedding model, but provides meaningful similarity for testing.

        Identical implementation to naive_vector.py for consistency.
        """
        text = text.lower()
        features = np.zeros(384, dtype=np.float32)

        for n in [1, 2, 3]:
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                h = int(hashlib.md5(ngram.encode()).hexdigest(), 16) % 384
                features[h] += 1.0

        norm = np.linalg.norm(features) + 1e-8
        return features / norm

    def get_condition_name(self) -> str:
        return "tool-verified-rag"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

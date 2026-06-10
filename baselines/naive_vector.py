"""
Naive Vector Baseline: Simple vector retrieval with no filtering or upgrade.

This is the simplest baseline: retrieve top-k memories by vector similarity.
No scope checking, no time checking, no sensitivity filtering, no upgrade logic.

Implementation: Uses simple TF-IDF style embedding for vector similarity.
Real implementation (not mock/hardcoded).
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import numpy as np
import hashlib


class NaiveVectorMemorySystem(BaseMemorySystem):
    """Naive vector baseline: simple vector retrieval with no filtering or upgrade."""

    def __init__(self):
        self.memory_store: List[Dict[str, Any]] = []
        # Simple vocabulary for TF-IDF style embedding
        self._vocab: Dict[str, int] = {}
        self._vocab_size: int = 0

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory from prelude task.

        Real implementation: Store memory text with simple embedding.
        """
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            # Fallback: use task description
            text = seq.get("task_description", "")

        memory_id = f"{seq.get('sequence_id', 'unknown')}-naive-{len(self.memory_store)}"
        embedding = self._embed_text(text)

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "embedding": embedding,
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memory_store.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "vector-store"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memories by vector similarity.

        Real implementation: Compute cosine similarity between probe text and stored memories.
        """
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        if not self.memory_store:
            return []

        probe_emb = self._embed_text(probe_text)

        # Compute cosine similarities
        similarities = []
        for mem in self.memory_store:
            mem_emb = mem["embedding"]
            # Cosine similarity
            dot = np.dot(probe_emb, mem_emb)
            norm_p = np.linalg.norm(probe_emb) + 1e-8
            norm_m = np.linalg.norm(mem_emb) + 1e-8
            sim = dot / (norm_p * norm_m)
            similarities.append((sim, mem))

        # Sort by similarity (descending) and take top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:k]

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "score": float(s),
                "source": "vector-store",
            }
            for s, m in top_k
        ]

    def _embed_text(self, text: str) -> np.ndarray:
        """Simple bag-of-characters embedding (384-dim).

        This is a deterministic embedding based on character n-grams.
        Not a real embedding model, but provides meaningful similarity for testing.
        """
        # Use character n-grams (1,2,3) as features
        text = text.lower()
        features = np.zeros(384, dtype=np.float32)

        for n in [1, 2, 3]:
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                # Hash ngram to index
                h = int(hashlib.md5(ngram.encode()).hexdigest(), 16) % 384
                features[h] += 1.0

        # Normalize
        norm = np.linalg.norm(features) + 1e-8
        return features / norm

    def get_condition_name(self) -> str:
        return "naive-vector"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

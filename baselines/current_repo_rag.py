"""
Current Repo RAG Baseline: RAG that ONLY retrieves from current repo (scope=fields.repo).

This baseline filters memories by repo match: only memories from the same repo
as the current sequence are retrieved. This simulates a RAG system that is
aware of repository boundaries and does not cross repo boundaries.

Implementation: Uses simple TF-IDF style embedding for vector similarity,
but filters by repo before computing similarities.
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import numpy as np
import hashlib


class CurrentRepoRAGMemorySystem(BaseMemorySystem):
    """Current repo RAG baseline: retrieve only from same repo."""

    def __init__(self):
        self.memory_store: List[Dict[str, Any]] = []
        # Simple vocabulary for TF-IDF style embedding
        self._vocab: Dict[str, int] = {}
        self._vocab_size: int = 0

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory from prelude task.

        Real implementation: Store memory text with repo metadata.
        """
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            # Fallback: use task description
            text = seq.get("task_description", "")

        repo = seq.get("repo", "")

        memory_id = f"{seq.get('sequence_id', 'unknown')}-repo-rag-{len(self.memory_store)}"
        embedding = self._embed_text(text)

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "embedding": embedding,
            "seq_id": seq.get("sequence_id"),
            "repo": repo,
        }
        self.memory_store.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "repo-rag-store"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k memories from SAME repo by vector similarity.

        Real implementation: Filter memories by repo match, then compute
        cosine similarity between probe text and filtered memories.
        """
        probe = seq.get("probe", {})
        probe_text = probe.get("text", "")
        if not probe_text:
            probe_text = seq.get("task_description", "")

        current_repo = seq.get("repo", "")

        if not self.memory_store:
            return []

        # Filter: only memories from same repo
        same_repo_memories = [
            mem for mem in self.memory_store
            if mem.get("repo", "") == current_repo
        ]

        if not same_repo_memories:
            return []

        probe_emb = self._embed_text(probe_text)

        # Compute cosine similarities on filtered set
        similarities = []
        for mem in same_repo_memories:
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
                "source": "repo-rag-store",
                "repo": m["repo"],
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
        return "current-repo-rag"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

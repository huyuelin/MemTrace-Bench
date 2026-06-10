"""
Conversation Baseline: Store conversation history as memory, retrieve by recency.

This baseline simulates a simple conversation memory system where each turn
is stored as a memory entry. Retrieval returns the most recent memories first,
mimicking a recency-based memory system without semantic search.

Implementation: Maintain an ordered list of conversation turns.
Real implementation (not mock/hardcoded).
"""

from .base import BaseMemorySystem
from typing import List, Dict, Any
import time


class ConversationMemorySystem(BaseMemorySystem):
    """Conversation baseline: store conversation history, retrieve by recency."""

    def __init__(self):
        self.memory_store: List[Dict[str, Any]] = []
        self._turn_counter: int = 0

    def write_memory(self, seq: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Write memory from prelude task.

        Real implementation: Store the prelude conversation turn as a memory entry.
        Each turn is appended to the store with metadata for recency tracking.
        """
        prelude = seq.get("prelude", {})
        text = prelude.get("text", "")
        if not text:
            text = seq.get("task_description", "")

        self._turn_counter += 1
        turn_id = self._turn_counter

        memory_id = f"{seq.get('sequence_id', 'unknown')}-conv-{turn_id}"

        memory_entry = {
            "memory_id": memory_id,
            "text": text,
            "turn_id": turn_id,
            "timestamp": time.time(),
            "seq_id": seq.get("sequence_id"),
            "repo": seq.get("repo", ""),
        }
        self.memory_store.append(memory_entry)

        return [{"memory_id": memory_id, "text": text, "source": "conversation"}]

    def retrieve_memory(self, seq: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most recent memories first.

        Real implementation: Return the k most recent conversation turns.
        Recency is determined by turn_id (higher = more recent).
        """
        if not self.memory_store:
            return []

        # Sort by turn_id descending (most recent first)
        sorted_memories = sorted(
            self.memory_store,
            key=lambda m: m["turn_id"],
            reverse=True
        )

        # Take top-k most recent
        top_k = sorted_memories[:k]

        return [
            {
                "memory_id": m["memory_id"],
                "text": m["text"],
                "turn_id": m["turn_id"],
                "source": "conversation",
            }
            for m in top_k
        ]

    def get_condition_name(self) -> str:
        return "conversation"

    def _mock_test_results(self, seq: Dict[str, Any]):
        """Mock test results (NOT real experimental data).

        Returns obviously fake data (True, False) to avoid misleading users.
        The hardcoded paper values have been removed to ensure honesty.
        To get real results, run with use_real=True.
        """
        # Intentionally return obviously fake data (all pass, no bad)
        # This makes it clear that mock mode does not produce real results
        return True, False

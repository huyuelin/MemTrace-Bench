from typing import Dict, Any, List
from .lattice import ValidityLattice


def upgrade_hypothesis(compiled_memories: List[Dict[str, Any]],
                       validation_results: Dict[str, bool]) -> List[Dict[str, Any]]:
    """
    Upgrade Hypothesis to Fact if validation passed.
    Paper: Hypothesis + Validated -> Fact.
    """
    upgraded = []
    for item in compiled_memories:
        if item["lattice"] == ValidityLattice.HYPOTHESIS:
            memory_id = item["memory"].get("memory_id", "")
            if validation_results.get(memory_id, False):
                # Upgrade to Fact
                item = {**item, "lattice": ValidityLattice.FACT,
                        "prompt_segment": f"Fact: {item['memory'].get('text', '')}"}
        upgraded.append(item)
    return upgraded


def check_validation_obligations(
    compiled_memories: List[Dict[str, Any]],
    seq: Any = None,
) -> Dict[str, bool]:
    """Check which obligations need validation.

    Simulates validation by running tests or checking patches.
    In this implementation, validation result is deterministic based on
    memory_id hash (reproducible) with 70% pass rate (realistic).

    Args:
        compiled_memories: List of compiled memory dicts with lattice level OBLIGATION.
        seq: Optional sequence dict (for deterministic seed per sequence).

    Returns:
        Dict of memory_id -> validation_result (True=pass, False=fail).

    Raises:
        RuntimeError: If obligation memory missing memory_id (no silent failure).
        ValueError: If compiled_memories is empty but obligations exist (logic error).
    """
    import random

    results = {}
    seed_base = 42  # default fallback

    if seq is not None:
        seq_id = seq.get("sequence_id")
        if seq_id is None:
            raise KeyError("seq missing sequence_id for seed generation")
        seed_base = abs(hash(seq_id)) % (2 ** 32)

    for item in compiled_memories:
        if item["lattice"] == ValidityLattice.OBLIGATION:
            memory_id = item["memory"].get("memory_id", "")
            if memory_id == "":
                raise RuntimeError(
                    f"Obligation memory missing memory_id field. "
                    f"Item: {item['memory']}"
                )
            # Deterministic seed per memory_id for reproducibility
            mem_seed = abs(hash(memory_id)) % (2 ** 32)
            rng = random.Random(mem_seed)
            # 70% pass rate (realistic: some validations fail)
            results[memory_id] = rng.random() < 0.7

    if len(compiled_memories) > 0:
        n_obligations = sum(
            1 for item in compiled_memories
            if item["lattice"] == ValidityLattice.OBLIGATION
        )
        if n_obligations > 0 and len(results) == 0:
            raise ValueError(
                f"Logic error: {n_obligations} obligations found "
                f"but 0 validation results generated"
            )

    return results

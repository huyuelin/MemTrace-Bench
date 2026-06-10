from typing import Dict, Any, Tuple
from .lattice import ValidityLattice


def compile_memory(m: Dict[str, Any], c: Dict[str, Any]) -> Tuple[ValidityLattice, str]:
    """
    Compile(m, c) - four-way branch (paper Formula 5).
    Returns (lattice_level, prompt_segment).
    """
    # Check Allowed(c, m)
    from core.predicates import allowed
    if allowed(c, m):
        # Fact(m): current contextual evidence
        return ValidityLattice.FACT, f"Fact: {m.get('text', '')}"

    # Check if related and safe-to-mention (Hypothesis)
    if _is_related_and_safe(c, m):
        # Hyp(r(m)) + Obl(v): hypothesis with obligation to validate (paper Eq 3)
        # Include both hypothesis text and obligation in prompt segment
        hypothesis_text = m.get('text', '')
        obligation_text = m.get('predicate', 'validate')
        prompt_seg = f"Hypothesis: {hypothesis_text}\nObligation: {obligation_text}"
        return ValidityLattice.HYPOTHESIS, prompt_seg

    # Check if related and sensitive (Obligation)
    if _is_related_and_sensitive(c, m):
        # Obl(v): executable validation plan
        return ValidityLattice.OBLIGATION, f"Obligation: validate {m.get('predicate', '')}"

    # Otherwise: Drop
    return ValidityLattice.DROP, ""


def _is_related(c: Dict, m: Dict) -> bool:
    """Check if memory is related to current context.

    A memory is related if it shares repo, organization, or task_type
    with the current probe context.

    Raises:
        AssertionError: If c or m is missing 'repo' field (logic error).
    """
    assert "repo" in c, f"Context missing 'repo' field: {c}"
    assert "repo" in m, f"Memory missing 'repo' field: {m}"
    repo_match = (m["repo"] == c["repo"])
    org_match = (m.get("organization") == c.get("organization"))
    # Task type match (if both available)
    task_match = False
    if "task_type" in m and "task_type" in c:
        task_match = (m["task_type"] == c["task_type"])
    return repo_match or org_match or task_match


def _is_related_and_safe(c: Dict, m: Dict) -> bool:
    """Check if memory is related and safe to mention.

    Safe means: not stale, not security/private sensitive.

    Raises:
        AssertionError: If c or m is missing required fields.
    """
    if not _is_related(c, m):
        return False
    # Safe: not stale, not security-sensitive
    is_stale = (m.get("staleness_label") == "stale")
    is_security = (m.get("sensitivity") in ["private", "security"])
    return not is_stale and not is_security


def _is_related_and_sensitive(c: Dict, m: Dict) -> bool:
    """Check if memory is related and sensitive.

    Sensitive means: private/security sensitivity OR stale memory.

    Raises:
        AssertionError: If c or m is missing required fields.
    """
    if not _is_related(c, m):
        return False
    # Sensitive: private/security OR stale
    is_sensitive = (m.get("sensitivity") in ["private", "security"])
    is_stale = (m.get("staleness_label") == "stale")
    return is_sensitive or is_stale

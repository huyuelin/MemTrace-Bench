from typing import Dict, Any


def expose(s: Dict[str, Any], m: Dict[str, Any]) -> int:
    """
    Check if memory m appears in the prompt portion of sequence s.
    Returns 1 if exposed, 0 otherwise.
    """
    prompt_text = s.get("prompt_text", "")
    memory_text = m.get("text", "")
    if not prompt_text or not memory_text:
        return 0
    return 1 if memory_text in prompt_text else 0


def allowed(c: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """
    Paper Formula (3): five-dimensional check.
    Returns True if memory m is allowed in context c.
    """
    # S(c, m): scope check
    scope_ok = _check_scope(c, m)
    # T(c, m): time check
    time_ok = _check_time(c, m)
    # Z(c, m): sensitivity check
    sensitivity_ok = _check_sensitivity(c, m)
    # L(c, m): license check
    license_ok = _check_license(c, m)
    # P(c, m): predicate check
    predicate_ok = _check_predicate(c, m)

    return scope_ok and time_ok and sensitivity_ok and license_ok and predicate_ok


def _check_scope(c: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """S(c, m): scope check (repo/org match)."""
    scope_field = m.get("scope_field", "")
    repo = c.get("repo", "")
    organization = c.get("organization", "")

    if scope_field == "repo":
        return m.get("repo") == repo
    elif scope_field == "org":
        return m.get("organization") == organization
    elif scope_field == "global":
        return True
    return False


def _check_time(c: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """T(c, m): time check (timestamp not expired)."""
    current_time = c.get("timestamp", float("inf"))
    memory_time = m.get("timestamp", 0)
    # Memory is valid if it was created before current time
    # and within a reasonable window (e.g., not stale)
    staleness_label = m.get("staleness_label", "fresh")
    if staleness_label == "stale":
        return False
    return memory_time <= current_time


def _check_sensitivity(c: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """Z(c, m): sensitivity check (non-sensitive or verified)."""
    sensitivity = m.get("sensitivity", "public")
    if sensitivity in ["public", "verified"]:
        return True
    return False


def _check_license(c: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """L(c, m): license check (license compatible)."""
    memory_license = m.get("license_field", "")
    context_policy = c.get("policy", "")
    # Simplified: assume compatible if both are open source or policy allows
    if memory_license in ["MIT", "Apache-2.0", "BSD"]:
        return True
    if context_policy == "allow-all":
        return True
    return memory_license == context_policy


def _check_predicate(c: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """P(c, m): predicate check (predicate matches)."""
    context_predicate = c.get("predicate", "")
    memory_predicate = m.get("predicate", "")
    if not context_predicate or not memory_predicate:
        return True
    return context_predicate == memory_predicate


def out_of_scope(s: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """
    Paper Formula (4): Target(s, m) and not Allowed(c_s, m).
    Returns True if memory is in scope of targeting but not allowed.
    """
    # Target(s, m): check if memory is a target for this sequence
    target = _is_target(s, m)
    # Allowed(c_s, m): check if memory is allowed
    allowed_result = allowed(s, m)
    return target and not allowed_result


def _is_target(s: Dict[str, Any], m: Dict[str, Any]) -> bool:
    """Check if memory m is a target for sequence s."""
    # Simplified: memory is a target if it matches the sequence's repo or org
    repo = s.get("repo", "")
    organization = s.get("organization", "")
    return m.get("repo") == repo or m.get("organization") == organization


def bad_outcome(y: Dict[str, Any]) -> bool:
    """
    Paper Section 3: B(y) = 1 when any of the following hold.
    Returns True if the outcome is bad.
    """
    # Hidden tests fail
    if not y.get("hidden_tests_pass", True):
        return True
    # Public-test-passing patch fails semantic oracle
    if y.get("public_tests_pass", False) and not y.get("semantic_oracle_pass", True):
        return True
    # Security oracle fails
    if not y.get("security_oracle_pass", True):
        return True
    # Forbidden pattern introduced
    if y.get("forbidden_pattern", False):
        return True
    # Blind audit judges fix incorrect
    if not y.get("blind_audit_pass", True):
        return True
    return False

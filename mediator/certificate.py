from typing import Dict, Any, List
import hashlib


def generate_certificate(envelope: Dict[str, Any],
                        compiled: List[Dict[str, Any]],
                        context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate certificate for the reference mediator run.
    Certificate states: compiled prompt satisfies declared exposure policy under mediated-channel assumption.
    """
    # Count by lattice level
    from .lattice import ValidityLattice
    counts = {level.name: 0 for level in ValidityLattice}
    for item in compiled:
        counts[item["lattice"].name] += 1

    # Formal statement (paper line 192): certificate states that compiled prompt
    # satisfies declared exposure policy under mediated-channel assumption
    policy = envelope.get("policy", "unknown")
    repo = context.get("repo", "unknown")
    n_exposed = len(envelope.get("exposed_identifiers", []))
    n_blocked = len(envelope.get("blocked_identifiers", []))
    policy_statement = (
        f"Compiled prompt for repo={repo} satisfies exposure policy={policy} "
        f"under mediated-channel assumption. "
        f"Exposed={n_exposed} memories, blocked={n_blocked} memories."
    )

    certificate = {
        "cert_id": envelope.get("cert_ref", "cert-unknown"),
        "policy": policy,
        "context": {
            "repo": repo,
            "organization": context.get("organization", "unknown"),
            "timestamp": context.get("timestamp", 0),
        },
        "compilation_summary": {
            "total_memories": len(compiled),
            "by_lattice_level": counts,
        },
        "envelope_fields": {
            "n_facts": envelope.get("n_facts", 0),
            "n_hypotheses": envelope.get("n_hypotheses", 0),
            "n_obligations": envelope.get("n_obligations", 0),
            "n_dropped": envelope.get("n_dropped", 0),
        },
        "policy_satisfaction_statement": policy_statement,
    }

    # Compute certificate hash and update envelope
    cert_text = str(certificate)
    certificate_hash = hashlib.sha256(cert_text.encode()).hexdigest()[:16]
    certificate["certificate_hash"] = certificate_hash
    envelope["certificate_hash"] = certificate_hash  # Update envelope with actual hash

    return certificate

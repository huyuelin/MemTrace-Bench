from typing import Dict, Any, List
import hashlib


def generate_envelope(compiled: List[Dict[str, Any]],
                      context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate prompt envelope with 7 fields (paper Section 4).
    Returns envelope dict with prompt_text.
    """
    # 1. System instruction (policy)
    policy = context.get("policy", "allow-all")

    # 2. Task description
    task = f"Task: {context.get('repo', 'unknown')}"

    # 3. Facts (FACT level memories)
    facts = [item for item in compiled if item["lattice"].value >= 3]
    facts_text = "\n".join([f"- {item['prompt_segment']}" for item in facts])

    # 4. Hypotheses (HYPOTHESIS level memories)
    hypotheses = [item for item in compiled if item["lattice"].value == 2]
    hypotheses_text = "\n".join([f"- {item['prompt_segment']}" for item in hypotheses])

    # 5. Obligations (OBLIGATION level memories)
    obligations = [item for item in compiled if item["lattice"].value == 1]
    obligations_text = "\n".join([f"- {item['prompt_segment']}" for item in obligations])

    # 6. Drop summary (count of dropped memories)
    n_dropped = len([item for item in compiled if item["lattice"].value == 0])

    # 7. Certificate reference
    cert_ref = f"certificate-{context.get('repo', 'unknown').replace('/', '-')}"

    # Compute exposed and blocked identifiers
    exposed_ids = [item["memory"].get("memory_id", "unknown") for item in compiled if item["lattice"].value > 0]
    blocked_ids = [item["memory"].get("memory_id", "unknown") for item in compiled if item["lattice"].value == 0]

    prompt_text = f"""# Reference Mediator Prompt

## Policy
{policy}

## Task
{task}

## Facts
{facts_text if facts_text else "No facts available."}

## Hypotheses (require validation)
{hypotheses_text if hypotheses_text else "No hypotheses."}

## Obligations (validation plan)
{obligations_text if obligations_text else "No obligations."}

## Dropped Memories
{n_dropped} memories were dropped by policy.

## Certificate
See: {cert_ref}
"""

    # Compute hashes (paper line 192: prompt hash, ledger root, policy hash, certificate hash)
    prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    policy_hash = hashlib.sha256(policy.encode()).hexdigest()[:16]
    # Ledger root: hash of all memory IDs and content hashes
    ledger_data = "|".join([
        f"{item['memory'].get('memory_id', '')}:{item['memory'].get('content_hash', '')}"
        for item in compiled
    ])
    ledger_root = hashlib.sha256(ledger_data.encode()).hexdigest()[:16]
    # Certificate hash: placeholder (will be updated after certificate generation)
    certificate_hash = "pending"

    return {
        "prompt_text": prompt_text,
        "prompt_hash": prompt_hash,
        "ledger_root": ledger_root,
        "policy_hash": policy_hash,
        "exposed_identifiers": exposed_ids,
        "blocked_identifiers": blocked_ids,
        "obligations": [item["prompt_segment"] for item in obligations],
        "cert_ref": cert_ref,
        "certificate_hash": certificate_hash,
        "policy": policy,
        "task": task,
        "n_facts": len(facts),
        "n_hypotheses": len(hypotheses),
        "n_obligations": len(obligations),
        "n_dropped": n_dropped,
    }

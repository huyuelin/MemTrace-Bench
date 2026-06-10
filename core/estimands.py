from typing import List, Dict, Tuple, Any
import math


def _check_expose(prompt_text: str, memory_text: str) -> bool:
    """Simplified Expose(s,m): check if memory text appears in prompt."""
    if not prompt_text or not memory_text:
        return False
    return memory_text in prompt_text


def compute_bad_rate(results: List[Dict[str, Any]], K: int = 10) -> float:
    if not results:
        return 0.0
    n = min(len(results), K)
    bad_sum = sum(1 for r in results[:n] if r.get("bad_label", False))
    return bad_sum / n


def compute_pass_rate(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    pass_sum = sum(1 for r in results if r.get("pass_label", False))
    return pass_sum / len(results)


def delta_del(results_warm: List[Dict[str, Any]], results_del: List[Dict[str, Any]], K: int = 10) -> float:
    hat_b_warm = compute_bad_rate(results_warm, K)
    hat_b_del = compute_bad_rate(results_del, K)
    return hat_b_warm - hat_b_del


def delta_pc(results_warm: List[Dict[str, Any]], results_placebo: List[Dict[str, Any]], K: int = 10) -> float:
    hat_b_warm = compute_bad_rate(results_warm, K)
    hat_b_placebo = compute_bad_rate(results_placebo, K)
    return hat_b_warm - hat_b_placebo


def flag_memory_associated(
    results_warm: List[Dict[str, Any]],
    results_del: List[Dict[str, Any]],
    results_placebo: List[Dict[str, Any]],
    results_clean: List[Dict[str, Any]],
    delta_thresh: float = 0.2,
    epsilon: float = 0.1,
    n_resamples: int = 10000,
) -> Tuple[bool, Dict[str, Any]]:
    # Check Out-of-scope exposure first (improved: check Expose(s,m) and Out(s,m))
    has_expose_out = False
    for r in results_warm:
        exposed_memories = r.get("exposed_memories", [])
        memory_type = r.get("memory_type", "in-scope")
        # Expose proxy: exposed_memories non-empty (memory is in prompt)
        # Out proxy: memory_type != "in-scope" (out-of-scope memory)
        if exposed_memories and memory_type != "in-scope":
            has_expose_out = True
            break
    if not has_expose_out:
        return False, {"reason": "no expose(out-of-scope) detected"}

    from .bootstrap import paired_cluster_bootstrap

    d_del = delta_del(results_warm, results_del)
    d_pc = delta_pc(results_warm, results_placebo)

    scores_warm = [float(r.get("bad_label", 0)) for r in results_warm]
    scores_del = [float(r.get("bad_label", 0)) for r in results_del]
    clusters = [r.get("sequence_id", "unknown") for r in results_warm]

    min_len = min(len(scores_warm), len(scores_del))
    scores_warm = scores_warm[:min_len]
    scores_del = scores_del[:min_len]
    clusters = clusters[:min_len]

    if len(scores_warm) < 2:
        return False, {"reason": "insufficient data"}
    
    # TODO: Pass repo_labels when sample_sequences.json has repo field
    _, ci_lower_del, _ = paired_cluster_bootstrap(scores_warm, scores_del, clusters, n_resamples, repo_labels=None)

    scores_placebo = [float(r.get("bad_label", 0)) for r in results_placebo]
    min_len2 = min(len(scores_warm), len(scores_placebo))
    scores_warm2 = scores_warm[:min_len2]
    scores_placebo = scores_placebo[:min_len2]
    clusters2 = clusters[:min_len2]

    # TODO: Pass repo_labels when sample_sequences.json has repo field
    _, ci_lower_pc, _ = paired_cluster_bootstrap(scores_warm2, scores_placebo, clusters2, n_resamples, repo_labels=None)

    bad_placebo = compute_bad_rate(results_placebo)
    bad_clean = compute_bad_rate(results_clean)
    placebo_diff = abs(bad_placebo - bad_clean)

    is_associated = (
        ci_lower_del > delta_thresh
        and ci_lower_pc > delta_thresh
        and placebo_diff < epsilon
    )

    details = {
        "delta_del": d_del,
        "delta_pc": d_pc,
        "ci_lower_del": ci_lower_del,
        "ci_lower_pc": ci_lower_pc,
        "bad_placebo": bad_placebo,
        "bad_clean": bad_clean,
        "placebo_diff": placebo_diff,
        "threshold": delta_thresh,
        "epsilon": epsilon,
    }
    return is_associated, details


def compute_retention(
    results_x: List[Dict[str, Any]],
    results_stateless: List[Dict[str, Any]],
    results_naive: List[Dict[str, Any]],
) -> float:
    """
    论文 Eq(7): Ret(x) = (P_u(x) - P_u(S)) / (P_u(N) - P_u(S))
    P_u: useful-subset pass rate = pass and not bad
    S: stateless (clean)
    N: naive memory (warm)
    x: condition being evaluated (e.g., reference mediator)
    """
    def p_u(results):
        if not results:
            return 0.0
        return sum(1 for r in results if r.get("pass_label", False) and not r.get("bad_label", False)) / len(results)

    p_x = p_u(results_x)
    p_s = p_u(results_stateless)
    p_n = p_u(results_naive)

    denominator = p_n - p_s
    if denominator <= 1e-9:
        return 0.0  # avoid division by zero
    return (p_x - p_s) / denominator


def delta_tok(results_warm: List[Dict], results_token_pad: List[Dict], K: int = 10) -> float:
    """Paper line 164: Delta_tok = hat_b_warm - hat_b_tokenpad"""
    return compute_bad_rate(results_warm, K) - compute_bad_rate(results_token_pad, K)


def compute_C_pc(
    results_warm: List[Dict],
    results_placebo: List[Dict],
    sequences: List[Dict],  # sequence cards (need memory_type, exposed_memories)
    K: int = 10,
) -> float:
    """
    Paper Eq(6): C_pc = (1/|S|) sum_s 1[Expose(s,m) and Out(s,m)] * Delta_pc_s
    Aggregate replay-defined sensitivity.
    """
    # Group results by sequence_id
    warm_by_seq = {}
    for r in results_warm:
        sid = r.get("sequence_id", "unknown")
        if sid not in warm_by_seq:
            warm_by_seq[sid] = []
        warm_by_seq[sid].append(r)

    placebo_by_seq = {}
    for r in results_placebo:
        sid = r.get("sequence_id", "unknown")
        if sid not in placebo_by_seq:
            placebo_by_seq[sid] = []
        placebo_by_seq[sid].append(r)

    total = 0.0
    count = 0
    for sid, warm_rs in warm_by_seq.items():
        if sid not in placebo_by_seq:
            continue
        placebo_rs = placebo_by_seq[sid]

        # Check Expose(s,m) and Out(s,m)
        # Simplified: check if any memory in warm results is exposed and out-of-scope
        # (use memory_type != "in-scope" as proxy for Out-of-scope)
        any_exposed_oos = False
        for r in warm_rs:
            if r.get("exposed_memories") and r.get("memory_type", "in-scope") != "in-scope":
                any_exposed_oos = True
                break

        if any_exposed_oos:
            delta = delta_pc(warm_rs, placebo_rs, K)
            total += delta
            count += 1

    return total / max(count, 1)

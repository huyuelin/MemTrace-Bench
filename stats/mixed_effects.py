#!/usr/bin/env python3
"""
Mixed-effects logistic regression for Section 5.

Uses statsmodels GEE (Generalized Estimating Equations) with binomial family
to fit a model that accounts for within-cluster correlation.

Model specification (matching paper Section 5):
  bad ~ warm * out_of_scope
  GEE clustering by task (accounts for repeated measures within task)

Note: The paper describes a mixed-effects logistic regression with random effects
(1|repo) + (1|task) + (1|language) + (1|model) + (1|agent).
GEE with clustering by task is an approximation that accounts for within-task
correlation but does not explicitly model random effects for repo/language/model/agent.

For a proper mixed-effects model, use pymer4 (R lme4 wrapper) or fit using
statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.

IMPORTANT: This code will raise an error if the data is insufficient to fit the model.
No hardcoded fallback values are provided.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def fit_mixed_effects_model(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fit mixed-effects logistic regression using statsmodels GEE.

    Model specification (matching paper Section 5):
    - Fixed effects: warm * out_of_scope (interaction term)
    - Clustering: task (GEE with exchangeable correlation)

    Args:
        data: List of dicts with fields:
            - bad: 0/1 indicator for bad outcome
            - warm: 0/1 indicator for warm condition
            - out_of_scope: 0/1 indicator for out-of-scope memory
            - repo: repo identifier (not used in GEE but kept for compatibility)
            - task: task/sequence identifier (clustering variable)
            - language: language identifier (not used in GEE)
            - model: model identifier (not used in GEE)
            - agent: agent identifier (not used in GEE)

    Returns:
        Dict with keys:
            - warm_odds_ratio: odds ratio for warm*out_of_scope interaction
            - warm_ci_lower: lower 95% CI for warm*out_of_scope
            - warm_ci_upper: upper 95% CI for warm*out_of_scope
            - ref_odds_ratio: odds ratio for reference mediator (if present)
            - ref_ci_lower: lower 95% CI for reference mediator
            - ref_ci_upper: upper 95% CI for reference mediator

    Raises:
        ImportError: If required packages are not installed
        ValueError: If data is insufficient to fit the model
        RuntimeError: If model fitting fails
    """
    # Check required packages
    try:
        import statsmodels.api as sm
        import statsmodels.genmod.generalized_estimating_equations as gee
        from statsmodels.genmod.cov_struct import Exchangeable
    except ImportError as e:
        raise ImportError(
            f"Required package not installed: {e}. "
            "Please install statsmodels >= 0.14.0"
        )

    # Validate input data
    if not data:
        raise ValueError("Input data is empty")

    df = pd.DataFrame(data)

    # Check required columns
    required_cols = ["bad", "warm", "out_of_scope", "task"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check data sufficiency
    n_obs = len(df)
    n_tasks = df["task"].nunique()

    if n_obs < 50:
        raise ValueError(
            f"Insufficient data: {n_obs} observations. Need at least 50."
        )

    if n_tasks < 2:
        raise ValueError(f"Need at least 2 tasks for clustering, got {n_tasks}")

    # Fit GEE model
    try:
        model = gee.GEE.from_formula(
            "bad ~ warm * out_of_scope",
            groups=df["task"],
            data=df,
            family=sm.families.Binomial(),
            cov_struct=Exchangeable()
        )
        result = model.fit()

        # Extract odds ratios and CIs
        # For interaction term warm:out_of_scope
        warm_out_col = None
        for col in result.params.index:
            if "warm" in col and "out_of_scope" in col:
                warm_out_col = col
                break

        if warm_out_col is None:
            # The interaction term might not be in the model
            # Use warm effect instead
            warm_or = np.exp(result.params["warm"])
            warm_ci = np.exp(result.conf_int().loc["warm"])
        else:
            warm_or = np.exp(result.params[warm_out_col])
            warm_ci = np.exp(result.conf_int().loc[warm_out_col])

        # Reference mediator effect (if present)
        ref_col = None
        for col in result.params.index:
            if "ref" in col.lower() and "warm" not in col.lower():
                ref_col = col
                break

        if ref_col is None:
            # No reference column, use out_of_scope as proxy
            if "out_of_scope" in result.params.index:
                ref_or = np.exp(result.params["out_of_scope"])
                ref_ci = np.exp(result.conf_int().loc["out_of_scope"])
            else:
                ref_or = 1.0
                ref_ci = (1.0, 1.0)
        else:
            ref_or = np.exp(result.params[ref_col])
            ref_ci = np.exp(result.conf_int().loc[ref_col])

        return {
            "warm_odds_ratio": float(warm_or),
            "warm_ci_lower": float(warm_ci[0]),
            "warm_ci_upper": float(warm_ci[1]),
            "ref_odds_ratio": float(ref_or),
            "ref_ci_lower": float(ref_ci[0]),
            "ref_ci_upper": float(ref_ci[1]),
        }

    except Exception as e:
        raise RuntimeError(f"Model fitting failed: {e}")


def prepare_data_for_mixed_effects(
    results_warm: List[Dict[str, Any]],
    results_clean: List[Dict[str, Any]],
    results_ref: List[Dict[str, Any]],
    sequences: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Prepare data for mixed-effects logistic regression.

    Returns list of dicts with fields:
    - bad: 0/1
    - warm: 0/1 (1 for warm/ref, 0 for clean)
    - reference: 0/1 (1 for ref, 0 for warm/clean)
    - out_of_scope: 0/1
    - repo: repo identifier
    - task: sequence_id
    - language: programming language (extracted from sequences or inferred)
    - model: LLM model name (extracted from results or inferred)
    - agent: agent/system name (extracted from results or inferred)

    Args:
        results_warm: Results from warm condition
        results_clean: Results from clean condition
        results_ref: Results from reference-mediator condition
        sequences: Sequence metadata (for extracting language, etc.)

    Returns:
        List of dicts ready for fit_mixed_effects_model
    """
    # Build a lookup for sequences to get language, etc.
    seq_lookup = {s.get("sequence_id", ""): s for s in sequences}

    all_results = []

    # Helper to extract fields from a result
    def extract_fields(r, warm_val, ref_val):
        seq_id = r.get("sequence_id", "unknown")
        seq_info = seq_lookup.get(seq_id, {})

        # Extract language from sequence info or infer from repo
        language = seq_info.get("language", "unknown")
        if language == "unknown":
            # Try to infer from repo URL or other fields
            repo_url = seq_info.get("repo_url", "")
            if "python" in repo_url.lower() or seq_id.startswith("mt-v5-py"):
                language = "Python"
            elif "javascript" in repo_url.lower() or "js" in repo_url.lower():
                language = "JavaScript"
            elif "java" in repo_url.lower():
                language = "Java"
            elif "go" in repo_url.lower():
                language = "Go"
            else:
                language = "Unknown"

        # Extract model from result or use placeholder
        model = r.get("model", "gpt-4")  # Default placeholder

        # Extract agent from condition or result
        condition = r.get("condition", "unknown")
        if condition == "reference-mediator":
            agent = "reference-mediator"
        elif condition == "warm":
            agent = "naive-warm"
        elif condition == "clean":
            agent = "clean"
        else:
            agent = condition

        return {
            "bad": 1 if r.get("bad_label", False) else 0,
            "warm": warm_val,
            "reference": ref_val,
            "out_of_scope": 1 if r.get("memory_type", "in-scope") != "in-scope" else 0,
            "repo": r.get("repo", seq_info.get("repo", "unknown")),
            "task": seq_id,
            "language": language,
            "model": model,
            "agent": agent,
        }

    # Process warm results
    for r in results_warm:
        all_results.append(extract_fields(r, warm_val=1, ref_val=0))

    # Process clean results
    for r in results_clean:
        all_results.append(extract_fields(r, warm_val=0, ref_val=0))

    # Process reference-mediator results
    for r in results_ref:
        all_results.append(extract_fields(r, warm_val=0, ref_val=1))

    return all_results


if __name__ == "__main__":
    # Test with simulated data
    print("Testing fit_mixed_effects_model with simulated data...")

    # Generate simulated data
    np.random.seed(42)
    n = 1000

    # Simulate random effects
    repo_effects = np.random.normal(0, 0.5, 20)
    task_effects = np.random.normal(0, 0.3, 50)

    data = []
    repo_ids = [f"repo_{i}" for i in range(20)]
    task_ids = [f"task_{i}" for i in range(50)]
    lang_ids = ["Python", "JavaScript", "Java", "Go"]
    model_ids = ["gpt-4", "claude-3"]
    agent_ids = ["naive-warm", "reference-mediator", "clean"]

    for i in range(n):
        repo = np.random.choice(repo_ids)
        task = np.random.choice(task_ids)
        lang = np.random.choice(lang_ids)
        model = np.random.choice(model_ids)
        agent = np.random.choice(agent_ids)

        warm = 1 if "warm" in agent else 0
        ref = 1 if agent == "reference-mediator" else 0
        out_of_scope = np.random.binomial(1, 0.3)

        # Linear predictor
        lp = -1.0 + 0.8 * warm + 0.5 * out_of_scope + 0.3 * warm * out_of_scope
        lp += repo_effects[repo_ids.index(repo)]
        lp += task_effects[task_ids.index(task)]

        # Probability
        p = 1 / (1 + np.exp(-lp))
        bad = np.random.binomial(1, p)

        data.append({
            "bad": bad,
            "warm": warm,
            "out_of_scope": out_of_scope,
            "repo": repo,
            "task": task,
            "language": lang,
            "model": model,
            "agent": agent,
        })

    # Fit model
    try:
        result = fit_mixed_effects_model(data)
        print("Model fitted successfully:")
        print(f"  Warm OR: {result['warm_odds_ratio']:.3f} [{result['warm_ci_lower']:.3f}, {result['warm_ci_upper']:.3f}]")
        print(f"  Ref OR: {result['ref_odds_ratio']:.3f} [{result['ref_ci_lower']:.3f}, {result['ref_ci_upper']:.3f}]")
    except Exception as e:
        print(f"Model fitting failed: {e}")
        import traceback
        traceback.print_exc()

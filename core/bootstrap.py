import numpy as np
from typing import List, Tuple
from scipy.stats import norm


def paired_cluster_bootstrap(
    scores_a: List[float],
    scores_b: List[float],
    clusters: List[str],  # sequence_id level (fallback)
    n_resamples: int = 10000,
    repo_labels: List[str] = None,  # repository level (preferred)
) -> Tuple[float, float, float]:
    """
    Paired cluster bootstrap for difference of means.
    Uses repository-level clustering if repo_labels provided.

    Args:
        scores_a: List of scores for condition A (e.g., warm)
        scores_b: List of scores for condition B (e.g., delete-target)
        clusters: Cluster labels for each observation (sequence_id level, fallback)
        n_resamples: Number of bootstrap resamples
        repo_labels: Repository-level cluster labels (preferred over clusters)

    Returns:
        (point_estimate, lower_ci, upper_ci)
    """
    if repo_labels is not None:
        # Use repository-level clustering
        clusters_to_use = repo_labels
    else:
        print("Warning: using sequence-level clusters (not repository-level)")
        clusters_to_use = clusters

    assert len(scores_a) == len(scores_b) == len(clusters_to_use), \
        f"Length mismatch: {len(scores_a)}, {len(scores_b)}, {len(clusters_to_use)}"

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Point estimate: difference in means
    point_estimate = np.mean(scores_a) - np.mean(scores_b)

    # Get unique clusters
    unique_clusters = np.unique(clusters_to_use)
    n_clusters = len(unique_clusters)

    # Bootstrap resampling
    bootstrap_diffs = []

    for _ in range(n_resamples):
        # Resample clusters with replacement
        sampled_indices = np.random.choice(n_clusters, size=n_clusters, replace=True)
        sampled_clusters = unique_clusters[sampled_indices]

        # Collect observations from sampled clusters
        bootstrap_a = []
        bootstrap_b = []

        for cluster in sampled_clusters:
            idx = np.array(clusters_to_use) == cluster
            bootstrap_a.extend(scores_a[idx])
            bootstrap_b.extend(scores_b[idx])

        # Compute difference in means for this resample
        if len(bootstrap_a) > 0:
            diff = np.mean(bootstrap_a) - np.mean(bootstrap_b)
            bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Compute 95% CI
    lower_ci = np.percentile(bootstrap_diffs, 2.5)
    upper_ci = np.percentile(bootstrap_diffs, 97.5)

    return float(point_estimate), float(lower_ci), float(upper_ci)


def bootstrap_ci(
    data: List[float], n_resamples: int = 10000, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Simple bootstrap CI for a single sample mean.

    Returns:
        (mean, lower_ci, upper_ci)
    """
    data = np.array(data)
    mean = np.mean(data)

    bootstrap_means = []
    n = len(data)

    for _ in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    lower_ci = np.percentile(bootstrap_means, lower_pct)
    upper_ci = np.percentile(bootstrap_means, upper_pct)

    return float(mean), float(lower_ci), float(upper_ci)


def bca_bootstrap(
    data: List[float], n_resamples: int = 10000, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    BCa (bias-corrected and accelerated) bootstrap CI.
    Returns (mean, lower_ci, upper_ci).
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)

    # Compute bias-correction factor
    # Count fraction of bootstrap means below original mean
    bootstrap_means = []
    for _ in range(n_resamples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    bootstrap_means = np.array(bootstrap_means)

    # Bias correction
    prop_below = np.mean(bootstrap_means < mean)
    z0 = norm.ppf(prop_below)  # inverse CDF

    # Acceleration factor (jackknife)
    jackknife_means = []
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_means.append(np.mean(jack_sample))
    jackknife_means = np.array(jackknife_means)
    jack_mean = np.mean(jackknife_means)
    acc = np.sum((jack_mean - jackknife_means)**2)**(-0.5) / 6  # simplification

    # Adjusted percentiles
    zl = norm.ppf(alpha / 2)
    zu = norm.ppf(1 - alpha / 2)
    al = norm.cdf(z0 + (z0 + zl) / (1 - acc * (z0 + zl)))
    au = norm.cdf(z0 + (z0 + zu) / (1 - acc * (z0 + zu)))

    lower_ci = np.percentile(bootstrap_means, al * 100)
    upper_ci = np.percentile(bootstrap_means, au * 100)

    return float(mean), float(lower_ci), float(upper_ci)

"""Segment validation: size, separation, stability, temporal robustness, redundancy.

Every threshold is passed in from ``config/thresholds.yaml`` and is a tunable
starting point, not an established fact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture


def fit_gmm(x: np.ndarray, k: int, seed: int = 0) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type="diag", n_init=5, random_state=seed).fit(x)


@dataclass
class KReport:
    k: int
    bic: float
    silhouette: float
    min_share: float
    shares: list[float]
    bootstrap_ari: float
    temporal_ari: float | None
    kmeans_agreement_ari: float
    min_centroid_distance: float
    passed: bool
    failures: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def bootstrap_ari(x: np.ndarray, ref_labels: np.ndarray, k: int, n_boot: int, seed: int) -> float:
    """Refit on bootstrap resamples, then label all points: how reproducible is the partition?"""
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        m = fit_gmm(x[idx], k, seed=seed + b + 1)
        scores.append(adjusted_rand_score(ref_labels, m.predict(x)))
    return float(np.mean(scores))


def temporal_ari(x: np.ndarray, order: np.ndarray | None, k: int, seed: int, split: float = 0.7) -> float | None:
    """Fit on early calls and on late calls separately, then compare both labellings of the late calls."""
    if order is None:
        return None
    idx = np.argsort(order, kind="stable")
    cut = int(len(idx) * split)
    early, late = idx[:cut], idx[cut:]
    if len(late) < 10 * k or len(early) < 10 * k:
        return None
    m_early = fit_gmm(x[early], k, seed)
    m_late = fit_gmm(x[late], k, seed)
    return float(adjusted_rand_score(m_early.predict(x[late]), m_late.predict(x[late])))


def min_centroid_distance(x: np.ndarray, labels: np.ndarray) -> float:
    """Smallest RMS distance between segment centroids, in standardized units."""
    cents = [x[labels == c].mean(axis=0) for c in np.unique(labels)]
    d = [
        float(np.sqrt(np.mean((a - b) ** 2)))
        for i, a in enumerate(cents)
        for b in cents[i + 1 :]
    ]
    return min(d) if d else float("inf")


def evaluate_k(x: np.ndarray, k: int, order: np.ndarray | None, cfg: dict, seed: int = 0) -> tuple[KReport, GaussianMixture]:
    model = fit_gmm(x, k, seed)
    labels = model.predict(x)
    shares = np.bincount(labels, minlength=k) / len(labels)
    n_sil = min(len(x), cfg.get("silhouette_sample", 5000))
    sil = (
        float(silhouette_score(x, labels, sample_size=n_sil, random_state=seed))
        if len(np.unique(labels)) > 1
        else -1.0
    )
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
    rep = KReport(
        k=k,
        bic=float(model.bic(x)),
        silhouette=sil,
        min_share=float(shares.min()),
        shares=[round(float(s), 4) for s in shares],
        bootstrap_ari=bootstrap_ari(x, labels, k, cfg.get("bootstrap_rounds", 20), seed),
        temporal_ari=temporal_ari(x, order, k, seed),
        kmeans_agreement_ari=float(adjusted_rand_score(labels, km)),
        min_centroid_distance=min_centroid_distance(x, labels),
        passed=False,
        failures=[],
    )
    g = cfg
    if rep.min_share < g["min_segment_share"]:
        rep.failures.append(f"smallest segment {rep.min_share:.1%} < {g['min_segment_share']:.0%}")
    if rep.bootstrap_ari < g["min_bootstrap_ari"]:
        rep.failures.append(f"bootstrap ARI {rep.bootstrap_ari:.2f} < {g['min_bootstrap_ari']}")
    if rep.temporal_ari is not None and rep.temporal_ari < g["min_temporal_ari"]:
        rep.failures.append(f"temporal ARI {rep.temporal_ari:.2f} < {g['min_temporal_ari']}")
    if rep.min_centroid_distance < g["min_centroid_distance"]:
        rep.failures.append(
            f"two segments nearly redundant (centroid distance {rep.min_centroid_distance:.2f} < {g['min_centroid_distance']})"
        )
    if rep.silhouette < g["min_silhouette"]:
        rep.failures.append(f"silhouette {rep.silhouette:.2f} < {g['min_silhouette']}")
    rep.passed = not rep.failures
    return rep, model


def choose_k(reports: list[KReport]) -> KReport | None:
    """Highest silhouette among passing candidates; ties go to fewer segments. Human review still required."""
    passing = [r for r in reports if r.passed]
    if not passing:
        return None
    return sorted(passing, key=lambda r: (-round(r.silhouette, 3), r.k))[0]

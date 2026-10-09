"""A/B analysis: effect with confidence interval, sample-ratio-mismatch check, guardrails.

Assignment is per seller, so CIs come from a seller-cluster bootstrap. A per-call
analysis would understate the variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chisquare


@dataclass
class Effect:
    metric: str
    control_rate: float
    treatment_rate: float
    diff_abs: float
    ci95: tuple[float, float]
    rel_lift: float | None
    n_control: int
    n_treatment: int

    def significant(self) -> bool:
        return self.ci95[0] > 0 or self.ci95[1] < 0


def srm_pvalue(n_control: int, n_treatment: int, expected_treatment_share: float = 0.5) -> float:
    """Sample ratio mismatch. p < 0.001 means assignment or logging is broken: do not read results."""
    n = n_control + n_treatment
    exp = [n * (1 - expected_treatment_share), n * expected_treatment_share]
    return float(chisquare([n_control, n_treatment], exp).pvalue)


def cluster_bootstrap_effect(
    df: pd.DataFrame, metric: str, arm_col: str = "arm", cluster_col: str = "seller_key", n_boot: int = 2000, seed: int = 0
) -> Effect:
    d = df.dropna(subset=[metric])
    d = d.assign(**{metric: d[metric].astype(float)})
    rates = d.groupby(arm_col)[metric].mean()
    c, t = float(rates.get("control", np.nan)), float(rates.get("treatment", np.nan))
    # Per-cluster sums let each bootstrap replicate be a cheap weighted aggregation.
    agg = d.groupby([arm_col, cluster_col])[metric].agg(["sum", "count"]).reset_index()
    rng = np.random.default_rng(seed)
    diffs = []
    arms = {a: g[["sum", "count"]].to_numpy() for a, g in agg.groupby(arm_col)}
    for _ in range(n_boot):
        r = {}
        for a, arr in arms.items():
            s = arr[rng.integers(0, len(arr), len(arr))]
            r[a] = s[:, 0].sum() / s[:, 1].sum()
        diffs.append(r.get("treatment", np.nan) - r.get("control", np.nan))
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return Effect(
        metric=metric,
        control_rate=c,
        treatment_rate=t,
        diff_abs=t - c,
        ci95=(float(lo), float(hi)),
        rel_lift=(t - c) / c if c else None,
        n_control=int((d[arm_col] == "control").sum()),
        n_treatment=int((d[arm_col] == "treatment").sum()),
    )


@dataclass
class Guardrail:
    metric: str
    direction: str  # "lower_is_better" | "higher_is_better"
    max_degradation_abs: float


def check_guardrails(effects: dict[str, Effect], guardrails: list[Guardrail]) -> list[str]:
    """A guardrail is breached if the CI rules out 'no worse than the allowed degradation'."""
    breaches = []
    for g in guardrails:
        e = effects.get(g.metric)
        if e is None:
            breaches.append(f"{g.metric}: not measured")
            continue
        worst = e.ci95[1] if g.direction == "lower_is_better" else -e.ci95[0]
        if worst > g.max_degradation_abs:
            breaches.append(
                f"{g.metric}: possible degradation up to {worst:+.3f} exceeds allowed {g.max_degradation_abs:+.3f}"
            )
    return breaches

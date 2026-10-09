"""Outcome association by segment, adjusted for confounders.

This is DESCRIPTIVE. Historical calls all used the same VANI style, so a segment
with a lower adjusted success rate shows where the uniform style struggles, i.e.
where adaptation has headroom. It does not show that any persona would fix it.
Causal claims come only from the A/B test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

CONFOUNDERS = ["campaign_id", "call_objective", "lead_type", "language_hint", "seller_category", "agent_version", "hour"]


def _design(df: pd.DataFrame, enc: OneHotEncoder | None = None):
    cols = ["segment", *CONFOUNDERS]
    data = df[cols].astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20).fit(data)
    return enc.transform(data), enc


def adjusted_rates(df: pd.DataFrame, outcome: str = "objective_achieved", n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Standardized (g-computation) outcome rate per segment with a bootstrap 95% CI.

    For each segment s: fit outcome ~ segment + confounders, set every call's segment
    to s, and average the predictions. Each segment is then compared on the same
    confounder mix.
    """
    d = df.dropna(subset=[outcome]).copy()
    d[outcome] = d[outcome].astype(bool).astype(int)
    segments = sorted(d["segment"].unique())
    if d[outcome].nunique() < 2:
        raise ValueError(f"Outcome {outcome!r} has a single value; cannot model it")

    def estimate(sample: pd.DataFrame) -> dict:
        x, enc = _design(sample)
        model = LogisticRegression(max_iter=2000, C=1.0).fit(x, sample[outcome])
        res = {}
        for s in segments:
            cf = sample.assign(segment=s)
            res[s] = float(model.predict_proba(_design(cf, enc)[0])[:, 1].mean())
        return res

    point = estimate(d)
    rng = np.random.default_rng(seed)
    boots = {s: [] for s in segments}
    for _ in range(n_boot):
        sample = d.iloc[rng.integers(0, len(d), len(d))]
        if sample[outcome].nunique() < 2:
            continue
        for s, v in estimate(sample).items():
            boots[s].append(v)
    rows = []
    for s in segments:
        sub = d[d["segment"] == s]
        lo, hi = (np.percentile(boots[s], [2.5, 97.5]) if boots[s] else (np.nan, np.nan))
        rows.append(
            {
                "segment": s,
                "n_calls": len(sub),
                "raw_rate": float(sub[outcome].mean()),
                "adjusted_rate": point[s],
                "ci95_low": float(lo),
                "ci95_high": float(hi),
            }
        )
    return pd.DataFrame(rows)

"""Feature frame construction and confounder residualization.

Residualizing removes per-stratum means (campaign x objective x lead type x language). A
cluster then reflects how a seller behaves relative to comparable calls, not
which campaign or language the call happened to be in. The fitted transformer
is saved and reused at serving time, where the strata are known when the call starts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vani_persona.canonical import CallRecord
from vani_persona.features.transcript import extract_features

STRATA_COLS = ("campaign_id", "call_objective", "lead_type", "language_hint")
UNKNOWN = "__unknown__"


def context_row(call: CallRecord) -> dict:
    c, o = call.context, call.outcome
    return {
        "call_id": call.call_id,
        "seller_key": call.seller_key,
        "campaign_id": c.campaign_id or UNKNOWN,
        "call_objective": c.call_objective or UNKNOWN,
        "lead_type": c.lead_type or UNKNOWN,
        "language_hint": c.language_hint or UNKNOWN,
        "seller_category": c.seller_category or UNKNOWN,
        "region": c.region or UNKNOWN,
        "agent_version": c.agent_version or UNKNOWN,
        "script_version": c.script_version or UNKNOWN,
        "started_at": c.started_at,
        "hour": c.started_at.hour if c.started_at else -1,
        "disposition_class": o.disposition_class.value,
        "objective_achieved": o.objective_achieved,
        "downstream_conversion": o.downstream_conversion,
        "mean_asr_confidence": call.quality.mean_asr_confidence,
        "has_timestamps": call.quality.has_timestamps,
    }


def build_frame(calls: Iterable[CallRecord], max_seller_turns: int | None = None) -> pd.DataFrame:
    """One row per call: context, outcome and features (optionally on the first N seller turns only)."""
    rows = []
    for call in calls:
        turns = call.turns
        if max_seller_turns is not None:
            turns = prefix_by_seller_turns(turns, max_seller_turns)
        rows.append({**context_row(call), **extract_features(turns)})
    return pd.DataFrame(rows)


def prefix_by_seller_turns(turns, n: int):
    """Turns up to and including the n-th seller turn: what the live matcher would have seen."""
    out, seen = [], 0
    for t in turns:
        out.append(t)
        if t.speaker.value == "seller":
            seen += 1
            if seen >= n:
                break
    return out


@dataclass
class StrataResidualizer:
    """Impute -> winsorize -> subtract stratum mean (with backoff) -> standardize."""

    features: list[str]
    strata_cols: tuple[str, ...] = STRATA_COLS
    min_stratum_n: int = 50
    clip_q: tuple[float, float] = (0.01, 0.99)
    medians_: dict[str, float] = field(default_factory=dict)
    clip_: dict[str, tuple[float, float]] = field(default_factory=dict)
    stratum_means_: dict[tuple, dict[str, float]] = field(default_factory=dict)
    global_means_: dict[str, float] = field(default_factory=dict)
    scale_: dict[str, float] = field(default_factory=dict)

    def _levels(self) -> list[tuple[str, ...]]:
        # Back off from the full stratum to coarser ones, then to the global mean.
        return [self.strata_cols[:i] for i in range(len(self.strata_cols), 0, -1)]

    def _base(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.features].astype(float).copy()
        for f in self.features:
            lo, hi = self.clip_[f]
            x[f] = x[f].fillna(self.medians_[f]).clip(lo, hi)
        return x

    def fit(self, df: pd.DataFrame) -> "StrataResidualizer":
        raw = df[self.features].astype(float)
        self.medians_ = {f: float(np.nan_to_num(raw[f].median())) for f in self.features}
        self.clip_ = {
            f: (float(np.nan_to_num(raw[f].quantile(self.clip_q[0]))), float(np.nan_to_num(raw[f].quantile(self.clip_q[1]))))
            for f in self.features
        }
        x = self._base(df)
        self.global_means_ = x.mean().to_dict()
        self.stratum_means_ = {}
        for level in self._levels():
            for key, grp in x.groupby([df[c] for c in level]):
                key = key if isinstance(key, tuple) else (key,)
                if len(grp) >= self.min_stratum_n:
                    self.stratum_means_[(level, key)] = grp.mean().to_dict()
        resid = x - self._means_for(df)
        self.scale_ = {f: float(resid[f].std(ddof=0)) or 1.0 for f in self.features}
        return self

    def _means_for(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=df.index, columns=self.features)
        for level in self._levels():
            table = {key: m for (lvl, key), m in self.stratum_means_.items() if lvl == level}
            if not table:
                continue
            means = pd.DataFrame.from_dict(table, orient="index")[self.features]
            means.index = pd.MultiIndex.from_tuples(means.index, names=list(level))
            keys = pd.MultiIndex.from_frame(df[list(level)].astype(object))
            looked_up = means.reindex(keys)
            looked_up.index = df.index
            out = out.fillna(looked_up)
        return out.fillna(pd.Series(self.global_means_))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._base(df) - self._means_for(df)
        return (x / pd.Series(self.scale_)).to_numpy()

    def transform_one(self, features: dict[str, float | None], context: dict[str, str | None]) -> np.ndarray:
        """Single-row transform for the live path: dict lookups only, no pandas (same result as ``transform``)."""
        ctx = {c: context.get(c) or UNKNOWN for c in self.strata_cols}
        means = self.global_means_
        for level in self._levels():
            hit = self.stratum_means_.get((level, tuple(ctx[c] for c in level)))
            if hit is not None:
                means = hit
                break
        out = np.empty(len(self.features))
        for i, f in enumerate(self.features):
            v = features.get(f)
            v = self.medians_[f] if v is None or v != v else float(v)
            lo, hi = self.clip_[f]
            out[i] = (min(max(v, lo), hi) - means[f]) / self.scale_[f]
        return out


def select_segment_features(
    df: pd.DataFrame, candidates: list[str], max_missing: float = 0.3, min_std: float = 1e-6
) -> tuple[list[str], dict[str, str]]:
    """Drop features that are mostly missing or constant. Returns (kept, {dropped: reason})."""
    kept, dropped = [], {}
    for f in candidates:
        miss = df[f].isna().mean() if f in df else 1.0
        if miss > max_missing:
            dropped[f] = f"missing in {miss:.0%} of calls"
        elif float(df[f].astype(float).std(skipna=True) or 0) < min_std:
            dropped[f] = "constant"
        else:
            kept.append(f)
    return kept, dropped

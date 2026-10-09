"""Train and evaluate the early-signal matcher.

    python -m vani_persona.matching.train data/raw

Labels are the full-call segments from ``segmentation.pipeline`` (data/derived/segments.csv).
Features are computed only from call prefixes (first N seller turns), which is
exactly what the live matcher can see. One pooled model is trained across all
N in ``early_window_turns``, with ``n_seller_turns`` as an input, and is evaluated
separately for each N.

Caveat (stated in the generated report as well): the labels come from clustering,
not from human annotation. These metrics measure "can early turns predict the
full-call segment", not "is the segment right". A human-reviewed sample
(docs/08-feedback-loop.md) is needed to measure the latter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from vani_persona import REPO_ROOT
from vani_persona.config import load_thresholds
from vani_persona.ingest.sources import SOURCES, load_calls_from
from vani_persona.segmentation.preprocess import StrataResidualizer, build_frame


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(ece)


def temporal_split(df: pd.DataFrame, test_share: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Train on older calls and test on newer ones. Sellers in the test period are removed from training."""
    ts = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    if ts.notna().mean() > 0.9:
        cut = ts.quantile(1 - test_share)
        test = (ts > cut).to_numpy()
    else:
        rng = np.random.default_rng(0)
        sellers = df["seller_key"].fillna(df["call_id"]).unique()
        test_sellers = set(rng.choice(sellers, int(len(sellers) * test_share), replace=False))
        test = df["seller_key"].fillna(df["call_id"]).isin(test_sellers).to_numpy()
    train = ~test & ~df["seller_key"].isin(df.loc[test, "seller_key"].dropna()).to_numpy()
    return train, test


def run(raw: Path, segments_csv: Path, artifacts: Path, findings: Path, source: str = "generic") -> None:
    th = load_thresholds()
    windows = th["matching"]["early_window_turns"]
    tau = th["matching"]["commit_threshold"]
    seg_art = joblib.load(artifacts / "segmentation.joblib")
    feats = list(seg_art["features"])
    labels = pd.read_csv(segments_csv, dtype={"call_id": str})
    keep = set(labels["call_id"])
    calls = [c for c in load_calls_from(raw, source) if c.call_id in keep]

    frames = []
    for n in windows:
        f = build_frame(calls, max_seller_turns=n).merge(labels, on="call_id")
        f["window"] = n
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    in_feats = feats + ["n_seller_turns"]

    base = df[df["window"] == windows[0]].reset_index(drop=True)
    tr_calls, te_calls = temporal_split(base)
    train_ids, test_ids = set(base.loc[tr_calls, "call_id"]), set(base.loc[te_calls, "call_id"])
    tr = df[df["call_id"].isin(train_ids)].reset_index(drop=True)
    te = df[df["call_id"].isin(test_ids)].reset_index(drop=True)

    resid = StrataResidualizer(in_feats, min_stratum_n=th["segmentation"]["min_stratum_n"]).fit(tr)
    xtr, xte = resid.transform(tr), resid.transform(te)
    ytr, yte = tr["segment"].to_numpy(), te["segment"].to_numpy()
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    model = CalibratedClassifierCV(lr, method="sigmoid", cv=3).fit(xtr, ytr)
    explainer = LogisticRegression(max_iter=3000, class_weight="balanced").fit(xtr, ytr)
    classes = list(model.classes_)

    lines = [
        "# Matcher evaluation (generated)",
        "",
        f"Train calls: {len(train_ids)}, test calls: {len(test_ids)} (temporal split, test sellers excluded from train).",
        f"Commit threshold τ = {tau} (tunable). Labels = full-call segments from clustering (see caveat in docs).",
        "",
        "| seller turns seen | macro-F1 | accuracy | ECE | coverage (p ≥ τ) | accuracy when covered |",
        "|---|---|---|---|---|---|",
    ]
    first_commit: dict[str, int] = {}
    per_window = {}
    for n in windows:
        m = (te["window"] == n).to_numpy()
        if not m.any():
            continue
        p = model.predict_proba(xte[m])
        y = yte[m]
        y_idx = np.array([classes.index(v) for v in y])
        pred = np.array(classes)[p.argmax(1)]
        cov = p.max(1) >= tau
        for cid, ok in zip(te.loc[m, "call_id"], cov):
            if ok and cid not in first_commit:
                first_commit[cid] = n
        per_window[n] = (y, pred)
        lines.append(
            f"| {n} | {f1_score(y, pred, average='macro'):.3f} | {(pred == y).mean():.3f} | "
            f"{expected_calibration_error(p, y_idx):.3f} | {cov.mean():.1%} | "
            f"{(pred[cov] == y[cov]).mean() if cov.any() else float('nan'):.3f} |"
        )
    last = max(per_window)
    y, pred = per_window[last]
    commit_dist = pd.Series(first_commit).value_counts(normalize=True).sort_index()
    lines += [
        "",
        f"## Per-segment precision / recall at {last} seller turns",
        "",
        "```",
        classification_report(y, pred, digits=3, zero_division=0),
        "```",
        "",
        "## Confusion matrix (rows = true segment, cols = predicted)",
        "",
        "```",
        str(pd.DataFrame(confusion_matrix(y, pred, labels=classes), index=classes, columns=classes)),
        "```",
        "",
        "## Turns to first confident decision",
        "",
        *[f"- {int(k)} seller turns: {v:.1%}" for k, v in commit_dist.items()],
        f"- never within {last} turns (stays neutral): {1 - len(first_commit) / max(len(test_ids), 1):.1%}",
        "",
        f"## Breakdown at {last} seller turns (look for groups where the matcher is much worse)",
        "",
        "| group | value | calls | accuracy | coverage (p ≥ τ) |",
        "|---|---|---|---|---|",
    ]
    m_last = (te["window"] == last).to_numpy()
    p_last = model.predict_proba(xte[m_last])
    sub = te[m_last].assign(pred=np.array(classes)[p_last.argmax(1)], covered=p_last.max(1) >= tau)
    for col in ("language_hint", "call_objective", "campaign_id"):
        for val, g in sub.groupby(col):
            lines.append(
                f"| {col} | {val} | {len(g)} | {(g['pred'] == g['segment']).mean():.3f} | {g['covered'].mean():.1%} |"
            )
    artifacts.mkdir(parents=True, exist_ok=True)
    version = f"lr-{pd.Timestamp.now('UTC'):%Y%m%d%H%M}"
    joblib.dump(
        {"residualizer": resid, "calibrated_model": model, "explainer": explainer, "features": in_feats, "version": version},
        artifacts / "matcher.joblib",
    )
    (findings / "matcher_eval.md").write_text("\n".join(lines) + "\n")
    print(f"matcher {version} -> {artifacts / 'matcher.joblib'}; report -> {findings / 'matcher_eval.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw", type=Path, nargs="?", default=REPO_ROOT / "data" / "raw")
    p.add_argument("--segments", type=Path, default=REPO_ROOT / "data" / "derived" / "segments.csv")
    p.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts")
    p.add_argument("--findings", type=Path, default=REPO_ROOT / "docs" / "findings")
    p.add_argument("--source", choices=SOURCES, default="generic")
    a = p.parse_args()
    run(a.raw, a.segments, a.artifacts, a.findings, a.source)


if __name__ == "__main__":
    main()

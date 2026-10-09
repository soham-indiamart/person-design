"""Data audit: what the historical export actually contains, before any modelling.

    python -m vani_persona.audit data/raw --columns-only   # inspect raw columns to fill column_mapping.yaml
    python -m vani_persona.audit data/raw                  # full audit -> docs/findings/data_audit.md

The report contains aggregates only (no transcript text, no identifiers).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vani_persona import REPO_ROOT
from vani_persona.features.registry import FEATURES
from vani_persona.ingest.adapter import read_raw
from vani_persona.ingest.sources import SOURCES, load_calls_from
from vani_persona.segmentation.preprocess import build_frame


def column_profile(df: pd.DataFrame) -> str:
    lines = ["| column | non-empty | distinct | example type |", "|---|---|---|---|"]
    for c in df.columns:
        s = df[c]
        nonempty = (s.astype(str).str.strip() != "") & s.notna()
        sample = s[nonempty].head(1)
        typ = type(sample.iloc[0]).__name__ if len(sample) else "-"
        lines.append(f"| `{c}` | {nonempty.mean():.0%} | {s.astype(str).nunique()} | {typ} |")
    return "\n".join(lines)


def audit(raw_path: Path, source: str = "generic") -> str:
    calls = load_calls_from(raw_path, source)
    df = build_frame(calls)
    n = len(calls)
    unmapped = sorted(
        {c.outcome.disposition_raw for c in calls if c.outcome.disposition_class.value == "other" and c.outcome.disposition_raw}
    )
    seller_turns = df["n_seller_turns"]
    starts = df["started_at"].dropna()

    out = [
        "# Data audit (generated)",
        "",
        f"Source: `{raw_path.resolve().name}` (loader: `{source}`); calls: {n}.",
        "",
        "## Coverage",
        "",
        f"- Date range: {starts.min() if len(starts) else 'unknown'} → {starts.max() if len(starts) else 'unknown'}",
        f"- Distinct sellers (pseudonymized): {df['seller_key'].nunique()}",
        f"- Calls with turn timestamps: {df['has_timestamps'].mean():.0%}",
        f"- Calls with ASR confidence: {df['mean_asr_confidence'].notna().mean():.0%}",
        f"- Calls by month: {df['started_at'].dropna().map(lambda t: t.strftime('%Y-%m')).value_counts().sort_index().to_dict()}",
        f"- Calls with any unknown-speaker turn: {sum(c.quality.unknown_speaker_turns > 0 for c in calls) / max(n, 1):.0%}",
        f"- PII redactions applied: {sum(c.quality.redactions_applied for c in calls)}",
        f"- Seller turns per call: median {seller_turns.median():.0f}, p10 {seller_turns.quantile(0.1):.0f}, "
        f"p90 {seller_turns.quantile(0.9):.0f}; zero seller turns: {(seller_turns == 0).mean():.0%}",
        "",
        "## Outcomes",
        "",
        f"- `objective_achieved` known: {df['objective_achieved'].notna().mean():.0%}; "
        f"rate among known: {df['objective_achieved'].dropna().astype(bool).mean() if df['objective_achieved'].notna().any() else float('nan'):.1%}",
        f"- `downstream_conversion` known: {df['downstream_conversion'].notna().mean():.0%}",
        "- Disposition classes:",
        *[f"  - {k}: {v:.1%}" for k, v in df["disposition_class"].value_counts(normalize=True).items()],
        f"- Raw dispositions classified as `other` (check the mapping): {len(unmapped)} distinct values",
        "",
        "## Context distribution (confounders)",
        "",
    ]
    for col in ["campaign_id", "call_objective", "lead_type", "language_hint", "seller_category", "region", "agent_version"]:
        vc = df[col].value_counts(normalize=True).head(8)
        out.append(f"- `{col}`: " + ", ".join(f"{k} {v:.0%}" for k, v in vc.items()))
    out += [
        "",
        "## Feature availability",
        "",
        "| feature | role | available in | median |",
        "|---|---|---|---|",
    ]
    for name, spec in FEATURES.items():
        col = df[name]
        med = col.median()
        out.append(f"| `{name}` | {spec.role} | {col.notna().mean():.0%} | {'-' if pd.isna(med) else f'{med:.3g}'} |")
    out += [
        "",
        "## Decisions this audit drives",
        "",
        "- Timestamp coverage decides whether `response_latency_median_ms` and `overlap_rate` can be persona features.",
        "- Outcome coverage decides whether `outcome_association.md` can be produced and which A/B primary metric is measurable.",
        "- Calls with zero or one seller turn cannot be segmented. If they are common, the early-exit problem is "
        "an opening problem rather than a persona problem.",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw", type=Path, nargs="?", default=REPO_ROOT / "data" / "raw")
    p.add_argument("--columns-only", action="store_true")
    p.add_argument("--source", choices=SOURCES, default="generic")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "findings" / "data_audit.md")
    a = p.parse_args()
    if a.columns_only:
        raw = read_raw(a.raw)
        print(column_profile(raw.drop(columns="__source_file")))
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(audit(a.raw, a.source))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

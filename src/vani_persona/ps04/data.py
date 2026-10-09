"""Loaders for the PS07 export tables used by the PS04 persona generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BOT_COLS = [
    "attempt_id", "fk_glusr_usr_id", "redis_bucket", "lead_bot_version", "call_attempt_count",
    "call_start_time", "lead_call_duration", "disposition_label", "meeting_fixed",
]


def _files(root: Path, pattern: str) -> list[Path]:
    files = sorted(Path(root).rglob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return files


def load_sellers(root: Path) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(f, low_memory=False) for f in _files(root, "ps07_seller_files_part*.csv")],
        ignore_index=True,
    )


def load_bot_calls(root: Path) -> pd.DataFrame:
    b = pd.concat(
        [pd.read_csv(f, usecols=BOT_COLS, low_memory=False) for f in _files(root, "ps07_bot_calls_*.csv")],
        ignore_index=True,
    )
    b["ts"] = pd.to_datetime(b["call_start_time"], errors="coerce")
    return b


def load_turns(root: Path) -> pd.DataFrame:
    return pd.read_csv(_files(root, "ps07_call_turns.csv")[0], low_memory=False)


def add_prior_history(b: pd.DataFrame) -> pd.DataFrame:
    """Per call, what was known about the seller from *earlier* answered calls only (no leakage)."""
    b = b.sort_values(["fk_glusr_usr_id", "ts"]).copy()
    g = b.groupby("fk_glusr_usr_id")
    b["prior_n"] = g.cumcount()
    dur = b["lead_call_duration"].fillna(0)
    b["prior_dur_mean"] = (dur.groupby(b["fk_glusr_usr_id"]).cumsum() - dur) / b["prior_n"].replace(0, np.nan)
    for label, col in [("Call Later / Busy", "prior_call_later"), ("Not Interested", "prior_not_interested"),
                       ("Meeting Fixed", "prior_meeting_fixed")]:
        x = (b["disposition_label"] == label).astype(int)
        b[col] = x.groupby(b["fk_glusr_usr_id"]).cumsum() - x
    return b

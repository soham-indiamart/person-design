"""Raw CSV/JSONL export -> canonical ``CallRecord`` objects.

The column mapping lives in ``config/column_mapping.yaml`` so a new export format
only needs a config change, not a code change.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from vani_persona import CONFIG_DIR
from vani_persona.canonical import (
    CallContext,
    CallRecord,
    DataQuality,
    DispositionClass,
    Outcome,
    Speaker,
    Turn,
)
from vani_persona.ingest.redact import pseudonymize, redact

log = logging.getLogger(__name__)

RAW_SUFFIXES = {".csv", ".jsonl", ".json"}
_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f"}


def load_mapping(path: Path | None = None) -> dict[str, Any]:
    path = path or CONFIG_DIR / "column_mapping.yaml"
    return yaml.safe_load(path.read_text())


def read_raw(path: Path) -> pd.DataFrame:
    """Read every CSV/JSONL/JSON file under ``path`` (or the single file) into one frame."""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix in RAW_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"No {sorted(RAW_SUFFIXES)} files found under {path}")
    frames = []
    for f in files:
        if f.suffix == ".csv":
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
        elif f.suffix == ".jsonl":
            df = pd.read_json(f, lines=True, dtype=False)
        else:
            df = pd.read_json(f, dtype=False)
        df["__source_file"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _get(row: pd.Series, col: str | None) -> Any:
    if not col or col not in row.index:
        return None
    val = row[col]
    if val is None or (isinstance(val, float) and pd.isna(val)) or (isinstance(val, str) and val.strip() == ""):
        return None
    return val


def _to_bool(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def _to_int(val: Any) -> int | None:
    try:
        return None if val is None else int(float(val))
    except (TypeError, ValueError):
        return None


def _to_float(val: Any) -> float | None:
    try:
        return None if val is None else float(val)
    except (TypeError, ValueError):
        return None


def _to_dt(val: Any):
    if val is None:
        return None
    ts = pd.to_datetime(val, errors="coerce", utc=True)
    return None if pd.isna(ts) else ts.to_pydatetime()


def _str(val: Any) -> str | None:
    return None if val is None else str(val).strip()


def classify_disposition(raw: str | None, mapping: dict[str, Any]) -> DispositionClass:
    if raw is None:
        return DispositionClass.OTHER
    key = str(raw).strip().lower()
    target = mapping.get("disposition_map", {}).get(key)
    try:
        return DispositionClass(target) if target else DispositionClass.OTHER
    except ValueError:
        return DispositionClass.OTHER


def parse_transcript(text: str, mapping: dict[str, Any]) -> list[tuple[Speaker, str]]:
    """Split a single-string transcript into (speaker, text) turns.

    Lines without a recognised prefix are appended to the previous turn, because
    exports often wrap long utterances.
    """
    cfg = mapping.get("transcript", {})
    agent = [p.lower() for p in cfg.get("agent_prefixes", [])]
    seller = [p.lower() for p in cfg.get("seller_prefixes", [])]
    prefix_re = re.compile(r"^\s*\[?([A-Za-z_ ]{1,20})\]?\s*[:\-]\s*(.*)$")
    turns: list[tuple[Speaker, str]] = []
    for line in str(text).splitlines():
        if not line.strip():
            continue
        m = prefix_re.match(line)
        label = m.group(1).strip().lower() if m else None
        if label in agent:
            turns.append((Speaker.AGENT, m.group(2).strip()))
        elif label in seller:
            turns.append((Speaker.SELLER, m.group(2).strip()))
        elif turns:
            spk, prev = turns[-1]
            turns[-1] = (spk, f"{prev} {line.strip()}")
        else:
            turns.append((Speaker.UNKNOWN, line.strip()))
    return turns


def _speaker(val: Any, mapping: dict[str, Any]) -> Speaker:
    s = (_str(val) or "").lower()
    cfg = mapping.get("transcript", {})
    if s in [p.lower() for p in cfg.get("agent_prefixes", [])]:
        return Speaker.AGENT
    if s in [p.lower() for p in cfg.get("seller_prefixes", [])]:
        return Speaker.SELLER
    return Speaker.UNKNOWN


def _call_level(row: pd.Series, mapping: dict[str, Any]) -> tuple[CallContext, Outcome]:
    c = mapping.get("context", {})
    o = mapping.get("outcome", {})
    ctx = CallContext(
        campaign_id=_str(_get(row, c.get("campaign_id"))),
        call_objective=_str(_get(row, c.get("call_objective"))),
        lead_type=_str(_get(row, c.get("lead_type"))),
        seller_category=_str(_get(row, c.get("seller_category"))),
        language_hint=_str(_get(row, c.get("language_hint"))),
        started_at=_to_dt(_get(row, c.get("started_at"))),
        agent_version=_str(_get(row, c.get("agent_version"))),
        script_version=_str(_get(row, c.get("script_version"))),
        region=_str(_get(row, c.get("region"))),
    )
    raw_disp = _str(_get(row, o.get("disposition_raw")))
    disp_class = classify_disposition(raw_disp, mapping)
    achieved = _to_bool(_get(row, o.get("objective_achieved")))
    if achieved is None and raw_disp is not None:
        achieved = disp_class.value in mapping.get("objective_achieved_classes", [])
    out = Outcome(
        disposition_raw=raw_disp,
        disposition_class=disp_class,
        objective_achieved=achieved,
        downstream_conversion=_to_bool(_get(row, o.get("downstream_conversion"))),
        termination_reason=_str(_get(row, o.get("termination_reason"))),
    )
    return ctx, out


def _finalize(call_id: str, seller_raw: Any, ctx: CallContext, out: Outcome, turns: list[Turn]) -> CallRecord:
    confs = [t.asr_confidence for t in turns if t.asr_confidence is not None]
    redactions = 0
    clean: list[Turn] = []
    for t in turns:
        text, n = redact(t.text)
        redactions += n
        clean.append(t.model_copy(update={"text": text}))
    quality = DataQuality(
        has_timestamps=any(t.start_ms is not None and t.end_ms is not None for t in clean),
        has_asr_confidence=bool(confs),
        mean_asr_confidence=sum(confs) / len(confs) if confs else None,
        redactions_applied=redactions,
        unknown_speaker_turns=sum(t.speaker == Speaker.UNKNOWN for t in clean),
    )
    return CallRecord(
        call_id=str(call_id),
        seller_key=pseudonymize(_str(seller_raw)),
        context=ctx,
        turns=clean,
        outcome=out,
        quality=quality,
    )


def iter_calls(df: pd.DataFrame, mapping: dict[str, Any]) -> Iterator[CallRecord]:
    layout = mapping.get("layout", "wide")
    call_col = mapping["call_id"]
    seller_col = mapping.get("seller_id")
    if call_col not in df.columns:
        raise KeyError(
            f"call_id column {call_col!r} not in export columns {list(df.columns)}; edit config/column_mapping.yaml"
        )

    if layout == "wide":
        tcol = mapping["transcript"]["column"]
        for _, row in df.iterrows():
            ctx, out = _call_level(row, mapping)
            parsed = parse_transcript(_get(row, tcol) or "", mapping)
            turns = [Turn(idx=i, speaker=s, text=t) for i, (s, t) in enumerate(parsed)]
            yield _finalize(row[call_col], _get(row, seller_col), ctx, out, turns)
        return

    if layout != "long":
        raise ValueError(f"Unknown layout {layout!r}")
    tm = mapping["turn"]
    for call_id, grp in df.groupby(call_col, sort=False):
        order = tm.get("order_by")
        if order and order in grp.columns:
            grp = grp.assign(__o=pd.to_numeric(grp[order], errors="coerce")).sort_values("__o")
        ctx, out = _call_level(grp.iloc[0], mapping)
        turns = []
        for i, (_, r) in enumerate(grp.iterrows()):
            turns.append(
                Turn(
                    idx=i,
                    speaker=_speaker(_get(r, tm.get("speaker")), mapping),
                    text=_str(_get(r, tm.get("text"))) or "",
                    start_ms=_to_int(_get(r, tm.get("start_ms"))),
                    end_ms=_to_int(_get(r, tm.get("end_ms"))),
                    asr_confidence=_to_float(_get(r, tm.get("asr_confidence"))),
                    interrupted=_to_bool(_get(r, tm.get("interrupted"))),
                )
            )
        yield _finalize(call_id, _get(grp.iloc[0], seller_col), ctx, out, turns)


def load_calls(path: Path, mapping: dict[str, Any] | None = None) -> list[CallRecord]:
    mapping = mapping or load_mapping()
    return list(iter_calls(read_raw(path), mapping))

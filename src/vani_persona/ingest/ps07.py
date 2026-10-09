"""Loader for the PS07 "Seller-Fit Adaptive Voice Persona" export (multi-table, July–Sept 2026).

Joins:
  ps07_call_turns.csv             turn-level transcript with timestamps -> Turn
  ps07_bot_calls_*.csv            call-level: bot version, lead bucket, start time, disposition, meeting_fixed
  ps07_evaluator_outputs_*.csv    'disposition' (detailed outcome) and 'incorrect_mf' (false meeting) evaluators
  ps07_seller_files_part*.csv     seller category group + state, used ONLY as confounders / proxy checks

The files are read in place from wherever they live. Nothing is copied. Raw seller IDs
are pseudonymized, and transcript text is redacted by the shared adapter (`_finalize`).
NOTE: the export is not masked. Seller/company *names* in transcripts are not removed
by the regex redactor (open item: NER pass). Reports must stay aggregate-only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from vani_persona.canonical import CallContext, CallRecord, DispositionClass, Outcome, Speaker, Turn
from vani_persona.ingest.adapter import _finalize

log = logging.getLogger(__name__)

CALL_OBJECTIVE = "fix_meeting"

LABEL_TO_CLASS = {
    "Meeting Fixed": DispositionClass.OBJECTIVE_COMPLETED,
    "Not Interested": DispositionClass.NOT_INTERESTED,
    "Call Later / Busy": DispositionClass.CALLBACK_REQUESTED,
    "General (talked)": DispositionClass.OTHER,
}
EVAL_TO_CLASS = {
    "meeting_fixed": DispositionClass.OBJECTIVE_COMPLETED,
    "online_meeting_fixed": DispositionClass.OBJECTIVE_COMPLETED,
    "callback_requested": DispositionClass.CALLBACK_REQUESTED,
    "not_interested": DispositionClass.NOT_INTERESTED,
    "budget_issue_denial": DispositionClass.NOT_INTERESTED,
    "do_not_call": DispositionClass.OPT_OUT,
    "call_dropped": DispositionClass.SELLER_HUNG_UP,
    "wrong_number": DispositionClass.WRONG_PARTY,
    "alternate_number_or_not_dm": DispositionClass.WRONG_PARTY,
    "already_in_touch_with_im": DispositionClass.OTHER,
    "info_requested": DispositionClass.OTHER,
    "other": DispositionClass.OTHER,
}
NO_RESPONSE_SUBS = {"no_response", "voicemail"}
SPEAKERS = {"bot": Speaker.AGENT, "seller": Speaker.SELLER}


def _find(root: Path, pattern: str) -> list[Path]:
    files = sorted(root.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} not found under {root}")
    return files


def _read(files: list[Path], **kw) -> pd.DataFrame:
    return pd.concat([pd.read_csv(f, low_memory=False, **kw) for f in files], ignore_index=True)


def _evaluators(root: Path, ids: set[int]) -> pd.DataFrame:
    """One row per attempt_id with the detailed disposition and the incorrect-meeting flag."""
    keep = []
    for f in _find(root, "ps07_evaluator_outputs_*.csv"):
        for chunk in pd.read_csv(f, chunksize=50_000, low_memory=False):
            keep.append(chunk[chunk.attempt_id.isin(ids) & chunk.eval_agent.isin(["disposition", "incorrect_mf"])])
    ev = pd.concat(keep, ignore_index=True).drop_duplicates(["attempt_id", "eval_agent"], keep="last")
    out = pd.DataFrame(index=sorted(ids))
    disp = ev[ev.eval_agent == "disposition"].set_index("attempt_id").agent_output.map(json.loads)
    out["eval_disposition"] = disp.map(lambda d: d.get("disposition"))
    out["eval_sub_disposition"] = disp.map(lambda d: d.get("sub_disposition"))
    mf = ev[ev.eval_agent == "incorrect_mf"].set_index("attempt_id").agent_output.map(json.loads)
    out["incorrect_mf"] = mf.map(lambda d: d.get("incorrect_mf"))
    return out


def load_ps07(root: Path, with_evaluators: bool = True) -> list[CallRecord]:
    turns = _read(_find(root, "ps07_call_turns.csv"))
    ids = set(turns.attempt_id.unique())
    calls = _read(
        _find(root, "ps07_bot_calls_*.csv"),
        usecols=["attempt_id", "fk_glusr_usr_id", "redis_bucket", "lead_bot_version", "call_attempt_count",
                 "call_start_time", "disposition_label", "meeting_fixed"],
    )
    calls = calls[calls.attempt_id.isin(ids)].set_index("attempt_id")
    sellers = _read(_find(root, "ps07_seller_files_part*.csv"), usecols=["fk_glusr_usr_id", "top_category_group", "seller_state"])
    sellers = sellers.drop_duplicates("fk_glusr_usr_id").set_index("fk_glusr_usr_id")
    ev = _evaluators(root, ids) if with_evaluators else pd.DataFrame(index=sorted(ids))

    missing = len(ids) - len(calls)
    if missing:
        log.warning("%d transcript calls have no bot_calls row; they are skipped", missing)

    records = []
    for attempt_id, grp in turns.sort_values(["attempt_id", "turn_no"]).groupby("attempt_id", sort=False):
        if attempt_id not in calls.index:
            continue
        c = calls.loc[attempt_id]
        seller_id = int(c.fk_glusr_usr_id)
        s = sellers.loc[seller_id] if seller_id in sellers.index else None
        started = pd.to_datetime(c.call_start_time, errors="coerce")
        if pd.notna(started) and started.tzinfo is None:
            started = started.tz_localize("Asia/Kolkata")  # export times are IST
        ctx = CallContext(
            campaign_id=str(c.lead_bot_version),
            call_objective=CALL_OBJECTIVE,
            lead_type=str(c.redis_bucket) if pd.notna(c.redis_bucket) else None,
            seller_category=str(s.top_category_group) if s is not None and pd.notna(s.top_category_group) else None,
            region=str(s.seller_state) if s is not None and pd.notna(s.seller_state) else None,
            started_at=None if pd.isna(started) else started.to_pydatetime(),
            agent_version=str(c.lead_bot_version),
        )
        e = ev.loc[attempt_id] if attempt_id in ev.index else None
        eval_disp = e.eval_disposition if e is not None and pd.notna(e.eval_disposition) else None
        sub = e.eval_sub_disposition if e is not None and pd.notna(e.eval_sub_disposition) else None
        if sub in NO_RESPONSE_SUBS:
            dclass = DispositionClass.NO_RESPONSE
        elif eval_disp in EVAL_TO_CLASS:
            dclass = EVAL_TO_CLASS[eval_disp]
        else:
            dclass = LABEL_TO_CLASS.get(c.disposition_label, DispositionClass.OTHER)
        incorrect = bool(e.incorrect_mf) if e is not None and pd.notna(e.incorrect_mf) else False
        out = Outcome(
            disposition_raw=f"{c.disposition_label}|{eval_disp or '-'}|{sub or '-'}",
            disposition_class=dclass,
            # The business label, minus meetings the incorrect_mf evaluator found to be wrong.
            objective_achieved=bool(int(c.meeting_fixed)) and not incorrect,
        )
        tlist = [
            Turn(
                idx=i,
                speaker=SPEAKERS.get(str(r.speaker), Speaker.UNKNOWN),
                text="" if pd.isna(r.text) else str(r.text),
                start_ms=None if pd.isna(r.start_sec) else int(round(r.start_sec * 1000)),
                end_ms=None if pd.isna(r.end_sec) else int(round(r.end_sec * 1000)),
            )
            for i, r in enumerate(grp.itertuples(index=False))
        ]
        records.append(_finalize(str(attempt_id), str(seller_id), ctx, out, tlist))
    return records

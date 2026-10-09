"""Transcript-only behavioural features.

The same ``extract_features`` function runs offline (full calls) and live (call
prefixes), so training and serving cannot drift apart. It is pure Python and
linear in the number of turns. Recomputing it on every seller turn of a phone
call costs well under a millisecond.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from vani_persona.canonical import Speaker, Turn
from vani_persona.features import lexicons as lx
from vani_persona.features.registry import FEATURES

_PUNCT_RE = re.compile(r"[^\w\s\u0900-\u0DFF?]", re.UNICODE)  # keep combining marks of all Indic scripts
_SPACE_RE = re.compile(r"\s+")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_REDACTION_RE = re.compile(r"<[A-Z]+>")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _REDACTION_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text.lower())
    return _SPACE_RE.sub(" ", text).strip()


def _compile(phrases: Sequence[str]) -> re.Pattern[str]:
    # Python's \b is unicode-aware but fails around Devanagari combining marks, so
    # use explicit "not a word character" lookarounds instead.
    alts = sorted({re.escape(normalize(p)) for p in phrases if normalize(p)}, key=len, reverse=True)
    return re.compile(r"(?<![\wऀ-ॿ])(?:" + "|".join(alts) + r")(?![\wऀ-ॿ])")


LEXICON_PATTERNS: dict[str, re.Pattern[str]] = {
    "clarification": _compile(lx.CLARIFICATION),
    "brevity_request": _compile(lx.BREVITY_REQUEST),
    "time_pressure": _compile(lx.TIME_PRESSURE),
    "objection_price": _compile(lx.OBJECTION_PRICE),
    "objection_trust": _compile(lx.OBJECTION_TRUST),
    "objection_not_interested": _compile(lx.OBJECTION_NOT_INTERESTED),
    "objection_already_have": _compile(lx.OBJECTION_ALREADY_HAVE),
    "info_request": _compile(lx.INFO_REQUEST),
    "opt_out": _compile(lx.OPT_OUT),
    "affirmation": _compile(lx.AFFIRMATION),
    "question_word": _compile(lx.QUESTION_WORDS),
}
LANGUAGE_REQUEST_PATTERNS = {lang: _compile(p) for lang, p in lx.LANGUAGE_REQUEST.items()}

OBJECTION_TAGS = frozenset({"objection_price", "objection_trust", "objection_not_interested", "objection_already_have"})
RUSH_TAGS = frozenset({"time_pressure", "brevity_request"})

RATE_FEATURES = {
    "clarification_rate": "clarification",
    "brevity_request_rate": "brevity_request",
    "time_pressure_rate": "time_pressure",
    "objection_price_rate": "objection_price",
    "objection_trust_rate": "objection_trust",
    "objection_not_interested_rate": "objection_not_interested",
    "objection_already_have_rate": "objection_already_have",
    "info_request_rate": "info_request",
    "affirmation_rate": "affirmation",
}


@dataclass(frozen=True)
class TurnSignals:
    """Per-turn lexical signals. Live components use them for explicit-signal handling."""

    tokens: int
    tags: frozenset[str]
    is_question: bool
    language_request: str | None
    hindi_fw: int
    english_fw: int
    devanagari_chars: int
    latin_chars: int


def turn_signals(text: str) -> TurnSignals:
    norm = normalize(text)
    toks = norm.replace("?", " ").split()
    tags = frozenset(name for name, pat in LEXICON_PATTERNS.items() if pat.search(norm))
    lang_req = next((lang for lang, pat in LANGUAGE_REQUEST_PATTERNS.items() if pat.search(norm)), None)
    hindi_fw = english_fw = 0
    for t in toks:
        if _DEVANAGARI_RE.search(t):
            hindi_fw += 1
        elif t in lx.AMBIGUOUS_FUNCTION_WORDS:
            continue
        elif t in lx.HINDI_FUNCTION_WORDS:
            hindi_fw += 1
        elif t in lx.ENGLISH_FUNCTION_WORDS:
            english_fw += 1
    return TurnSignals(
        tokens=len(toks),
        tags=tags,
        is_question="?" in text or "question_word" in tags,
        language_request=lang_req,
        hindi_fw=hindi_fw,
        english_fw=english_fw,
        devanagari_chars=len(_DEVANAGARI_RE.findall(text)),
        latin_chars=len(_LATIN_RE.findall(text)),
    )


def _slope(ys: list[float]) -> float | None:
    n = len(ys)
    if n < 3:
        return None
    xbar = (n - 1) / 2
    ybar = sum(ys) / n
    den = sum((i - xbar) ** 2 for i in range(n))
    return sum((i - xbar) * (y - ybar) for i, y in enumerate(ys)) / den


def _timing(turns: Sequence[Turn]) -> tuple[float | None, float | None]:
    """Return (median response latency ms, overlap rate) from timestamps or interruption flags."""
    gaps: list[float] = []
    overlaps = 0
    seller_after_agent = 0
    for prev, cur in zip(turns, turns[1:]):
        if prev.speaker != Speaker.AGENT or cur.speaker != Speaker.SELLER:
            continue
        if prev.end_ms is None or cur.start_ms is None:
            continue
        seller_after_agent += 1
        gap = cur.start_ms - prev.end_ms
        if gap < 0:
            overlaps += 1
        gaps.append(max(gap, 0))
    latency = statistics.median(gaps) if gaps else None
    if seller_after_agent:
        return latency, overlaps / seller_after_agent
    flags = [t.interrupted for t in turns if t.speaker == Speaker.AGENT and t.interrupted is not None]
    return latency, (sum(flags) / len(flags) if flags else None)


def extract_features(turns: Sequence[Turn]) -> dict[str, float | None]:
    """Compute every registered feature over the given turns (a full call or a live prefix).

    Features that cannot be computed (no seller turns, no timestamps) are None and never 0,
    because "not observed" is different from "observed to be zero".
    """
    seller = [t for t in turns if t.speaker == Speaker.SELLER and t.text.strip()]
    sigs = [turn_signals(t.text) for t in seller]
    n = len(sigs)
    out: dict[str, float | None] = {name: None for name in FEATURES}
    out["n_seller_turns"] = float(n)

    starts = [t.start_ms for t in turns if t.start_ms is not None]
    ends = [t.end_ms for t in turns if t.end_ms is not None]
    if starts and ends:
        out["duration_s"] = (max(ends) - min(starts)) / 1000.0
    out["response_latency_median_ms"], out["overlap_rate"] = _timing(turns)

    if n == 0:
        return out

    lens = [s.tokens for s in sigs]
    out["seller_mean_tokens"] = sum(lens) / n
    out["short_reply_ratio"] = sum(x <= 3 for x in lens) / n
    out["long_reply_ratio"] = sum(x >= 15 for x in lens) / n
    out["question_rate"] = sum(s.is_question for s in sigs) / n
    for feat, tag in RATE_FEATURES.items():
        out[feat] = sum(tag in s.tags for s in sigs) / n
    out["engagement_slope"] = _slope([float(x) for x in lens])
    out["log_mean_tokens"] = math.log1p(out["seller_mean_tokens"])
    out["resistance_rate"] = sum(bool(s.tags & OBJECTION_TAGS) for s in sigs) / n
    out["rush_rate"] = sum(bool(s.tags & RUSH_TAGS) for s in sigs) / n

    dev = sum(s.devanagari_chars for s in sigs)
    lat = sum(s.latin_chars for s in sigs)
    out["devanagari_ratio"] = dev / (dev + lat) if dev + lat else None
    hi = sum(s.hindi_fw for s in sigs)
    en = sum(s.english_fw for s in sigs)
    out["hindi_function_ratio"] = hi / (hi + en) if hi + en else None
    out["code_switch_rate"] = sum(s.hindi_fw > 0 and s.english_fw > 0 for s in sigs) / n
    return out

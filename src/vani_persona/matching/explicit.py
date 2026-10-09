"""Explicit seller requests: direct evidence that needs no classifier and no gating.

A seller saying "Hindi mein bolo" or "jaldi bolo" is not something to infer. It
is applied as an immediate style override on top of whatever persona is active.
Opt-out is passed to the policy layer and is never treated as a style signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vani_persona.features.transcript import TurnSignals


@dataclass(frozen=True)
class ExplicitSignal:
    kind: str  # language_request | brevity_request | repeated_clarification | opt_out
    value: str | None = None
    source: str = "direct_user_statement"


@dataclass
class StyleOverrides:
    """Overrides that take precedence over the persona style for the rest of the call."""

    language: str | None = None
    max_sentences_per_turn: int | None = None
    density: str | None = None
    tts_rate: float | None = None
    rephrase_simpler: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, False, [])}


def detect(last: TurnSignals | None, recent: list[TurnSignals]) -> list[ExplicitSignal]:
    if last is None:
        return []
    out = []
    if "opt_out" in last.tags:
        out.append(ExplicitSignal("opt_out"))
    if last.language_request:
        out.append(ExplicitSignal("language_request", last.language_request))
    if "brevity_request" in last.tags:
        out.append(ExplicitSignal("brevity_request"))
    # A single "kya?" is often line noise; two clarification requests within the
    # last three seller turns is a pattern.
    if "clarification" in last.tags and sum("clarification" in s.tags for s in recent) >= 2:
        out.append(ExplicitSignal("repeated_clarification", source="observed_pattern"))
    return out


def apply(signals: list[ExplicitSignal], overrides: StyleOverrides) -> StyleOverrides:
    for s in signals:
        if s.kind == "language_request":
            overrides.language = s.value
            overrides.reasons.append(f"seller asked for language={s.value}")
        elif s.kind == "brevity_request":
            overrides.max_sentences_per_turn = 1
            overrides.density = "minimal"
            overrides.reasons.append("seller asked VANI to be brief")
        elif s.kind == "repeated_clarification":
            overrides.tts_rate = min(overrides.tts_rate or 1.0, 0.95)
            overrides.rephrase_simpler = True
            overrides.reasons.append("seller asked for clarification repeatedly")
    return overrides

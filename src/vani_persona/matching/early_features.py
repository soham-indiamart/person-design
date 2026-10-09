"""Live per-call feature state, updated on every finalized turn.

Only finalized ASR turns are processed; partial hypotheses are ignored. A turn
whose ASR confidence is below ``min_asr_confidence`` still goes to the dialogue
manager, but it is left out of behavioural evidence and recorded as uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vani_persona.canonical import Speaker, Turn
from vani_persona.features.transcript import TurnSignals, extract_features, turn_signals


@dataclass
class LiveFeatureState:
    min_asr_confidence: float = 0.6
    turns: list[Turn] = field(default_factory=list)
    low_confidence_turns: int = 0
    silent_turns: int = 0
    last_signals: TurnSignals | None = None
    recent_signals: list[TurnSignals] = field(default_factory=list)

    def add_turn(self, turn: Turn) -> bool:
        """Register a finalized turn. Returns False if it yields no behavioural evidence."""
        if turn.speaker == Speaker.SELLER:
            if not turn.text.strip():
                self.silent_turns += 1
                self.last_signals = None
                return False
            if turn.asr_confidence is not None and turn.asr_confidence < self.min_asr_confidence:
                self.low_confidence_turns += 1
                self.last_signals = None
                return False
            self.last_signals = turn_signals(turn.text)
            self.recent_signals = (self.recent_signals + [self.last_signals])[-3:]
        self.turns.append(turn)
        return True

    def features(self, recent_seller_turns: int | None = None) -> dict[str, float | None]:
        """Features over the whole prefix, or over the window that starts at the K-th most recent seller turn.

        Whole-prefix features suit the initial decision. Mid-call switching uses a
        recent window, because cumulative averages react too slowly when a seller's
        behaviour changes. Recomputing is O(turns), and phone calls have tens of turns.
        """
        turns = self.turns
        if recent_seller_turns is not None:
            seller_idx = [i for i, t in enumerate(turns) if t.speaker == Speaker.SELLER]
            if len(seller_idx) > recent_seller_turns:
                start = seller_idx[-recent_seller_turns]
                # Keep the agent turn just before the window so latency/overlap stay computable.
                turns = turns[max(start - 1, 0):]
        return extract_features(turns)

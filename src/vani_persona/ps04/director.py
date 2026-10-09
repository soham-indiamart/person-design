"""Persona Director: turns Behaviour Analyst verdicts into persona switches.

Deterministic, so every switch can be explained in the log:

- **Clear signal** (intensity >= 0.7) for a state whose persona is not active -> switch now.
- **Aggression** switches at >= 0.6, or on any aggression phrase (safety first).
- **Weak signal** (0.5 to the threshold) must repeat on 2 consecutive seller turns, unless
  the analyst and the phrase lexicon agree on it (two independent detectors) -> switch now.
- **Recovery**: 2 consecutive calm turns (trend calming or steady) -> back to the *home*
  persona the live seller profile picks (Quick & Direct for an impatient seller, else
  Friendly Guide). Between those two resting personas the profile switches at once.
- **Anti ping-pong**: switching straight back to the persona left on the previous turn
  needs intensity >= 0.8.
- **Escalation**: a 3rd consecutive aggressive turn while already de-escalating ->
  apologise, offer a callback and end the call.
"""

from __future__ import annotations

from dataclasses import dataclass

from vani_persona.ps04.analyst import Verdict
from vani_persona.ps04.live_personas import PERSONAS, STATE_TO_PERSONA

CLEAR = 0.7
AGGRESSION_CLEAR = 0.6
WEAK = 0.5
PING_PONG = 0.8
CALM_TURNS_TO_RECOVER = 2
AGGRESSIVE_TURNS_TO_END = 3


@dataclass
class Decision:
    action: str            # switch | stay | pending | escalate
    from_persona: str
    to_persona: str
    reason: str

    @property
    def switched(self) -> bool:
        return self.from_persona != self.to_persona


class PersonaDirector:
    def __init__(self, start: str):
        self.current = start
        self.previous: str | None = None      # persona active before the last switch
        self.turns_since_switch = 0
        self.pending: tuple[str, int] | None = None
        self.calm_streak = 0
        self.aggressive_streak = 0

    def _switch(self, to: str, reason: str) -> Decision:
        d = Decision("switch", self.current, to, reason)
        self.previous, self.current = self.current, to
        self.turns_since_switch = 0
        self.pending = None
        return d

    def decide(self, v: Verdict, home: str = "normal", home_reason: str = "") -> Decision:
        """``home``: the persona the live seller profile says to rest in while the seller is calm."""
        self.turns_since_switch += 1
        cur = self.current
        self.aggressive_streak = self.aggressive_streak + 1 if v.state == "aggressive" else 0

        if v.state == "calm":
            self.pending = None
            self.calm_streak = self.calm_streak + 1 if v.trend in ("calming", "steady") else 0
            if cur != home and cur in ("normal", "quick_direct"):
                # Between the two resting personas the profile decides at once.
                self.calm_streak = 0
                return self._switch(home, f"Live profile: {home_reason or 'seller profile changed'}; "
                                          f"switch to {PERSONAS[home].name}.")
            if cur != home and self.calm_streak >= CALM_TURNS_TO_RECOVER:
                self.calm_streak = 0
                return self._switch(home, f"Seller calm for {CALM_TURNS_TO_RECOVER} turns ({v.trend}): "
                                          f"back to {PERSONAS[home].name}.")
            return Decision("stay", cur, cur, f"Seller calm ({v.trend}); keep {PERSONAS[cur].name}.")
        self.calm_streak = 0

        target = STATE_TO_PERSONA[v.state]
        name = PERSONAS[target].name
        if target == cur:
            self.pending = None
            if v.state == "aggressive" and cur == "de_escalator" and self.aggressive_streak >= AGGRESSIVE_TURNS_TO_END:
                return Decision("escalate", cur, cur,
                                f"Still aggressive after {self.aggressive_streak} turns: escalation rule -> "
                                "apologise, offer a callback and end the call.")
            return Decision("stay", cur, cur, f"Seller still {v.state}; {name} already active.")

        urgent = v.state == "aggressive" and (v.intensity >= AGGRESSION_CLEAR or "aggressive" in v.lexicon)
        agreed = v.source == "analyst+lexicon" and v.intensity >= WEAK
        ping_pong = target == self.previous and self.turns_since_switch <= 1 and not urgent
        bar = PING_PONG if ping_pong else CLEAR
        if urgent or v.intensity >= bar or (agreed and not ping_pong):
            why = ("aggression phrase" if (urgent and v.intensity < AGGRESSION_CLEAR)
                   else f"analyst + phrase '{v.lexicon.get(v.state)}' agree" if v.intensity < bar
                   else f"intensity {v.intensity:.2f}")
            return self._switch(target, f"Seller is {v.state} ({why}, {v.trend}): switch to {name}.")
        if v.intensity >= WEAK:
            n = self.pending[1] + 1 if self.pending and self.pending[0] == target else 1
            self.pending = (target, n)
            if n >= 2:
                return self._switch(target, f"Seller {v.state} on {n} consecutive turns (intensity "
                                            f"{v.intensity:.2f}): switch to {name}.")
            return Decision("pending", cur, cur, f"Weak {v.state} signal ({v.intensity:.2f}); "
                                                 f"switch to {name} if it repeats.")
        self.pending = None
        return Decision("stay", cur, cur, f"{v.state} too faint ({v.intensity:.2f}); keep {PERSONAS[cur].name}.")

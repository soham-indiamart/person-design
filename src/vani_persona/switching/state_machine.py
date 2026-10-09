"""Persona switching state machine.

States:
  NEUTRAL    - neutral_default active; waiting for enough evidence
  COMMITTED  - a persona is active
A challenger streak is tracked in both states. A change happens only after the
challenger has won ``confirm_evaluations`` consecutive evaluations and every
gate passes (threshold, margin, dwell, cooldown, switch budget). Explicit seller
requests do not go through this machine: they become immediate StyleOverrides
(see ``matching.explicit``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum

from vani_persona.matching.matcher import MatchResult
from vani_persona.personas import NEUTRAL_ID


class Phase(str, Enum):
    NEUTRAL = "NEUTRAL"
    COMMITTED = "COMMITTED"


class Action(str, Enum):
    RETAIN = "retain"
    COMMIT = "commit"
    SWITCH = "switch"
    REVERT_TO_NEUTRAL = "revert_to_neutral"


@dataclass
class SwitchConfig:
    commit_threshold: float = 0.60
    switch_threshold: float = 0.65
    switch_margin: float = 0.15
    confirm_evaluations: int = 2
    min_seller_turns_before_commit: int = 2
    min_dwell_seller_turns: int = 2
    switch_back_cooldown_turns: int = 4
    max_switches_per_call: int = 2
    revert_floor: float = 0.35
    revert_evaluations: int = 3

    @classmethod
    def from_thresholds(cls, th: dict) -> "SwitchConfig":
        s, m = th["switching"], th["matching"]
        return cls(
            commit_threshold=m["commit_threshold"],
            min_seller_turns_before_commit=m["min_seller_turns_before_commit"],
            **{k: s[k] for k in (
                "switch_threshold", "switch_margin", "confirm_evaluations", "min_dwell_seller_turns",
                "switch_back_cooldown_turns", "max_switches_per_call", "revert_floor", "revert_evaluations",
            )},
        )

    def with_overrides(self, overrides: dict[str, float]) -> "SwitchConfig":
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown switching overrides: {sorted(unknown)}")
        return replace(self, **overrides)


@dataclass
class SwitchDecision:
    turn_index: int
    seller_turn: int
    action: Action
    from_persona: str
    to_persona: str
    phase: Phase
    reason: str
    challenger: str | None
    challenger_streak: int
    scores: dict[str, float]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["action"], d["phase"] = self.action.value, self.phase.value
        return d


@dataclass
class SwitchState:
    phase: Phase = Phase.NEUTRAL
    active: str = NEUTRAL_ID
    previous: str | None = None
    seller_turn_of_last_change: int = 0
    switches: int = 0
    challenger: str | None = None
    streak: int = 0
    low_streak: int = 0
    history: list[SwitchDecision] = field(default_factory=list)


class PersonaSwitcher:
    """``persona_overrides`` maps persona_id -> threshold overrides (persona YAML
    ``switching.threshold_overrides``). They apply while that persona is active, e.g.
    a persona that is costly to leave can demand a larger margin."""

    def __init__(self, cfg: SwitchConfig, persona_overrides: dict[str, dict[str, float]] | None = None):
        self.base_cfg = cfg
        self._per_persona = {pid: cfg.with_overrides(o) for pid, o in (persona_overrides or {}).items() if o}
        self.state = SwitchState()

    @property
    def cfg(self) -> SwitchConfig:
        return self._per_persona.get(self.state.active, self.base_cfg)

    def _track_challenger(self, best: str) -> int:
        st = self.state
        st.streak = st.streak + 1 if st.challenger == best else 1
        st.challenger = best
        return st.streak

    def _reset_challenger(self) -> None:
        self.state.challenger, self.state.streak = None, 0

    def _change(self, to: str, seller_turn: int, counts_as_switch: bool) -> None:
        st = self.state
        st.previous, st.active = st.active, to
        st.phase = Phase.NEUTRAL if to == NEUTRAL_ID else Phase.COMMITTED
        st.seller_turn_of_last_change = seller_turn
        st.switches += int(counts_as_switch)
        st.low_streak = 0
        self._reset_challenger()

    def update(self, m: MatchResult, seller_turn: int) -> SwitchDecision:
        """Consume one matcher result (after a seller turn) and decide whether to change persona."""
        cfg, st = self.cfg, self.state
        before = st.active
        scores = m.scores
        candidates = {k: v for k, v in scores.items() if k != NEUTRAL_ID}
        best = max(candidates, key=candidates.get) if candidates else None
        p_best = candidates.get(best, 0.0) if best else 0.0
        action, reason = Action.RETAIN, ""

        if not scores:
            reason = f"matcher unavailable ({m.reason}); keeping {st.active}"
            self._reset_challenger()
        elif st.phase == Phase.NEUTRAL:
            if seller_turn < cfg.min_seller_turns_before_commit:
                reason = f"collecting evidence ({seller_turn}/{cfg.min_seller_turns_before_commit} seller turns)"
            elif best is None or p_best < cfg.commit_threshold:
                reason = f"no persona above commit threshold ({best} {p_best:.2f} < {cfg.commit_threshold})"
                self._reset_challenger()
            elif st.switches >= cfg.max_switches_per_call:
                reason = "switch budget exhausted; staying neutral"
            elif best == st.previous and seller_turn - st.seller_turn_of_last_change < cfg.switch_back_cooldown_turns:
                reason = f"cooldown: {best} was just left"
            else:
                streak = self._track_challenger(best)
                if streak >= cfg.confirm_evaluations:
                    action, reason = Action.COMMIT, f"{best} {p_best:.2f} >= {cfg.commit_threshold} for {streak} evaluations"
                    # The first commit from neutral does not use up the switch budget.
                    self._change(best, seller_turn, counts_as_switch=st.previous is not None)
                else:
                    reason = f"{best} leading ({streak}/{cfg.confirm_evaluations} confirmations)"
        else:
            p_cur = scores.get(st.active, 0.0)
            dwell = seller_turn - st.seller_turn_of_last_change
            st.low_streak = st.low_streak + 1 if p_cur < cfg.revert_floor else 0
            challenger_ok = (
                best is not None
                and best != st.active
                and p_best >= cfg.switch_threshold
                and p_best - p_cur >= cfg.switch_margin
            )
            if challenger_ok:
                if dwell < cfg.min_dwell_seller_turns:
                    reason = f"dwell: {dwell}/{cfg.min_dwell_seller_turns} seller turns on {st.active}"
                    self._track_challenger(best)
                elif st.switches >= cfg.max_switches_per_call:
                    reason = "switch budget exhausted"
                elif best == st.previous and dwell < cfg.switch_back_cooldown_turns:
                    reason = f"cooldown: switching back to {best} blocked"
                else:
                    streak = self._track_challenger(best)
                    if streak >= cfg.confirm_evaluations:
                        action = Action.SWITCH
                        reason = (
                            f"{best} {p_best:.2f} beats {st.active} {p_cur:.2f} by >= {cfg.switch_margin} "
                            f"for {streak} evaluations"
                        )
                        self._change(best, seller_turn, counts_as_switch=True)
                    else:
                        reason = f"challenger {best} ({streak}/{cfg.confirm_evaluations} confirmations)"
            elif st.low_streak >= cfg.revert_evaluations and st.switches < cfg.max_switches_per_call:
                action = Action.REVERT_TO_NEUTRAL
                reason = f"{st.active} below {cfg.revert_floor} for {st.low_streak} evaluations and no clear challenger"
                self._change(NEUTRAL_ID, seller_turn, counts_as_switch=True)
            else:
                reason = f"{st.active} retained (p={p_cur:.2f})"
                self._reset_challenger()

        decision = SwitchDecision(
            turn_index=m.turn_index,
            seller_turn=seller_turn,
            action=action,
            from_persona=before,
            to_persona=st.active,
            phase=st.phase,
            reason=reason,
            challenger=st.challenger,
            challenger_streak=st.streak,
            scores=scores,
        )
        st.history.append(decision)
        return decision

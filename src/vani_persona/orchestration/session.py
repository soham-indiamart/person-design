"""Per-call runtime: wires feature state, matcher, switcher, overrides, composer and events.

Modes:
  control    - today's VANI: neutral style only; matcher not run.
  shadow     - matcher and switcher run and are logged, but neutral style is always applied.
  treatment  - matcher decisions are applied.

A finalized seller turn is processed in two phases:
  ingest_seller_turn  - inline, sub-millisecond: features, explicit signals, opt-out.
                        A direct request shapes the very next reply.
  evaluate_persona    - off the response path: matcher + switching state machine.
``compose_next`` reads only already-decided state. If evaluation has not finished
when the next reply starts, that reply uses the previous persona: the style
arrives one turn late, but the reply is never delayed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vani_persona.canonical import Speaker, Turn
from vani_persona.events import EventLog
from vani_persona.matching import explicit
from vani_persona.matching.early_features import LiveFeatureState
from vani_persona.matching.matcher import MatchRequest, MatchResult, PersonaMatcher
from vani_persona.orchestration.composer import ComposedPrompt, compose
from vani_persona.orchestration.dialogue_state import DialogueState, ObjectivePolicy, Stage
from vani_persona.personas import NEUTRAL_ID, PersonaRegistry
from vani_persona.switching.state_machine import Action, PersonaSwitcher, SwitchConfig, SwitchDecision

OBJECTION_TAGS = {
    "objection_price": "price",
    "objection_trust": "trust",
    "objection_not_interested": "not_interested",
    "objection_already_have": "already_have",
    "time_pressure": "no_time",
    "info_request": "info_request",
}


@dataclass
class TurnOutcome:
    match: MatchResult | None
    decision: SwitchDecision | None
    explicit_signals: list[explicit.ExplicitSignal] = field(default_factory=list)
    end_call: bool = False


class CallSession:
    def __init__(
        self,
        call_id: str,
        policy: ObjectivePolicy,
        registry: PersonaRegistry,
        matcher: PersonaMatcher,
        switch_cfg: SwitchConfig,
        mode: str = "treatment",
        context: dict | None = None,
        language: str = "hi",
        events: EventLog | None = None,
        recent_window: int = 3,
    ):
        if mode not in {"control", "shadow", "treatment"}:
            raise ValueError(mode)
        self.mode = mode
        self.recent_window = recent_window
        self.policy = policy
        self.registry = registry
        self.pinned = registry.snapshot()  # persona versions fixed for the whole call
        # Only personas approved for this call objective may be selected.
        self.allowed = {NEUTRAL_ID} | {
            pid
            for pid, v in self.pinned.items()
            if {"*", policy.objective_id} & set(registry.get(pid, v)["audience"]["call_objectives"])
            and policy.objective_id not in registry.get(pid, v)["audience"].get("excluded_objectives", [])
        }
        self.matcher = matcher
        self.switcher = PersonaSwitcher(
            switch_cfg,
            {pid: registry.get(pid, v).get("switching", {}).get("threshold_overrides", {}) for pid, v in self.pinned.items()},
        )
        self.features = LiveFeatureState()
        self.overrides = explicit.StyleOverrides()
        self.context = context or {}
        self.state = DialogueState(call_id=call_id, objective_id=policy.objective_id, language=language)
        self.events = events or EventLog(call_id, call_id, {"experiment_id": None, "arm": mode})
        self.turn_index = 0
        self.seller_turns = 0
        self._pending_turn_idx: int | None = None
        self.events.emit("call_started", self._persona_ref(NEUTRAL_ID), {"mode": mode, "pinned_personas": self.pinned})

    def _persona_ref(self, pid: str) -> dict:
        return {"persona_id": pid, "version": self.pinned[pid]}

    @property
    def applied_persona_id(self) -> str:
        return self.switcher.state.active if self.mode == "treatment" else NEUTRAL_ID

    def compose_next(self) -> ComposedPrompt:
        pid = self.applied_persona_id
        persona = self.registry.get(pid, self.pinned[pid])
        overrides = self.overrides if self.mode != "control" else explicit.StyleOverrides(language=self.overrides.language)
        prompt = compose(self.policy, self.state, persona, overrides)
        self.events.emit(
            "agent_turn_composed",
            self._persona_ref(pid),
            {"tts": prompt.tts, "overrides": overrides.as_dict(), "stage": self.state.stage.value},
            self.turn_index,
        )
        return prompt

    def on_agent_turn(self, text: str) -> None:
        self.features.add_turn(Turn(idx=self.turn_index, speaker=Speaker.AGENT, text=text))
        self.state.agent_turns += 1
        if self.state.stage == Stage.OPENING:
            self.state.consent.ai_disclosure_given = True
            self.state.stage = Stage.DISCOVERY
        self.turn_index += 1

    def on_seller_turn(self, text: str, asr_confidence: float | None = None) -> TurnOutcome:
        """Both phases back to back. Production runs ``ingest_seller_turn`` inline and ``evaluate_persona`` async."""
        r = self.ingest_seller_turn(text, asr_confidence)
        if r.end_call or self._pending_turn_idx is None or self.mode == "control":
            return r
        m, d = self.evaluate_persona()
        return TurnOutcome(m, d, r.explicit_signals)

    def ingest_seller_turn(self, text: str, asr_confidence: float | None = None) -> TurnOutcome:
        """Synchronous, sub-millisecond: features, state bookkeeping, explicit signals.

        Runs before the next reply is composed, so a direct request ("jaldi bolo",
        "Hindi mein bolo", "call mat karo") shapes the very next agent turn.
        """
        turn = Turn(idx=self.turn_index, speaker=Speaker.SELLER, text=text, asr_confidence=asr_confidence)
        self.turn_index += 1
        self._pending_turn_idx = None
        if not self.features.add_turn(turn):  # silence or low-confidence ASR: no behavioural evidence
            return TurnOutcome(None, None)
        sig = self.features.last_signals
        self.seller_turns += 1

        # Dialogue-state bookkeeping (the dialogue manager's job; shown here for completeness).
        for tag, objection in OBJECTION_TAGS.items():
            if tag in sig.tags and objection not in self.state.open_objections:
                self.state.open_objections.append(objection)
                self.state.stage = Stage.OBJECTION

        # Explicit signals: immediate, no gating. Opt-out goes to policy, not style.
        signals = explicit.detect(sig, self.features.recent_signals)
        if any(s.kind == "opt_out" for s in signals):
            self.state.consent.opted_out = True
            self.state.stage = Stage.CLOSING
            self.events.emit("opt_out_detected", self._persona_ref(self.applied_persona_id), {}, turn.idx)
            return TurnOutcome(None, None, signals, end_call=True)
        if signals:
            explicit.apply(signals, self.overrides)
            if self.overrides.language:
                self.state.language = self.overrides.language
            self.events.emit(
                "style_override_applied",
                self._persona_ref(self.applied_persona_id),
                {"signals": [s.kind for s in signals], "overrides": self.overrides.as_dict()},
                turn.idx,
            )
        self._pending_turn_idx = turn.idx
        return TurnOutcome(None, None, signals)

    def evaluate_persona(self) -> tuple[MatchResult | None, SwitchDecision | None]:
        """Off the response path: behavioural matching + gated switching for the last ingested seller turn."""
        if self.mode == "control" or self._pending_turn_idx is None:
            return None, None
        turn_idx, self._pending_turn_idx = self._pending_turn_idx, None
        feats = self.features.features(recent_seller_turns=self.recent_window)
        req = MatchRequest(self.state.call_id, turn_idx, feats, self.context, self.switcher.state.active)
        m = self.matcher.match(req)
        if any(k not in self.allowed for k in m.scores):
            m.scores = {k: v for k, v in m.scores.items() if k in self.allowed}
            if m.best_persona_id not in self.allowed:
                m.best_persona_id, m.abstained = NEUTRAL_ID, True
                m.reason = f"best persona not approved for objective {self.policy.objective_id}"
        if not m.scores:
            self.events.emit("matcher_fallback", self._persona_ref(self.applied_persona_id), {"reason": m.reason}, turn_idx)
        self.events.emit(
            "match_evaluated",
            self._persona_ref(self.applied_persona_id),
            {
                "scores": m.scores,
                "best": m.best_persona_id,
                "confidence": m.confidence,
                "abstained": m.abstained,
                "reason": m.reason,
                "scorer": m.scorer,
                "model_version": m.model_version,
                "latency_ms": m.latency_ms,
            },
            turn_idx,
        )
        d = self.switcher.update(m, self.seller_turns)
        event = {
            Action.COMMIT: "persona_committed",
            Action.SWITCH: "persona_switched",
            Action.REVERT_TO_NEUTRAL: "persona_reverted",
        }.get(d.action)
        if event:
            # In shadow mode the decision is logged with the persona that WOULD be active.
            self.events.emit(event, self._persona_ref(d.to_persona), d.as_dict(), turn_idx)
        return m, d

    def end_call(self, end_reason: str, disposition: str | None = None) -> dict:
        """Emit ``call_ended`` with a compact persona history. Outcomes arrive separately and later."""
        self.state.stage = Stage.ENDED
        history = [
            {"seller_turn": d.seller_turn, "action": d.action.value, "from": d.from_persona, "to": d.to_persona}
            for d in self.switcher.state.history
            if d.action != Action.RETAIN
        ]
        return self.events.emit(
            "call_ended",
            self._persona_ref(self.applied_persona_id),
            {
                "end_reason": end_reason,
                "disposition": disposition,
                "seller_turns": self.seller_turns,
                "agent_turns": self.state.agent_turns,
                "low_confidence_turns": self.features.low_confidence_turns,
                "silent_turns": self.features.silent_turns,
                "persona_history": history,
                "overrides": self.overrides.as_dict(),
                "opted_out": self.state.consent.opted_out,
            },
            self.turn_index,
        )

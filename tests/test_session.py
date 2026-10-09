import json

import jsonschema

from vani_persona import SCHEMA_DIR
from vani_persona.events import outcome_record
from vani_persona.matching.matcher import default_matcher
from vani_persona.orchestration.session import CallSession
from vani_persona.personas import NEUTRAL_ID
from vani_persona.switching.state_machine import SwitchConfig

OBJECTION_TURNS = [
    "aap kahan se bol rahe ho? kaun bol raha hai?",
    "mujhe nahi chahiye, meri pehle se baat ho chuki hai aapki team se",
    "ye paid hai kya? kitna charge lagega iska?",
    "interest nahi hai, aur aapko mera number kahan se mila?",
]


def session(registry, policy, thresholds, mode):
    return CallSession("c1", policy, registry, default_matcher(registry, thresholds), SwitchConfig.from_thresholds(thresholds), mode=mode)


def play(s, turns):
    for t in turns:
        s.compose_next()
        s.on_agent_turn("agent says something")
        s.on_seller_turn(t)


def test_treatment_commits_and_logs(registry, policy, thresholds):
    s = session(registry, policy, thresholds, "treatment")
    play(s, OBJECTION_TURNS)
    assert s.applied_persona_id == "objection_raising"
    types = [e["event_type"] for e in s.events.events]
    assert "persona_committed" in types and types[0] == "call_started"
    assert [e["seq"] for e in s.events.events] == list(range(1, len(types) + 1))


def test_shadow_decides_but_applies_neutral(registry, policy, thresholds):
    s = session(registry, policy, thresholds, "shadow")
    play(s, OBJECTION_TURNS)
    assert s.switcher.state.active == "objection_raising" and s.applied_persona_id == NEUTRAL_ID
    assert s.compose_next().persona_id == NEUTRAL_ID


def test_control_never_runs_matcher(registry, policy, thresholds):
    s = session(registry, policy, thresholds, "control")
    play(s, OBJECTION_TURNS)
    types = {e["event_type"] for e in s.events.events}
    assert "match_evaluated" not in types and s.applied_persona_id == NEUTRAL_ID


def test_opt_out_ends_call_in_every_mode(registry, policy, thresholds):
    for mode in ("control", "shadow", "treatment"):
        s = session(registry, policy, thresholds, mode)
        r = s.on_seller_turn("mujhe dobara call mat karo")
        assert r.end_call and s.state.consent.opted_out


def test_language_request_applies_immediately(registry, policy, thresholds):
    s = session(registry, policy, thresholds, "treatment")
    s.on_agent_turn("namaste")
    s.on_seller_turn("please speak in english")
    assert s.state.language == "en" and "English" in s.compose_next().sections["STYLE"]


def test_low_confidence_asr_is_not_evidence(registry, policy, thresholds):
    s = session(registry, policy, thresholds, "treatment")
    r = s.on_seller_turn("jaldi bolo busy hoon", asr_confidence=0.2)
    assert r.match is None and s.overrides.as_dict() == {}


def test_outcome_record_is_idempotent_and_valid():
    a = outcome_record("c1", "objective_achieved", True, "crm")
    b = outcome_record("c1", "objective_achieved", True, "crm")
    assert a["outcome_id"] == b["outcome_id"]
    jsonschema.validate(a, json.loads((SCHEMA_DIR / "outcome.schema.json").read_text()))


def test_explicit_request_shapes_next_reply_before_persona_evaluation(registry, policy, thresholds):
    s = session(registry, policy, thresholds, "treatment")
    s.on_agent_turn("namaste")
    s.ingest_seller_turn("bhai jaldi bolo")  # async evaluation has NOT run yet
    assert "at most 1 sentence" in s.compose_next().sections["STYLE"]
    m, d = s.evaluate_persona()
    assert m is not None and d is not None
    assert s.evaluate_persona() == (None, None)  # each ingested turn is evaluated once


def test_call_ended_summarises_persona_history(registry, policy, thresholds):
    s = session(registry, policy, thresholds, "treatment")
    play(s, OBJECTION_TURNS)
    ev = s.end_call("completed", "callback")
    assert ev["event_type"] == "call_ended" and ev["payload"]["persona_history"][0]["action"] == "commit"
    assert s.state.stage.value == "ended"


def test_personas_restricted_to_their_call_objectives(registry, thresholds):
    from vani_persona.orchestration.dialogue_state import ObjectivePolicy

    other = ObjectivePolicy(**{**ObjectivePolicy.load("fix_meeting").model_dump(), "objective_id": "other_objective"})
    s = CallSession("c1", other, registry, default_matcher(registry, thresholds), SwitchConfig.from_thresholds(thresholds))
    play(s, OBJECTION_TURNS)
    assert s.allowed == {NEUTRAL_ID} and s.switcher.state.active == NEUTRAL_ID

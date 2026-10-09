from vani_persona.matching.explicit import StyleOverrides
from vani_persona.orchestration.composer import compose
from vani_persona.orchestration.dialogue_state import DialogueState, Stage


def mid_call_state():
    return DialogueState(
        call_id="c", objective_id="fix_meeting", stage=Stage.OBJECTION, agent_turns=3,
        completed_steps=["confirm_right_person", "state_purpose"], open_objections=["price"],
        answered_questions=["which products"], commitments=["send link"],
    )


def test_switch_changes_only_style_and_tts(registry, policy):
    state = mid_call_state()
    snapshot = state.model_dump()
    prompts = [compose(policy, state, registry.get(pid), StyleOverrides()) for pid in registry.ids(include_neutral=True)]
    for key in ("POLICY", "OBJECTIVE", "STATE"):
        assert len({p.sections[key] for p in prompts}) == 1, f"{key} differs between personas"
    assert len({p.sections["STYLE"] for p in prompts}) == len(prompts)
    assert state.model_dump() == snapshot  # composing never mutates state


def test_opening_only_before_first_agent_turn(registry, policy):
    p = registry.get("objection_raising")
    fresh = DialogueState(call_id="c", objective_id=policy.objective_id)
    assert "Opening:" in compose(policy, fresh, p, StyleOverrides()).sections["STYLE"]
    assert "Opening:" not in compose(policy, mid_call_state(), p, StyleOverrides()).sections["STYLE"]


def test_open_objection_uses_persona_strategy_with_policy_facts_only(registry, policy):
    style = compose(policy, mid_call_state(), registry.get("time_pressed"), StyleOverrides()).sections["STYLE"]
    assert "Handling open objection 'price'" in style and "approved answer" in style


def test_explicit_overrides_beat_persona(registry, policy):
    o = StyleOverrides(language="en", max_sentences_per_turn=1, density="minimal", tts_rate=0.95, reasons=["x"])
    p = compose(policy, mid_call_state(), registry.get("objection_raising"), o)
    assert "at most 1 sentence" in p.sections["STYLE"] and "English" in p.sections["STYLE"]
    assert p.tts["tts_rate"] == 0.95


def test_policy_section_lists_forbidden_and_opt_out(policy, registry):
    text = compose(policy, mid_call_state(), registry.neutral, StyleOverrides()).sections["POLICY"]
    assert "override everything below" in text and "not to be called" in text
    assert all(f in text for f in policy.forbidden)

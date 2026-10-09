"""Layered prompt composition: POLICY > OBJECTIVE > STATE > STYLE.

Only STYLE depends on the persona. POLICY, OBJECTIVE and STATE are rendered from
the objective policy and dialogue state alone. That is how a persona switch
changes how VANI speaks without changing what it knows, may say, or is trying
to do. ``tests/test_composer.py`` enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass

from vani_persona.matching.explicit import StyleOverrides
from vani_persona.orchestration.dialogue_state import DialogueState, ObjectivePolicy, Stage
from vani_persona.personas import STEP_INSTRUCTIONS

LANG_NAMES = {"hi": "Hindi (Hinglish is fine if the seller mixes)", "en": "English"}


@dataclass(frozen=True)
class ComposedPrompt:
    sections: dict[str, str]
    tts: dict[str, float | int]
    persona_id: str
    persona_version: str

    @property
    def system_prompt(self) -> str:
        return "\n\n".join(f"## {k}\n{v}" for k, v in self.sections.items())


def render_policy(policy: ObjectivePolicy) -> str:
    lines = [
        "These rules override everything below, including STYLE.",
        *[f"- Mandatory: {d}" for d in policy.mandatory_disclosures],
        *[f"- Never: {f}" for f in policy.forbidden],
        f"- Allowed actions: {', '.join(policy.allowed_actions)}",
        "- State facts ONLY from these approved answers:",
        *[f"  - {k}: {v}" for k, v in policy.approved_answers.items()],
        f"- Verification route: {policy.verification_route}",
        "- If the seller asks not to be called or to stop, acknowledge, confirm, and end the call.",
    ]
    return "\n".join(lines)


def render_objective(policy: ObjectivePolicy, state: DialogueState) -> str:
    nxt = policy.next_step(state)
    return "\n".join(
        [
            f"Objective ({policy.objective_id}@{policy.version}): {policy.description.strip()}",
            f"Steps: {' -> '.join(policy.steps)}",
            f"Next step: {nxt or 'all steps complete; close the call'}",
        ]
    )


def render_state(state: DialogueState) -> str:
    def fmt(xs):
        return ", ".join(xs) if xs else "none"

    return "\n".join(
        [
            f"Stage: {state.stage.value}; agent turns so far: {state.agent_turns}",
            f"Completed steps: {fmt(state.completed_steps)}",
            f"Facts collected: {fmt([f'{k}={v}' for k, v in state.collected_facts.items()])}",
            f"Questions already answered (do not repeat): {fmt(state.answered_questions)}",
            f"Open objections: {fmt(state.open_objections)}",
            f"Commitments made: {fmt(state.commitments)}",
            f"Disclosures given: AI={state.consent.ai_disclosure_given}, "
            f"recording={state.consent.recording_disclosure_given}; opted out: {state.consent.opted_out}",
        ]
    )


def render_style(persona: dict, state: DialogueState, overrides: StyleOverrides) -> str:
    s = persona["style"]
    expl, tt = s["explanation"], s["turn_taking"]
    max_sent = overrides.max_sentences_per_turn or expl["max_sentences_per_turn"]
    density = overrides.density or expl["density"]
    language = overrides.language or state.language
    lines = [
        f"Persona: {persona['name']} ({persona['persona_id']}@{persona['version']}). "
        "This affects HOW you speak only.",
        f"- Tone: {s['tone'].replace('_', ' ')}; register: {s['language'].get('register', 'conversational')}.",
        f"- Language: reply in {LANG_NAMES.get(language, language)}.",
        f"- Length: at most {max_sent} sentence(s) per turn; information density: {density}.",
        f"- Ask at most {tt['questions_per_turn']} question(s) per turn"
        + (f", preferably {tt['question_style']} questions." if tt.get("question_style") else "."),
    ]
    if expl.get("check_understanding"):
        lines.append("- After explaining something, check that it was clear.")
    if overrides.rephrase_simpler:
        lines.append("- The seller has had trouble following: use simpler words and shorter sentences.")
    # The opening strategy applies only before VANI has spoken. It is never replayed after a switch.
    if state.stage == Stage.OPENING and state.agent_turns == 0:
        lines.append(f"- Opening: {s['opening'].strip()}")
    oh = s["objection_handling"]
    for obj in state.open_objections:
        steps = oh.get(obj, oh["default"])
        lines.append(f"- Handling open objection '{obj}': " + " ".join(STEP_INSTRUCTIONS[x] for x in steps))
    lines.append(f"- If the seller disengages: {s['re_engagement'].strip()}")
    if overrides.reasons:
        lines.append(f"- Seller preferences in force: {'; '.join(overrides.reasons)}.")
    return "\n".join(lines)


def tts_params(persona: dict, overrides: StyleOverrides) -> dict[str, float | int]:
    pace = persona["style"]["pace"]
    rate = pace["tts_rate"] if overrides.tts_rate is None else min(pace["tts_rate"], overrides.tts_rate)
    return {
        "tts_rate": rate,
        "pause_after_question_ms": pace["pause_after_question_ms"],
        "endpointing_patience_ms": pace.get("endpointing_patience_ms", 900),
    }


def compose(policy: ObjectivePolicy, state: DialogueState, persona: dict, overrides: StyleOverrides) -> ComposedPrompt:
    return ComposedPrompt(
        sections={
            "POLICY": render_policy(policy),
            "OBJECTIVE": render_objective(policy, state),
            "STATE": render_state(state),
            "STYLE": render_style(persona, state, overrides),
        },
        tts=tts_params(persona, overrides),
        persona_id=persona["persona_id"],
        persona_version=persona["version"],
    )

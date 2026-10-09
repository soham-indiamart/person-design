"""Two-persona demo: initial selection, a mid-call switch, and preserved context.

    python -m vani_persona.demo.run_demo [--json events.jsonl]

ILLUSTRATIVE MECHANICS. The personas are data-backed CANDIDATES from run seg-63b6bc47e6
(segment evidence is real; their styles are unvalidated). The scorer is the
uncalibrated prototype scorer, so the demo needs no trained model. Seller and agent
lines are scripted; agent lines are examples of what an LLM would produce from the
printed STYLE block. The demo shows the mechanics (signals -> selection -> switch ->
continuity). It is not evidence that the personas improve outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vani_persona.config import load_thresholds
from vani_persona.events import EventLog
from vani_persona.matching.matcher import default_matcher
from vani_persona.orchestration.dialogue_state import ObjectivePolicy
from vani_persona.orchestration.session import CallSession
from vani_persona.personas import NEUTRAL_ID, PersonaRegistry
from vani_persona.switching.state_machine import SwitchConfig

# (seller utterance, {persona_id: agent reply}). The reply is picked by the persona active AFTER the seller turn.
O, T, N = "objection_raising", "time_pressed", NEUTRAL_ID

SCRIPT: list[tuple[str, dict[str, str]]] = [
    (
        "Haan ji, boliye.",
        {N: "Dhanyavaad. Kya pehle kabhi kisi IndiaMART executive se aapki baat hui hai?"},
    ),
    (
        "Aap kahan se bol rahe ho? Mera number kahan se mila aapko?",
        {N: "Main IndiaMART se VANI bol rahi hoon, aapke IndiaMART account ke silsile mein call kiya hai."},
    ),
    (
        "Meri pehle se baat ho chuki hai aapki team se, mujhe nahi chahiye abhi ye sab.",
        {
            O: "Samajh gayi, aapki baat sahi hai. Kya aap chahenge ki wahi executive aapse milein jinse baat hui thi?",
            N: "Theek hai. Kya aap chahenge ki wahi executive aapse milein?",
        },
    ),
    (
        "Theek hai, par abhi main busy hoon, baad mein baat karte hain.",
        {
            O: "Bilkul, aapka samay keemti hai. Kya main kal ke liye ek samay note kar loon?",
            N: "Theek hai. Kya main kal ke liye ek samay note kar loon?",
        },
    ),
    (
        "Abhi time nahi hai.",
        {
            O: "Koi baat nahi. Kal subah theek rahega ya kal shaam?",
            N: "Koi baat nahi. Kal subah theek rahega ya kal shaam?",
            T: "Ji. Kal subah ya kal shaam?",
        },
    ),
    (
        "Kal shaam.",
        {
            O: "Theek hai, kal shaam. Kitne baje aapke liye theek rahega?",
            T: "Kal shaam. 5 baje ya 7 baje?",
            N: "Theek hai, kal shaam. Kitne baje theek rahega?",
        },
    ),
    (
        "7 baje ke baad.",
        {
            T: "Theek hai, kal 7 baje ke baad. Meeting aapki dukaan par hi?",
            O: "Theek hai, kal 7 baje ke baad. Kya executive aapki dukaan ke pate par aa sakte hain?",
            N: "Theek hai, kal 7 baje ke baad. Meeting aapki dukaan par hi rakhein?",
        },
    ),
    (
        "Haan, wahi.",
        {T: "Done. Dhanyavaad, namaste.", O: "Theek hai, kal 7 baje dukaan par. Dhanyavaad.", N: "Theek hai. Dhanyavaad."},
    ),
]

OPENING = (
    "Namaste, main VANI, IndiaMART ki automated assistant bol rahi hoon. Aapke IndiaMART account ke liye "
    "ek executive se meeting ke silsile mein call kiya hai, kya do minute baat ho sakti hai?"
)

# Steps the dialogue manager would mark complete after each agent turn (index = agent turn).
STEPS_AFTER_AGENT_TURN = {
    0: ["confirm_right_person"],
    2: ["state_purpose"],
    3: ["check_existing_executive_contact"],
    6: ["propose_meeting_slot"],
    7: ["confirm_location"],
    8: ["confirm_and_close"],
}


def _hr(title: str = "") -> str:
    return f"\n{'─' * 8} {title} {'─' * max(0, 70 - len(title))}"


def run(events_path: Path | None = None, out=print) -> CallSession:
    th = load_thresholds()
    registry = PersonaRegistry()
    policy = ObjectivePolicy.load("fix_meeting")
    matcher = default_matcher(registry, th)
    log = EventLog("demo-call-001", "demo-session-001", {"experiment_id": "demo", "arm": "treatment"}, events_path)
    s = CallSession(
        "demo-call-001",
        policy,
        registry,
        matcher,
        SwitchConfig.from_thresholds(th),
        mode="treatment",
        context={"campaign_id": "main_vani", "call_objective": policy.objective_id, "lead_type": "PIM"},
        events=log,
        recent_window=th["matching"]["recent_window_seller_turns"],
    )
    out(__doc__.split("\n\n")[2].strip())
    out(_hr("CALL START: persona = neutral_default"))
    p = s.compose_next()
    out(f"[STYLE]\n{p.sections['STYLE']}\n[TTS] {p.tts}")
    out(f"\nVANI  ({s.applied_persona_id}): {OPENING}")
    s.on_agent_turn(OPENING)
    for step in STEPS_AFTER_AGENT_TURN.get(0, []):
        s.state.completed_steps.append(step)

    for i, (seller, replies) in enumerate(SCRIPT, start=1):
        out(_hr(f"seller turn {i}"))
        out(f"SELLER: {seller}")
        before = s.applied_persona_id
        r = s.on_seller_turn(seller)
        sig = s.features.last_signals
        out(f"  signals: tags={sorted(sig.tags) if sig else []}, tokens={sig.tokens if sig else 0}")
        if r.explicit_signals:
            out(f"  explicit: {[x.kind for x in r.explicit_signals]} -> overrides {s.overrides.as_dict()}")
        if r.match:
            sc = ", ".join(f"{k}={v:.2f}" for k, v in sorted(r.match.scores.items(), key=lambda kv: -kv[1]))
            out(f"  match:   [{sc}] calibrated={r.match.calibrated} -> {r.match.reason}")
            if r.match.evidence:
                ev = "; ".join(f"{e.feature}={e.value:.2f} ({e.detail})" for e in r.match.evidence[:3] if e.value is not None)
                out(f"  evidence: {ev}")
        if r.decision:
            out(f"  switch:  {r.decision.action.value.upper()}: {r.decision.reason}")
        after = s.applied_persona_id
        if after != before:
            out(f"  >>> PERSONA {before} -> {after} (dialogue state untouched: stage={s.state.stage.value}, "
                f"completed_steps={s.state.completed_steps}, open_objections={s.state.open_objections})")
        if r.end_call:
            out("  opt-out -> policy layer ends the call")
            break

        prompt = s.compose_next()
        reply = replies.get(after) or replies.get(NEUTRAL_ID) or next(iter(replies.values()))
        out(f"[STYLE]\n{prompt.sections['STYLE']}\n[TTS] {prompt.tts}")
        out(f"\nVANI  ({after}): {reply}")
        s.on_agent_turn(reply)
        for step in STEPS_AFTER_AGENT_TURN.get(i, []):
            s.state.completed_steps.append(step)
        if i == 2 and "trust" in s.state.open_objections:
            # Identity and purpose were answered: the dialogue manager marks the trust objection resolved.
            s.state.open_objections.remove("trust")
            s.state.resolved_objections.append("trust")
        if i == 6:
            # Seller chose a slot: the dialogue manager marks the open objections resolved.
            s.state.resolved_objections += s.state.open_objections
            s.state.open_objections = []
        if i == 7:
            s.state.collected_facts["meeting_slot"] = "tomorrow after 19:00"
        if i == 8:
            s.state.collected_facts["meeting_location"] = "seller's shop"
            s.state.commitments.append("executive visit tomorrow after 19:00 at the seller's shop")

    s.end_call("completed", disposition="callback")
    out(_hr("CONTINUITY CHECK"))
    hist = [d for d in s.switcher.state.history if d.action.value != "retain"]
    for d in hist:
        out(f"  seller turn {d.seller_turn}: {d.action.value} {d.from_persona} -> {d.to_persona} ({d.reason})")
    out(f"  final state: stage={s.state.stage.value}, completed_steps={s.state.completed_steps}")
    out(f"  open_objections={s.state.open_objections}, resolved={s.state.resolved_objections}, commitments={s.state.commitments}")
    out(f"  opening replayed after switch: {'no' if s.state.agent_turns > 0 else 'n/a'} (opening is gated on agent_turns == 0)")
    out(f"  events logged: {len(log.events)} ({', '.join(sorted({e['event_type'] for e in log.events}))})")
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None, help="also write call events to this JSONL file")
    a = p.parse_args()
    run(a.json)


if __name__ == "__main__":
    main()

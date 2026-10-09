# 10 · Demonstration plan

*Deliverable 11, plus brief §9.*

## Stage A: mechanics demo (built; runs today)

```bash
.venv/bin/python -m vani_persona.demo.run_demo            # prints the full trace
.venv/bin/python -m vani_persona.demo.run_demo --json /tmp/demo_events.jsonl   # also writes the events
```

**What it is.**
- A scripted Hinglish seller call for the real objective `fix_meeting` (`config/objectives/fix_meeting.yaml`), using the three data-backed personas.
- It runs through the real `CallSession`: feature extraction, explicit signals, matcher, switching state machine, composer and event log.
- Agent replies are **pre-written** for each persona, to show what an LLM would produce from the printed STYLE block. There is no LLM or TTS in the loop.

**Labelling.**
- The personas are data-backed **candidates** (run `seg-63b6bc47e6`): the segments are real, the styles are unvalidated.
- The scorer is the uncalibrated prototype scorer with prototypes taken from the real segments, so the demo runs without the gitignored trained model.
- The seller lines are written to resemble patterns seen in the PS07 transcripts. They are not quotes.
- The demo proves the **mechanics**, not that any persona works.

| Requirement | Where in the trace |
|---|---|
| 1. Initial utterance and extracted signals | Seller turn 1 "Haan ji, boliye.": `tags=[affirmation]`; insufficient evidence, so neutral stays |
| 2. Initial persona selection and confidence | Seller turn 2 (identity question "kahan se bol rahe ho?") puts `objection_raising` at 0.71 (1/2 confirmations). Seller turn 3 ("pehle se baat ho chuki hai, nahi chahiye") gives `COMMIT neutral_default → objection_raising (0.71 ≥ 0.6 for 2 evaluations)` |
| 3. Response under the selected persona | Agent reply after turn 3: acknowledges, offers the same executive, one closed question; TTS 0.97 |
| 4. A later signal that changes the style | Turns 4–5: "abhi main busy hoon, baad mein", "abhi time nahi hai". The persona is **held** while objections and rush both sit in the window (hysteresis); a challenger appears at turn 6 |
| 5. Mid-call switch without losing context | Turn 7: `SWITCH objection_raising → time_pressed`. The trace prints the unchanged stage, completed steps and resolved objections at the moment of the switch |
| 6. Revised response under the new style | Agent reply after turn 7: one sentence, confirms the slot and asks the location as a single closed question; TTS 1.06 |
| 7. Explanation of the switch | The `reason` string + `evidence` lines + the `persona_switched` event payload (`schemas/examples/call_event.persona_switched.json`) |

Explicit overrides (language, brevity, repeated clarification) are covered by `tests/test_session.py`. The scripted call doesn't use them, because the data shows brevity requests are rare (0.1% of calls).

Isolation of style:
- The POLICY, OBJECTIVE and STATE sections are identical for every persona, as `tests/test_composer.py` asserts.
- Facts in the agent replies come only from the objective's approved answers.
- Nothing about the offer, the objective or the permissions differs between personas.

## Stage B: voice demo (after personas are validated)

Goal: let stakeholders *hear* the difference, still without claiming effectiveness.

1. **Personas.** `objection_raising` and `time_pressed` once `validated_offline`, compared with `neutral_default`.
2. **Calls.** Three scripted seller recordings or role-played sellers per persona, drawn from real (redacted, consented) transcript patterns of each segment, plus one call that changes behaviour mid-way.
3. **Pipeline.** The real VANI stack (ASR → orchestrator + `vani_persona` → LLM → TTS), same objective policy and same voice. Only STYLE and TTS params differ.
4. **Comparison.** For each seller script, generate (a) neutral, (b) the matched persona, (c) the mismatched persona.
5. **Blind listening panel.** Sales/CX staff rate clarity, appropriateness and respectfulness on a 1–5 scale and flag any factual or policy deviation, without knowing which version is which. Any policy deviation blocks the persona.
6. **Output.** A short report with audio, the event trace for each call, and the panel ratings. It is explicitly *not* outcome evidence: that comes only from the A/B test (doc 09).

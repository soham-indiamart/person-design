# 15 · PS04 submission: seller-fit adaptive persona on Sarvam

The brief is Voice AI Hackathon 2.0, Problem Statement 04 "Persona Design": build a persona for each seller from their seller data before the call, adapt it live, and show it in voice on Sarvam. Docs 00–14 were written for an earlier, broader brief (persona library, A/B test, rollout). They remain background; this document maps the PS04 deliverables to the code.

## Deliverables

| PS04 deliverable | Where | How to produce it |
|---|---|---|
| Persona generator running on the sample seller files | `src/vani_persona/ps04/generator.py` | `python -m vani_persona.ps04.generate .` |
| Persona spec per seller, with a reason for each choice | `artifacts/ps04/persona_specs.jsonl.gz` (149,363 specs, gitignored) + `shared_policy.json`; summary in [findings/ps04_generator_report.md](findings/ps04_generator_report.md) | same command |
| Evidence from past calls | [findings/ps04_evidence.md](findings/ps04_evidence.md), `evidence/ps04_evidence.json` | `python -m vani_persona.ps04.evidence .` |
| Live demo with at least one mid-call switch, 2–3 contrasting sellers | Browser demo `src/vani_persona/ps04/server.py` + `static/index.html` | `python -m vani_persona.ps04.server`, open http://localhost:8765 |
| Switch log (signal, change, reason) | On screen during the call; **download** button; `artifacts/ps04/switch_logs/<call>.json` | written automatically for every call |
| Backup recording (same scenario, 3 sellers) | `artifacts/ps04/demo_audio/*.wav` + `README.md` + switch logs | `python -m vani_persona.ps04.simulate` |
| One-slide presentation | [ps04_slide.html](ps04_slide.html) / [ps04_slide.pdf](ps04_slide.pdf) | open in a browser |

## Pipeline

```
seller file row ──► generator (rules + ps04_evidence.json) ──► persona spec
                                                                   │ start of call
                                         starting live persona (from seller file)
seller audio ──► Saaras v3 (transcribe ‖ translit, language ID)
                                                                   │
        Behaviour Analyst agent (sarvam-105b-conversations, reads last 6 turns of transcript)
              + phrase lexicons (fallback / aggression safety net)  ──► seller state, intensity, trend
                                                                   │
        Persona Director (deterministic rules) ──► switch named persona? write switch log
        Language tracker (Saaras language ID)  ──► mirror language?  write switch log
                                                                   │
            Sarvam LLM (sarvam-105b-conversations): facts + guardrails + plan + ACTIVE PERSONA ──► reply
                                                                   │
                     Bulbul v3: speaker, pace, expressiveness (temperature), language ──► seller
```

- **Speech to text:** `saaras:v3`, called twice in parallel: `transcribe` (native script, used for the LLM and the log) and `translit` (romanized, matches the Hinglish lexicons). Language ID covers 23 codes.
- **LLM:** `sarvam-105b-conversations`, Sarvam's model for real-time voice agents. It is used twice per turn, as two separate agents: the Behaviour Analyst and the reply writer. One key, one vendor.
- **Text to speech:** `bulbul:v3`, 16 kHz. Each reply's sentences are synthesized in parallel and joined.
- **Languages:** Saaras understands 23 languages. Bulbul speaks 11 of them: Hindi, English, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu. If a seller switches to a language Saaras understands but Bulbul cannot speak (e.g. Assamese, Urdu), VANI keeps replying in simple Hindi, and the switch log records why.

## Persona spec (per seller)

| Part | Settings | Driven by (seller file) |
|---|---|---|
| Voice | gender, speaker, accent, pitch (register), pace, expressiveness | type, past call length, frustration flag |
| Language | primary language, Hinglish mix, formality, prepared regional language | state (with measured English share), business type, turnover |
| Tone | warmth, empathy | GST year, turnover, flags, past outcomes |
| Plan | opening steps, personalisation hooks, objection playbook (seller's own past objections first), escalation rules, guardrails | category, city, nature of business, enquiries, catalog score, past objections and outcomes |

Every setting is `{value, reason, evidence: {key, strength, stat}, confidence}`. Missing fields get defaults with `low` confidence and are listed in `missing_fields` (FAQ Q6). Do-not-call sellers get no persona; the call is suppressed.

**Honest limits** (the brief asks for these):
- VANI used one voice and one style on every past call. So the data shows what each seller type struggles with: short calls, refusals, language. It cannot show that a given voice works better. Voice gender, speaker and pitch are marked `informed_guess`.
- Sarvam's current text-to-speech, `bulbul:v3`, has no pitch parameter, and `bulbul:v2` is retired. Pitch is therefore a register (lower / neutral / higher) carried by the speaker choice.
- The seller-file flags (frustration, asked-if-bot, already-in-touch) summarize the same calls they are compared with. The call-history rules use earlier calls only.

## Live personas, chosen during the call

The seller spec fixes *who* VANI is for this seller: speaker, language, formality and plan. A **live persona** fixes *how* VANI talks right now. The live persona is picked during the call from what the seller actually does, not chosen by the operator.

| Live persona | Seller state that selects it | Delivery | Behaviour |
|---|---|---|---|
| Friendly Guide | calm / neutral | the seller's baseline from the spec | follow the plan, explain the benefit, move to a time |
| Calm De-escalator | aggressive / frustrated | pace 0.88, low expressiveness, short, simple | apologise once, never argue or re-pitch, offer a callback or to end |
| Quick & Direct | rushed | pace 1.2, very short | skip pleasantries; one line on why, then ask for a time or offer a callback |
| Patient Explainer | confused | pace 0.85, short, simple | re-explain with an example, one idea per sentence; never "did you understand?" |
| Enthusiastic Closer | interested | warm, expressive | propose a concrete 11 am–5 pm slot, visit or online, and confirm it |
| Transparent Reassurer | suspicious (fraud, spam, bot) | calm, short | say plainly it's IndiaMART's AI assistant; never ask for payment or OTP |

- **Same voice throughout.** A persona switch changes pace, expressiveness, warmth, empathy, sentence length, vocabulary and behaviour. It never changes the speaker, language, facts or guardrails.
- **Starting persona from the seller file:** Quick & Direct for time-pressed sellers, Transparent Reassurer for sellers who asked before whether VANI is a bot, Friendly Guide otherwise. "Unknown seller" also starts on Friendly Guide.

**Behaviour Analyst agent** (`ps04/analyst.py`)
- A separate LLM call after every seller turn.
- **Input:** the last 6 turns of the transcript (both sides), the active persona and its previous verdict.
- **Output:** `state` (calm, aggressive, rushed, confused, interested or suspicious), `intensity` 0–1, `trend` (escalating, steady or calming), the seller's words as evidence, and a one-line rationale.
- Because it sees the conversation rather than a single utterance, it can tell "irritated" from "calming down".
- **Fallback and safety net:** the phrase lexicons. They take over if the call fails, times out (2.5 s) or hits the credit cap, and an explicit aggression phrase always wins over a softer reading.
- **Self-test** (`python -m vani_persona.ps04.analyst --selftest`): 29/30 on labelled Hinglish and English lines. The lexicons alone score 26–28/30.

**Persona Director** (`ps04/director.py`) is a set of deterministic rules, so every switch can be explained:
- **Switch now:** intensity ≥ 0.7. Aggression switches at ≥ 0.6 or on any aggression phrase.
- **Weaker signals** (0.5–0.7) must repeat on a second turn, unless the analyst and the lexicon agree.
- **Back to Friendly Guide** after 2 calm turns.
- **No ping-pong:** switching straight back to the persona just left needs ≥ 0.8.
- **Escalation:** a 3rd consecutive aggressive turn → apologise, offer a callback and end the call.

**Language mirroring** is independent of persona:
- It uses Saaras language ID (p ≥ 0.6) or an explicit request ("English mein bolo").
- Hindi↔English needs two utterances; other languages switch at once.

**Switch log:** every switch is logged with the turn, from → to, seller state, intensity, trend, detector, the seller's words, the reason and the settings changed. Each call also saves the analyst verdicts and the persona timeline.

## Running the demo

```bash
# once
.venv/bin/python -m vani_persona.ps04.evidence .
.venv/bin/python -m vani_persona.ps04.generate .
# live demo (Chrome; allow the microphone; use headphones so VANI doesn't hear itself)
.venv/bin/python -m vani_persona.ps04.server
# backup recording of the same scenario for 3 sellers
.venv/bin/python -m vani_persona.ps04.simulate
```

The Sarvam key is read from `SARVAM_API_KEY` or the gitignored `.env`.

**Suggested live script (same for every seller), with the expected persona:**
1. "haan ji, kaun?": stays Friendly Guide.
2. "kitni baar call karoge yaar, dimaag kharab mat karo": **Calm De-escalator**.
3. "achha theek hai, bolo": still De-escalator (1 calm turn).
4. "haan sun raha hoon, aage bolo": back to **Friendly Guide**.
5. "matlab? kaun si meeting? samajh nahi aaya": **Patient Explainer**.
6. Switch language (Tamil, English or Gujarati) or say "English mein bolo": language switch, same persona.
7. "theek hai, kab aa sakte ho?": **Enthusiastic Closer**.
8. "kal 12 baje": meeting confirmed.

Try also "fraud call hai kya? number kahan se mila?" → **Transparent Reassurer**, and "abhi busy hoon, jaldi bolo" → **Quick & Direct**.

Run it with "Unknown seller" and with the contrasting sellers (new seller in Odisha, time-pressed in UP, English-first in Tamil Nadu). They start from different baselines, and the time-pressed seller starts as Quick & Direct.

**Latency:** about 3.5–5 s from the end of the seller's speech to the start of the reply. Roughly 0.6–1.5 s is speech-to-text, 0.4–0.8 s the Behaviour Analyst, 0.5–1.5 s the reply, and about 2 s text-to-speech. This is slower than production VANI. Streaming text-to-speech and running the scorer speculatively are the next steps.

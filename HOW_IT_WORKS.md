# How `person-design` works

This is a working guide to the code: what each part does, how data moves through it, and where the obvious improvements are. It was written from the source, not only from `docs/`.

---

## 1. The big picture

The repo holds **two systems** that share one Python package, `src/vani_persona/`:

| | **A. Offline persona-discovery pipeline (PS07)** | **B. Live voice demo (PS04 hackathon)** |
|---|---|---|
| Goal | Discover personas from ~12k historical VANI calls, train a matcher, and prove it with an A/B test | Build a persona per seller *before* the call, adapt it *during* the call, in real voice on Sarvam |
| Code | `ingest/`, `features/`, `segmentation/`, `matching/`, `switching/`, `orchestration/`, `experiment/`, `audit.py` | `ps04/` (all of it) |
| Brain | GMM clustering → calibrated logistic regression → hysteresis state machine | LLM "Behaviour Analyst" + phrase lexicons → deterministic "Persona Director" |
| Status | Run on real data; 3 personas (`neutral_default`, `objection_raising`, `time_pressed`) | Working browser demo + scripted backup recording |

**Important:** System B does **not** use System A's trained matcher or state machine at runtime. It reuses only the lexicons (`features/lexicons.py`) and text normalisation (`features/transcript.py`). The two are parallel designs for the same problem (see §7.1).

The core rule in both is the same: **a persona changes *how* VANI speaks (pace, tone, length, wording style), never *what* it may say** (facts, guardrails, call purpose).

---

## 2. System B: the PS04 live demo (the part you'll most likely improve)

### 2.1 End-to-end flow

```
            BEFORE THE CALL                                    DURING THE CALL (every seller turn)
 ┌──────────────────────────────┐            ┌──────────────────────────────────────────────────────────┐
 │ seller file row (CSV/JSON/XL)│            │ Browser mic → VAD → WAV → POST /api/turn                 │
 │        │                     │            │        │                                                 │
 │        ▼                     │            │        ▼                                                 │
 │ generator.py  ◄── evidence/  │            │ Saaras v3 STT ×2 in parallel (native + romanised, lang ID)│
 │  (rules)          ps04_      │            │        │                                                 │
 │        │          evidence   │            │        ├─► looks_incomplete()? → hold & wait for more     │
 │        ▼          .json      │            │        │                                                 │
 │ use_cases.py (pick use case, │            │        ├─► SPECULATIVE DRAFT reply (current persona) ──┐  │
 │  archetype, checklist)       │            │        │                                              │  │
 │        │                     │            │        ▼                                              │  │
 │        ▼                     │            │ analyst.py: LLM reads last 6 turns → state, intensity, │  │
 │ PERSONA SPEC                 │──start────►│   trend, profile  (+ lexicon fallback / safety net)    │  │
 │ voice/language/tone/plan,    │  persona   │        │                                              │  │
 │ each with reason+evidence    │            │        ▼                                              │  │
 └──────────────────────────────┘            │ LanguageTracker → mirror language?                     │  │
                                             │ LiveProfile.observe → pace bias, patience, formality   │  │
                                             │ PersonaDirector.decide → switch / stay / pending /     │  │
                                             │   escalate                                             │  │
                                             │        │                                              │  │
                                             │        ▼                                              │  │
                                             │ text-shaping settings unchanged? ──yes──► reuse draft ◄┘  │
                                             │        │ no                                              │
                                             │        ▼                                                 │
                                             │ regenerate reply (sarvam-105b-conversations)             │
                                             │        ▼                                                 │
                                             │ strip ##STAGE/NOTE/MEETING/HANGUP## tags                 │
                                             │        ▼                                                 │
                                             │ Bulbul v3 TTS, sentence by sentence, streamed as NDJSON  │
                                             │        ▼                                                 │
                                             │ switch log → artifacts/ps04/switch_logs/<call>.json      │
                                             └──────────────────────────────────────────────────────────┘
```

### 2.2 Before the call: building the persona spec

**`ps04/evidence.py`** (`python -m vani_persona.ps04.evidence .`)
- Reads the PS07 export (bot calls, turns, seller files) via `ps04/data.py`.
- Computes aggregate comparisons, e.g. "new sellers fix meetings at X% vs Y%", with a 95% CI.
- Writes `evidence/ps04_evidence.json`. Each key has a strength: `moderate` (CI excludes zero on ≥1,000 calls), `weak`, or `informed_guess`.
- Caveat built in: VANI used one voice on every past call. The evidence shows what each seller type struggles with, not which voice works better.

**`ps04/generator.py`**: `generate(row, evidence) -> spec`
1. `traits(row)` derives booleans from the seller file: `new_seller` (GST ≥ 2024), `high_turnover`, `short_calls` (avg answered call < 25 s), `prior_call_later`, `frustration`, `asked_if_bot`, `in_touch`, `dnc`, past objections, and so on.
2. `seller_type(traits)` picks one label using a **priority order**:
   `do_not_call` → `guarded` → `already_engaged` → `time_pressed` → `established_formal` → `new_receptive` → `standard`.
3. It builds the spec. Every setting is `{value, reason, evidence: {key, strength, stat}, confidence}`:
   - **voice**: speaker (`priya` / `ishita` / `ritu`), pace, expressiveness, pitch register
   - **language**: primary language (from state, e.g. Gujarat → `gu-IN`), Hinglish mix, formality
   - **tone**: warmth, empathy
   - **conversation_plan**: opening steps, personalisation hooks, objection playbook (the seller's own past objections first), escalation rules, guardrails
4. `FACTS` and `GUARDRAILS` are constants shared by every seller and never edited.
5. Do-not-call sellers get no spec: `call_policy: suppress`.

**`ps04/use_cases.py`**: applied on top of the spec
- There are 8 use cases: `seller_enrichment`, `meeting_scheduling`, `lead_qualification`, `response_recovery`, `onboarding`, `renewal`, `issue_resolution`, `catalogue_verification`.
- `infer_use_case(row)`: an explicit `use_case` column wins. Otherwise the order is open ticket → renewal ≤ 60 days → unreplied enquiries → enquiry to qualify → onboarding → listings to verify → missing profile fields.
- Each use case picks an **archetype** (e.g. busy shopkeeper vs manufacturer). The archetype sets the starting/resting persona, the approach, the delivery tweaks, the call context and a **checklist**. The checklist is what the call must collect, and it is recorded live through `##NOTE: key=value##` tags.

**`ps04/seller_file.py`** parses an uploaded CSV/JSON/Excel. **`ps04/cook.py`** writes `samples/cooked_sellers.csv`, a set of synthetic sellers covering every use case. **`ps04/generate.py`** runs the generator over all 149,363 real sellers and writes `artifacts/ps04/persona_specs.jsonl.gz`, `shared_policy.json`, `demo_sellers.json` and a report.

### 2.3 During the call: the live loop (`ps04/live.py` → `CallSession`)

Each call has three layers of settings, composed in `CallSession.compose()`:

```
seller baseline (from spec)  →  live language  →  active persona's delivery template  →  live profile adjustments
```

**Per seller turn (`_seller_turn`)**

1. **STT** (`seller_audio`): Saaras is called twice in parallel. `transcribe` returns native script (used by the LLM and the log). `translit` returns romanised text (matched against the Hinglish lexicons). The call also returns language ID and probability.
2. **End-of-utterance check:** `signals.looks_incomplete()` looks for a trailing connector, a filler or a trailing comma. If the utterance seems cut off, the session holds it (up to `MAX_WAITS = 2`) and merges it with the next chunk.
3. **Speculative draft:** a reply is generated *in parallel* with the analyst, using the current persona and a fast language guess (`LanguageTracker.peek`).
4. **Behaviour Analyst** (`analyst.analyse`):
   - A separate LLM call (`temperature=0`, JSON mode, timeout `TIMEOUT_S = 3.0`, no retries) over the last 6 turns.
   - Returns `state` (one of 10: calm, aggressive, rushed, confused, interested, suspicious, hesitant, complaining, inquisitive, uninterested), plus `intensity`, `trend`, `evidence` and `rationale`.
   - Also returns a **profile**: language, mix, pace wanted, patience, formality, and a short description of the seller.
   - The phrase lexicons always run as well:
     - On LLM failure or timeout, or when the credit cap is hit, the lexicon verdict is used instead.
     - An aggression phrase **always overrides** a softer LLM reading.
     - A speed complaint ("itna dheere kyu") turns "confused" into "rushed".
     - If the LLM and the lexicon agree, `source = "analyst+lexicon"`, which lowers the switch bar.
5. **Language** (`signals.LanguageTracker`):
   - Mirrors the seller when Saaras language ID has p ≥ 0.6, or on an explicit request ("English mein bolo").
   - Hindi↔English needs 2 consecutive utterances; any other language switches at once.
   - If Saaras understands the language but Bulbul can't speak it (e.g. Urdu, Assamese), VANI replies in simple Hindi and offers an executive who speaks it.
6. **Live profile** (`profile.LiveProfile.observe`):
   - **Pace bias:** an explicit faster/slower request moves it by ±0.12 and stacks. It is sticky across persona switches.
   - **Speech rate:** the seller's words per second (EMA) nudges pace.
   - **Patience:** "low" takes effect at once; leaving "low" needs 2 calmer reads in a row.
   - **Home persona:** `quick_direct` if the seller is impatient or wants it faster, otherwise `normal` (or the use case's rest persona). This is where the Director returns to once the seller is calm.
7. **Persona Director** (`director.PersonaDirector.decide`): deterministic rules.

   | Rule | Threshold |
   |---|---|
   | Clear signal → switch now | intensity ≥ 0.7 |
   | Aggression → switch now | ≥ 0.6, or any aggression phrase |
   | Weak signal → must repeat on 2 turns (or analyst+lexicon agree) | 0.5 – 0.7 |
   | Calm → return to home persona | 2 calm turns (instant between `normal` ↔ `quick_direct`) |
   | Anti ping-pong (going straight back to the persona just left) | ≥ 0.8 |
   | Escalation: end the call with a callback offer | 3rd consecutive aggressive turn while in De-escalator |

8. **Settings and switch log:** each setting change is attributed to its cause (language, persona or profile). Each entry records turn, from → to, signal, detector, score, trend, the seller's words, the settings changed and the reason.
9. **Reply:** the draft is reused if `(persona, escalate, TEXT_KEYS)` didn't change; otherwise the reply is regenerated. `TEXT_KEYS` are language, mix, formality, sentence_length, vocabulary and focus. Pace and expressiveness only affect TTS, so they never invalidate the draft.
10. **Post-processing** (`_finish`):
    - Parses the `##STAGE: id##`, `##NOTE: k=v##`, `##MEETING: ...##` and `##HANGUP##` tags, then strips them from the spoken text.
    - Hang-up is only honoured if a meeting was fixed, escalation is on, or the seller said an end phrase, and only if the text doesn't end in "?".
11. **TTS:** `split_speech` splits the reply into sentences and cuts a long first sentence at a comma. All sentences are synthesised in parallel with Bulbul v3, and each one is streamed to the browser as NDJSON as soon as it's ready.
12. `save()` writes the full call (transcript, verdicts, timeline, profile history, switch log, outcome) to `artifacts/ps04/switch_logs/<id>.json` after every turn.

**Other turn types**
- `open()`: the opening line.
- `nudge()`: after 8 s of silence, "are you still there?". After `MAX_NUDGES = 2` it ends the call.
- `mark_interrupted()`: the seller barged in, so the last bot turn is marked as cut off in the LLM history.
- `flush()`: answers a held, unfinished utterance after 2.5 s of silence.

### 2.4 The 10 live personas (`ps04/live_personas.py`)

| Persona id | Name | Selected by state | Delivery highlights |
|---|---|---|---|
| `normal` | Friendly Guide | calm | seller baseline |
| `de_escalator` | Calm De-escalator | aggressive | pace 0.88, low expressiveness, short, simple |
| `quick_direct` | Quick & Direct | rushed | pace 1.2, very short |
| `patient_explainer` | Patient Explainer | confused | pace 0.85, short, simple |
| `closer` | Enthusiastic Closer | interested | warm, expressive, proposes a slot |
| `reassurer` | Transparent Reassurer | suspicious | calm; "I'm IndiaMART's AI assistant" |
| `gentle_nudger` | Gentle Encourager | hesitant | low pressure, smaller next step |
| `empathetic_listener` | Empathetic Listener | complaining | acknowledge, apologise once |
| `advisor` | Business Advisor | inquisitive | precise answers from FACTS only |
| `respectful_persuader` | Respectful Persuader | uninterested | accept the no, one reason, graceful exit |

Each persona has `behaviour` (seller-meeting calls), `buyer_behaviour` (buyer help-desk calls) and a `GENERAL_BEHAVIOUR` variant for non-meeting use cases. The **speaker never changes** on a switch.

**Starting persona** (`starting_persona`): the use case's `start_persona` if it has one; else `quick_direct` for time-pressed sellers; else `reassurer` if the seller asked before whether VANI is a bot; else `normal`.

### 2.5 The prompt (`CallSession.system_prompt` + `_messages`)

The prompt is built fresh every turn, in this order:

```
ROLE → USE CASE (goal, archetype approach, known context, open checklist)
→ FACTS → GUARDRAILS → SELLER summary + hooks
→ CALL FLOW (stages, current stage, notes learned)
→ COMMON QUESTIONS → SOUND HUMAN
→ DELIVERY (language, mix, script, formality, warmth, empathy, vocab, max words)
→ ACTIVE PERSONA + focus (+ ESCALATION NOW)
→ OPENING PLAN, OBJECTION PLAYBOOK, ESCALATION RULES
→ OUTPUT rules + tag protocol
```

The history follows. A **final system message restates language, length and persona**, because models follow the latest instruction over the language used in earlier turns. Call flows (stages, handlers, note keys) are defined in `ps04/flows.py`, for seller and buyer modes.

### 2.6 Server and browser

**`ps04/server.py`**: standard library `ThreadingHTTPServer` on `127.0.0.1:8765`.

| Endpoint | What it does |
|---|---|
| `GET /` | `static/index.html` |
| `GET /api/sellers` | demo seller cards, personas, modes, use cases, uploaded-file cards, credits, languages |
| `GET /api/spec?seller=&use_case=` | full persona spec |
| `GET /api/log?session=` | full call log (for download) |
| `POST /api/upload` | load a seller file (or `?sample=1`) |
| `POST /api/start` | create a `CallSession`, return the opening (optionally streamed) |
| `POST /api/turn?session=&stream=1&interrupted=&nudge=&flush=` | seller audio (WAV body) or typed text (JSON) → NDJSON stream: `turn`, then `audio` × N, then `done` |

Sessions live in an in-memory dict, and each has a lock.

**`static/index.html`** (single file, vanilla JS):
- Mic capture runs through `AudioContext` with a homemade energy VAD (`PAUSE_MS = 750`, `PAUSE_SHORT_MS = 1000`, `MAX_UTTERANCE_MS = 25000`) and is encoded to 16 kHz WAV (`toWav`).
- Barge-in (300 ms of speech while VANI talks) stops playback and sends `interrupted=1`.
- Silence timers trigger flush (2.5 s) and nudge (8 s).
- Panels show the spec, live settings, analyst verdict, profile, persona timeline, call-flow checklist, switch log and credits.

**`ps04/sarvam.py`**: minimal standard-library client for STT, TTS and chat.
- **Credit cap:** every call is priced *before* it's sent, using published rates, and refused once the running estimate passes `SARVAM_CREDIT_LIMIT` (default 400). The estimate is kept in `artifacts/ps04/sarvam_usage.json`.
- The key comes from `SARVAM_API_KEY` or `.env`.

**`ps04/simulate.py`**: the backup demo. It plays scripted seller lines, spoken by a second Sarvam voice (`aditya`), through the *real* pipeline (Saaras → analyst → director → reply → Bulbul) for 3 sellers. It writes WAVs, switch logs and a README to `artifacts/ps04/demo_audio/`.

---

## 3. System A: the offline PS07 pipeline (summary)

```
raw export ─► ingest/ (adapter, ps07 loader, redact PII, pseudonymise w/ VANI_PSEUDONYM_SALT)
          ─► audit.py → docs/findings/data_audit.md
          ─► features/ (registry, lexicons: Hinglish/Devanagari/English, transcript features)
          ─► segmentation/ preprocess (residualise vs campaign/objective/language)
                          → GMM k∈3..6 + k-means cross-check → validate (stability gates)
                          → outcomes (adjusted meeting rate per segment)
          ─► personas/*.yaml (versioned, schema + lint checked by personas.py)
          ─► matching/ train (calibrated multinomial LR on early-turn features) → matcher.joblib
                       replay (run held-out calls through the live session)
          ─► switching/state_machine.py (threshold, margin, confirmations, dwell, cooldown, budget)
          ─► orchestration/ (DialogueState, layered composer POLICY›OBJECTIVE›STATE›STYLE, CallSession)
          ─► events.py (schema-validated, idempotent events/outcomes)
          ─► experiment/ (seller-level assignment, power calc, cluster bootstrap, SRM, guardrails)
```

**Key results** (from `docs/00-executive-summary.md`):
- 3 segments: receptive 57%, objection-raising 27%, time-pressed 16%.
- They are **in-call states, not seller traits**: earlier calls don't predict them (AUC ≈ 0.5).
- Matcher: macro-F1 0.75 at 3 seller turns. Replay: 82.5% agreement.
- Two Phase-3 criteria fail: calibration at 4–5 turns, and >1 persona change in 11.6% of calls.

**Contracts** live in `schemas/*.json` (with examples). **Thresholds** live in `config/thresholds.yaml`, all marked `validated: false`.

---

## 4. How to run

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows path
.venv/Scripts/pytest -q

# PS04
python -m vani_persona.ps04.evidence .      # needs the PS07 export folders in the repo root
python -m vani_persona.ps04.generate .
python -m vani_persona.ps04.server          # http://localhost:8765, needs SARVAM_API_KEY
python -m vani_persona.ps04.simulate
python -m vani_persona.ps04.analyst --selftest [--lexicon-only]
python -m vani_persona.ps04.cook            # regenerate the synthetic sample sellers
python -m vani_persona.ps04.sarvam --usage  # credit estimate
```

**Tests:** about 100 test functions. `tests/test_ps04.py` has 29 of them; the rest cover the PS07 pipeline.

---

## 5. File map (PS04)

| File | Lines | Role |
|---|---|---|
| `ps04/live.py` | 665 | `CallSession`: the whole live loop, prompt, tags, TTS chunking, logging |
| `ps04/use_cases.py` | 668 | 8 use cases, archetypes, checklists, inference |
| `ps04/generator.py` | 407 | seller row → persona spec with reasons/evidence |
| `ps04/server.py` | 302 | HTTP API + NDJSON streaming |
| `ps04/analyst.py` | 281 | LLM Behaviour Analyst + lexicon fallback + self-test |
| `ps04/signals.py` | 278 | lexicons, language tracker, pace requests, incomplete-utterance check |
| `ps04/sarvam.py` | 244 | API client + credit budget |
| `ps04/evidence.py` | 227 | past-call evidence → JSON/MD |
| `ps04/flows.py` | 207 | seller / buyer call flows (stages, handlers, note keys) |
| `ps04/cook.py` | 193 | synthetic sample sellers |
| `ps04/live_personas.py` | 178 | the 10 personas + starting persona |
| `ps04/profile.py` | 159 | live seller profile |
| `ps04/director.py` | 110 | switching rules |
| `ps04/simulate.py` | 122 | scripted backup recording |
| `ps04/static/index.html` | 573 | browser UI, VAD, playback |

---

## 6. Where to tune behaviour quickly

| You want to change… | Edit |
|---|---|
| When personas switch | constants at the top of `ps04/director.py` (`CLEAR`, `WEAK`, `PING_PONG`, `CALM_TURNS_TO_RECOVER`, …) |
| What a persona sounds like or does | `PERSONAS` / `GENERAL_BEHAVIOUR` in `ps04/live_personas.py` |
| How the seller is read | `PROMPT` and the lexicons in `ps04/analyst.py`; phrase lists in `ps04/signals.py` |
| Pace and patience adaptation | constants in `ps04/profile.py` |
| Reply length | `LENGTH_WORDS` in `ps04/live.py` |
| Seller-type rules and spec defaults | `seller_type()` and `generate()` in `ps04/generator.py` |
| Use-case priority and checklists | `infer_use_case()` and the `UseCase(...)` blocks in `ps04/use_cases.py` |
| Turn-taking feel (VAD) | `PAUSE_MS`, `FLUSH_AFTER_MS`, `NUDGE_AFTER_MS`, `BARGE_IN_MS` in `static/index.html` |

---

## 7. Improvement opportunities

These come from reading the code. Each one names the file and the reason.

### 7.1 Architecture
1. **Two parallel brains.** PS07 built a calibrated matcher and a hysteresis state machine. PS04 uses an LLM analyst plus a hand-written director. Either:
   - feed the PS07 matcher's scores into the Director as a third detector next to analyst and lexicon, or
   - replay the PS07 held-out transcripts through `analyst` + `director` and compare agreement and oscillation with the PS07 numbers (82.5% / 0.6%).

   There are also two classes named `CallSession`, in `orchestration/session.py` and `ps04/live.py`, which is confusing.
2. **`_seller_turn` is ~160 lines** (`ps04/live.py:472`). It handles STT merge, drafting, analysis, language, profile, director, diffing, logging and reply. Splitting it into `decide()` (pure, no I/O) and `respond()` would make the decision logic unit-testable without mocking Sarvam.
3. **Tag protocol in free text** (`##STAGE##`, `##NOTE##`, …) is parsed with regexes (`_finish`, `_track_flow`). It is brittle: there is already code to handle unterminated tags and stray punctuation left behind. Use JSON mode or a structured `{speech, stage, notes, meeting, hangup}` output, or have the analyst extract notes.

### 7.2 Latency (3.5–5 s today; production VANI is faster)
4. **The LLM reply isn't streamed.** TTS waits for the full reply. Streaming tokens and sending each completed sentence to Bulbul right away would cut time-to-first-audio the most.
5. **Two STT calls per turn** (`transcribe` + `translit`) double STT cost. Transliterating locally (e.g. `indic-transliteration`), or making the lexicons script-aware, would remove one call.
6. **The draft is thrown away whenever any `TEXT_KEY` changes**, and that includes formality flips from the profile. Log `draft_reused` rates from the switch logs. If reuse is low, consider letting the analyst's verdict apply from the *next* turn for minor profile changes (as PS07's async design does).
7. **The analyst can block for up to 3 s** before the reply is chosen. A tighter timeout for the lexicon fallback, or a smaller and faster model for the analyst, would help.
8. **Browser VAD is energy-based.** A real VAD (e.g. Silero VAD in WASM), or server-side endpointing, would reduce false cuts and the need for `looks_incomplete` holds.

### 7.3 Robustness and security
9. **`Access-Control-Allow-Origin: *` on a localhost server with no auth** (`server.py:_set_cors`). Any website open in the same browser could call `/api/start` and `/api/turn` and spend Sarvam credits. Restrict the origin to `http://localhost:8765`, or add a session token.
10. **Sessions are never evicted** (`State.sessions`). Add a TTL or delete sessions when a call ends.
11. **The credit budget is a file-based estimate** with a process-local lock. Two processes (server + simulate) can race. The fix is fine for a demo; for anything longer-lived use a proper store or Sarvam's own usage API.
12. **`save()` rewrites the full JSON log every turn.** That's fine for demos, but an append-only JSONL would be cheaper and crash-safe.

### 7.4 Quality of the "brain"
13. **The analyst self-test is tiny** (a few dozen lines in `analyst.SELFTEST`). Build a labelled set of a few hundred turns from real PS07 transcripts, especially for the 4 newer states (hesitant, complaining, inquisitive, uninterested). Then report per-state precision and recall, not only accuracy.
14. **The Director thresholds are hand-picked** and the intensity numbers come from the LLM, so they aren't calibrated. Replay real calls, measure switch rate, revert rate and ping-pong, and tune against that. Optionally calibrate intensity per state.
15. **The doc and the code disagree:** `docs/15-ps04-submission.md` says the analyst times out at 2.5 s, but `analyst.TIMEOUT_S = 3.0`. Keep the doc and code in sync.
16. **`seller_type` is a strict priority chain**, so a seller who is both new *and* time-pressed loses the "new" signal. Consider multi-label traits that each contribute settings (the spec is already per setting, so this fits).
17. **Voice choices are `informed_guess`** because past data had one voice. The only way to validate them is an experiment. The PS07 `experiment/` tooling (seller-level assignment, power, cluster bootstrap) already exists and could be wired to PS04 personas.

### 7.5 Product and demo
18. **Outcome tracking is thin:** `outcome` holds only the meeting and the checklist progress. Add per-call metrics (turns to goal, number of switches, time in each persona, hang-up reason) and a small dashboard over `switch_logs/`.
19. **The UI is one 573-line HTML file.** It works, but splitting out the JS (or moving to a small framework) would make UI changes easier.
20. **Test coverage for PS04 is 29 tests.** Add golden-path tests for the Director (sequences of verdicts → expected timeline) and for `_finish` tag parsing edge cases. Both are pure and cheap to test.

---

## 8. Mental model in one paragraph

Before the call, a rules engine turns the seller's file into a **spec**: voice, language, tone and plan, each setting with a reason and the evidence behind it. A use-case layer adds *what the call needs* (a checklist). During the call, every seller utterance is transcribed. An **LLM analyst**, backed by phrase lexicons, labels the seller's state and builds a live **profile**. A deterministic **Director** decides whether to switch to one of **10 named personas**, using thresholds, confirmation and anti-ping-pong rules. Language mirroring runs on its own track. The persona and profile only change *delivery* settings, which feed a layered prompt with fixed facts and guardrails. The reply is drafted speculatively to save time, then spoken by Bulbul sentence by sentence. Every decision is written to a **switch log** with its signal and reason.

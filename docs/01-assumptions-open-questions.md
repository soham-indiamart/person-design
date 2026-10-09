# 01 · Assumptions and open questions

*Deliverable 2.* Every assumption lists what changes if it turns out to be false.

## Confirmed requirements

| # | Requirement | Source |
|---|---|---|
| C1 | Scope is **seller-facing** VANI calls only | Stakeholder answer, 2026-10-09 |
| C2 | Deliverable: architecture docs **and** runnable scaffolding | Stakeholder answer |
| C3 | Historical data: the PS07 multi-table export (bot calls, turn transcripts, evaluator outputs, seller file), delivered 2026-10-09 and read in place | Delivered; `ingest/ps07.py` |
| C7 | VANI's seller calls are **outbound** calls to fix a meeting with an IndiaMART executive (`meeting_fixed`); dispositions: Meeting Fixed / Not Interested / General / Call Later | PS07 data dictionary |
| C8 | Turn timestamps exist but are evaluator-derived and snapped (49.6% zero gaps), so they are unusable for latency/overlap | Data audit |
| C9 | Transcripts are mostly romanized Hinglish (0.2% Devanagari) | Data audit |
| C10 | ~1.09 calls per seller in the transcript subset | Data audit |
| C4 | 3–6 personas, each evidence-backed; fewer than 3 if the evidence doesn't support more | Brief §2.3 |
| C5 | Personas never override business rules, facts, compliance or call purpose | Brief §1.2, §5.4 |
| C6 | Improvement must be shown by A/B against the current VANI | Brief §8 |

## Working assumptions

| # | Assumption | If false |
|---|---|---|
| A1 | ~~Calls are mainly outbound~~: **confirmed** (C7) | — |
| A2 | Pipeline is telephony → streaming ASR → LLM response generation → TTS, with a dialogue layer that keeps state | If responses are scripted/templated rather than LLM-generated, the STYLE block maps to template variants instead of prompt text. Everything else holds |
| A3 | Transcripts separate the speakers | If not, the adapter's `unknown_speaker_turns` will be high and diarization must be fixed first. **Blocking** |
| A4 | Turn timestamps may be missing or unreliable | **Confirmed unreliable** (C8): latency/overlap excluded by config; transcript-only features carry the analysis |
| A5 | Audio is not available for analysis | Already handled: no acoustic features are required. If audio is permitted later, add speech rate and pitch range as optional features |
| A6 | ~~A per-call outcome exists~~: **confirmed**. `meeting_fixed`, corrected by the `incorrect_mf` evaluator. Still unknown: whether fixed meetings actually happen (downstream) | A meeting-held feed is needed as a secondary metric |
| A7 | Languages are mainly Hindi, English and Hinglish; regional languages are less common | Lexicons cover Hindi/English only. Other languages get language mirroring only, with no behavioural lexicon features, until lexicons are added and validated |
| A8 | Sellers can receive several calls over time | Randomization is per seller (not per call) to avoid contamination; analysis uses a seller-cluster bootstrap |
| A9 | VANI can accept a per-turn STYLE block and TTS rate/pause parameters | If TTS rate is fixed, drop pace from personas and keep the textual style. Re-check persona distinctiveness |
| A10 | Each call has a deterministic objective and a policy owner who can approve "approved answers" | Without approved answers, objection handling falls back to the neutral, existing behaviour |

## Missing data (needed from the VANI team)

1. ~~Historical export~~: delivered. Open: transcripts for a **representative** sample of calls (the current subset over-represents meetings: 22.1% vs 11.5%).
2. ~~Disposition dictionary~~: delivered.
3. A downstream outcome feed: did the fixed meeting happen, and did it lead to a sale? Plus the lag.
4. Campaign, objective, agent-version and script-version identifiers per call (confounders).
5. Current latency profile: p50/p95 of end-of-speech to start-of-TTS. This sets the real latency budget.
6. Call volume per day per campaign and calls per seller, which determine experiment duration and design effect.
7. Consent/recording basis for analysing historical calls, and retention rules.

## Open questions that would materially change the design

| Question | Why it matters |
|---|---|
| Is the LLM response generator able to follow a style block reliably at current latency? | If not, personas must be implemented as prompt variants per persona (precompiled) rather than a dynamic block |
| Does VANI already handle language switching? | If yes, the explicit language override should call into the existing mechanism rather than duplicate it |
| Are there hard rules on call length or number of attempts? | These go into POLICY and can cap personas (e.g. "detailed" density) |
| Is there an existing experimentation platform? | Use its assignment service instead of `experiment/assignment.py` (same contract: deterministic seller → arm) |
| Who approves persona text and objective policies? | Defines the `approved_by` field and the release workflow |
| Is the "objective achieved" outcome trustworthy, or just a bot self-report? | Partly answered: `incorrect_mf` flags 2% of audited meetings. A meeting-held signal from the executive side would be the independent outcome |
| Should a persona revert to neutral once objections or rush fade? | Replay shows reverts in 9.9% of calls. This decides the Phase 3 switching target |
| What did the `personalis…` (personalised-script) trial show? | 452 recorded calls in the export: a prior style experiment whose results could inform MDE and novelty effects |

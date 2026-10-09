# 00 · Executive summary and final recommendation

*Deliverables 1 and 14.*

## Problem

VANI uses one conversational style with every seller. Sellers differ in how much time they have, how much explanation they want, which language they use, and which objections they raise. A single style is too verbose for some sellers and too terse for others. Today none of this is measured, so nobody knows how much it costs.

## Proposed solution

An evidence-driven adaptation layer that sits **beside** VANI's existing dialogue logic and never replaces it.

1. **Discover** 3–6 behavioural segments from historical transcripts, dispositions and outcomes. The segments are residualized against campaign, objective and language, and accepted only if they pass explicit stability gates (`src/vani_persona/segmentation/`).
2. **Encode** each accepted segment as a versioned, schema-validated persona. A persona controls *how* VANI speaks: tone, pace, length, question style, objection-handling moves. It never controls *what* VANI may say or do (`personas/`, `schemas/persona.schema.json`).
3. **Match** live calls from early behavioural signals with a calibrated, interpretable classifier. The call starts in a safe neutral style and the matcher abstains when the evidence is weak (`src/vani_persona/matching/`).
4. **Adapt mid-call** in two ways:
   - Explicit requests ("Hindi mein bolo", "jaldi bolo") take effect immediately.
   - Inferred persona changes go through a hysteresis state machine (threshold, margin, confirmations, dwell, cooldown, switch budget) so that the style cannot oscillate (`src/vani_persona/switching/`).
5. **Preserve context.** Prompts are layered POLICY › OBJECTIVE › STATE › STYLE. A switch replaces only STYLE and the TTS parameters, and a test enforces this (`tests/test_composer.py`).
6. **Learn safely.** Structured events and delayed outcomes feed periodic batch retraining. Exploration and a permanent holdout prevent the system from only confirming its own choices. Persona changes ship through offline eval, then shadow, then A/B.
7. **Prove it.** A seller-randomized A/B test against today's VANI, with a fixed horizon, guardrails, and a cluster-bootstrap analysis (`src/vani_persona/experiment/`).

## Major design decisions

| Decision | Choice | Why |
|---|---|---|
| Persona vs. business logic | Strict layering; personas lint-checked to contain no facts, offers or pressure tactics | Style can then be tested without risking compliance or accuracy |
| Segmentation | Residualized interpretable features → GMM with k-means cross-check, k ∈ 3..6, hard validation gates, human review | Interpretable, stability-tested, and able to say "no personas supported" |
| Language | A separate runtime axis (mirror the seller), **not** a persona dimension | Avoids language-as-persona stereotyping; language preference is directly observable |
| Matcher | Calibrated multinomial logistic regression; prototype scorer only as bootstrap/fallback; never an LLM | Fast (<5 ms target), explainable, calibratable, auditable |
| Switching | Explicit signals immediate; inferred switches gated | Respects direct requests without flapping on noise |
| Latency | Matching runs after a seller turn is finalized, off the response path | Adds no blocking work to the voice loop |
| Learning | Periodic batch retraining with promotion gates | Safer and more auditable than online learning at this stage |
| Proof | A/B randomized by seller; shadow mode first | Correlation in historical data is not causation |

## What the real data showed (PS07 export, 12,354 bot calls with transcripts, Jul–Sep 2026)

- **Three evidence-backed personas** (run `seg-63b6bc47e6`, all stability gates passed; proxy check V ≤ 0.12):

  | Persona | Share | Adjusted meeting rate under today's style |
  |---|---|---|
  | receptive majority (keeps today's style) | 57% | 40.5% |
  | objection-raising | 27% | 18.8% |
  | time-pressed | 16% | 15.9% |

  These gaps show **headroom**, not a promised lift.
- **The segments are in-call states, not seller traits.** Earlier calls don't predict them (AUC ≈ 0.5), and they appear after VANI's pitch. Live switching is essential; a classification at hello would not work. Every call starts neutral.
- **The gates worked.** The first run on real data produced no valid personas and the pipeline refused to force them. Documented, outcome-blind fixes followed: dense features, removing a filler word from the lexicon, and routing rare direct requests to overrides. See `findings/discovery_log.md`.
- **Matcher** (temporal holdout): macro-F1 0.75 and ECE 0.036 at 3 seller turns.
- **Replay through the live session:** 82.5% agreement with the full-call segment, 0.6% oscillation, p99 matcher latency 3.4 ms on a dev machine.
- **Two Phase 3 criteria fail:**
  - calibration at 4–5 turns;
  - more than one persona change in 11.6% of calls, mostly reverts to neutral once objections fade.

## Expected measurable benefits (hypotheses, not claims)

No lift is promised and no experiment has run. Hypotheses to test:
- more fixed meetings in the objection-raising and time-pressed states, through identity-first reassurance and single-choice scheduling;
- fewer callbacks that never convert;
- no regression in opt-outs, complaints, incorrect meetings or latency (guardrails).

## Current status (2026-10-09)

| Area | Status |
|---|---|
| Ingest (generic + PS07 multi-table), PII redaction, audit, features | Built, tested, run on real data |
| Segmentation + validation + outcome association | Run on real data: 3 segments, all gates passed |
| Matcher training/evaluation, replay, runtime matcher, switching, composer, session, events | Built, tested, evaluated on a real temporal holdout |
| A/B tooling (assignment, power, cluster-bootstrap analysis, SRM, guardrails) | Built and tested; not yet run (needs live traffic) |
| Persona library | `neutral_default` (approved), `objection_raising` and `time_pressed` (**candidates**: segments validated, styles not yet) |
| Open before an experiment | Human transcript review; approved answers for `fix_meeting`; revert-rate decision; per-window calibration; name redaction (NER) |

## Final recommendation

**Architecture.** Adopt the layered design above. It needs no change to VANI's telephony, ASR or TTS. It needs three integration points:
- a hook on finalized seller turns,
- a style block plus TTS parameters injected into the existing prompt/TTS call,
- event emission.

**First milestone.** Finish Phase 3 and start the Phase 4–5 shadow run:
- human review of 20 transcripts per segment, plus a hand-labelled lexicon precision check;
- the business supplies approved answers for `fix_meeting`;
- decide the revert-to-neutral policy and fix per-window calibration;
- ship `CallSession` in **shadow mode** on live `main_vani` traffic, which logs predictions and changes nothing;
- measure the live baseline meeting rate (the transcript subset over-represents meetings: 22% vs 11.5%).

**Most important unresolved dependency.** Approved policy content and a representative live baseline:
- **Approved answers.** For identity, purpose, cost and "already in touch with an executive" (currently placeholders). Personas can change *how* VANI handles objections, but only the business can supply *what* it may say.
- **Live baseline.** The meeting rate and ICC on all eligible live calls, not just the transcript subset. Without it the A/B test can't be sized.

**Before any production traffic gets an adaptive persona:**
1. Personas reach `status: in_experiment`. That requires segment evidence and a human review, and the lint enforces it.
2. Offline matcher evaluation meets the Phase 3 acceptance criteria, including per-segment recall and calibration (ECE).
3. Shadow mode shows the added latency and failure rates within budget, and a switching rate with no oscillation.
4. Compliance signs off on the objective policy and on persona style text.

**Before general rollout:** the A/B test reaches its pre-registered horizon with:
- a primary-metric CI excluding zero, in the right direction and above the practical-significance threshold,
- no guardrail breached,
- no SRM.

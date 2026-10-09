# 04 · Persona library

*Deliverable 5, plus brief §3.*

> **Evidence status (2026-10-09).** Three personas are backed by segments discovered in the PS07 export:
> - **8,722 eligible bot calls**, Jul–Sep 2026;
> - run `seg-63b6bc47e6`, which passes every validation gate;
> - full history in [`findings/discovery_log.md`](findings/discovery_log.md).
>
> The **segments** are statistically validated. The **styles** are design proposals (`evaluation.validated: false`). They become validated only after the human transcript review and the A/B test. Two earlier illustrative hypotheses are retired in `personas/retired/`.

## Library

| Persona | Status | Segment (share, n) | Defining behaviour (evidence) | Communication strategy | Today's adjusted meeting rate (95% CI) | Confidence / validation |
|---|---|---|---|---|---|---|
| `neutral_default` | approved | 0: receptive (56.7%, 4,947) | No distinctive features: short cooperative replies (median 50% of turns ≤ 3 tokens), no objections, no time pressure | **Keep today's VANI style**: polite, ≤ 3 sentences, one closed question per turn. It is also the fallback for abstentions | 40.5% (39.0–42.2) | High. This is the control style |
| `objection_raising` | candidate | 1: objection-raising (27.0%, 2,358) | `resistance_rate` z = +1.29. Detail (segment vs all): not-interested 9.7% vs 2.6% of turns, already-in-touch 5.2% vs 1.4%, identity/trust 4.3% vs 1.2%, price/paid 3.6% vs 1.0%; longer turns (11.1 vs 8.2 tokens) and more questions | Reassuring; identity and purpose first; ≤ 2 sentences; comprehension check; slower TTS 0.97; approved answers per objection; at most one reason question for "not interested", then respect and close | 18.8% (17.5–20.4) | Medium. Lexicon precision moderate ('already' ~8/12); early recall low (objections appear after the pitch); human review pending |
| `time_pressed` | candidate | 2: time-pressed (16.2%, 1,417) | `rush_rate` z = +1.64. Time-pressure turns 24.8% vs 5.6% ('abhi nahi', 'baad mein', 'time nahi', 'busy hoon'); 58% end as callback requests | Brisk; 1 sentence; the single ask up front; **a choice of two concrete slots** (meeting or callback); TTS 1.06; shorter endpointing | 15.9% (14.1–17.6) | Medium. A state of the moment: earlier calls don't predict it (AUC 0.53); human review pending |

**Rejected or retired:**
- **Clarification-seeking.** A very stable but small group: 3.5% of calls, below the 5% gate. It is handled by the `repeated_clarification` override instead.
- **Brevity-requesting.** Explicit "be brief" requests appear in 0.1% of calls. Handled by the brevity override.
- `hyp_detail_seeking`, `hyp_time_pressed_direct`: illustrative hypotheses from before the data arrived; retired.

### Why the receptive majority keeps today's style

Segment 0 has no distinctive features and already has the highest adjusted meeting rate. There is no evidence that a different style would help it. The matcher's errors are also one-directional: segments 1 and 2 get confused with 0 before objections or rush appear, never the reverse. So the receptive persona and the safe default are the same style. That makes the most common case behave like control, which simplifies the A/B test.

### What the outcome gaps mean (and don't)

Adjusted meeting rates are 18.8% and 15.9% against 40.5%. These show where today's uniform style struggles, i.e. where adaptation has headroom. They are **not** evidence that the proposed styles will close the gap:
- the segments are partly defined by behaviour that overlaps the outcome (objections ↔ refusals, rush ↔ callbacks);
- the transcript subset is not representative of all bot calls (22.1% vs 11.5% meeting rate).

Only the A/B test can establish an effect.

## Required fields: where each lives

All 18 fields required by brief §3.1 are in the persona YAML (`schemas/persona.schema.json`). Complete examples: `personas/objection_raising.yaml`, `personas/time_pressed.yaml`, `personas/neutral_default.yaml`.

| # | Field | YAML path |
|---|---|---|
| 1 | ID and version | `persona_id`, `version` |
| 2 | Name | `name` |
| 3 | Behavioural summary | `definition.behavioural_summary` |
| 4 | Seller type supported | `audience.party` (`seller`), `audience.call_objectives` (`fix_meeting`) |
| 5 | Supporting evidence | `definition.evidence`: run id, segment id, n, share, bootstrap/temporal ARI, adjusted outcome, human review |
| 6 | Distinctive features | `definition.distinctive_features[]` (feature, direction, z from the report) |
| 7 | Confidence and limitations | `definition.confidence`, `definition.limitations[]` |
| 8 | Tone | `style.tone` |
| 9 | Pace and response timing | `style.pace.{tts_rate, pause_after_question_ms, endpointing_patience_ms}` |
| 10 | Language behaviour | `style.language.{policy, register}`. Always `mirror_seller` |
| 11 | Opening strategy | `style.opening`. Used only before VANI's first turn. In practice every call opens neutral; the field matters for future campaign-specific defaults |
| 12 | Explanation length / density | `style.explanation.{max_sentences_per_turn, density, check_understanding}` |
| 13 | Question style and turn-taking | `style.turn_taking.{questions_per_turn, question_style, on_interruption}` |
| 14 | Objection handling | `style.objection_handling.{category: [steps]}` from the closed vocabulary `personas.py: STEP_INSTRUCTIONS` |
| 15 | Re-engagement | `style.re_engagement` |
| 16 | Switch-away conditions | `switching.switch_away_signals` + state-machine thresholds |
| 17 | Example dialogue | `style.example_dialogue[]` |
| 18 | Applicable objectives and restrictions | `audience.call_objectives`, `audience.excluded_objectives`. Enforced at runtime by `CallSession` |

## Defined behaviourally, not stereotypically

- Every persona is defined by what the seller does **in this call**: objecting, signalling time pressure, or neither.
- Category, region, bot version and lead bucket are excluded from clustering, and the proxy check shows the segments don't track them (Cramér's V ≤ 0.12).
- Seller history is not used as a prior.
- The personas describe communication needs ("needs reassurance and identity first", "needs a fast, single choice"), never the seller's worth, intent or character.

## Operational, not vague

Every style field maps to something the runtime executes:
- **Prompt instructions.** `max_sentences_per_turn`, `density` and the question rules become explicit instructions in the STYLE block (`composer.render_style`).
- **Voice parameters.** Pace fields become TTS/ASR parameters (`composer.tts_params`).
- **Objection handling.** Named moves from a closed vocabulary. Moves needing content pull from POLICY's approved answers.
- **Lint.** Forbids prices, offers, guarantees and pressure language. It requires every `not_interested` sequence to end with `respect_decision_and_close`, and requires segment evidence for `approved`/`in_experiment`.

All numeric pace values are **tunable proposals** until the A/B test.

## Next steps to promote the candidates

1. **Human review.** 20 random transcripts per segment, plus the checklist in `findings/segmentation_report.md`. Re-check 'already' precision.
2. **Policy content.** The business owner fills in the `<TO BE SUPPLIED>` approved answers in `config/objectives/fix_meeting.yaml`.
3. **Phase 3 decision** on the revert-to-neutral rate (`findings/discovery_log.md`, Replay), plus per-window calibration.
4. Then `validated_offline`, then compliance sign-off, then `in_experiment` (doc 09).

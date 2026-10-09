# Discovery log: PS07 export (written by hand; numbers from the generated reports)

This log records every analysis run and every decision taken between them, including the runs that failed. Its purpose is to make the path to the persona library auditable and to show that no threshold was tuned until something passed.

## Data

- **Source:** "Seller-Fit Adaptive Voice Persona" export (PS07), read in place by `ingest/ps07.py`.
- **Joined tables:**
  - `ps07_call_turns.csv`: 12,788 calls with turn transcripts and timestamps;
  - `ps07_bot_calls_*`: bot version, lead bucket, start time, disposition, `meeting_fixed`;
  - evaluator outputs: `disposition`, `incorrect_mf`;
  - seller file: category group and state, used only as confounders and proxy checks.
- **Join coverage:** 12,354 calls (434 transcript calls have no bot-call row).
- **Dates:** 2026-07-08 → 2026-09-01. Sellers: 11,298, at ~1.09 calls per seller.
- **Objective:** fix a meeting with an IndiaMART executive.
- **Outcome definition:** `objective_achieved` = `meeting_fixed = 1` and **not** flagged by the `incorrect_mf` evaluator (52 of 2,550 audited meetings were flagged).
- **Representativeness caveat.** The meeting rate is 22.1% in the transcript subset against 11.5% across all 269,455 answered bot calls. The subset is **not** representative: transcripts exist only where the `latency_repetition` evaluator ran. Every rate below describes this subset only, and the A/B baseline must be measured on live traffic.
- **PII:** the export is not masked. Regex redaction removes phone numbers, emails and IDs (255 redactions). It does not remove names; an NER pass is an open item. All reports here are aggregates.

## Audit findings that changed the design

| Finding | Evidence | Decision |
|---|---|---|
| Timestamps are evaluator-derived and snapped | 49.6% of seller turns start exactly when the bot turn ends; 3.4% start before it | `response_latency_median_ms` and `overlap_rate` excluded from clustering (`config/thresholds.yaml: exclude_features`) |
| Language hint absent; transcripts romanized | 0.2% of seller turns contain Devanagari | Language axis runs on romanized function-word lexicons only |
| Explicit brevity requests ≈ absent | 0.1% of eligible calls | Brevity stays an explicit override, not a persona dimension |
| 28% of calls have < 3 seller turns, or are wrong party / no response | 2,402 short + 1,230 wrong-party/no-response/technical | Excluded from discovery (counted, not labelled) |

## Runs

| Run | Change from previous | Result |
|---|---|---|
| 1 | Original design: 14 behavioural features incl. sparse lexicon rates | **No k passed.** Bootstrap ARI 0.53–0.57, temporal ARI 0.43–0.49, near-redundant segments, silhouette ≈ 0. The cause: zero-inflated lexicon features (most present in < 11% of calls) and redundancy (`long_reply_ratio`/`seller_mean_tokens` r = 0.77) |
| 2 | **Dense features only** (decided from sparsity/redundancy, before looking at outcomes): log verbosity, terseness, questions, affirmation, clarification, and two unions, `resistance_rate` (any objection) and `rush_rate` (time pressure or brevity). Sparse components kept as `detail` | k = 3 and 4 passed; k = 4 chosen. One segment was defined by `clarification_rate` |
| — | **Lexicon spot check** (model-assisted, 12 random contexts per phrase; *not* the human gate) | "matlab" was 75% of clarification hits and ~1/12 genuine: a filler. "kahan se" 11/12 identity questions (keep). "abhi nahi" ~10/12 postponement (keep). "already" ~8/12 (below 0.8; flagged for human review) |
| 3 | Bare "matlab" removed (question forms kept) | Every k failed **only** on minimum size. The genuine-clarification segment was very stable (bootstrap ARI 0.98, temporal 0.85) but just 3.5% of calls. The 5% gate was **not** lowered |
| 4 | `clarification_rate` → `detail`. Clarification requests are direct signals and are already handled by the `repeated_clarification` override, like brevity | **k = 3 passes**: bootstrap ARI 1.00, temporal ARI 1.00, silhouette 0.18, min share 16.2%. k = 4 and 6 also pass. k = 3 chosen by the pre-set rule (best silhouette). Run id `seg-63b6bc47e6` |

## Result (run seg-63b6bc47e6)

| Segment | Share | Distinctive (residualized z) | Adjusted meeting rate (95% CI) | Persona |
|---|---|---|---|---|
| 0 receptive | 56.7% (4,947) | none: short cooperative replies, no objection, no rush | 40.5% (39.0–42.2) | `neutral_default` (keep today's style) |
| 1 objection-raising | 27.0% (2,358) | `resistance_rate` +1.29; more questions, longer turns | 18.8% (17.5–20.4) | `objection_raising` |
| 2 time-pressed | 16.2% (1,417) | `rush_rate` +1.64; 58% callbacks | 15.9% (14.1–17.6) | `time_pressed` |

**Proxy check.** Cramér's V with category 0.08, region 0.08, bot version 0.07 and lead bucket 0.12, all well below 0.5. The segments do not track who the seller is.

**Caveats:**
1. **Method sensitivity.** GMM vs k-means agreement is only ARI 0.22. With zero-inflated union features, the GMM partition is close to the rule "raised any objection / signalled rush / neither", and k-means instead splits by verbosity. The segments should be read as these behavioural states.
2. **Circularity.** The segments are defined by in-call behaviour that overlaps the outcome: objections relate to "not interested", and rush relates to callbacks. Lower adjusted rates show *headroom under today's style*, not that a different style would help.
3. **Human review is pending** (20 transcripts per segment).

## Segments are in-call states, not seller traits

- Point-in-time check: for the 42.6% of calls whose seller had an earlier bot call, the earlier calls' "call later" share predicts the time-pressed segment with AUC 0.53, and the earlier "not interested" share predicts objection-raising with AUC 0.51.
- Seller history is therefore **not** used as a prior; every call starts neutral.
- The seller file's aggregate history columns cover Apr–Sep and include the call itself and later calls, so using them would leak.

## Matcher (lr-202610090826, temporal holdout of 1,745 calls)

| Seller turns seen | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Macro-F1 | 0.32 | 0.54 | **0.75** | 0.84 | 0.89 |
| ECE | 0.056 | 0.045 | **0.036** | 0.082 | 0.102 |

- At 3 turns the Phase 3 bar is met (macro-F1 ≥ 0.6, ECE ≤ 0.05).
- Calibration degrades at 4–5 turns: the pooled calibration should become per-window. This is open.
- Errors are one-directional: segments 1 and 2 get predicted as 0 (objections and rush have not appeared yet); 0 is never predicted as 1 or 2.
- Accuracy is lower on `arrowhead` (0.83) than on `main_vani` (0.92).

## Replay (shadow `CallSession`, same holdout)

| Measure | Result | Target |
|---|---|---|
| Calls never leaving neutral | 62.2% | — |
| First commit at seller turn 3 / 4 / 5 | 34% / 25% / 15% of committed calls | — |
| Switch between two personas | 2.3% | — |
| Revert to neutral | 9.9% | — |
| Return to an earlier persona | 0.6% | — |
| Calls with > 1 change | **11.6%** | ≤ 5% (**fails**, mostly from reverts) |
| Final persona = full-call segment | 82.5% | objection-raising 57%, time-pressed 68%, receptive 100% |
| Matcher latency p99 | 3.4 ms (dev machine) | < 5 ms |

**Open decision for Phase 3.** A revert after objections fade is arguably the intended behaviour ("objection resolved → back to neutral"). If the product owner accepts that, the target should be restated as "≤ 5% oscillation (return to an earlier persona)", which currently measures 0.6%. Otherwise, raise `revert_evaluations`. The target was **not** redefined here.

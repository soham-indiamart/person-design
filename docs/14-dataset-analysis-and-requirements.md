# 14 · PS07 dataset analysis and what we need from IndiaMART

*Written 2026-10-09.* Numbers are from the generated reports:
- [`findings/dataset_overview.md`](findings/dataset_overview.md): whole export (`python -m vani_persona.analysis.ps07_overview .`);
- [`findings/discovery_log.md`](findings/discovery_log.md): persona discovery on the transcript subset.

All analysis ran locally on the export and everything written out is aggregate-only.

## 1. What the export contains

The three folders contain one documentation file, `Dataset 1 of 3/PS07_DATA_DICTIONARY.md`. Datasets 2 and 3 contain CSVs only. Window: 1 Apr – 30 Sep 2026. Join keys: `fk_glusr_usr_id` (seller) and `attempt_id` (bot call).

| Table (folder) | Rows | What it is | How this project uses it |
|---|---|---|---|
| `ps07_bot_calls_*` (2) | 269,455 answered bot calls, 130,129 sellers | Bot version, lead bucket, attempt number, time, duration, disposition, `meeting_fixed`, LLM call summary (96%), recording URL (88%) | Outcome, confounders, baseline, experiment sizing |
| `ps07_evaluator_outputs_*` (3) | 278,332 JSON results from 10 evaluators | Detailed disposition (56% of calls), questions, objections, who-ended, IVR, executive contact, location, **latency/repetition + timestamped transcript** (4.7%) | Outcome correction (`incorrect_mf`), drop analysis, latency analysis |
| `ps07_call_turns.csv` (1) | 139,555 turns, 12,788 calls | Turn-level transcript (bot/seller, start/end seconds, text), extracted from the `latency_repetition` evaluator | **The only live-comparable input**: features, segmentation, matcher training |
| `ps07_seller_files_part*` (1) | 149,363 sellers × 58 columns | Profile (location, category, turnover, business type), engagement (64%), aggregated bot history | Confounders and proxy checks only. Profile is never a persona input; the history aggregates leak future calls |
| `ps07_recording_sample_5000.csv` (1) | 5,000 calls (2,494 with transcripts) | Balanced across outcomes and bot versions; public MP3 links | Intended for ASR checks. **Not downloaded** (needs your approval: §4) |
| `ps07_executive_calls.csv` (1) | 67,151 calls, 6,933 sellers, 21 Jul – 30 Sep | Human executive click-to-call log | Weak downstream signal (§2.6) |

## 2. Findings

### 2.1 Baseline and experiment arithmetic

- **Meeting rate: 11.5%** of answered bot calls. It falls over time (14.7% in April, 9.7% in August, 10.6% in September) while volume grows 7×.
- **2.07 calls per seller**, and the intra-seller correlation of `meeting_fixed` is **0.215**. Randomizing by seller therefore costs a design effect of **≈ 1.23**. (Earlier docs said ≈ 1; that came from the transcript subset and was wrong.)
- Sample size at α = 0.05 and power 0.8, with the 11.5% baseline and ~2,300 answered calls/day (September):

  | Overall lift to detect | Calls per arm | Duration (both arms) |
  |---|---|---|
  | +1.0 pt | ~20,400 | ~3 weeks |
  | +1.5 pt | ~9,200 | ~2 weeks |
  | +2.0 pt | ~5,300 | ~1 week (run ≥ 2 full weeks anyway) |

- **Dilution.** In replay only ~38% of calls ever leave neutral; the rest behave exactly like control. An overall +1 pt therefore needs about **+2.6 pt in the adapted calls**. The MDE must be agreed with that in mind (§3).

### 2.2 The biggest losses happen before personas can act

The detailed-disposition evaluator covers 150,538 calls:
- **55.1% `call_dropped`**: no response 25.2%; dropped mid-pitch 11.5%; dropped right after VANI says "IndiaMART" 10.0%; dropped after the seller-identity check 7.3%.
- Not interested 10.9%, callback 12.4%, meeting fixed 12.0% (incl. online).

A persona needs 2–3 seller turns of evidence. **17.3% of evaluated calls drop at the identity moments** (after "IndiaMART" or the seller-identity check), and a further 25.2% never respond: all of this happens **before any adaptation is possible**. The opening line and the identity moment are a separate, larger lever: a neutral-style A/B on opening variants. They are outside the persona system, which starts after the opening by design.

### 2.3 Attempt number and dialer timing dominate outcomes (confounders)

- Meeting rate by attempt: **16.3%** (1st) → 9.4% → 7.2% → 5.7% → 5.1% → **2.1%** (6th+). Later attempts are mostly "no response".
- The 11:00 dip (6.3% vs ~13%) is a **retry batch**: only 13% of 11:00 calls are first attempts, and 3rd+ attempts at 11:00 do worse still.
- **Action:** attempt number and hour must be stratification variables in the A/B and covariates in every outcome comparison. Attempt number should also be added to the residualization strata: it is known before dialing, but is not yet in the code.

### 2.4 Bot latency strongly tracks outcome

From `latency_repetition`, 12,788 calls:

| Per-call median bot lag | ≤ 1 s | 1–1.5 s | 1.5–2 s | 2–3 s | > 3 s |
|---|---|---|---|---|---|
| Meeting fixed | 25.6% | 24.9% | 23.0% | **13.8%** | **4.6%** |

This is correlational, but the cliff after 2 s is large. Two consequences:
1. The persona system must add no blocking latency (design: ≤ 2 ms inline plus STYLE tokens).
2. p95 bot latency is a hard A/B guardrail.

Other signals:
- Seller had to repeat themselves in 15.9% of calls; the bot was "loopy" in 13.0%. Both co-occur with *more* meetings, because longer calls have more chances for both. Not used as quality signals without a length adjustment.
- Bot pronunciation of locations is poor or flawed in 29.8% of `location_confirmation` calls.
- Location was rejected or corrected in 7.5% of those calls.

### 2.5 Outcome quality

- `incorrect_mf` flags **2.4%** of audited meetings as not really agreed. The project outcome is `meeting_fixed AND NOT incorrect_mf`.
- The disposition evaluator and the `meeting_fixed` label disagree on 3.5% of calls (2.4% evaluator-only meetings, 1.1% label-only).
- `execdrop`: 14.8% of sellers were **still trying to talk after the call closed**. Worth a check of the bot's hang-up logic, independent of personas.
- `se_contact`: 12.6% of evaluated sellers have already met or are in touch with an executive. Their meeting rate is 16.5%/15.8% vs 24.7%. This confounder is now handled by the `objection_raising` persona's "already in touch" move, which needs approved wording.

### 2.6 Downstream outcome is missing

Within 14 days, only 5.6% of meeting-fixed calls are followed by an answered executive click-to-call, against 2.1% of others. The click-to-call log covers just 6,933 sellers and Aug–Sep, and it can't show whether the meeting happened. **There is no "meeting held" signal**, which is the outcome that matters commercially.

### 2.7 Transcripts, language and Sarvam

- `ps07_call_turns.csv` covers only **4.7% of calls** (Jul–Aug), and with a meeting rate of 22.1% vs 11.5% it is **not a random sample**. The selection rule is unknown.
- Text is **99.9% Latin script**: romanized Hinglish. Calls with romanized Tamil, Bengali or Gujarati markers are each ~1%; mostly-English calls ~3%.
- Turn timestamps are snapped: 49.6% of seller turns start exactly when the bot turn ends. Latency/overlap features are excluded for that reason. The evaluator's own latency measurements look usable at call level.
- **Why this matters with Sarvam.** The behavioural features are phrase lexicons over romanized Hinglish, so live features only match training if live VANI receives text in the **same script and style**. Two things are unknown:
  - which Sarvam model and output mode produced (or will produce) the text;
  - whether `ps07_call_turns` text is raw Sarvam output at all, or an evaluator rewrite.

  This is the largest technical risk to the live matcher. It is cheap to check with a sample of live output (§3, item A1).

### 2.8 Prior style trials in the data

Several bot versions ran briefly in September. They were **not randomized** and each ran on one day. Compared with `main_vani` on the same attempt × lead-bucket mix:

| Version | Calls | Observed | Expected for main_vani |
|---|---|---|---|
| `personalis` (personalised script) | 452 | 14.2% | 11.5% |
| `base_vani` | 289 | 18.3% | 14.8% |
| `summary_va` | 91 | 3.3% | 7.0% |

These differences are suggestive only. They do show that style changes plausibly move the metric by a few points, which supports an MDE of +1.5–2 pt overall.

### 2.9 Seller history is sparse and not predictive per call

- `past_objections` is filled for 2.7% of sellers, `past_questions_asked` for 1.8%, and frustration / "asked if bot" for 0.1%.
- A strictly point-in-time history built from earlier bot calls predicts a call's persona segment at AUC 0.51–0.53.
- Conclusion: start every call neutral; no history prior. This also avoids profiling.

## 3. What we need from you

Ordered by how much each item blocks progress. **Bold** items block an A/B test.

### A. Sarvam / ASR (live-matcher correctness)

1. **A1. 300–500 live Sarvam transcripts of current VANI calls**, exactly as the VANI orchestrator receives them: raw text per finalized seller utterance. These let us check script, romanization and lexicon hit rates against the training data.
2. **A2. The Sarvam configuration in production:**
   - model and version;
   - output mode (native script, romanized/transliterated, or translated to English);
   - streaming vs batch;
   - language setting: fixed or auto-detect;
   - which per-utterance fields are returned: language code, confidence, word/segment timestamps.

   Each returned field unlocks a feature that is currently disabled: the language axis, the ASR-confidence filter, real latency/overlap.
3. **A3.** How `ps07_call_turns.csv` text was produced: raw Sarvam output (which config?), or a rewrite by the `latency_repetition` evaluator (which model?).
4. **A4. Approval (and an API key or budget) to transcribe recordings with your production Sarvam config.** I have not downloaded any audio. This would be used for:
   - (a) the 2,494 recording-sample calls that also have transcripts, to quantify how the training text differs from Sarvam text;
   - (b) a **random** sample of ~10,000 recent calls, to rebuild discovery on representative data with real timestamps.
5. **A5.** The current endpointing/VAD settings and the measured p50/p95 from seller-stop to bot-speech, given the 2 s cliff in §2.4.

### B. Business and policy content (persona style needs it)

6. **B1. Approved wording** for the `fix_meeting` objective: identity, purpose of the meeting, what the meeting covers and how long it takes, "is this paid?", "already in touch with an executive", verification route, and mandatory disclosures (AI, recording). These are placeholders in `config/objectives/fix_meeting.yaml` today.
7. **B2.** Meanings of the lead buckets (`PIM`, `PUA`, `UA`, `PAM`, `OLP`, `SCHD`, `NUR`, …) and bot versions (`main` vs `main_vani`, `last_met_d/a`, `arrowhead`, `personalis`, `base_vani`, `summary_va`), plus the scripts used in the September trials. Arrowhead has no recordings: is it a different telephony/ASR stack?
8. **B3.** Approvers for persona style text (conversation design + compliance), the people who sign `approved_by`.

### C. Data

9. **C1. A "meeting held" feed** (did the executive actually meet the seller?), and ideally a sale/subscription outcome, keyed by `attempt_id` or seller + date. This becomes the secondary metric and guards against fixing meetings that don't happen.
10. **C2.** The rule that selected the 4.7% of calls with turn transcripts, or transcripts for a random sample (see A4).
11. **C3.** How `lead_call_summary` is generated (model, prompt). At 96% coverage it could supply weak behavioural labels for all 269k calls, as offline analysis only.
12. **C4.** Dialer scheduling rules (retry batches at 10–12 h). These are needed to stratify the experiment correctly.

### D. Decisions

13. **D1. MDE for the A/B test:** the smallest overall lift in valid meeting rate worth shipping. Recommended +1.5 pt, which is ~2 weeks per §2.1 and implies ~+4 pt in adapted calls.
14. **D2.** Experiment scope: `main_vani` only (75% of calls) or all versions; first attempts only or all attempts.
15. **D3. Revert policy.** Should a persona fall back to neutral once objections or time pressure fade? Replay shows this in 9.9% of calls, and it decides the Phase 3 switching target.
16. **D4.** Whether to run an opening-line experiment in parallel (§2.2). It is the larger lever and stays separate from the persona experiment.

### E. People and access

17. **E1. Two Hindi/Hinglish-fluent reviewers for ~2 days:**
    - 20 transcripts per segment × 3 segments, against the checklist in `findings/segmentation_report.md`;
    - 200 seller turns per lexicon category for the precision check.
18. **E2. Access to the VANI orchestrator code or an integration contact.** We need to know:
    - where finalized ASR turns are available;
    - how the LLM prompt is assembled;
    - which LLM and TTS are used, and whether TTS rate and pauses are controllable;
    - the event pipeline (stream, warehouse);
    - any existing experiment-assignment service.
19. **E3.** A `VANI_PSEUDONYM_SALT` secret from your secret store, so pseudonymous seller keys are stable across runs.

### F. Legal and privacy

20. **F1. Written approval** to analyse this unmasked export (names are present in transcripts, summaries and evaluator quotes), with a retention period.
21. **F2.** Whether call audio and transcripts may be sent to Sarvam (A4) and to an LLM for NER-based name redaction, or whether redaction must run on-premises.

## 4. What I did not do, and why

- **Recordings.** I did not download any recordings, even though the URLs are public. They are personal audio, so downloading them needs F1/F2.
- **Summaries and quotes.** I did not use `lead_call_summary` or evaluator quotes as features. They are not available live, and they contain names.
- **Profile attributes.** I did not use seller-file profile data (location, turnover, category) for personas. It is used only to check that personas are *not* proxies for it: Cramér's V ≤ 0.12.

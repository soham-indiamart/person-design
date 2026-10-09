# 11 · Implementation roadmap

*Deliverable 12.* "Built" means the code exists in this repo and has tests. Every phase still needs to be run on VANI's real data and infrastructure.

## Phase 1: Data audit and baseline measurement

| | |
|---|---|
| **Tasks** | Receive the export into `data/raw/` (never committed). Run `audit --columns-only` and fill in `config/column_mapping.yaml` and `disposition_map`. Run the full audit. Confirm the outcome definition per objective with the business. Hand-label 200 seller turns per lexicon category and fix lexicons to precision ≥ 0.8. Measure the baseline primary metric, calls per seller and ICC over ≥ 4 weeks. Measure the current response-latency profile. Confirm the legal basis and retention for analysis. Integrate `CallSession` in **shadow mode**: logging only, behaviour unchanged |
| **Dependencies** | Export + data dictionary; legal approval; an orchestrator hook for finalized turns |
| **Deliverables** | `docs/findings/data_audit.md`; lexicon precision report; baseline metric note; shadow events flowing to the warehouse |
| **Acceptance** | ≥ 90% of calls have speaker-separated turns; an outcome is known for ≥ 70% of calls (else open an outcome-feed workstream); lexicon precision ≥ 0.8 per category; shadow adds no measurable p95 latency (≤ +5 ms, tunable) |
| **Risks** | Diarization missing (blocking). Outcomes missing or bot-reported only. Lexicons too noisy for some languages |
| **Built** | Adapter, redaction, audit, lexicons, `CallSession(mode="shadow")`, event schema |
| **Done on PS07 (2026-10-09)** | Export loaded (`ingest/ps07.py`), audit run, timestamps found unreliable (excluded), "matlab" lexicon defect found and fixed (spot check, not yet the human gate), transcript-subset bias quantified. **Remaining:** human lexicon precision check, live baseline, legal sign-off, NER for names, shadow integration |

## Phase 2: Historical analysis and persona discovery

| | |
|---|---|
| **Tasks** | Run `segmentation.pipeline`. Review the gates, proxy check and noisy-transcript robustness re-run. Human review: 20 transcripts per segment, plus the checklist. Draft `candidate` personas from approved segments (doc 04 procedure). Retire unsupported hypotheses |
| **Dependencies** | Phase 1 mapping + lexicons |
| **Deliverables** | `segmentation_report.md`, `outcome_association.md`, 3–6 candidate persona YAMLs **or** a documented "no persona library supported" result |
| **Acceptance** | Chosen k passes all gates; every segment has a behavioural description agreed by two reviewers; no proxy V > 0.5 without explanation; personas pass schema + lint |
| **Risks** | Weak structure (fewer than 3 segments). Segments that mirror campaigns. Over-interpretation of outcome association as causal |
| **Built** | Pipeline, validation gates, outcome association, report generation; verified on synthetic data |
| **Done on PS07** | 4 documented runs; k = 3 accepted (`seg-63b6bc47e6`); 3 persona files written (`neutral_default` + 2 candidates); hypotheses retired. **Remaining:** human review (20 transcripts per segment) |

## Phase 3: Persona library and offline validation

| | |
|---|---|
| **Tasks** | Train the matcher (`matching.train`). Evaluate on a temporal holdout. Human-review ≥ 50 calls per persona blind to predictions. Tune `commit_threshold`/switching on replayed historical calls (simulate `CallSession` turn by turn and count switches and reversals). Compliance review of persona style text and objective policies. Write style text per persona from segment needs |
| **Dependencies** | Phase 2 candidates |
| **Deliverables** | `matcher_eval.md`, `artifacts/matcher.joblib`, reviewed-agreement report, personas → `validated_offline` |
| **Acceptance** | Macro-F1 ≥ 0.6 at 3 seller turns; every persona's recall ≥ 0.5; ECE ≤ 0.05; coverage ≥ 40% by turn 3; no language/objective group > 0.1 below overall; replay: ≤ 5% of calls with > 1 switch; compliance sign-off |
| **Risks** | Early turns are not predictive (personas only recognizable late, so value is limited to later turns). Label circularity. Reviewers disagree |
| **Built** | Training/eval script, calibrated scorer, evidence, abstention, replay (`matching/replay.py`) |
| **Done on PS07** | Macro-F1 0.75 / ECE 0.036 at 3 turns. Replay: 82.5% final agreement, 0.6% oscillation. **Not met:** ECE at 4–5 turns (0.08–0.10), > 1 change in 11.6% of calls (mostly reverts). **Remaining:** per-window calibration, revert policy decision, blind human review, compliance review, approved answers |

## Phase 4: Runtime matching and safe persona switching

| | |
|---|---|
| **Tasks** | Productionize the integration: `ingest_seller_turn` inline, `evaluate_persona` async; STYLE block + TTS params passed to the LLM/TTS calls; pinned persona snapshot; circuit breaker; dashboards (doc 02 monitoring list). Benchmark latency on production hardware. Verify that the control path is byte-identical to today's prompt |
| **Dependencies** | Phase 3 artifacts; orchestrator team |
| **Deliverables** | Integrated runtime behind a feature flag; latency benchmark replacing the doc-02 targets; runbook |
| **Acceptance** | p95 added response latency ≤ 2 ms + measured STYLE-token effect ≤ 50 ms; fallback rate < 0.5% in load test; zero persona events in control; the continuity test passes in integration (state identical across a forced switch) |
| **Risks** | The LLM ignores the style block (measure instruction-following with an offline judge rubric). TTS rate unsupported. Prompt-length cost |
| **Built** | Session, composer, state machine, overrides, events (in-process library) |

## Phase 5: Shadow evaluation and controlled A/B testing

| | |
|---|---|
| **Tasks** | Shadow on 100% of eligible traffic for ≥ 2 weeks: matching distribution, switch rates, latency. Pre-register the experiment (metrics, MDE, horizon, segments, guardrails). Run the A/B with seller randomization and the matcher in shadow inside control. Daily SRM and guardrail checks. Final analysis at the horizon |
| **Dependencies** | Phase 4; agreed MDE; outcome pipeline |
| **Deliverables** | Shadow report; pre-registration doc; experiment readout (`experiment.analysis`) |
| **Acceptance** | Rollout criteria of doc 09 §4 |
| **Risks** | Underpowered (low volume, high ICC). Novelty effects. Contamination via shared numbers. Outcome delay lengthening the test |
| **Built** | Assignment, power calculator, cluster-bootstrap analysis, SRM, guardrails |

## Phase 6: Feedback-loop integration and gradual production rollout

| | |
|---|---|
| **Tasks** | Limited rollout per doc 09. Permanent 5% holdout. Wire ε-exploration into `CallSession`. Monthly human-review sampling. Drift monitors (doc 08 §5). Retraining and promotion pipeline (offline eval → shadow → canary). Expand to new objectives through new experiments |
| **Dependencies** | Phase 5 success |
| **Deliverables** | Production rollout; monitoring dashboards; retraining job; quarterly persona review |
| **Acceptance** | Holdout comparison remains positive after 8 weeks; no drift alarm open > 1 week; every model/persona version traceable in events |
| **Risks** | Effect decay. Silent ASR/script changes. Persona sprawl (cap at 6, enforced by the registry) |
| **Built** | Exploration flag, holdout config, versioned events |

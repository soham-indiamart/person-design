# VANI adaptive persona system (seller calls)

> **Hackathon (PS04 · Persona Design):** start with [docs/15-ps04-submission.md](docs/15-ps04-submission.md) and the one-slide [docs/ps04_slide.html](docs/ps04_slide.html).
>
> ```bash
> .venv/bin/python -m vani_persona.ps04.evidence .   # past-call evidence -> evidence/ps04_evidence.json
> .venv/bin/python -m vani_persona.ps04.generate .   # a persona spec for every seller (149,363)
> .venv/bin/python -m vani_persona.ps04.server       # live Sarvam voice demo on http://localhost:8765
> .venv/bin/python -m vani_persona.ps04.simulate     # backup recording: same scenario, 3 sellers
> ```
>
> Needs `SARVAM_API_KEY` in the environment or in a gitignored `.env`.

An evidence-driven architecture and working scaffolding that lets VANI adapt **how** it speaks to each seller. It does this without changing **what** VANI is allowed to say or do.

- Personas are discovered from historical calls.
- A persona is matched from early live-call signals.
- The style switches safely mid-call without losing context.
- The system improves through a governed feedback loop.
- Any gain must be proven by an A/B test against today's VANI.

> **Evidence status (2026-10-09).** The pipeline has run on the real PS07 export (12,354 VANI bot calls with transcripts, Jul–Sep 2026). It produced **three evidence-backed personas**, run `seg-63b6bc47e6`, with all stability gates passed:
> - `neutral_default`: the receptive 57%, who keep today's style;
> - `objection_raising` (27%);
> - `time_pressed` (16%).
>
> The segments are validated; the persona **styles** are unvalidated candidates until the human transcript review and the A/B test. **No outcome lift is claimed.** Start with [docs/findings/discovery_log.md](docs/findings/discovery_log.md).

## Documents (deliverables)

| # | Deliverable | Document |
|---|---|---|
| 1, 14 | Executive summary and final recommendation | [docs/00-executive-summary.md](docs/00-executive-summary.md) |
| 2 | Assumptions and open questions | [docs/01-assumptions-open-questions.md](docs/01-assumptions-open-questions.md) |
| 3 | End-to-end architecture (component + sequence diagrams, latency, reliability) | [docs/02-architecture.md](docs/02-architecture.md) |
| 4 | Historical data and segmentation design | [docs/03-historical-analysis-segmentation.md](docs/03-historical-analysis-segmentation.md) |
| 5 | Persona library | [docs/04-persona-library.md](docs/04-persona-library.md) |
| 6 | Persona configuration | [docs/05-persona-configuration.md](docs/05-persona-configuration.md) |
| 7 | Real-time matching and switching | [docs/06-realtime-matching-switching.md](docs/06-realtime-matching-switching.md) |
| 8 | Integration contracts | [docs/07-integration-contracts.md](docs/07-integration-contracts.md) |
| 9 | Feedback loop | [docs/08-feedback-loop.md](docs/08-feedback-loop.md) |
| 10 | A/B testing | [docs/09-ab-testing.md](docs/09-ab-testing.md) |
| 11 | Demonstration plan | [docs/10-demo-plan.md](docs/10-demo-plan.md) |
| 12 | Implementation roadmap | [docs/11-roadmap.md](docs/11-roadmap.md) |
| 13 | Risks and trade-offs | [docs/12-risks-tradeoffs.md](docs/12-risks-tradeoffs.md) |
| — | Privacy, security, responsible adaptation | [docs/13-privacy-security-responsible-adaptation.md](docs/13-privacy-security-responsible-adaptation.md) |
| — | **PS07 dataset analysis and what we need from IndiaMART** | [docs/14-dataset-analysis-and-requirements.md](docs/14-dataset-analysis-and-requirements.md) |
| — | **PS04 submission (hackathon brief)** | [docs/15-ps04-submission.md](docs/15-ps04-submission.md) |
| — | Generated findings (after running on real data) | `docs/findings/` |

## Setup

```bash
python3 -m venv .venv            # if ensurepip is missing: python3 -m venv --without-pip .venv, then bootstrap pip
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q              # full test suite
```

## Running

```bash
# Demo: selection, explicit override, mid-call switch, context preserved (illustrative personas)
.venv/bin/python -m vani_persona.demo.run_demo

# PS07 export (the three "Seller-Fit Adaptive Voice Persona - Dataset N of 3" folders in the repo root, read in place)
export VANI_PSEUDONYM_SALT=...                                       # from a secret store
.venv/bin/python -m vani_persona.audit . --source ps07                # -> docs/findings/data_audit.md
.venv/bin/python -m vani_persona.segmentation.pipeline . --source ps07  # -> segmentation_report.md, outcome_association.md
.venv/bin/python -m vani_persona.matching.train . --source ps07       # -> matcher_eval.md, artifacts/matcher.joblib
.venv/bin/python -m vani_persona.matching.replay . --source ps07      # -> replay_report.md (live session, held-out calls)
.venv/bin/python -m vani_persona.analysis.ps07_overview .            # -> dataset_overview.md (every table, aggregates)

# Any other single-table export (CSV/JSONL in data/raw/)
.venv/bin/python -m vani_persona.audit data/raw --columns-only       # then edit config/column_mapping.yaml

# Experiment sizing (use the measured baseline, not a guess)
.venv/bin/python -m vani_persona.experiment.power --baseline 0.20 --mde 0.02 --calls-per-seller 1.5 --icc 0.05

# Regenerate code-derived schemas after changing the feature registry or canonical models
.venv/bin/python -m vani_persona.schemagen
```

Set `VANI_PSEUDONYM_SALT` (from a secret store) before ingesting real data. Call data, derived data and model artifacts are gitignored and must never be committed.

## Repository map

```
config/            thresholds.yaml (all tunable, validated: false), column_mapping.yaml, objectives/fix_meeting.yaml (policy; approved answers TO BE SUPPLIED)
personas/          versioned persona YAML: neutral_default, objection_raising, time_pressed; retired/ = earlier hypotheses
schemas/           JSON Schemas (persona, features, match req/resp, switch decision, call event, outcome, canonical call) + examples/
src/vani_persona/
  canonical.py     observed call record (inferred labels live elsewhere)
  ingest/          generic CSV/JSONL adapter, PS07 multi-table loader, PII redaction, pseudonymization
  audit.py         data audit report
  features/        feature registry, Hinglish/Devanagari/English lexicons, transcript features
  segmentation/    residualization, GMM k-search + validation gates, outcome association, reports
  matching/        live feature state, explicit signals, matcher (calibrated LR / prototype), training + eval, replay
  switching/       hysteresis state machine
  orchestration/   dialogue state + objective policy, layered prompt composer, per-call session
  events.py        ordered, idempotent, schema-validated events and outcomes
  experiment/      seller-level assignment, power, cluster-bootstrap analysis, SRM, guardrails
  demo/            two-persona mid-call switch demo
tests/             tests/synthetic.py is SYNTHETIC data for code paths only
```

## Status legend for personas

`hypothesis` (no data) → `candidate` (from an approved segment) → `validated_offline` (matcher eval passed) → `in_experiment` (compliance sign-off) → `approved` (A/B success) / `retired`. The lint blocks `in_experiment`/`approved` without segment evidence.

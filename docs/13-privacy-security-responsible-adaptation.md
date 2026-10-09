# 13 · Privacy, security and responsible adaptation

*Brief §10.* These are architecture requirements: each row names the mechanism that enforces it.

| Requirement | Mechanism | Status |
|---|---|---|
| Authorization to analyse historical calls | Written purpose approval from legal/compliance before Phase 1; recorded in the audit report header | **To obtain** |
| Data minimization | Analysis uses transcripts + metadata only; audio not required. Features are aggregates. Events carry no transcript text | Built (event schema, features) |
| PII protection at ingest | `ingest/redact.py` removes phone numbers, emails, GSTIN, PAN and long digit runs **before** features or reports. Seller IDs become salted SHA-256 keys (`VANI_PSEUDONYM_SALT` from a secret store, never in the repo) | Built + tested. **The PS07 export is not masked**: transcripts contain seller and company names, which the regex redactor does not remove. An NER pass is required before any wider sharing; until then, analysis stays local and reports stay aggregate-only (tested) |
| Call data never in git | `.gitignore` excludes `data/`, `artifacts/`, `*.parquet` and the PS07 export folders. Reports are aggregate-only, and tests assert that no transcript text leaks into findings | Built |
| Retention | Doc 08 storage table: transcripts kept for the shortest permitted period; derived data 12 months (proposal) | To confirm with legal |
| Access control | Separate stores for transcripts / features / predictions / actions / outcomes; transcript access restricted and audited; warehouse joins only | Infra task (Phase 1) |
| Auditability of persona decisions | Every match, switch, override and opt-out is an ordered, idempotent event with persona version, model version, scores and reason | Built |
| Auditability of configuration changes | Personas and policies in git, PR review by conversation design + compliance, `approved_by`/`approved_at`, immutable versions | Built (schema/lint); process to adopt |
| Supported-language limits | Lexicons: Hindi (romanized + Devanagari) and English. Other languages get language mirroring only; no behavioural lexicon features until validated. `ObjectivePolicy.supported_languages` bounds the language override | Built |
| Low-confidence language detection | Language changes only through an explicit request (override) or the campaign default. Language is never switched on an inferred, low-confidence signal | Built |
| Testing across languages, accents, populations | Matcher eval breakdown by language/objective/campaign; promotion blocked if any group is > 0.1 below overall; A/B segment analysis by language | Built (report), process |
| Safeguards against stereotyping | Personas are defined by observable conversational behaviour only. Language, region and category are excluded from clustering and are not matcher predictors. Campaign, objective and language are used only to normalize features within comparable calls. Cramér's V proxy check on every segment. Human review checklist item. No persona may reference demographics | Built |
| Opt-outs and consent | Opt-out detection is handled by the policy layer in **every** arm and mode, ends the call, and is logged. AI disclosure is tracked in `DialogueState.consent`. POLICY states mandatory disclosures first | Built; the lexicon must be reviewed against real opt-out phrasing |
| Calling regulations (e.g. TRAI/DND, calling hours) | Outside the persona system: enforced by the existing dialer and policy. Personas cannot change who, when or how often to call | Existing |
| Style ≠ sensitive decisions | Personas affect only wording, length, pace and the order of approved moves. They never affect eligibility, pricing, offers, prioritization or whether a seller is called. Inferred segments are not stored on seller profiles and not used for other decisions | Built (layering + lint); policy commitment |
| No exploitation of vulnerabilities | Optimize for objective completion **with** complaint and opt-out guardrails; pressure phrases banned by lint; lint requires every persona's `not_interested` sequence to end with `respect_decision_and_close` (at most one reason question); no persona may target "hesitant" sellers with more persistence | Built (lint + step vocabulary), guardrails in doc 09 |

**Principle.** Persona assignment is a *communication-style* decision about the current conversation. It is ephemeral, it is not a judgement about the person, and it is not reused for anything else.

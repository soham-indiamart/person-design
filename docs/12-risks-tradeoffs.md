# 12 · Risks and trade-offs

*Deliverable 13.*

## Major choices

| Choice | Recommended | Alternatives considered | Rationale | Trade-off | Change the recommendation if… |
|---|---|---|---|---|---|
| Segmentation method | Residualized interpretable features → diagonal GMM, k-means cross-check, validation gates, human review | Rules only; embedding clustering; HDBSCAN | Interpretable centroids, soft membership, stability measurable, cheap | Elliptical-cluster assumption; lexicon dependence | Gates fail repeatedly while human reviewers see clear behaviour types → try HDBSCAN or LLM-assisted, schema-validated tagging as features |
| Confounder handling | Within-stratum residualization (campaign × objective × language) + covariate-adjusted outcome model | Raw features; propensity weighting | Removes campaign/language mix from clusters with no extra modelling | Can erase genuinely campaign-specific behaviour | Most campaigns have < 50 calls (backoff dominates) → pool campaigns by objective |
| Language | Separate runtime axis (mirror seller; explicit override) | Language-specific personas | Directly observable; avoids stereotyping; halves the persona count | Ignores possible language × style interactions | A/B segment analysis shows a style works in one language and harms in another → persona × language templates |
| Matcher | Calibrated multinomial LR; prototype scorer only as bootstrap/fallback | GBDT; LLM classifier; kNN | Fast, calibratable, explainable coefficients, stable | Linear decision boundaries | GBDT beats it on per-persona recall at equal ECE on a temporal holdout |
| LLM role | Generates text under a style block; never classifies | LLM persona judge per turn | Latency, cost, auditability, stereotyping risk | Misses nuanced cues lexicons can't capture | An LLM tagger (schema-validated, measured) adds ≥ 5 points macro-F1 within the latency budget |
| Explicit vs inferred signals | Explicit applied immediately; inferred gated | Everything through the classifier | Direct requests are ground truth; gating stops flapping | Inferred adaptation is ≥ 2 turns late | Replay shows most value is in turns 1–3 → lower `confirm_evaluations` for high-confidence cases |
| Switching | Threshold + margin + confirmations + dwell + cooldown + budget + revert | Lock at first match; switch on argmax | Adaptive without oscillation; verified by tests | Conservative; may miss short-lived needs | Replay/shadow show many missed adaptations and no oscillation → relax margin/confirmations |
| Feature windowing | Last 3 seller turns for live evaluation | Cumulative prefix; exponential decay | Mid-call change becomes visible within 2–3 turns | Matcher is trained on prefixes from call start; mid-call windows are a slight distribution shift | Matcher accuracy on mid-call windows (replayed) is much worse → add sliding windows to training |
| Execution | In-process library; inline lexical phase + async evaluation | Separate microservice | No network hop; failure contained per call | Shares the orchestrator's runtime/language | The orchestrator isn't Python, or multiple consumers need it → service with the same JSON contracts and a 30 ms deadline |
| Learning | Periodic batch retraining with promotion gates | Online learning; contextual bandit | Auditable; handles delayed outcomes; safe | Slower adaptation | System proven, outcomes fast, guardrails mature → bandit with off-policy evaluation |
| Randomization unit | Seller | Call | Avoids within-seller contamination over repeated calls | Fewer independent units; needs a design effect | Sellers almost never receive > 1 call → call-level randomization is fine |
| Persona storage | Versioned YAML in git + JSON Schema + lint + per-call pinning | Database/CMS | Reviewable diffs, CI validation, simple rollback | Needs a deploy (or config sync) to change | Non-engineers must edit frequently → CMS that exports the same schema and still runs lint |

## Likely failure modes

| Failure mode | Detection | Mitigation |
|---|---|---|
| No stable segments in real data | Gates fail in the segmentation report | Ship explicit-signal adaptation only; collect timestamps; revisit features |
| Segments mirror campaigns or language | Proxy check (Cramér's V) | Stronger residualization; drop the segment |
| Lexicon false positives (already seen on synthetic data) | Phase-1 precision review; lexicon tag-rate monitor | Context-aware phrases; per-language review |
| Early turns don't predict the segment | Turns-to-decision, coverage | Value comes from explicit overrides + later switching; reduce the persona count |
| LLM ignores the style block | Offline rubric eval; sentence counts in shadow transcripts | Stronger prompt placement; per-persona precompiled prompts; post-generation length check |
| Persona text drifts into persuasion or claims | Lint; compliance review; complaint/opt-out guardrails | Closed step vocabulary; POLICY precedence |
| Oscillating style | Switch-rate and reversal monitors | Gates; budget; per-persona overrides |
| Latency regression | p95 guardrail | Async evaluation; circuit breaker; neutral fallback |
| Feedback loop self-confirms | Holdout; exploration; blind review | Doc 08 §4 |
| Outcome definition gamed by the bot | Mismatch between bot disposition and CRM action | Use independent outcomes as primary |
| Underpowered experiment | Power calc from the measured baseline | Larger MDE, longer test, more campaigns, CUPED with pre-period covariates |

## Why this fits the evidence available now

- The design put reproducible, gated analysis ahead of persona invention, and that paid off on real data. The first run on the PS07 export produced **no** valid personas, and the gates refused to force them. The library was found only after documented, outcome-blind fixes (dense features, lexicon precision, routing direct requests to overrides). See `findings/discovery_log.md`.
- The system does something useful and low-risk even if no personas survive: the neutral default plus explicit language and brevity overrides.
- It adds no blocking work to the voice path.
- Every adaptive choice is logged with its reason and model version, so the A/B test and later audits can attribute effects.

The minimum viable architecture is Phases 1–5 with the logistic matcher and at most 3–4 personas. Everything beyond that (more personas, LLM tagging, bandits) has explicit triggers above and should wait until they fire.

# 07 · Integration contracts

*Deliverable 8, plus brief §6.4.* Schemas: `schemas/`. Real examples captured from the demo run: `schemas/examples/`. `tests/test_schemas.py` validates every example against its schema and checks that the generated schemas are current.

## Deployment shape

Recommended: the persona runtime runs **in-process** inside the dialogue orchestrator, as a library (`vani_persona`). There is no network hop, it shares the call's memory, and its failure is contained per call. The JSON contracts below are the wire formats for the case where the matcher becomes a separate service, and they are already the event/log formats.

| Interface | Direction | Mode | Contract |
|---|---|---|---|
| Behavioural feature extraction | orchestrator → persona runtime | in-process function | `schemas/features.schema.json` (generated from the feature registry) |
| Persona matching | runtime → matcher | in-process, or RPC with a 30 ms deadline | `match_request.schema.json` → `match_response.schema.json` |
| Switching decision | matcher result → state machine | in-process | `switch_decision.schema.json` |
| Persona configuration retrieval | registry → composer | load at start; pinned per call | `persona.schema.json`; `GET /personas/{id}?version=` if served |
| Call-event logging | runtime → event stream | async, at-least-once | `call_event.schema.json` |
| Outcome submission | CRM/telephony/QA → outcome intake | async, idempotent | `outcome.schema.json` |

## Shared rules

- **Session identity.** `call_id` is unique per call. `session_id` identifies the runtime session (they differ when a call is transferred or resumed). `seller_key` is always the salted hash, never the raw ID.
- **Schema versioning.** Every payload carries `schema_version` (`MAJOR.MINOR`). Consumers accept any minor version of their major. Additive changes bump the minor version; breaking changes bump the major and run in parallel streams during migration.
- **Ordering.** `seq` increases monotonically per call and gives a total order within a call. Consumers order by (`call_id`, `seq`), never by `ts`, because clocks drift.
- **Idempotency.**
  - `event_id` = UUIDv5(`call_id:seq:event_type`), so redelivery keeps the id and consumers dedupe on it.
  - `outcome_id` = UUIDv5(`call_id:outcome_type:source`), so a resubmitted outcome is a no-op.
  - Corrections are new records with `supersedes`.
- **Missing fields.**
  - Unobservable features are `null`, never 0.
  - Matchers skip `null` features (prototype) or impute training medians (logistic).
  - Unknown context values map to `__unknown__` and back off to coarser residualization strata.
  - Missing outcomes stay missing; they are never inferred as failure.
- **No transcript text** in events. Transcripts stay in the access-controlled transcript store, joined by `call_id`.

## Examples

### Match request (`schemas/examples/match_request.json`, abridged)

```json
{
  "schema_version": "1.0",
  "call_id": "demo-call-001",
  "turn_index": 13,
  "features": {
    "n_seller_turns": 3.0,
    "short_reply_ratio": 0.3333,
    "question_rate": 0.0,
    "affirmation_rate": 0.0,
    "log_mean_tokens": 1.4663,
    "resistance_rate": 0.0,
    "rush_rate": 0.3333,
    "response_latency_median_ms": null,
    "overlap_rate": null,
    "...": "every feature in features.schema.json; null = not observable"
  },
  "context": {"campaign_id": "main_vani", "call_objective": "fix_meeting", "lead_type": "PIM"},
  "current_persona_id": "objection_raising",
  "deadline_ms": 30
}
```

### Match response (`schemas/examples/match_response.json`)

This example uses the prototype scorer, so `calibrated` is false. With the trained logistic scorer, `scorer` is `"logistic"`, `calibrated` is true, and `evidence` lists per-feature log-odds contributions.

```json
{
  "schema_version": "1.0",
  "call_id": "demo-call-001",
  "turn_index": 13,
  "scores": {"neutral_default": 0.2549, "objection_raising": 0.0523, "time_pressed": 0.6929},
  "best_persona_id": "time_pressed",
  "confidence": 0.6929,
  "calibrated": false,
  "abstained": false,
  "reason": "time_pressed 0.69 >= 0.6",
  "evidence": [{"feature": "rush_rate", "value": 0.3333, "detail": "+0.00 scale units off the time_pressed prototype"}],
  "uncertain_signals": ["question_rate"],
  "scorer": "prototype",
  "model_version": "prototype:objection_raising@1.0.0,time_pressed@1.0.0",
  "latency_ms": 0.036
}
```

### Switch decision (`schemas/examples/switch_decision.json`)

```json
{
  "turn_index": 13,
  "seller_turn": 7,
  "action": "switch",
  "from_persona": "objection_raising",
  "to_persona": "time_pressed",
  "phase": "COMMITTED",
  "reason": "time_pressed 0.69 beats objection_raising 0.05 by >= 0.15 for 2 evaluations",
  "challenger": null,
  "challenger_streak": 0,
  "scores": {"neutral_default": 0.2549, "objection_raising": 0.0523, "time_pressed": 0.6929}
}
```

### Call event (`schemas/examples/call_event.persona_switched.json`)

```json
{
  "schema_version": "1.0",
  "event_id": "…uuid5…",
  "call_id": "demo-call-001",
  "session_id": "demo-session-001",
  "seq": 17,
  "event_type": "persona_switched",
  "ts": "2026-10-09T08:32:35.177+00:00",
  "turn_index": 13,
  "experiment": {"experiment_id": "demo", "arm": "treatment"},
  "persona": {"persona_id": "time_pressed", "version": "1.0.0"},
  "payload": {"…": "the switch decision above"}
}
```

Event types:

| Event type | Meaning |
|---|---|
| `call_started` | Carries `pinned_personas` and the mode |
| `match_evaluated` | Every matcher run: scores, abstained, reason, latency |
| `persona_committed`, `persona_switched`, `persona_reverted` | Persona changes |
| `style_override_applied` | Explicit signals |
| `opt_out_detected` | Opt-out handed to the policy layer |
| `matcher_fallback` | Matcher failed or timed out |
| `agent_turn_composed` | TTS params and overrides used for each reply |
| `call_ended` | End of call |

### Outcome (`schemas/examples/outcome.json`)

```json
{
  "schema_version": "1.0",
  "outcome_id": "…uuid5(call_id:objective_achieved:crm)…",
  "call_id": "demo-call-001",
  "outcome_type": "objective_achieved",
  "value": true,
  "source": "crm",
  "observed_at": "2026-10-09T10:15:00+00:00",
  "supersedes": null,
  "details": {"objective_id": "fix_meeting", "evidence": "meeting held by executive; not flagged by incorrect_mf evaluator"}
}
```

### Persona configuration retrieval (if served)

```
GET /personas/time_pressed?version=1.0.0
200 → persona document (persona.schema.json)
GET /personas?status=approved,in_experiment
200 → [{"persona_id": "...", "version": "...", "status": "..."}]
```

Clients cache by (`persona_id`, `version`). A version is immutable, so the cache needs no invalidation.

## Main runtime data flow

See the sequence diagram in `docs/02-architecture.md`. In short:
1. connect;
2. assign arm and pin personas;
3. neutral opening;
4. each finalized seller turn → `ingest_seller_turn` (inline: explicit signals, opt-out) → compose and reply → `evaluate_persona` (async: features → match → switch → events);
5. call end → `call_ended`;
6. outcomes arrive later through the outcome intake and are joined on `call_id`.
